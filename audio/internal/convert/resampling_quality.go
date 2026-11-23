// Copyright 2026 The Ebiten Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package convert

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
)

// ResampleQuality selects the interpolation kernel used by a [Resampling].
//
// The kernels differ in tap count and window, and therefore in CPU cost,
// memory use, and fidelity. All qualities share the exact same Read, Seek,
// Length, buffering and EOF semantics; only the per-sample weights change.
type ResampleQuality int

const (
	// ResampleQualityDefault is the quality [NewResampling] uses.
	// It is equivalent to [ResampleQualityBalanced], so existing streams are
	// reproduced sample for sample.
	ResampleQualityDefault ResampleQuality = iota

	// ResampleQualityFast is a two-tap linear interpolator. It touches only the
	// two source samples adjacent to each output sample, making it the cheapest
	// option. It has no stop-band rejection beyond the natural roll-off of the
	// triangular kernel, so high-frequency content and steep transients are
	// attenuated and aliasing is audible on aggressive rate changes.
	ResampleQualityFast

	// ResampleQualityBalanced is the historical Ebitengine kernel: a sinc
	// interpolator evaluated with 17 taps (half width 8) and a Hann window.
	ResampleQualityBalanced

	// ResampleQualityHighFidelity is a 33-tap (half width 16) sinc interpolator
	// with a Kaiser window (beta 10.063, roughly 100 dB stop-band attenuation).
	// Its weights are precomputed in a fixed-size polyphase table, so its
	// per-sample cost is a table lookup rather than transcendentals.
	ResampleQualityHighFidelity
)

// ErrInvalidResampleQuality is returned when a quality value is not one of the
// declared [ResampleQuality] constants.
var ErrInvalidResampleQuality = errors.New("convert: invalid resample quality")

// kernelHalfWidthMax bounds the half width, and therefore the tap count, of any
// windowed-sinc kernel. It keeps per-output-sample work, the polyphase table
// size and the source-block look-ahead finite even for a deliberately long
// custom kernel.
const kernelHalfWidthMax = 64

// kernelPhaseBits is the resolution of the high-fidelity polyphase table.
// The fractional part of each interpolation position is quantised to
// 1<<kernelPhaseBits phases.
const kernelPhaseBits = 11
const kernelPhaseCount = 1 << kernelPhaseBits
const kernelPhaseMask = kernelPhaseCount - 1

// kernelSpec describes one windowed-sinc resampling kernel.
type kernelSpec struct {
	// halfWidth is the kernel extent on each side of the interpolation point;
	// the tap count is 2*halfWidth+1.
	halfWidth int

	// window maps an offset d with |d| <= halfWidth to a window weight.
	window func(d float64) float64
}

// hannWindow is the window used by [ResampleQualityBalanced]. It is evaluated
// through fastCos01 to stay bit-identical to the historical implementation.
func hannWindow(d, halfWidth float64) float64 {
	return 0.5 + 0.5*fastCos01(d/(halfWidth*2+1))
}

// kaiserWindow is the window used by [ResampleQualityHighFidelity]. beta 10.063
// is the textbook value for roughly 100 dB of stop-band attenuation.
func kaiserWindow(d, halfWidth, beta float64) float64 {
	arg := 2 * d / (halfWidth * 2)
	v := 1 - arg*arg
	if v <= 0 {
		// The window reaches zero exactly at the outermost taps.
		return 0
	}
	return besselI0(beta*math.Sqrt(v)) / besselI0(beta)
}

// normalizedSinc returns sin(πx)/(πx), the standard band-limited interpolation
// kernel, with value 1 at x == 0.
func normalizedSinc(x float64) float64 {
	if math.Abs(x) < 1e-12 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// besselI0 returns the value of the modified Bessel function I0 at x using its
// power series. It is used only while building the polyphase table, never on
// the per-sample path. The series converges for every finite x; iterating a
// fixed number of terms drives the remainder below float64 round-off for the
// arguments the Kaiser window reaches (|x| around its beta, ~10).
func besselI0(x float64) float64 {
	// I0 is even.
	y := x * x / 4
	var sum, term float64 = 1, 1
	for k := 1; k < 64; k++ {
		term *= y / float64(k*k)
		sum += term
	}
	return sum
}

// qualityKernelSpec returns the kernel specification for q. The fast quality
// has no windowed-sinc kernel and is handled separately in [Resampling.at].
func qualityKernelSpec(q ResampleQuality) (kernelSpec, bool) {
	switch q {
	case ResampleQualityDefault, ResampleQualityBalanced:
		hw := 8.0
		return kernelSpec{
			halfWidth: int(hw),
			window:    func(d float64) float64 { return hannWindow(d, hw) },
		}, true
	case ResampleQualityHighFidelity:
		hw := 16.0
		beta := 10.063
		return kernelSpec{
			halfWidth: int(hw),
			window:    func(d float64) float64 { return kaiserWindow(d, hw, beta) },
		}, true
	default:
		return kernelSpec{}, false
	}
}

// polyphaseKernel is a fixed-size lookup table of windowed-sinc weights.
// weights[phase*stride+k] is the weight of the source sample at
// center-halfWidth+k (the unclamped first tap plus k) for a fractional
// position of phase/kernelPhaseCount.
type polyphaseKernel struct {
	halfWidth int
	stride    int
	weights   []float64
}

var (
	polyphaseKernels     = map[ResampleQuality]*polyphaseKernel{}
	polyphaseKernelsLock sync.Mutex
)

// getPolyphaseKernel builds (once) and returns the polyphase table for q.
// The table has a fixed, quality-derived size: at most
// kernelPhaseCount*(2*kernelHalfWidthMax+1) float64 entries, so the total
// memory cached across all qualities is bounded.
func getPolyphaseKernel(q ResampleQuality) (*polyphaseKernel, error) {
	polyphaseKernelsLock.Lock()
	defer polyphaseKernelsLock.Unlock()
	if k := polyphaseKernels[q]; k != nil {
		return k, nil
	}

	spec, ok := qualityKernelSpec(q)
	if !ok {
		return nil, ErrInvalidResampleQuality
	}
	if spec.halfWidth <= 0 || spec.halfWidth > kernelHalfWidthMax {
		return nil, fmt.Errorf("convert: kernel half width %d is out of range (1..%d)", spec.halfWidth, kernelHalfWidthMax)
	}

	stride := spec.halfWidth*2 + 1
	k := &polyphaseKernel{
		halfWidth: spec.halfWidth,
		stride:    stride,
		weights:   make([]float64, kernelPhaseCount*stride),
	}
	for phase := range kernelPhaseCount {
		frac := float64(phase) / kernelPhaseCount
		base := phase * stride
		var sum float64
		for j := range stride {
			// Tap j corresponds to source sample (center-halfWidth+j), matching
			// the n-firstN indexing used while interpolating. Its offset from the
			// interpolation point (center+frac) is therefore frac+halfWidth-j.
			d := frac + float64(spec.halfWidth) - float64(j)
			s := normalizedSinc(d) * spec.window(d)
			k.weights[base+j] = s
			sum += s
		}
		// Normalise each phase to unit DC gain. This compensates for the
		// truncation of the sinc tails, keeping constant signals unchanged
		// regardless of fractional position.
		if sum != 0 {
			inv := 1 / sum
			for j := range stride {
				k.weights[base+j] *= inv
			}
		}
	}
	polyphaseKernels[q] = k
	return k, nil
}

// atLinear evaluates the fast two-tap linear kernel at output frame t.
func (r *Resampling) atLinear(t int64) (float64, float64, error) {
	tInSrc := float64(t) * float64(r.from) / float64(r.to)
	if math.IsNaN(tInSrc) || math.IsInf(tInSrc, 0) {
		return 0, 0, fmt.Errorf("convert: interpolation position is not finite for output frame %d", t)
	}
	n0 := int64(math.Floor(tInSrc))
	frac := tInSrc - float64(n0)

	l0, r0, err0 := r.src(n0)
	if err0 != nil && err0 != io.EOF {
		return 0, 0, err0
	}
	l1, r1, err1 := r.src(n0 + 1)
	if err1 != nil && err1 != io.EOF {
		return 0, 0, err1
	}
	lv := l0*(1-frac) + l1*frac
	rv := r0*(1-frac) + r1*frac
	if err0 == io.EOF || err1 == io.EOF {
		return lv, rv, io.EOF
	}
	return lv, rv, nil
}

// atPolyphase evaluates a precomputed windowed-sinc kernel at output frame t.
func (r *Resampling) atPolyphase(t int64, k *polyphaseKernel) (float64, float64, error) {
	tInSrc := float64(t) * float64(r.from) / float64(r.to)
	if math.IsNaN(tInSrc) || math.IsInf(tInSrc, 0) {
		return 0, 0, fmt.Errorf("convert: interpolation position is not finite for output frame %d", t)
	}
	center := int64(math.Floor(tInSrc))
	frac := tInSrc - float64(center)
	phase := int(frac * kernelPhaseCount)
	if phase >= kernelPhaseCount {
		phase = kernelPhaseMask
	}
	base := phase * k.stride
	firstN := center - int64(k.halfWidth)
	startN := max(firstN, 0)
	endN := center + int64(k.halfWidth)

	var lv, rv float64
	var eof bool
	for n := startN; n <= endN; n++ {
		srcL, srcR, err := r.src(n)
		if err != nil && err != io.EOF {
			return 0, 0, err
		}
		if err == io.EOF {
			eof = true
		}
		// The table is indexed relative to the unclamped first tap, so a
		// boundary clamp cannot shift the kernel off centre.
		s := k.weights[base+int(n-firstN)]
		lv += srcL * s
		rv += srcR * s
	}
	if eof {
		return lv, rv, io.EOF
	}
	return lv, rv, nil
}

// newResamplingWithQuality is the quality-aware constructor behind
// [NewResampling]. It validates every parameter so that an invalid quality or an
// unusable rate is reported as an error instead of failing later inside Read.
func newResamplingWithQuality(source io.Reader, length int64, from, to int, bitDepthInBytes int, quality ResampleQuality) (*Resampling, error) {
	if from <= 0 {
		return nil, fmt.Errorf("convert: original sample rate must be positive but was %d", from)
	}
	if to <= 0 {
		return nil, fmt.Errorf("convert: target sample rate must be positive but was %d", to)
	}
	switch bitDepthInBytes {
	case 2, 4:
	default:
		return nil, fmt.Errorf("convert: bit depth in bytes must be 2 or 4 but was %d", bitDepthInBytes)
	}
	switch quality {
	case ResampleQualityDefault, ResampleQualityFast, ResampleQualityBalanced, ResampleQualityHighFidelity:
	default:
		return nil, fmt.Errorf("%w: %d", ErrInvalidResampleQuality, quality)
	}
	// Resolve and cache the polyphase table up front so that a kernel whose
	// half width exceeds the cap fails at construction rather than during the
	// first Read. Only the high-fidelity kernel interpolates through the table;
	// balanced keeps the historical on-the-fly evaluation and fast is linear,
	// so neither pays for a table it never reads.
	var kernel *polyphaseKernel
	if quality == ResampleQualityHighFidelity {
		k, err := getPolyphaseKernel(quality)
		if err != nil {
			return nil, err
		}
		kernel = k
	}

	if length < 0 {
		length = -1
	}
	r := &Resampling{
		source:                source,
		declaredSrcLength:     length,
		from:                  from,
		bitDepthInBytes:       bitDepthInBytes,
		to:                    to,
		quality:               quality,
		kernel:                kernel,
		srcBlock:              -1,
		lastReadSrcBlock:      -1,
		lastReadSrcBlockValid: true,
		srcBufL:               map[int64][]float64{},
		srcBufR:               map[int64][]float64{},
	}
	r.derivedSrcLength.Store(-1)
	return r, nil
}

// NewResamplingWithQuality returns a stream that converts the sample rate of
// source using the given quality. length is the length of source in bytes;
// a negative length indicates the length is unknown.
//
// The returned stream behaves exactly like the one from [NewResampling] with
// respect to Read, Seek and Length; quality only selects the interpolation
// kernel. An error is returned for an invalid quality, a non-positive sample
// rate, or an unsupported bit depth.
func NewResamplingWithQuality(source io.Reader, length int64, from, to int, bitDepthInBytes int, quality ResampleQuality) (*Resampling, error) {
	return newResamplingWithQuality(source, length, from, to, bitDepthInBytes, quality)
}
