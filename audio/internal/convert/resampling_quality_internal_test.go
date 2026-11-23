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
	"bytes"
	"errors"
	"io"
	"math"
	"testing"
)

func TestResampleQualityConstants(t *testing.T) {
	// The numeric values are part of the package contract.
	cases := []struct {
		q    ResampleQuality
		want int
	}{
		{ResampleQualityDefault, 0},
		{ResampleQualityFast, 1},
		{ResampleQualityBalanced, 2},
		{ResampleQualityHighFidelity, 3},
	}
	for _, c := range cases {
		if int(c.q) != c.want {
			t.Errorf("quality %d: got %d, want %d", c.q, c.q, c.want)
		}
	}
}

func TestQualityKernelSpecs(t *testing.T) {
	t.Run("balanced", func(t *testing.T) {
		spec, ok := qualityKernelSpec(ResampleQualityBalanced)
		if !ok {
			t.Fatal("balanced quality has no kernel spec")
		}
		if got, want := spec.halfWidth, 8; got != want {
			t.Errorf("half width: got %d, want %d (17 taps)", got, want)
		}
		if got, want := spec.halfWidth*2+1, 17; got != want {
			t.Errorf("tap count: got %d, want %d", got, want)
		}
		// The Hann window must be 1 at the centre and symmetric. Its zero
		// crossing sits at halfWidth+0.5; the outermost tap (halfWidth) keeps a
		// small nonzero weight, which is why the kernel spans 2*halfWidth+1 taps.
		if got := spec.window(0); math.Abs(got-1) > 1e-15 {
			t.Errorf("window at 0: got %v, want 1", got)
		}
		if got := spec.window(float64(spec.halfWidth)); got <= 0 || got >= 0.01 {
			t.Errorf("Hann window at outermost tap: got %v, want a small positive weight", got)
		}
		if got := spec.window(float64(spec.halfWidth) + 0.5); math.Abs(got) > 1e-15 {
			t.Errorf("Hann window at halfWidth+0.5: got %v, want 0", got)
		}
		for _, x := range []float64{-8, -3.5, -0.25, 0.25, 3.5, 8} {
			if got := math.Abs(spec.window(x) - spec.window(-x)); got > 1e-15 {
				t.Errorf("Hann window is not symmetric at %v: diff %v", x, got)
			}
		}
	})

	t.Run("default matches balanced", func(t *testing.T) {
		d, okd := qualityKernelSpec(ResampleQualityDefault)
		b, okb := qualityKernelSpec(ResampleQualityBalanced)
		if !okd || !okb || d.halfWidth != b.halfWidth {
			t.Fatalf("default (%+v, %t) must share the balanced (%+v, %t) kernel", d, okd, b, okb)
		}
		for _, x := range []float64{-8, -3.5, -0.25, 0, 0.25, 3.5, 8} {
			if d.window(x) != b.window(x) {
				t.Errorf("window mismatch at %v: %v vs %v", x, d.window(x), b.window(x))
			}
		}
	})

	t.Run("high fidelity", func(t *testing.T) {
		spec, ok := qualityKernelSpec(ResampleQualityHighFidelity)
		if !ok {
			t.Fatal("high-fidelity quality has no kernel spec")
		}
		if got, want := spec.halfWidth, 16; got != want {
			t.Errorf("half width: got %d, want %d (33 taps)", got, want)
		}
		if got, want := spec.halfWidth*2+1, 33; got != want {
			t.Errorf("tap count: got %d, want %d", got, want)
		}
		if got := spec.window(0); math.Abs(got-1) > 1e-15 {
			t.Errorf("Kaiser window at 0: got %v, want 1", got)
		}
		if got := spec.window(float64(spec.halfWidth)); math.Abs(got) > 1e-12 {
			t.Errorf("Kaiser window at edge: got %v, want ~0", got)
		}
		// Symmetry is required so the two stereo channels stay phase-aligned.
		for _, x := range []float64{0.5, 1.5, 7.25, 15.75} {
			if got := math.Abs(spec.window(x) - spec.window(-x)); got > 1e-15 {
				t.Errorf("Kaiser window is not symmetric at %v: diff %v", x, got)
			}
		}
	})

	t.Run("fast has no windowed kernel", func(t *testing.T) {
		if _, ok := qualityKernelSpec(ResampleQualityFast); ok {
			t.Error("the fast linear kernel must not use a windowed-sinc spec")
		}
	})
}

func TestBesselI0(t *testing.T) {
	// Reference values from the defining series I0(x) = Σ (x²/4)^k / (k!)²,
	// cross-checked independently with a Python reference implementation.
	cases := []struct {
		x    float64
		want float64
	}{
		{0, 1},
		{1, 1.2660658777520082},
		{3.5, 7.3782034322254795},
		{10.063, 2989.1499890042824},
		{-1, 1.2660658777520082}, // I0 is even.
	}
	for _, c := range cases {
		got := besselI0(c.x)
		if math.Abs(got-c.want) > 1e-9*max(1, math.Abs(c.want)) {
			t.Errorf("besselI0(%v): got %v, want %v", c.x, got, c.want)
		}
	}
}

func TestPolyphaseKernelShapeAndGain(t *testing.T) {
	for _, q := range []ResampleQuality{ResampleQualityDefault, ResampleQualityBalanced, ResampleQualityHighFidelity} {
		t.Run(qualityName(q), func(t *testing.T) {
			k, err := getPolyphaseKernel(q)
			if err != nil {
				t.Fatal(err)
			}
			if k.halfWidth <= 0 || k.halfWidth > kernelHalfWidthMax {
				t.Fatalf("half width %d outside 1..%d", k.halfWidth, kernelHalfWidthMax)
			}
			if got, want := k.stride, k.halfWidth*2+1; got != want {
				t.Errorf("stride: got %d, want %d", got, want)
			}
			if got, want := len(k.weights), kernelPhaseCount*k.stride; got != want {
				t.Errorf("table size: got %d, want %d", got, want)
			}
			// Every phase must normalise to unit DC gain, so constant signals
			// survive at every fractional position.
			for phase := range kernelPhaseCount {
				base := phase * k.stride
				var sum float64
				for j := range k.stride {
					sum += k.weights[base+j]
				}
				if math.Abs(sum-1) > 1e-12 {
					t.Fatalf("phase %d DC gain: got %v, want 1", phase, sum)
				}
			}
			// The integer-position phase (frac == 0) must be a unit impulse at the
			// centre tap: sampling an integer grid point reproduces it exactly.
			const phase0 = 0
			base := phase0 * k.stride
			for j := range k.stride {
				want := 0.0
				if j == k.halfWidth {
					want = 1
				}
				if got := k.weights[base+j]; math.Abs(got-want) > 1e-12 {
					t.Errorf("phase 0 tap %d: got %v, want %v", j, got, want)
				}
			}
			// Table sharing: a second lookup returns the same allocation.
			k2, err := getPolyphaseKernel(q)
			if err != nil {
				t.Fatal(err)
			}
			if k2 != k {
				t.Error("polyphase kernel was rebuilt instead of cached")
			}
		})
	}
}

func TestPolyphaseKernelMemoryBound(t *testing.T) {
	// Even if every valid quality is in use, the cached tables occupy a fixed,
	// bounded number of float64 entries independent of stream count or length.
	qualities := []ResampleQuality{ResampleQualityBalanced, ResampleQualityHighFidelity}
	var entries int
	for _, q := range qualities {
		k, err := getPolyphaseKernel(q)
		if err != nil {
			t.Fatal(err)
		}
		entries += len(k.weights)
	}
	bound := len(qualities) * kernelPhaseCount * (2*kernelHalfWidthMax + 1)
	if entries > bound {
		t.Errorf("cached entries %d exceed bound %d", entries, bound)
	}
}

func TestNewResamplingWithQualityValidation(t *testing.T) {
	src := bytes.NewReader(make([]byte, 64))

	t.Run("invalid quality", func(t *testing.T) {
		for _, q := range []ResampleQuality{-1, 4, 42, 1 << 30} {
			if _, err := newResamplingWithQuality(src, 64, 44100, 48000, 2, q); !errors.Is(err, ErrInvalidResampleQuality) {
				t.Errorf("quality %d: got %v, want ErrInvalidResampleQuality", q, err)
			}
		}
	})

	t.Run("rates", func(t *testing.T) {
		for _, rates := range [][2]int{{0, 48000}, {-44100, 48000}, {44100, 0}, {44100, -1}} {
			if _, err := newResamplingWithQuality(src, 64, rates[0], rates[1], 2, ResampleQualityBalanced); err == nil {
				t.Errorf("rates %v: got no error", rates)
			}
		}
	})

	t.Run("bit depth", func(t *testing.T) {
		for _, depth := range []int{0, 1, 3, 8} {
			if _, err := newResamplingWithQuality(src, 64, 44100, 48000, depth, ResampleQualityBalanced); err == nil {
				t.Errorf("bit depth %d: got no error", depth)
			}
		}
	})

	t.Run("valid", func(t *testing.T) {
		for _, q := range []ResampleQuality{ResampleQualityDefault, ResampleQualityFast, ResampleQualityBalanced, ResampleQualityHighFidelity} {
			for _, depth := range []int{2, 4} {
				r, err := newResamplingWithQuality(src, 64, 44100, 48000, depth, q)
				if err != nil {
					t.Errorf("quality %d depth %d: unexpected error %v", q, depth, err)
					continue
				}
				if r.quality != q {
					t.Errorf("stored quality: got %d, want %d", r.quality, q)
				}
			}
		}
	})

	t.Run("unknown length stays unknown", func(t *testing.T) {
		r, err := newResamplingWithQuality(src, -99, 44100, 48000, 2, ResampleQualityFast)
		if err != nil {
			t.Fatal(err)
		}
		if got := r.declaredSrcLength; got != -1 {
			t.Errorf("declared source length: got %d, want -1", got)
		}
	})
}

func TestNonFiniteAndExtremeRatiosDoNotReadOutOfBounds(t *testing.T) {
	// NaN/Inf source samples must flow through without panicking or growing the
	// block cache, and extreme rate ratios must keep every source index in range.
	for _, depth := range []int{2, 4} {
		t.Run(qualityDepthName(ResampleQualityHighFidelity, depth), func(t *testing.T) {
			const blockFrames = 4096 // resamplingBufferSize in resampling.go
			const frames = 3 * blockFrames
			in := make([]byte, frames*depth*2)
			if depth == 4 {
				// Plant NaN, +Inf and -Inf in the float32 source, both channels.
				specials := []uint32{
					math.Float32bits(float32(math.NaN())),
					math.Float32bits(float32(math.Inf(1))),
					math.Float32bits(float32(math.Inf(-1))),
				}
				for i, sp := range specials {
					for ch := 0; ch < 2; ch++ {
						off := ((i*7+1)*2 + ch) * 4
						in[off] = byte(sp)
						in[off+1] = byte(sp >> 8)
						in[off+2] = byte(sp >> 16)
						in[off+3] = byte(sp >> 24)
					}
				}
			}
			// An extreme but finite ratio that still yields many output frames.
			// Every source index reached must stay within the stream.
			for _, q := range []ResampleQuality{ResampleQualityFast, ResampleQualityBalanced, ResampleQualityHighFidelity} {
				r, err := newResamplingWithQuality(bytes.NewReader(in), int64(len(in)), 192000, 300, depth, q)
				if err != nil {
					t.Fatal(err)
				}
				out, err := io.ReadAll(r)
				if err != nil {
					t.Fatalf("quality %d: ReadAll: %v", q, err)
				}
				if len(out) == 0 {
					t.Fatalf("quality %d produced no output", q)
				}
				// The LRU block cache is capped at four regardless of quality.
				if len(r.lruSrcBlocks) > 4 {
					t.Errorf("quality %d cached %d blocks, cap is 4", q, len(r.lruSrcBlocks))
				}
				if len(r.srcBufL) > 4 || len(r.srcBufR) > 4 {
					t.Errorf("quality %d cache maps grew to %d/%d", q, len(r.srcBufL), len(r.srcBufR))
				}
			}

			// The most aggressive down-conversion possible (to == 1) yields at most
			// one output frame; it must still terminate without an out-of-range read.
			r, err := newResamplingWithQuality(bytes.NewReader(in), int64(len(in)), 192000, 1, depth, ResampleQualityHighFidelity)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := io.ReadAll(r); err != nil {
				t.Fatalf("extreme ratio ReadAll: %v", err)
			}
		})
	}
}

func TestLongTapsAreBounded(t *testing.T) {
	if kernelHalfWidthMax < 16 {
		t.Fatalf("the high-fidelity half width (16) must fit under the cap %d", kernelHalfWidthMax)
	}
	spec, ok := qualityKernelSpec(ResampleQualityHighFidelity)
	if !ok {
		t.Fatal("missing high-fidelity spec")
	}
	if spec.halfWidth > kernelHalfWidthMax {
		t.Errorf("half width %d exceeds cap %d", spec.halfWidth, kernelHalfWidthMax)
	}
}

func qualityName(q ResampleQuality) string {
	switch q {
	case ResampleQualityDefault:
		return "default"
	case ResampleQualityFast:
		return "fast"
	case ResampleQualityBalanced:
		return "balanced"
	case ResampleQualityHighFidelity:
		return "high"
	default:
		return "unknown"
	}
}

func qualityDepthName(q ResampleQuality, depth int) string {
	return qualityName(q) + "-depth" + string(rune('0'+depth))
}
