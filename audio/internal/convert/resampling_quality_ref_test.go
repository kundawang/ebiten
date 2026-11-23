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

package convert_test

import (
	"encoding/binary"
	"math"
)

// This file holds the high-precision offline reference used to judge the
// quality of each resampling kernel. It deliberately uses a much longer,
// directly evaluated windowed-sinc kernel (no phase quantisation, no lookup
// table) and runs only in tests.

// besselI0Ref is a power-series evaluation of the modified Bessel function I0,
// used by the reference Kaiser window.
func besselI0Ref(x float64) float64 {
	y := x * x / 4
	var sum, term float64 = 1, 1
	for k := 1; k < 200; k++ {
		term *= y / float64(k*k)
		sum += term
	}
	return sum
}

// kaiserWindowRef returns w(d) for |d| <= halfWidth, a symmetric Kaiser window
// with the given beta.
func kaiserWindowRef(d, halfWidth, beta float64) float64 {
	v := 1 - (d/halfWidth)*(d/halfWidth)
	if v <= 0 {
		return 0
	}
	return besselI0Ref(beta*math.Sqrt(v)) / besselI0Ref(beta)
}

// referenceResample resamples a mono float64 signal from rate `from` to rate
// `to` using a 129-tap Kaiser-windowed (beta 14) sinc. Positions before the
// first and after the last sample read as zero, matching Resampling.src's
// out-of-range behaviour. It is treated as ground truth for error reporting.
func referenceResample(in []float64, from, to int) []float64 {
	const halfWidth = 64.0
	const beta = 14.0
	outLen := int(math.Floor(float64(len(in)) * float64(to) / float64(from)))
	out := make([]float64, outLen)
	for t := range outLen {
		ts := float64(t) * float64(from) / float64(to)
		start := max(int64(math.Floor(ts-halfWidth)), 0)
		end := min(int64(math.Floor(ts+halfWidth)), int64(len(in))-1)
		var acc float64
		for n := start; n <= end; n++ {
			d := ts - float64(n)
			sinc := 1.0
			if math.Abs(d) > 1e-12 {
				sinc = math.Sin(math.Pi*d) / (math.Pi * d)
			}
			acc += in[n] * sinc * kaiserWindowRef(d, halfWidth, beta)
		}
		out[t] = acc
	}
	return out
}

// rmsError returns the root-mean-square difference between got and want over
// the indices both slices share. Edges where the reference needs warm-up
// samples are therefore included but weighted with all other samples.
func rmsError(got, want []float64) float64 {
	n := min(len(got), len(want))
	if n == 0 {
		return math.NaN()
	}
	var sum float64
	for i := range n {
		d := got[i] - want[i]
		sum += d * d
	}
	return math.Sqrt(sum / float64(n))
}

// maxError returns the largest absolute difference over shared indices.
func maxError(got, want []float64) float64 {
	n := min(len(got), len(want))
	var m float64
	for i := range n {
		if d := math.Abs(got[i] - want[i]); d > m {
			m = d
		}
	}
	return m
}

// sineSignal builds a mono sine of freq Hz lasting seconds at rate.
func sineSignal(rate int, seconds, freq float64) []float64 {
	n := int(seconds * float64(rate))
	out := make([]float64, n)
	for i := range n {
		out[i] = 0.8 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate))
	}
	return out
}

// sweepSignal builds a logarithmic sine sweep from f0 to f1 over seconds.
func sweepSignal(rate int, seconds, f0, f1 float64) []float64 {
	n := int(seconds * float64(rate))
	out := make([]float64, n)
	k := math.Log(f1 / f0)
	for i := range n {
		t := float64(i) / float64(rate)
		phase := 2 * math.Pi * f0 * seconds * (math.Exp(k*t/seconds) - 1) / k
		out[i] = 0.7 * math.Sin(phase)
	}
	return out
}

// impulseSignal returns a signal that is one at the given index and zero
// elsewhere, exposing a kernel's tap shape directly.
func impulseSignal(n, at int) []float64 {
	out := make([]float64, n)
	if at >= 0 && at < n {
		out[at] = 1
	}
	return out
}

// encodeF32Stereo writes a mono signal duplicated into both stereo channels as
// little-endian float32 bytes.
func encodeF32Stereo(mono []float64) []byte {
	b := make([]byte, len(mono)*8)
	for i, v := range mono {
		lv := math.Float32bits(float32(v))
		binary.LittleEndian.PutUint32(b[i*8:], lv)
		binary.LittleEndian.PutUint32(b[i*8+4:], lv)
	}
	return b
}

// decodeF32StereoMono decodes little-endian stereo float32 bytes and returns
// the left channel as float64.
func decodeF32StereoMono(b []byte) []float64 {
	n := len(b) / 8
	out := make([]float64, n)
	for i := range n {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*8:])))
	}
	return out
}

// encodeI16Stereo writes a mono signal into both channels as int16 bytes.
func encodeI16Stereo(mono []float64) []byte {
	b := make([]byte, len(mono)*4)
	for i, v := range mono {
		v = math.Max(-1, math.Min(1, v))
		s := int16(v * (1<<15 - 1))
		binary.LittleEndian.PutUint16(b[i*4:], uint16(s))
		binary.LittleEndian.PutUint16(b[i*4+2:], uint16(s))
	}
	return b
}

// decodeI16StereoMono decodes stereo int16 bytes and returns the left channel.
func decodeI16StereoMono(b []byte) []float64 {
	n := len(b) / 4
	out := make([]float64, n)
	for i := range n {
		out[i] = float64(int16(binary.LittleEndian.Uint16(b[i*4:]))) / (1<<15 - 1)
	}
	return out
}
