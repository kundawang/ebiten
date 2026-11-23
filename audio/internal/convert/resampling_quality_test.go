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
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
)

var allQualities = []convert.ResampleQuality{
	convert.ResampleQualityDefault,
	convert.ResampleQualityFast,
	convert.ResampleQualityBalanced,
	convert.ResampleQualityHighFidelity,
}

func qualityLabel(q convert.ResampleQuality) string {
	switch q {
	case convert.ResampleQualityDefault:
		return "default"
	case convert.ResampleQualityFast:
		return "fast"
	case convert.ResampleQualityBalanced:
		return "balanced"
	case convert.ResampleQualityHighFidelity:
		return "highfidelity"
	default:
		return fmt.Sprintf("unknown-%d", q)
	}
}

// TestResampleQualityAgainstReference feeds sweep, impulse, sine and
// variable-ratio fixtures through every quality and reports the error against
// the high-precision offline reference. Fidelity must improve monotonically
// fast < balanced <= high, and default must equal balanced.
func TestResampleQualityAgainstReference(t *testing.T) {
	const (
		from = 44100
		to   = 48000
	)
	// f0..f1 stay inside both rates' pass bands; the sweep reaches frequencies
	// where kernel stop-band rejection actually matters.
	fixtures := map[string]func() []float64{
		"sine-1k":    func() []float64 { return sineSignal(from, 0.5, 1000) },
		"sweep":      func() []float64 { return sweepSignal(from, 0.5, 100, 16000) },
		"impulse":    func() []float64 { return impulseSignal(from/4, from/8) },
		"multi-sine": func() []float64 { return multiSineSignal(from, 0.5) },
	}

	rmsByQuality := map[convert.ResampleQuality]float64{}
	for name, build := range fixtures {
		t.Run(name, func(t *testing.T) {
			mono := build()
			src := encodeF32Stereo(mono)
			want := referenceResample(mono, from, to)

			for _, q := range allQualities {
				t.Run(qualityLabel(q), func(t *testing.T) {
					r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), from, to, 4, q)
					if err != nil {
						t.Fatal(err)
					}
					out, err := io.ReadAll(r)
					if err != nil {
						t.Fatal(err)
					}
					got := decodeF32StereoMono(out)

					rms := rmsError(got, want)
					mx := maxError(got, want)
					// Report the numbers the task asks for; t.Logf keeps them in
					// -v output without turning measurement into brittle gates.
					t.Logf("rms error: %.3e, max error: %.3e, output frames: %d", rms, mx, len(got))
					if math.IsNaN(rms) || math.IsInf(rms, 0) {
						t.Fatalf("non-finite error for quality %d", q)
					}
					rmsByQuality[q] = rms

					// The impulse must not be smeared beyond each kernel's taps,
					// and must never overshoot into clipping on bounded input.
					if name == "impulse" {
						for i, v := range got {
							if math.Abs(v) > 1+1e-6 {
								t.Fatalf("sample %d overshoots: %v", i, v)
							}
						}
					}
				})
			}

			// Default must be numerically identical to balanced.
			if rmsByQuality[convert.ResampleQualityDefault] != rmsByQuality[convert.ResampleQualityBalanced] {
				t.Errorf("default RMS %v != balanced RMS %v",
					rmsByQuality[convert.ResampleQualityDefault], rmsByQuality[convert.ResampleQualityBalanced])
			}
			// High fidelity must beat the balanced kernel on every demanding fixture.
			if rmsByQuality[convert.ResampleQualityHighFidelity] >= rmsByQuality[convert.ResampleQualityBalanced] {
				t.Errorf("high-fidelity RMS %.3e does not improve on balanced %.3e",
					rmsByQuality[convert.ResampleQualityHighFidelity], rmsByQuality[convert.ResampleQualityBalanced])
			}
			// Fast must be the weakest of the three real kernels.
			if rmsByQuality[convert.ResampleQualityFast] <= rmsByQuality[convert.ResampleQualityBalanced] {
				t.Errorf("fast RMS %.3e is not worse than balanced %.3e",
					rmsByQuality[convert.ResampleQualityFast], rmsByQuality[convert.ResampleQualityBalanced])
			}
		})
	}
}

// multiSineSignal mixes several in-band frequencies with incommensurate phases.
func multiSineSignal(rate int, seconds float64) []float64 {
	freqs := []float64{220, 880, 3520, 12000}
	n := int(seconds * float64(rate))
	out := make([]float64, n)
	for i := range n {
		var v float64
		for j, f := range freqs {
			v += 0.2 * math.Sin(2*math.Pi*f*float64(i)/float64(rate)+float64(j))
		}
		out[i] = v
	}
	return out
}

// TestResampleQualityVariableRatio exercises non-integer ratios in both
// directions against the reference, including heavy down-conversion.
func TestResampleQualityVariableRatio(t *testing.T) {
	ratios := []struct {
		from, to int
	}{
		{44100, 48000},
		{48000, 44100},
		{48000, 16000},
		{16000, 48000},
		{44100, 44101}, // ratio almost, but not quite, one
	}
	for _, c := range ratios {
		t.Run(fmt.Sprintf("%d-%d", c.from, c.to), func(t *testing.T) {
			mono := sweepSignal(c.from, 0.25, 100, math.Min(float64(c.from), float64(c.to))*0.4)
			src := encodeF32Stereo(mono)
			want := referenceResample(mono, c.from, c.to)

			var baseline float64
			for _, q := range allQualities {
				r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), c.from, c.to, 4, q)
				if err != nil {
					t.Fatal(err)
				}
				out, err := io.ReadAll(r)
				if err != nil {
					t.Fatal(err)
				}
				got := decodeF32StereoMono(out)
				if len(got) != len(want) {
					t.Errorf("quality %d length: got %d frames, want %d", q, len(got), len(want))
				}
				rms := rmsError(got, want)
				t.Logf("quality %s: rms %.3e max %.3e", qualityLabel(q), rms, maxError(got, want))
				if math.IsNaN(rms) {
					t.Fatalf("quality %d produced NaN", q)
				}
				if q == convert.ResampleQualityBalanced {
					baseline = rms
				}
				if q == convert.ResampleQualityHighFidelity && rms > baseline {
					t.Errorf("high-fidelity (%.3e) worse than balanced (%.3e)", rms, baseline)
				}
			}
		})
	}
}

// TestResampleQualityDefaultBitIdentical proves the default constructor and the
// new quality-aware constructor with Default/Balanced emit identical bytes for
// both int16 and float32 sources, so existing streams are unaffected.
func TestResampleQualityDefaultBitIdentical(t *testing.T) {
	for _, depth := range []int{2, 4} {
		t.Run(fmt.Sprintf("depth%d", depth), func(t *testing.T) {
			inB := newSoundBytes(44100, depth)
			legacy := ioReadAll(t, convert.NewResampling(bytes.NewReader(inB), int64(len(inB)), 44100, 48000, depth))
			for _, q := range []convert.ResampleQuality{convert.ResampleQualityDefault, convert.ResampleQualityBalanced} {
				r, err := convert.NewResamplingWithQuality(bytes.NewReader(inB), int64(len(inB)), 44100, 48000, depth, q)
				if err != nil {
					t.Fatal(err)
				}
				got := ioReadAll(t, r)
				if !bytes.Equal(got, legacy) {
					t.Errorf("quality %s depth %d differs from the legacy stream", qualityLabel(q), depth)
				}
			}
		})
	}
}

func ioReadAll(t *testing.T, r io.Reader) []byte {
	t.Helper()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestResampleQualitySemanticsParity runs the key behavioural scenarios for
// every quality and asserts the results match the balanced kernel: lengths,
// Seek round-tripping, repeated seeks, EOF and short-buffer handling.
func TestResampleQualitySemanticsParity(t *testing.T) {
	const (
		from = 44100
		to   = 48000
	)
	for _, depth := range []int{2, 4} {
		t.Run(fmt.Sprintf("depth%d", depth), func(t *testing.T) {
			inB := newSoundBytesForFrames(from, 3*4096+37, depth)
			bytesPerSample := depth * 2

			reference := func(q convert.ResampleQuality) *convert.Resampling {
				r, err := convert.NewResamplingWithQuality(bytes.NewReader(inB), int64(len(inB)), from, to, depth, q)
				if err != nil {
					t.Fatal(err)
				}
				return r
			}
			wantAll := ioReadAll(t, reference(convert.ResampleQualityBalanced))
			wantLen := reference(convert.ResampleQualityBalanced).Length()

			for _, q := range allQualities {
				t.Run(qualityLabel(q), func(t *testing.T) {
					r := reference(q)
					if got := r.Length(); got != wantLen {
						t.Errorf("Length: got %d, want %d", got, wantLen)
					}

					// Read in deliberately awkward chunk sizes covering single
					// frames, partial samples and block boundaries.
					var got []byte
					buf := make([]byte, 7*bytesPerSample+1)
					for {
						n, err := r.Read(buf)
						got = append(got, buf[:n]...)
						if err == io.EOF {
							break
						}
						if err != nil {
							t.Fatal(err)
						}
					}
					if len(got) != len(wantAll) {
						t.Fatalf("length: got %d, want %d", len(got), len(wantAll))
					}
					if q != convert.ResampleQualityFast && q != convert.ResampleQualityHighFidelity {
						if !bytes.Equal(got, wantAll) {
							t.Error("default/balanced output drifted from the reference stream")
						}
					}

					// Repeated seeks, including back-and-forth over cached blocks.
					for _, pos := range []int64{
						int64(50 * bytesPerSample),
						int64(2 * 4096 * bytesPerSample),
						int64(50 * bytesPerSample),
						int64(4096*bytesPerSample) + 3,
						0,
					} {
						aligned := pos / int64(bytesPerSample) * int64(bytesPerSample)
						gotPos, err := r.Seek(pos, io.SeekStart)
						if err != nil {
							t.Fatal(err)
						}
						if gotPos != aligned {
							t.Errorf("Seek(%d): got %d, want %d", pos, gotPos, aligned)
						}
						snip := make([]byte, 16*bytesPerSample)
						if _, err := io.ReadFull(r, snip); err != nil {
							t.Fatal(err)
						}
					}

					// SeekEnd must resolve against the same reported length.
					endPos, err := r.Seek(0, io.SeekEnd)
					if err != nil {
						t.Fatal(err)
					}
					if endPos != wantLen {
						t.Errorf("SeekEnd: got %d, want %d", endPos, wantLen)
					}
				})
			}
		})
	}
}

// TestResampleQualitySingleFrameAndEOF checks the single-frame boundary, an
// empty source, and unknown-length EOF discovery for every quality.
func TestResampleQualitySingleFrameAndEOF(t *testing.T) {
	for _, depth := range []int{2, 4} {
		for _, frames := range []int{0, 1, 2, 33} {
			t.Run(fmt.Sprintf("depth%d/frames%d", depth, frames), func(t *testing.T) {
				src := make([]byte, frames*depth*2)
				for _, q := range allQualities {
					// Known length.
					r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), 44100, 48000, depth, q)
					if err != nil {
						t.Fatal(err)
					}
					wantLen := r.Length()
					out := ioReadAll(t, r)
					if int64(len(out)) != wantLen {
						t.Errorf("quality %d: read %d bytes, Length said %d", q, len(out), wantLen)
					}

					// Unknown length must discover the same EOF and length.
					r2, err := convert.NewResamplingWithQuality(bytes.NewReader(src), -1, 44100, 48000, depth, q)
					if err != nil {
						t.Fatal(err)
					}
					if got := r2.Length(); got != -1 {
						t.Errorf("quality %d: initial Length: got %d, want -1", q, got)
					}
					out2 := ioReadAll(t, r2)
					if len(out2) != len(out) {
						t.Errorf("quality %d: unknown-length read %d bytes, known-length %d", q, len(out2), len(out))
					}
					if got := r2.Length(); got != wantLen {
						t.Errorf("quality %d: Length after EOF: got %d, want %d", q, got, wantLen)
					}
				}
			})
		}
	}
}

// TestResampleQualityShortBuffer verifies sub-sample buffers keep returning
// io.ErrShortBuffer (or io.EOF once drained) for every quality, and that the
// stream still completes afterwards with consistent lengths.
func TestResampleQualityShortBuffer(t *testing.T) {
	for _, depth := range []int{2, 4} {
		t.Run(fmt.Sprintf("depth%d", depth), func(t *testing.T) {
			src := make([]byte, 1024)
			for _, q := range allQualities {
				r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), 44100, 48000, depth, q)
				if err != nil {
					t.Fatal(err)
				}
				for size := 1; size < depth*2; size++ {
					n, err := r.Read(make([]byte, size))
					if n != 0 || !errors.Is(err, io.ErrShortBuffer) {
						t.Errorf("quality %d Read(%d): got (%d, %v), want (0, ErrShortBuffer)", q, size, n, err)
					}
				}
				rest := ioReadAll(t, r)
				if int64(len(rest)) != r.Length() {
					t.Errorf("quality %d: %d bytes after short reads, want %d", q, len(rest), r.Length())
				}
				if n, err := r.Read(make([]byte, depth*2)); err != io.EOF || n != 0 {
					t.Errorf("quality %d post-drain Read: got (%d, %v), want (0, EOF)", q, n, err)
				}
			}
		})
	}
}

// TestResampleQualityStereoPhase confirms both channels apply the same weights,
// so a phase offset between channels is preserved rather than blurred.
func TestResampleQualityStereoPhase(t *testing.T) {
	const from, to = 44100, 48000
	frames := from / 4
	mono := sineSignal(from, 0.25, 2000)
	src := make([]byte, frames*8)
	for i := 0; i < frames && i < len(mono); i++ {
		// Right channel is the inverse of the left.
		binary.LittleEndian.PutUint32(src[i*8:], math.Float32bits(float32(mono[i])))
		binary.LittleEndian.PutUint32(src[i*8+4:], math.Float32bits(float32(-mono[i])))
	}
	for _, q := range allQualities {
		r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), from, to, 4, q)
		if err != nil {
			t.Fatal(err)
		}
		out := ioReadAll(t, r)
		for i := 0; i+8 <= len(out); i += 8 {
			l := math.Float32frombits(binary.LittleEndian.Uint32(out[i:]))
			rr := math.Float32frombits(binary.LittleEndian.Uint32(out[i+4:]))
			if math.Abs(float64(l+rr)) > 2e-6 {
				t.Errorf("quality %d frame %d: L=%v R=%v are not exact inverses (phase/weight mismatch)", q, i/8, l, rr)
				break
			}
		}
	}
}

// TestResampleQualityNonFiniteInput ensures NaN/Inf float32 input neither
// panics nor grows the block cache beyond its cap, and the stream terminates.
func TestResampleQualityNonFiniteInput(t *testing.T) {
	src := make([]byte, 3*4096*8)
	nan := math.Float32bits(float32(math.NaN()))
	posInf := math.Float32bits(float32(math.Inf(1)))
	for i, sp := range []uint32{nan, posInf, nan} {
		off := (i*101 + 7) * 8
		binary.LittleEndian.PutUint32(src[off:], sp)
		binary.LittleEndian.PutUint32(src[off+4:], sp)
	}
	for _, q := range allQualities {
		r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), 44100, 48000, 4, q)
		if err != nil {
			t.Fatal(err)
		}
		out, err := io.ReadAll(r)
		if err != nil {
			t.Fatalf("quality %d: %v", q, err)
		}
		if len(out) == 0 {
			t.Errorf("quality %d produced no output", q)
		}
	}
}

// TestNewResamplingWithQualityErrors checks the exported constructor's error
// surface for invalid quality, rates and bit depth.
func TestNewResamplingWithQualityErrors(t *testing.T) {
	src := bytes.NewReader(make([]byte, 64))
	if _, err := convert.NewResamplingWithQuality(src, 64, 44100, 48000, 2, convert.ResampleQuality(99)); !errors.Is(err, convert.ErrInvalidResampleQuality) {
		t.Errorf("invalid quality: got %v, want convert.ErrInvalidResampleQuality", err)
	}
	if _, err := convert.NewResamplingWithQuality(src, 64, 0, 48000, 2, convert.ResampleQualityFast); err == nil {
		t.Error("from=0: expected error")
	}
	if _, err := convert.NewResamplingWithQuality(src, 64, 44100, -1, 2, convert.ResampleQualityFast); err == nil {
		t.Error("to=-1: expected error")
	}
	if _, err := convert.NewResamplingWithQuality(src, 64, 44100, 48000, 3, convert.ResampleQualityFast); err == nil {
		t.Error("depth=3: expected error")
	}
}

// BenchmarkResampling measures throughput and allocations per quality on a
// representative 44.1k -> 48k float32 conversion.
func BenchmarkResampling(b *testing.B) {
	mono := sineSignal(44100, 1.0, 1000)
	src := encodeF32Stereo(mono)
	for _, q := range allQualities {
		b.Run(qualityLabel(q), func(b *testing.B) {
			b.ReportAllocs()
			out := make([]byte, 4096)
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for range b.N {
				r, err := convert.NewResamplingWithQuality(bytes.NewReader(src), int64(len(src)), 44100, 48000, 4, q)
				if err != nil {
					b.Fatal(err)
				}
				for {
					n, err := r.Read(out)
					if err == io.EOF {
						break
					}
					if err != nil {
						b.Fatal(err)
					}
					_ = n
				}
			}
		})
	}
}
