// Copyright 2026 The Ebitengine Authors
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

package audio_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// crossfadeFormat describes one PCM encoding used by the crossfade tests.
type crossfadeFormat struct {
	name     string
	bitDepth int
	newLoop  func(io.ReadSeeker, audio.LoopCrossfadePlan) (*audio.InfiniteLoop, error)
	// encode writes one frame (two channel values) to dst.
	encode func(dst []byte, left, right float64)
	// decode reads one frame from b.
	decode func(b []byte) (left, right float64)
	// toBytes converts a mixed channel value back to its encoded form.
	mix func(dst []byte, orig, tail []byte, loopGain, afterLoopGain float64)
}

func int16CrossfadeFormat() crossfadeFormat {
	return crossfadeFormat{
		name:     "int16",
		bitDepth: 2,
		newLoop:  audio.NewInfiniteLoopWithCrossfade,
		encode: func(dst []byte, left, right float64) {
			binary.LittleEndian.PutUint16(dst[0:], uint16(int16(left)))
			binary.LittleEndian.PutUint16(dst[2:], uint16(int16(right)))
		},
		decode: func(b []byte) (float64, float64) {
			return float64(int16(binary.LittleEndian.Uint16(b[0:]))), float64(int16(binary.LittleEndian.Uint16(b[2:])))
		},
		mix: func(dst []byte, orig, tail []byte, loopGain, afterLoopGain float64) {
			o := int16(binary.LittleEndian.Uint16(orig))
			a := int16(binary.LittleEndian.Uint16(tail))
			binary.LittleEndian.PutUint16(dst, uint16(int16(float64(a)*afterLoopGain+float64(o)*loopGain)))
		},
	}
}

func float32CrossfadeFormat() crossfadeFormat {
	return crossfadeFormat{
		name:     "float32",
		bitDepth: 4,
		newLoop:  audio.NewInfiniteLoopWithCrossfadeF32,
		encode: func(dst []byte, left, right float64) {
			binary.LittleEndian.PutUint32(dst[0:], math.Float32bits(float32(left)))
			binary.LittleEndian.PutUint32(dst[4:], math.Float32bits(float32(right)))
		},
		decode: func(b []byte) (float64, float64) {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[0:]))), float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4:])))
		},
		mix: func(dst []byte, orig, tail []byte, loopGain, afterLoopGain float64) {
			o := math.Float32frombits(binary.LittleEndian.Uint32(orig))
			a := math.Float32frombits(binary.LittleEndian.Uint32(tail))
			binary.LittleEndian.PutUint32(dst, math.Float32bits(float32(float64(a)*afterLoopGain+float64(o)*loopGain)))
		},
	}
}

func crossfadeFormats() []crossfadeFormat {
	return []crossfadeFormat{int16CrossfadeFormat(), float32CrossfadeFormat()}
}

// encodeCrossfadeFrames encodes frames as a stereo stream. value returns the frame's channel values.
func encodeCrossfadeFrames(f crossfadeFormat, frames int, value func(frame int) (float64, float64)) []byte {
	data := make([]byte, frames*f.bitDepth*2)
	for i := range frames {
		left, right := value(i)
		f.encode(data[i*f.bitDepth*2:], left, right)
	}
	return data
}

func crossfadeBytesPerSample(f crossfadeFormat) int64 {
	return int64(f.bitDepth * 2)
}

// crossfadeWindow returns the number of frames actually crossfaded with the given plan and source.
//
// It mirrors the documented degradation rules: the window is shortened to the loop and to the whole
// frames the source has after the loop end.
func crossfadeWindow(plan audio.LoopCrossfadePlan, bitDepth int, srcLen int64) int64 {
	bps := int64(bitDepth * 2)
	llength := plan.LoopLength / bps * bps
	frames := min(plan.FadeFrames, llength/bps, int64(8<<20)/bps)
	length := plan.IntroLength/bps*bps + llength
	tailFrames := (srcLen - length) / bps
	if tailFrames < 0 {
		tailFrames = 0
	}
	return max(min(frames, tailFrames), 0)
}

// crossfadeGainsRef mirrors the gain computation of the implementation.
func crossfadeGainsRef(curve audio.LoopCrossfadeCurve, frame, window int64) (loopGain, afterLoopGain float64) {
	if window <= 0 || frame < 0 || frame >= window {
		return 1, 0
	}
	t := float64(frame) / float64(window)
	loopGain, afterLoopGain = curve(t)
	if math.IsNaN(loopGain) || math.IsInf(loopGain, 0) || math.IsNaN(afterLoopGain) || math.IsInf(afterLoopGain, 0) {
		loopGain, afterLoopGain = audio.LoopCrossfadeLinear(t)
	}
	clamp := func(v float64) float64 {
		return min(max(v, 0), 1)
	}
	return clamp(loopGain), clamp(afterLoopGain)
}

// offlineCrossfadeMix independently mixes the intro and loop repetitions of src, applying the
// crossfade with the tail that follows the loop end.
//
// captured reports whether the tail was already captured: the first lap is raw because a sequential
// reader reaches the loop end, and therefore captures the tail, only after emitting the first lap.
// With captured set, every lap is faded, as it is after a seek once the tail was captured.
func offlineCrossfadeMix(f crossfadeFormat, src []byte, plan audio.LoopCrossfadePlan, outFrames int, curve audio.LoopCrossfadeCurve, captured bool) []byte {
	bps := crossfadeBytesPerSample(f)
	bitDepth := int64(f.bitDepth)
	lstart := plan.IntroLength / bps * bps
	llength := plan.LoopLength / bps * bps
	length := lstart + llength
	if curve == nil {
		curve = audio.LoopCrossfadeLinear
	}
	window := crossfadeWindow(plan, f.bitDepth, int64(len(src)))

	out := make([]byte, 0, outFrames*int(bps))
	for i := range outFrames {
		pos := int64(i) * bps
		if pos < lstart {
			out = append(out, src[pos:pos+bps]...)
			continue
		}
		rel := (pos - lstart) % llength
		lap := (pos - lstart) / llength
		srcOff := lstart + rel
		orig := src[srcOff : srcOff+bps]

		faded := captured || lap > 0
		frame := rel / bps
		if !faded || frame >= window {
			out = append(out, orig...)
			continue
		}

		tail := src[length+rel : length+rel+bps]
		loopGain, afterLoopGain := crossfadeGainsRef(curve, frame, window)
		frameOut := make([]byte, bps)
		for ch := range 2 {
			chOff := int64(ch) * bitDepth
			f.mix(frameOut[chOff:], orig[chOff:], tail[chOff:], loopGain, afterLoopGain)
		}
		out = append(out, frameOut...)
	}
	return out
}

// readAllReadSizes reads exactly n bytes from r, requesting the given buffer sizes in rotation.
func readAllReadSizes(t *testing.T, r io.Reader, n int, sizes []int) []byte {
	t.Helper()
	out := make([]byte, 0, n)
	for i := 0; len(out) < n; i++ {
		size := sizes[i%len(sizes)]
		buf := make([]byte, size)
		m, err := r.Read(buf)
		if m > 0 {
			out = append(out, buf[:m]...)
		}
		if err != nil {
			if errors.Is(err, io.ErrShortBuffer) {
				continue
			}
			t.Fatalf("Read(a buffer of %d bytes) at %d: %v", size, len(out), err)
		}
	}
	return out[:n]
}

func crossfadeTestSource(f crossfadeFormat, frames int, step bool) []byte {
	switch {
	case step:
		// A step fixture: a silence, a jump, and a different value after the loop end, so that every
		// boundary sample of a crossfade is distinguishable.
		return encodeCrossfadeFrames(f, frames, func(i int) (float64, float64) {
			switch {
			case i < 4:
				return 0, 0
			case i < frames-2:
				return 0.5, -0.25
			default:
				return 0.75, 0.125
			}
		})
	default:
		r := float64(frames)
		return encodeCrossfadeFrames(f, frames, func(i int) (float64, float64) {
			s := math.Sin(2 * math.Pi * 3 * float64(i) / r)
			return 0.75 * s, 0.5 * math.Cos(2*math.Pi*5*float64(i)/r)
		})
	}
}

func TestInfiniteLoopCrossfadeMatchesOfflineMix(t *testing.T) {
	const (
		introFrames = 5
		loopFrames  = 16
		tailFrames  = 6
		fadeFrames  = 4
		lapCount    = 4
	)
	curves := []struct {
		name  string
		curve audio.LoopCrossfadeCurve
	}{
		{"linear", audio.LoopCrossfadeLinear},
		{"equal-power", audio.LoopCrossfadeEqualPower},
		{"nil", nil},
	}

	for _, f := range crossfadeFormats() {
		bps := crossfadeBytesPerSample(f)
		for _, curve := range curves {
			for _, step := range []bool{false, true} {
				t.Run(f.name+"/"+curve.name+"/step="+map[bool]string{true: "yes", false: "no"}[step], func(t *testing.T) {
					plan := audio.LoopCrossfadePlan{
						IntroLength: int64(introFrames) * bps,
						LoopLength:  int64(loopFrames) * bps,
						FadeFrames:  fadeFrames,
						Curve:       curve.curve,
					}
					fixture := crossfadeTestSource(f, introFrames+loopFrames+tailFrames, step)
					outFrames := introFrames + loopFrames*lapCount
					want := offlineCrossfadeMix(f, fixture, plan, outFrames, curve.curve, false)

					l, err := f.newLoop(bytes.NewReader(fixture), plan)
					if err != nil {
						t.Fatal(err)
					}
					wantLen := int64(outFrames) * bps
					if got := l.Length(); got != int64(introFrames+loopFrames)*bps {
						t.Errorf("Length: got %d, want %d", got, int64(introFrames+loopFrames)*bps)
					}
					got := readAllReadSizes(t, l, int(wantLen), []int{1, 3, 5, 7, 11, 13, 17})

					// Read in short buffers must append up to whole values only; the helper reads
					// exactly wantLen bytes, which are whole frames.
					if !bytes.Equal(got, want) {
						for i := 0; i < len(got) && i < len(want); i++ {
							if got[i] != want[i] {
								t.Fatalf("the audio differs at byte %d: got %v, want %v", i, got[i], want[i])
							}
						}
						t.Fatalf("the audio differs in length: got %d, want %d", len(got), len(want))
					}

					// The first frame of every faded lap must be exactly the first frame after the
					// loop end, since the crossfade starts with the tail at full gain.
					tailStart := int64(introFrames+loopFrames) * bps
					tail := fixture[tailStart : tailStart+bps]
					for lap := 1; lap < lapCount; lap++ {
						off := int64(introFrames+loopFrames*lap) * bps
						if got := want[off : off+bps]; !bytes.Equal(got, tail) {
							t.Fatalf("lap %d boundary frame: got %v, want the tail frame %v", lap, got, tail)
						}
					}
				})
			}
		}
	}
}

func TestInfiniteLoopCrossfadeWholeFrameBoundaries(t *testing.T) {
	// The source has one whole frame less than the requested fade plus a trailing partial frame. The
	// window must end at the last whole frame, and the trailing partial frame must be dropped.
	const (
		introFrames = 2
		loopFrames  = 8
		fadeFrames  = 4
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, introFrames+loopFrames+2, true)
			// Append a partial frame: it cannot be mixed with a whole frame.
			fixture = append(fixture, 1, 2, 3)

			plan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
				FadeFrames:  fadeFrames,
				Curve:       audio.LoopCrossfadeLinear,
			}
			// The source has only two whole frames after the loop end, so the window is two frames.
			want := offlineCrossfadeMix(f, fixture, plan, introFrames+loopFrames*3, audio.LoopCrossfadeLinear, false)
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			got := readAllReadSizes(t, l, len(want), []int{int(bps), int(bps) + 1, 3})
			if !bytes.Equal(got, want) {
				t.Fatalf("got %v, want %v", got, want)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeDisabled(t *testing.T) {
	const (
		introFrames = 3
		loopFrames  = 12
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, introFrames+loopFrames+5, false)

			// FadeFrames == 0 must behave like a loop with no blending at all, including not reading
			// the data after the loop end.
			plan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
			}
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			got := readAllReadSizes(t, l, (introFrames+loopFrames*3)*int(bps), []int{5, 7, 9})
			want := offlineCrossfadeMix(f, fixture, plan, introFrames+loopFrames*3, audio.LoopCrossfadeLinear, false)
			if !bytes.Equal(got, want) {
				t.Fatalf("got %v, want %v", got, want)
			}

			// A legacy loop with the same regions and blending disabled must produce the same bytes.
			var legacy *audio.InfiniteLoop
			switch f.bitDepth {
			case 2:
				legacy = audio.NewInfiniteLoopWithIntro(bytes.NewReader(fixture), plan.IntroLength, plan.LoopLength)
			case 4:
				legacy = audio.NewInfiniteLoopWithIntroF32(bytes.NewReader(fixture), plan.IntroLength, plan.LoopLength)
			default:
				t.Fatal("unexpected bit depth")
			}
			legacy.SetNoBlendForTesting(true)
			legacyGot := readAllReadSizes(t, legacy, (introFrames+loopFrames*3)*int(bps), []int{5, 7, 9})
			if !bytes.Equal(got, legacyGot) {
				t.Fatalf("a disabled crossfade differs from a legacy loop without blending: got %v, want %v", got, legacyGot)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeDoesNotChangePositions(t *testing.T) {
	const (
		introFrames = 3
		loopFrames  = 10
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, introFrames+loopFrames+4, false)
			length := int64(introFrames+loopFrames) * bps

			plainPlan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
			}
			fadedPlan := plainPlan
			fadedPlan.FadeFrames = 4

			plain, err := f.newLoop(bytes.NewReader(fixture), plainPlan)
			if err != nil {
				t.Fatal(err)
			}
			faded, err := f.newLoop(bytes.NewReader(fixture), fadedPlan)
			if err != nil {
				t.Fatal(err)
			}
			if plain.Length() != length || faded.Length() != length {
				t.Fatalf("Length: got %d and %d, want %d", plain.Length(), faded.Length(), length)
			}

			offsets := []int64{-1, 0, 1, int64(bps) - 1, int64(bps), length - 1, length, length + 1, length * 3, length*7 + 2}
			for _, whence := range []int{io.SeekStart, io.SeekCurrent} {
				for _, offset := range offsets {
					want, wantErr := plain.Seek(offset, whence)
					got, gotErr := faded.Seek(offset, whence)
					if (gotErr != nil) != (wantErr != nil) {
						t.Fatalf("Seek(%d, %d): got error %v, want %v", offset, whence, gotErr, wantErr)
					}
					if got != want {
						t.Fatalf("Seek(%d, %d): got %d, want %d", offset, whence, got, want)
					}
				}
			}

			// A query must report the same position and must not move either stream.
			wantPos, wantErr := plain.Seek(0, io.SeekCurrent)
			gotPos, gotErr := faded.Seek(0, io.SeekCurrent)
			if (gotErr != nil) != (wantErr != nil) || gotPos != wantPos {
				t.Fatalf("Seek(0, io.SeekCurrent): got (%d, %v), want (%d, %v)", gotPos, gotErr, wantPos, wantErr)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeMultipleSeeks(t *testing.T) {
	const (
		introFrames = 2
		loopFrames  = 12
		fadeFrames  = 5
		laps        = 3
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			plan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
				FadeFrames:  fadeFrames,
				Curve:       audio.LoopCrossfadeEqualPower,
			}
			fixture := crossfadeTestSource(f, introFrames+loopFrames+7, true)
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}

			// Read the first lap, which also captures the data after the loop end.
			firstLap := readAllReadSizes(t, l, int(l.Length()), []int{7, 11})
			wantFirst := offlineCrossfadeMix(f, fixture, plan, introFrames+loopFrames, audio.LoopCrossfadeEqualPower, false)
			if !bytes.Equal(firstLap, wantFirst) {
				t.Fatalf("the first lap: got %v, want %v", firstLap, wantFirst)
			}

			// From now on the crossfade depends only on the position, whatever seeks happened.
			for _, tc := range []struct {
				offset int64
				frames int
			}{
				{0, introFrames + loopFrames},
				{int64(introFrames) * bps, loopFrames * 2},
				{int64(introFrames+loopFrames/2) * bps, loopFrames},
				{l.Length() + int64(introFrames+2)*bps, loopFrames},
				{-1, 0},
			} {
				if tc.frames == 0 {
					if _, err := l.Seek(tc.offset, io.SeekStart); err == nil {
						t.Fatalf("Seek(%d, io.SeekStart): got no error, want an error", tc.offset)
					}
					continue
				}
				pos, err := l.Seek(tc.offset, io.SeekStart)
				if err != nil {
					t.Fatalf("Seek(%d, io.SeekStart): %v", tc.offset, err)
				}
				got := readAllReadSizes(t, l, tc.frames*int(bps), []int{3, 5, 9})

				// Build the expected data from the stream position the Seek folded the offset to.
				want := make([]byte, 0, tc.frames)
				window := crossfadeWindow(plan, f.bitDepth, int64(len(fixture)))
				length := int64(introFrames+loopFrames) * bps
				llength := int64(loopFrames) * bps
				lstart := int64(introFrames) * bps
				for i := range tc.frames {
					p := pos + int64(i)*bps
					if p < lstart {
						want = append(want, fixture[p:p+bps]...)
						continue
					}
					rel := (p - lstart) % llength
					srcOff := lstart + rel
					frame := rel / bps
					if frame >= window {
						want = append(want, fixture[srcOff:srcOff+bps]...)
						continue
					}
					tail := fixture[length+rel : length+rel+bps]
					loopGain, afterLoopGain := crossfadeGainsRef(audio.LoopCrossfadeEqualPower, frame, window)
					out := make([]byte, bps)
					for ch := range 2 {
						chOff := int64(ch) * int64(f.bitDepth)
						f.mix(out[chOff:], fixture[srcOff+chOff:], tail[chOff:], loopGain, afterLoopGain)
					}
					want = append(want, out...)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("Seek(%d) then %d frames: got %v, want %v", tc.offset, tc.frames, got, want)
				}
			}
		})
	}
}

func TestInfiniteLoopCrossfadeNonFiniteCurve(t *testing.T) {
	const (
		loopFrames = 12
		fadeFrames = 4
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames+6, true)
			base := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: fadeFrames,
				Curve:      audio.LoopCrossfadeLinear,
			}
			want := offlineCrossfadeMix(f, fixture, base, loopFrames*2, audio.LoopCrossfadeLinear, false)

			for _, tc := range []struct {
				name  string
				curve audio.LoopCrossfadeCurve
			}{
				{"nan", func(float64) (float64, float64) { return math.NaN(), math.NaN() }},
				{"inf", func(float64) (float64, float64) { return math.Inf(1), math.Inf(-1) }},
				{"mixed", func(t float64) (float64, float64) { return math.NaN(), t }},
			} {
				plan := base
				plan.Curve = tc.curve
				l, err := f.newLoop(bytes.NewReader(fixture), plan)
				if err != nil {
					t.Fatalf("%s: %v", tc.name, err)
				}
				got := readAllReadSizes(t, l, len(want), []int{int(bps), int(bps) + 3, int(bps) + 7})
				if !bytes.Equal(got, want) {
					t.Errorf("%s: a non-finite curve must degrade to the linear curve: got %v, want %v", tc.name, got, want)
				}
			}
		})
	}
}

func TestInfiniteLoopCrossfadePlanValidation(t *testing.T) {
	cases := []struct {
		name string
		plan audio.LoopCrossfadePlan
	}{
		{"ZeroLoop", audio.LoopCrossfadePlan{LoopLength: 0, FadeFrames: 1}},
		{"NegativeLoop", audio.LoopCrossfadePlan{LoopLength: -4, FadeFrames: 1}},
		{"LoopShorterThanAFrame", audio.LoopCrossfadePlan{LoopLength: 3, FadeFrames: 1}},
		{"NegativeIntro", audio.LoopCrossfadePlan{IntroLength: -1, LoopLength: 16}},
		{"NegativeFade", audio.LoopCrossfadePlan{LoopLength: 16, FadeFrames: -1}},
		{"Overflow", audio.LoopCrossfadePlan{IntroLength: math.MaxInt64, LoopLength: math.MaxInt64}},
	}
	for _, f := range crossfadeFormats() {
		for _, tc := range cases {
			t.Run(f.name+"/"+tc.name, func(t *testing.T) {
				src := bytes.NewReader(make([]byte, 1024))
				if l, err := f.newLoop(src, tc.plan); err == nil {
					t.Fatalf("got no error for %+v, want an error (stream: %v)", tc.plan, l != nil)
				}
			})
		}
	}
}

func TestInfiniteLoopCrossfadeLongFadeIsClamped(t *testing.T) {
	const loopFrames = 8
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames*3, true)
			// A fade much longer than the loop must not error and must behave like a full-loop fade.
			longPlan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 1 << 40,
			}
			fullPlan := longPlan
			fullPlan.FadeFrames = loopFrames

			long, err := f.newLoop(bytes.NewReader(fixture), longPlan)
			if err != nil {
				t.Fatal(err)
			}
			full, err := f.newLoop(bytes.NewReader(fixture), fullPlan)
			if err != nil {
				t.Fatal(err)
			}
			// The source has only twice the loop after the loop end, so the window is the loop.
			gotLong := readAllReadSizes(t, long, loopFrames*3*int(bps), []int{int(bps), int(bps) + 5})
			gotFull := readAllReadSizes(t, full, loopFrames*3*int(bps), []int{int(bps), int(bps) + 5})
			if !bytes.Equal(gotLong, gotFull) {
				t.Fatalf("a long fade differs from a full-loop fade: got %v, want %v", gotLong, gotFull)
			}
			if long.Length() != int64(loopFrames)*bps {
				t.Fatalf("Length: got %d, want %d", long.Length(), int64(loopFrames)*bps)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeShortSource(t *testing.T) {
	const (
		introFrames = 2
		loopFrames  = 10
		tailFrames  = 3
		fadeFrames  = 6
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, introFrames+loopFrames+tailFrames, true)
			plan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
				FadeFrames:  fadeFrames,
				Curve:       audio.LoopCrossfadeLinear,
			}
			want := offlineCrossfadeMix(f, fixture, plan, introFrames+loopFrames*3, audio.LoopCrossfadeLinear, false)
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			got := readAllReadSizes(t, l, len(want), []int{int(bps), int(bps) * 2, int(bps) + 1})
			if !bytes.Equal(got, want) {
				t.Fatalf("a short tail must shorten the window: got %v, want %v", got, want)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeNoTail(t *testing.T) {
	// The source ends exactly at the loop end, so there is nothing to crossfade with.
	const loopFrames = 10
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames, true)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			want := make([]byte, 0, loopFrames*3)
			for i := range 3 {
				_ = i
				want = append(want, fixture...)
			}
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			got := readAllReadSizes(t, l, loopFrames*3*int(bps), []int{5, 13})
			if !bytes.Equal(got, want) {
				t.Fatalf("a loop without a tail must be raw: got %v, want %v", got, want)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeShortBuffer(t *testing.T) {
	const loopFrames = 12
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			fixture := crossfadeTestSource(f, loopFrames+5, true)
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}

			// A buffer shorter than a frame keeps the documented short-buffer behavior and the stream
			// continues unchanged.
			for size := 0; size < int(bps); size++ {
				if size == 0 {
					if n, err := l.Read(nil); n != 0 || err != nil {
						t.Fatalf("Read(nil): got (%d, %v), want (0, <nil>)", n, err)
					}
					continue
				}
				if n, err := l.Read(make([]byte, size)); n != 0 || !errors.Is(err, io.ErrShortBuffer) {
					t.Fatalf("Read(a buffer of %d bytes): got (%d, %v), want (0, %v)", size, n, err, io.ErrShortBuffer)
				}
			}
			if pos, err := l.Seek(0, io.SeekCurrent); err != nil || pos != 0 {
				t.Fatalf("Seek(0, io.SeekCurrent) after short reads: got (%d, %v), want (0, <nil>)", pos, err)
			}

			want := offlineCrossfadeMix(f, fixture, plan, loopFrames*3, audio.LoopCrossfadeLinear, false)
			got := readAllReadSizes(t, l, len(want), []int{int(bps), int(bps) + 1, 2, 3, 5, 7, 9, 11})
			if !bytes.Equal(got, want) {
				t.Fatalf("short reads changed the audio: got %v, want %v", got, want)
			}
		})
	}
}

// crossfadeTailRecordingReader records the source ranges it reads, to check that the tail is read
// only once.
type crossfadeTailRecordingReader struct {
	src   io.ReadSeeker
	reads [][2]int64
}

func (r *crossfadeTailRecordingReader) Read(buf []byte) (int, error) {
	pos, err := r.src.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	n, err := r.src.Read(buf)
	r.reads = append(r.reads, [2]int64{pos, pos + int64(n)})
	return n, err
}

func (r *crossfadeTailRecordingReader) Seek(offset int64, whence int) (int64, error) {
	return r.src.Seek(offset, whence)
}

func TestInfiniteLoopCrossfadeReadsTailOnce(t *testing.T) {
	const (
		loopFrames = 16
		tailFrames = 8
		fadeFrames = 4
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames+tailFrames, false)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: fadeFrames,
			}
			r := &crossfadeTailRecordingReader{src: bytes.NewReader(fixture)}
			l, err := f.newLoop(r, plan)
			if err != nil {
				t.Fatal(err)
			}
			// Read many laps: the data after the loop end must be buffered once, not re-read per lap.
			const laps = 10
			buf := make([]byte, loopFrames)
			for range laps {
				if _, err := io.ReadFull(l, buf); err != nil {
					t.Fatal(err)
				}
			}
			loopEnd := int64(loopFrames) * bps
			tailReads := 0
			for _, rd := range r.reads {
				if rd[0] >= loopEnd {
					tailReads++
				}
			}
			if tailReads > 1 {
				t.Fatalf("the data after the loop was read %d times, want at most once", tailReads)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeSourceError(t *testing.T) {
	const loopFrames = 8
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames*2+4, false)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			// Fail the read of the data after the loop end, after the loop itself was read.
			src := &dataWithErrorReadSeeker{
				src:    bytes.NewReader(fixture),
				failAt: 2,
				dataN:  int(bps),
			}
			l, err := f.newLoop(src, plan)
			if err != nil {
				t.Fatal(err)
			}
			buf := make([]byte, int(l.Length()))
			n, err := l.Read(buf)
			if !errors.Is(err, errSourceFailed) {
				t.Fatalf("Read: got %v, want %v (n=%d)", err, errSourceFailed, n)
			}
			if n != len(buf) {
				t.Fatalf("Read: got %d bytes, want %d", n, len(buf))
			}
		})
	}
}

func TestInfiniteLoopCrossfadeStalledSourceDoesNotHang(t *testing.T) {
	const loopFrames = 8
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames+4, false)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			l, err := f.newLoop(&stalledReader{src: bytes.NewReader(fixture)}, plan)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			go func() {
				defer close(done)
				buf := make([]byte, int(l.Length())*2)
				for range 4 {
					if _, err := l.Read(buf); err != nil && !errors.Is(err, io.EOF) {
						t.Error(err)
					}
				}
			}()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("Read hung with a stalled source after the loop")
			}
		})
	}
}

func TestInfiniteLoopCrossfadeFailedSeekKeepsState(t *testing.T) {
	const loopFrames = 12
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames+6, true)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
				Curve:      audio.LoopCrossfadeLinear,
			}
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}

			// Capture the tail and start the second lap.
			readFrames := int64(loopFrames + 1)
			secondLap := readAllReadSizes(t, l, int(readFrames)*int(bps), []int{int(bps), int(bps) + 1})
			want := offlineCrossfadeMix(f, fixture, plan, loopFrames+1, audio.LoopCrossfadeLinear, false)
			if !bytes.Equal(secondLap, want) {
				t.Fatalf("before the failed seek: got %v, want %v", secondLap, want)
			}

			if _, err := l.Seek(-1, io.SeekStart); err == nil {
				t.Fatal("Seek(-1, io.SeekStart): got no error, want an error")
			}

			// The stream must continue from where it was, still crossfaded.
			next := readAllReadSizes(t, l, loopFrames*int(bps), []int{int(bps) + 5, int(bps) + 9})
			wantAll := offlineCrossfadeMix(f, fixture, plan, loopFrames+1+loopFrames, audio.LoopCrossfadeLinear, false)
			wantNext := wantAll[readFrames*bps:]
			if !bytes.Equal(next, wantNext) {
				t.Fatalf("after the failed seek: got %v, want %v", next, wantNext)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeSourceShorterThanLoop(t *testing.T) {
	const loopFrames = 32
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			// The source ends in the middle of the loop, so the loop is shortened to end there and
			// the crossfade has no tail to use.
			fixture := crossfadeTestSource(f, loopFrames/2, false) // 4-byte and 8-byte aligned
			l, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			// Reading the shortened loop repeatedly must not hang and must return the source data.
			got := readAllReadSizes(t, l, len(fixture)*2, []int{int(bps) + 1, int(bps) + 5})
			want := append(append([]byte{}, fixture...), fixture...)
			if !bytes.Equal(got, want) {
				t.Fatalf("a source ending inside the loop: got %v, want %v", got, want)
			}
		})
	}
}

func TestInfiniteLoopCrossfadeSlowSourceMatchesAlignedReads(t *testing.T) {
	const (
		introFrames = 2
		loopFrames  = 10
		tailFrames  = 5
		fadeFrames  = 3
	)
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			plan := audio.LoopCrossfadePlan{
				IntroLength: int64(introFrames) * bps,
				LoopLength:  int64(loopFrames) * bps,
				FadeFrames:  fadeFrames,
				Curve:       audio.LoopCrossfadeEqualPower,
			}
			fixture := crossfadeTestSource(f, introFrames+loopFrames+tailFrames, true)

			// The reference reads with one large buffer.
			ref, err := f.newLoop(bytes.NewReader(fixture), plan)
			if err != nil {
				t.Fatal(err)
			}
			n := (introFrames + loopFrames*4) * int(bps)
			want := readAllReadSizes(t, ref, n, []int{n})

			// A source returning one byte per Read, and buffers of odd sizes, must give the same audio.
			slow, err := f.newLoop(&slowReader{src: bytes.NewReader(fixture)}, plan)
			if err != nil {
				t.Fatal(err)
			}
			got := readAllReadSizes(t, slow, n, []int{1, 3, 5, 7, 11, 13, 17, 19})
			if !bytes.Equal(got, want) {
				for i := range min(len(got), len(want)) {
					if got[i] != want[i] {
						t.Fatalf("the audio differs at byte %d: got %v, want %v", i, got[i], want[i])
					}
				}
				t.Fatalf("the audio differs in length: got %d, want %d", len(got), len(want))
			}
		})
	}
}

func TestInfiniteLoopCrossfadePartialWholeSampleAlignment(t *testing.T) {
	// A source positioned inside a frame before the loop is created produces channel values that
	// straddle two source values. Such values must be left raw rather than mixed with unrelated tail
	// bytes, and the stream must stay finite.
	const loopFrames = 8
	for _, f := range crossfadeFormats() {
		t.Run(f.name, func(t *testing.T) {
			bps := crossfadeBytesPerSample(f)
			fixture := crossfadeTestSource(f, loopFrames*2, true)
			plan := audio.LoopCrossfadePlan{
				LoopLength: int64(loopFrames) * bps,
				FadeFrames: 4,
			}
			src := bytes.NewReader(fixture)
			// Start one byte into the source.
			if _, err := src.Seek(1, io.SeekStart); err != nil {
				t.Fatal(err)
			}
			l, err := f.newLoop(src, plan)
			if err != nil {
				t.Fatal(err)
			}
			n := loopFrames * int(bps)
			done := make(chan struct{})
			var got []byte
			go func() {
				defer close(done)
				got = readAllReadSizes(t, l, n, []int{5, 7, 9})
			}()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("Read hung for a source starting inside a frame")
			}
			// The first value is raw, since it straddles two source values.
			if want := fixture[1 : 1+int(bps)]; !bytes.Equal(got[:bps], want) {
				t.Errorf("got %v, want %v", got[:bps], want)
			}
		})
	}
}
