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

package convert_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"sync/atomic"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
)

func f32Bytes(v ...float32) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, v)
	return b.Bytes()
}

func decodeF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(b[4*i:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func i16Bytes(v ...int16) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, v)
	return b.Bytes()
}

func masterStatePointer(s convert.MixState) *atomic.Pointer[convert.MixState] {
	var p atomic.Pointer[convert.MixState]
	p.Store(&s)
	return &p
}

// referenceMix is the offline matrix reference: out[frame][outCh] = clamp(gain *
// sum_in(coeff * in), -1, 1). Auto-normalization divides each output row by the sum of the
// absolute coefficient values.
func referenceMix(in [][]float64, sourceChannels int, state convert.MixState) [][2]float64 {
	out := make([][2]float64, len(in))
	rows := make([][]float64, 2)
	for o := range rows {
		row := make([]float64, sourceChannels)
		var sum float64
		if state.AutoNormalize {
			for i := range sourceChannels {
				sum += math.Abs(state.Coeffs[o*sourceChannels+i])
			}
		}
		for i := range sourceChannels {
			c := state.Coeffs[o*sourceChannels+i]
			if state.AutoNormalize && sum != 0 {
				c /= sum
			}
			row[i] = c * state.Gain
		}
		rows[o] = row
	}
	for f, frame := range in {
		for o := 0; o < 2; o++ {
			var v float64
			for i := range sourceChannels {
				sample := frame[i]
				if math.IsNaN(sample) || math.IsInf(sample, 0) {
					sample = 0
				}
				v += rows[o][i] * sample
			}
			if v > 1 {
				v = 1
			} else if v < -1 {
				v = -1
			} else if math.IsNaN(v) {
				v = 0
			}
			out[f][o] = v
		}
	}
	return out
}

func readAll(t *testing.T, r io.Reader) []byte {
	t.Helper()
	var out bytes.Buffer
	buf := make([]byte, 13) // deliberately unaligned to exercise partial frames
	for {
		n, err := r.Read(buf)
		out.Write(buf[:n])
		if errors.Is(err, io.EOF) {
			break
		}
		if errors.Is(err, io.ErrShortBuffer) {
			// The destination cannot hold one output frame yet; retry with the same buffer.
			continue
		}
		if err != nil {
			t.Fatalf("Read returned unexpected error: %v", err)
		}
		if n == 0 {
			t.Fatal("Read returned (0, nil)")
		}
	}
	return out.Bytes()
}

func TestChannelMixerF32AgainstReference(t *testing.T) {
	// Pulse fixture: each source channel emits a distinct full-scale pulse on its own frame,
	// repeated so every routing can be checked sample by sample.
	sourceChannels := 4
	frames := 8
	raw := make([]float32, 0, frames*sourceChannels)
	in := make([][]float64, frames)
	for f := range frames {
		frame := make([]float64, sourceChannels)
		for ch := range sourceChannels {
			var v float32
			if f%sourceChannels == ch {
				// Alternate signs so phase inversion is visible.
				if f/sourceChannels%2 == 0 {
					v = 1
				} else {
					v = -1
				}
			}
			frame[ch] = float64(v)
			raw = append(raw, v)
		}
		in[f] = frame
	}

	coeffs := []float64{
		// out L: in0 -> 1, in1 -> -0.5 (phase invert), in2 -> 0.25, in3 -> 0
		1, -0.5, 0.25, 0,
		// out R: in0 -> 0, in1 -> 0.25, in2 -> -0.5, in3 -> 1
		0, 0.25, -0.5, 1,
	}

	for _, autoNormalize := range []bool{false, true} {
		for _, gain := range []float64{1, 0.75, 4} {
			state := convert.MixState{
				SourceChannels: sourceChannels,
				OutputChannels: 2,
				Coeffs:         coeffs,
				Gain:           gain,
				AutoNormalize:  autoNormalize,
			}
			master := masterStatePointer(convert.DefaultMixState(2))
			var player atomic.Pointer[convert.MixState]
			player.Store(&state)
			mixer := convert.NewChannelMixer(bytes.NewReader(f32Bytes(raw...)), sourceChannels, convert.BitDepthInBytesFloat32, &player, master)

			got := decodeF32(readAll(t, mixer))
			want := referenceMix(in, sourceChannels, state)
			if len(got) != frames*2 {
				t.Fatalf("autoNormalize=%t gain=%v: got %d output samples, want %d", autoNormalize, gain, len(got), frames*2)
			}
			for f := range frames {
				for o := 0; o < 2; o++ {
					if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 1e-6 {
						t.Errorf("autoNormalize=%t gain=%v frame=%d out=%d: got %v, want %v", autoNormalize, gain, f, o, got[2*f+o], want[f][o])
					}
				}
			}
		}
	}
}

func TestChannelMixerSineSurroundAgainstReference(t *testing.T) {
	sourceChannels := 6
	frames := 256
	raw := make([]float32, 0, frames*sourceChannels)
	in := make([][]float64, frames)
	for f := range frames {
		frame := make([]float64, sourceChannels)
		for ch := range sourceChannels {
			// Distinct frequency and phase per channel.
			v := 0.6 * math.Sin(2*math.Pi*float64(f)*(1+float64(ch))/64+float64(ch))
			frame[ch] = v
			raw = append(raw, float32(v))
		}
		in[f] = frame
	}

	// 5.1 -> stereo downmix, with a negative coefficient to check phase.
	coeffs := make([]float64, 2*sourceChannels)
	for ch := range sourceChannels {
		coeffs[0*sourceChannels+ch] = 0.5
		coeffs[1*sourceChannels+ch] = 0.5
	}
	coeffs[0*sourceChannels+2] = -0.5 // center inverted on the left
	state := convert.MixState{
		SourceChannels: sourceChannels,
		OutputChannels: 2,
		Coeffs:         coeffs,
		Gain:           1,
		AutoNormalize:  true,
	}
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	player.Store(&state)
	mixer := convert.NewChannelMixer(bytes.NewReader(f32Bytes(raw...)), sourceChannels, convert.BitDepthInBytesFloat32, &player, master)

	got := decodeF32(readAll(t, mixer))
	want := referenceMix(in, sourceChannels, state)
	for f := range frames {
		for o := 0; o < 2; o++ {
			if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 2e-6 {
				t.Fatalf("frame=%d out=%d: got %v, want %v", f, o, got[2*f+o], want[f][o])
			}
		}
	}
}

func TestChannelMixerInt16AgainstReference(t *testing.T) {
	sourceChannels := 3
	frames := 32
	rawI16 := make([]int16, 0, frames*sourceChannels)
	in := make([][]float64, frames)
	for f := range frames {
		frame := make([]float64, sourceChannels)
		for ch := range sourceChannels {
			v := int16((f*7 + ch*13 - 200) * 111)
			frame[ch] = float64(v) / (1 << 15)
			rawI16 = append(rawI16, v)
		}
		in[f] = frame
	}

	coeffs := []float64{
		0.7, 0, -0.3,
		-0.2, 0.8, 0.4,
	}
	state := convert.MixState{
		SourceChannels: sourceChannels,
		OutputChannels: 2,
		Coeffs:         coeffs,
		Gain:           1.25,
	}
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	player.Store(&state)
	mixer := convert.NewChannelMixer(bytes.NewReader(i16Bytes(rawI16...)), sourceChannels, convert.BitDepthInBytesInt16, &player, master)

	got := decodeF32(readAll(t, mixer))
	want := referenceMix(in, sourceChannels, state)
	for f := range frames {
		for o := 0; o < 2; o++ {
			// int16 quantization error is at most one quantization step.
			if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 1.5/(1<<15) {
				t.Fatalf("frame=%d out=%d: got %v, want %v", f, o, got[2*f+o], want[f][o])
			}
		}
	}
}

func TestChannelMixerMonoAndStereoPassthrough(t *testing.T) {
	// Mono identity-ish default state: channel 0 routes to both outputs only when explicitly
	// configured; DefaultMixState(1) maps it to the left output only.
	mono := f32Bytes(0.25, -0.5, 0.75, 1)
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(bytes.NewReader(mono), 1, convert.BitDepthInBytesFloat32, nil, master)
	got := decodeF32(readAll(t, mixer))
	want := []float32{0.25, 0, -0.5, 0, 0.75, 0, 1, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mono default mix at %d: got %v, want %v", i, got[i], want[i])
		}
	}

	// Stereo default state must be a byte-identical passthrough.
	stereo := f32Bytes(0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
	mixer = convert.NewChannelMixer(bytes.NewReader(stereo), 2, convert.BitDepthInBytesFloat32, nil, master)
	gotBytes := readAll(t, mixer)
	if !bytes.Equal(gotBytes, stereo) {
		t.Fatalf("stereo passthrough changed the bytes: got %v, want %v", gotBytes, stereo)
	}
}

func TestChannelMixerMuteInvertAndClamp(t *testing.T) {
	// Two in-phase full-scale channels summed without normalization must clamp instead of
	// overflowing.
	raw := f32Bytes(1, 1, 1, 1)
	state := convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs: []float64{
			1, 1, // left = L + R = 2 -> clamped to 1
			1, -1, // right = L - R = 0, with inversion
		},
		Gain: 1,
	}
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	player.Store(&state)
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)
	got := decodeF32(readAll(t, mixer))
	if got[0] != 1 || got[1] != 0 {
		t.Fatalf("clamp/invert: got %v, want [1 0]", got[:2])
	}

	// Two channels at partial scale: the raw sum reaches full scale without clamping, while
	// auto-normalization halves it. This distinguishes normalization from clamping.
	raw = f32Bytes(0.6, 0.4, 0.6, 0.4)
	state = convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs: []float64{
			1, 1, // left = L + R
			1, -1, // right = L - R, with inversion
		},
		Gain: 1,
	}
	player.Store(&state)
	mixer = convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)
	got = decodeF32(readAll(t, mixer))
	if diff := math.Abs(float64(got[0]) - 1.0); diff > 1e-6 {
		t.Fatalf("unnormalized partial-scale sum: got %v, want 1", got[0])
	}

	// Auto-normalization divides the row by two unit coefficients, halving the result.
	normalized := state
	normalized.AutoNormalize = true
	player.Store(&normalized)
	mixer = convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)
	got = decodeF32(readAll(t, mixer))
	if diff := math.Abs(float64(got[0]) - 0.5); diff > 1e-6 {
		t.Fatalf("auto-normalized sum: got %v, want 0.5", got[0])
	}

	// A zero row is silence.
	state2 := convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs:         make([]float64, 4),
		Gain:           1,
	}
	player.Store(&state2)
	mixer = convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)
	got = decodeF32(readAll(t, mixer))
	for i, v := range got {
		if v != 0 {
			t.Fatalf("muted output at %d: got %v, want 0", i, v)
		}
	}
}

func TestChannelMixerExtremeGainAndNonFinite(t *testing.T) {
	raw := f32Bytes(0.5, -0.5)
	state := convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs: []float64{
			1, 0,
			0, 1,
		},
		Gain: 1e18, // extreme but finite
	}
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	player.Store(&state)
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)
	got := decodeF32(readAll(t, mixer))
	if got[0] != 1 || got[1] != -1 {
		t.Fatalf("extreme gain clamp: got %v, want [1 -1]", got)
	}
	for _, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output must be finite but got %v", v)
		}
	}

	// A non-finite source sample is replaced by silence for that frame.
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	rawBad := f32Bytes(nan, inf, 0.25, 0.5)
	state.Gain = 1
	player.Store(&state)
	mixer = convert.NewChannelMixer(bytes.NewReader(rawBad), 2, convert.BitDepthInBytesFloat32, &player, master)
	got = decodeF32(readAll(t, mixer))
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("non-finite input must be silence: got %v, want [0 0]", got[:2])
	}
	if got[2] != 0.25 || got[3] != 0.5 {
		t.Fatalf("frame after non-finite input must be intact: got %v", got[2:])
	}
}

func TestChannelMixerEOFAndShortBuffer(t *testing.T) {
	// Two and a half frames of a mono source (10 bytes = 2.5 input frames).
	raw := f32Bytes(0.1, 0.2, 0.3)[:10]
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 1, convert.BitDepthInBytesFloat32, nil, master)

	// A destination shorter than one output frame yields ErrShortBuffer while whole frames
	// remain available.
	small := make([]byte, 4)
	n, err := mixer.Read(small)
	if n != 0 || !errors.Is(err, io.ErrShortBuffer) {
		t.Fatalf("short buffer: got (%d, %v), want (0, ErrShortBuffer)", n, err)
	}

	// The two whole source frames become two stereo output frames; the trailing half-frame
	// cannot form a frame and is discarded, so reading ends with io.EOF.
	out, err := io.ReadAll(mixer)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(out) != 2*8 {
		t.Fatalf("output length: got %d bytes, want %d", len(out), 2*8)
	}
	got := decodeF32(out)
	want := []float32{0.1, 0, 0.2, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output at %d: got %v, want %v", i, got[i], want[i])
		}
	}

	// A subsequent read keeps reporting EOF with no bytes.
	buf := make([]byte, 8)
	if n, err := mixer.Read(buf); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("drained stream: got (%d, %v), want (0, EOF)", n, err)
	}
}

// dataAndEOFReader returns the final bytes together with io.EOF, as some sources (e.g. the
// virtual audio device's players) do.
type dataAndEOFReader struct {
	b []byte
	i int
}

func (r *dataAndEOFReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.i:])
	r.i += n
	if r.i >= len(r.b) {
		return n, io.EOF
	}
	return n, nil
}

// TestChannelMixerEOFWithWholeFrames uses a source whose length is an exact number of frames.
// bytes.Reader reports EOF one read after the final data; a source returning data and EOF
// together delivers the last frame with the EOF.
func TestChannelMixerEOFWithWholeFrames(t *testing.T) {
	// Exactly two mono frames (8 bytes).
	raw := f32Bytes(0.5, 0.25)
	master := masterStatePointer(convert.DefaultMixState(2))

	// With bytes.Reader, EOF arrives on the read after the final frame.
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 1, convert.BitDepthInBytesFloat32, nil, master)
	buf := make([]byte, 8)
	if n, err := mixer.Read(buf); n != 8 || err != nil {
		t.Fatalf("first frame: got (%d, %v), want (8, nil)", n, err)
	}
	if n, err := mixer.Read(buf); n != 8 || err != nil {
		t.Fatalf("second frame: got (%d, %v), want (8, nil)", n, err)
	}
	if n, err := mixer.Read(buf); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("drained: got (%d, %v), want (0, EOF)", n, err)
	}

	// With a source returning data and EOF together, the last frame arrives with io.EOF.
	mixer = convert.NewChannelMixer(&dataAndEOFReader{b: raw}, 1, convert.BitDepthInBytesFloat32, nil, master)
	if n, err := mixer.Read(buf); n != 8 || err != nil {
		t.Fatalf("first frame (joint EOF source): got (%d, %v), want (8, nil)", n, err)
	}
	if n, err := mixer.Read(buf); n != 8 || !errors.Is(err, io.EOF) {
		t.Fatalf("last frame (joint EOF source): got (%d, %v), want (8, EOF)", n, err)
	}
}

type errReader struct {
	err error
}

func (e *errReader) Read(p []byte) (int, error) {
	return 0, e.err
}

func TestChannelMixerSourceError(t *testing.T) {
	sentinel := errors.New("boom")
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(&errReader{err: sentinel}, 2, convert.BitDepthInBytesFloat32, nil, master)
	buf := make([]byte, 16)
	if _, err := mixer.Read(buf); !errors.Is(err, sentinel) {
		t.Fatalf("got %v, want the source error %v", err, sentinel)
	}
}

// oneByteAtATimeReader returns at most one byte per Read to exercise the partial-frame
// buffering path heavily.
type oneByteAtATimeReader struct {
	b []byte
	i int
}

func (r *oneByteAtATimeReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	p[0] = r.b[r.i]
	r.i++
	return 1, nil
}

func TestChannelMixerBytewiseReads(t *testing.T) {
	raw := f32Bytes(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(&oneByteAtATimeReader{b: raw}, 2, convert.BitDepthInBytesFloat32, nil, master)
	got := readAll(t, mixer)
	if !bytes.Equal(got, raw) {
		t.Fatalf("bytewise passthrough mismatch: got %v, want %v", got, raw)
	}
}

func TestChannelMixerSeek(t *testing.T) {
	sourceChannels := 3
	frames := 10
	raw := make([]float32, frames*sourceChannels)
	for f := range frames {
		for ch := range sourceChannels {
			raw[f*sourceChannels+ch] = float32(f*sourceChannels + ch + 1)
		}
	}
	state := convert.MixState{
		SourceChannels: sourceChannels,
		OutputChannels: 2,
		// Identity-ish first two channels.
		Coeffs: []float64{1, 0, 0, 0, 1, 0},
		Gain:   1,
	}
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	player.Store(&state)
	seeker := bytes.NewReader(f32Bytes(raw...))
	mixer := convert.NewChannelMixer(seeker, sourceChannels, convert.BitDepthInBytesFloat32, &player, master)
	seekMixer, ok := any(mixer).(io.Seeker)
	if !ok {
		t.Fatal("a mixer over a seekable source must itself be seekable")
	}

	// Seek to output frame 4 (= 32 output bytes; source frame 4 at 48 source bytes).
	pos, err := seekMixer.Seek(4*8, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}
	if pos != 4*8 {
		t.Fatalf("seek position: got %d, want %d", pos, 4*8)
	}
	got := decodeF32(readAll(t, seekMixer.(io.Reader)))
	wantFrames := referenceMix(framesFromF32(raw, sourceChannels), sourceChannels, state)
	for i := range frames - 4 {
		for o := 0; o < 2; o++ {
			if diff := math.Abs(float64(got[2*i+o]) - wantFrames[4+i][o]); diff > 1e-6 {
				t.Fatalf("after seek at output %d: got %v, want %v", 2*i+o, got[2*i+o], wantFrames[4+i][o])
			}
		}
	}

	// Seek back to the start and verify identical output (no latency introduced by the mixer).
	if _, err := seekMixer.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	again := decodeF32(readAll(t, seekMixer.(io.Reader)))
	first := decodeF32(readAll(t, convert.NewChannelMixer(bytes.NewReader(f32Bytes(raw...)), sourceChannels, convert.BitDepthInBytesFloat32, &player, master)))
	if len(again) != len(first) {
		t.Fatalf("rewind length: got %d, want %d", len(again), len(first))
	}
	for i := range again {
		if again[i] != first[i] {
			t.Fatalf("rewind changed output at %d: got %v, want %v", i, again[i], first[i])
		}
	}

	// A query reports the current position without moving it.
	pos, err = seekMixer.Seek(0, io.SeekCurrent)
	if err != nil || pos != int64(frames*8) {
		t.Fatalf("current-position query: got (%d, %v), want (%d, nil)", pos, err, frames*8)
	}

	// SeekEnd lands on the last whole frame.
	pos, err = seekMixer.Seek(0, io.SeekEnd)
	if err != nil || pos != int64(frames*8) {
		t.Fatalf("end seek: got (%d, %v), want (%d, nil)", pos, err, frames*8)
	}
}

func framesFromF32(raw []float32, channels int) [][]float64 {
	frames := make([][]float64, len(raw)/channels)
	for f := range frames {
		frame := make([]float64, channels)
		for ch := range channels {
			frame[ch] = float64(raw[f*channels+ch])
		}
		frames[f] = frame
	}
	return frames
}

func TestChannelMixerNonSeekableSource(t *testing.T) {
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(&oneByteAtATimeReader{b: f32Bytes(1, 2, 3, 4)}, 2, convert.BitDepthInBytesFloat32, nil, master)
	if _, err := mixer.Seek(0, io.SeekStart); !errors.Is(err, errors.ErrUnsupported) {
		t.Fatalf("seeking a non-seekable source: got %v, want an error wrapping errors.ErrUnsupported", err)
	}
}

func TestCombineMixStatesMasterAfterPlayer(t *testing.T) {
	// Player: mono -> stereo with both outputs fed by the single input.
	player := convert.MixState{
		SourceChannels: 1,
		OutputChannels: 2,
		Coeffs:         []float64{0.5, 0.5},
		Gain:           2,
	}
	// Master: invert the left output only.
	master := convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs:         []float64{-1, 0, 0, 1},
		Gain:           1,
	}
	eff := convert.CombineMixStates(&player, master)
	// Left = -1 * (2 * 0.5) = -1; right = 1 * (2 * 0.5) = 1.
	if diff := math.Abs(eff.Coeffs[0] - -1); diff > 1e-12 {
		t.Errorf("left coefficient: got %v, want -1", eff.Coeffs[0])
	}
	if diff := math.Abs(eff.Coeffs[1] - 1); diff > 1e-12 {
		t.Errorf("right coefficient: got %v, want 1", eff.Coeffs[1])
	}
}

func TestChannelMixerIdentityFloatRoundTrip(t *testing.T) {
	// Every float32 must survive an identity mix byte-identically: float32 -> float64 ->
	// times 1.0 -> float32 is an exact round-trip because float64 represents every float32.
	values := []float32{0.1, -0.2, 1.0, -1.0, 0.33333334, 1.5e-7, 0.99999994, -0.7853982}
	raw := f32Bytes(values...)
	master := masterStatePointer(convert.DefaultMixState(2))
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, nil, master)
	out, err := io.ReadAll(mixer)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, raw) {
		t.Fatalf("identity mix changed float32 bytes: got %v, want %v", decodeF32(out), values)
	}
}

func TestChannelMixerStateUpdateAtFrameBoundary(t *testing.T) {
	raw := f32Bytes(1, 0, 0, 1, 1, 1, 0, 1)
	master := masterStatePointer(convert.DefaultMixState(2))
	var player atomic.Pointer[convert.MixState]
	mixer := convert.NewChannelMixer(bytes.NewReader(raw), 2, convert.BitDepthInBytesFloat32, &player, master)

	// Start with identity.
	identity := convert.DefaultMixState(2)
	player.Store(&identity)
	buf := make([]byte, 8)
	if n, err := mixer.Read(buf); n != 8 || err != nil {
		t.Fatalf("first read: got (%d, %v)", n, err)
	}
	if got := decodeF32(buf); got[0] != 1 || got[1] != 0 {
		t.Fatalf("identity first frame: got %v", got)
	}

	// Swap to an inverting matrix before the second frame.
	inverted := convert.MixState{
		SourceChannels: 2,
		OutputChannels: 2,
		Coeffs:         []float64{-1, 0, 0, -1},
		Gain:           1,
	}
	player.Store(&inverted)
	if n, err := mixer.Read(buf); n != 8 || err != nil {
		t.Fatalf("second read: got (%d, %v)", n, err)
	}
	if got := decodeF32(buf); got[0] != 0 || got[1] != -1 {
		t.Fatalf("inverted second frame: got %v, want [0 -1]", got)
	}
}
