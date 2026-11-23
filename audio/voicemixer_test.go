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

package audio_test

import (
	"bytes"
	"errors"
	"io"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

const testSampleRate = 48000

// f32FrameBytes is one stereo 32bit float frame.
const f32FrameBytes = 8

// frameDuration is the exact duration of n frames at testSampleRate. Multiplying before dividing
// avoids the truncation that time.Second/rate would accumulate for small frame counts.
func frameDuration(n int) time.Duration {
	return time.Duration(int64(n) * int64(time.Second) / int64(testSampleRate))
}

func mixerSetup() {
	context = audio.NewContext(testSampleRate)
}

// f32Source builds an n-frame F32 stereo source whose every frame carries the given left/right
// amplitudes. Frames are uniform so a reference mix is just a sum of scaled amplitudes.
func f32Source(n int, l, r float32) *audio.AudioVoiceSource {
	data := make([]byte, n*f32FrameBytes)
	for i := range n {
		putF32(data[(i*2)*4:], l)
		putF32(data[(i*2+1)*4:], r)
	}
	return &audio.AudioVoiceSource{Data: data, SampleRate: testSampleRate}
}

func putF32(b []byte, v float32) {
	bits := math.Float32bits(v)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
	b[2] = byte(bits >> 16)
	b[3] = byte(bits >> 24)
}

func getF32(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}

// readFrames reads n frames (8 bytes each) from the mixer in fixed pulses.
func readFrames(t *testing.T, m *audio.AudioVoiceMixer, n int) []byte {
	t.Helper()
	buf := make([]byte, n*f32FrameBytes)
	if _, err := io.ReadFull(m, buf); err != nil {
		t.Fatalf("mixer Read: %v", err)
	}
	return buf
}

// frameAt returns the left/right amplitudes of frame i.
func frameAt(buf []byte, i int) (float32, float32) {
	off := i * f32FrameBytes
	return getF32(buf[off:]), getF32(buf[off+4:])
}

func approxEqual(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-5
}

// TestAudioVoiceMixerUnalignedReadCarry: reads ending in the middle of a frame carry the
// frame-aligned remainder into the next read, keep the frame clock consistent, and still apply
// the bus volume to the carried bytes.
func TestAudioVoiceMixerUnalignedReadCarry(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()
	m.SetVolume(0.5)

	if _, err := m.Play(f32Source(16, 1, 1), nil); err != nil {
		t.Fatal(err)
	}

	// Read 13 bytes: one whole frame (8) plus 5 bytes of a second frame, whose remaining 3 bytes
	// are carried. Every complete emitted sample should be 1.0 (source) * 0.5 (bus) = 0.5.
	buf := make([]byte, 13)
	if n, err := m.Read(buf); err != nil || n != 13 {
		t.Fatalf("unaligned Read: n=%d err=%v", n, err)
	}
	// Only the first three complete float32 samples (12 bytes) are checkable; the 13th starts a
	// fourth sample that completes on the next read.
	for i := range 12 / 4 {
		if got := getF32(buf[i*4:]); !approxEqual(got, 0.5) {
			t.Fatalf("unaligned sample %d: got %v, want 0.5", i, got)
		}
	}
	// The carried 3 bytes complete the second frame's left channel; rest's layout is:
	//   rest[0:3]   = carried tail of frame 2's left channel
	//   rest[3:7]   = frame 2's right channel (fresh)
	//   rest[7:11]  = frame 3's left channel
	//   ... and so on.
	rest := make([]byte, 3+2*f32FrameBytes)
	if n, err := io.ReadFull(m, rest); err != nil || n != len(rest) {
		t.Fatalf("carry drain: n=%d err=%v", n, err)
	}
	// Reassemble frame 2's left channel from the 3 carried bytes and one fresh byte (rest[3] is
	// actually the start of the right channel, so instead verify the right channel directly).
	if got := getF32(rest[3:7]); !approxEqual(got, 0.5) {
		t.Fatalf("carried frame's right channel: got %v, want 0.5", got)
	}
	// Frame 3 starts at rest[7].
	if got := getF32(rest[7:11]); !approxEqual(got, 0.5) {
		t.Fatalf("post-carry frame left channel: got %v, want 0.5", got)
	}

	// Exactly four frames have been consumed overall: the partial second frame still advances the
	// deterministic clock by one frame.
	if got := m.FramePositionForTesting(); got != 4 {
		t.Fatalf("frame clock: got %d, want 4", got)
	}
}

// TestAudioVoiceMixerZeroVoicesEmitsSilence: with no voice, every pulse is silence and no error
// is ever returned, even across oddly sized reads.
func TestAudioVoiceMixerZeroVoicesEmitsSilence(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()

	// Frame-aligned and deliberately unaligned reads must all come back as silence.
	for _, size := range []int{8, 16, 3, 13, 1, 32} {
		buf := make([]byte, size)
		n, err := m.Read(buf)
		if err != nil {
			t.Fatalf("Read(%d): unexpected error %v", size, err)
		}
		if n != size {
			t.Fatalf("Read(%d): got %d bytes, want %d", size, n, size)
		}
		for _, b := range buf {
			if b != 0 {
				t.Fatalf("Read(%d): idle mixer emitted non-silence", size)
			}
		}
	}
}

// TestAudioVoiceMixerSingleVoiceMatchesSource: one voice reproduces its source frames exactly and
// is recycled at the pulse where its last frame is consumed.
func TestAudioVoiceMixerSingleVoiceMatchesSource(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()

	src := f32Source(4, 0.5, -0.25)
	v, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !v.IsActive() {
		t.Fatal("a new voice must be active")
	}

	buf := readFrames(t, m, 4)
	for i := range 4 {
		l, r := frameAt(buf, i)
		if !approxEqual(l, 0.5) || !approxEqual(r, -0.25) {
			t.Fatalf("frame %d: got (%v, %v), want (0.5, -0.25)", i, l, r)
		}
	}
	// Recycling happens at the pulse boundary of the next read attempt.
	readFrames(t, m, 1)
	if v.IsActive() {
		t.Fatal("the voice must be recycled after its source ended")
	}
	if err := v.Err(); err != nil {
		t.Fatalf("a cleanly ending voice reports no error, got %v", err)
	}
	stats := m.Stats()
	if stats.StartedVoices != 1 || stats.RecycledVoices != 1 || stats.FailedVoices != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// After recycling, further pulses are silence.
	silent := readFrames(t, m, 2)
	for i := range 2 {
		if l, r := frameAt(silent, i); l != 0 || r != 0 {
			t.Fatalf("post-recycle frame %d must be silence, got (%v, %v)", i, l, r)
		}
	}
}

// TestAudioVoiceMixerSameSourceRepeatedPlay: the same source object playing concurrently yields
// independent voices whose samples are added, and replaying after recycle restarts from frame 0.
func TestAudioVoiceMixerSameSourceRepeatedPlay(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(8)
	defer m.Close()

	src := f32Source(3, 0.2, 0.2)

	v1, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	// One frame later, start the same source again.
	readFrames(t, m, 1)
	v2, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Frame 1: v1's second frame (0.2) plus v2's first frame (0.2) = 0.4.
	buf := readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.4) {
		t.Fatalf("overlapping same-source play: got %v, want 0.4", l)
	}
	// Frame 2: v1's third frame + v2's second frame = 0.4.
	buf = readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.4) {
		t.Fatalf("overlapping frame: got %v, want 0.4", l)
	}
	// Frame 3: v1 ended; only v2's third frame remains.
	buf = readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.2) {
		t.Fatalf("after first voice ended: got %v, want 0.2", l)
	}
	if v1.IsActive() {
		t.Fatal("v1 must have been recycled")
	}
	if !v2.IsActive() {
		t.Fatal("v2 must still be active")
	}

	// Replay the same source after both ended: it restarts from the beginning.
	readFrames(t, m, 1)
	v3, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	buf = readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.2) {
		t.Fatalf("replay after recycle: got %v, want 0.2", l)
	}
	readFrames(t, m, 3)
	if v3.IsActive() {
		t.Fatal("v3 must be recycled after its 3 frames")
	}
}

// TestAudioVoiceMixerPreemptionAndLimit: preemption only evicts strictly lower-priority,
// non-paused voices; same-priority traffic is dropped, and the high-priority "music" voice is
// never silently interrupted.
func TestAudioVoiceMixerPreemptionAndLimit(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	music, err := m.Play(f32Source(100, 0.9, 0.9), &audio.AudioVoicePlayOptions{Priority: 10})
	if err != nil {
		t.Fatal(err)
	}
	sfx, err := m.Play(f32Source(100, 0.1, 0.1), &audio.AudioVoicePlayOptions{Priority: 1})
	if err != nil {
		t.Fatal(err)
	}

	// With both slots taken, a newcomer whose priority is not strictly higher than every
	// preemptable voice is dropped. Priority 1 only equals the sfx and loses to the music.
	if _, err := m.Play(f32Source(1, 1, 1), &audio.AudioVoicePlayOptions{Priority: 1}); !errors.Is(err, audio.ErrAudioVoiceLimit) {
		t.Fatalf("same-priority play: got %v, want ErrAudioVoiceLimit", err)
	}

	// A higher-priority effect preempts the lower-priority sfx, never the music.
	urgent, err := m.Play(f32Source(1, 0.5, 0.5), &audio.AudioVoicePlayOptions{Priority: 5})
	if err != nil {
		t.Fatal(err)
	}
	if sfx.IsActive() {
		t.Fatal("the low-priority sfx must have been preempted")
	}
	if !music.IsActive() {
		t.Fatal("the high-priority music must never be preempted")
	}
	if !urgent.IsActive() {
		t.Fatal("the preempting voice must be active")
	}

	// Another low-priority play while full is dropped: nothing is strictly below priority 0.
	if _, err := m.Play(f32Source(1, 1, 1), &audio.AudioVoicePlayOptions{Priority: 0}); !errors.Is(err, audio.ErrAudioVoiceLimit) {
		t.Fatalf("low-priority play while full: got %v, want ErrAudioVoiceLimit", err)
	}

	stats := m.Stats()
	if stats.PreemptedVoices != 1 || stats.DroppedPlays != 2 || stats.ActiveVoices != 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// Paused voices are protected from preemption. Pause the urgent voice, then apply enough
	// preemption pressure that it would otherwise be the oldest victim: a priority-20 newcomer
	// can evict the music (10), but never the paused urgent voice (5).
	if err := urgent.Pause(); err != nil {
		t.Fatal(err)
	}
	high, err := m.Play(f32Source(1, 1, 1), &audio.AudioVoicePlayOptions{Priority: 20})
	if err != nil {
		t.Fatalf("a priority-20 play should evict the music: %v", err)
	}
	if music.IsActive() {
		t.Fatal("the lower-priority music must be preempted by the priority-20 voice")
	}
	if !urgent.IsPaused() || !urgent.IsActive() {
		t.Fatal("a paused voice must survive preemption pressure")
	}
	if !high.IsActive() {
		t.Fatal("the preempting high-priority voice must be active")
	}
}

// TestAudioVoiceMixerZeroLimit: a zero-capacity mixer drops every play and only emits silence.
func TestAudioVoiceMixerZeroLimit(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(0)
	defer m.Close()

	if _, err := m.Play(f32Source(1, 1, 1), nil); !errors.Is(err, audio.ErrAudioVoiceLimit) {
		t.Fatalf("zero-limit play: got %v, want ErrAudioVoiceLimit", err)
	}
	buf := readFrames(t, m, 4)
	for i := range 4 {
		if l, r := frameAt(buf, i); l != 0 || r != 0 {
			t.Fatalf("zero-limit mixer must emit silence")
		}
	}
	if got := m.Stats().DroppedPlays; got != 1 {
		t.Fatalf("DroppedPlays: got %d, want 1", got)
	}
}

// TestAudioVoiceMixerFadeIn: the first frames follow a deterministic linear fade-in.
func TestAudioVoiceMixerFadeIn(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	// Three frames at 48kHz is an exactly representable duration (3*time.Second/48000).
	const fadeFrames = 3
	v, err := m.Play(f32Source(8, 1, 1), &audio.AudioVoicePlayOptions{
		FadeIn: frameDuration(fadeFrames),
	})
	if err != nil {
		t.Fatal(err)
	}

	buf := readFrames(t, m, 6)
	wantFade := []float32{1.0 / 3, 2.0 / 3, 1.0}
	for i, want := range wantFade {
		if l, _ := frameAt(buf, i); !approxEqual(l, want) {
			t.Fatalf("fade-in frame %d: got %v, want %v", i, l, want)
		}
	}
	// After the fade, full volume.
	for i := fadeFrames; i < 6; i++ {
		if l, _ := frameAt(buf, i); !approxEqual(l, 1) {
			t.Fatalf("post-fade frame %d: got %v, want 1", i, l)
		}
	}
	readFrames(t, m, 3)
	if v.IsActive() {
		t.Fatal("voice must be recycled after its 8 frames")
	}
}

// TestAudioVoiceMixerFadeOut: gain ramps down linearly and the voice is recycled exactly when the
// fade reaches zero.
func TestAudioVoiceMixerFadeOut(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	v, err := m.Play(f32Source(100, 1, 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	// Let two full-volume frames pass, then fade over three. The first fade frame keeps 2/3 of
	// the volume, the Nth reaches zero and recycles the voice.
	readFrames(t, m, 2)
	if err := v.FadeOut(frameDuration(3)); err != nil {
		t.Fatal(err)
	}

	buf := readFrames(t, m, 5)
	wants := []float32{2.0 / 3, 1.0 / 3, 0, 0, 0}
	for i, want := range wants {
		if l, _ := frameAt(buf, i); !approxEqual(l, want) {
			t.Fatalf("fade-out frame %d: got %v, want %v", i, l, want)
		}
	}
	// The voice is recycled at the zero-gain frame.
	if v.IsActive() {
		t.Fatal("voice must be recycled when the fade-out reaches zero")
	}
}

// TestAudioVoiceMixerPauseResume: a paused voice neither advances its source nor its envelope,
// and resumes deterministically; paused frames contribute silence.
func TestAudioVoiceMixerPauseResume(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	v, err := m.Play(f32Source(6, 0.3, 0.3), nil)
	if err != nil {
		t.Fatal(err)
	}
	readFrames(t, m, 3)
	if err := v.Pause(); err != nil {
		t.Fatal(err)
	}
	// While paused, pulses pass but produce silence and the source does not advance.
	buf := readFrames(t, m, 3)
	for i := range 3 {
		if l, r := frameAt(buf, i); l != 0 || r != 0 {
			t.Fatalf("paused frame %d must be silence", i)
		}
	}
	if pos := v.Position(); pos != frameDuration(3) {
		t.Fatalf("paused voice source advanced: got %v, want %v", pos, frameDuration(3))
	}
	if err := v.Resume(); err != nil {
		t.Fatal(err)
	}
	// Resuming continues from frame 3 of the source, still amplitude 0.3.
	buf = readFrames(t, m, 3)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.3) {
		t.Fatalf("resumed frame: got %v, want 0.3", l)
	}
	readFrames(t, m, 1)
	if v.IsActive() {
		t.Fatal("voice must end after its six source frames")
	}
}

// TestAudioVoiceMixerVoiceVolumeAndBusVolume: per-voice volume and the bus volume multiply, and
// invalid volumes are treated as zero.
func TestAudioVoiceMixerVoiceVolumeAndBusVolume(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()

	v, err := m.Play(f32Source(8, 1, 1), &audio.AudioVoicePlayOptions{Volume: 0.4})
	if err != nil {
		t.Fatal(err)
	}
	m.SetVolume(0.5)
	buf := readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.2) {
		t.Fatalf("voice*bus gain: got %v, want 0.2", l)
	}

	if err := v.SetVolume(math.NaN()); err != nil {
		t.Fatal(err)
	}
	buf = readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); l != 0 {
		t.Fatalf("NaN voice volume must be treated as zero, got %v", l)
	}
	if got := v.Volume(); got != 0 {
		t.Fatalf("Volume after NaN: got %v, want 0", got)
	}

	m.SetVolume(2)
	if err := v.SetVolume(0.5); err != nil {
		t.Fatal(err)
	}
	buf = readFrames(t, m, 1)
	if l, _ := frameAt(buf, 0); !approxEqual(l, 1.0) {
		t.Fatalf("bus amplification: got %v, want 1.0", l)
	}
	if got := m.Volume(); !approxEqual(float32(got), 2) {
		t.Fatalf("bus Volume: got %v, want 2", got)
	}
}

// TestAudioVoiceMixerSeek: SetPosition rewinds the source deterministically and resets envelopes.
func TestAudioVoiceMixerSeek(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	// Distinct amplitudes per frame so a seek is observable.
	data := make([]byte, 6*f32FrameBytes)
	for i := range 6 {
		putF32(data[(i*2)*4:], float32(i+1)*0.1)
		putF32(data[(i*2+1)*4:], 0)
	}
	src := &audio.AudioVoiceSource{Data: data, SampleRate: testSampleRate}
	v, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	readFrames(t, m, 2)
	// Three frames at 48kHz is an exactly round-trippable duration.
	frameDur := frameDuration(3)
	if err := v.SetPosition(frameDur); err != nil {
		t.Fatal(err)
	}
	if got := v.Position(); got != frameDur {
		t.Fatalf("Position after seek: got %v, want %v", got, frameDur)
	}
	buf := readFrames(t, m, 1)
	// Source frame 3 carries 0.4.
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.4) {
		t.Fatalf("after seek to frame 3: got %v, want 0.4", l)
	}

	// Seeking a non-seekable PlayReader voice reports an unsupported error.
	nr, err := m.PlayReader(&nonSeekableReader{r: bytes.NewReader(data)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := nr.SetPosition(0); !errors.Is(err, errors.ErrUnsupported) {
		t.Fatalf("seek on non-seekable voice: got %v, want ErrUnsupported", err)
	}
}

// nonSeekableReader hides the Seek method of an underlying reader, so a test can build a voice
// whose source is genuinely not seekable.
type nonSeekableReader struct {
	r io.Reader
}

func (n *nonSeekableReader) Read(p []byte) (int, error) {
	return n.r.Read(p)
}

// TestAudioVoiceMixerSigned16Source: a signed 16bit source is converted to 32bit float and its
// amplitudes scale by 1/(1<<15), matching the existing Player conversion.
func TestAudioVoiceMixerSigned16Source(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()

	// Two stereo int16 frames: full scale positive and half scale negative.
	i16 := []int16{32767, 32767, -16384, -16384}
	data := make([]byte, len(i16)*2)
	for i, s := range i16 {
		bits := uint16(s)
		data[i*2] = byte(bits)
		data[i*2+1] = byte(bits >> 8)
	}
	src := &audio.AudioVoiceSource{Data: data, Signed16: true, SampleRate: testSampleRate}
	v, err := m.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}

	buf := readFrames(t, m, 2)
	l0, r0 := frameAt(buf, 0)
	want0 := float32(32767) / float32(1<<15)
	if !approxEqual(l0, want0) || !approxEqual(r0, want0) {
		t.Fatalf("int16 full-scale frame: got (%v, %v), want %v", l0, r0, want0)
	}
	l1, r1 := frameAt(buf, 1)
	want1 := float32(-16384) / float32(1<<15)
	if !approxEqual(l1, want1) || !approxEqual(r1, want1) {
		t.Fatalf("int16 half-negative frame: got (%v, %v), want %v", l1, r1, want1)
	}
	readFrames(t, m, 1)
	if v.IsActive() {
		t.Fatal("int16 voice must recycle after its frames")
	}
}

// TestAudioVoiceMixerLoop: a looping voice rewinds and keeps producing its source forever, and
// stop recycles it.
func TestAudioVoiceMixerLoop(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(2)
	defer m.Close()

	v, err := m.Play(f32Source(3, 0.6, 0.6), &audio.AudioVoicePlayOptions{Loop: true})
	if err != nil {
		t.Fatal(err)
	}
	// Read well past one source length; the amplitude must remain constant.
	buf := readFrames(t, m, 10)
	for i := range 10 {
		if l, _ := frameAt(buf, i); !approxEqual(l, 0.6) {
			t.Fatalf("loop frame %d: got %v, want 0.6", i, l)
		}
	}
	if !v.IsActive() {
		t.Fatal("a looping voice must not recycle on its own")
	}
	if err := v.Stop(); err != nil {
		t.Fatal(err)
	}
	if v.IsActive() {
		t.Fatal("Stop must recycle the voice")
	}
	if err := v.Stop(); !errors.Is(err, audio.ErrAudioVoiceClosed) {
		t.Fatalf("Stop on a recycled voice: got %v, want ErrAudioVoiceClosed", err)
	}
	// All mutating operations on a stale handle are observable errors, not panics.
	if err := v.Pause(); !errors.Is(err, audio.ErrAudioVoiceClosed) {
		t.Fatalf("Pause on stale voice: got %v", err)
	}
	if err := v.SetVolume(1); !errors.Is(err, audio.ErrAudioVoiceClosed) {
		t.Fatalf("SetVolume on stale voice: got %v", err)
	}
	if err := v.FadeOut(time.Millisecond); !errors.Is(err, audio.ErrAudioVoiceClosed) {
		t.Fatalf("FadeOut on stale voice: got %v", err)
	}
}

// TestAudioVoiceMixerResampling: a source at a different sample rate is converted to the context
// rate and its frame count follows the rate ratio.
func TestAudioVoiceMixerResampling(t *testing.T) {
	// Context rate 48000; a 24000Hz source of 4 frames should resample to ~8 output frames.
	c := audio.NewOfflineAudioVoiceMixerWithSampleRateForTesting(48000, 4)
	defer c.Close()

	src := f32Source(4, 0.7, 0.7)
	src.SampleRate = 24000
	v, err := c.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The resampled voice should last 8 frames. Read across them and confirm eventual silence.
	buf := readFrames(t, c, 10)
	nonSilent := 0
	for i := range 10 {
		if l, _ := frameAt(buf, i); l != 0 {
			nonSilent++
		}
	}
	if nonSilent != 8 {
		t.Fatalf("resampled frame count: got %d non-silent frames, want 8", nonSilent)
	}
	readFrames(t, c, 1)
	if v.IsActive() {
		t.Fatal("resampled voice must recycle after its converted length")
	}
}

// TestAudioVoiceMixerSigned16Resampling: an int16 source at a different sample rate is first widened
// to float32 and then resampled; the resampler's declared length must match the widened stream, so
// the voice lasts the rate-ratio number of frames rather than ending early.
func TestAudioVoiceMixerSigned16Resampling(t *testing.T) {
	// Context 48000; a 24000Hz int16 source of 4 frames widens to float32 then resamples to 8
	// output frames.
	c := audio.NewOfflineAudioVoiceMixerWithSampleRateForTesting(48000, 4)
	defer c.Close()

	i16 := make([]int16, 4*2) // 4 stereo frames
	for i := range i16 {
		i16[i] = 16384 // quarter scale, positive
	}
	data := make([]byte, len(i16)*2)
	for i, s := range i16 {
		bits := uint16(s)
		data[i*2] = byte(bits)
		data[i*2+1] = byte(bits >> 8)
	}
	src := &audio.AudioVoiceSource{Data: data, Signed16: true, SampleRate: 24000}
	v, err := c.Play(src, nil)
	if err != nil {
		t.Fatal(err)
	}

	buf := readFrames(t, c, 10)
	nonSilent := 0
	for i := range 10 {
		if l, _ := frameAt(buf, i); l != 0 {
			nonSilent++
		}
	}
	if nonSilent != 8 {
		t.Fatalf("int16 resampled frame count: got %d non-silent frames, want 8", nonSilent)
	}
	// The first output frame carries the widened amplitude 16384/32768 = 0.5 (resampler clamps).
	if l, _ := frameAt(buf, 0); !approxEqual(l, 0.5) {
		t.Fatalf("int16 resampled amplitude: got %v, want 0.5", l)
	}
	readFrames(t, c, 1)
	if v.IsActive() {
		t.Fatal("int16 resampled voice must recycle after its converted length")
	}
}

// TestAudioVoiceMixerErrorSource: a failing source recycles its voice with an observable error but
// never disturbs the other voices or the mixer output.
func TestAudioVoiceMixerErrorSource(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(8)
	defer m.Close()

	good, err := m.Play(f32Source(20, 0.2, 0.2), nil)
	if err != nil {
		t.Fatal(err)
	}
	bad, err := m.PlayReader(&errorAfterFramesReader{frames: 2, err: errors.New("boom")}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Three frames in: the bad voice fails during the third, the good one keeps playing.
	buf := readFrames(t, m, 3)
	if l, _ := frameAt(buf, 2); !approxEqual(l, 0.2) {
		t.Fatalf("good voice must keep playing after a sibling failed, got %v", l)
	}
	if bad.IsActive() {
		t.Fatal("the failed voice must be recycled")
	}
	if err := bad.Err(); err == nil || err.Error() != "boom" {
		t.Fatalf("bad voice Err: got %v, want boom", err)
	}
	if !good.IsActive() {
		t.Fatal("the good voice must survive a sibling source error")
	}
	if got := m.Stats().FailedVoices; got != 1 {
		t.Fatalf("FailedVoices: got %d, want 1", got)
	}
}

// TestAudioVoiceMixerShortReads: a source returning one byte at a time and (0, nil) never stalls
// the mixer and the assembled frames match the source.
func TestAudioVoiceMixerShortReads(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	defer m.Close()

	src := f32Source(3, 0.4, 0.4)
	v, err := m.PlayReader(&byteAtATimeReader{data: src.Data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	buf := readFrames(t, m, 3)
	for i := range 3 {
		if l, _ := frameAt(buf, i); !approxEqual(l, 0.4) {
			t.Fatalf("short-read frame %d: got %v, want 0.4", i, l)
		}
	}
	readFrames(t, m, 1)
	if v.IsActive() {
		t.Fatal("voice must recycle after slow source ends")
	}
}

// TestAudioVoiceMixerClosedMixer: operations after Close return observable errors, Read keeps
// emitting silence, and Close is idempotent.
func TestAudioVoiceMixerClosedMixer(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	v, err := m.Play(f32Source(10, 1, 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); !errors.Is(err, audio.ErrAudioMixerClosed) {
		t.Fatalf("second Close: got %v, want ErrAudioMixerClosed", err)
	}
	if _, err := m.Play(f32Source(1, 1, 1), nil); !errors.Is(err, audio.ErrAudioMixerClosed) {
		t.Fatalf("Play after close: got %v, want ErrAudioMixerClosed", err)
	}
	// Existing voices were recycled.
	if v.IsActive() {
		t.Fatal("voices must be recycled when the mixer closes")
	}
	// Read remains safe and silent.
	buf := readFrames(t, m, 2)
	for i := range 2 {
		if l, r := frameAt(buf, i); l != 0 || r != 0 {
			t.Fatal("a closed mixer must emit silence")
		}
	}
}

// TestAudioVoiceMixerDeterministicOrder: two runs with the same operation sequence and the same
// fixed pulses produce byte-identical output, and the voice creation order fixes the summation.
func TestAudioVoiceMixerDeterministicOrder(t *testing.T) {
	mixerSetup()
	defer teardown()

	run := func() []byte {
		m := audio.NewOfflineAudioVoiceMixerForTesting(8)
		defer m.Close()
		_, _ = m.Play(f32Source(5, 0.1, 0.1), &audio.AudioVoicePlayOptions{Priority: 1})
		readFrames(t, m, 1)
		_, _ = m.Play(f32Source(5, 0.2, 0.2), &audio.AudioVoicePlayOptions{Priority: 2})
		readFrames(t, m, 1)
		_, _ = m.Play(f32Source(5, 0.3, 0.3), &audio.AudioVoicePlayOptions{Priority: 3})
		return readFrames(t, m, 6)
	}
	a := run()
	b := run()
	if !bytes.Equal(a, b) {
		t.Fatal("identical operation sequences must produce byte-identical output")
	}
}

// TestAudioVoiceMixerConcurrent: hammering Play/control/Read from many goroutines is race-free and
// never leaves an active voice beyond the limit.
func TestAudioVoiceMixerConcurrent(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewOfflineAudioVoiceMixerForTesting(16)
	defer m.Close()

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				v, err := m.Play(f32Source(8, 0.1, 0.1), nil)
				if err != nil {
					continue
				}
				_ = v.SetVolume(0.5)
				_ = v.Pause()
				_ = v.Resume()
				buf := make([]byte, f32FrameBytes)
				_, _ = m.Read(buf)
				_ = v.IsActive()
			}
		}()
	}
	wg.Wait()

	if got := m.Stats().ActiveVoices; got > 16 {
		t.Fatalf("active voices %d exceed the limit 16", got)
	}
	// Drain everything; the mixer must settle back to zero voices without hanging.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		readFrames(t, m, 8)
		if m.Stats().ActiveVoices == 0 {
			return
		}
	}
	t.Fatalf("mixer did not drain, %d voices remain", m.Stats().ActiveVoices)
}

// TestAudioVoiceMixerNoGoroutineLeak: the mixer itself starts no goroutine; its bus Player's
// reading goroutine is stopped by Close. The offline mixer has no bus at all.
func TestAudioVoiceMixerNoGoroutineLeak(t *testing.T) {
	mixerSetup()
	defer teardown()

	before := runtime.NumGoroutine()
	m := audio.NewOfflineAudioVoiceMixerForTesting(4)
	_, err := m.Play(f32Source(4, 1, 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	readFrames(t, m, 8)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	// An offline mixer never spawned a reader goroutine, so the count must not grow.
	after := runtime.NumGoroutine()
	if after > before+1 {
		t.Fatalf("goroutine count grew from %d to %d", before, after)
	}
}

// TestAudioVoiceMixerBusCloseStopsPlayer: a bus-backed mixer's Close closes its Player, so the
// context stops tracking it and no source-reading goroutine survives.
func TestAudioVoiceMixerBusCloseStopsPlayer(t *testing.T) {
	mixerSetup()
	defer teardown()

	m := audio.NewAudioVoiceMixerForTestingWithBus()
	bus := m.Player()
	if bus == nil {
		t.Fatal("a bus-backed mixer must expose its Player")
	}
	// The bus is playing and therefore tracked by the context.
	if got := audio.PlayersCountForTesting(); got < 1 {
		t.Fatalf("tracked players after creation: got %d, want >= 1", got)
	}

	if _, err := m.Play(f32Source(8, 1, 1), nil); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	// Closing released the bus: the context no longer tracks a playing player for this mixer.
	if got := audio.PlayersCountForTesting(); got != 0 {
		t.Fatalf("tracked players after Close: got %d, want 0", got)
	}
	// A second Close is an observable error rather than a panic or nil dereference.
	if err := m.Close(); !errors.Is(err, audio.ErrAudioMixerClosed) {
		t.Fatalf("second Close: got %v, want ErrAudioMixerClosed", err)
	}
}

// TestAudioVoiceMixerDeviceRebuild: closing and recreating the bus Player leaves voices,
// envelopes, and the frame clock intact, and playback continues through the new player.
func TestAudioVoiceMixerDeviceRebuild(t *testing.T) {
	mixerSetup()
	defer teardown()

	// A bus-backed mixer, driven manually through the underlying stream.
	m := audio.NewAudioVoiceMixerForTestingWithBus()
	defer m.Close()

	v, err := m.Play(f32Source(12, 0.5, 0.5), nil)
	if err != nil {
		t.Fatal(err)
	}
	readFrames(t, m, 3)
	clockBefore := m.FramePositionForTesting()

	if err := m.RebuildForTesting(); err != nil {
		t.Fatal(err)
	}
	if m.FramePositionForTesting() != clockBefore {
		t.Fatal("the frame clock must survive a device rebuild")
	}
	if !v.IsActive() {
		t.Fatal("voices must survive a device rebuild")
	}
	// The source continues right where it was: amplitude unchanged, and the voice lasts the
	// remaining nine frames before recycling.
	buf := readFrames(t, m, 9)
	for i := range 9 {
		if l, _ := frameAt(buf, i); !approxEqual(l, 0.5) {
			t.Fatalf("post-rebuild frame %d: got %v, want 0.5", i, l)
		}
	}
	readFrames(t, m, 1)
	if v.IsActive() {
		t.Fatal("voice must finish after its full 12 frames despite the rebuild")
	}
}

// errorAfterFramesReader returns a fixed error after emitting frames F32 stereo frames.
type errorAfterFramesReader struct {
	frames int
	err    error
	read   int
}

func (e *errorAfterFramesReader) Read(p []byte) (int, error) {
	if e.read >= e.frames*f32FrameBytes {
		return 0, e.err
	}
	n := min(len(p), e.frames*f32FrameBytes-e.read)
	for i := range n {
		p[i] = 0
	}
	e.read += n
	return n, nil
}

// byteAtATimeReader returns one byte per call and, on every third call, (0, nil) to exercise
// short-read handling without stalling forever.
type byteAtATimeReader struct {
	data  []byte
	pos   int
	calls int
}

func (s *byteAtATimeReader) Read(p []byte) (int, error) {
	if s.pos >= len(s.data) {
		return 0, io.EOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	s.calls++
	if s.calls%3 == 0 {
		// A no-progress, no-error read. io.ReadFull retries, so the frame is still assembled.
		return 0, nil
	}
	p[0] = s.data[s.pos]
	s.pos++
	return 1, nil
}
