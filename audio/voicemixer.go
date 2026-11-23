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

package audio

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
	"github.com/hajimehoshi/ebiten/v2/internal/mathutil"
)

// Audio voice mixer errors.
var (
	// ErrAudioVoiceLimit is returned by [AudioVoiceMixer.Play] when no further voice can be
	// started: the mixer already holds its maximum number of active voices and none of them
	// can be preempted by the new voice.
	ErrAudioVoiceLimit = errors.New("audio: voice limit reached")

	// ErrAudioVoiceClosed is returned by an [AudioVoice] operation after the voice has been
	// recycled, stopped, or preempted. A recycled voice is a stale handle and must not be reused.
	ErrAudioVoiceClosed = errors.New("audio: voice is closed")

	// ErrAudioMixerClosed is returned by an [AudioVoiceMixer] operation after the mixer was closed.
	ErrAudioMixerClosed = errors.New("audio: voice mixer is closed")
)

const (
	// bytesPerF32Frame is the size of one stereo frame in the 32bit float format.
	bytesPerF32Frame = bitDepthInBytesFloat32 * channelCount

	// defaultAudioVoiceMixerMaxVoices is the maximum number of simultaneous voices a mixer
	// created without an explicit option manages.
	defaultAudioVoiceMixerMaxVoices = 32
)

// AudioVoiceSource describes a reusable source for [AudioVoiceMixer.Play].
//
// Data is 2-channel (stereo) linear PCM without a header.
// When Signed16 is false, the data is 32bit float little endian; when it is true,
// the data is signed 16bit little endian.
//
// SampleRate is the sample rate of Data. When it differs from the audio context's sample rate,
// Play resamples the data to the context's rate.
//
// An AudioVoiceSource value is immutable and may be played any number of times, including
// concurrently: every Play creates an independent reader over Data, so playing the same source
// repeatedly never disturbs another voice.
type AudioVoiceSource struct {
	Data       []byte
	Signed16   bool
	SampleRate int
}

// AudioVoicePlayOptions configures one [AudioVoiceMixer.Play] call.
type AudioVoicePlayOptions struct {
	// Priority is the voice's preemption priority. A larger value means a higher priority.
	// When the mixer is full, a new voice preempts only an active, non-paused voice with a
	// strictly lower priority. Voices with the same priority never preempt each other, so
	// high-priority audio (for example background music) is never silently interrupted by
	// sound effects with the same or a lower priority.
	Priority int

	// Volume is the voice's initial volume, which must be in the range of [0, math.MaxFloat32].
	// A value out of the range, including NaN, is treated as 0, as with [Player.SetVolume].
	// Volume defaults to 1.
	Volume float64

	// Loop makes the voice repeat the source until it is stopped or the mixer is closed.
	Loop bool

	// FadeIn is the duration of the linear fade-in applied when the voice starts.
	// A nonpositive value starts the voice at its full volume.
	FadeIn time.Duration
}

// AudioVoiceStats reports the observable counters of an [AudioVoiceMixer].
type AudioVoiceStats struct {
	// ActiveVoices is the number of voices currently held by the mixer, including paused voices.
	ActiveVoices int

	// StartedVoices is the total number of voices started since the mixer was created.
	StartedVoices uint64

	// RecycledVoices is the total number of voices recycled after reaching their source's end,
	// failing with an error, being stopped, or being preempted.
	RecycledVoices uint64

	// PreemptedVoices is the number of voices recycled by preemption when the mixer was full.
	PreemptedVoices uint64

	// DroppedPlays is the number of Play calls rejected with [ErrAudioVoiceLimit], including
	// plays on a mixer whose maximum voice count is zero.
	DroppedPlays uint64

	// FailedVoices is the number of voices whose source returned a non-EOF error.
	FailedVoices uint64
}

// AudioVoiceMixer mixes many short voices into one stream and feeds a single [Player].
//
// A mixer manages voice handles, preemption priorities, a bus volume, fade-ins and fade-outs,
// pausing, and automatic recycling of finished voices, so that an application playing dozens of
// short sound effects does not have to manage a Player per effect itself.
//
// The mixer is itself an [io.Reader] producing 32bit float stereo data at the context's sample
// rate. Its output never ends: with no active voice it produces silence. The mixer owns a Player
// reading this stream, so its output goes through the same pipeline as any other Player and
// mixing with regular Players is done by the audio device.
//
// The mixer starts no goroutine of its own: all voice state advances in fixed pulses, namely the
// Player's reads of the mixer. Voice ordering, preemption, fades, and recycling therefore have
// deterministic results for a fixed sequence of reads.
//
// A mixer must be created with [Context.NewAudioVoiceMixer] and is closed with
// [AudioVoiceMixer.Close].
type AudioVoiceMixer struct {
	impl *audioVoiceMixerImpl

	cleanup runtime.Cleanup
}

// AudioVoice is a handle of one voice managed by an [AudioVoiceMixer].
//
// A voice is automatically recycled when its source ends, its source fails, it is stopped
// (including after a fade-out), or it is preempted by a higher-priority voice. A recycled handle
// is stale: its mutating operations return [ErrAudioVoiceClosed] and it no longer affects the mix.
type AudioVoice struct {
	mixer *audioVoiceMixerImpl
	voice *audioVoice
}

type audioVoiceMixerImpl struct {
	context    *Context
	sampleRate int
	maxVoices  int

	// bus is the Player reading this mixer. It is nil in offline tests, which drive Read directly.
	bus *Player

	// voices are ordered by creation time, oldest first. Mixing and preemption scans follow this
	// order so the result is deterministic for a fixed sequence of operations.
	voices []*audioVoice

	// carry holds trailing bytes already mixed but not yet handed to a reader whose buffer ended
	// in the middle of a frame.
	carry []byte

	volume float64
	closed bool

	started   uint64
	recycled  uint64
	preempted uint64
	dropped   uint64
	failed    uint64

	// framePos is the number of frames emitted since the mixer was created. It only advances while
	// Read mixes whole frames, never while a read merely drains carry, and is the deterministic
	// clock against which every voice envelope is evaluated.
	framePos uint64

	m sync.Mutex
}

type audioVoice struct {
	// mixer is the owning mixer, used by recycle to update the mixer's counters.
	mixer *audioVoiceMixerImpl

	// stream is the voice's source chain, already in 32bit float stereo at the mixer's sample rate.
	// It implements io.Seeker whenever the original source is seekable.
	stream io.Reader

	// seekable reports whether stream supports seeking.
	seekable bool

	// loop reports whether the source is wrapped so that it never ends on its own.
	loop bool

	priority int
	volume   float64

	// active is false once the voice has been recycled. A recycled handle is stale.
	active bool

	// paused freezes both source reads and the envelope.
	paused bool

	// err is the first non-EOF error the source returned, if any.
	err error

	// envPos is the envelope position in frames, advanced only for frames the voice emits.
	envPos uint64

	// fadeIn is the fade-in length in frames.
	fadeIn uint64

	// fadeOutStart is the mixer frame at which a requested fade-out begins.
	fadeOutStart uint64

	// fadeOutLen is the fade-out length in frames. A zero length means no fade-out is pending.
	fadeOutLen uint64

	// stopping reports that a fade-out is in progress or finished and the voice must be recycled
	// once its gain reaches zero.
	stopping bool
}

// NewAudioVoiceMixer creates an [AudioVoiceMixer] with up to 32 simultaneous voices.
func (c *Context) NewAudioVoiceMixer() *AudioVoiceMixer {
	return c.NewAudioVoiceMixerWithOptions(defaultAudioVoiceMixerMaxVoices)
}

// NewAudioVoiceMixerWithOptions creates an [AudioVoiceMixer] managing at most maxVoices
// simultaneous voices.
//
// A maxVoices of zero is valid: every Play then fails with [ErrAudioVoiceLimit] and the mixer
// only emits silence. NewAudioVoiceMixerWithOptions panics when maxVoices is negative.
func (c *Context) NewAudioVoiceMixerWithOptions(maxVoices int) *AudioVoiceMixer {
	if maxVoices < 0 {
		panic(fmt.Sprintf("audio: maxVoices must be non-negative but was %d", maxVoices))
	}
	return c.newAudioVoiceMixer(maxVoices, true)
}

// newAudioVoiceMixer constructs a mixer. When createBus is true, a Player reading the mixer is
// created and started; offline tests pass false and drive the mixer through Read themselves.
func (c *Context) newAudioVoiceMixer(maxVoices int, createBus bool) *AudioVoiceMixer {
	impl := &audioVoiceMixerImpl{
		context:    c,
		sampleRate: c.sampleRate,
		maxVoices:  maxVoices,
		volume:     1,
	}

	mixer := &AudioVoiceMixer{impl: impl}

	if createBus {
		// The mixer is an infinite stream: an exhausted source makes its voices fall silent,
		// but Read itself never returns io.EOF.
		bus, err := c.NewPlayerF32(impl)
		if err != nil {
			panic(fmt.Sprintf("audio: NewAudioVoiceMixer failed: %v", err))
		}
		impl.bus = bus
		// Start the bus immediately. Play before the device exists is recorded as a pending play
		// and materialized by the context's before-update hook, exactly as for a regular Player.
		bus.Play()

		// Close the bus if the mixer wrapper is abandoned, so its source-reading goroutine and
		// its context registration do not outlive the mixer.
		mixer.cleanup = runtime.AddCleanup(mixer, (*audioVoiceMixerImpl).finalize, impl)
	}

	return mixer
}

func (m *audioVoiceMixerImpl) finalize() {
	_ = m.Close()
}

// Player returns the [Player] that reads the mixer and feeds the audio device.
//
// The returned player is owned by the mixer and closed by [AudioVoiceMixer.Close].
// It can be used to query the playback position or to adjust the device-side buffer size.
func (m *AudioVoiceMixer) Player() *Player {
	defer runtime.KeepAlive(m)
	return m.impl.bus
}

// Read implements [io.Reader] by mixing one fixed pulse of 32bit float stereo data.
//
// Read never returns an error and never ends: when the mixer has no active voice, or after it is
// closed, it fills p with silence. A destination ending in the middle of a frame gets a
// frame-aligned remainder carried into the next read.
func (m *AudioVoiceMixer) Read(p []byte) (int, error) {
	defer runtime.KeepAlive(m)
	return m.impl.Read(p)
}

// Close stops and closes the mixer, recycles every voice, and closes the player reading it.
//
// Close is idempotent: closing an already closed mixer returns [ErrAudioMixerClosed].
// Close does not close any source passed to Play; closing sources is their owner's responsibility.
func (m *AudioVoiceMixer) Close() error {
	defer runtime.KeepAlive(m)
	return m.impl.Close()
}

func (m *audioVoiceMixerImpl) Close() error {
	m.m.Lock()
	if m.closed {
		m.m.Unlock()
		return ErrAudioMixerClosed
	}
	m.closed = true
	bus := m.bus
	m.bus = nil
	voices := m.voices
	m.voices = nil
	m.recycled += uint64(len(voices))
	for _, v := range voices {
		v.active = false
	}
	m.m.Unlock()

	if bus != nil {
		// PauseAndStopReading blocks until an in-flight read of the mixer finishes and stops the
		// goroutine reading it; Close then releases the device player.
		bus.PauseAndStopReading()
		if err := bus.Close(); err != nil {
			return err
		}
	}
	return nil
}

// rebuild closes and recreates the player reading the mixer, simulating an audio device being
// torn down and recreated. Voices, their envelopes, and the mixer's frame clock are unaffected:
// the new player resumes reading the same stream at its current position.
func (m *AudioVoiceMixer) rebuild() error {
	impl := m.impl
	impl.m.Lock()
	if impl.closed {
		impl.m.Unlock()
		return ErrAudioMixerClosed
	}
	old := impl.bus
	ctx := impl.context
	impl.bus = nil
	impl.m.Unlock()

	if old != nil {
		old.PauseAndStopReading()
		if err := old.Close(); err != nil {
			return err
		}
	}

	bus, err := ctx.NewPlayerF32(impl)
	if err != nil {
		return err
	}
	impl.m.Lock()
	impl.bus = bus
	impl.m.Unlock()
	bus.Play()
	return nil
}

// Play starts a voice playing src and returns its handle.
//
// When the mixer already holds its maximum number of active voices, Play first preempts the
// oldest active, non-paused voice whose priority is strictly lower than opts.Priority. If no such
// voice exists, Play returns [ErrAudioVoiceLimit] and records a dropped play: a voice with the
// same or a higher priority, or a paused voice, is never preempted.
//
// Play returns [ErrAudioMixerClosed] when the mixer is closed.
// opts may be nil, in which case the voice uses the default options.
func (m *AudioVoiceMixer) Play(src *AudioVoiceSource, opts *AudioVoicePlayOptions) (*AudioVoice, error) {
	defer runtime.KeepAlive(m)
	return m.impl.play(src, opts)
}

// PlayReader starts a voice over an arbitrary 32bit float stereo reader at the context's sample
// rate.
//
// Unlike [AudioVoiceMixer.Play], the reader is consumed directly: it must not be shared between
// voices, and the resulting voice cannot seek unless r implements [io.Seeker]. This entry point
// exists for callers that synthesize or stream audio that does not live in a byte slice.
func (m *AudioVoiceMixer) PlayReader(r io.Reader, opts *AudioVoicePlayOptions) (*AudioVoice, error) {
	defer runtime.KeepAlive(m)
	if r == nil {
		return nil, fmt.Errorf("audio: Play source must not be nil")
	}
	_, seekable := r.(io.Seeker)
	return m.impl.playReader(r, seekable, opts)
}

func (m *audioVoiceMixerImpl) play(src *AudioVoiceSource, opts *AudioVoicePlayOptions) (*AudioVoice, error) {
	if src == nil {
		return nil, fmt.Errorf("audio: Play source must not be nil")
	}
	if src.SampleRate <= 0 {
		return nil, fmt.Errorf("audio: source sample rate must be positive but was %d", src.SampleRate)
	}

	// Every voice gets its own bytes.Reader, so the same AudioVoiceSource can be played any
	// number of times, even concurrently, without the readers disturbing each other.
	base := bytes.NewReader(src.Data)
	seekable := true

	var stream io.Reader
	// streamLen is the byte length of the stream handed to the resampler, which must measure that
	// stream's own format. A 16bit source doubles in size when converted to 32bit float.
	streamLen := int64(len(src.Data))
	if src.Signed16 {
		stream = convert.NewFloat32BytesReaderFromInt16BytesReader(base)
		streamLen = streamLen * int64(bitDepthInBytesFloat32) / int64(bitDepthInBytesInt16)
	} else {
		stream = base
	}
	if src.SampleRate != m.sampleRate {
		stream = convert.NewResampling(stream, streamLen, src.SampleRate, m.sampleRate, bitDepthInBytesFloat32)
	}
	return m.playReader(stream, seekable, opts)
}

func (m *audioVoiceMixerImpl) playReader(r io.Reader, seekable bool, opts *AudioVoicePlayOptions) (*AudioVoice, error) {
	volume := 1.0
	var priority int
	var loop bool
	var fadeIn time.Duration
	if opts != nil {
		priority = opts.Priority
		// A zero-value opts.Volume means "unset", so keep the default of 1 rather than muting.
		// An explicitly negative, NaN, or too-large volume is still normalized to zero below.
		if opts.Volume != 0 {
			volume = opts.Volume
		}
		loop = opts.Loop
		fadeIn = opts.FadeIn
	}
	// !(volume > 0) is true for NaN as well as for negative values, matching Player.SetVolume.
	if volume < 0 || math.IsNaN(volume) || volume > math.MaxFloat32 {
		volume = 0
	}

	m.m.Lock()
	defer m.m.Unlock()

	if m.closed {
		return nil, ErrAudioMixerClosed
	}

	if len(m.voices) >= m.maxVoices {
		if !m.preemptLocked(priority) {
			m.dropped++
			return nil, ErrAudioVoiceLimit
		}
	}

	stream := r
	if loop {
		seeker, ok := r.(io.ReadSeeker)
		if !ok {
			return nil, fmt.Errorf("audio: a looping voice requires a seekable source: %w", errors.ErrUnsupported)
		}
		stream = newLoopingF32Reader(seeker)
		seekable = true
	}

	m.started++
	v := &audioVoice{
		mixer:    m,
		stream:   stream,
		seekable: seekable,
		loop:     loop,
		priority: priority,
		volume:   volume,
		active:   true,
		fadeIn:   durationToFrames(fadeIn, m.sampleRate),
	}
	m.voices = append(m.voices, v)
	return &AudioVoice{mixer: m, voice: v}, nil
}

// preemptLocked recycles the oldest active, non-paused voice whose priority is strictly lower
// than priority. It reports whether a voice was preempted. The caller holds m.m.
func (m *audioVoiceMixerImpl) preemptLocked(priority int) bool {
	for i, v := range m.voices {
		if !v.active || v.paused || v.priority >= priority {
			continue
		}
		v.active = false
		m.voices = append(m.voices[:i], m.voices[i+1:]...)
		m.preempted++
		m.recycled++
		return true
	}
	return false
}

// Read implements [io.Reader]. It is the fixed pulse at which all voice state advances.
func (m *audioVoiceMixerImpl) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	m.m.Lock()
	defer m.m.Unlock()

	if m.closed {
		// A closed mixer keeps the stream contract: it emits silence forever instead of failing a
		// read a device player may already be waiting on.
		clear(p)
		return len(p), nil
	}

	total := 0

	// First serve any frame-aligned remainder carried from a previous short destination buffer.
	if len(m.carry) > 0 {
		n := copy(p, m.carry)
		m.carry = m.carry[n:]
		p = p[n:]
		total += n
		if len(p) == 0 {
			return total, nil
		}
	}

	// Mix whole frames into p. An unaligned tail is mixed into one extra frame and carried to the
	// next read, so every mixing pulse stays frame aligned.
	frames := len(p) / bytesPerF32Frame
	headBytes := frames * bytesPerF32Frame
	if frames > 0 {
		m.mixFramesLocked(p[:headBytes], frames)
		total += headBytes
		p = p[headBytes:]
	}
	if len(p) > 0 {
		// Mix one extra frame for the unaligned tail. mixFramesLocked also applies the bus volume,
		// so the carried bytes are bus-scaled exactly like the aligned output.
		var one [bytesPerF32Frame]byte
		m.mixFramesLocked(one[:], 1)
		n := copy(p, one[:])
		m.carry = append(m.carry[:0], one[n:]...)
		total += n
	}
	return total, nil
}

// mixFramesLocked mixes frames frames into dst, which must be frames*bytesPerF32Frame long.
// The caller holds m.m.
func (m *audioVoiceMixerImpl) mixFramesLocked(dst []byte, frames int) {
	clear(dst)

	var in [bytesPerF32Frame]byte
	var dead []*audioVoice
	for _, v := range m.voices {
		if !v.active || v.paused {
			continue
		}
		for f := range frames {
			frame := m.framePos + uint64(f)
			gain := v.gainAt(frame)
			// A fade-out reaching zero recycles the voice at this deterministic frame, before the
			// source would be read again, so no further bytes of it enter the mix. The Nth frame of
			// an N-frame fade-out is the one that reaches zero.
			if v.stopping && v.fadeOutLen > 0 && frame >= v.fadeOutStart &&
				frame-v.fadeOutStart+1 >= v.fadeOutLen {
				v.recycle(nil)
				break
			}
			if !v.readFrameLocked(in[:]) {
				// The voice ended or failed inside this pulse; its remaining frames stay silent.
				break
			}
			if gain == 0 {
				continue
			}
			off := f * bytesPerF32Frame
			for ch := range channelCount {
				chOff := ch * bitDepthInBytesFloat32
				s := math.Float32frombits(loadUint32(in[chOff:]))
				d := math.Float32frombits(loadUint32(dst[off+chOff:]))
				d += s * float32(gain)
				storeUint32(dst[off+chOff:], math.Float32bits(d))
			}
		}
		if !v.active {
			dead = append(dead, v)
		}
	}
	m.dropVoicesLocked(dead)

	if m.volume != 1 {
		g := float32(m.volume)
		for i := range frames * channelCount {
			pos := i * bitDepthInBytesFloat32
			v := math.Float32frombits(loadUint32(dst[pos:])) * g
			storeUint32(dst[pos:], math.Float32bits(v))
		}
	}

	m.framePos += uint64(frames)
}

// readFrameLocked reads exactly one stereo frame into dst. A voice that ends or fails is
// recycled immediately, at this pulse boundary. It reports whether the voice remains active.
// The caller holds m.m.
func (v *audioVoice) readFrameLocked(dst []byte) bool {
	_, err := io.ReadFull(v.stream, dst)
	if err == nil {
		v.envPos++
		return true
	}

	switch {
	case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
		if v.loop {
			// A looping reader must never end; treat its end as a failure.
			v.recycle(fmt.Errorf("audio: looping source ended unexpectedly: %w", io.EOF))
			return false
		}
		v.recycle(nil)
	default:
		v.recycle(err)
	}
	return false
}

// recycle marks the voice stale and updates the mixer's counters. The caller holds the mixer mutex.
func (v *audioVoice) recycle(err error) {
	if !v.active {
		return
	}
	v.active = false
	if err != nil {
		v.err = err
		v.mixer.failed++
	}
	v.mixer.recycled++
}

// dropVoicesLocked removes the given already-recycled voices from the active list, preserving the
// creation order of the rest. The caller holds m.m.
func (m *audioVoiceMixerImpl) dropVoicesLocked(dead []*audioVoice) {
	if len(dead) == 0 {
		return
	}
	stale := make(map[*audioVoice]struct{}, len(dead))
	for _, v := range dead {
		stale[v] = struct{}{}
	}
	kept := m.voices[:0]
	for _, v := range m.voices {
		if _, ok := stale[v]; !ok {
			kept = append(kept, v)
		}
	}
	m.voices = kept
}

// gainAt returns the linear gain of the voice for the frame it is about to emit.
// mixerFrame is the mixer's frame clock; envPos is the number of frames the voice has emitted so
// far, so the upcoming frame is numbered envPos+1 within the voice's own envelope.
func (v *audioVoice) gainAt(mixerFrame uint64) float64 {
	gain := v.volume

	// Fade-in, evaluated against the voice's own envelope position. The first emitted frame has
	// envelope index 1, so its gain is 1/fadeIn rather than 0.
	if v.fadeIn > 0 && v.envPos < v.fadeIn {
		gain *= float64(v.envPos+1) / float64(v.fadeIn)
	}

	// Fade-out, evaluated against the mixer clock so that it lines up with the requested time.
	// The fade starts at full volume on its first frame and reaches zero exactly on its Nth frame,
	// symmetric with the fade-in.
	if v.stopping && v.fadeOutLen > 0 {
		if mixerFrame < v.fadeOutStart {
			return gain
		}
		elapsed := mixerFrame - v.fadeOutStart
		if elapsed+1 >= v.fadeOutLen {
			return 0
		}
		gain *= 1 - float64(elapsed+1)/float64(v.fadeOutLen)
	}
	return gain
}

// SetVolume sets the bus volume of the mixer, which is multiplied onto every mixed voice.
// The value rules are the same as [Player.SetVolume]. Setting the volume on a closed mixer is
// ignored.
func (m *AudioVoiceMixer) SetVolume(volume float64) {
	defer runtime.KeepAlive(m)
	m.impl.setVolume(volume)
}

func (m *audioVoiceMixerImpl) setVolume(volume float64) {
	if !(volume > 0) || volume > math.MaxFloat32 {
		volume = 0
	}
	m.m.Lock()
	defer m.m.Unlock()
	if m.closed {
		return
	}
	// The bus player stays at full scale: the bus volume is applied per pulse in the mix so that
	// it is observable in offline reads without a device.
	m.volume = volume
}

// Volume returns the bus volume of the mixer.
func (m *AudioVoiceMixer) Volume() float64 {
	defer runtime.KeepAlive(m)
	m.impl.m.Lock()
	defer m.impl.m.Unlock()
	return m.impl.volume
}

// Pause pauses the player reading the mixer. Voices keep their state; Resume continues them.
// On an offline mixer without a bus player, Pause does nothing.
func (m *AudioVoiceMixer) Pause() {
	defer runtime.KeepAlive(m)
	m.impl.pause()
}

func (m *audioVoiceMixerImpl) pause() {
	m.m.Lock()
	bus := m.bus
	closed := m.closed
	m.m.Unlock()
	if closed || bus == nil {
		return
	}
	bus.Pause()
}

// Resume resumes the player reading the mixer.
func (m *AudioVoiceMixer) Resume() {
	defer runtime.KeepAlive(m)
	m.impl.resume()
}

func (m *audioVoiceMixerImpl) resume() {
	m.m.Lock()
	bus := m.bus
	closed := m.closed
	m.m.Unlock()
	if closed || bus == nil {
		return
	}
	bus.Play()
}

// IsPaused reports whether the player reading the mixer is currently paused. An offline mixer
// without a bus player reports false.
func (m *AudioVoiceMixer) IsPaused() bool {
	defer runtime.KeepAlive(m)
	m.impl.m.Lock()
	bus := m.impl.bus
	m.impl.m.Unlock()
	if bus == nil {
		return false
	}
	return !bus.IsPlaying()
}

// Stats returns a snapshot of the mixer's counters.
func (m *AudioVoiceMixer) Stats() AudioVoiceStats {
	defer runtime.KeepAlive(m)
	impl := m.impl
	impl.m.Lock()
	defer impl.m.Unlock()
	return AudioVoiceStats{
		ActiveVoices:    len(impl.voices),
		StartedVoices:   impl.started,
		RecycledVoices:  impl.recycled,
		PreemptedVoices: impl.preempted,
		DroppedPlays:    impl.dropped,
		FailedVoices:    impl.failed,
	}
}

// IsActive reports whether the voice is still part of the mix.
func (v *AudioVoice) IsActive() bool {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	return v.voice.active
}

// Err returns the first non-EOF error the voice's source returned, if any. A recycled voice whose
// source ended cleanly returns nil.
func (v *AudioVoice) Err() error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	return v.voice.err
}

// Priority returns the voice's preemption priority.
func (v *AudioVoice) Priority() int {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	return v.voice.priority
}

// IsPaused reports whether the voice is paused.
func (v *AudioVoice) IsPaused() bool {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	return v.voice.active && v.voice.paused
}

// Volume returns the voice's volume, excluding any fade envelope.
func (v *AudioVoice) Volume() float64 {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	return v.voice.volume
}

// SetVolume sets the voice's volume. The value rules are the same as [Player.SetVolume].
// A fade already in progress keeps running and multiplies this volume.
// It returns [ErrAudioVoiceClosed] when the voice has been recycled.
func (v *AudioVoice) SetVolume(volume float64) error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	if !(volume > 0) || volume > math.MaxFloat32 {
		volume = 0
	}
	v.voice.volume = volume
	return nil
}

// Pause freezes the voice's source reads and envelope until [AudioVoice.Resume] is called.
// A paused voice cannot be preempted. It returns [ErrAudioVoiceClosed] on a recycled voice.
func (v *AudioVoice) Pause() error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	v.voice.paused = true
	return nil
}

// Resume continues a paused voice. Its envelope resumes from where it stopped.
// It returns [ErrAudioVoiceClosed] on a recycled voice.
func (v *AudioVoice) Resume() error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	v.voice.paused = false
	return nil
}

// FadeOut fades the voice to silence over d and recycles it when the fade finishes.
// A nonpositive d recycles the voice immediately. It returns [ErrAudioVoiceClosed] on a recycled
// voice.
func (v *AudioVoice) FadeOut(d time.Duration) error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	if d <= 0 {
		v.voice.recycle(nil)
		v.mixer.dropVoicesLocked([]*audioVoice{v.voice})
		return nil
	}
	v.voice.stopping = true
	v.voice.fadeOutStart = v.mixer.framePos
	v.voice.fadeOutLen = durationToFrames(d, v.mixer.sampleRate)
	return nil
}

// Stop recycles the voice immediately. A stopped voice contributes no more audio and its handle
// becomes stale. It returns [ErrAudioVoiceClosed] on an already recycled voice.
func (v *AudioVoice) Stop() error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	v.voice.recycle(nil)
	v.mixer.dropVoicesLocked([]*audioVoice{v.voice})
	return nil
}

// SetPosition seeks the voice's source to offset. It requires the voice's source to be seekable,
// which the sources created by [AudioVoiceMixer.Play] always are.
// The envelope position is reset to the beginning. It returns [ErrAudioVoiceClosed] on a recycled
// voice and an error wrapping [errors.ErrUnsupported] for a non-seekable source.
func (v *AudioVoice) SetPosition(offset time.Duration) error {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return ErrAudioVoiceClosed
	}
	if !v.voice.seekable {
		return fmt.Errorf("audio: voice source is not seekable: %w", errors.ErrUnsupported)
	}
	pos := int64(durationToFrames(offset, v.mixer.sampleRate)) * bytesPerF32Frame
	if _, err := v.voice.stream.(io.Seeker).Seek(pos, io.SeekStart); err != nil {
		return addErrorInfo(err)
	}
	// Reset the envelope so a seek replays fades deterministically.
	v.voice.envPos = 0
	v.voice.fadeOutStart = 0
	v.voice.fadeOutLen = 0
	v.voice.stopping = false
	return nil
}

// Position returns the voice's source position. A non-seekable voice reports zero.
func (v *AudioVoice) Position() time.Duration {
	v.mixer.m.Lock()
	defer v.mixer.m.Unlock()
	if !v.voice.active {
		return 0
	}
	seeker, ok := v.voice.stream.(io.Seeker)
	if !ok {
		return 0
	}
	pos, err := seeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0
	}
	frames := uint64(pos) / bytesPerF32Frame
	return framesToDuration(frames, v.mixer.sampleRate)
}

func durationToFrames(d time.Duration, sampleRate int) uint64 {
	if d <= 0 {
		return 0
	}
	frames, ok := mathutil.MulDiv(int64(d), int64(sampleRate), int64(time.Second))
	if !ok {
		return math.MaxUint64
	}
	return uint64(frames)
}

func framesToDuration(frames uint64, sampleRate int) time.Duration {
	nanos, ok := mathutil.MulDiv(int64(frames), int64(time.Second), int64(sampleRate))
	if !ok {
		return 0
	}
	return time.Duration(nanos)
}

func loadUint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func storeUint32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

// loopingF32Reader wraps a seekable 32bit float stereo reader and rewinds it to the start whenever
// it reaches its end, so the wrapped stream never returns io.EOF on its own.
//
// The underlying reader can return short reads, so Read keeps reading until p is full or a
// non-EOF error occurs. A (0, io.EOF) pair right after a rewind is retried once, so that an
// empty or already-exhausted source cannot make Read spin or return (0, io.EOF).
type loopingF32Reader struct {
	r io.ReadSeeker
}

func newLoopingF32Reader(r io.ReadSeeker) *loopingF32Reader {
	return &loopingF32Reader{r: r}
}

func (l *loopingF32Reader) Read(p []byte) (int, error) {
	total := 0
	rewound := false
	for total < len(p) {
		n, err := l.r.Read(p[total:])
		total += n
		if err != nil && !errors.Is(err, io.EOF) {
			return total, err
		}
		if errors.Is(err, io.EOF) {
			if _, serr := l.r.Seek(0, io.SeekStart); serr != nil {
				return total, serr
			}
			if total > 0 {
				// Hand back what was read; the next call continues from the start.
				return total, nil
			}
			if rewound {
				// The source yielded nothing even after a rewind, so it is empty. Surface EOF:
				// the voice recycles itself instead of emitting silence forever for a loop over
				// no data.
				return 0, io.EOF
			}
			rewound = true
			// The first read hit EOF immediately (a fresh, fully consumed source). Read again
			// from the start instead of returning (0, io.EOF), which the caller would treat as
			// an unexpected end of a looping stream.
			continue
		}
	}
	return total, nil
}

func (l *loopingF32Reader) Seek(offset int64, whence int) (int64, error) {
	return l.r.Seek(offset, whence)
}
