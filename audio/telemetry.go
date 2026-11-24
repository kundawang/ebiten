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

package audio

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hajimehoshi/ebiten/v2/internal/mathutil"
)

var (
	// ErrPlayerClosed is wrapped by the errors [AudioUnderrunTelemetry.Snapshot],
	// [AudioUnderrunTelemetry.Reset], and [AudioUnderrunTelemetry.SetEnabled] return after the
	// owning player was closed.
	ErrPlayerClosed = errors.New("audio: player is closed")

	// ErrSampleRateMismatch is wrapped by the error [AudioUnderrunTelemetry.Snapshot] returns
	// when the player's sample rate differs from the rate the telemetry was created with. Byte
	// counts are never interpreted at a rate other than the one they were collected with.
	ErrSampleRateMismatch = errors.New("audio: the sample rate changed since the telemetry started")

	// ErrInvalidRead is wrapped by the error the device player receives when a source returns a
	// byte count that is negative or larger than the destination buffer. The bad count is never
	// passed on, so a misbehaving source cannot corrupt the device or panic the audio thread.
	ErrInvalidRead = errors.New("audio: the source returned an invalid byte count")
)

const (
	// audioUnderrunEventHistory is the maximum number of events a snapshot carries.
	audioUnderrunEventHistory = 64
	// audioUnderrunErrorHistory is the maximum number of errors a snapshot carries.
	audioUnderrunErrorHistory = 8
	// audioUnderrunMaxDuration is the largest representable duration, used as the saturation
	// point of the duration statistics.
	audioUnderrunMaxDuration = time.Duration(math.MaxInt64)
)

// AudioUnderrunEventKind identifies what an [AudioUnderrunEvent] reports.
type AudioUnderrunEventKind int

const (
	// AudioUnderrunEventShortRead reports a source read that returned some data but fewer bytes
	// than the device requested, without an error.
	AudioUnderrunEventShortRead AudioUnderrunEventKind = iota + 1

	// AudioUnderrunEventStall reports a source read that returned no data and no error.
	AudioUnderrunEventStall

	// AudioUnderrunEventEOF reports that the source reached its end.
	AudioUnderrunEventEOF

	// AudioUnderrunEventSourceError reports a source error other than [io.EOF], including an
	// error reported by the underlying device player.
	AudioUnderrunEventSourceError

	// AudioUnderrunEventInvalidRead reports a source read whose byte count was negative or
	// larger than the destination buffer.
	AudioUnderrunEventInvalidRead

	// AudioUnderrunEventPause reports that the player was paused with [Player.Pause].
	AudioUnderrunEventPause

	// AudioUnderrunEventPauseAndStopReading reports that the player stopped reading with
	// [Player.PauseAndStopReading].
	AudioUnderrunEventPauseAndStopReading

	// AudioUnderrunEventResume reports that the player's underlying voice started playing.
	AudioUnderrunEventResume

	// AudioUnderrunEventSeek reports a successful [Player.SetPosition] or [Player.Rewind]. The
	// requested position is the event's Offset.
	AudioUnderrunEventSeek

	// AudioUnderrunEventLoop reports that an [InfiniteLoop] wrapped around its source.
	AudioUnderrunEventLoop

	// AudioUnderrunEventDeviceSuspend reports that the audio device was suspended.
	AudioUnderrunEventDeviceSuspend

	// AudioUnderrunEventDeviceResume reports that the audio device was resumed.
	AudioUnderrunEventDeviceResume

	// AudioUnderrunEventDeviceRebuild reports that the underlying voice was re-created, e.g.
	// after the audio device was rebuilt. The data buffered in the previous voice is lost.
	// PreviousVoiceID names the replaced voice.
	AudioUnderrunEventDeviceRebuild

	// AudioUnderrunEventUnderrunStart reports that the player was playing with an empty device
	// buffer after having supplied data, that is, the device ran dry.
	AudioUnderrunEventUnderrunStart

	// AudioUnderrunEventUnderrunEnd reports that a buffered-data underrun ended. The event's
	// Duration is the length of the underrun.
	AudioUnderrunEventUnderrunEnd
)

// audioUnderrunEventKindCount is one past the largest event kind.
const audioUnderrunEventKindCount = int(AudioUnderrunEventUnderrunEnd) + 1

// String returns a human-readable name of the kind.
func (k AudioUnderrunEventKind) String() string {
	switch k {
	case AudioUnderrunEventShortRead:
		return "ShortRead"
	case AudioUnderrunEventStall:
		return "Stall"
	case AudioUnderrunEventEOF:
		return "EOF"
	case AudioUnderrunEventSourceError:
		return "SourceError"
	case AudioUnderrunEventInvalidRead:
		return "InvalidRead"
	case AudioUnderrunEventPause:
		return "Pause"
	case AudioUnderrunEventPauseAndStopReading:
		return "PauseAndStopReading"
	case AudioUnderrunEventResume:
		return "Resume"
	case AudioUnderrunEventSeek:
		return "Seek"
	case AudioUnderrunEventLoop:
		return "Loop"
	case AudioUnderrunEventDeviceSuspend:
		return "DeviceSuspend"
	case AudioUnderrunEventDeviceResume:
		return "DeviceResume"
	case AudioUnderrunEventDeviceRebuild:
		return "DeviceRebuild"
	case AudioUnderrunEventUnderrunStart:
		return "UnderrunStart"
	case AudioUnderrunEventUnderrunEnd:
		return "UnderrunEnd"
	default:
		return fmt.Sprintf("AudioUnderrunEventKind(%d)", int(k))
	}
}

// AudioUnderrunEvent is one event observed by [AudioUnderrunTelemetry].
//
// Sequence increases by one per recorded event. A missing sequence number means the event was
// dropped because a snapshot held the telemetry; DroppedEventCount reports how many were
// dropped.
type AudioUnderrunEvent struct {
	// Sequence is the event's sequence number, starting at 1 after a reset.
	Sequence uint64
	// Kind is what the event reports.
	Kind AudioUnderrunEventKind
	// At is the time the event was recorded, as reported by the telemetry's clock.
	At time.Time
	// PlayerID identifies the player. IDs are unique within a process.
	PlayerID int64
	// VoiceID identifies the underlying device player that was in use. VoiceID is 0 before the
	// device exists. A device rebuild assigns a new voice.
	VoiceID int64
	// PreviousVoiceID is the voice replaced by a device rebuild, and 0 for other events.
	PreviousVoiceID int64
	// LoopCount is the number of loops an [InfiniteLoop] wrapped since the previous report.
	LoopCount uint64
	// FillLevelBytes is the byte size of the device buffer at the time of the event.
	FillLevelBytes int
	// FillLevelDuration is FillLevelBytes as playing time.
	FillLevelDuration time.Duration
	// BytesRequested is the destination buffer size of the source read.
	BytesRequested int
	// BytesRead is the number of bytes the source returned.
	BytesRead int
	// ReadLatency is how long the source read took. This is also the delay the device's read
	// loop experienced while it was waiting for this player's data.
	ReadLatency time.Duration
	// Duration is the length of a finished underrun for AudioUnderrunEventUnderrunEnd.
	Duration time.Duration
	// Offset is the requested position for AudioUnderrunEventSeek.
	Offset time.Duration
	// Err is the source error for AudioUnderrunEventSourceError and
	// AudioUnderrunEventInvalidRead.
	Err error
}

// AudioUnderrunError is one recent source error observed by [AudioUnderrunTelemetry].
type AudioUnderrunError struct {
	// Sequence is the sequence number the error got as an event.
	Sequence uint64
	// Kind is the event kind that carried the error.
	Kind AudioUnderrunEventKind
	// At is the time the error was observed.
	At time.Time
	// PlayerID identifies the player.
	PlayerID int64
	// VoiceID identifies the underlying device player.
	VoiceID int64
	// Err is the source error.
	Err error
}

// AudioUnderrunLatency summarizes a set of measured durations: source reads, or the intervals
// between device observations. A source read longer than the buffered playing time, or a device
// observation that arrives late, is what makes a device run dry.
type AudioUnderrunLatency struct {
	// Count is the number of measured durations.
	Count uint64
	// Last is the most recent duration.
	Last time.Duration
	// Max is the longest duration observed since the last reset.
	Max time.Duration
	// Total is the sum of all measured durations. The value saturates at the largest
	// representable duration instead of overflowing.
	Total time.Duration
}

// Average returns the average measured duration, or 0 when nothing was measured.
func (l AudioUnderrunLatency) Average() time.Duration {
	if l.Count == 0 {
		return 0
	}
	return time.Duration(uint64(l.Total) / l.Count)
}

// AudioUnderrunSnapshot is a consistent copy of [AudioUnderrunTelemetry]'s statistics.
type AudioUnderrunSnapshot struct {
	// Enabled reports whether collection was enabled when the snapshot was taken.
	Enabled bool
	// PlayerID identifies the player.
	PlayerID int64
	// VoiceID identifies the current underlying device player, or 0 before the device exists.
	VoiceID int64
	// SampleRate is the sample rate the statistics were collected with. Durations are derived
	// from byte counts with this rate.
	SampleRate int

	// BufferSizeBytes is the configured device buffer size in bytes, or 0 when it is unknown
	// or the driver uses its default.
	BufferSizeBytes int
	// FillLevelBytes is the most recently observed device buffer fill in bytes.
	FillLevelBytes int
	// FillLevelDuration is FillLevelBytes as playing time.
	FillLevelDuration time.Duration
	// FillLevelRatio is FillLevelBytes divided by BufferSizeBytes, clamped to [0, 1], or 0
	// when BufferSizeBytes is 0.
	FillLevelRatio float64
	// PeakFillLevelBytes is the largest fill observed since the last reset.
	PeakFillLevelBytes int
	// MinFillLevelBytes is the smallest fill observed since the last reset, or -1 when the
	// fill has not been observed yet.
	MinFillLevelBytes int

	// InUnderrun reports whether the player is currently playing with an empty device buffer.
	InUnderrun bool
	// UnderrunCount is the number of buffered-data underruns since the last reset.
	UnderrunCount uint64
	// TotalUnderrunDuration is the total length of the finished underruns.
	TotalUnderrunDuration time.Duration
	// CurrentUnderrunDuration is the length of the ongoing underrun, or 0 when there is none.
	CurrentUnderrunDuration time.Duration
	// LongestUnderrunDuration is the longest finished or ongoing underrun.
	LongestUnderrunDuration time.Duration

	// ReadCount is the number of source reads.
	ReadCount uint64
	// ShortReadCount is the number of short source reads without an error.
	ShortReadCount uint64
	// StallCount is the number of source reads that returned no data and no error.
	StallCount uint64
	// EOFCount is the number of times the source reached its end.
	EOFCount uint64
	// LoopCount is the number of times an [InfiniteLoop] wrapped around its source.
	LoopCount uint64
	// SeekCount is the number of successful position changes.
	SeekCount uint64
	// PauseCount is the number of [Player.Pause] and [Player.PauseAndStopReading] calls that
	// changed the device's state.
	PauseCount uint64
	// ResumeCount is the number of times a voice started playing.
	ResumeCount uint64
	// DeviceSuspendCount is the number of device suspensions.
	DeviceSuspendCount uint64
	// DeviceResumeCount is the number of device resumptions.
	DeviceResumeCount uint64
	// DeviceRebuildCount is the number of times the underlying voice was re-created.
	DeviceRebuildCount uint64
	// InvalidReadCount is the number of source reads whose byte count was out of range.
	InvalidReadCount uint64
	// ErrorCount is the number of source errors other than [io.EOF], including invalid reads.
	ErrorCount uint64
	// DroppedEventCount is the number of events dropped because a snapshot was being taken.
	DroppedEventCount uint64

	// ReadLatency summarizes the measured source read durations. A source read runs on the
	// device's pull path, so a long read is a device callback that arrived late for this player.
	ReadLatency AudioUnderrunLatency
	// DeviceCallbackLatency summarizes the intervals between consecutive device observations of
	// this player. A growing interval means the audio callback was delayed or starved.
	DeviceCallbackLatency AudioUnderrunLatency

	// LastError is the most recent source error, or nil.
	LastError error

	// Events holds up to the latest 64 events in sequence order.
	Events []AudioUnderrunEvent
	// Errors holds up to the latest 8 errors in sequence order.
	Errors []AudioUnderrunError
}

// audioUnderrunClock returns the current time. It is replaced in tests.
type audioUnderrunClock func() time.Time

// AudioUnderrunTelemetry collects underrun diagnostics for one [Player].
//
// Collection is disabled by default. While it is disabled, the player takes no timestamps and
// records no events, so playback timing and volume are unaffected.
//
// All methods are safe for concurrent use. The read path a device uses never blocks: if a
// snapshot holds the telemetry, the event is dropped and counted in DroppedEventCount instead
// of delaying the audio thread.
type AudioUnderrunTelemetry struct {
	// sampleRate is the rate byte counts are interpreted with.
	sampleRate int
	// sampleRateFn returns the rate the owning player currently uses.
	sampleRateFn func() int

	playerID       int64
	bytesPerSample int

	enabled atomic.Bool
	closed  atomic.Bool

	// fillBytes is the most recently observed device buffer fill in bytes.
	fillBytes  atomic.Int64
	prevFill   atomic.Int64
	peakFill   atomic.Int64
	minFill    atomic.Int64 // -1 means no observation yet.
	bufferSize atomic.Int64
	hasData    atomic.Bool
	// starved is set when the most recent source read could not fill the request, and cleared by
	// a read that filled it. An empty device buffer together with a starved source is what makes
	// an underrun audible; an empty buffer alone is normal on a device that consumes in real time.
	starved atomic.Bool

	readCount        atomic.Uint64
	underrunCount    atomic.Uint64
	loopCount        atomic.Uint64
	dropped          atomic.Uint64
	totalUnderrunNs  atomic.Uint64
	readLatencyCount atomic.Uint64
	readLatencyLast  atomic.Int64
	readLatencyMax   atomic.Int64
	readLatencyTotal atomic.Uint64
	kindCounts       [audioUnderrunEventKindCount]atomic.Uint64

	// deviceCallback* measure the intervals between consecutive device observations.
	lastObservedAt        atomic.Int64 // Unix nanoseconds, 0 when never observed.
	deviceCallbackCount   atomic.Uint64
	deviceCallbackLast    atomic.Int64
	deviceCallbackMax     atomic.Int64
	deviceCallbackTotalNs atomic.Uint64

	clock atomic.Pointer[audioUnderrunClock]

	mu sync.Mutex

	// voiceID is the current underlying voice, and 0 before the device exists.
	voiceID int64
	seq     uint64

	events     [audioUnderrunEventHistory]AudioUnderrunEvent
	eventHead  int
	eventCount int

	errors     [audioUnderrunErrorHistory]AudioUnderrunError
	errorHead  int
	errorCount int

	lastError error

	inUnderrun      bool
	underrunStart   time.Time
	currentUnderrun time.Duration
	longestUnderrun time.Duration
}

func newAudioUnderrunTelemetry(playerID int64, sampleRate int, sampleRateFn func() int) *AudioUnderrunTelemetry {
	t := &AudioUnderrunTelemetry{
		playerID:       playerID,
		sampleRate:     sampleRate,
		sampleRateFn:   sampleRateFn,
		bytesPerSample: bitDepthInBytesFloat32 * channelCount,
	}
	t.minFill.Store(-1)
	return t
}

// Enabled reports whether collection is enabled.
func (t *AudioUnderrunTelemetry) Enabled() bool {
	return t.enabled.Load()
}

// SetEnabled enables or disables collection.
//
// Enabling resets no statistics and does not change playback. Disabling stops recording new
// events and read timings; a snapshot keeps returning the data collected so far. SetEnabled
// returns an error wrapping [ErrPlayerClosed] after the player was closed.
func (t *AudioUnderrunTelemetry) SetEnabled(enabled bool) error {
	if t.closed.Load() {
		return fmt.Errorf("audio: AudioUnderrunTelemetry.SetEnabled: %w", ErrPlayerClosed)
	}
	if enabled && !t.enabled.Swap(true) {
		// The fill level sampled while collection was disabled is not the previous sample of a
		// continuous observation, so a drop from it must not be reported as an underrun.
		t.prevFill.Store(-1)
		t.starved.Store(false)
	}
	if !enabled {
		t.enabled.Store(false)
	}
	return nil
}

// Snapshot returns a consistent copy of the collected statistics.
//
// After the player is closed, Snapshot still returns the final statistics, together with an
// error wrapping [ErrPlayerClosed]. If the player's sample rate changed since the telemetry was
// created, Snapshot returns an error wrapping [ErrSampleRateMismatch].
func (t *AudioUnderrunTelemetry) Snapshot() (AudioUnderrunSnapshot, error) {
	if t.sampleRateFn != nil && t.sampleRateFn() != t.sampleRate {
		return AudioUnderrunSnapshot{}, fmt.Errorf("audio: AudioUnderrunTelemetry.Snapshot: %w", ErrSampleRateMismatch)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	now := t.now()
	bufferSize := t.bufferSize.Load()
	fill := t.fillBytes.Load()
	s := AudioUnderrunSnapshot{
		Enabled:                 t.enabled.Load(),
		PlayerID:                t.playerID,
		VoiceID:                 t.voiceID,
		SampleRate:              t.sampleRate,
		BufferSizeBytes:         int(max(0, bufferSize)),
		FillLevelBytes:          int(max(0, fill)),
		PeakFillLevelBytes:      int(max(0, t.peakFill.Load())),
		MinFillLevelBytes:       int(t.minFill.Load()),
		InUnderrun:              t.inUnderrun,
		UnderrunCount:           t.underrunCount.Load(),
		TotalUnderrunDuration:   saturatingDuration(t.totalUnderrunNs.Load()),
		CurrentUnderrunDuration: t.currentUnderrun,
		LongestUnderrunDuration: t.longestUnderrun,
		ReadCount:               t.readCount.Load(),
		ShortReadCount:          t.kindCounts[AudioUnderrunEventShortRead].Load(),
		StallCount:              t.kindCounts[AudioUnderrunEventStall].Load(),
		EOFCount:                t.kindCounts[AudioUnderrunEventEOF].Load(),
		LoopCount:               t.loopCount.Load(),
		SeekCount:               t.kindCounts[AudioUnderrunEventSeek].Load(),
		PauseCount:              t.kindCounts[AudioUnderrunEventPause].Load() + t.kindCounts[AudioUnderrunEventPauseAndStopReading].Load(),
		ResumeCount:             t.kindCounts[AudioUnderrunEventResume].Load(),
		DeviceSuspendCount:      t.kindCounts[AudioUnderrunEventDeviceSuspend].Load(),
		DeviceResumeCount:       t.kindCounts[AudioUnderrunEventDeviceResume].Load(),
		DeviceRebuildCount:      t.kindCounts[AudioUnderrunEventDeviceRebuild].Load(),
		InvalidReadCount:        t.kindCounts[AudioUnderrunEventInvalidRead].Load(),
		ErrorCount:              t.kindCounts[AudioUnderrunEventSourceError].Load() + t.kindCounts[AudioUnderrunEventInvalidRead].Load(),
		DroppedEventCount:       t.dropped.Load(),
		ReadLatency: AudioUnderrunLatency{
			Count: t.readLatencyCount.Load(),
			Last:  saturatingDuration(uint64(max(0, t.readLatencyLast.Load()))),
			Max:   saturatingDuration(uint64(max(0, t.readLatencyMax.Load()))),
			Total: saturatingDuration(t.readLatencyTotal.Load()),
		},
		DeviceCallbackLatency: AudioUnderrunLatency{
			Count: t.deviceCallbackCount.Load(),
			Last:  saturatingDuration(uint64(max(0, t.deviceCallbackLast.Load()))),
			Max:   saturatingDuration(uint64(max(0, t.deviceCallbackMax.Load()))),
			Total: saturatingDuration(t.deviceCallbackTotalNs.Load()),
		},
		LastError: t.lastError,
	}
	s.FillLevelDuration = t.bytesToDuration(int64(s.FillLevelBytes))
	if s.BufferSizeBytes > 0 {
		s.FillLevelRatio = math.Min(1, float64(s.FillLevelBytes)/float64(s.BufferSizeBytes))
	}
	if t.inUnderrun {
		s.CurrentUnderrunDuration = nonNegativeDuration(now.Sub(t.underrunStart))
		if s.CurrentUnderrunDuration > s.LongestUnderrunDuration {
			s.LongestUnderrunDuration = s.CurrentUnderrunDuration
		}
	}

	for i := range t.eventCount {
		s.Events = append(s.Events, t.events[t.eventIndex(i)])
	}
	for i := range t.errorCount {
		s.Errors = append(s.Errors, t.errors[t.errorIndex(i)])
	}

	if t.closed.Load() {
		return s, fmt.Errorf("audio: AudioUnderrunTelemetry.Snapshot: %w", ErrPlayerClosed)
	}
	return s, nil
}

func (t *AudioUnderrunTelemetry) eventIndex(i int) int {
	idx := (t.eventHead - t.eventCount + i) % audioUnderrunEventHistory
	if idx < 0 {
		idx += audioUnderrunEventHistory
	}
	return idx
}

func (t *AudioUnderrunTelemetry) errorIndex(i int) int {
	idx := (t.errorHead - t.errorCount + i) % audioUnderrunErrorHistory
	if idx < 0 {
		idx += audioUnderrunErrorHistory
	}
	return idx
}

// Reset clears all statistics, events, and errors, and restarts sequence numbers at 1.
//
// Reset is intended to be called while the player is idle. A source read running concurrently
// may record an event right after Reset and is then counted as dropped.
//
// Reset returns an error wrapping [ErrPlayerClosed] after the player was closed.
func (t *AudioUnderrunTelemetry) Reset() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed.Load() {
		return fmt.Errorf("audio: AudioUnderrunTelemetry.Reset: %w", ErrPlayerClosed)
	}

	t.seq = 0
	t.eventHead = 0
	t.eventCount = 0
	t.errorHead = 0
	t.errorCount = 0
	t.lastError = nil
	t.inUnderrun = false
	t.underrunStart = time.Time{}
	t.currentUnderrun = 0
	t.longestUnderrun = 0

	t.fillBytes.Store(0)
	t.prevFill.Store(-1)
	t.peakFill.Store(0)
	t.minFill.Store(-1)
	t.hasData.Store(false)
	t.starved.Store(false)
	t.readCount.Store(0)
	t.underrunCount.Store(0)
	t.loopCount.Store(0)
	t.dropped.Store(0)
	t.totalUnderrunNs.Store(0)
	t.readLatencyCount.Store(0)
	t.readLatencyLast.Store(0)
	t.readLatencyMax.Store(0)
	t.readLatencyTotal.Store(0)
	t.lastObservedAt.Store(0)
	t.deviceCallbackCount.Store(0)
	t.deviceCallbackLast.Store(0)
	t.deviceCallbackMax.Store(0)
	t.deviceCallbackTotalNs.Store(0)
	for i := range t.kindCounts {
		t.kindCounts[i].Store(0)
	}
	return nil
}

// close finishes collection for good and closes an ongoing underrun.
func (t *AudioUnderrunTelemetry) close() {
	if t.closed.Load() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.inUnderrun {
		t.finishUnderrunLocked(t.now(), AudioUnderrunEventUnderrunEnd)
	}
	t.closed.Store(true)
}

func (t *AudioUnderrunTelemetry) now() time.Time {
	if c := t.clock.Load(); c != nil {
		return (*c)()
	}
	return time.Now()
}

func nonNegativeDuration(d time.Duration) time.Duration {
	if d < 0 {
		return 0
	}
	return d
}

func saturatingDuration(ns uint64) time.Duration {
	if ns > uint64(math.MaxInt64) {
		return audioUnderrunMaxDuration
	}
	return time.Duration(ns)
}

func saturatingAddUint64(v *atomic.Uint64, delta uint64) {
	if delta == 0 {
		return
	}
	for {
		old := v.Load()
		n := old + delta
		if n < old {
			n = math.MaxUint64
		}
		if v.CompareAndSwap(old, n) {
			return
		}
	}
}

func saturatingAddInt64(v *atomic.Int64, delta int64) {
	if delta <= 0 {
		return
	}
	for {
		old := max(0, v.Load())
		n := old + delta
		if n < old {
			n = math.MaxInt64
		}
		if v.CompareAndSwap(old, n) {
			return
		}
	}
}

func (t *AudioUnderrunTelemetry) bytesToDuration(bytes int64) time.Duration {
	if bytes <= 0 {
		return 0
	}
	bytesPerSecond := int64(t.sampleRate) * int64(t.bytesPerSample)
	if bytesPerSecond <= 0 {
		return 0
	}
	ns, ok := mathutil.MulDiv(bytes, int64(time.Second), bytesPerSecond)
	if !ok {
		return audioUnderrunMaxDuration
	}
	return time.Duration(ns)
}

// audioUnderrunReadSeeker wraps a player's source with the telemetry. Reads are measured and
// reported; the wrapper never changes the bytes, the position, or the errors the device sees,
// except that an invalid byte count is replaced by an error wrapping [ErrInvalidRead].
type audioUnderrunReadSeeker struct {
	r     io.ReadSeeker
	t     *AudioUnderrunTelemetry
	loops loopCounter
}

func (r *audioUnderrunReadSeeker) Read(buf []byte) (int, error) {
	// The loop counter is drained even while collection is disabled, so that enabling the
	// telemetry later does not report the wraps that happened in between as one burst.
	loops := uint64(0)
	if r.loops != nil {
		loops = r.loops.takeLoops()
	}

	if !r.t.enabled.Load() || r.t.closed.Load() {
		n, err := r.r.Read(buf)
		if n < 0 || n > len(buf) {
			// A source must not corrupt the device with an out-of-range count, even when the
			// telemetry is disabled: the caller must not see the bad count.
			invalidErr := r.t.invalidReadError(n, len(buf))
			r.t.recordInvalidRead(n, len(buf), invalidErr)
			return 0, invalidErr
		}
		return n, err
	}

	start := r.t.now()
	n, err := r.r.Read(buf)
	latency := nonNegativeDuration(r.t.now().Sub(start))
	if n < 0 || n > len(buf) {
		invalidErr := r.t.invalidReadError(n, len(buf))
		r.t.recordInvalidRead(n, len(buf), invalidErr)
		return 0, invalidErr
	}
	r.t.recordRead(n, len(buf), err, latency)
	r.t.recordLoops(loops)
	return n, err
}

func (r *audioUnderrunReadSeeker) Seek(offset int64, whence int) (int64, error) {
	// A seek discards the wraps accumulated at the old position; they are not a playback loop.
	if r.loops != nil {
		r.loops.takeLoops()
	}
	return r.r.Seek(offset, whence)
}

func (t *AudioUnderrunTelemetry) invalidReadError(n, requested int) error {
	return fmt.Errorf("%w: read %d bytes into a %d-byte buffer", ErrInvalidRead, n, requested)
}

// setVoice records the underlying voice in use. voiceID is 0 when there is none.
func (t *AudioUnderrunTelemetry) setVoice(voiceID int64) {
	t.hasData.Store(false)
	t.prevFill.Store(-1)
	t.mu.Lock()
	t.voiceID = voiceID
	t.mu.Unlock()
}

// setBufferSize records the configured device buffer size. A nonpositive size means unknown.
func (t *AudioUnderrunTelemetry) setBufferSize(size int) {
	t.bufferSize.Store(int64(max(0, size)))
}

func (t *AudioUnderrunTelemetry) countKind(kind AudioUnderrunEventKind) {
	if kind <= 0 || int(kind) >= len(t.kindCounts) {
		return
	}
	saturatingAddUint64(&t.kindCounts[kind], 1)
}

// appendLocked appends e with the next sequence number, the current voice, and a timestamp.
// t.mu must be held.
func (t *AudioUnderrunTelemetry) appendLocked(kind AudioUnderrunEventKind, e AudioUnderrunEvent) {
	t.seq++
	e.Sequence = t.seq
	e.Kind = kind
	e.At = t.now()
	e.PlayerID = t.playerID
	e.VoiceID = t.voiceID
	t.events[t.eventHead] = e
	t.eventHead = (t.eventHead + 1) % audioUnderrunEventHistory
	if t.eventCount < audioUnderrunEventHistory {
		t.eventCount++
	}
	if e.Err != nil {
		t.lastError = e.Err
		t.errors[t.errorHead] = AudioUnderrunError{
			Sequence: e.Sequence,
			Kind:     kind,
			At:       e.At,
			PlayerID: e.PlayerID,
			VoiceID:  e.VoiceID,
			Err:      e.Err,
		}
		t.errorHead = (t.errorHead + 1) % audioUnderrunErrorHistory
		if t.errorCount < audioUnderrunErrorHistory {
			t.errorCount++
		}
	}
}

// record records an event without blocking. It is called from the goroutine reading the source,
// so it never waits for a snapshot: a contended event is dropped and counted.
func (t *AudioUnderrunTelemetry) record(kind AudioUnderrunEventKind, e AudioUnderrunEvent) {
	if !t.enabled.Load() || t.closed.Load() {
		return
	}
	t.countKind(kind)
	if !t.mu.TryLock() {
		saturatingAddUint64(&t.dropped, 1)
		return
	}
	defer t.mu.Unlock()
	t.appendLocked(kind, e)
}

// recordBlocking records an event from a goroutine that may wait for a snapshot. It is never
// called from the source-reading goroutine.
func (t *AudioUnderrunTelemetry) recordBlocking(kind AudioUnderrunEventKind, e AudioUnderrunEvent) {
	if !t.enabled.Load() || t.closed.Load() {
		return
	}
	t.countKind(kind)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.appendLocked(kind, e)
}

func (t *AudioUnderrunTelemetry) fillEvent() AudioUnderrunEvent {
	fill := int(max(0, t.fillBytes.Load()))
	return AudioUnderrunEvent{
		FillLevelBytes:    fill,
		FillLevelDuration: t.bytesToDuration(int64(fill)),
	}
}

// recordRead reports one source read. It is called from the source-reading goroutine and never
// blocks. The event carries the fill level sampled at the last context update, because querying
// the device player here could deadlock a driver.
func (t *AudioUnderrunTelemetry) recordRead(bytesRead, bytesRequested int, err error, latency time.Duration) {
	if !t.enabled.Load() || t.closed.Load() {
		return
	}

	saturatingAddUint64(&t.readCount, 1)
	saturatingAddUint64(&t.readLatencyCount, 1)
	saturatingAddUint64(&t.readLatencyTotal, uint64(nonNegativeDuration(latency)))
	t.readLatencyLast.Store(int64(nonNegativeDuration(latency)))
	for {
		old := t.readLatencyMax.Load()
		if old >= int64(latency) || t.readLatencyMax.CompareAndSwap(old, int64(nonNegativeDuration(latency))) {
			break
		}
	}

	if bytesRead > 0 {
		t.hasData.Store(true)
	}
	// A read that could not fill the request means the source cannot keep up; a read that filled
	// it means the source recovered.
	t.starved.Store(!errors.Is(err, io.EOF) && bytesRead < bytesRequested)

	event := t.fillEvent()
	event.BytesRequested = bytesRequested
	event.BytesRead = bytesRead
	event.ReadLatency = nonNegativeDuration(latency)

	var kind AudioUnderrunEventKind
	switch {
	case err != nil && !errors.Is(err, io.EOF):
		kind = AudioUnderrunEventSourceError
		event.Err = err
	case err != nil:
		kind = AudioUnderrunEventEOF
	case bytesRead == 0:
		kind = AudioUnderrunEventStall
	case bytesRead < bytesRequested:
		kind = AudioUnderrunEventShortRead
	default:
		return
	}

	t.record(kind, event)
}

// recordInvalidRead reports a read whose byte count was out of range. It never blocks.
//
// An invalid read is counted even while collection is disabled: it is an error the caller must
// see regardless, and counting it takes no clock reading, so playback timing is unaffected.
func (t *AudioUnderrunTelemetry) recordInvalidRead(bytesRead, bytesRequested int, err error) {
	t.countKind(AudioUnderrunEventInvalidRead)
	if !t.enabled.Load() || t.closed.Load() {
		return
	}
	event := t.fillEvent()
	event.BytesRequested = bytesRequested
	event.BytesRead = bytesRead
	event.Err = err
	// The kind was already counted above, so append directly instead of going through record.
	if !t.mu.TryLock() {
		saturatingAddUint64(&t.dropped, 1)
		return
	}
	defer t.mu.Unlock()
	t.appendLocked(AudioUnderrunEventInvalidRead, event)
}

// recordLoops reports that an InfiniteLoop wrapped count times since the previous report. It is
// called from the source-reading goroutine and never blocks.
func (t *AudioUnderrunTelemetry) recordLoops(count uint64) {
	if count == 0 || !t.enabled.Load() || t.closed.Load() {
		return
	}
	saturatingAddUint64(&t.loopCount, count)
	event := t.fillEvent()
	event.LoopCount = count
	t.record(AudioUnderrunEventLoop, event)
}

// recordTransport records an event that changes the flow of data to the device: a pause, seek,
// resume, or device state change. Such an event ends an ongoing underrun, as the device is no
// longer running dry while the flow is stopped or redirected.
func (t *AudioUnderrunTelemetry) recordTransport(kind AudioUnderrunEventKind, e AudioUnderrunEvent, endUnderrun bool) {
	if !t.enabled.Load() || t.closed.Load() {
		return
	}
	t.countKind(kind)
	t.mu.Lock()
	defer t.mu.Unlock()
	if endUnderrun && t.inUnderrun {
		t.finishUnderrunLocked(t.now(), AudioUnderrunEventUnderrunEnd)
	}
	t.appendLocked(kind, e)
	// The next observation is not continuous with the samples before this event, so it must not
	// report a drop from a stale fill level.
	t.prevFill.Store(-1)
	t.hasData.Store(false)
	t.starved.Store(false)
}

func (t *AudioUnderrunTelemetry) fillEventFor(fillBytes int) AudioUnderrunEvent {
	fillBytes = max(0, fillBytes)
	return AudioUnderrunEvent{
		FillLevelBytes:    fillBytes,
		FillLevelDuration: t.bytesToDuration(int64(fillBytes)),
	}
}

func (t *AudioUnderrunTelemetry) recordSeek(offset time.Duration, fillBytes int) {
	e := t.fillEventFor(fillBytes)
	e.Offset = nonNegativeDuration(offset)
	t.recordTransport(AudioUnderrunEventSeek, e, true)
}

func (t *AudioUnderrunTelemetry) recordPause(stopReading bool, fillBytes int) {
	kind := AudioUnderrunEventPause
	if stopReading {
		kind = AudioUnderrunEventPauseAndStopReading
	}
	t.recordTransport(kind, t.fillEventFor(fillBytes), true)
}

func (t *AudioUnderrunTelemetry) recordResume(fillBytes int) {
	t.recordTransport(AudioUnderrunEventResume, t.fillEventFor(fillBytes), true)
}

func (t *AudioUnderrunTelemetry) recordDeviceSuspend(fillBytes int) {
	t.recordTransport(AudioUnderrunEventDeviceSuspend, t.fillEventFor(fillBytes), true)
}

func (t *AudioUnderrunTelemetry) recordDeviceResume(fillBytes int) {
	t.recordTransport(AudioUnderrunEventDeviceResume, t.fillEventFor(fillBytes), true)
}

func (t *AudioUnderrunTelemetry) recordDeviceRebuild(previousVoiceID int64, fillBytes int) {
	e := t.fillEventFor(fillBytes)
	e.PreviousVoiceID = previousVoiceID
	t.recordTransport(AudioUnderrunEventDeviceRebuild, e, true)
}

// recordDeviceError reports an error from the underlying device player. It is called from a
// goroutine that may wait.
func (t *AudioUnderrunTelemetry) recordDeviceError(err error) {
	if err == nil {
		return
	}
	e := t.fillEvent()
	e.Err = err
	t.recordBlocking(AudioUnderrunEventSourceError, e)
}

// observeTick samples the device buffer. playing reports whether the underlying voice is playing,
// buffered is its buffered byte size, bufferSize is the configured buffer size (0 if unknown),
// and voiceID identifies the sampled voice.
//
// An underrun starts when the voice is playing with an empty buffer while data is available but
// cannot keep up: either the fill dropped from a non-empty sample, or the source is returning
// short reads. It ends when data is buffered again, when the voice stops, or when the data flow
// is redirected by a seek, pause, or device change. observeTick is called from the context's
// update goroutine, so it may wait for a snapshot.
func (t *AudioUnderrunTelemetry) observeTick(playing bool, buffered, bufferSize int, voiceID int64) {
	if !t.enabled.Load() || t.closed.Load() {
		return
	}

	buffered = max(0, buffered)
	prev := t.prevFill.Swap(int64(buffered))
	t.fillBytes.Store(int64(buffered))
	for {
		old := t.peakFill.Load()
		if old >= int64(buffered) || t.peakFill.CompareAndSwap(old, int64(buffered)) {
			break
		}
	}
	for {
		old := t.minFill.Load()
		if old >= 0 && old <= int64(buffered) {
			break
		}
		if t.minFill.CompareAndSwap(old, int64(buffered)) {
			break
		}
	}
	if bufferSize > 0 {
		t.bufferSize.Store(int64(bufferSize))
	}
	if buffered > 0 {
		t.hasData.Store(true)
	}

	// The interval between observations is the delay the device callback experienced for this
	// player. The clock is read once and shared with the underrun bookkeeping below.
	now := t.now()
	nowNs := now.UnixNano()
	if prevNs := t.lastObservedAt.Swap(max(0, nowNs)); prevNs > 0 && nowNs > prevNs {
		d := nowNs - prevNs
		saturatingAddUint64(&t.deviceCallbackCount, 1)
		saturatingAddUint64(&t.deviceCallbackTotalNs, uint64(d))
		t.deviceCallbackLast.Store(d)
		for {
			old := t.deviceCallbackMax.Load()
			if old >= d || t.deviceCallbackMax.CompareAndSwap(old, d) {
				break
			}
		}
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.voiceID = voiceID

	if t.inUnderrun {
		if buffered > 0 || !playing {
			t.finishUnderrunLocked(now, AudioUnderrunEventUnderrunEnd)
		}
		return
	}

	if !playing {
		return
	}
	// A buffer that is empty now means the device has nothing to output. That is an underrun when
	// it fell from a non-empty sample, or when the source cannot fill the device's requests: an
	// empty buffer with a source that always fills a request is a device consuming in real time,
	// not a starved player. The first sample of an enabled observation is not a drop: prevFill is
	// negative after enabling, a seek, or a change of voice, which breaks the continuity of the
	// observations.
	if buffered > 0 || !t.hasData.Load() {
		return
	}
	if prev <= 0 && !t.starved.Load() {
		return
	}

	t.inUnderrun = true
	t.underrunStart = now
	t.currentUnderrun = 0
	saturatingAddUint64(&t.underrunCount, 1)
	event := t.fillEvent()
	t.appendLocked(AudioUnderrunEventUnderrunStart, event)
}

// noteSeekState resets the data observation after a seek, so that the data buffered by the old
// position is not taken as a sign that the new position is filling the device.
func (t *AudioUnderrunTelemetry) noteSeekState() {
	t.hasData.Store(false)
	t.prevFill.Store(-1)
}

// finishUnderrunLocked ends an ongoing underrun. t.mu must be held.
func (t *AudioUnderrunTelemetry) finishUnderrunLocked(now time.Time, kind AudioUnderrunEventKind) {
	if !t.inUnderrun {
		return
	}
	d := nonNegativeDuration(now.Sub(t.underrunStart))
	t.inUnderrun = false
	t.currentUnderrun = 0
	t.underrunStart = time.Time{}
	saturatingAddUint64(&t.totalUnderrunNs, uint64(d))
	if d > t.longestUnderrun {
		t.longestUnderrun = d
	}
	event := t.fillEvent()
	event.Duration = d
	t.appendLocked(kind, event)
}
