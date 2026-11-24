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
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
)

// fakeClock is a controllable clock for the telemetry.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// readSeekerSource is a scripted source for the telemetry wrapper.
type readSeekerSource struct {
	reads []readSeekerRead
	pos   int64
}

type readSeekerRead struct {
	data []byte
	err  error
}

func (s *readSeekerSource) Read(buf []byte) (int, error) {
	if len(s.reads) == 0 {
		return 0, io.EOF
	}
	r := s.reads[0]
	s.reads = s.reads[1:]
	n := copy(buf, r.data)
	s.pos += int64(n)
	return n, r.err
}

func (s *readSeekerSource) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
		s.pos = offset
	case io.SeekCurrent:
		s.pos += offset
	default:
		return 0, errors.New("unsupported")
	}
	return s.pos, nil
}

func newTestTelemetry(t *testing.T, sampleRate int) (*AudioUnderrunTelemetry, *fakeClock) {
	t.Helper()
	clock := newFakeClock()
	tel := newAudioUnderrunTelemetry(7, sampleRate, func() int { return sampleRate })
	tel.SetClockForTesting(clock.Now)
	tel.SetEnabledForTesting(true)
	t.Cleanup(func() { tel.CloseForTesting() })
	return tel, clock
}

func TestTelemetryDisabledByDefault(t *testing.T) {
	clock := newFakeClock()
	tel := newAudioUnderrunTelemetry(1, 44100, nil)
	tel.SetClockForTesting(clock.Now)

	if tel.Enabled() {
		t.Fatal("telemetry must be disabled by default")
	}

	src := &readSeekerSource{reads: []readSeekerRead{{data: make([]byte, 4)}}}
	r := &audioUnderrunReadSeeker{r: src, t: tel}
	buf := make([]byte, 16)
	n, err := r.Read(buf)
	if n != 4 || err != nil {
		t.Fatalf("Read: got (%d, %v), want (4, nil)", n, err)
	}

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.Enabled {
		t.Error("snapshot reports enabled for a disabled telemetry")
	}
	if snap.ReadCount != 0 || snap.ShortReadCount != 0 || len(snap.Events) != 0 {
		t.Errorf("disabled telemetry recorded data: %+v", snap)
	}
}

// TestTelemetryDisabledPreservesPlaybackTiming checks that the disabled path does no timing at
// all: the clock must not be consulted while reading.
func TestTelemetryDisabledPreservesPlaybackTiming(t *testing.T) {
	var clockCalls atomic.Int64
	tel := newAudioUnderrunTelemetry(1, 44100, nil)
	tel.SetClockForTesting(func() time.Time {
		clockCalls.Add(1)
		return time.Now()
	})
	defer tel.CloseForTesting()

	src := &readSeekerSource{reads: []readSeekerRead{
		{data: make([]byte, 8)},
		{data: make([]byte, 3)},
		{err: io.EOF},
	}}
	r := &audioUnderrunReadSeeker{r: src, t: tel}
	buf := make([]byte, 16)
	for range 3 {
		if _, err := r.Read(buf); err != nil && !errors.Is(err, io.EOF) {
			t.Fatal(err)
		}
	}
	if got := clockCalls.Load(); got != 0 {
		t.Errorf("the clock was consulted %d times while the telemetry was disabled; want 0", got)
	}
}

func TestTelemetryReadClassification(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	srcErr := errors.New("source failed")
	src := &readSeekerSource{reads: []readSeekerRead{
		{data: make([]byte, 16)},             // full read: not an underrun event
		{data: make([]byte, 4)},              // short read
		{},                                   // stall
		{err: io.EOF},                        // EOF
		{data: make([]byte, 16)},             // full read
		{data: make([]byte, 2), err: srcErr}, // a source error carrying data
	}}
	// The reads are sized so that the short read and the data-carrying error are distinct.
	r := &audioUnderrunReadSeeker{r: src, t: tel}
	buf := make([]byte, 16)
	for i := 0; i < 6; i++ {
		clock.Advance(time.Millisecond)
		if _, err := r.Read(buf); err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, srcErr) {
			t.Fatalf("read %d: %v", i, err)
		}
	}

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.ReadCount, uint64(6); got != want {
		t.Errorf("ReadCount: got %d, want %d", got, want)
	}
	if got, want := snap.ShortReadCount, uint64(1); got != want {
		t.Errorf("ShortReadCount: got %d, want %d", got, want)
	}
	if got, want := snap.StallCount, uint64(1); got != want {
		t.Errorf("StallCount: got %d, want %d", got, want)
	}
	if got, want := snap.EOFCount, uint64(1); got != want {
		t.Errorf("EOFCount: got %d, want %d", got, want)
	}
	if got, want := snap.ErrorCount, uint64(1); got != want {
		t.Errorf("ErrorCount: got %d, want %d", got, want)
	}
	if !errors.Is(snap.LastError, srcErr) {
		t.Errorf("LastError: got %v, want %v", snap.LastError, srcErr)
	}

	// The events are in sequence order and each carries the read's byte counts.
	var kinds []AudioUnderrunEventKind
	for i, e := range snap.Events {
		if want := uint64(i + 1); e.Sequence != want {
			t.Errorf("event %d: Sequence: got %d, want %d", i, e.Sequence, want)
		}
		if e.PlayerID != 7 {
			t.Errorf("event %d: PlayerID: got %d, want 7", i, e.PlayerID)
		}
		kinds = append(kinds, e.Kind)
	}
	want := []AudioUnderrunEventKind{
		AudioUnderrunEventShortRead,
		AudioUnderrunEventStall,
		AudioUnderrunEventEOF,
		AudioUnderrunEventSourceError,
	}
	if len(kinds) != len(want) {
		t.Fatalf("event kinds: got %v, want %v", kinds, want)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Errorf("event kind %d: got %v, want %v", i, kinds[i], want[i])
		}
	}
}

func TestTelemetryReadLatency(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	readDurations := []time.Duration{3 * time.Millisecond, 10 * time.Millisecond, 5 * time.Millisecond}
	// The source advances the clock by the scripted duration during each read, so the wrapper
	// measures exactly that as the read latency.
	src := &slowClockReadSeeker{clock: clock, durations: readDurations}
	r := &audioUnderrunReadSeeker{r: src, t: tel}
	buf := make([]byte, 16)
	for range readDurations {
		if _, err := r.Read(buf); err != nil {
			t.Fatal(err)
		}
	}

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.ReadLatency.Count, uint64(3); got != want {
		t.Fatalf("ReadLatency.Count: got %d, want %d", got, want)
	}
	if got, want := snap.ReadLatency.Last, 5*time.Millisecond; got != want {
		t.Errorf("ReadLatency.Last: got %v, want %v", got, want)
	}
	if got, want := snap.ReadLatency.Max, 10*time.Millisecond; got != want {
		t.Errorf("ReadLatency.Max: got %v, want %v", got, want)
	}
	if got, want := snap.ReadLatency.Total, 18*time.Millisecond; got != want {
		t.Errorf("ReadLatency.Total: got %v, want %v", got, want)
	}
	if got, want := snap.ReadLatency.Average(), 6*time.Millisecond; got != want {
		t.Errorf("ReadLatency.Average(): got %v, want %v", got, want)
	}
}

// slowClockReadSeeker advances a fake clock while reading, to simulate a slow source.
type slowClockReadSeeker struct {
	clock     *fakeClock
	durations []time.Duration
}

func (s *slowClockReadSeeker) Read(buf []byte) (int, error) {
	if len(s.durations) == 0 {
		return 0, io.EOF
	}
	d := s.durations[0]
	s.durations = s.durations[1:]
	s.clock.Advance(d)
	return copy(buf, make([]byte, 16)), nil
}

func (s *slowClockReadSeeker) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func TestTelemetryNegativeLatencyIsClamped(t *testing.T) {
	tel := newAudioUnderrunTelemetry(1, 48000, nil)
	defer tel.CloseForTesting()
	tel.SetEnabledForTesting(true)
	tel.recordRead(16, 16, nil, -time.Second)

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.ReadLatency.Last != 0 || snap.ReadLatency.Max != 0 || snap.ReadLatency.Total != 0 {
		t.Errorf("a negative latency must be clamped to 0: %+v", snap.ReadLatency)
	}
}

func TestTelemetryCountersSaturate(t *testing.T) {
	tel := newAudioUnderrunTelemetry(1, 48000, nil)
	defer tel.CloseForTesting()
	tel.SetEnabledForTesting(true)

	tel.readCount.Store(math.MaxUint64)
	// The helper used by the telemetry must saturate instead of wrapping.
	saturatingAddUint64(&tel.readCount, 1)
	if got := tel.readCount.Load(); got != math.MaxUint64 {
		t.Errorf("uint64 counter: got %d, want %d", got, uint64(math.MaxUint64))
	}

	tel.readLatencyTotal.Store(math.MaxUint64)
	saturatingAddUint64(&tel.readLatencyTotal, 1)
	if got := tel.readLatencyTotal.Load(); got != math.MaxUint64 {
		t.Errorf("latency total: got %d, want %d", got, uint64(math.MaxUint64))
	}

	tel.readLatencyMax.Store(math.MaxInt64)
	saturatingAddInt64(&tel.readLatencyMax, 1)
	if got := tel.readLatencyMax.Load(); got != math.MaxInt64 {
		t.Errorf("latency max: got %d, want %d", got, math.MaxInt64)
	}

	// A saturated duration sum must report the maximum duration, not a wrapped negative one.
	if got := saturatingDuration(math.MaxUint64); got != audioUnderrunMaxDuration {
		t.Errorf("saturatingDuration: got %v, want %v", got, audioUnderrunMaxDuration)
	}
}

func TestTelemetryDeviceCallbackLatency(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	intervals := []time.Duration{5 * time.Millisecond, 20 * time.Millisecond, 10 * time.Millisecond}
	// The first observation has no previous time, so it does not produce a latency sample.
	tel.observeTick(true, 64, 0, 1)
	for _, d := range intervals {
		clock.Advance(d)
		tel.observeTick(true, 64, 0, 1)
	}

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	got := snap.DeviceCallbackLatency
	if got.Count != 3 {
		t.Fatalf("DeviceCallbackLatency.Count: got %d, want 3", got.Count)
	}
	if got.Last != 10*time.Millisecond {
		t.Errorf("DeviceCallbackLatency.Last: got %v, want 10ms", got.Last)
	}
	if got.Max != 20*time.Millisecond {
		t.Errorf("DeviceCallbackLatency.Max: got %v, want 20ms", got.Max)
	}
	if got.Total != 35*time.Millisecond {
		t.Errorf("DeviceCallbackLatency.Total: got %v, want 35ms", got.Total)
	}
}

func TestTelemetryUnderrunLifecycle(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	// The first observation has no previous fill, so it is not a drop.
	tel.observeTick(true, 100, 0, 1)
	clock.Advance(10 * time.Millisecond)
	tel.observeTick(true, 0, 0, 1)

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if !snap.InUnderrun {
		t.Fatal("expected an ongoing underrun")
	}
	if got, want := snap.UnderrunCount, uint64(1); got != want {
		t.Errorf("UnderrunCount: got %d, want %d", got, want)
	}
	if got, want := snap.MinFillLevelBytes, 0; got != want {
		t.Errorf("MinFillLevelBytes: got %d, want %d", got, want)
	}

	clock.Advance(40 * time.Millisecond)
	snap, err = tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.CurrentUnderrunDuration, 40*time.Millisecond; got != want {
		t.Errorf("CurrentUnderrunDuration: got %v, want %v", got, want)
	}

	clock.Advance(10 * time.Millisecond)
	tel.observeTick(true, 64, 0, 1)
	snap, err = tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.InUnderrun {
		t.Error("the underrun must end once data is buffered")
	}
	if got, want := snap.CurrentUnderrunDuration, time.Duration(0); got != want {
		t.Errorf("CurrentUnderrunDuration after end: got %v, want %v", got, want)
	}
	if got, want := snap.TotalUnderrunDuration, 50*time.Millisecond; got != want {
		t.Errorf("TotalUnderrunDuration: got %v, want %v", got, want)
	}
	if got, want := snap.LongestUnderrunDuration, 50*time.Millisecond; got != want {
		t.Errorf("LongestUnderrunDuration: got %v, want %v", got, want)
	}

	var kinds []AudioUnderrunEventKind
	for _, e := range snap.Events {
		kinds = append(kinds, e.Kind)
	}
	want := []AudioUnderrunEventKind{AudioUnderrunEventUnderrunStart, AudioUnderrunEventUnderrunEnd}
	if fmt.Sprint(kinds) != fmt.Sprint(want) {
		t.Errorf("event kinds: got %v, want %v", kinds, want)
	}
	if got, want := snap.Events[1].Duration, 50*time.Millisecond; got != want {
		t.Errorf("UnderrunEnd duration: got %v, want %v", got, want)
	}

	// The fill statistics tracked the samples.
	if got, want := snap.PeakFillLevelBytes, 100; got != want {
		t.Errorf("PeakFillLevelBytes: got %d, want %d", got, want)
	}
	if got, want := snap.FillLevelBytes, 64; got != want {
		t.Errorf("FillLevelBytes: got %d, want %d", got, want)
	}
	// 64 bytes of 32bit float stereo at 48000Hz.
	if want := time.Duration(int64(64) * int64(time.Second) / (48000 * 8)); snap.FillLevelDuration != want {
		t.Errorf("FillLevelDuration: got %v, want %v", snap.FillLevelDuration, want)
	}
}

func TestTelemetryUnderrunEndsWhenVoiceStops(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	tel.observeTick(true, 32, 0, 1)
	clock.Advance(5 * time.Millisecond)
	tel.observeTick(true, 0, 0, 1)
	clock.Advance(5 * time.Millisecond)
	// The voice stops without refilling: the underrun is finalized rather than left open.
	tel.observeTick(false, 0, 0, 1)

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.InUnderrun {
		t.Error("an underrun must not stay open after the voice stopped")
	}
	if got, want := snap.LongestUnderrunDuration, 5*time.Millisecond; got != want {
		t.Errorf("LongestUnderrunDuration: got %v, want %v", got, want)
	}
}

func TestTelemetryReset(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	tel.observeTick(true, 64, 0, 1)
	clock.Advance(time.Millisecond)
	tel.observeTick(true, 0, 0, 1)
	tel.recordRead(4, 16, nil, time.Millisecond)

	if err := tel.Reset(); err != nil {
		t.Fatal(err)
	}

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.InUnderrun || snap.UnderrunCount != 0 || snap.ReadCount != 0 || len(snap.Events) != 0 {
		t.Errorf("Reset left statistics behind: %+v", snap)
	}
	if got, want := snap.MinFillLevelBytes, -1; got != want {
		t.Errorf("MinFillLevelBytes after reset: got %d, want %d", got, want)
	}
	if !snap.Enabled {
		t.Error("Reset must not disable collection")
	}

	// Sequence numbers restart at 1.
	tel.recordRead(4, 16, nil, time.Millisecond)
	snap, err = tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Events) != 1 || snap.Events[0].Sequence != 1 {
		t.Errorf("after reset: events = %+v, want one event with sequence 1", snap.Events)
	}
}

func TestTelemetryClose(t *testing.T) {
	tel, clock := newTestTelemetry(t, 48000)

	tel.observeTick(true, 64, 0, 1)
	clock.Advance(2 * time.Millisecond)
	tel.observeTick(true, 0, 0, 1)
	clock.Advance(3 * time.Millisecond)

	tel.CloseForTesting()

	snap, err := tel.Snapshot()
	if !errors.Is(err, ErrPlayerClosed) {
		t.Errorf("Snapshot after close: got %v, want %v", err, ErrPlayerClosed)
	}
	// The final statistics are still returned.
	if snap.InUnderrun {
		t.Error("closing must finalize the ongoing underrun")
	}
	if got, want := snap.LongestUnderrunDuration, 3*time.Millisecond; got != want {
		t.Errorf("LongestUnderrunDuration after close: got %v, want %v", got, want)
	}
	if err := tel.Reset(); !errors.Is(err, ErrPlayerClosed) {
		t.Errorf("Reset after close: got %v, want %v", err, ErrPlayerClosed)
	}
	if err := tel.SetEnabled(true); !errors.Is(err, ErrPlayerClosed) {
		t.Errorf("SetEnabled after close: got %v, want %v", err, ErrPlayerClosed)
	}

	// A closed telemetry records nothing more.
	before := snap.ReadCount
	tel.recordRead(4, 16, nil, time.Millisecond)
	snap, _ = tel.Snapshot()
	if snap.ReadCount != before {
		t.Error("a closed telemetry recorded a read")
	}
}

func TestTelemetrySampleRateChange(t *testing.T) {
	rate := int32(48000)
	tel := newAudioUnderrunTelemetry(1, 48000, func() int { return int(atomic.LoadInt32(&rate)) })
	defer tel.CloseForTesting()

	if _, err := tel.Snapshot(); err != nil {
		t.Fatalf("Snapshot at the original rate: %v", err)
	}

	atomic.StoreInt32(&rate, 44100)
	if _, err := tel.Snapshot(); !errors.Is(err, ErrSampleRateMismatch) {
		t.Errorf("Snapshot after a sample rate change: got %v, want %v", err, ErrSampleRateMismatch)
	}
}

func TestTelemetryInvalidRead(t *testing.T) {
	tel, _ := newTestTelemetry(t, 48000)
	r := &audioUnderrunReadSeeker{r: &invalidCountReader{n: 100}, t: tel}
	if _, err := r.Read(make([]byte, 16)); !errors.Is(err, ErrInvalidRead) {
		t.Fatalf("Read: got %v, want %v", err, ErrInvalidRead)
	}
	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.InvalidReadCount, uint64(1); got != want {
		t.Errorf("InvalidReadCount: got %d, want %d", got, want)
	}
	if !errors.Is(snap.LastError, ErrInvalidRead) {
		t.Errorf("LastError: got %v, want %v", snap.LastError, ErrInvalidRead)
	}
}

func TestTelemetryInvalidReadWhenDisabled(t *testing.T) {
	tel := newAudioUnderrunTelemetry(1, 48000, nil)
	defer tel.CloseForTesting()
	r := &audioUnderrunReadSeeker{r: &invalidCountReader{n: -1}, t: tel}
	if _, err := r.Read(make([]byte, 16)); !errors.Is(err, ErrInvalidRead) {
		t.Fatalf("Read: got %v, want %v", err, ErrInvalidRead)
	}
	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.InvalidReadCount, uint64(1); got != want {
		t.Errorf("InvalidReadCount: got %d, want %d", got, want)
	}
}

// invalidCountReader reports a byte count that cannot be true for the destination buffer.
type invalidCountReader struct {
	n int
}

func (r *invalidCountReader) Read(buf []byte) (int, error) {
	return r.n, nil
}

func (r *invalidCountReader) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func TestTelemetryEventRingSaturates(t *testing.T) {
	tel, _ := newTestTelemetry(t, 48000)
	for range audioUnderrunEventHistory + 10 {
		tel.recordRead(0, 16, nil, 0)
	}
	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(snap.Events), audioUnderrunEventHistory; got != want {
		t.Fatalf("events: got %d, want %d", got, want)
	}
	// The events kept are the latest ones, in order.
	for i, e := range snap.Events {
		want := uint64(i + 11)
		if e.Sequence != want {
			t.Errorf("event %d: Sequence: got %d, want %d", i, e.Sequence, want)
		}
	}
	if got, want := snap.StallCount, uint64(audioUnderrunEventHistory+10); got != want {
		t.Errorf("StallCount: got %d, want %d", got, want)
	}
}

func TestTelemetryErrorRingSaturates(t *testing.T) {
	tel, _ := newTestTelemetry(t, 48000)
	for i := range audioUnderrunErrorHistory + 3 {
		tel.recordRead(0, 16, fmt.Errorf("err %d", i), 0)
	}
	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(snap.Errors), audioUnderrunErrorHistory; got != want {
		t.Fatalf("errors: got %d, want %d", got, want)
	}
	if got, want := snap.Errors[len(snap.Errors)-1].Err.Error(), fmt.Sprintf("err %d", audioUnderrunErrorHistory+2); got != want {
		t.Errorf("last error: got %q, want %q", got, want)
	}
}

// TestTelemetryReadDoesNotBlockOnSnapshot checks that a read while a snapshot holds the telemetry
// is dropped and counted rather than waiting.
func TestTelemetryReadDoesNotBlockOnSnapshot(t *testing.T) {
	tel, _ := newTestTelemetry(t, 48000)

	// Hold the telemetry as a snapshot does, then record from another goroutine.
	tel.mu.Lock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		tel.recordRead(0, 16, nil, 0)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		tel.mu.Unlock()
		t.Fatal("recordRead blocked while the telemetry was locked")
	}
	tel.mu.Unlock()

	snap, err := tel.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := snap.DroppedEventCount, uint64(1); got != want {
		t.Errorf("DroppedEventCount: got %d, want %d", got, want)
	}
	if got, want := snap.StallCount, uint64(1); got != want {
		t.Errorf("a dropped event must still be counted: got %d, want %d", got, want)
	}
}

// TestTelemetryConcurrentReadsAndSnapshots runs the read and snapshot paths concurrently under
// the race detector.
func TestTelemetryConcurrentReadsAndSnapshots(t *testing.T) {
	tel, _ := newTestTelemetry(t, 48000)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			src := &readSeekerSource{}
			r := &audioUnderrunReadSeeker{r: src, t: tel}
			buf := make([]byte, 16)
			for {
				select {
				case <-stop:
					return
				default:
				}
				src.reads = []readSeekerRead{{data: make([]byte, 16)}}
				_, _ = r.Read(buf)
				tel.observeTick(true, 0, 0, 1)
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := tel.Snapshot(); err != nil {
				t.Errorf("Snapshot: %v", err)
				return
			}
		}
	}()

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestTelemetryDisabledDoesNotAllocate checks the zero-overhead default path: a read through a
// disabled telemetry allocates nothing, so it cannot perturb playback.
func TestTelemetryDisabledDoesNotAllocate(t *testing.T) {
	tel := newAudioUnderrunTelemetry(1, 48000, nil)
	defer tel.CloseForTesting()
	src := &staticReadSeeker{}
	r := &audioUnderrunReadSeeker{r: src, t: tel}
	buf := make([]byte, 16)
	// Warm up so that any lazy initialization is done.
	_, _ = r.Read(buf)

	allocs := testing.AllocsPerRun(100, func() {
		_, _ = r.Read(buf)
	})
	if allocs > 0 {
		t.Errorf("a disabled telemetry read allocated %v times; want 0", allocs)
	}
}

// staticReadSeeker returns a fixed amount of data without allocating.
type staticReadSeeker struct {
	pos int64
}

func (s *staticReadSeeker) Read(buf []byte) (int, error) {
	n := min(16, len(buf))
	for i := range n {
		buf[i] = 0
	}
	return n, nil
}

func (s *staticReadSeeker) Seek(offset int64, whence int) (int64, error) {
	return s.pos, nil
}

func TestFindLoopCounter(t *testing.T) {
	src := bytes.NewReader(make([]byte, 16))
	loop := NewInfiniteLoopF32(src, 16)
	if got := findLoopCounter(loop); got != loop {
		t.Fatalf("findLoopCounter(loop): got %v, want the loop", got)
	}

	// A loop nested in another loop reports the outermost one.
	outer := NewInfiniteLoopF32(loop, 16)
	if got := findLoopCounter(outer); got != outer {
		t.Errorf("findLoopCounter(outer): got %v, want the outer loop", got)
	}

	if got := findLoopCounter(src); got != nil {
		t.Errorf("findLoopCounter(plain source): got %v, want nil", got)
	}

	// A time stream hides its source; the counter is still found through it.
	ts, err := newTimeStream(loop, true, 48000, bitDepthInBytesFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if got := findLoopCounter(ts); got != loop {
		t.Errorf("findLoopCounter(timeStream): got %v, want the loop", got)
	}

	// A 16bit source goes through the float32 converter; the counter is still found.
	conv := convert.NewFloat32BytesReaderFromInt16BytesReader(loop)
	if got := findLoopCounter(conv); got != loop {
		t.Errorf("findLoopCounter(float32 converter): got %v, want the loop", got)
	}
}

func TestInfiniteLoopCountsWraps(t *testing.T) {
	src := make([]byte, 32)
	l := NewInfiniteLoopF32(bytes.NewReader(src), int64(len(src)))
	if got, want := l.takeLoops(), uint64(0); got != want {
		t.Fatalf("takeLoops before reading: got %d, want %d", got, want)
	}
	// Read more than the loop length: the stream wraps at least once.
	buf := make([]byte, 64)
	if _, err := io.ReadFull(l, buf); err != nil {
		t.Fatal(err)
	}
	if got := l.takeLoops(); got == 0 {
		t.Errorf("takeLoops after a wrap: got 0, want at least 1")
	}
	if got, want := l.takeLoops(), uint64(0); got != want {
		t.Errorf("takeLoops after drain: got %d, want %d", got, want)
	}
}
