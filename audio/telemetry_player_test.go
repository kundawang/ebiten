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
	"errors"
	"io"
	"runtime"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// waitForEvent waits until cond holds or the deadline passes.
func waitForEvent(t *testing.T, cond func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		if err := audio.UpdateForTesting(); err != nil {
			t.Fatalf("UpdateForTesting: %v", err)
		}
		time.Sleep(time.Millisecond)
	}
	return false
}

// newTelemetryPlayer creates a player over src with collection enabled.
func newTelemetryPlayer(t *testing.T, src io.Reader) (*audio.Player, *audio.AudioUnderrunTelemetry) {
	t.Helper()
	p, err := context.NewPlayerF32(src)
	if err != nil {
		t.Fatal(err)
	}
	tel := p.AudioUnderrunTelemetry()
	if tel == nil {
		t.Fatal("AudioUnderrunTelemetry returned nil")
	}
	if tel.Enabled() {
		t.Fatal("the telemetry must be disabled by default")
	}
	if err := tel.SetEnabled(true); err != nil {
		t.Fatal(err)
	}
	return p, tel
}

// TestTelemetryPlayerDefaultDisabled checks that a player's telemetry starts disabled and that a
// disabled telemetry does not record anything while a player reads.
func TestTelemetryPlayerDefaultDisabled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		p, err := context.NewPlayerF32(bytes.NewReader(make([]byte, 44100*8)))
		if err != nil {
			t.Fatal(err)
		}
		defer runtime.KeepAlive(p)
		tel := p.AudioUnderrunTelemetry()
		if tel.Enabled() {
			t.Fatal("the telemetry must be disabled by default")
		}

		p.Play()
		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.Enabled {
			t.Error("snapshot reports enabled")
		}
		if snap.ReadCount != 0 || snap.UnderrunCount != 0 || len(snap.Events) != 0 {
			t.Errorf("a disabled telemetry recorded data: %+v", snap)
		}
		// The player still advances normally.
		if snap.FillLevelBytes != 0 && snap.BufferSizeBytes == 0 {
			t.Errorf("a disabled telemetry observed a fill level: %+v", snap)
		}
	})
}

// TestTelemetryLoopInt16Source checks that a looped 16bit source, which the player wraps with the
// float32 converter, still reports loops.
func TestTelemetryLoopInt16Source(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		// A tiny 16bit loop that wraps many times while the test plays it.
		loop := audio.NewInfiniteLoop(bytes.NewReader(make([]byte, 64)), 64)
		p, err := context.NewPlayer(loop)
		if err != nil {
			t.Fatal(err)
		}
		defer runtime.KeepAlive(p)
		tel := p.AudioUnderrunTelemetry()
		if err := tel.SetEnabled(true); err != nil {
			t.Fatal(err)
		}
		p.Play()

		if !waitForEvent(t, func() bool {
			snap, _ := tel.Snapshot()
			return snap.LoopCount > 0
		}) {
			snap, _ := tel.Snapshot()
			t.Fatalf("no loop was reported: %+v", snap)
		}
	})
}

// TestTelemetryDeviceSuspendResume checks that suspending and resuming the device is reported with
// distinguishable events.
func TestTelemetryDeviceSuspendResume(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		p, tel := newTelemetryPlayer(t, &infiniteReader{})
		defer runtime.KeepAlive(p)
		p.Play()
		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}

		if err := audio.SuspendForTesting(); err != nil {
			t.Fatal(err)
		}
		if err := audio.ResumeForTesting(); err != nil {
			t.Fatal(err)
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.DeviceSuspendCount != 1 {
			t.Errorf("DeviceSuspendCount: got %d, want 1", snap.DeviceSuspendCount)
		}
		if snap.DeviceResumeCount != 1 {
			t.Errorf("DeviceResumeCount: got %d, want 1", snap.DeviceResumeCount)
		}
		var sawSuspend, sawResume bool
		for _, e := range snap.Events {
			switch e.Kind {
			case audio.AudioUnderrunEventDeviceSuspend:
				sawSuspend = true
			case audio.AudioUnderrunEventDeviceResume:
				sawResume = true
			}
		}
		if !sawSuspend || !sawResume {
			t.Errorf("missing suspend/resume events in %v", snap.Events)
		}
	})
}

// TestTelemetrySlowReaderShortBuffer checks the event order for a slow source with a short
// buffer: short reads and stalls are reported before an underrun, and the underrun ends once
// data is buffered again.
func TestTelemetrySlowReaderShortBuffer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		// A source that returns 8 bytes at a time, which is far less than the device asks for.
		src := &smallChunkReader{chunk: 8, total: 44100 * 8}
		p, tel := newTelemetryPlayer(t, src)
		defer runtime.KeepAlive(p)
		// A buffer of one millisecond forces the device to ask often, so the slow source cannot
		// keep it filled.
		p.SetBufferSize(time.Millisecond)
		p.Play()

		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.ShortReadCount == 0 {
			t.Errorf("no short reads recorded: %+v", snap)
		}
		if snap.ReadCount == 0 {
			t.Errorf("no reads recorded: %+v", snap)
		}
		if snap.ReadLatency.Count == 0 {
			t.Errorf("no read latency recorded: %+v", snap)
		}

		// The short reads must appear as events, and they must precede any underrun event.
		var sawShort bool
		var unterrunIndex = -1
		for i, e := range snap.Events {
			if e.Kind == audio.AudioUnderrunEventShortRead || e.Kind == audio.AudioUnderrunEventStall {
				sawShort = true
			}
			if e.Kind == audio.AudioUnderrunEventUnderrunStart && unterrunIndex < 0 {
				unterrunIndex = i
			}
			if e.Sequence == 0 {
				t.Errorf("event %d has sequence 0", i)
			}
			if e.PlayerID != snap.PlayerID {
				t.Errorf("event %d: PlayerID = %d, want %d", i, e.PlayerID, snap.PlayerID)
			}
		}
		if !sawShort {
			t.Errorf("no short-read or stall event: %v", snap.Events)
		}
		if unterrunIndex >= 0 && !sawShort {
			t.Errorf("an underrun was reported without a short read or stall before it: %v", snap.Events)
		}
		// Sequences increase by exactly one across the retained events.
		for i := 1; i < len(snap.Events); i++ {
			if snap.Events[i].Sequence != snap.Events[i-1].Sequence+1 {
				t.Errorf("event sequences are not consecutive at %d: %d -> %d",
					i, snap.Events[i-1].Sequence, snap.Events[i].Sequence)
			}
		}
	})
}

// TestTelemetryNoUnderrunControl is the control fixture: a source that always fills the request
// produces no underrun and no short-read events.
func TestTelemetryNoUnderrunControl(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		// 4 seconds of audio, so the player does not run out during the test.
		p, tel := newTelemetryPlayer(t, &infiniteReader{})
		defer runtime.KeepAlive(p)
		p.Play()

		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}
		// Let several updates observe the fill.
		for range 5 {
			if err := audio.UpdateForTesting(); err != nil {
				t.Fatal(err)
			}
			time.Sleep(2 * time.Millisecond)
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.UnderrunCount != 0 || snap.InUnderrun {
			t.Errorf("an underrun was reported for a source that always fills the buffer: %+v", snap)
		}
		if snap.ShortReadCount != 0 || snap.StallCount != 0 {
			t.Errorf("a short read was reported for a full source: %+v", snap)
		}
		if snap.ReadCount == 0 {
			t.Errorf("no reads recorded: %+v", snap)
		}
	})
}

// TestTelemetryDeviceRebuild checks that a rebuilt device yields a rebuild event with both voice
// IDs and that the player keeps playing.
func TestTelemetryDeviceRebuild(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		p, tel := newTelemetryPlayer(t, &infiniteReader{})
		defer runtime.KeepAlive(p)
		p.Play()

		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}
		before, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if before.VoiceID == 0 {
			t.Fatal("the voice ID is 0 while playing")
		}

		audio.RebuildDriverForTesting()

		if !waitForEvent(t, func() bool {
			snap, _ := tel.Snapshot()
			return snap.DeviceRebuildCount > 0
		}) {
			t.Fatal("no device rebuild was reported")
		}
		after, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if after.VoiceID == 0 || after.VoiceID == before.VoiceID {
			t.Errorf("the voice ID did not change: before %d, after %d", before.VoiceID, after.VoiceID)
		}

		var rebuild *audio.AudioUnderrunEvent
		for i := range after.Events {
			if after.Events[i].Kind == audio.AudioUnderrunEventDeviceRebuild {
				rebuild = &after.Events[i]
			}
		}
		if rebuild == nil {
			t.Fatalf("no rebuild event in %v", after.Events)
		}
		if rebuild.PreviousVoiceID != before.VoiceID {
			t.Errorf("PreviousVoiceID: got %d, want %d", rebuild.PreviousVoiceID, before.VoiceID)
		}
		if rebuild.VoiceID != after.VoiceID {
			t.Errorf("VoiceID: got %d, want %d", rebuild.VoiceID, after.VoiceID)
		}

		// The player keeps playing after the rebuild.
		pos := p.Position()
		if !waitForEvent(t, func() bool { return p.Position() > pos }) {
			t.Errorf("Position did not advance after the device rebuild: %v", pos)
		}
	})
}

// TestTelemetryPauseResumeSeekLoop checks the distinguishable events for pause, resume, seek,
// and loop.
func TestTelemetryPauseResumeSeekLoop(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		// A tiny loop that wraps many times while the test plays it.
		loop := audio.NewInfiniteLoopF32(bytes.NewReader(make([]byte, 64)), 64)
		p, tel := newTelemetryPlayer(t, loop)
		defer runtime.KeepAlive(p)
		p.Play()

		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}
		if !waitForEvent(t, func() bool {
			snap, _ := tel.Snapshot()
			return snap.LoopCount > 0
		}) {
			t.Fatal("no loop was reported")
		}

		p.Pause()
		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}
		p.Play()
		if !waitForEvent(t, func() bool {
			snap, _ := tel.Snapshot()
			return snap.ResumeCount > 0
		}) {
			t.Fatal("no resume was reported")
		}

		p.PauseAndStopReading()
		if err := p.SetPosition(0); err != nil {
			t.Fatal(err)
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.PauseCount < 2 {
			t.Errorf("PauseCount: got %d, want at least 2", snap.PauseCount)
		}
		if snap.SeekCount != 1 {
			t.Errorf("SeekCount: got %d, want 1", snap.SeekCount)
		}

		want := map[audio.AudioUnderrunEventKind]bool{
			audio.AudioUnderrunEventLoop:                false,
			audio.AudioUnderrunEventPause:               false,
			audio.AudioUnderrunEventPauseAndStopReading: false,
			audio.AudioUnderrunEventResume:              false,
			audio.AudioUnderrunEventSeek:                false,
		}
		for _, e := range snap.Events {
			if _, ok := want[e.Kind]; ok {
				want[e.Kind] = true
			}
		}
		for kind, seen := range want {
			if !seen {
				t.Errorf("no %v event in %v", kind, snap.Events)
			}
		}
	})
}

// TestTelemetryEOFFixture checks that reaching the end of the source is reported as an EOF event
// and not as an error.
func TestTelemetryEOFFixture(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		// A short source: 100ms of audio.
		src := bytes.NewReader(make([]byte, 44100*8/10))
		p, tel := newTelemetryPlayer(t, src)
		defer runtime.KeepAlive(p)
		p.Play()

		// Wait for the player to finish without taking snapshots, so that the EOF event cannot be
		// dropped by contention with a snapshot.
		deadline := time.Now().Add(5 * time.Second)
		for p.IsPlaying() && time.Now().Before(deadline) {
			if err := audio.UpdateForTesting(); err != nil {
				t.Fatal(err)
			}
			time.Sleep(time.Millisecond)
		}
		if p.IsPlaying() {
			t.Fatal("the player did not finish")
		}

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.EOFCount == 0 {
			t.Fatal("no EOF was reported")
		}
		if snap.ErrorCount != 0 {
			t.Errorf("EOF was counted as an error: %+v", snap)
		}
		if snap.LastError != nil {
			t.Errorf("LastError: got %v, want nil", snap.LastError)
		}
		var sawEOF bool
		for _, e := range snap.Events {
			if e.Kind == audio.AudioUnderrunEventEOF {
				sawEOF = true
				if e.BytesRead != 0 {
					t.Errorf("EOF event BytesRead: got %d, want 0", e.BytesRead)
				}
			}
		}
		if !sawEOF {
			// The ring keeps the latest events; a full ring can have rotated the EOF out only if
			// later events were recorded, which the short source does not produce.
			t.Errorf("no EOF event in %v", snap.Events)
		}
	})
}

// TestTelemetrySourceErrorFixture checks that an abnormal source error is reported with the
// error and stops the player.
func TestTelemetrySourceErrorFixture(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		errSource := errors.New("audio_test: broken source")
		src := &errorAfterReader{n: 44100 * 8, err: errSource}
		p, tel := newTelemetryPlayer(t, src)
		defer runtime.KeepAlive(p)
		p.Play()

		waitForEvent(t, func() bool {
			snap, _ := tel.Snapshot()
			return snap.ErrorCount > 0
		})

		snap, err := tel.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap.ErrorCount == 0 {
			t.Fatalf("no error was reported: %+v", snap)
		}
		if !errors.Is(snap.LastError, errSource) {
			t.Errorf("LastError: got %v, want %v", snap.LastError, errSource)
		}
		if len(snap.Errors) == 0 {
			t.Fatal("no errors were retained")
		}
		if !errors.Is(snap.Errors[len(snap.Errors)-1].Err, errSource) {
			t.Errorf("last retained error: got %v, want %v", snap.Errors[len(snap.Errors)-1].Err, errSource)
		}
	})
}

// TestTelemetryMultiplePlayers checks that events of different players are distinguishable.
func TestTelemetryMultiplePlayers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		p1, tel1 := newTelemetryPlayer(t, &smallChunkReader{chunk: 8, total: 44100 * 8})
		p2, tel2 := newTelemetryPlayer(t, &smallChunkReader{chunk: 8, total: 44100 * 8})
		defer runtime.KeepAlive(p1)
		defer runtime.KeepAlive(p2)
		p1.SetBufferSize(time.Millisecond)
		p2.SetBufferSize(time.Millisecond)
		p1.Play()
		p2.Play()

		if !waitForEvent(t, func() bool { return p1.Position() > 0 && p2.Position() > 0 }) {
			t.Fatal("the players did not start")
		}

		snap1, err := tel1.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		snap2, err := tel2.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if snap1.PlayerID == snap2.PlayerID {
			t.Errorf("both players have PlayerID %d", snap1.PlayerID)
		}
		if snap1.VoiceID == snap2.VoiceID {
			t.Errorf("both players have VoiceID %d", snap1.VoiceID)
		}
		for _, e := range snap1.Events {
			if e.PlayerID != snap1.PlayerID {
				t.Errorf("player 1 recorded an event for player %d", e.PlayerID)
			}
		}
		for _, e := range snap2.Events {
			if e.PlayerID != snap2.PlayerID {
				t.Errorf("player 2 recorded an event for player %d", e.PlayerID)
			}
		}
	})
}

// TestTelemetryClosedPlayer checks that querying a closed player's telemetry reports the close
// while keeping the final statistics.
func TestTelemetryClosedPlayer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}

		p, tel := newTelemetryPlayer(t, &infiniteReader{})
		p.Play()
		if !waitForEvent(t, func() bool { return p.Position() > 0 }) {
			t.Fatal("the player did not start")
		}
		reads := func() uint64 {
			snap, _ := tel.Snapshot()
			return snap.ReadCount
		}
		before := reads()
		if before == 0 {
			t.Fatal("no reads before closing")
		}

		p.PauseAndStopReading()
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}

		snap, err := tel.Snapshot()
		if !errors.Is(err, audio.ErrPlayerClosed) {
			t.Fatalf("Snapshot after Close: got %v, want %v", err, audio.ErrPlayerClosed)
		}
		if snap.ReadCount < before {
			t.Errorf("ReadCount shrank after Close: %d -> %d", before, snap.ReadCount)
		}
		if err := tel.Reset(); !errors.Is(err, audio.ErrPlayerClosed) {
			t.Errorf("Reset after Close: got %v, want %v", err, audio.ErrPlayerClosed)
		}
		if err := tel.SetEnabled(true); !errors.Is(err, audio.ErrPlayerClosed) {
			t.Errorf("SetEnabled after Close: got %v, want %v", err, audio.ErrPlayerClosed)
		}
	})
}

// TestTelemetryConcurrentSnapshotWithPlayback hammers Snapshot while a player reads, under the
// race detector. It is not a synctest test: a snapshot loop with no blocking would prevent
// synctest's fake time from advancing.
func TestTelemetryConcurrentSnapshotWithPlayback(t *testing.T) {
	setup()
	defer teardown()

	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	p, tel := newTelemetryPlayer(t, &smallChunkReader{chunk: 64, total: 44100 * 8 * 10})
	defer runtime.KeepAlive(p)
	p.Play()

	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Go(func() {
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
	})

	deadline := time.Now().Add(5 * time.Second)
	for p.Position() == 0 && time.Now().Before(deadline) {
		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	close(stop)
	wg.Wait()
	if p.Position() == 0 {
		t.Fatal("the player did not start")
	}
}

// smallChunkReader returns at most chunk bytes per read.
type smallChunkReader struct {
	chunk int
	total int
}

func (r *smallChunkReader) Read(buf []byte) (int, error) {
	if r.total <= 0 {
		return 0, io.EOF
	}
	n := min(r.chunk, len(buf), r.total)
	clear(buf[:n])
	r.total -= n
	return n, nil
}

// errorAfterReader returns n bytes and then fails.
type errorAfterReader struct {
	n   int
	err error
}

func (r *errorAfterReader) Read(buf []byte) (int, error) {
	if r.n <= 0 {
		return 0, r.err
	}
	n := min(len(buf), r.n)
	clear(buf[:n])
	r.n -= n
	return n, nil
}
