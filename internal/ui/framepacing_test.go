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

package ui_test

import (
	"math"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

// referenceScheduler is an independent implementation of the frame-pacing specification that the
// tests cross-check ui.FramePacingController against. It derives each frame's scheduled time from
// an anchor and a frame index instead of advancing a deadline incrementally.
type referenceScheduler struct {
	interval       float64
	optsJitter     float64
	policy         ui.FramePacingPolicy
	maxCatchUp     int
	maxConsecutive uint64
	maxPause       float64

	follow      bool
	refreshRate float64
	started     bool
	anchor      float64
	index       int64
	paused      bool
	pauseStart  float64
	consec      uint64
	cumErr      float64
	reanchors   uint64
}

type referenceDecision struct {
	present bool
	wait    float64
	late    bool
}

func referenceRate(rate, refreshRate float64) float64 {
	switch {
	case rate == 0:
		rate = refreshRate
		if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			rate = ui.DefaultFrameRate
		}
	case rate < 0 || math.IsNaN(rate) || math.IsInf(rate, 0):
		rate = ui.DefaultFrameRate
	}
	return min(max(rate, 1), 1000)
}

func newReferenceScheduler(opts ui.FramePacingOptions, refreshRate float64) *referenceScheduler {
	r := &referenceScheduler{
		optsJitter: opts.JitterWindow.Seconds(),
		policy:     opts.Policy,
		maxCatchUp: opts.MaxCatchUpFrames,
		maxPause:   opts.MaxPause.Seconds(),
		follow:     opts.TargetFrameRate == 0,
	}
	if r.maxCatchUp <= 0 {
		r.maxCatchUp = 3
	}
	r.maxConsecutive = uint64(opts.MaxConsecutiveOverBudget)
	if r.maxConsecutive <= 0 {
		r.maxConsecutive = 8
	}
	if r.maxPause <= 0 {
		r.maxPause = 5
	}
	if refreshRate > 0 && !math.IsNaN(refreshRate) && !math.IsInf(refreshRate, 0) {
		r.refreshRate = refreshRate
	}
	r.interval = 1 / referenceRate(opts.TargetFrameRate, r.refreshRate)
	return r
}

func (r *referenceScheduler) jitter() float64 {
	j := r.optsJitter
	if j <= 0 {
		j = r.interval / 4
	}
	return min(j, r.interval/2)
}

func (r *referenceScheduler) planFrame(t float64) referenceDecision {
	if r.paused {
		return referenceDecision{}
	}
	if !r.started {
		r.started = true
		r.anchor = t
		r.index = 0
	}
	jitter := r.jitter()
	deadline := r.anchor + float64(r.index)*r.interval
	d := referenceDecision{present: true}
	var err float64
	presentTime := t
	if delta := t - deadline; delta < -jitter {
		d.wait = min(-delta, r.interval)
		presentTime = t + d.wait
	} else {
		err = delta
	}
	r.cumErr += err
	if err > jitter {
		d.late = true
		r.consec++
	} else {
		r.consec = 0
	}
	debt := presentTime - deadline
	var reanchor bool
	switch {
	case r.consec > 0 && r.consec >= r.maxConsecutive:
		reanchor = true
	case r.policy == ui.FramePacingDrop && debt > jitter:
		reanchor = true
	case r.policy == ui.FramePacingCatchUp && debt > float64(r.maxCatchUp)*r.interval:
		reanchor = true
	}
	if reanchor {
		r.anchor = presentTime + r.interval
		r.index = 0
		r.consec = 0
		r.reanchors++
	} else {
		r.index++
	}
	return d
}

func (r *referenceScheduler) pause(t float64) {
	if r.paused {
		return
	}
	r.paused = true
	r.pauseStart = t
}

func (r *referenceScheduler) resume(t float64) {
	if !r.paused {
		return
	}
	r.paused = false
	if !r.started {
		return
	}
	if d := t - r.pauseStart; d > r.maxPause {
		r.anchor = t
		r.index = 0
		r.consec = 0
		r.reanchors++
	} else {
		r.anchor += d
	}
}

func (r *referenceScheduler) notifyRefreshRate(hz float64) {
	if hz <= 0 || math.IsNaN(hz) || math.IsInf(hz, 0) {
		hz = 0
	}
	if r.refreshRate == hz {
		return
	}
	r.refreshRate = hz
	if r.follow {
		r.interval = 1 / referenceRate(0, hz)
		if r.started {
			r.started = false
			r.consec = 0
			r.reanchors++
		}
	}
}

// steadyTimes returns n frame times at the given rate in Hz, starting at start.
func steadyTimes(rate float64, n int, start float64) []float64 {
	ts := make([]float64, n)
	for i := range ts {
		ts[i] = start + float64(i)/rate
	}
	return ts
}

// shiftTimes returns ts with every time from index from on shifted by d.
func shiftTimes(ts []float64, from int, d float64) []float64 {
	out := make([]float64, len(ts))
	copy(out, ts)
	for i := from; i < len(out); i++ {
		out[i] += d
	}
	return out
}

type pacingEvent struct {
	atFrame     int
	pause       bool
	resume      bool
	refreshRate float64
}

// runPacingScenario feeds the fixed frame-time sequence to both the controller and the reference
// scheduler, and checks that every per-frame decision, the frame order, and the cumulative error
// agree. It also checks the decision bounds that must always hold: no negative wait, and no wait
// longer than one frame interval.
func runPacingScenario(t *testing.T, opts ui.FramePacingOptions, refreshRate float64, times []float64, events []pacingEvent) {
	t.Helper()

	c := ui.NewFramePacingController(opts)
	if refreshRate > 0 {
		c.NotifyRefreshRate(refreshRate)
	}
	ref := newReferenceScheduler(opts, refreshRate)
	interval := c.FrameInterval()

	for i, now := range times {
		for _, e := range events {
			if e.atFrame != i {
				continue
			}
			switch {
			case e.pause:
				c.Pause(now)
				ref.pause(now)
			case e.resume:
				c.Resume(now)
				ref.resume(now)
			case e.refreshRate != 0:
				c.NotifyRefreshRate(e.refreshRate)
				ref.notifyRefreshRate(e.refreshRate)
				interval = c.FrameInterval()
			}
		}

		got := c.PlanFrame(now)
		want := ref.planFrame(now)

		if got.Wait < 0 {
			t.Errorf("frame %d: negative wait: %v", i, got.Wait)
		}
		if got.Wait > interval {
			t.Errorf("frame %d: wait %v longer than the frame interval %v", i, got.Wait, interval)
		}
		if got.Present != want.present {
			t.Errorf("frame %d: Present: got: %t, want: %t", i, got.Present, want.present)
		}
		if got.Late != want.late {
			t.Errorf("frame %d: Late: got: %t, want: %t", i, got.Late, want.late)
		}
		if diff := math.Abs(got.Wait.Seconds() - want.wait); diff > 1e-6 {
			t.Errorf("frame %d: Wait: got: %v, want: %v", i, got.Wait, time.Duration(want.wait*float64(time.Second)))
		}
	}

	stats := c.Stats()
	if got, want := stats.Frames, uint64(len(times)); got != want {
		t.Errorf("Frames: got: %d, want: %d", got, want)
	}
	if stats.Frames != stats.Presented+stats.Dropped {
		t.Errorf("Frames (%d) != Presented (%d) + Dropped (%d)", stats.Frames, stats.Presented, stats.Dropped)
	}
	if diff := math.Abs(stats.CumulativeError.Seconds() - ref.cumErr); diff > 1e-6 {
		t.Errorf("CumulativeError: got: %v, want: %v", stats.CumulativeError, time.Duration(ref.cumErr*float64(time.Second)))
	}
}

func TestFramePacingAgainstReferenceScheduler(t *testing.T) {
	const rate = 60.0

	// A 30 [ms] stall at frame 30, after which the frames keep arriving at 60 [Hz].
	stallTimes := shiftTimes(steadyTimes(rate, 120, 0), 30, 0.030)

	// Frames arriving every 1.35 frame intervals, so that the debt crosses the catch-up bound
	// strictly between frames instead of landing exactly on it.
	slowTimes := make([]float64, 120)
	for i := range slowTimes {
		slowTimes[i] = float64(i) * 1.35 / rate
	}

	// 60 frames at 60 [Hz], then a refresh-rate change to 144 [Hz] and frames at 144 [Hz].
	changeTimes := steadyTimes(rate, 120, 0)
	for i := 60; i < len(changeTimes); i++ {
		changeTimes[i] = 1.0 + float64(i-60)/144.0
	}

	// A 10 [s] pause at frame 40.
	longPauseTimes := shiftTimes(steadyTimes(rate, 120, 0), 41, 10)

	cases := []struct {
		name        string
		opts        ui.FramePacingOptions
		refreshRate float64
		times       []float64
		events      []pacingEvent
	}{
		{
			name:  "steady 60Hz",
			opts:  ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60},
			times: steadyTimes(rate, 120, 0),
		},
		{
			name: "jittered 60Hz",
			opts: ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60},
			times: func() []float64 {
				ts := steadyTimes(rate, 120, 0)
				for i := range ts {
					ts[i] += float64(i%5-2) * 1e-3
				}
				return ts
			}(),
		},
		{
			name:  "stall with drop policy",
			opts:  ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, Policy: ui.FramePacingDrop},
			times: stallTimes,
		},
		{
			name:  "stall with catch-up policy",
			opts:  ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, Policy: ui.FramePacingCatchUp},
			times: stallTimes,
		},
		{
			name:  "slow frames with drop policy",
			opts:  ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, Policy: ui.FramePacingDrop},
			times: slowTimes,
		},
		{
			name:  "slow frames with catch-up policy",
			opts:  ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, Policy: ui.FramePacingCatchUp},
			times: slowTimes,
		},
		{
			name:        "following a 144Hz display",
			opts:        ui.FramePacingOptions{Enabled: true},
			refreshRate: 144,
			times:       steadyTimes(144, 120, 0),
		},
		{
			name:        "refresh rate change",
			opts:        ui.FramePacingOptions{Enabled: true},
			refreshRate: 60,
			times:       changeTimes,
			events:      []pacingEvent{{atFrame: 60, refreshRate: 144}},
		},
		{
			name:   "short pause",
			opts:   ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60},
			times:  steadyTimes(rate, 120, 0),
			events: []pacingEvent{{atFrame: 40, pause: true}, {atFrame: 50, resume: true}},
		},
		{
			name:   "long pause",
			opts:   ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60},
			times:  longPauseTimes,
			events: []pacingEvent{{atFrame: 40, pause: true}, {atFrame: 41, resume: true}},
		},
		{
			name: "custom jitter and bounds",
			opts: ui.FramePacingOptions{
				Enabled:                  true,
				TargetFrameRate:          60,
				JitterWindow:             time.Millisecond,
				Policy:                   ui.FramePacingCatchUp,
				MaxCatchUpFrames:         5,
				MaxConsecutiveOverBudget: 3,
			},
			times: func() []float64 {
				ts := steadyTimes(rate, 120, 0)
				for i := range ts {
					ts[i] += 2e-3 * float64(i)
				}
				return ts
			}(),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			runPacingScenario(t, c.opts, c.refreshRate, c.times, c.events)
		})
	}
}

func TestFramePacingFrameOrder(t *testing.T) {
	interval := 1.0 / 60

	c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})

	cases := []struct {
		now     float64
		present bool
		wait    time.Duration
		late    bool
	}{
		// The first frame anchors the schedule and is presented immediately.
		{now: 0, present: true},
		// 6.7 [ms] early, beyond the jitter window of interval/4: wait for the deadline.
		{now: 0.010, present: true, wait: time.Duration((interval - 0.010) * float64(time.Second))},
		// On the deadline.
		{now: 2 * interval, present: true},
		// 10 [ms] late: presented immediately and counted late. The drop policy re-anchors.
		{now: 3*interval + 0.010, present: true, late: true},
		// On the re-anchored deadline.
		{now: 4*interval + 0.010, present: true},
	}
	for i, want := range cases {
		got := c.PlanFrame(want.now)
		if got.Present != want.present {
			t.Errorf("frame %d: Present: got: %t, want: %t", i, got.Present, want.present)
		}
		if got.Late != want.late {
			t.Errorf("frame %d: Late: got: %t, want: %t", i, got.Late, want.late)
		}
		if diff := got.Wait - want.wait; diff < -time.Microsecond || diff > time.Microsecond {
			t.Errorf("frame %d: Wait: got: %v, want: %v", i, got.Wait, want.wait)
		}
		if got.Wait < 0 {
			t.Errorf("frame %d: negative wait: %v", i, got.Wait)
		}
	}

	// Only the 10 [ms] late frame contributes to the cumulative error.
	if got, want := c.Stats().CumulativeError, 10*time.Millisecond; got < want-time.Microsecond || got > want+time.Microsecond {
		t.Errorf("CumulativeError: got: %v, want: %v", got, want)
	}
	stats := c.Stats()
	if got, want := stats.Late, uint64(1); got != want {
		t.Errorf("Late: got: %d, want: %d", got, want)
	}
	if got, want := stats.Reanchors, uint64(1); got != want {
		t.Errorf("Reanchors: got: %d, want: %d", got, want)
	}
}

func TestFramePacingPauseResume(t *testing.T) {
	const interval = 1.0 / 60

	t.Run("short pause keeps the rhythm", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, MaxPause: time.Second})
		c.PlanFrame(0)
		c.PlanFrame(interval)
		c.Pause(2 * interval)
		if !c.Paused() {
			t.Error("Paused: got: false, want: true")
		}
		// Pausing again does nothing.
		c.Pause(2 * interval)
		if got := c.PlanFrame(3 * interval); got.Present {
			t.Error("PlanFrame while paused: Present: got: true, want: false")
		}
		// Resume after 0.5 [s]: the schedule shifts by the pause duration.
		c.Resume(2*interval + 0.5)
		if c.Paused() {
			t.Error("Paused: got: true, want: false")
		}
		if got := c.PlanFrame(2*interval + 0.5); !got.Present || got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame after resume: got: %+v, want: {Present:true}", got)
		}
		stats := c.Stats()
		if got, want := stats.Pauses, uint64(1); got != want {
			t.Errorf("Pauses: got: %d, want: %d", got, want)
		}
		if got, want := stats.Dropped, uint64(1); got != want {
			t.Errorf("Dropped: got: %d, want: %d", got, want)
		}
		if got, want := stats.Reanchors, uint64(0); got != want {
			t.Errorf("Reanchors: got: %d, want: %d", got, want)
		}
	})

	t.Run("long pause restarts the schedule", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60, MaxPause: time.Second})
		c.PlanFrame(0)
		c.PlanFrame(interval)
		c.Pause(2 * interval)
		// Resume after 10 [s]: the schedule restarts at the resume time instead of bursting
		// catch-up frames or waiting far into the future.
		c.Resume(2*interval + 10)
		if got := c.PlanFrame(2*interval + 10); !got.Present || got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame after a long pause: got: %+v, want: {Present:true}", got)
		}
		if got := c.PlanFrame(2*interval + 10 + interval); !got.Present || got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame one interval after a long pause: got: %+v, want: {Present:true}", got)
		}
		if got, want := c.Stats().Reanchors, uint64(1); got != want {
			t.Errorf("Reanchors: got: %d, want: %d", got, want)
		}
	})

	t.Run("resume without pause does nothing", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})
		c.PlanFrame(0)
		c.Resume(interval)
		if got := c.PlanFrame(interval); !got.Present || got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame after a stray resume: got: %+v, want: {Present:true}", got)
		}
	})
}

func TestFramePacingFallbacks(t *testing.T) {
	t.Run("frame rate fallback", func(t *testing.T) {
		for _, rate := range []float64{0, -1, -60, math.NaN(), math.Inf(1), math.Inf(-1)} {
			c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: rate})
			if got, want := c.FrameInterval(), time.Second/ui.DefaultFrameRate; got != want {
				t.Errorf("FrameInterval for rate %v: got: %v, want: %v", rate, got, want)
			}
		}
	})

	t.Run("frame rate clamp", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 1e9})
		if got, want := c.FrameInterval(), time.Millisecond; got != want {
			t.Errorf("FrameInterval for rate 1e9: got: %v, want: %v", got, want)
		}
		c = ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 1e-9})
		if got, want := c.FrameInterval(), time.Second; got != want {
			t.Errorf("FrameInterval for rate 1e-9: got: %v, want: %v", got, want)
		}
	})

	t.Run("follow the refresh rate", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true})
		c.NotifyRefreshRate(120)
		if got, want := c.FrameInterval(), time.Second/120; got != want {
			t.Errorf("FrameInterval at 120 [Hz]: got: %v, want: %v", got, want)
		}
		// An unknown rate falls back to the default.
		c.NotifyRefreshRate(0)
		if got, want := c.FrameInterval(), time.Second/ui.DefaultFrameRate; got != want {
			t.Errorf("FrameInterval at an unknown rate: got: %v, want: %v", got, want)
		}
		c.NotifyRefreshRate(math.NaN())
		if got, want := c.FrameInterval(), time.Second/ui.DefaultFrameRate; got != want {
			t.Errorf("FrameInterval at a NaN rate: got: %v, want: %v", got, want)
		}
	})

	t.Run("non-finite time", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})
		c.PlanFrame(0)
		for _, now := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			if got := c.PlanFrame(now); !got.Present || got.Wait != 0 {
				t.Errorf("PlanFrame(%v): got: %+v, want: {Present:true}", now, got)
			}
		}
		stats := c.Stats()
		if got, want := stats.BadSamples, uint64(3); got != want {
			t.Errorf("BadSamples: got: %d, want: %d", got, want)
		}
		// The schedule is unchanged: the next frame is due one interval after the first.
		if got := c.PlanFrame(1.0 / 60); got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame after non-finite samples: got: %+v, want: {Present:true}", got)
		}
	})

	t.Run("clock going backwards", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})
		c.PlanFrame(1.0)
		// A sample before the previous one is clamped to it: no negative wait, no time travel.
		if got := c.PlanFrame(0.5); got.Wait < 0 {
			t.Errorf("PlanFrame with a backwards clock: negative wait: %v", got.Wait)
		}
		if got, want := c.Stats().BadSamples, uint64(1); got != want {
			t.Errorf("BadSamples: got: %d, want: %d", got, want)
		}
	})

	t.Run("consecutive over-budget frames", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{
			Enabled:                  true,
			TargetFrameRate:          60,
			Policy:                   ui.FramePacingCatchUp,
			MaxCatchUpFrames:         100,
			MaxConsecutiveOverBudget: 3,
		})
		const interval = 1.0 / 60
		// Every frame arrives two intervals after the previous one, so the debt grows every
		// frame. The catch-up policy's debt bound is set so high that only the consecutive
		// over-budget bound can re-anchor the schedule.
		for i := range 10 {
			got := c.PlanFrame(float64(i+1) * 2 * interval)
			if got.Wait < 0 {
				t.Fatalf("frame %d: negative wait: %v", i, got.Wait)
			}
			if got.Wait > time.Second/60 {
				t.Fatalf("frame %d: wait %v longer than the frame interval", i, got.Wait)
			}
		}
		stats := c.Stats()
		if stats.Reanchors == 0 {
			t.Error("Reanchors: got: 0, want: > 0")
		}
		if got, want := stats.Late, uint64(9); got != want {
			t.Errorf("Late: got: %d, want: %d", got, want)
		}
	})

	t.Run("device recreation re-anchors", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})
		c.PlanFrame(0)
		c.PlanFrame(1.0 / 60)
		c.NotifyDeviceRecreated()
		// The schedule restarts at the next frame instead of waiting for the old phase.
		if got := c.PlanFrame(1.0/60 + 1e-3); !got.Present || got.Wait != 0 || got.Late {
			t.Errorf("PlanFrame after a device recreation: got: %+v, want: {Present:true}", got)
		}
		if got, want := c.Stats().Reanchors, uint64(1); got != want {
			t.Errorf("Reanchors: got: %d, want: %d", got, want)
		}
	})

	t.Run("disabled controller is inert", func(t *testing.T) {
		c := ui.NewFramePacingController(ui.FramePacingOptions{})
		for i := range 10 {
			if got := c.PlanFrame(float64(i) / 1000); !got.Present || got.Wait != 0 || got.Late {
				t.Errorf("frame %d: got: %+v, want: {Present:true}", i, got)
			}
		}
		if got := c.Stats(); got.Frames != 0 {
			t.Errorf("Frames: got: %d, want: 0", got.Frames)
		}
	})
}

func TestFramePacingConcurrentReads(t *testing.T) {
	c := ui.NewFramePacingController(ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60})

	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				_ = c.Stats()
				_ = c.Options()
				_ = c.Enabled()
				_ = c.Paused()
				_ = c.FrameInterval()
			}
		}()
	}
	for i := range 100 {
		now := float64(i) / 60
		c.PlanFrame(now)
		if i == 50 {
			c.Pause(now)
			c.Resume(now + 0.1)
		}
	}
	wg.Wait()
}

func TestFramePacingFlushCommandsAndWait(t *testing.T) {
	var driver frameDriver
	now := 0.0
	ctx := ui.NewContextWithFramePacingForTest(
		ui.FramePacingOptions{Enabled: true, TargetFrameRate: 60},
		func() float64 { return now },
	)
	const interval = 1.0 / 60

	flush := func() {
		graphicscommand.NewImage(1, 1, false, "")
		if err := ctx.FlushCommandsAndWait(&driver, true); err != nil {
			t.Fatal(err)
		}
	}

	// The first frame anchors the schedule.
	flush()
	// Frames on the deadline are not delayed.
	now = interval
	flush()
	if got, want := ctx.FramePacing().Stats().Frames, uint64(2); got != want {
		t.Errorf("Frames: got: %d, want: %d", got, want)
	}
	if got := ctx.FramePacing().Stats().CumulativeError; got != 0 {
		t.Errorf("CumulativeError: got: %v, want: 0", got)
	}

	// A frame 5 [ms] early waits the rest of the interval in real time.
	now = 2*interval - 0.005
	start := time.Now()
	flush()
	if elapsed := time.Since(start); elapsed < 2*time.Millisecond {
		t.Errorf("an early frame did not wait: elapsed: %v, want: >= 2 [ms]", elapsed)
	}

	// A frame 20 [ms] late is counted late and re-anchors the schedule with the drop policy.
	now = 3*interval + 0.020
	flush()
	stats := ctx.FramePacing().Stats()
	if got, want := stats.Late, uint64(1); got != want {
		t.Errorf("Late: got: %d, want: %d", got, want)
	}
	if got, want := stats.Reanchors, uint64(1); got != want {
		t.Errorf("Reanchors: got: %d, want: %d", got, want)
	}
}
