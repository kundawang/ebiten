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

package ui

import (
	"math"
	"sync"
	"time"
)

// DefaultFrameRate is the frame rate in Hz that a FramePacingController falls back to when no
// usable target frame rate is configured or reported.
const DefaultFrameRate = 60.0

const (
	// minFrameRate and maxFrameRate bound the effective frame rate in Hz, so that the frame
	// interval stays meaningful: the wait a decision asks for is always bounded, and the interval
	// never gets so short that applying the decisions would busy loop.
	minFrameRate = 1.0
	maxFrameRate = 1000.0

	// defaultMaxCatchUpFrames bounds, in frame intervals, the debt that a FramePacingCatchUp
	// schedule carries before it is re-anchored.
	defaultMaxCatchUpFrames = 3

	// defaultMaxConsecutiveOverBudget is how many frames in a row may miss their budget before
	// the schedule is re-anchored regardless of the policy.
	defaultMaxConsecutiveOverBudget = 8

	// defaultMaxPause is how long a pause can last before resuming restarts the schedule fresh
	// instead of shifting it by the pause duration.
	defaultMaxPause = 5 * time.Second
)

// FramePacingPolicy determines how the presentation schedule reacts when a frame misses its
// budget.
type FramePacingPolicy int

const (
	// FramePacingDrop re-anchors the schedule at a late frame, dropping the frame slots that
	// were missed. The following frames are paced from the late frame on.
	FramePacingDrop FramePacingPolicy = iota

	// FramePacingCatchUp keeps the schedule when a frame is late, so that the following frames
	// run back to back until the debt is repaid. The debt is bounded by
	// FramePacingOptions.MaxCatchUpFrames; beyond it the schedule is re-anchored as with
	// FramePacingDrop.
	FramePacingCatchUp
)

// FramePacingOptions configures a FramePacingController.
//
// The zero value disables the controller. A disabled controller makes no pacing decisions and
// collects no statistics, and the UI context paces frames as it always has.
type FramePacingOptions struct {
	// Enabled reports whether the controller paces frames.
	Enabled bool

	// TargetFrameRate is the target presentation rate in Hz. 0 follows the display refresh rate
	// reported with NotifyRefreshRate, falling back to DefaultFrameRate while the rate is
	// unknown. A negative or non-finite value falls back to DefaultFrameRate. The effective rate
	// is clamped so that the frame interval stays between 1 [ms] and 1 [s].
	TargetFrameRate float64

	// JitterWindow is how far a frame may deviate from its scheduled time before the controller
	// waits for an early frame or counts a frame as late. Non-positive values use a quarter of
	// the frame interval. The window is clamped to half the frame interval.
	JitterWindow time.Duration

	// Policy is the catch-up/drop policy. See FramePacingPolicy. Values other than FramePacingDrop
	// and FramePacingCatchUp are treated as FramePacingDrop.
	Policy FramePacingPolicy

	// MaxCatchUpFrames bounds the debt, in frame intervals, that FramePacingCatchUp carries
	// before the schedule is re-anchored. Non-positive values use a default.
	MaxCatchUpFrames int

	// MaxConsecutiveOverBudget is how many frames in a row may miss their budget before the
	// schedule is re-anchored regardless of the policy. Non-positive values use a default.
	MaxConsecutiveOverBudget int

	// MaxPause is how long a pause may last before resuming restarts the schedule fresh instead
	// of shifting it by the pause duration. Non-positive values use a default.
	MaxPause time.Duration
}

// normalize replaces invalid option values with their fallbacks.
func (o *FramePacingOptions) normalize() {
	if o.Policy != FramePacingDrop && o.Policy != FramePacingCatchUp {
		o.Policy = FramePacingDrop
	}
}

// FrameDecision is the controller's decision for one frame.
type FrameDecision struct {
	// Present reports whether the frame should be presented. It is false only while paused.
	Present bool

	// Wait is how long the caller should wait before presenting the frame. It is never negative
	// and never longer than one frame interval.
	Wait time.Duration

	// Late reports whether the frame missed its scheduled time by more than the jitter window.
	Late bool
}

// FramePacingStats is a snapshot of the controller's counters. Frames always equals Presented
// plus Dropped.
type FramePacingStats struct {
	// Frames is the number of frames the controller decided on.
	Frames uint64

	// Presented is the number of frames decided to be presented.
	Presented uint64

	// Dropped is the number of frames decided not to be presented, i.e. frames while paused.
	Dropped uint64

	// Late is the number of presented frames that missed their scheduled time by more than the
	// jitter window.
	Late uint64

	// Reanchors is the number of times the schedule was re-anchored: by the policy, by too many
	// consecutive over-budget frames, by a long pause, by a refresh-rate change, or by a device
	// recreation.
	Reanchors uint64

	// Pauses is the number of pauses.
	Pauses uint64

	// BadSamples is the number of rejected time samples: non-finite times and times before the
	// previous sample.
	BadSamples uint64

	// ConsecutiveOverBudget is the current run of late frames.
	ConsecutiveOverBudget uint64

	// CumulativeError is the signed sum of the presented frames' errors against the schedule.
	// It is positive when frames were presented late on average.
	CumulativeError time.Duration
}

// FramePacingController is the presentation clock of a UI context, separate from the update
// clock in internal/clock: the update clock decides how many ticks a frame runs, while the frame
// pacing controller decides when a frame is presented.
//
// The controller never blocks. PlanFrame returns a decision for each frame — whether to present
// it and how long to wait first — and applying the decision is up to the caller, so game updates
// are never blocked by the controller itself.
//
// All methods are safe for concurrent use: the game loop can plan frames while another goroutine
// reads the statistics or changes the options.
type FramePacingController struct {
	mu sync.Mutex

	opts FramePacingOptions

	// refreshRate is the last display refresh rate reported with NotifyRefreshRate, in Hz.
	// 0 means unknown.
	refreshRate float64

	// started reports whether the schedule has been anchored.
	started bool

	// deadline is the scheduled presentation time of the next frame, in seconds in the same time
	// base as PlanFrame's argument.
	deadline float64

	// lastNow is the time sample of the last frame, for rejecting non-monotonic samples.
	lastNow float64

	paused     bool
	pauseStart float64

	frames                uint64
	presented             uint64
	dropped               uint64
	late                  uint64
	reanchors             uint64
	pauses                uint64
	badSamples            uint64
	consecutiveOverBudget uint64
	cumulativeError       float64
}

// NewFramePacingController returns a new FramePacingController configured with opts.
func NewFramePacingController(opts FramePacingOptions) *FramePacingController {
	opts.normalize()
	return &FramePacingController{
		opts: opts,
	}
}

// Enabled reports whether the controller paces frames.
func (c *FramePacingController) Enabled() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.opts.Enabled
}

// Options returns the current options.
func (c *FramePacingController) Options() FramePacingOptions {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.opts
}

// SetOptions sets the options. Changing the options re-anchors the schedule at the next frame.
func (c *FramePacingController) SetOptions(opts FramePacingOptions) {
	opts.normalize()
	c.mu.Lock()
	defer c.mu.Unlock()
	c.opts = opts
	c.started = false
	c.consecutiveOverBudget = 0
}

// FrameInterval returns the current frame interval, including all fallbacks and clamps.
func (c *FramePacingController) FrameInterval() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return secondsToDuration(c.frameInterval())
}

// PlanFrame returns the decision for a frame at now, in seconds in a monotonic time base of the
// caller's choice. The controller never blocks: applying the decision, including waiting, is up
// to the caller.
//
// A non-finite now is rejected and reported in FramePacingStats.BadSamples; the decision is then
// to present without waiting, and the schedule is left unchanged. A now before the previous
// sample is clamped to it, so the schedule never moves backwards.
func (c *FramePacingController) PlanFrame(now float64) FrameDecision {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.opts.Enabled {
		return FrameDecision{Present: true}
	}
	now, ok := c.sanitizeNow(now)
	if !ok {
		// The safe fallback for an unusable time sample: present without waiting.
		return FrameDecision{Present: true}
	}
	if c.paused {
		c.frames++
		c.dropped++
		c.lastNow = now
		return FrameDecision{}
	}

	interval := c.frameInterval()
	jitter := c.jitterWindow(interval)
	if !c.started {
		c.started = true
		c.deadline = now
	}

	d := FrameDecision{Present: true}
	var err float64
	presentTime := now
	if delta := now - c.deadline; delta < -jitter {
		// The frame is early beyond the jitter window: wait until its scheduled time. The wait
		// is never negative and never longer than one frame interval.
		wait := min(-delta, interval)
		d.Wait = secondsToDuration(wait)
		presentTime = now + wait
	} else {
		err = delta
	}

	c.frames++
	c.presented++
	c.cumulativeError += err
	if err > jitter {
		d.Late = true
		c.late++
		c.consecutiveOverBudget++
	} else {
		c.consecutiveOverBudget = 0
	}

	// Advance the schedule. Re-anchor it when the debt grows beyond what the policy carries, or
	// when too many frames in a row miss their budget: catching up further would either spiral
	// into ever larger debt or starve the frames after it.
	debt := presentTime - c.deadline
	var reanchor bool
	switch {
	case c.consecutiveOverBudget > 0 && c.consecutiveOverBudget >= c.maxConsecutiveOverBudget():
		reanchor = true
	case c.opts.Policy == FramePacingDrop && debt > jitter:
		reanchor = true
	case c.opts.Policy == FramePacingCatchUp && debt > float64(c.maxCatchUpFrames())*interval:
		reanchor = true
	}
	if reanchor {
		c.deadline = presentTime + interval
		c.reanchors++
		c.consecutiveOverBudget = 0
	} else {
		c.deadline += interval
	}
	c.lastNow = now
	return d
}

// Pause pauses the presentation schedule at now, in the same time base as PlanFrame's argument.
// While paused, PlanFrame decides not to present. Pausing while already paused does nothing.
func (c *FramePacingController) Pause(now float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now, ok := c.sanitizeNow(now)
	if !ok {
		return
	}
	if c.paused {
		return
	}
	c.paused = true
	c.pauseStart = now
	c.pauses++
	c.lastNow = now
}

// Resume resumes the presentation schedule at now, in the same time base as PlanFrame's
// argument. A pause not longer than FramePacingOptions.MaxPause shifts the schedule by the pause
// duration, keeping the rhythm. A longer pause restarts the schedule fresh at now, so that
// resuming neither bursts catch-up frames nor schedules a wait far in the future. Resuming while
// not paused does nothing.
func (c *FramePacingController) Resume(now float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now, ok := c.sanitizeNow(now)
	if !ok {
		return
	}
	if !c.paused {
		return
	}
	c.paused = false
	if c.started {
		if pauseDuration := now - c.pauseStart; pauseDuration > c.maxPause() {
			c.deadline = now
			c.reanchors++
			c.consecutiveOverBudget = 0
		} else {
			c.deadline += pauseDuration
		}
	}
	c.lastNow = now
}

// Paused reports whether the presentation schedule is paused.
func (c *FramePacingController) Paused() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.paused
}

// NotifyRefreshRate reports the refresh rate of the display the game is presented on, in Hz,
// e.g. after the window moved to another display. A non-positive or non-finite rate is treated
// as unknown. When the controller follows the display refresh rate
// (FramePacingOptions.TargetFrameRate == 0), a changed rate re-anchors the schedule.
func (c *FramePacingController) NotifyRefreshRate(hz float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if hz <= 0 || math.IsNaN(hz) || math.IsInf(hz, 0) {
		hz = 0
	}
	if c.refreshRate == hz {
		return
	}
	c.refreshRate = hz
	if c.opts.Enabled && c.opts.TargetFrameRate == 0 {
		// The presentation phase measured against the old display no longer applies.
		c.reanchor()
	}
}

// NotifyDeviceRecreated reports that the graphics device was recreated, e.g. after the app was
// suspended. The presentation schedule measured against the old device no longer applies.
func (c *FramePacingController) NotifyDeviceRecreated() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reanchor()
}

// Stats returns a snapshot of the controller's counters.
func (c *FramePacingController) Stats() FramePacingStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return FramePacingStats{
		Frames:                c.frames,
		Presented:             c.presented,
		Dropped:               c.dropped,
		Late:                  c.late,
		Reanchors:             c.reanchors,
		Pauses:                c.pauses,
		BadSamples:            c.badSamples,
		ConsecutiveOverBudget: c.consecutiveOverBudget,
		CumulativeError:       secondsToDuration(c.cumulativeError),
	}
}

// sanitizeNow validates a time sample. ok is false for a non-finite sample, which the caller
// must reject. A sample before the previous one is clamped to it: the schedule never moves
// backwards. c.mu must be held.
func (c *FramePacingController) sanitizeNow(now float64) (float64, bool) {
	if math.IsNaN(now) || math.IsInf(now, 0) {
		c.badSamples++
		return 0, false
	}
	if c.started && now < c.lastNow {
		c.badSamples++
		return c.lastNow, true
	}
	return now, true
}

// frameInterval returns the frame interval in seconds. c.mu must be held.
func (c *FramePacingController) frameInterval() float64 {
	rate := c.opts.TargetFrameRate
	switch {
	case rate == 0:
		// Follow the display refresh rate, falling back to the default while it is unknown.
		rate = c.refreshRate
		if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			rate = DefaultFrameRate
		}
	case rate < 0 || math.IsNaN(rate) || math.IsInf(rate, 0):
		rate = DefaultFrameRate
	}
	return 1 / min(max(rate, minFrameRate), maxFrameRate)
}

// jitterWindow returns the jitter window in seconds for the given frame interval. c.mu must be
// held.
func (c *FramePacingController) jitterWindow(interval float64) float64 {
	j := c.opts.JitterWindow.Seconds()
	if j <= 0 {
		j = interval / 4
	}
	return min(j, interval/2)
}

// maxCatchUpFrames returns the catch-up bound in frames. c.mu must be held.
func (c *FramePacingController) maxCatchUpFrames() int {
	if c.opts.MaxCatchUpFrames <= 0 {
		return defaultMaxCatchUpFrames
	}
	return c.opts.MaxCatchUpFrames
}

// maxConsecutiveOverBudget returns the bound on the run of late frames. c.mu must be held.
func (c *FramePacingController) maxConsecutiveOverBudget() uint64 {
	if c.opts.MaxConsecutiveOverBudget <= 0 {
		return defaultMaxConsecutiveOverBudget
	}
	return uint64(c.opts.MaxConsecutiveOverBudget)
}

// maxPause returns the pause bound in seconds. c.mu must be held.
func (c *FramePacingController) maxPause() float64 {
	if c.opts.MaxPause <= 0 {
		return defaultMaxPause.Seconds()
	}
	return c.opts.MaxPause.Seconds()
}

// reanchor restarts the schedule at the next frame. c.mu must be held.
func (c *FramePacingController) reanchor() {
	if !c.started {
		return
	}
	c.started = false
	c.consecutiveOverBudget = 0
	c.reanchors++
}

// secondsToDuration converts seconds to a time.Duration, saturating instead of overflowing.
func secondsToDuration(seconds float64) time.Duration {
	const maxSeconds = float64(math.MaxInt64) / float64(time.Second)
	if seconds >= maxSeconds {
		return time.Duration(math.MaxInt64)
	}
	if seconds <= -maxSeconds {
		return time.Duration(math.MinInt64)
	}
	return time.Duration(seconds * float64(time.Second))
}
