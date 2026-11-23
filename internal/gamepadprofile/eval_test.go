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

package gamepadprofile_test

import (
	"math"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

const evalGUID = "guid-eval"

func evalProfile(t *testing.T, opts ...func(*profileBuilder)) *gamepadprofile.GamepadMappingProfile {
	t.Helper()
	b := &profileBuilder{
		guid:    evalGUID,
		actions: []string{"jump", "fire", "left", "right", "up", "screenshot", "throttle"},
	}
	for _, opt := range opts {
		opt(b)
	}
	p, err := gamepadprofile.NewGamepadMappingProfile(b.guid, "", b.actions, b.bindings, b.chords, b.axes)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

type profileBuilder struct {
	guid     string
	actions  []string
	bindings []gamepadprofile.GamepadBinding
	chords   []gamepadprofile.GamepadChordBinding
	axes     []gamepadprofile.GamepadAxisConfig
}

func install(t *testing.T, p *gamepadprofile.GamepadMappingProfile) {
	t.Helper()
	resetRegistry(t)
	if err := gamepadprofile.SetProfile(p); err != nil {
		t.Fatal(err)
	}
}

func approxEqual(got, want, eps float64) bool {
	return math.Abs(got-want) <= eps
}

func TestButtonAndHatBindings(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.bindings = []gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
			gamepadprofile.NewGamepadHatBinding(0, 1, "up"),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, nil, []bool{false}, []int{0})
	if gamepadprofile.IsActionPressed(g, "jump") {
		t.Error("jump is pressed before any input; want false")
	}

	g.buttons[0] = true
	if !gamepadprofile.IsActionPressed(g, "jump") {
		t.Error("jump is not pressed while button 0 is held; want true")
	}
	if got, want := gamepadprofile.ActionValue(g, "jump"), 1.0; got != want {
		t.Errorf("ActionValue(jump) = %v; want %v", got, want)
	}

	g.hats[0] = 1 // hat up
	if !gamepadprofile.IsActionPressed(g, "up") {
		t.Error("up is not pressed while the hat points up; want true")
	}
	g.hats[0] = 2 // hat right
	if gamepadprofile.IsActionPressed(g, "up") {
		t.Error("up stays pressed after the hat turned right; want false")
	}
}

func TestChordBinding(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.chords = []gamepadprofile.GamepadChordBinding{
			gamepadprofile.NewGamepadChordBinding([]int{0, 1}, "screenshot"),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, nil, []bool{false, false, false}, nil)
	if gamepadprofile.IsActionPressed(g, "screenshot") {
		t.Error("the chord fires with no buttons held; want false")
	}

	g.buttons[0] = true
	if gamepadprofile.IsActionPressed(g, "screenshot") {
		t.Error("the chord fires with only one of its two buttons held; want false")
	}

	g.buttons[1] = true
	if !gamepadprofile.IsActionPressed(g, "screenshot") {
		t.Error("the chord does not fire while both buttons are held; want true")
	}
	if got, want := gamepadprofile.ActionValue(g, "screenshot"), 1.0; got != want {
		t.Errorf("ActionValue(screenshot) = %v; want %v", got, want)
	}

	g.buttons[0] = false
	if gamepadprofile.IsActionPressed(g, "screenshot") {
		t.Error("the chord stays pressed after one button was released; want false")
	}
}

func TestChordOnDeviceMissingButton(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.chords = []gamepadprofile.GamepadChordBinding{
			gamepadprofile.NewGamepadChordBinding([]int{0, 5}, "screenshot"),
		}
	})
	install(t, p)

	// The device only has two buttons, so it can never satisfy the chord.
	g := newFakeGamepad(evalGUID, nil, []bool{true, true}, nil)
	if gamepadprofile.IsActionPressed(g, "screenshot") {
		t.Error("a chord referencing a missing button fired; want false")
	}
}

func TestHalfAxisBindingsAndDeadZone(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.bindings = []gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadHalfAxisBinding(0, true, "left"),
			gamepadprofile.NewGamepadHalfAxisBinding(0, false, "right"),
		}
		b.axes = []gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.2, false),
		}
	})
	install(t, p)

	tests := []struct {
		name       string
		raw        float64
		wantLeft   bool
		wantRight  bool
		wantRightV float64
	}{
		{name: "centered inside dead zone", raw: 0.1, wantLeft: false, wantRight: false, wantRightV: 0},
		{name: "negative inside dead zone", raw: -0.19, wantLeft: false, wantRight: false, wantRightV: 0},
		{name: "full right", raw: 1, wantLeft: false, wantRight: true, wantRightV: 1},
		{name: "full left", raw: -1, wantLeft: true, wantRight: false, wantRightV: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := newFakeGamepad(evalGUID, []float64{test.raw}, nil, nil)
			if got := gamepadprofile.IsActionPressed(g, "left"); got != test.wantLeft {
				t.Errorf("IsActionPressed(left) = %v; want %v", got, test.wantLeft)
			}
			if got := gamepadprofile.IsActionPressed(g, "right"); got != test.wantRight {
				t.Errorf("IsActionPressed(right) = %v; want %v", got, test.wantRight)
			}
			if got := gamepadprofile.ActionValue(g, "right"); !approxEqual(got, test.wantRightV, 1e-9) {
				t.Errorf("ActionValue(right) = %v; want %v", got, test.wantRightV)
			}
		})
	}

	// Just past the dead zone, the value must start from 0 (rescaled), not jump to 0.2.
	g := newFakeGamepad(evalGUID, []float64{0.2 + 0.8*0.5}, nil, nil) // rescaled value 0.5
	if got, want := gamepadprofile.ActionValue(g, "right"), 0.5; !approxEqual(got, want, 1e-9) {
		t.Errorf("ActionValue(right) halfway = %v; want %v (dead zone rescaled)", got, want)
	}
}

func TestWholeAxisBinding(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.bindings = []gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadAxisBinding(0, "throttle"),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, []float64{0.75}, nil, nil)
	if !gamepadprofile.IsActionPressed(g, "throttle") {
		t.Error("throttle is not pressed on a positive axis value; want true")
	}
	if got, want := gamepadprofile.ActionValue(g, "throttle"), 0.75; !approxEqual(got, want, 1e-9) {
		t.Errorf("ActionValue(throttle) = %v; want %v", got, want)
	}

	// The negative half does not drive a positive action.
	g = newFakeGamepad(evalGUID, []float64{-0.9}, nil, nil)
	if gamepadprofile.IsActionPressed(g, "throttle") {
		t.Error("throttle fires on a negative axis value; want false")
	}
	if got := gamepadprofile.ActionValue(g, "throttle"); got != 0 {
		t.Errorf("ActionValue(throttle) on negative input = %v; want 0", got)
	}
}

func TestLogicalAxisDeadZoneAndInversion(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.axes = []gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.2, false),
			gamepadprofile.NewGamepadAxisConfig(1, gamepadprofile.LogicalAxisLeftStickVertical, 0.2, true),
			gamepadprofile.NewGamepadAxisConfig(2, gamepadprofile.LogicalAxisLeftTrigger, 0.1, false),
		}
	})
	install(t, p)

	// Left stick X: inside the dead zone reads 0.
	g := newFakeGamepad(evalGUID, []float64{0.1, 0, 0}, nil, nil)
	if got := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickHorizontal); got != 0 {
		t.Errorf("left X inside the dead zone = %v; want 0", got)
	}

	// Left stick X at full travel reads 1.
	g = newFakeGamepad(evalGUID, []float64{1, 0, 0}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickHorizontal), 1.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("left X full travel = %v; want %v", got, want)
	}

	// Inverted left stick Y flips the sign.
	g = newFakeGamepad(evalGUID, []float64{0, 0.5, 0}, nil, nil)
	gotY := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickVertical)
	// 0.5 is past the 0.2 dead zone; processed = applyDeadZone(0.5, 0.2) = 0.375, then inverted.
	if want := -0.375; !approxEqual(gotY, want, 1e-9) {
		t.Errorf("inverted left Y = %v; want %v", gotY, want)
	}

	// A trigger maps the -1..1 stick range onto 0..1: raw -1 means released.
	g = newFakeGamepad(evalGUID, []float64{0, 0, -1}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftTrigger), 0.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("trigger released = %v; want %v", got, want)
	}
	g = newFakeGamepad(evalGUID, []float64{0, 0, 1}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftTrigger), 1.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("trigger fully held = %v; want %v", got, want)
	}
}

func TestAxisRangeNormalization(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		// A device whose axis only reports 0.2..0.8 must still span the full -1..1 logical range.
		b.axes = []gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfigWithRange(0, gamepadprofile.LogicalAxisRightStickHorizontal, 0.2, 0.8, 0, false),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, []float64{0.2}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisRightStickHorizontal), -1.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("range minimum = %v; want %v", got, want)
	}
	g = newFakeGamepad(evalGUID, []float64{0.5}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisRightStickHorizontal), 0.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("range midpoint = %v; want %v", got, want)
	}
	g = newFakeGamepad(evalGUID, []float64{0.8}, nil, nil)
	if got, want := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisRightStickHorizontal), 1.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("range maximum = %v; want %v", got, want)
	}
}

func TestBothSticksDeadZones(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.axes = []gamepadprofile.GamepadAxisConfig{
			// Left stick: 0.2 dead zone, not inverted.
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.2, false),
			gamepadprofile.NewGamepadAxisConfig(1, gamepadprofile.LogicalAxisLeftStickVertical, 0.2, false),
			// Right stick: 0.4 dead zone, both axes inverted.
			gamepadprofile.NewGamepadAxisConfig(2, gamepadprofile.LogicalAxisRightStickHorizontal, 0.4, true),
			gamepadprofile.NewGamepadAxisConfig(3, gamepadprofile.LogicalAxisRightStickVertical, 0.4, true),
		}
	})
	install(t, p)

	tests := []struct {
		name string
		axis gamepadprofile.LogicalAxis
		raw  float64
		want float64
	}{
		// Left stick: 0.1 is inside the 0.2 dead zone; 0.6 rescales to 0.5.
		{name: "left X inside dead zone", axis: gamepadprofile.LogicalAxisLeftStickHorizontal, raw: 0.1, want: 0},
		{name: "left X past dead zone", axis: gamepadprofile.LogicalAxisLeftStickHorizontal, raw: 0.6, want: 0.5},
		// Right stick: 0.3 is inside its larger 0.4 dead zone, even though it would pass the left one.
		{name: "right X inside its larger dead zone", axis: gamepadprofile.LogicalAxisRightStickHorizontal, raw: 0.3, want: 0},
		// 0.7 is past 0.4: applyDeadZone gives 0.5, then inversion negates it.
		{name: "right X inverted past dead zone", axis: gamepadprofile.LogicalAxisRightStickHorizontal, raw: 0.7, want: -0.5},
		{name: "right Y inverted full travel", axis: gamepadprofile.LogicalAxisRightStickVertical, raw: 1, want: -1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			axes := []float64{0, 0, 0, 0}
			axes[int(test.axis)] = test.raw
			g := newFakeGamepad(evalGUID, axes, nil, nil)
			if got := gamepadprofile.LogicalAxisValue(g, test.axis); !approxEqual(got, test.want, 1e-9) {
				t.Errorf("LogicalAxisValue(%v) with raw %v = %v; want %v", test.axis, test.raw, got, test.want)
			}
		})
	}
}

func TestNotReadyAxisReadsZero(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.axes = []gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.1, false),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, []float64{0.9}, nil, nil)
	g.axesReady[0] = false
	if got := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickHorizontal); got != 0 {
		t.Errorf("an axis that is not ready = %v; want 0", got)
	}
}

func TestMultipleBindingsTakeMaximum(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		// jump is reachable from a button and from a half axis; the larger value wins.
		b.bindings = []gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
			gamepadprofile.NewGamepadHalfAxisBinding(0, false, "jump"),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, []float64{0.4}, []bool{false}, nil)
	if got, want := gamepadprofile.ActionValue(g, "jump"), 0.4; !approxEqual(got, want, 1e-9) {
		t.Errorf("ActionValue with only the axis held = %v; want %v", got, want)
	}
	g.buttons[0] = true
	if got, want := gamepadprofile.ActionValue(g, "jump"), 1.0; !approxEqual(got, want, 1e-9) {
		t.Errorf("ActionValue with the button also held = %v; want %v (maximum, not a sum)", got, want)
	}
}

func TestUnknownActionAndAvailability(t *testing.T) {
	p := evalProfile(t, func(b *profileBuilder) {
		b.bindings = []gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
		}
	})
	install(t, p)

	g := newFakeGamepad(evalGUID, nil, nil, nil)
	if gamepadprofile.IsActionAvailable(g, "ghost") {
		t.Error("IsActionAvailable() = true for an unknown action; want false")
	}
	if gamepadprofile.IsActionPressed(g, "ghost") {
		t.Error("IsActionPressed() = true for an unknown action; want false")
	}
	if !gamepadprofile.IsActionAvailable(g, "jump") {
		t.Error("IsActionAvailable(jump) = false; want true")
	}
}
