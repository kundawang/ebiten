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
	"errors"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

func TestNewGamepadMappingProfileValid(t *testing.T) {
	p, err := gamepadprofile.NewGamepadMappingProfile(
		"guid-1",
		"Pad",
		[]string{"jump", "fire"},
		[]gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
			gamepadprofile.NewGamepadHalfAxisBinding(2, true, "fire"),
			gamepadprofile.NewGamepadHatBinding(0, 1, "jump"),
		},
		[]gamepadprofile.GamepadChordBinding{
			gamepadprofile.NewGamepadChordBinding([]int{0, 1}, "fire"),
		},
		[]gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.2, false),
		},
	)
	if err != nil {
		t.Fatalf("NewGamepadMappingProfile returned an error: %v", err)
	}
	if got, want := p.DeviceGUID(), "guid-1"; got != want {
		t.Errorf("DeviceGUID() = %q; want %q", got, want)
	}
	if got := p.AppendActions(nil); len(got) != 2 {
		t.Errorf("AppendActions() = %v; want two actions", got)
	}
	if got := p.AppendBindings(nil); len(got) != 3 {
		t.Errorf("AppendBindings() length = %d; want 3", len(got))
	}
	if got := p.AppendChordBindings(nil); len(got) != 1 {
		t.Errorf("AppendChordBindings() length = %d; want 1", len(got))
	}
	if got := p.AppendAxisConfigs(nil); len(got) != 1 {
		t.Errorf("AppendAxisConfigs() length = %d; want 1", len(got))
	}
}

// TestProfileValidationErrors checks that every rejected profile reports a field-context error
// naming the offending part.
func TestProfileValidationErrors(t *testing.T) {
	actions := []string{"jump"}
	validBinding := gamepadprofile.NewGamepadButtonBinding(0, "jump")

	tests := []struct {
		name      string
		guid      string
		actions   []string
		bindings  []gamepadprofile.GamepadBinding
		chords    []gamepadprofile.GamepadChordBinding
		axes      []gamepadprofile.GamepadAxisConfig
		wantPath  string
		wantInMsg string
	}{
		{
			name:      "empty guid",
			guid:      "",
			actions:   actions,
			bindings:  []gamepadprofile.GamepadBinding{validBinding},
			wantPath:  "device_guid",
			wantInMsg: "GUID",
		},
		{
			name:      "duplicate action",
			guid:      "g",
			actions:   []string{"jump", "jump"},
			bindings:  []gamepadprofile.GamepadBinding{validBinding},
			wantPath:  "actions[1]",
			wantInMsg: "duplicate action",
		},
		{
			name:      "unknown binding target",
			guid:      "g",
			actions:   actions,
			bindings:  []gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadButtonBinding(0, "nope")},
			wantPath:  "bindings[0].target",
			wantInMsg: "unknown action",
		},
		{
			name:    "duplicate binding source",
			guid:    "g",
			actions: []string{"a", "b"},
			bindings: []gamepadprofile.GamepadBinding{
				gamepadprofile.NewGamepadButtonBinding(0, "a"),
				gamepadprofile.NewGamepadButtonBinding(0, "b"),
			},
			wantPath:  "bindings[1]",
			wantInMsg: "duplicate binding",
		},
		{
			name:      "button index out of range",
			guid:      "g",
			actions:   actions,
			bindings:  []gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadButtonBinding(32, "jump")},
			wantPath:  "bindings[0].index",
			wantInMsg: "out of range",
		},
		{
			name:      "negative axis index",
			guid:      "g",
			actions:   actions,
			bindings:  []gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadAxisBinding(-1, "jump")},
			wantPath:  "bindings[0].index",
			wantInMsg: "out of range",
		},
		{
			name:      "invalid hat direction",
			guid:      "g",
			actions:   actions,
			bindings:  []gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadHatBinding(0, 0, "jump")},
			wantPath:  "bindings[0].hat_direction",
			wantInMsg: "invalid hat direction",
		},
		{
			name:      "empty chord",
			guid:      "g",
			actions:   actions,
			chords:    []gamepadprofile.GamepadChordBinding{gamepadprofile.NewGamepadChordBinding(nil, "jump")},
			wantPath:  "chord_bindings[0].buttons",
			wantInMsg: "at least one button",
		},
		{
			name:      "repeated chord button",
			guid:      "g",
			actions:   actions,
			chords:    []gamepadprofile.GamepadChordBinding{gamepadprofile.NewGamepadChordBinding([]int{1, 1}, "jump")},
			wantPath:  "chord_bindings[0].buttons[1]",
			wantInMsg: "duplicate button",
		},
		{
			name:      "dead zone too large",
			guid:      "g",
			actions:   actions,
			axes:      []gamepadprofile.GamepadAxisConfig{gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisNone, 1, false)},
			wantPath:  "axes[0].dead_zone",
			wantInMsg: "dead zone",
		},
		{
			name:      "illegal axis range",
			guid:      "g",
			actions:   actions,
			axes:      []gamepadprofile.GamepadAxisConfig{gamepadprofile.NewGamepadAxisConfigWithRange(0, gamepadprofile.LogicalAxisNone, 0.5, 0.5, 0.1, false)},
			wantPath:  "axes[0]",
			wantInMsg: "range min must be smaller",
		},
		{
			name:      "range endpoint out of bounds",
			guid:      "g",
			actions:   actions,
			axes:      []gamepadprofile.GamepadAxisConfig{gamepadprofile.NewGamepadAxisConfigWithRange(0, gamepadprofile.LogicalAxisNone, -2, 1, 0.1, false)},
			wantPath:  "axes[0]",
			wantInMsg: "within [-1, 1]",
		},
		{
			name:    "duplicate axis config",
			guid:    "g",
			actions: actions,
			axes: []gamepadprofile.GamepadAxisConfig{
				gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisNone, 0.1, false),
				gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisNone, 0.2, false),
			},
			wantPath:  "axes[1]",
			wantInMsg: "duplicate axis configuration",
		},
		{
			name:    "two raw axes feed one logical axis",
			guid:    "g",
			actions: actions,
			axes: []gamepadprofile.GamepadAxisConfig{
				gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.1, false),
				gamepadprofile.NewGamepadAxisConfig(1, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.1, false),
			},
			wantPath:  "axes[1].logical_axis",
			wantInMsg: "more than one raw axis",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := gamepadprofile.NewGamepadMappingProfile(test.guid, "", test.actions, test.bindings, test.chords, test.axes)
			if err == nil {
				t.Fatal("NewGamepadMappingProfile returned nil error; want a field error")
			}
			var fe *gamepadprofile.FieldError
			if !errors.As(err, &fe) {
				t.Fatalf("error %v is not a *GamepadFieldError", err)
			}
			if fe.Path != test.wantPath {
				t.Errorf("field path = %q; want %q", fe.Path, test.wantPath)
			}
			if !strings.Contains(err.Error(), test.wantInMsg) {
				t.Errorf("error message %q does not contain %q", err.Error(), test.wantInMsg)
			}
		})
	}
}

func TestProfileAccessors(t *testing.T) {
	binding := gamepadprofile.NewGamepadHatBinding(1, 2, "jump")
	if binding.Kind() != gamepadprofile.BindSourceHat {
		t.Errorf("Kind() = %v; want BindSourceHat", binding.Kind())
	}
	if binding.Index() != 1 || binding.HatDirection() != 2 || binding.Target() != "jump" {
		t.Errorf("hat binding accessors = (%d, %d, %q); want (1, 2, jump)", binding.Index(), binding.HatDirection(), binding.Target())
	}

	half := gamepadprofile.NewGamepadHalfAxisBinding(3, true, "fire")
	if !half.Negative() || half.Index() != 3 {
		t.Errorf("half-axis accessors = (index %d, negative %v); want (3, true)", half.Index(), half.Negative())
	}

	chord := gamepadprofile.NewGamepadChordBinding([]int{4, 5}, "fire")
	if got := chord.AppendButtons(nil); len(got) != 2 || got[0] != 4 || got[1] != 5 || chord.Target() != "fire" {
		t.Errorf("chord accessors = buttons %v target %q; want [4 5] fire", got, chord.Target())
	}
	// The constructor copies its input slice.
	src := []int{6}
	c2 := gamepadprofile.NewGamepadChordBinding(src, "fire")
	src[0] = 9
	if got := c2.AppendButtons(nil); got[0] != 6 {
		t.Errorf("chord aliases the constructor's input slice: %v; want [6]", got)
	}

	cfg := gamepadprofile.NewGamepadAxisConfigWithRange(2, gamepadprofile.LogicalAxisRightTrigger, -0.5, 0.5, 0.1, true)
	if cfg.Axis() != 2 || cfg.LogicalAxis() != gamepadprofile.LogicalAxisRightTrigger || cfg.DeadZone() != 0.1 || !cfg.Inverted() {
		t.Errorf("axis config accessors returned unexpected values: %+v", cfg)
	}
	if min, max := cfg.Range(); min != -0.5 || max != 0.5 {
		t.Errorf("Range() = (%v, %v); want (-0.5, 0.5)", min, max)
	}
}

func TestSetProfileRejectsNil(t *testing.T) {
	resetRegistry(t)
	if err := gamepadprofile.SetProfile(nil); err == nil {
		t.Error("SetProfile(nil) returned nil error; want an error")
	}
}

func TestFieldErrorUnwrap(t *testing.T) {
	inner := errors.New("inner")
	fe := &gamepadprofile.FieldError{Path: "x", Err: inner}
	if !errors.Is(fe, inner) {
		t.Error("FieldError does not unwrap to its inner error")
	}
	if fe.Error() != "x: inner" {
		t.Errorf("Error() = %q; want %q", fe.Error(), "x: inner")
	}
	empty := &gamepadprofile.FieldError{Err: inner}
	if empty.Error() != "inner" {
		t.Errorf("Error() with empty path = %q; want %q", empty.Error(), "inner")
	}
}

func TestProfileConstructorCopiesInputs(t *testing.T) {
	actions := []string{"jump"}
	bindings := []gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadButtonBinding(0, "jump")}
	chords := []gamepadprofile.GamepadChordBinding{gamepadprofile.NewGamepadChordBinding([]int{0, 1}, "jump")}
	axes := []gamepadprofile.GamepadAxisConfig{gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisNone, 0.1, false)}

	p, err := gamepadprofile.NewGamepadMappingProfile("g", "", actions, bindings, chords, axes)
	if err != nil {
		t.Fatal(err)
	}

	actions[0] = "mutated"
	bindings[0] = gamepadprofile.NewGamepadButtonBinding(9, "jump")
	chords[0] = gamepadprofile.NewGamepadChordBinding([]int{7}, "jump")

	if got := p.AppendActions(nil); got[0] != "jump" {
		t.Errorf("action mutated through the caller's slice: %v", got)
	}
	if got := p.AppendBindings(nil); got[0].Index() != 0 {
		t.Errorf("binding mutated through the caller's slice: %v", got[0])
	}
	if got := p.AppendChordBindings(nil); len(got[0].AppendButtons(nil)) != 2 {
		t.Errorf("chord mutated through the caller's slice: %v", got[0].AppendButtons(nil))
	}
}
