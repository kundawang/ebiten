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
	"slices"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepad"
	"github.com/hajimehoshi/ebiten/v2/internal/gamepaddb"
	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

const (
	guidPlayerOne = "0000000000000000000000000000b001"
	guidPlayerTwo = "0000000000000000000000000000b002"
)

// updateVirtual drives the virtual gamepad backend and fails on error.
func updateVirtual(t *testing.T, states []gamepad.VirtualGamepadState) {
	t.Helper()
	if err := gamepad.Update(0, states); err != nil {
		t.Fatal(err)
	}
}

// twoPlayerProfiles builds different profiles for two devices, each with per-stick dead zones.
func twoPlayerProfiles(t *testing.T) []*gamepadprofile.GamepadMappingProfile {
	t.Helper()
	p1, err := gamepadprofile.NewGamepadMappingProfile(
		guidPlayerOne,
		"Player One Pad",
		[]string{"jump", "fire"},
		[]gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
			gamepadprofile.NewGamepadHalfAxisBinding(0, false, "fire"),
		},
		nil,
		[]gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.25, false),
			gamepadprofile.NewGamepadAxisConfig(1, gamepadprofile.LogicalAxisLeftStickVertical, 0.25, false),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := gamepadprofile.NewGamepadMappingProfile(
		guidPlayerTwo,
		"Player Two Pad",
		[]string{"jump"},
		[]gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(2, "jump"),
		},
		nil,
		[]gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisRightStickHorizontal, 0.1, true),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return []*gamepadprofile.GamepadMappingProfile{p1, p2}
}

func TestVirtualMultiDeviceProfiles(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	for _, p := range twoPlayerProfiles(t) {
		if err := gamepadprofile.SetProfile(p); err != nil {
			t.Fatal(err)
		}
	}

	updateVirtual(t, []gamepad.VirtualGamepadState{
		{
			ID:      0,
			SDLID:   guidPlayerOne,
			Name:    "P1",
			Axes:    []float64{0.1, 0}, // inside the 0.25 dead zone
			Buttons: []bool{true},
		},
		{
			ID:      1,
			SDLID:   guidPlayerTwo,
			Name:    "P2",
			Axes:    []float64{0.5},
			Buttons: []bool{false, false, true}, // button 2 is P2's jump
		},
	})

	p1 := gamepad.Get(0)
	p2 := gamepad.Get(1)
	if p1 == nil || p2 == nil {
		t.Fatalf("Get: p1=%v p2=%v; want both connected", p1, p2)
	}

	// Player one: button 0 fires jump; the small axis deflection is inside the dead zone.
	if !gamepadprofile.IsActionPressed(p1, "jump") {
		t.Error("player one jump = false; want true (button 0 held)")
	}
	if got := gamepadprofile.LogicalAxisValue(p1, gamepadprofile.LogicalAxisLeftStickHorizontal); got != 0 {
		t.Errorf("player one left X = %v; want 0 inside the dead zone", got)
	}

	// Player two has no "fire" action; its jump is button 2.
	if !gamepadprofile.IsActionAvailable(p1, "fire") {
		t.Error("player one should have the fire action")
	}
	if gamepadprofile.IsActionAvailable(p2, "fire") {
		t.Error("player two unexpectedly has the fire action")
	}
	if !gamepadprofile.IsActionPressed(p2, "jump") {
		t.Error("player two jump = false; want true (button 2 held)")
	}

	// The two players must not share state: P1's held button 0 is not P2's jump (its button 2),
	// and a profile lookup is per device GUID.
	if gamepadprofile.IsActionPressed(p2, "jump") == false {
		t.Error("player two jump should read its own device (button 2)")
	}

	// P1's button 0 must not be readable through P2's profile.
	if p2.Button(0) {
		t.Error("the two virtual gamepads share raw button state")
	}
}

func TestVirtualHotplugProfileFallback(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	if err := gamepadprofile.SetProfile(twoPlayerProfiles(t)[0]); err != nil {
		t.Fatal(err)
	}

	// Connect the profiled device.
	updateVirtual(t, []gamepad.VirtualGamepadState{
		{ID: 0, SDLID: guidPlayerOne, Name: "P1", Buttons: []bool{true}},
	})
	p1 := gamepad.Get(0)
	if p1 == nil {
		t.Fatal("the profiled gamepad did not connect")
	}
	if !gamepadprofile.HasProfile(p1) {
		t.Error("HasProfile() = false for the profiled device; want true")
	}
	if !gamepadprofile.IsActionPressed(p1, "jump") {
		t.Error("jump = false while button 0 is held; want true")
	}

	// Unplug it: the ID now resolves to nil and every query falls back to the default.
	updateVirtual(t, []gamepad.VirtualGamepadState{})
	if gamepad.Get(0) != nil {
		t.Error("Get(0) is non-nil after disconnection; want nil")
	}
	if gamepadprofile.HasProfile(gamepad.Get(0)) {
		t.Error("HasProfile() = true for a vanished device; want false")
	}

	// Replug a different, unprofiled device at the same ID.
	updateVirtual(t, []gamepad.VirtualGamepadState{
		{ID: 0, SDLID: "0000000000000000000000000000b099", Name: "Generic", Buttons: []bool{true}},
	})
	generic := gamepad.Get(0)
	if generic == nil {
		t.Fatal("the replacement gamepad did not connect")
	}
	if gamepadprofile.HasProfile(generic) {
		t.Error("HasProfile() = true for an unprofiled device; want false")
	}
	if gamepadprofile.IsActionPressed(generic, "jump") {
		t.Error("jump = true on an unprofiled device; want the default false")
	}

	// Replug the original device: its profile applies again by GUID, not by slot.
	updateVirtual(t, []gamepad.VirtualGamepadState{
		{ID: 0, SDLID: guidPlayerOne, Name: "P1", Buttons: []bool{true}},
	})
	if !gamepadprofile.HasProfile(gamepad.Get(0)) {
		t.Error("the profile was not re-applied when the device returned")
	}
}

func TestVirtualStickDeadZones(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	profiles := twoPlayerProfiles(t)
	if err := gamepadprofile.SetProfile(profiles[0]); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		rawX        float64
		wantLogical float64
	}{
		{name: "centered", rawX: 0, wantLogical: 0},
		{name: "inside dead zone", rawX: 0.2, wantLogical: 0},
		{name: "edge of dead zone", rawX: 0.25, wantLogical: 0},
		{name: "halfway after dead zone", rawX: 0.625, wantLogical: 0.5},
		{name: "full deflection", rawX: 1, wantLogical: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			updateVirtual(t, []gamepad.VirtualGamepadState{
				{ID: 0, SDLID: guidPlayerOne, Name: "P1", Axes: []float64{test.rawX}},
			})
			g := gamepad.Get(0)
			got := gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickHorizontal)
			if math.Abs(got-test.wantLogical) > 1e-9 {
				t.Errorf("logical left X = %v; want %v", got, test.wantLogical)
			}
		})
	}
}

// TestVirtualProfileDoesNotAffectStandardLayout proves the new layer is purely additive: with a
// profile active, the standard-layout mapping, its availability and the raw queries behave
// exactly as before.
func TestVirtualProfileDoesNotAffectStandardLayout(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	if err := gamepadprofile.SetProfile(twoPlayerProfiles(t)[0]); err != nil {
		t.Fatal(err)
	}

	updateVirtual(t, []gamepad.VirtualGamepadState{
		{
			ID:      0,
			SDLID:   guidPlayerOne,
			Name:    "P1",
			Axes:    []float64{0.9},
			Buttons: []bool{true, false, true},
			StandardButtons: map[gamepaddb.StandardButton]gamepad.VirtualStandardGamepadButton{
				gamepaddb.StandardButtonRightBottom: {Pressed: true, Value: 1},
			},
			StandardAxes: map[gamepaddb.StandardAxis]float64{
				gamepaddb.StandardAxisLeftStickHorizontal: 0.42,
			},
		},
	})
	g := gamepad.Get(0)

	// The raw view is untouched.
	if !g.Button(0) || g.Button(1) || !g.Button(2) {
		t.Errorf("raw buttons changed by the profile: %v", []bool{g.Button(0), g.Button(1), g.Button(2)})
	}
	if got := g.Axis(0); got != 0.9 {
		t.Errorf("raw axis = %v; want 0.9 (the profile must not alter raw queries)", got)
	}

	// The standard layout is the host's forwarded view, independent of the profile.
	if !g.IsStandardLayoutAvailable() {
		t.Error("standard layout became unavailable with a profile active")
	}
	if !g.IsStandardButtonPressed(gamepaddb.StandardButtonRightBottom) {
		t.Error("standard button press changed with a profile active")
	}
	if got := g.StandardAxisValue(gamepaddb.StandardAxisLeftStickHorizontal); math.Abs(got-0.42) > 1e-9 {
		t.Errorf("standard axis = %v; want 0.42", got)
	}
}

func TestVirtualProfileRuntimeSwitch(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	// First profile: button 0 is jump.
	if err := gamepadprofile.SetProfile(twoPlayerProfiles(t)[0]); err != nil {
		t.Fatal(err)
	}
	updateVirtual(t, []gamepad.VirtualGamepadState{
		{ID: 0, SDLID: guidPlayerOne, Name: "P1", Buttons: []bool{true, true}},
	})
	g := gamepad.Get(0)
	if !gamepadprofile.IsActionPressed(g, "jump") {
		t.Error("jump = false under the first profile (button 0); want true")
	}

	// Switch at runtime to a profile where button 1 is jump and button 0 is unrelated.
	swapped, err := gamepadprofile.NewGamepadMappingProfile(
		guidPlayerOne,
		"Swapped",
		[]string{"jump"},
		[]gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadButtonBinding(1, "jump")},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := gamepadprofile.SetProfile(swapped); err != nil {
		t.Fatal(err)
	}
	// Both buttons are still held, so jump stays pressed; now delete the profile entirely and
	// confirm the default fallback.
	if !gamepadprofile.IsActionPressed(g, "jump") {
		t.Error("jump = false after the runtime switch to the button-1 profile; want true")
	}
	gamepadprofile.DeleteProfile(guidPlayerOne)
	if gamepadprofile.IsActionPressed(g, "jump") {
		t.Error("jump = true after the profile was deleted; want the default false")
	}
	if gamepadprofile.HasProfile(g) {
		t.Error("HasProfile() = true after deletion; want false")
	}
}

// TestVirtualTwoPlayersInputSequence replays a scripted input sequence on two devices and checks
// the action stream each player observes, mirroring a game's per-frame polling.
func TestVirtualTwoPlayersInputSequence(t *testing.T) {
	resetRegistry(t)
	t.Cleanup(func() { updateVirtual(t, []gamepad.VirtualGamepadState{}) })

	for _, p := range twoPlayerProfiles(t) {
		if err := gamepadprofile.SetProfile(p); err != nil {
			t.Fatal(err)
		}
	}

	type frame struct {
		p1Buttons []bool
		p2Buttons []bool
		p1Jump    bool
		p2Jump    bool
	}
	frames := []frame{
		{
			p1Buttons: []bool{false}, p2Buttons: []bool{false, false, false},
			p1Jump: false, p2Jump: false,
		},
		{
			p1Buttons: []bool{true}, p2Buttons: []bool{false, false, false},
			p1Jump: true, p2Jump: false,
		},
		{
			p1Buttons: []bool{true}, p2Buttons: []bool{false, false, true},
			p1Jump: true, p2Jump: true,
		},
		{
			p1Buttons: []bool{false}, p2Buttons: []bool{false, false, true},
			p1Jump: false, p2Jump: true,
		},
		{
			p1Buttons: []bool{false}, p2Buttons: []bool{false, false, false},
			p1Jump: false, p2Jump: false,
		},
	}

	for i, fr := range frames {
		updateVirtual(t, []gamepad.VirtualGamepadState{
			{ID: 0, SDLID: guidPlayerOne, Name: "P1", Buttons: fr.p1Buttons},
			{ID: 1, SDLID: guidPlayerTwo, Name: "P2", Buttons: fr.p2Buttons},
		})
		p1 := gamepad.Get(0)
		p2 := gamepad.Get(1)
		if got := gamepadprofile.IsActionPressed(p1, "jump"); got != fr.p1Jump {
			t.Errorf("frame %d: player one jump = %v; want %v", i, got, fr.p1Jump)
		}
		if got := gamepadprofile.IsActionPressed(p2, "jump"); got != fr.p2Jump {
			t.Errorf("frame %d: player two jump = %v; want %v", i, got, fr.p2Jump)
		}
	}

	if ids := gamepad.AppendGamepadIDs(nil); !slices.Equal(ids, []gamepad.ID{0, 1}) {
		t.Errorf("gamepad IDs = %v; want [0 1]", ids)
	}
}
