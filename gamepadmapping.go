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

package ebiten

import (
	"github.com/hajimehoshi/ebiten/v2/internal/gamepad"
	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

// GamepadMappingProfile is a per-device, user-customizable mapping of a gamepad model.
//
// A profile is keyed by the device's SDL GUID (see [GamepadSDLID]) and maps the device's raw
// buttons, axes and hats onto named, game-defined actions, together with per-axis dead zones and
// inversion. Profiles live in a process-wide registry, can be switched at runtime and can be
// exported to and imported from a versioned JSON document.
//
// A profile does not change the standard-layout queries ([IsStandardGamepadButtonPressed] and
// related functions), vibration, hot-plugging or the raw gamepad queries.
type GamepadMappingProfile = gamepadprofile.GamepadMappingProfile

// GamepadBinding maps one physical gamepad input (a button, a whole or half axis, or a hat
// direction) to an action.
type GamepadBinding = gamepadprofile.GamepadBinding

// GamepadChordBinding maps a simultaneous press of several buttons to an action.
type GamepadChordBinding = gamepadprofile.GamepadChordBinding

// GamepadAxisConfig configures one raw axis: the logical stick or trigger axis it feeds, its dead
// zone, its input range and whether it is inverted.
type GamepadAxisConfig = gamepadprofile.GamepadAxisConfig

// GamepadBindSourceKind identifies the physical input a binding reads.
type GamepadBindSourceKind = gamepadprofile.BindSourceKind

// Gamepad binding source kinds.
const (
	GamepadBindSourceNone     GamepadBindSourceKind = gamepadprofile.BindSourceNone
	GamepadBindSourceButton   GamepadBindSourceKind = gamepadprofile.BindSourceButton
	GamepadBindSourceAxis     GamepadBindSourceKind = gamepadprofile.BindSourceAxis
	GamepadBindSourceHalfAxis GamepadBindSourceKind = gamepadprofile.BindSourceHalfAxis
	GamepadBindSourceHat      GamepadBindSourceKind = gamepadprofile.BindSourceHat
)

// GamepadLogicalAxis identifies a stick or trigger axis that a raw axis feeds.
type GamepadLogicalAxis = gamepadprofile.LogicalAxis

// Logical axes a raw axis can feed.
const (
	GamepadLogicalAxisNone                 GamepadLogicalAxis = gamepadprofile.LogicalAxisNone
	GamepadLogicalAxisLeftStickHorizontal  GamepadLogicalAxis = gamepadprofile.LogicalAxisLeftStickHorizontal
	GamepadLogicalAxisLeftStickVertical    GamepadLogicalAxis = gamepadprofile.LogicalAxisLeftStickVertical
	GamepadLogicalAxisRightStickHorizontal GamepadLogicalAxis = gamepadprofile.LogicalAxisRightStickHorizontal
	GamepadLogicalAxisRightStickVertical   GamepadLogicalAxis = gamepadprofile.LogicalAxisRightStickVertical
	GamepadLogicalAxisLeftTrigger          GamepadLogicalAxis = gamepadprofile.LogicalAxisLeftTrigger
	GamepadLogicalAxisRightTrigger         GamepadLogicalAxis = gamepadprofile.LogicalAxisRightTrigger
)

// GamepadFieldError carries the profile field or array element that caused an error. Its Path is
// a dotted field path such as "profiles[0].bindings[2].target"; errors returned by the profile
// API can be inspected with errors.As.
type GamepadFieldError = gamepadprofile.FieldError

// NewGamepadButtonBinding returns a binding of a raw button to the action target.
func NewGamepadButtonBinding(button int, target string) GamepadBinding {
	return gamepadprofile.NewGamepadButtonBinding(button, target)
}

// NewGamepadAxisBinding returns a binding of a whole raw axis to the action target.
func NewGamepadAxisBinding(axis int, target string) GamepadBinding {
	return gamepadprofile.NewGamepadAxisBinding(axis, target)
}

// NewGamepadHalfAxisBinding returns a binding of one half of a raw axis to the action target.
// negative selects the negative half; otherwise the positive half is used.
func NewGamepadHalfAxisBinding(axis int, negative bool, target string) GamepadBinding {
	return gamepadprofile.NewGamepadHalfAxisBinding(axis, negative, target)
}

// NewGamepadHatBinding returns a binding of one direction mask of a raw hat to the action target.
// direction is a mask of hat direction constants such as 1 (up), 2 (right), 4 (down) or 8 (left),
// possibly combined.
func NewGamepadHatBinding(hat, direction int, target string) GamepadBinding {
	return gamepadprofile.NewGamepadHatBinding(hat, direction, target)
}

// NewGamepadChordBinding returns a multi-button binding to the action target.
func NewGamepadChordBinding(buttons []int, target string) GamepadChordBinding {
	return gamepadprofile.NewGamepadChordBinding(buttons, target)
}

// NewGamepadAxisConfig returns an axis configuration for the raw axis over the default -1..1
// range. It is validated when the profile is created.
func NewGamepadAxisConfig(axis int, logicalAxis GamepadLogicalAxis, deadZone float64, inverted bool) GamepadAxisConfig {
	return gamepadprofile.NewGamepadAxisConfig(axis, logicalAxis, deadZone, inverted)
}

// NewGamepadAxisConfigWithRange returns an axis configuration with an explicit input range.
// rangeMin must be smaller than rangeMax and both must lie within -1..1.
func NewGamepadAxisConfigWithRange(axis int, logicalAxis GamepadLogicalAxis, rangeMin, rangeMax, deadZone float64, inverted bool) GamepadAxisConfig {
	return gamepadprofile.NewGamepadAxisConfigWithRange(axis, logicalAxis, rangeMin, rangeMax, deadZone, inverted)
}

// NewGamepadMappingProfile builds and validates a profile for the device with the given SDL GUID.
//
// NewGamepadMappingProfile returns a [GamepadFieldError] for an empty GUID, an unknown binding
// target, an out-of-range index, an illegal axis range or dead zone, a repeated binding, or a
// chord with zero or repeated buttons. The returned profile is a private copy of the arguments.
func NewGamepadMappingProfile(deviceGUID string, name string, actions []string, bindings []GamepadBinding, chordBindings []GamepadChordBinding, axisConfigs []GamepadAxisConfig) (*GamepadMappingProfile, error) {
	return gamepadprofile.NewGamepadMappingProfile(deviceGUID, name, actions, bindings, chordBindings, axisConfigs)
}

// SetGamepadMappingProfile installs or replaces the profile for its device, taking effect on the
// next query even while the game is running. The registry keeps a private copy.
//
// SetGamepadMappingProfile is concurrent-safe.
func SetGamepadMappingProfile(profile *GamepadMappingProfile) error {
	return gamepadprofile.SetProfile(profile)
}

// GamepadMappingProfileForDevice returns a private copy of the profile registered for the device
// GUID, or nil when none exists.
//
// GamepadMappingProfileForDevice is concurrent-safe.
func GamepadMappingProfileForDevice(deviceGUID string) *GamepadMappingProfile {
	return gamepadprofile.Profile(deviceGUID)
}

// DeleteGamepadMappingProfile removes the profile for the device GUID and reports whether one was
// present. The device falls back to its default mapping on its next query.
//
// DeleteGamepadMappingProfile is concurrent-safe.
func DeleteGamepadMappingProfile(deviceGUID string) bool {
	return gamepadprofile.DeleteProfile(deviceGUID)
}

// ResetGamepadMappingProfiles removes every registered profile, restoring the default mappings.
//
// ResetGamepadMappingProfiles is concurrent-safe.
func ResetGamepadMappingProfiles() {
	gamepadprofile.Reset()
}

// AppendGamepadMappingDeviceGUIDs appends the GUIDs that currently have a profile to dst and
// returns the extended slice, in sorted order.
//
// AppendGamepadMappingDeviceGUIDs is concurrent-safe.
func AppendGamepadMappingDeviceGUIDs(dst []string) []string {
	return gamepadprofile.AppendDeviceGUIDs(dst)
}

// AppendGamepadMappingProfiles appends private copies of every registered profile to dst and
// returns the extended slice, ordered by device GUID.
//
// AppendGamepadMappingProfiles is concurrent-safe.
func AppendGamepadMappingProfiles(dst []*GamepadMappingProfile) []*GamepadMappingProfile {
	return gamepadprofile.AppendProfiles(dst)
}

// LoadGamepadMappingProfiles parses the versioned JSON document and atomically replaces every
// registered profile. On a parse or validation error the existing profiles are left untouched, so
// a bad document never leaves a half-loaded configuration. A successful load with an empty profile
// list clears the registry.
//
// LoadGamepadMappingProfiles is concurrent-safe.
func LoadGamepadMappingProfiles(data []byte) error {
	return gamepadprofile.LoadProfiles(data)
}

// ExportGamepadMappingProfiles serializes every registered profile as a versioned JSON document.
//
// ExportGamepadMappingProfiles is concurrent-safe.
func ExportGamepadMappingProfiles() ([]byte, error) {
	return gamepadprofile.ExportProfiles()
}

// HasGamepadMappingProfile reports whether the gamepad's device has a registered profile. A
// disconnected or unknown gamepad reports false, which is the default-mapping fallback.
//
// HasGamepadMappingProfile is concurrent-safe.
func HasGamepadMappingProfile(id GamepadID) bool {
	g := gamepad.Get(id)
	if g == nil {
		return false
	}
	return gamepadprofile.HasProfile(g)
}

// IsGamepadActionPressed reports whether any binding of the action is currently pressed on the
// gamepad. It reports false for a disconnected gamepad, an unknown action or a device without a
// profile.
//
// IsGamepadActionPressed is concurrent-safe.
func IsGamepadActionPressed(id GamepadID, action string) bool {
	g := gamepad.Get(id)
	if g == nil {
		return false
	}
	return gamepadprofile.IsActionPressed(g, action)
}

// GamepadActionValue returns the analog value of the action in 0..1. Several bindings of one
// action combine by taking the maximum, so they never stack. It returns 0 for a disconnected
// gamepad, an unknown action or a device without a profile.
//
// GamepadActionValue is concurrent-safe.
func GamepadActionValue(id GamepadID, action string) float64 {
	g := gamepad.Get(id)
	if g == nil {
		return 0
	}
	return gamepadprofile.ActionValue(g, action)
}

// GamepadLogicalAxisValue returns the value of a logical stick or trigger axis for the gamepad,
// with the feeding raw axis's range normalization, inversion and dead zone applied. Stick axes
// return -1..1 and trigger axes 0..1; an unmapped or disconnected gamepad returns 0.
//
// GamepadLogicalAxisValue is concurrent-safe.
func GamepadLogicalAxisValue(id GamepadID, logicalAxis GamepadLogicalAxis) float64 {
	g := gamepad.Get(id)
	if g == nil {
		return 0
	}
	return gamepadprofile.LogicalAxisValue(g, logicalAxis)
}

// IsGamepadActionAvailable reports whether the profile defines at least one binding for the
// action on the gamepad's device.
//
// IsGamepadActionAvailable is concurrent-safe.
func IsGamepadActionAvailable(id GamepadID, action string) bool {
	g := gamepad.Get(id)
	if g == nil {
		return false
	}
	return gamepadprofile.IsActionAvailable(g, action)
}
