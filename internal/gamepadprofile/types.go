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

// Package gamepadprofile stores per-device, user-customizable gamepad mappings.
//
// A profile is keyed by a device's SDL GUID and remaps the device's raw buttons, axes and hats onto
// named, game-defined actions, together with per-axis dead zones and inversion. Profiles can be
// serialized to and parsed from a versioned JSON document and switched at runtime through a
// process-wide registry. The standard-layout mapping and the raw gamepad queries are unaffected.
package gamepadprofile

// CurrentConfigVersion is the version of the JSON document written and accepted by this package.
const CurrentConfigVersion = 1

// BindSourceKind identifies the physical input a binding reads.
type BindSourceKind int

const (
	// BindSourceNone is the zero value and maps no input.
	BindSourceNone BindSourceKind = iota

	// BindSourceButton is a raw button, identified by its index.
	BindSourceButton

	// BindSourceAxis is a whole raw axis in -1..1, identified by its index.
	BindSourceAxis

	// BindSourceHalfAxis is one half of a raw axis in 0..1, identified by its index and its
	// positive or negative direction.
	BindSourceHalfAxis

	// BindSourceHat is one direction of a raw hat, identified by its hat index and direction mask.
	BindSourceHat
)

// LogicalAxis identifies a stick or trigger axis a physical raw axis feeds.
type LogicalAxis int

const (
	// LogicalAxisNone reports that a raw axis feeds no logical stick or trigger axis.
	LogicalAxisNone LogicalAxis = -1

	LogicalAxisLeftStickHorizontal  LogicalAxis = 0
	LogicalAxisLeftStickVertical    LogicalAxis = 1
	LogicalAxisRightStickHorizontal LogicalAxis = 2
	LogicalAxisRightStickVertical   LogicalAxis = 3
	LogicalAxisLeftTrigger          LogicalAxis = 4
	LogicalAxisRightTrigger         LogicalAxis = 5

	LogicalAxisMax = LogicalAxisRightTrigger
)

// GamepadMappingProfile is the customizable mapping of one gamepad model, identified by its SDL GUID.
//
// A profile value is immutable once validated: the registry stores and returns copies, so a caller
// cannot change a registered profile by mutating a value it holds.
type GamepadMappingProfile struct {
	// deviceGUID is the SDL GUID of the device this profile belongs to. It is never empty.
	deviceGUID string

	// name is an optional human-readable label.
	name string

	// actions are the game-defined action names a binding may target. Every binding target must
	// appear here.
	actions []string

	// bindings maps each physical input to at most one action. Two entries may never share a source.
	bindings []GamepadBinding

	// chordBindings are the multi-button bindings. They may target the same actions as bindings.
	chordBindings []GamepadChordBinding

	// axisConfigs is the per-raw-axis configuration (dead zone, inversion and the like), at most one
	// entry per raw axis.
	axisConfigs []GamepadAxisConfig
}

// DeviceGUID returns the SDL GUID of the device the profile belongs to.
func (p *GamepadMappingProfile) DeviceGUID() string {
	return p.deviceGUID
}

// Name returns the optional human-readable label of the profile.
func (p *GamepadMappingProfile) Name() string {
	return p.name
}

// AppendActions appends the profile's action names to dst and returns the extended slice. The
// returned slice is the caller's own: mutating it never changes the profile.
func (p *GamepadMappingProfile) AppendActions(dst []string) []string {
	return append(dst, p.actions...)
}

// AppendBindings appends a copy of the profile's single-input bindings to dst and returns the
// extended slice.
func (p *GamepadMappingProfile) AppendBindings(dst []GamepadBinding) []GamepadBinding {
	return append(dst, p.bindings...)
}

// AppendChordBindings appends a copy of the profile's multi-button bindings to dst and returns the
// extended slice.
func (p *GamepadMappingProfile) AppendChordBindings(dst []GamepadChordBinding) []GamepadChordBinding {
	return append(dst, p.chordBindings...)
}

// AppendAxisConfigs appends a copy of the profile's per-axis configurations to dst and returns the
// extended slice.
func (p *GamepadMappingProfile) AppendAxisConfigs(dst []GamepadAxisConfig) []GamepadAxisConfig {
	return append(dst, p.axisConfigs...)
}

// GamepadBinding maps one physical input (a button, a whole or half axis, or a hat direction) to an
// action.
type GamepadBinding struct {
	// kind is the kind of physical input.
	kind BindSourceKind

	// index is the raw button, axis or hat index.
	index int

	// negative selects the negative half of a half-axis binding. It is ignored by the other kinds.
	negative bool

	// hatDirection is the direction mask of a hat binding. It is ignored by the other kinds.
	hatDirection int

	// target is the action the input drives. It must be one of the profile's actions.
	target string
}

// NewGamepadBinding returns a binding of a raw button to the action target.
func NewGamepadButtonBinding(button int, target string) GamepadBinding {
	return GamepadBinding{kind: BindSourceButton, index: button, target: target}
}

// NewGamepadAxisBinding returns a binding of a whole raw axis to the action target.
func NewGamepadAxisBinding(axis int, target string) GamepadBinding {
	return GamepadBinding{kind: BindSourceAxis, index: axis, target: target}
}

// NewGamepadHalfAxisBinding returns a binding of one half of a raw axis to the action target.
// negative selects the negative half; otherwise the positive half is used.
func NewGamepadHalfAxisBinding(axis int, negative bool, target string) GamepadBinding {
	return GamepadBinding{kind: BindSourceHalfAxis, index: axis, negative: negative, target: target}
}

// NewGamepadHatBinding returns a binding of one direction mask of a raw hat to the action target.
// direction is a mask of the hat direction constants, such as 1 (up), 2 (right), 4 (down) or
// 8 (left), possibly combined.
func NewGamepadHatBinding(hat, direction int, target string) GamepadBinding {
	return GamepadBinding{kind: BindSourceHat, index: hat, hatDirection: direction, target: target}
}

// Kind returns the kind of physical input the binding reads.
func (b GamepadBinding) Kind() BindSourceKind {
	return b.kind
}

// Index returns the raw button, axis or hat index of the binding.
func (b GamepadBinding) Index() int {
	return b.index
}

// Negative reports whether a half-axis binding reads the negative half.
func (b GamepadBinding) Negative() bool {
	return b.negative
}

// HatDirection returns the direction mask of a hat binding.
func (b GamepadBinding) HatDirection() int {
	return b.hatDirection
}

// Target returns the action the binding drives.
func (b GamepadBinding) Target() string {
	return b.target
}

// GamepadChordBinding maps a simultaneous press of several raw buttons to an action.
type GamepadChordBinding struct {
	// buttons are the raw button indices of the chord. They are distinct and never empty.
	buttons []int

	// target is the action the chord drives. It must be one of the profile's actions.
	target string
}

// NewGamepadChordBinding returns a multi-button binding to the action target. It does not validate
// the buttons; [NewGamepadMappingProfile] reports an empty or repeated list as an error.
func NewGamepadChordBinding(buttons []int, target string) GamepadChordBinding {
	return GamepadChordBinding{buttons: append([]int(nil), buttons...), target: target}
}

// AppendButtons appends a copy of the chord's button indices to dst and returns the extended slice.
func (b GamepadChordBinding) AppendButtons(dst []int) []int {
	return append(dst, b.buttons...)
}

// Target returns the action the chord drives.
func (b GamepadChordBinding) Target() string {
	return b.target
}

// GamepadAxisConfig is the per-raw-axis configuration: the logical axis it feeds, its dead zone,
// its input range and whether its direction is inverted.
type GamepadAxisConfig struct {
	// axis is the raw axis index.
	axis int

	// logicalAxis is the logical stick or trigger axis the raw axis feeds, or LogicalAxisNone.
	logicalAxis LogicalAxis

	// deadZone is the axial dead zone in 0..1. Inputs within it read as centered.
	deadZone float64

	// rangeMin and rangeMax are the reported input range, with rangeMin < rangeMax. A zero pair
	// means the default -1..1 range.
	rangeMin float64
	rangeMax float64

	// inverted reverses the axis direction.
	inverted bool
}

// NewGamepadAxisConfig returns an axis configuration for the raw axis over the default -1..1 range.
// Validate it through [NewGamepadMappingProfile]; a dead zone outside 0..1 is rejected there.
func NewGamepadAxisConfig(axis int, logicalAxis LogicalAxis, deadZone float64, inverted bool) GamepadAxisConfig {
	return GamepadAxisConfig{
		axis:        axis,
		logicalAxis: logicalAxis,
		deadZone:    deadZone,
		rangeMin:    -1,
		rangeMax:    1,
		inverted:    inverted,
	}
}

// NewGamepadAxisConfigWithRange returns an axis configuration with an explicit input range.
// rangeMin must be smaller than rangeMax and both must lie within -1..1.
func NewGamepadAxisConfigWithRange(axis int, logicalAxis LogicalAxis, rangeMin, rangeMax, deadZone float64, inverted bool) GamepadAxisConfig {
	return GamepadAxisConfig{
		axis:        axis,
		logicalAxis: logicalAxis,
		deadZone:    deadZone,
		rangeMin:    rangeMin,
		rangeMax:    rangeMax,
		inverted:    inverted,
	}
}

// Axis returns the raw axis index.
func (c GamepadAxisConfig) Axis() int {
	return c.axis
}

// LogicalAxis returns the logical axis the raw axis feeds, or LogicalAxisNone.
func (c GamepadAxisConfig) LogicalAxis() LogicalAxis {
	return c.logicalAxis
}

// DeadZone returns the axial dead zone.
func (c GamepadAxisConfig) DeadZone() float64 {
	return c.deadZone
}

// Range returns the reported input range.
func (c GamepadAxisConfig) Range() (min, max float64) {
	return c.rangeMin, c.rangeMax
}

// Inverted reports whether the axis direction is reversed.
func (c GamepadAxisConfig) Inverted() bool {
	return c.inverted
}
