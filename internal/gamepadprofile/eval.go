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

package gamepadprofile

import (
	"math"
	"reflect"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepaddb"
)

// RawGamepad is the raw-state surface the evaluation needs from a gamepad. *gamepad.Gamepad
// satisfies it, and so does a test fake. It deliberately excludes standard-layout queries, which a
// profile must never alter.
type RawGamepad interface {
	SDLID() string
	AxisCount() int
	ButtonCount() int
	HatCount() int
	IsAxisReady(axis int) bool
	Axis(axis int) float64
	Button(button int) bool
	Hat(hat int) int
}

// HasProfile reports whether a registered profile exists for the gamepad's device GUID. A
// disconnected gamepad (nil, including a typed nil interface) or an unknown device simply reports
// false, which is the default mapping fallback.
//
// HasProfile is concurrent-safe.
func HasProfile(g RawGamepad) bool {
	if isNilGamepad(g) {
		return false
	}
	return theRegistry.profileForDevice(g.SDLID()) != nil
}

// IsActionPressed reports whether any binding of the action is currently pressed on the gamepad.
// It reports false for a nil gamepad, an unknown action or a device without a profile.
//
// IsActionPressed is concurrent-safe.
func IsActionPressed(g RawGamepad, action string) bool {
	p, ok := profileAndGamepad(g)
	if !ok {
		return false
	}
	return p.actionPressed(g, action)
}

// ActionValue returns the analog value of the action in 0..1. Whole-axis and half-axis bindings
// contribute their magnitude; the maximum across all of the action's bindings wins, so several
// bindings of one action do not stack. A chord contributes 1 while held.
//
// ActionValue returns 0 for a nil gamepad, an unknown action or a device without a profile.
//
// ActionValue is concurrent-safe.
func ActionValue(g RawGamepad, action string) float64 {
	p, ok := profileAndGamepad(g)
	if !ok {
		return 0
	}
	return p.actionValue(g, action)
}

// LogicalAxisValue returns the value of a logical stick or trigger axis in -1..1 (triggers use
// 0..1). The raw axis feeding it has its range normalization, inversion and dead zone applied.
//
// LogicalAxisValue returns 0 for a nil gamepad, a device without a profile or a logical axis that
// no configured raw axis feeds.
//
// LogicalAxisValue is concurrent-safe.
func LogicalAxisValue(g RawGamepad, logicalAxis LogicalAxis) float64 {
	p, ok := profileAndGamepad(g)
	if !ok {
		return 0
	}
	return p.logicalAxisValue(g, logicalAxis)
}

// IsActionAvailable reports whether the profile defines at least one binding for the action.
func IsActionAvailable(g RawGamepad, action string) bool {
	p, ok := profileAndGamepad(g)
	if !ok {
		return false
	}
	for i := range p.bindings {
		if p.bindings[i].target == action {
			return true
		}
	}
	for i := range p.chordBindings {
		if p.chordBindings[i].target == action {
			return true
		}
	}
	return false
}

// profileAndGamepad resolves the registered profile for the gamepad. The second result is false on
// every fallback path: a nil gamepad or an unknown device. The returned profile is read-only and
// must not be mutated.
func profileAndGamepad(g RawGamepad) (*GamepadMappingProfile, bool) {
	if isNilGamepad(g) {
		return nil, false
	}
	p := theRegistry.profileForDevice(g.SDLID())
	if p == nil {
		return nil, false
	}
	return p, true
}

// isNilGamepad reports whether the interface is nil or holds a typed nil pointer, as a
// disconnected gamepad lookup can produce the latter.
func isNilGamepad(g RawGamepad) bool {
	if g == nil {
		return true
	}
	v := reflect.ValueOf(g)
	return v.Kind() == reflect.Ptr && v.IsNil()
}

func (p *GamepadMappingProfile) axisConfig(axis int) (GamepadAxisConfig, bool) {
	for i := range p.axisConfigs {
		if p.axisConfigs[i].axis == axis {
			return p.axisConfigs[i], true
		}
	}
	return GamepadAxisConfig{}, false
}

// processedAxis reads one raw axis and applies range normalization, inversion and the dead zone.
// The result is in -1..1; an axis that is not ready or is inside its dead zone reads 0.
func (p *GamepadMappingProfile) processedAxis(g RawGamepad, axis int) float64 {
	if axis < 0 || axis >= g.AxisCount() || !g.IsAxisReady(axis) {
		return 0
	}
	raw := g.Axis(axis)
	if math.IsNaN(raw) {
		return 0
	}

	cfg, ok := p.axisConfig(axis)
	if ok {
		// Normalize the configured input range onto -1..1.
		span := cfg.rangeMax - cfg.rangeMin
		v := (raw-cfg.rangeMin)/span*2 - 1
		if v < -1 {
			v = -1
		} else if v > 1 {
			v = 1
		}
		if cfg.inverted {
			v = -v
		}
		return applyDeadZone(v, cfg.deadZone)
	}

	// An axis without a configuration still gets the default clamping so out-of-range hardware never
	// escapes the documented range.
	if raw < -1 {
		return -1
	}
	if raw > 1 {
		return 1
	}
	return raw
}

// applyDeadZone maps values inside the dead zone to 0 and rescales the remainder to fill -1..1 so
// the first input past the dead zone starts from 0 rather than jumping.
func applyDeadZone(v, deadZone float64) float64 {
	av := math.Abs(v)
	if av <= deadZone {
		return 0
	}
	out := (av - deadZone) / (1 - deadZone)
	if out > 1 {
		out = 1
	}
	if v < 0 {
		return -out
	}
	return out
}

// bindingPressed evaluates the physical input of one binding.
func (p *GamepadMappingProfile) bindingPressed(g RawGamepad, b GamepadBinding) bool {
	switch b.kind {
	case BindSourceButton:
		if b.index >= g.ButtonCount() {
			return false
		}
		return g.Button(b.index)
	case BindSourceHat:
		if b.index >= g.HatCount() {
			return false
		}
		return g.Hat(b.index)&b.hatDirection != 0
	case BindSourceAxis:
		// A whole axis "pressed" matches the standard-layout trigger convention: the positive half
		// beyond the dead zone.
		return p.bindingValue(g, b) > gamepaddb.ButtonPressedThreshold
	case BindSourceHalfAxis:
		return p.bindingValue(g, b) > gamepaddb.ButtonPressedThreshold
	default:
		return false
	}
}

// bindingValue evaluates the 0..1 analog magnitude of one binding.
func (p *GamepadMappingProfile) bindingValue(g RawGamepad, b GamepadBinding) float64 {
	switch b.kind {
	case BindSourceButton:
		if b.index < g.ButtonCount() && g.Button(b.index) {
			return 1
		}
		return 0
	case BindSourceHat:
		if b.index < g.HatCount() && g.Hat(b.index)&b.hatDirection != 0 {
			return 1
		}
		return 0
	case BindSourceAxis:
		v := p.processedAxis(g, b.index)
		// A whole axis maps its positive travel to the action, as an analog trigger does.
		if v <= 0 {
			return 0
		}
		return v
	case BindSourceHalfAxis:
		v := p.processedAxis(g, b.index)
		if b.negative {
			v = -v
		}
		if v <= 0 {
			return 0
		}
		return v
	default:
		return 0
	}
}

func (p *GamepadMappingProfile) actionPressed(g RawGamepad, action string) bool {
	for i := range p.bindings {
		b := &p.bindings[i]
		if b.target == action && p.bindingPressed(g, *b) {
			return true
		}
	}
	for i := range p.chordBindings {
		c := &p.chordBindings[i]
		if c.target == action && p.chordPressed(g, c) {
			return true
		}
	}
	return false
}

func (p *GamepadMappingProfile) actionValue(g RawGamepad, action string) float64 {
	var best float64
	for i := range p.bindings {
		b := &p.bindings[i]
		if b.target != action {
			continue
		}
		if v := p.bindingValue(g, *b); v > best {
			best = v
		}
	}
	for i := range p.chordBindings {
		c := &p.chordBindings[i]
		if c.target == action && p.chordPressed(g, c) {
			best = 1
		}
	}
	return best
}

// chordPressed reports whether every button of the chord is held. A button the device does not
// have can never be pressed, so such a chord never fires.
func (p *GamepadMappingProfile) chordPressed(g RawGamepad, c *GamepadChordBinding) bool {
	for _, button := range c.buttons {
		if button >= g.ButtonCount() || !g.Button(button) {
			return false
		}
	}
	return true
}

func (p *GamepadMappingProfile) logicalAxisValue(g RawGamepad, logicalAxis LogicalAxis) float64 {
	for i := range p.axisConfigs {
		c := &p.axisConfigs[i]
		if c.logicalAxis != logicalAxis {
			continue
		}
		v := p.processedAxis(g, c.axis)
		if logicalAxis == LogicalAxisLeftTrigger || logicalAxis == LogicalAxisRightTrigger {
			// Triggers are one-sided: map -1..1 onto 0..1.
			return (v + 1) / 2
		}
		return v
	}
	return 0
}
