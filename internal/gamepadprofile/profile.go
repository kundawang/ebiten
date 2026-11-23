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
	"fmt"
	"math"
	"slices"
)

// NewGamepadMappingProfile builds and validates a profile for the device with the given SDL GUID.
//
// NewGamepadMappingProfile returns a field-context error for an empty GUID, an unknown binding
// target, an out-of-range button, axis or hat index, an illegal axis range or dead zone, a repeated
// binding source, or a chord with zero or repeated buttons. On success the returned value is a
// private copy, so later mutation of the argument slices does not affect it.
func NewGamepadMappingProfile(deviceGUID string, name string, actions []string, bindings []GamepadBinding, chordBindings []GamepadChordBinding, axisConfigs []GamepadAxisConfig) (*GamepadMappingProfile, error) {
	p := &GamepadMappingProfile{
		deviceGUID:    deviceGUID,
		name:          name,
		actions:       slices.Clone(actions),
		bindings:      slices.Clone(bindings),
		chordBindings: slices.Clone(chordBindings),
		axisConfigs:   slices.Clone(axisConfigs),
	}
	if err := p.validate(); err != nil {
		return nil, err
	}
	return p, nil
}

// validate checks every invariant a profile must hold. Error paths name the offending JSON-like
// field so a rejected document points at its cause.
func (p *GamepadMappingProfile) validate() error {
	if p.deviceGUID == "" {
		return &FieldError{Path: "device_guid", Err: fmt.Errorf("gamepadprofile: device GUID must not be empty")}
	}

	actionSet := make(map[string]struct{}, len(p.actions))
	for i, action := range p.actions {
		if action == "" {
			return &FieldError{Path: fmt.Sprintf("actions[%d]", i), Err: fmt.Errorf("gamepadprofile: action name must not be empty")}
		}
		if _, ok := actionSet[action]; ok {
			return &FieldError{Path: fmt.Sprintf("actions[%d]", i), Err: fmt.Errorf("gamepadprofile: duplicate action %q", action)}
		}
		actionSet[action] = struct{}{}
	}

	// Every single-input binding gets a unique source key: a duplicate key means two bindings claim
	// the same physical input.
	sources := make(map[string]struct{}, len(p.bindings))
	for i := range p.bindings {
		b := &p.bindings[i]
		path := fmt.Sprintf("bindings[%d]", i)
		if _, ok := actionSet[b.target]; !ok {
			return &FieldError{Path: path + ".target", Err: fmt.Errorf("gamepadprofile: unknown action %q", b.target)}
		}
		if err := validateBindingSource(b, path); err != nil {
			return err
		}
		key := b.sourceKey()
		if _, ok := sources[key]; ok {
			return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: duplicate binding for %s", key)}
		}
		sources[key] = struct{}{}
	}

	for i := range p.chordBindings {
		c := &p.chordBindings[i]
		path := fmt.Sprintf("chord_bindings[%d]", i)
		if _, ok := actionSet[c.target]; !ok {
			return &FieldError{Path: path + ".target", Err: fmt.Errorf("gamepadprofile: unknown action %q", c.target)}
		}
		if len(c.buttons) == 0 {
			return &FieldError{Path: path + ".buttons", Err: fmt.Errorf("gamepadprofile: a chord must contain at least one button")}
		}
		seen := make(map[int]struct{}, len(c.buttons))
		for j, button := range c.buttons {
			if err := validateRawIndex(button, "button", fmt.Sprintf("%s.buttons[%d]", path, j)); err != nil {
				return err
			}
			if _, ok := seen[button]; ok {
				return &FieldError{Path: fmt.Sprintf("%s.buttons[%d]", path, j), Err: fmt.Errorf("gamepadprofile: duplicate button %d in chord", button)}
			}
			seen[button] = struct{}{}
		}
	}

	axisByRaw := make(map[int]struct{}, len(p.axisConfigs))
	logicalAxes := make(map[LogicalAxis]struct{}, len(p.axisConfigs))
	for i := range p.axisConfigs {
		c := &p.axisConfigs[i]
		path := fmt.Sprintf("axes[%d]", i)
		if err := validateRawIndex(c.axis, "axis", path+".axis"); err != nil {
			return err
		}
		if _, ok := axisByRaw[c.axis]; ok {
			return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: duplicate axis configuration for raw axis %d", c.axis)}
		}
		axisByRaw[c.axis] = struct{}{}

		if c.logicalAxis != LogicalAxisNone {
			if c.logicalAxis < 0 || c.logicalAxis > LogicalAxisMax {
				return &FieldError{Path: path + ".logical_axis", Err: fmt.Errorf("gamepadprofile: invalid logical axis %d", c.logicalAxis)}
			}
			if _, ok := logicalAxes[c.logicalAxis]; ok {
				return &FieldError{Path: path + ".logical_axis", Err: fmt.Errorf("gamepadprofile: logical axis %d is fed by more than one raw axis", c.logicalAxis)}
			}
			logicalAxes[c.logicalAxis] = struct{}{}
		}

		if math.IsNaN(c.deadZone) || c.deadZone < 0 || c.deadZone >= 1 {
			return &FieldError{Path: path + ".dead_zone", Err: fmt.Errorf("gamepadprofile: dead zone must be in [0, 1), got %v", c.deadZone)}
		}
		if math.IsNaN(c.rangeMin) || math.IsNaN(c.rangeMax) {
			return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: axis range must not be NaN")}
		}
		if c.rangeMin < -1 || c.rangeMin > 1 || c.rangeMax < -1 || c.rangeMax > 1 {
			return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: axis range endpoints must be within [-1, 1], got [%v, %v]", c.rangeMin, c.rangeMax)}
		}
		if c.rangeMin >= c.rangeMax {
			return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: axis range min must be smaller than max, got [%v, %v]", c.rangeMin, c.rangeMax)}
		}
	}

	return nil
}

// validateBindingSource checks the physical input half of a binding.
func validateBindingSource(b *GamepadBinding, path string) error {
	switch b.kind {
	case BindSourceButton:
		return validateRawIndex(b.index, "button", path+".index")
	case BindSourceAxis, BindSourceHalfAxis:
		return validateRawIndex(b.index, "axis", path+".index")
	case BindSourceHat:
		if err := validateRawIndex(b.index, "hat", path+".index"); err != nil {
			return err
		}
		if b.hatDirection <= 0 || b.hatDirection&^hatDirectionMask != 0 {
			return &FieldError{Path: path + ".hat_direction", Err: fmt.Errorf("gamepadprofile: invalid hat direction %d", b.hatDirection)}
		}
		return nil
	default:
		return &FieldError{Path: path + ".kind", Err: fmt.Errorf("gamepadprofile: unknown binding kind %d", b.kind)}
	}
}

// hatDirectionMask is every legal hat direction bit (up, right, down and left).
const hatDirectionMask = 1 | 2 | 4 | 8

// validateRawIndex rejects negative indices and indices past the engine's fixed hardware limits.
func validateRawIndex(index int, kind, path string) error {
	max := rawAxisCount - 1
	switch kind {
	case "button":
		max = rawButtonCount - 1
	case "hat":
		max = rawHatCount - 1
	}
	if index < 0 || index > max {
		return &FieldError{Path: path, Err: fmt.Errorf("gamepadprofile: %s index out of range [0, %d]: %d", kind, max, index)}
	}
	return nil
}

// sourceKey renders the physical input of a binding as a unique string for duplicate detection.
func (b *GamepadBinding) sourceKey() string {
	switch b.kind {
	case BindSourceButton:
		return fmt.Sprintf("b%d", b.index)
	case BindSourceAxis:
		return fmt.Sprintf("a%d", b.index)
	case BindSourceHalfAxis:
		if b.negative {
			return fmt.Sprintf("-a%d", b.index)
		}
		return fmt.Sprintf("+a%d", b.index)
	case BindSourceHat:
		return fmt.Sprintf("h%d.%d", b.index, b.hatDirection)
	default:
		return "none"
	}
}

// clone returns a deep enough copy that no slice inside shares storage with p.
func (p *GamepadMappingProfile) clone() *GamepadMappingProfile {
	chords := make([]GamepadChordBinding, len(p.chordBindings))
	for i := range p.chordBindings {
		chords[i].buttons = slices.Clone(p.chordBindings[i].buttons)
		chords[i].target = p.chordBindings[i].target
	}
	return &GamepadMappingProfile{
		deviceGUID:    p.deviceGUID,
		name:          p.name,
		actions:       slices.Clone(p.actions),
		bindings:      slices.Clone(p.bindings),
		chordBindings: chords,
		axisConfigs:   slices.Clone(p.axisConfigs),
	}
}

// FieldError wraps an error with the profile field or array element that caused it.
type FieldError struct {
	// Path is the dotted field path, with array indices such as bindings[2].source.
	Path string
	Err  error
}

func (e *FieldError) Error() string {
	if e.Path == "" {
		return e.Err.Error()
	}
	return e.Path + ": " + e.Err.Error()
}

func (e *FieldError) Unwrap() error {
	return e.Err
}
