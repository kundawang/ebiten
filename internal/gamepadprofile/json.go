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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// configDocument is the versioned JSON container of an exported profile set.
type configDocument struct {
	Version  int                  `json:"version"`
	Profiles []gamepadProfileWire `json:"profiles"`
}

type gamepadProfileWire struct {
	DeviceGUID    string               `json:"device_guid"`
	Name          string               `json:"name,omitempty"`
	Actions       []string             `json:"actions"`
	Bindings      []gamepadBindingWire `json:"bindings"`
	ChordBindings []gamepadChordWire   `json:"chord_bindings,omitempty"`
	Axes          []gamepadAxisWire    `json:"axes,omitempty"`
}

type gamepadBindingWire struct {
	Kind         string `json:"kind"`
	Index        int    `json:"index"`
	Negative     bool   `json:"negative,omitempty"`
	HatDirection int    `json:"hat_direction,omitempty"`
	Target       string `json:"target"`
}

type gamepadChordWire struct {
	Buttons []int  `json:"buttons"`
	Target  string `json:"target"`
}

type gamepadAxisWire struct {
	Axis        int     `json:"axis"`
	LogicalAxis int     `json:"logical_axis"`
	DeadZone    float64 `json:"dead_zone"`
	RangeMin    float64 `json:"range_min"`
	RangeMax    float64 `json:"range_max"`
	Inverted    bool    `json:"inverted"`
}

// MarshalProfiles serializes the profiles as a versioned, indented JSON document.
//
// The profiles must have passed [NewGamepadMappingProfile]; marshaling an unvalidated value can
// still fail and then returns an error without producing a document.
func MarshalProfiles(profiles []*GamepadMappingProfile) ([]byte, error) {
	doc := configDocument{
		Version:  CurrentConfigVersion,
		Profiles: make([]gamepadProfileWire, 0, len(profiles)),
	}
	for i, p := range profiles {
		if p == nil {
			return nil, &FieldError{Path: fmt.Sprintf("profiles[%d]", i), Err: fmt.Errorf("gamepadprofile: profile must not be nil")}
		}
		doc.Profiles = append(doc.Profiles, p.toWire())
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(&doc); err != nil {
		return nil, fmt.Errorf("gamepadprofile: failed to encode profiles: %w", err)
	}
	return buf.Bytes(), nil
}

// UnmarshalProfiles parses a versioned JSON document and returns validated profiles.
//
// Unknown JSON fields, a bad or incompatible version, malformed JSON and every validation failure
// are reported with a field-context [FieldError]. Nothing is partially returned: on error the
// result is nil.
func UnmarshalProfiles(data []byte) ([]*GamepadMappingProfile, error) {
	// First pass: read only the version, tolerating unknown fields so an old document with a
	// different shape is reported as a version incompatibility rather than a syntax failure. A
	// decoder (not json.Unmarshal) is used so trailing data is deferred to the strict second pass.
	var header struct {
		Version int `json:"version"`
	}
	headerDec := json.NewDecoder(bytes.NewReader(data))
	if err := headerDec.Decode(&header); err != nil {
		return nil, jsonFieldError(err)
	}
	if header.Version != CurrentConfigVersion {
		return nil, &FieldError{
			Path: "version",
			Err:  fmt.Errorf("gamepadprofile: incompatible config version %d; this build supports version %d", header.Version, CurrentConfigVersion),
		}
	}

	// Second pass: the full, strict decode that rejects unknown fields and wrong field types.
	var doc configDocument
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return nil, jsonFieldError(err)
	}
	// Reject trailing data after the document rather than silently ignoring it. The first document
	// was decoded fully, so anything the second decode finds (a value or garbage) is trailing data.
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != io.EOF {
		return nil, &FieldError{Path: "", Err: fmt.Errorf("gamepadprofile: unexpected trailing data after the document")}
	}

	profiles := make([]*GamepadMappingProfile, 0, len(doc.Profiles))
	for i, w := range doc.Profiles {
		p, err := profileFromWire(w, fmt.Sprintf("profiles[%d]", i))
		if err != nil {
			return nil, err
		}
		profiles = append(profiles, p)
	}
	return profiles, nil
}

func (p *GamepadMappingProfile) toWire() gamepadProfileWire {
	w := gamepadProfileWire{
		DeviceGUID:    p.deviceGUID,
		Name:          p.name,
		Actions:       append([]string(nil), p.actions...),
		Bindings:      make([]gamepadBindingWire, 0, len(p.bindings)),
		ChordBindings: make([]gamepadChordWire, 0, len(p.chordBindings)),
		Axes:          make([]gamepadAxisWire, 0, len(p.axisConfigs)),
	}
	for _, b := range p.bindings {
		w.Bindings = append(w.Bindings, gamepadBindingWire{
			Kind:         bindingKindString(b.kind),
			Index:        b.index,
			Negative:     b.negative,
			HatDirection: b.hatDirection,
			Target:       b.target,
		})
	}
	for _, c := range p.chordBindings {
		w.ChordBindings = append(w.ChordBindings, gamepadChordWire{
			Buttons: append([]int(nil), c.buttons...),
			Target:  c.target,
		})
	}
	for _, c := range p.axisConfigs {
		w.Axes = append(w.Axes, gamepadAxisWire{
			Axis:        c.axis,
			LogicalAxis: int(c.logicalAxis),
			DeadZone:    c.deadZone,
			RangeMin:    c.rangeMin,
			RangeMax:    c.rangeMax,
			Inverted:    c.inverted,
		})
	}
	return w
}

func profileFromWire(w gamepadProfileWire, prefix string) (*GamepadMappingProfile, error) {
	p := &GamepadMappingProfile{
		deviceGUID: w.DeviceGUID,
		name:       w.Name,
		actions:    append([]string(nil), w.Actions...),
	}
	for i, b := range w.Bindings {
		kind, ok := parseBindingKindString(b.Kind)
		if !ok {
			return nil, &FieldError{Path: fmt.Sprintf("%s.bindings[%d].kind", prefix, i), Err: fmt.Errorf("gamepadprofile: unknown binding kind %q", b.Kind)}
		}
		p.bindings = append(p.bindings, GamepadBinding{
			kind:         kind,
			index:        b.Index,
			negative:     b.Negative,
			hatDirection: b.HatDirection,
			target:       b.Target,
		})
	}
	for _, c := range w.ChordBindings {
		p.chordBindings = append(p.chordBindings, GamepadChordBinding{
			buttons: append([]int(nil), c.Buttons...),
			target:  c.Target,
		})
	}
	for _, c := range w.Axes {
		p.axisConfigs = append(p.axisConfigs, GamepadAxisConfig{
			axis:        c.Axis,
			logicalAxis: LogicalAxis(c.LogicalAxis),
			deadZone:    c.DeadZone,
			rangeMin:    c.RangeMin,
			rangeMax:    c.RangeMax,
			inverted:    c.Inverted,
		})
	}

	if err := p.validate(); err != nil {
		// Prefix every generated path with the profile's position in the document.
		var fe *FieldError
		if errors.As(err, &fe) {
			return nil, &FieldError{Path: prefix + "." + fe.Path, Err: fe.Err}
		}
		return nil, &FieldError{Path: prefix, Err: err}
	}
	return p, nil
}

func bindingKindString(kind BindSourceKind) string {
	switch kind {
	case BindSourceButton:
		return "button"
	case BindSourceAxis:
		return "axis"
	case BindSourceHalfAxis:
		return "half_axis"
	case BindSourceHat:
		return "hat"
	default:
		return "none"
	}
}

func parseBindingKindString(s string) (BindSourceKind, bool) {
	switch s {
	case "button":
		return BindSourceButton, true
	case "axis":
		return BindSourceAxis, true
	case "half_axis":
		return BindSourceHalfAxis, true
	case "hat":
		return BindSourceHat, true
	default:
		return BindSourceNone, false
	}
}

// jsonFieldError turns a decoder error into a FieldError, using the structured field path that a
// type or unknown-field error carries so the failure points at the offending document field.
func jsonFieldError(err error) error {
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return &FieldError{Path: typeErr.Field, Err: fmt.Errorf("gamepadprofile: invalid JSON: %w", err)}
	}
	var fieldErr *json.SyntaxError
	if errors.As(err, &fieldErr) {
		return &FieldError{Path: "", Err: fmt.Errorf("gamepadprofile: invalid JSON at byte %d: %w", fieldErr.Offset, err)}
	}
	return &FieldError{Path: "", Err: fmt.Errorf("gamepadprofile: invalid JSON: %w", err)}
}
