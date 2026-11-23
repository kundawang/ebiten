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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

func validProfile(t *testing.T, guid string) *gamepadprofile.GamepadMappingProfile {
	t.Helper()
	p, err := gamepadprofile.NewGamepadMappingProfile(
		guid,
		"Pad "+guid,
		[]string{"jump", "fire", "left", "right"},
		[]gamepadprofile.GamepadBinding{
			gamepadprofile.NewGamepadButtonBinding(0, "jump"),
			gamepadprofile.NewGamepadHalfAxisBinding(0, true, "left"),
			gamepadprofile.NewGamepadHalfAxisBinding(0, false, "right"),
			gamepadprofile.NewGamepadHatBinding(0, 1, "jump"),
		},
		[]gamepadprofile.GamepadChordBinding{
			gamepadprofile.NewGamepadChordBinding([]int{0, 1}, "fire"),
		},
		[]gamepadprofile.GamepadAxisConfig{
			gamepadprofile.NewGamepadAxisConfig(0, gamepadprofile.LogicalAxisLeftStickHorizontal, 0.2, false),
			gamepadprofile.NewGamepadAxisConfig(1, gamepadprofile.LogicalAxisLeftStickVertical, 0.15, true),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestMarshalUnmarshalRoundTrip(t *testing.T) {
	original := []*gamepadprofile.GamepadMappingProfile{
		validProfile(t, "guid-roundtrip-1"),
		validProfile(t, "guid-roundtrip-2"),
	}
	data, err := gamepadprofile.MarshalProfiles(original)
	if err != nil {
		t.Fatal(err)
	}

	got, err := gamepadprofile.UnmarshalProfiles(data)
	if err != nil {
		t.Fatalf("UnmarshalProfiles returned an error: %v", err)
	}
	if len(got) != len(original) {
		t.Fatalf("profile count = %d; want %d", len(got), len(original))
	}
	for i, p := range got {
		if p.DeviceGUID() != original[i].DeviceGUID() {
			t.Errorf("profile %d GUID = %q; want %q", i, p.DeviceGUID(), original[i].DeviceGUID())
		}
		if got := p.AppendBindings(nil); len(got) != len(original[i].AppendBindings(nil)) {
			t.Errorf("profile %d binding count = %d; want %d", i, len(got), len(original[i].AppendBindings(nil)))
		}
		if got := p.AppendChordBindings(nil); len(got) != 1 || len(got[0].AppendButtons(nil)) != 2 {
			t.Errorf("profile %d chord did not round-trip: %+v", i, got)
		}
		axes := p.AppendAxisConfigs(nil)
		if len(axes) != 2 || !axes[1].Inverted() || axes[0].DeadZone() != 0.2 {
			t.Errorf("profile %d axis configs did not round-trip: %+v", i, axes)
		}
	}

	// Re-marshaling the parsed profiles must produce the same document, proving the wire form is
	// canonical and stable.
	data2, err := gamepadprofile.MarshalProfiles(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(data2) {
		t.Errorf("marshaled documents differ after a round trip\nfirst:\n%s\nthen:\n%s", data, data2)
	}
}

func TestMarshalRejectsNilProfile(t *testing.T) {
	_, err := gamepadprofile.MarshalProfiles([]*gamepadprofile.GamepadMappingProfile{nil})
	var fe *gamepadprofile.FieldError
	if !errors.As(err, &fe) {
		t.Fatalf("error %v is not a *FieldError", err)
	}
	if fe.Path != "profiles[0]" {
		t.Errorf("field path = %q; want profiles[0]", fe.Path)
	}
}

func TestUnmarshalFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "profiles_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	profiles, err := gamepadprofile.UnmarshalProfiles(data)
	if err != nil {
		t.Fatalf("UnmarshalProfiles on the v1 fixture returned an error: %v", err)
	}
	if len(profiles) != 2 {
		t.Fatalf("profile count = %d; want 2", len(profiles))
	}
	if got := profiles[0].DeviceGUID(); got != "0000000000000000000000000000a001" {
		t.Errorf("first GUID = %q; want the fixture's GUID", got)
	}
	if got := profiles[0].AppendAxisConfigs(nil); len(got) != 4 {
		t.Errorf("first profile axis config count = %d; want 4", len(got))
	}
}

func TestUnmarshalLegacyVersionFallsBack(t *testing.T) {
	// Seed the registry so a failed import can be proven not to touch it.
	if err := gamepadprofile.SetProfile(validProfile(t, "seed-guid")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(gamepadprofile.Reset)

	data, err := os.ReadFile(filepath.Join("testdata", "profiles_v0_legacy.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := gamepadprofile.LoadProfiles(data); err == nil {
		t.Fatal("LoadProfiles on a version-0 document returned nil error; want an incompatibility error")
	} else {
		var fe *gamepadprofile.FieldError
		if !errors.As(err, &fe) {
			t.Fatalf("error %v is not a *GamepadFieldError", err)
		}
		if fe.Path != "version" {
			t.Errorf("field path = %q; want %q", fe.Path, "version")
		}
	}

	// The existing registry must survive the failed load.
	if gamepadprofile.Profile("seed-guid") == nil {
		t.Error("the legacy import replaced the registry; want it left untouched")
	}
	if got := gamepadprofile.AppendDeviceGUIDs(nil); len(got) != 1 || got[0] != "seed-guid" {
		t.Errorf("registry GUIDs after failed load = %v; want only the seed", got)
	}
}

func TestUnmarshalBadJSON(t *testing.T) {
	tests := []struct {
		name    string
		data    string
		wantMsg string
	}{
		{name: "syntax error", data: "{not json", wantMsg: "invalid JSON"},
		{name: "empty", data: "", wantMsg: "invalid JSON"},
		{name: "unknown field", data: `{"version":1,"profiles":[],"surprise":true}`, wantMsg: "unknown field"},
		{name: "wrong field type", data: `{"version":"one","profiles":[]}`, wantMsg: "invalid JSON"},
		{name: "trailing data", data: `{"version":1,"profiles":[]} garbage`, wantMsg: "trailing data"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := gamepadprofile.UnmarshalProfiles([]byte(test.data))
			if err == nil {
				t.Fatal("UnmarshalProfiles returned nil error; want an error")
			}
			var fe *gamepadprofile.FieldError
			if !errors.As(err, &fe) {
				t.Fatalf("error %v is not a *GamepadFieldError", err)
			}
			if !strings.Contains(err.Error(), test.wantMsg) {
				t.Errorf("error %q does not contain %q", err.Error(), test.wantMsg)
			}
		})
	}
}

func TestUnmarshalValidationErrorKeepsContext(t *testing.T) {
	// The second profile duplicates a binding source; the error must name its position in the
	// document.
	data := `{
		"version": 1,
		"profiles": [
			{
				"device_guid": "g1",
				"actions": ["a", "b"],
				"bindings": [
					{"kind": "button", "index": 0, "target": "a"},
					{"kind": "button", "index": 0, "target": "b"}
				]
			}
		]
	}`
	_, err := gamepadprofile.UnmarshalProfiles([]byte(data))
	if err == nil {
		t.Fatal("UnmarshalProfiles returned nil error; want a duplicate-binding error")
	}
	var fe *gamepadprofile.FieldError
	if !errors.As(err, &fe) {
		t.Fatalf("error %v is not a *GamepadFieldError", err)
	}
	if !strings.HasPrefix(fe.Path, "profiles[0].bindings[1]") {
		t.Errorf("field path = %q; want a profiles[0].bindings[1] path", fe.Path)
	}
}

func TestUnmarshalUnknownBindingKind(t *testing.T) {
	data := `{
		"version": 1,
		"profiles": [
			{"device_guid": "g", "actions": ["a"], "bindings": [{"kind": "pedal", "index": 0, "target": "a"}]}
		]
	}`
	_, err := gamepadprofile.UnmarshalProfiles([]byte(data))
	var fe *gamepadprofile.FieldError
	if !errors.As(err, &fe) {
		t.Fatalf("error %v is not a *GamepadFieldError", err)
	}
	if fe.Path != "profiles[0].bindings[0].kind" {
		t.Errorf("field path = %q; want profiles[0].bindings[0].kind", fe.Path)
	}
}
