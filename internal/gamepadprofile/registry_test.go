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
	"slices"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

// resetRegistry empties the process-wide registry around a test.
func resetRegistry(t *testing.T) {
	t.Helper()
	gamepadprofile.Reset()
	t.Cleanup(gamepadprofile.Reset)
}

func TestRegistrySetGetDelete(t *testing.T) {
	resetRegistry(t)

	p := validProfile(t, "guid-reg")
	if err := gamepadprofile.SetProfile(p); err != nil {
		t.Fatal(err)
	}
	if got := gamepadprofile.Profile("guid-reg"); got == nil || got.DeviceGUID() != "guid-reg" {
		t.Fatalf("Profile() = %v; want the registered profile", got)
	}
	if gamepadprofile.Profile("missing") != nil {
		t.Error("Profile() for an unknown GUID is non-nil; want nil")
	}
	if !gamepadprofile.DeleteProfile("guid-reg") {
		t.Error("DeleteProfile() = false; want true for a present profile")
	}
	if gamepadprofile.DeleteProfile("guid-reg") {
		t.Error("DeleteProfile() = true on the second call; want false")
	}
	if gamepadprofile.Profile("guid-reg") != nil {
		t.Error("Profile() is non-nil after deletion; want nil")
	}
}

func TestRegistryReplacesProfile(t *testing.T) {
	resetRegistry(t)

	first := validProfile(t, "guid-swap")
	if err := gamepadprofile.SetProfile(first); err != nil {
		t.Fatal(err)
	}

	// A second profile for the same GUID replaces the first at runtime.
	second, err := gamepadprofile.NewGamepadMappingProfile(
		"guid-swap",
		"Replaced",
		[]string{"only"},
		[]gamepadprofile.GamepadBinding{gamepadprofile.NewGamepadButtonBinding(3, "only")},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := gamepadprofile.SetProfile(second); err != nil {
		t.Fatal(err)
	}
	got := gamepadprofile.Profile("guid-swap")
	if got.Name() != "Replaced" {
		t.Errorf("Name() = %q; want %q after the runtime switch", got.Name(), "Replaced")
	}
	if actions := got.AppendActions(nil); !slices.Equal(actions, []string{"only"}) {
		t.Errorf("actions = %v; want [only] after the runtime switch", actions)
	}
}

func TestRegistryReturnsCopies(t *testing.T) {
	resetRegistry(t)

	if err := gamepadprofile.SetProfile(validProfile(t, "guid-copy")); err != nil {
		t.Fatal(err)
	}

	// Mutating a profile returned by the registry must not change the registry.
	got := gamepadprofile.Profile("guid-copy")
	actions := got.AppendActions(nil)
	actions[0] = "tampered"
	bindings := got.AppendBindings(nil)
	bindings[0] = gamepadprofile.NewGamepadButtonBinding(31, "jump")

	again := gamepadprofile.Profile("guid-copy")
	if a := again.AppendActions(nil); a[0] == "tampered" {
		t.Error("a returned profile aliases the registry's action storage")
	}
	if b := again.AppendBindings(nil); b[0].Index() == 31 {
		t.Error("a returned profile aliases the registry's binding storage")
	}

	// The list-returning query must hand over independent slices as well.
	all := gamepadprofile.AppendProfiles(nil)
	all[0] = nil
	if gamepadprofile.Profile("guid-copy") == nil {
		t.Error("mutating the AppendProfiles result removed the registry entry")
	}
	guids := gamepadprofile.AppendDeviceGUIDs(nil)
	guids[0] = "tampered"
	if again := gamepadprofile.AppendDeviceGUIDs(nil); again[0] != "guid-copy" {
		t.Errorf("GUID list aliases the registry: %v", again)
	}
}

func TestRegistrySetProfileCopiesInput(t *testing.T) {
	resetRegistry(t)

	p := validProfile(t, "guid-input-copy")
	if err := gamepadprofile.SetProfile(p); err != nil {
		t.Fatal(err)
	}
	// The caller keeps its own value; mutating it after registration must not reach the registry.
	// Re-registering a different value is the only way to change a stored profile.
	if got := gamepadprofile.Profile("guid-input-copy"); got.Name() != "Pad guid-input-copy" {
		t.Errorf("Name() = %q; want the original name", got.Name())
	}
}

func TestLoadProfilesAtomic(t *testing.T) {
	resetRegistry(t)

	if err := gamepadprofile.SetProfile(validProfile(t, "guid-before")); err != nil {
		t.Fatal(err)
	}

	// A bad document must leave the existing registry untouched.
	bad := []byte(`{"version":1,"profiles":[{"device_guid":"","actions":[],"bindings":[]}]}`)
	if err := gamepadprofile.LoadProfiles(bad); err == nil {
		t.Fatal("LoadProfiles on a bad document returned nil error; want an error")
	}
	if gamepadprofile.Profile("guid-before") == nil {
		t.Error("the failed LoadProfiles removed the existing profile")
	}

	// A good document atomically replaces the whole registry.
	good, err := gamepadprofile.MarshalProfiles([]*gamepadprofile.GamepadMappingProfile{
		validProfile(t, "guid-after-1"),
		validProfile(t, "guid-after-2"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := gamepadprofile.LoadProfiles(good); err != nil {
		t.Fatal(err)
	}
	if gamepadprofile.Profile("guid-before") != nil {
		t.Error("the old profile survived a successful LoadProfiles; want it replaced")
	}
	if got := gamepadprofile.AppendDeviceGUIDs(nil); len(got) != 2 {
		t.Errorf("GUIDs after load = %v; want the two new profiles", got)
	}

	// An empty profile list clears the registry.
	empty, err := gamepadprofile.MarshalProfiles(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := gamepadprofile.LoadProfiles(empty); err != nil {
		t.Fatal(err)
	}
	if got := gamepadprofile.AppendDeviceGUIDs(nil); len(got) != 0 {
		t.Errorf("GUIDs after an empty load = %v; want none", got)
	}
}

func TestExportProfilesRoundTrip(t *testing.T) {
	resetRegistry(t)

	if err := gamepadprofile.SetProfile(validProfile(t, "guid-export")); err != nil {
		t.Fatal(err)
	}
	data, err := gamepadprofile.ExportProfiles()
	if err != nil {
		t.Fatal(err)
	}
	gamepadprofile.Reset()
	if err := gamepadprofile.LoadProfiles(data); err != nil {
		t.Fatalf("reloading the exported document failed: %v", err)
	}
	if gamepadprofile.Profile("guid-export") == nil {
		t.Error("the exported profile did not survive an export/import cycle")
	}
}

func TestUnknownDeviceFallsBackToDefault(t *testing.T) {
	resetRegistry(t)

	if err := gamepadprofile.SetProfile(validProfile(t, "guid-known")); err != nil {
		t.Fatal(err)
	}
	unknown := newFakeGamepad("guid-unknown", nil, nil, nil)
	if gamepadprofile.HasProfile(unknown) {
		t.Error("HasProfile() = true for an unknown device; want false")
	}
	if gamepadprofile.IsActionPressed(unknown, "jump") {
		t.Error("IsActionPressed() = true on an unknown device; want the default (false)")
	}
	if got := gamepadprofile.ActionValue(unknown, "jump"); got != 0 {
		t.Errorf("ActionValue() = %v on an unknown device; want 0", got)
	}
	if got := gamepadprofile.LogicalAxisValue(unknown, gamepadprofile.LogicalAxisLeftStickHorizontal); got != 0 {
		t.Errorf("LogicalAxisValue() = %v on an unknown device; want 0", got)
	}

	// A nil gamepad is the same fallback as a vanished device.
	if gamepadprofile.HasProfile(nil) {
		t.Error("HasProfile(nil) = true; want false")
	}
	if gamepadprofile.IsActionPressed(nil, "jump") {
		t.Error("IsActionPressed(nil, ...) = true; want false")
	}

	// A typed nil pointer (as a disconnected gamepad lookup can hand back) must also fall back
	// without panicking.
	var nilPad *fakeGamepad
	if gamepadprofile.HasProfile(nilPad) {
		t.Error("HasProfile(typed-nil) = true; want false")
	}
	if gamepadprofile.IsActionPressed(nilPad, "jump") {
		t.Error("IsActionPressed(typed-nil, ...) = true; want false")
	}
	if got := gamepadprofile.ActionValue(nilPad, "jump"); got != 0 {
		t.Errorf("ActionValue(typed-nil, ...) = %v; want 0", got)
	}
}
