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

package ebiten_test

// This smoke test exercises only the pure, non-graphical surface of the gamepad mapping profile
// API. It never opens a window or starts the run loop, so it does not need a virtual display, but
// it lives in the root package's test binary and is therefore run under xvfb-run like the rest of
// the package on Linux.

import (
	"errors"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
)

func TestGamepadMappingProfilePublicAPISmoke(t *testing.T) {
	t.Cleanup(ebiten.ResetGamepadMappingProfiles)

	profile, err := ebiten.NewGamepadMappingProfile(
		"0000000000000000000000000000c001",
		"Smoke Pad",
		[]string{"jump", "roll"},
		[]ebiten.GamepadBinding{
			ebiten.NewGamepadButtonBinding(0, "jump"),
			ebiten.NewGamepadHalfAxisBinding(1, true, "roll"),
			ebiten.NewGamepadHatBinding(0, 1, "jump"),
		},
		[]ebiten.GamepadChordBinding{
			ebiten.NewGamepadChordBinding([]int{0, 1}, "roll"),
		},
		[]ebiten.GamepadAxisConfig{
			ebiten.NewGamepadAxisConfig(0, ebiten.GamepadLogicalAxisLeftStickHorizontal, 0.2, false),
		},
	)
	if err != nil {
		t.Fatalf("NewGamepadMappingProfile returned an error: %v", err)
	}
	if err := ebiten.SetGamepadMappingProfile(profile); err != nil {
		t.Fatalf("SetGamepadMappingProfile returned an error: %v", err)
	}

	// The registry is keyed by GUID and returns private copies.
	got := ebiten.GamepadMappingProfileForDevice("0000000000000000000000000000c001")
	if got == nil || got.Name() != "Smoke Pad" {
		t.Fatalf("GamepadMappingProfileForDevice = %v; want the registered profile", got)
	}
	guids := ebiten.AppendGamepadMappingDeviceGUIDs(nil)
	if len(guids) != 1 || guids[0] != "0000000000000000000000000000c001" {
		t.Errorf("AppendGamepadMappingDeviceGUIDs = %v; want the one registered GUID", guids)
	}
	if ebiten.AppendGamepadMappingProfiles(nil)[0].DeviceGUID() != got.DeviceGUID() {
		t.Error("AppendGamepadMappingProfiles did not return the registered profile")
	}

	// Export then atomically reload the versioned document.
	data, err := ebiten.ExportGamepadMappingProfiles()
	if err != nil {
		t.Fatalf("ExportGamepadMappingProfiles returned an error: %v", err)
	}
	if !strings.Contains(string(data), `"version": 1`) {
		t.Errorf("exported document is missing the version field:\n%s", data)
	}
	ebiten.ResetGamepadMappingProfiles()
	if len(ebiten.AppendGamepadMappingDeviceGUIDs(nil)) != 0 {
		t.Error("ResetGamepadMappingProfiles did not clear the registry")
	}
	if err := ebiten.LoadGamepadMappingProfiles(data); err != nil {
		t.Fatalf("LoadGamepadMappingProfiles returned an error: %v", err)
	}
	if ebiten.GamepadMappingProfileForDevice("0000000000000000000000000000c001") == nil {
		t.Error("the profile did not survive an export/import cycle")
	}

	// A bad document is rejected with a field-context error and leaves the registry intact.
	bad := []byte(`{"version":1,"profiles":[{"device_guid":"","actions":[],"bindings":[]}]}`)
	err = ebiten.LoadGamepadMappingProfiles(bad)
	var fieldErr *ebiten.GamepadFieldError
	if !errors.As(err, &fieldErr) {
		t.Fatalf("LoadGamepadMappingProfiles error %v is not a *GamepadFieldError", err)
	}
	if !strings.Contains(fieldErr.Path, "device_guid") {
		t.Errorf("field path = %q; want a device_guid path", fieldErr.Path)
	}
	if ebiten.GamepadMappingProfileForDevice("0000000000000000000000000000c001") == nil {
		t.Error("the failed import removed the existing profile; want it untouched")
	}

	// Deletion falls back to the default.
	if !ebiten.DeleteGamepadMappingProfile("0000000000000000000000000000c001") {
		t.Error("DeleteGamepadMappingProfile = false; want true")
	}
	if ebiten.GamepadMappingProfileForDevice("0000000000000000000000000000c001") != nil {
		t.Error("the profile survived deletion")
	}

	// A disconnected gamepad ID is a safe default fallback.
	if ebiten.HasGamepadMappingProfile(999) {
		t.Error("HasGamepadMappingProfile(999) = true; want false for a disconnected gamepad")
	}
	if ebiten.IsGamepadActionPressed(999, "jump") {
		t.Error("IsGamepadActionPressed(999, ...) = true; want false")
	}
	if got := ebiten.GamepadActionValue(999, "jump"); got != 0 {
		t.Errorf("GamepadActionValue(999, ...) = %v; want 0", got)
	}
}
