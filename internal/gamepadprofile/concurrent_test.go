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
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/gamepadprofile"
)

// TestRegistryConcurrentUpdate exercises the registry under simultaneous readers, writers and
// profile swaps. Run with -race to detect data races; the test itself asserts only that nothing
// panics and the invariants stay observable.
func TestRegistryConcurrentUpdate(t *testing.T) {
	resetRegistry(t)

	guid := "guid-race"
	if err := gamepadprofile.SetProfile(validProfile(t, guid)); err != nil {
		t.Fatal(err)
	}
	g := newFakeGamepad(guid, []float64{0.5}, []bool{true}, nil)

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Writers keep replacing and deleting the profile.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = gamepadprofile.SetProfile(validProfile(t, guid))
			gamepadprofile.DeleteProfile(guid)
			_ = gamepadprofile.SetProfile(validProfile(t, guid))
		}
	}()

	// Readers evaluate actions against the sometimes-present profile.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = gamepadprofile.IsActionPressed(g, "jump")
			_ = gamepadprofile.ActionValue(g, "jump")
			_ = gamepadprofile.LogicalAxisValue(g, gamepadprofile.LogicalAxisLeftStickHorizontal)
			_ = gamepadprofile.HasProfile(g)
		}
	}()

	// Listers read the whole registry and mutate their own results.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			guids := gamepadprofile.AppendDeviceGUIDs(nil)
			for i := range guids {
				guids[i] = "tampered"
			}
			profiles := gamepadprofile.AppendProfiles(nil)
			for _, p := range profiles {
				if p != nil {
					_ = p.DeviceGUID()
				}
			}
		}
	}()

	// An importer swaps the whole registry atomically, including a deliberately bad document that
	// must be rejected without disturbing the other goroutines.
	wg.Add(1)
	go func() {
		defer wg.Done()
		good, err := gamepadprofile.MarshalProfiles([]*gamepadprofile.GamepadMappingProfile{validProfile(t, guid)})
		if err != nil {
			t.Error(err)
			return
		}
		bad := []byte(`{"version":1,"profiles":[{"device_guid":"","actions":[],"bindings":[]}]}`)
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = gamepadprofile.LoadProfiles(good)
			_ = gamepadprofile.LoadProfiles(bad)
		}
	}()

	// Let the goroutines contend, then stop and wait.
	for range 200 {
		_ = gamepadprofile.Profile(guid)
	}
	close(stop)
	wg.Wait()
}
