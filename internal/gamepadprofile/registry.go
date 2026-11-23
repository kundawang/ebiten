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
	"slices"
	"sync"
)

// registry is the process-wide profile store. Profiles are keyed by device SDL GUID.
type registry struct {
	m        sync.RWMutex
	profiles map[string]*GamepadMappingProfile
}

var theRegistry registry

// SetProfile installs or replaces the profile for its device, taking effect immediately for every
// later query even while the game is running. The stored value is a private copy, so mutating the
// argument after the call does not change the registry.
//
// SetProfile is concurrent-safe.
func SetProfile(profile *GamepadMappingProfile) error {
	if profile == nil {
		return &FieldError{Path: "profile", Err: fmt.Errorf("gamepadprofile: profile must not be nil")}
	}
	if err := profile.validate(); err != nil {
		return err
	}
	theRegistry.set(profile)
	return nil
}

func (r *registry) set(profile *GamepadMappingProfile) {
	r.m.Lock()
	defer r.m.Unlock()

	if r.profiles == nil {
		r.profiles = map[string]*GamepadMappingProfile{}
	}
	r.profiles[profile.deviceGUID] = profile.clone()
}

// Profile returns a private copy of the profile registered for the device GUID, or nil when none
// exists. The returned value can be changed freely; it never aliases the registry's profile.
//
// Profile is concurrent-safe.
func Profile(deviceGUID string) *GamepadMappingProfile {
	return theRegistry.profile(deviceGUID)
}

func (r *registry) profile(deviceGUID string) *GamepadMappingProfile {
	r.m.RLock()
	defer r.m.RUnlock()

	if p, ok := r.profiles[deviceGUID]; ok {
		return p.clone()
	}
	return nil
}

// DeleteProfile removes the profile for the device GUID and reports whether one was present. A
// device whose profile is removed falls back to its default mapping on its next query.
//
// DeleteProfile is concurrent-safe.
func DeleteProfile(deviceGUID string) bool {
	return theRegistry.delete(deviceGUID)
}

func (r *registry) delete(deviceGUID string) bool {
	r.m.Lock()
	defer r.m.Unlock()

	if _, ok := r.profiles[deviceGUID]; !ok {
		return false
	}
	delete(r.profiles, deviceGUID)
	return true
}

// Reset removes every registered profile, restoring the default mappings.
//
// Reset is concurrent-safe.
func Reset() {
	theRegistry.reset()
}

func (r *registry) reset() {
	r.m.Lock()
	defer r.m.Unlock()

	clear(r.profiles)
}

// AppendDeviceGUIDs appends the GUIDs that currently have a profile to dst and returns the
// extended slice, in sorted order. The caller owns the returned slice.
//
// AppendDeviceGUIDs is concurrent-safe.
func AppendDeviceGUIDs(dst []string) []string {
	return theRegistry.appendDeviceGUIDs(dst)
}

func (r *registry) appendDeviceGUIDs(dst []string) []string {
	r.m.RLock()
	defer r.m.RUnlock()

	for guid := range r.profiles {
		dst = append(dst, guid)
	}
	slices.Sort(dst[len(dst)-len(r.profiles):])
	return dst
}

// AppendProfiles appends private copies of every registered profile to dst and returns the
// extended slice, ordered by device GUID.
//
// AppendProfiles is concurrent-safe.
func AppendProfiles(dst []*GamepadMappingProfile) []*GamepadMappingProfile {
	return theRegistry.appendProfiles(dst)
}

func (r *registry) appendProfiles(dst []*GamepadMappingProfile) []*GamepadMappingProfile {
	r.m.RLock()
	defer r.m.RUnlock()

	guids := make([]string, 0, len(r.profiles))
	for guid := range r.profiles {
		guids = append(guids, guid)
	}
	slices.Sort(guids)
	for _, guid := range guids {
		dst = append(dst, r.profiles[guid].clone())
	}
	return dst
}

// profileForDevice returns the registered profile for the GUID without copying it. The caller must
// not mutate the result and must only read it while holding the registry's read lock.
func (r *registry) profileForDevice(deviceGUID string) *GamepadMappingProfile {
	r.m.RLock()
	defer r.m.RUnlock()

	return r.profiles[deviceGUID]
}

// LoadProfiles parses the versioned JSON document and atomically replaces every registered
// profile with the ones it contains.
//
// On a parse or validation error the existing registry is left untouched, so a bad document never
// leaves a half-loaded configuration. A successful load with an empty profile list clears the
// registry.
//
// LoadProfiles is concurrent-safe.
func LoadProfiles(data []byte) error {
	profiles, err := UnmarshalProfiles(data)
	if err != nil {
		return err
	}
	theRegistry.replaceAll(profiles)
	return nil
}

func (r *registry) replaceAll(profiles []*GamepadMappingProfile) {
	r.m.Lock()
	defer r.m.Unlock()

	next := make(map[string]*GamepadMappingProfile, len(profiles))
	for _, p := range profiles {
		next[p.deviceGUID] = p.clone()
	}
	r.profiles = next
}

// ExportProfiles serializes every registered profile as a versioned JSON document.
//
// ExportProfiles is concurrent-safe.
func ExportProfiles() ([]byte, error) {
	return MarshalProfiles(AppendProfiles(nil))
}
