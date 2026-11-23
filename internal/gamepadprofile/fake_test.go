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

// fakeGamepad is a RawGamepad driven entirely in memory, so evaluation tests need no platform
// backend.
type fakeGamepad struct {
	sdlID     string
	axes      []float64
	axesReady []bool
	buttons   []bool
	hats      []int
}

func (f *fakeGamepad) SDLID() string {
	return f.sdlID
}

func (f *fakeGamepad) AxisCount() int {
	return len(f.axes)
}

func (f *fakeGamepad) ButtonCount() int {
	return len(f.buttons)
}

func (f *fakeGamepad) HatCount() int {
	return len(f.hats)
}

func (f *fakeGamepad) IsAxisReady(axis int) bool {
	if axis < 0 || axis >= len(f.axesReady) {
		return false
	}
	return f.axesReady[axis]
}

func (f *fakeGamepad) Axis(axis int) float64 {
	if axis < 0 || axis >= len(f.axes) {
		return 0
	}
	return f.axes[axis]
}

func (f *fakeGamepad) Button(button int) bool {
	if button < 0 || button >= len(f.buttons) {
		return false
	}
	return f.buttons[button]
}

func (f *fakeGamepad) Hat(hat int) int {
	if hat < 0 || hat >= len(f.hats) {
		return 0
	}
	return f.hats[hat]
}

func newFakeGamepad(sdlID string, axes []float64, buttons []bool, hats []int) *fakeGamepad {
	ready := make([]bool, len(axes))
	for i := range ready {
		ready[i] = true
	}
	return &fakeGamepad{
		sdlID:     sdlID,
		axes:      axes,
		axesReady: ready,
		buttons:   buttons,
		hats:      hats,
	}
}
