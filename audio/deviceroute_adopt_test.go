// Copyright 2026 The Ebitengine Authors
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

package audio_test

import (
	"testing"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// TestRouteMigratesPlayerCreatedBeforeRouting covers a player created and started on the
// context's initial driver before the route is touched. Enabling the route and then switching must
// keep the player playing and promote it onto a routed device rather than abandoning it.
func TestRouteMigratesPlayerCreatedBeforeRouting(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	defer teardownRoutingTest(t)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})

	// Create and start the player before obtaining the route. It binds to the context's initial
	// driver, the same driver a real context creates before routing is enabled.
	p, err := c.NewPlayerF32(newConstF32Source(0.5, 60, rate))
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	posBefore := p.Position()

	// Enabling the route and switching migrates the pre-routing player onto the new device.
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)
	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatalf("switch: %v", err)
	}

	if !audio.PlayerIsRoutedForTesting(p) {
		t.Fatalf("player was not promoted onto a routed device")
	}
	if !p.IsPlaying() {
		t.Fatalf("pre-routing player stopped after migration")
	}
	if got := p.Position(); got < posBefore {
		t.Fatalf("position moved backwards: before %v after %v", posBefore, got)
	}
	if got := route.Snapshot().CurrentID; got != "bt" {
		t.Fatalf("active device: got %q, want bt", got)
	}
	if got := reg.OpenCount("bt"); got != 1 {
		t.Fatalf("new device not open: %d", got)
	}
}
