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

package audio

import (
	"math"
	"time"
)

// RouteRegistryForTesting returns the virtual registry backing the current context's route, or
// nil when the route has not been created.
func RouteRegistryForTesting() *VirtualDeviceRegistry {
	if deviceOpenerForTesting == nil {
		return nil
	}
	reg, _ := deviceOpenerForTesting.(*VirtualDeviceRegistry)
	return reg
}

// RouteSnapshotForTesting exposes the internal snapshot for white-box assertions.
func RouteSnapshotForTesting() AudioDeviceRouteSnapshot {
	c := CurrentContext()
	if c == nil || c.route == nil {
		return AudioDeviceRouteSnapshot{}
	}
	return c.route.Snapshot()
}

// SetRouteFadeDurationForTesting sets the route fade duration without going through the public
// once-guard, for tests that build the route explicitly.
func SetRouteFadeDurationForTesting(d time.Duration) {
	c := CurrentContext()
	if c == nil || c.route == nil {
		return
	}
	c.route.SetFadeDuration(d)
}

// RouteActiveDeviceIDForTesting returns the active device ID, or empty.
func RouteActiveDeviceIDForTesting() string {
	c := CurrentContext()
	if c == nil || c.route == nil {
		return ""
	}
	d := c.route.CurrentDevice()
	if d == nil {
		return ""
	}
	return d.ID
}

// RouteKnownPlayersForTesting reports how many players the router currently tracks.
func RouteKnownPlayersForTesting() int {
	c := CurrentContext()
	if c == nil {
		return 0
	}
	c.playerFactory.m.Lock()
	router := c.playerFactory.router
	c.playerFactory.m.Unlock()
	if router == nil {
		return 0
	}
	router.mu.Lock()
	n := len(router.knownPlayers)
	router.mu.Unlock()
	return n
}

// PlayerIsRoutedForTesting reports whether a player is bound to a routed device.
func PlayerIsRoutedForTesting(p *Player) bool {
	p.p.m.Lock()
	defer p.p.m.Unlock()
	return p.p.routed != nil
}

// PullRoutedFramesForTesting pulls n frames from a player's current routed stream, exercising the
// gain and resampling chain on pull-based virtual devices. It returns the number of bytes read.
func PullRoutedFramesForTesting(p *Player, n int) int {
	p.p.m.Lock()
	rp := p.p.routed
	p.p.m.Unlock()
	if rp == nil {
		return 0
	}
	buf := make([]byte, n*8)
	got, _ := rp.stream.Read(buf)
	return got
}

// CapturedFramesForTesting returns the frames a virtual device captured.
func CapturedFramesForTesting(id string) []float32 {
	reg := CurrentVirtualDeviceRegistryForTesting()
	if reg == nil {
		return nil
	}
	return reg.CapturedFrames(id)
}

// CapturePullForTesting reads n frames from the player's routed stream and also records them on
// the device's capture buffer, so frame-reference assertions see the exact produced samples.
func CapturePullForTesting(p *Player, deviceID string, n int) int {
	p.p.m.Lock()
	rp := p.p.routed
	p.p.m.Unlock()
	if rp == nil {
		return 0
	}
	reg := CurrentVirtualDeviceRegistryForTesting()
	buf := make([]byte, n*8)
	got, _ := rp.stream.Read(buf)
	if reg != nil && got > 0 {
		frames := make([]float32, got/4)
		for i := range frames {
			bits := uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 |
				uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24
			frames[i] = math.Float32frombits(bits)
		}
		if d := reg.deviceForID(deviceID); d != nil {
			reg.capture(d, frames)
		}
	}
	return got
}
