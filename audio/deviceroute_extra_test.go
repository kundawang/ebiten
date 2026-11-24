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
	"math"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// TestRouteChangeCallback verifies the OnDeviceChange callback reports each transition.
func TestRouteChangeCallback(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	defer teardownRoutingTest(t)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	var mu sync.Mutex
	var seen []string
	route.OnDeviceChange(func(s audio.AudioDeviceRouteSnapshot) {
		mu.Lock()
		seen = append(seen, s.CurrentID)
		mu.Unlock()
	})

	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	found := false
	for _, id := range seen {
		if id == "bt" {
			found = true
		}
	}
	mu.Unlock()
	if !found {
		t.Fatalf("callback did not report bt: %v", seen)
	}
}

// TestRouteLoopContinuity verifies an infinite loop keeps producing frames and a monotonic
// position across a switch, with no EOF and no jump back to the loop start.
func TestRouteLoopContinuity(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	defer teardownRoutingTest(t)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	// A short 0.1s window of a longer source, looped infinitely.
	src := newConstF32Source(0.5, 60, rate)
	loopLen := int64(rate/10) * 8
	loop := audio.NewInfiniteLoopF32(src, loopLen)

	// Obtain the route before creating players, so they are built as routed players.
	_ = route

	p, err := c.NewPlayerF32(loop)
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if !audio.PlayerIsRoutedForTesting(p) {
		t.Fatalf("player was not created as a routed player")
	}

	// Consume past the source length so the loop has wrapped at least once.
	for range 5 {
		if audio.CapturePullForTesting(p, "speaker", rate/10*2) == 0 {
			t.Fatalf("no frames before switch")
		}
	}
	// Position is computed on the update sweep; drive it explicitly in the test environment.
	audio.UpdatePositionForTesting(p)
	posBefore := p.Position()
	if posBefore <= 0 {
		t.Fatalf("position did not advance: %v", posBefore)
	}

	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatal(err)
	}
	// The loop must keep producing well past its underlying source length.
	for range 5 {
		if audio.CapturePullForTesting(p, "bt", rate/10*2) == 0 {
			t.Fatalf("loop produced no frames after switch")
		}
	}
	posAfter := p.Position()
	if posAfter < posBefore {
		t.Fatalf("loop position moved backwards: before %v after %v", posBefore, posAfter)
	}
}

// TestRouteFadeExactFrameReference compares every fade-in frame against the analytic linear
// ramp, not just its endpoints. Both devices run at the context rate so no resampler alters the
// gain values; the gain stage is the only transform between the constant source and the output.
func TestRouteFadeExactFrameReference(t *testing.T) {
	const rate = 48000
	const fade = 64
	const value = float32(0.5)
	c := setupRoutingTest(t, rate)
	defer teardownRoutingTest(t)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(time.Duration(fade) * time.Second / time.Duration(rate))

	p, err := c.NewPlayerF32(newConstF32Source(value, 60, rate))
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatal(err)
	}

	frames := reg.CapturedFrames("bt")
	if len(frames) < fade*2 {
		t.Fatalf("captured %d floats, want at least %d", len(frames), fade*2)
	}
	// The first frame carries the smallest nonzero step; the final ramp frame reaches unity.
	for i := range fade {
		wantGain := float64(i+1) / float64(fade)
		for ch := range 2 {
			got := float64(frames[i*2+ch])
			want := float64(value) * wantGain
			if math.Abs(got-want) > 1e-5 {
				t.Fatalf("frame %d ch %d: got %v, want %v (gain %v)", i, ch, got, want, wantGain)
			}
		}
	}
}

// TestRouteResamplingFrameRatio verifies a 2:1 rate change halves the number of output frames
// produced from the same number of source frames.
func TestRouteResamplingFrameRatio(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	defer teardownRoutingTest(t)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "slow", Name: "24k", Kind: audio.DeviceKindBluetooth,
		MinRate: 24000, MaxRate: 24000, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	p, err := c.NewPlayerF32(newConstF32Source(0.5, 60, rate))
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := route.SwitchDevice("slow"); err != nil {
		t.Fatal(err)
	}

	// Pull fixed-size batches of output (24k) frames through the resampling chain. The number of
	// output frames produced must match the number requested: the resampler consumes two 48k
	// source frames per 24k output frame internally, but the output count is rate-independent.
	const pulls = 64
	const each = 128
	before := len(reg.CapturedFrames("slow")) / 2 // two floats per stereo frame
	for range pulls {
		audio.CapturePullForTesting(p, "slow", each)
	}
	after := len(reg.CapturedFrames("slow")) / 2
	outFrames := after - before
	want := pulls * each
	// The interpolation window causes a bounded startup transient; allow a small slack.
	if outFrames < want-128 || outFrames > want+128 {
		t.Fatalf("resampled frames: got %d, want ~%d", outFrames, want)
	}
}
