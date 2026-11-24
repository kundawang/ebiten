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
	"errors"
	"io"
	"math"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

// constF32Source is a seekable float32 stereo source whose every sample equals value. It is long
// enough for any test in this file and reports a known byte length.
type constF32Source struct {
	value float32
	pos   int64
	size  int64
}

func newConstF32Source(value float32, seconds int, sampleRate int) *constF32Source {
	return &constF32Source{
		value: value,
		size:  int64(seconds) * int64(sampleRate) * 8,
	}
}

func (s *constF32Source) Read(p []byte) (int, error) {
	if s.pos >= s.size {
		return 0, io.EOF
	}
	n := len(p)
	if int64(n) > s.size-s.pos {
		n = int(s.size - s.pos)
	}
	bits := math.Float32bits(s.value)
	for i := range n {
		p[i] = byte(bits >> ((i % 4) * 8))
	}
	s.pos += int64(n)
	return n, nil
}

func (s *constF32Source) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
		s.pos = offset
	case io.SeekCurrent:
		s.pos += offset
	case io.SeekEnd:
		s.pos = s.size + offset
	}
	return s.pos, nil
}

// setupRoutingTest builds a context backed by a fresh virtual device registry. The registry is
// reachable through audio.CurrentVirtualDeviceRegistryForTesting.
func setupRoutingTest(t *testing.T, sampleRate int) *audio.Context {
	t.Helper()
	audio.NewVirtualDeviceRegistryForTesting()
	c := audio.NewContext(sampleRate)
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatalf("UpdateForTesting: %v", err)
	}
	return c
}

func teardownRoutingTest(t *testing.T) {
	t.Helper()
	audio.ResetContextForTesting()
	audio.ResetVirtualDeviceRegistryForTesting()
}

// nearlyEqual compares two float32 gains within an absolute tolerance.
func nearlyEqual(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestRouteEnumerationAndSnapshot(t *testing.T) {
	c := setupRoutingTest(t, 48000)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)

	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: 44100, ChannelCounts: []int{2},
	})

	route := c.AudioDeviceRoute()
	devs := route.AvailableDevices()
	if len(devs) != 2 {
		t.Fatalf("AvailableDevices count: got %d, want 2", len(devs))
	}
	// Sorted by ID: bt then speaker.
	if devs[0].ID != "bt" || devs[1].ID != "speaker" {
		t.Fatalf("AvailableDevices order: got %q, %q", devs[0].ID, devs[1].ID)
	}
	if devs[1].Kind != audio.DeviceKindSpeaker || !devs[1].IsDefault {
		t.Fatalf("default device wrong: %+v", devs[1])
	}

	snap := route.Snapshot()
	if snap.State != audio.RouteStateActive || snap.CurrentID != "speaker" {
		t.Fatalf("snapshot: %+v", snap)
	}
	if snap.Generation == 0 {
		t.Fatalf("generation should advance")
	}

	cur := route.CurrentDevice()
	if cur == nil || cur.ID != "speaker" {
		t.Fatalf("CurrentDevice: %+v", cur)
	}
}

func TestRouteSwitchPreservesPositionLoopVolume(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	src := newConstF32Source(0.5, 60, rate)
	p, err := c.NewPlayerF32(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetPosition(250 * time.Millisecond); err != nil {
		t.Fatal(err)
	}
	p.SetVolume(0.7)
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	posBefore := p.Position()
	volBefore := p.Volume()

	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatalf("SwitchDevice: %v", err)
	}

	if got := route.Snapshot().CurrentID; got != "bt" {
		t.Fatalf("active after switch: got %q, want bt", got)
	}
	if !p.IsPlaying() {
		t.Fatalf("player should still be playing after switch")
	}
	if got := p.Volume(); got != volBefore {
		t.Fatalf("volume changed: got %v, want %v", got, volBefore)
	}
	// Position must not jump backwards and should stay within a small window of the pre-switch value.
	posAfter := p.Position()
	if posAfter < posBefore {
		t.Fatalf("position moved backwards: before %v, after %v", posBefore, posAfter)
	}
	if posAfter-posBefore > 100*time.Millisecond {
		t.Fatalf("position jumped too far: before %v, after %v", posBefore, posAfter)
	}

	// The old device's handle must be closed; the new one open.
	if got := reg.OpenCount("speaker"); got != 0 {
		t.Fatalf("speaker handle leaked: %d open", got)
	}
	if got := reg.OpenCount("bt"); got != 1 {
		t.Fatalf("bt handle not open: %d", got)
	}
	if reg.TotalOpens() != reg.TotalCloses()+1 {
		t.Fatalf("open/close imbalance: opens %d closes %d", reg.TotalOpens(), reg.TotalCloses())
	}
}

func TestRouteFadeGainRamp(t *testing.T) {
	const rate = 48000
	const fade = 100 // frames
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(time.Duration(fade) * time.Second / time.Duration(rate))

	src := newConstF32Source(0.5, 60, rate)
	p, err := c.NewPlayerF32(src)
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatalf("SwitchDevice: %v", err)
	}

	// The captured fade-in on the new device must start near silence and reach unity.
	frames := reg.CapturedFrames("bt")
	if len(frames) < fade*2 {
		t.Fatalf("captured too few frames: %d", len(frames))
	}
	// First frame is the smallest nonzero step (1/fade of the source value), not exact silence.
	firstWant := float32(0.5) / float32(fade)
	if !nearlyEqual(frames[0], firstWant, 0.02) || !nearlyEqual(frames[1], firstWant, 0.02) {
		t.Fatalf("fade-in first step: got %v, want ~%v", frames[0], firstWant)
	}
	// The last fade frame reaches unity gain: the full source value.
	last := frames[(fade-1)*2]
	if !nearlyEqual(last, 0.5, 0.02) {
		t.Fatalf("fade-in end gain: got %v, want ~0.5", last)
	}
}

func TestRouteUnknownAndUnavailableAndDenied(t *testing.T) {
	c := setupRoutingTest(t, 48000)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "denied", Name: "Restricted", Kind: audio.DeviceKindUSB,
		SampleRate: 48000, ChannelCounts: []int{2}, Denied: true,
	})
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "off", Name: "Docked", Kind: audio.DeviceKindHDMI,
		SampleRate: 48000, ChannelCounts: []int{2}, IsUnavailable: true,
	})
	route := c.AudioDeviceRoute()

	if err := route.SwitchDevice("nope"); !errors.Is(err, audio.ErrAudioDeviceNotFound) {
		t.Fatalf("unknown id: got %v, want ErrAudioDeviceNotFound", err)
	}
	if err := route.SwitchDevice("denied"); !errors.Is(err, audio.ErrAudioDevicePermissionDenied) {
		t.Fatalf("denied: got %v, want ErrAudioDevicePermissionDenied", err)
	}
	if err := route.SwitchDevice("off"); !errors.Is(err, audio.ErrNoAudioDevice) {
		t.Fatalf("unavailable: got %v, want ErrNoAudioDevice", err)
	}
	// The unavailable selection is remembered and the current device is unchanged.
	if got := route.Snapshot().CurrentID; got != "speaker" {
		t.Fatalf("current device changed on failed switch: %q", got)
	}
	if got := route.Snapshot().PendingID; got != "off" {
		t.Fatalf("pending selection: got %q, want off", got)
	}

	// Bringing the pending device online should migrate to it automatically.
	reg.SetAvailable("off", true)
	if got := route.Snapshot().CurrentID; got != "off" {
		t.Fatalf("after hot-plug current: got %q, want off", got)
	}
}

func TestRouteIncompatibleFormat(t *testing.T) {
	c := setupRoutingTest(t, 48000)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	// Mono-only device: the context is stereo and the router only supports stereo routing.
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "mono", Name: "Mono", Kind: audio.DeviceKindSpeaker,
		SampleRate: 48000, ChannelCounts: []int{1},
	})
	route := c.AudioDeviceRoute()
	err := route.SwitchDevice("mono")
	if !errors.Is(err, audio.ErrAudioDeviceIncompatible) {
		t.Fatalf("incompatible: got %v, want ErrAudioDeviceIncompatible", err)
	}
}

func TestRouteResamplingDifferentRate(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	// A 24kHz-only device forces resampling.
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "slow", Name: "24k", Kind: audio.DeviceKindBluetooth,
		MinRate: 24000, MaxRate: 24000, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	src := newConstF32Source(0.5, 60, rate)
	p, err := c.NewPlayerF32(src)
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := route.SwitchDevice("slow"); err != nil {
		t.Fatalf("switch to resampled device: %v", err)
	}
	if !p.IsPlaying() {
		t.Fatalf("player stopped after resampled switch")
	}
	// The virtual device is pull-based: explicitly pull frames through the resampling chain.
	if got := audio.CapturePullForTesting(p, "slow", 256); got == 0 {
		t.Fatalf("pull produced no bytes")
	}
	if frames := reg.CapturedFrames("slow"); len(frames) == 0 {
		t.Fatalf("no frames captured on resampled device")
	}
}

func TestRouteMultiplePlayersDeterministic(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	var players []*audio.Player
	for range 3 {
		p, err := c.NewPlayerF32(newConstF32Source(0.4, 60, rate))
		if err != nil {
			t.Fatal(err)
		}
		p.SetVolume(0.5)
		p.Play()
		players = append(players, p)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatalf("switch: %v", err)
	}
	for i, p := range players {
		if !p.IsPlaying() {
			t.Fatalf("player %d stopped", i)
		}
		if p.Volume() != 0.5 {
			t.Fatalf("player %d volume: got %v", i, p.Volume())
		}
	}
	if got := audio.RouteKnownPlayersForTesting(); got != 3 {
		t.Fatalf("known players: got %d, want 3", got)
	}
	if reg.OpenCount("speaker") != 0 || reg.OpenCount("bt") != 1 {
		t.Fatalf("handle state: speaker=%d bt=%d", reg.OpenCount("speaker"), reg.OpenCount("bt"))
	}
}

func TestRouteHotPlugRemovalFallback(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
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
	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatal(err)
	}

	// Pull the Bluetooth device: the route must fall back to the speaker.
	reg.RemoveDevice("bt")
	if got := route.Snapshot().CurrentID; got != "speaker" {
		t.Fatalf("fallback after removal: got %q, want speaker", got)
	}
	if !p.IsPlaying() {
		t.Fatalf("player stopped after fallback")
	}
}

func TestRouteDefaultDeviceChangeFollowed(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "dock", Name: "Dock", Kind: audio.DeviceKindHDMI,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)

	// An implicit route follows the default device change.
	reg.SetDefault("dock")
	if got := route.Snapshot().CurrentID; got != "dock" {
		t.Fatalf("follow default: got %q, want dock", got)
	}

	// An explicit selection is not overridden by a later default change.
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	if err := route.SwitchDevice("bt"); err != nil {
		t.Fatal(err)
	}
	reg.SetDefault("speaker")
	if got := route.Snapshot().CurrentID; got != "bt" {
		t.Fatalf("explicit route overridden: got %q, want bt", got)
	}
}

func TestRouteNoDeviceUnavailableThenRecover(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
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

	// Remove every device: the route goes unavailable. The player keeps its pending play request,
	// exactly as a player does before the first device is created, so it can resume automatically.
	reg.RemoveDevice("speaker")
	if got := route.Snapshot().State; got != audio.RouteStateUnavailable {
		t.Fatalf("state after all removed: got %v, want unavailable", got)
	}
	if got := audio.PlayersCountForTesting(); got != 0 {
		t.Fatalf("player should be untracked while no device exists: %d", got)
	}

	// Re-add a device: the pending player resumes.
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "speaker", Name: "Speaker", Kind: audio.DeviceKindSpeaker, IsDefault: true,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if got := route.Snapshot().State; got != audio.RouteStateActive {
		t.Fatalf("state after recover: got %v, want active", got)
	}
}

func TestRouteOpenErrorFallbackKeepsOldDevice(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
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

	reg.SetOpenError("bt", audio.ErrNoAudioDevice)
	if err := route.SwitchDevice("bt"); !errors.Is(err, audio.ErrNoAudioDevice) {
		t.Fatalf("open error: got %v", err)
	}
	// The old device stays active and the player keeps playing.
	if got := route.Snapshot().CurrentID; got != "speaker" {
		t.Fatalf("active after failed open: got %q, want speaker", got)
	}
	if !p.IsPlaying() {
		t.Fatalf("player stopped after failed switch")
	}
}

func TestRouteConcurrentSwitchSerialized(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "dock", Name: "Dock", Kind: audio.DeviceKindHDMI,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)
	if _, err := c.NewPlayerF32(newConstF32Source(0.5, 60, rate)); err != nil {
		t.Fatal(err)
	}

	// Block the first switch so a second one races it and must be rejected.
	reg.BlockNextOpen("bt")
	done := make(chan error, 2)
	go func() { done <- route.SwitchDevice("bt") }()

	// Wait until the first switch is actually blocked in OpenDevice.
	reg.WaitForBlockedOpen("bt")

	err := route.SwitchDevice("dock")
	if !errors.Is(err, audio.ErrAudioDeviceSwitchInProgress) {
		t.Fatalf("concurrent switch: got %v, want ErrAudioDeviceSwitchInProgress", err)
	}

	reg.ReleaseBlockedOpen("bt")
	if err := <-done; err != nil {
		t.Fatalf("blocked switch failed on release: %v", err)
	}
	if got := route.Snapshot().CurrentID; got != "bt" {
		t.Fatalf("final device: got %q, want bt", got)
	}
}

func TestRouteCloseIdempotentAndReopen(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
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

	if err := route.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := route.Close(); !errors.Is(err, audio.ErrAudioDeviceClosed) {
		t.Fatalf("second close: got %v, want ErrAudioDeviceClosed", err)
	}
	if reg.OpenCount("speaker") != 0 {
		t.Fatalf("handle not closed: %d", reg.OpenCount("speaker"))
	}

	// Operations on a closed route fail with the context error.
	if err := route.SwitchDevice("speaker"); !errors.Is(err, audio.ErrAudioDeviceClosed) {
		t.Fatalf("switch after close: got %v", err)
	}

	// Reopen rebuilds the device and reattaches the still-live player.
	if err := route.Reopen(); err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if got := route.Snapshot().State; got != audio.RouteStateActive {
		t.Fatalf("state after reopen: %v", got)
	}
	if reg.OpenCount("speaker") != 1 {
		t.Fatalf("handle after reopen: %d", reg.OpenCount("speaker"))
	}
}

func TestRouteNoLeakOnRepeatedSwitches(t *testing.T) {
	const rate = 48000
	c := setupRoutingTest(t, rate)
	reg := audio.CurrentVirtualDeviceRegistryForTesting()
	defer teardownRoutingTest(t)
	reg.AddDevice(audio.VirtualDeviceSpecForTesting{
		ID: "bt", Name: "Bluetooth", Kind: audio.DeviceKindBluetooth,
		SampleRate: rate, ChannelCounts: []int{2},
	})
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)
	if _, err := c.NewPlayerF32(newConstF32Source(0.5, 60, rate)); err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	for range 10 {
		if err := route.SwitchDevice("bt"); err != nil {
			t.Fatal(err)
		}
		if err := route.SwitchDevice("speaker"); err != nil {
			t.Fatal(err)
		}
	}
	// Exactly one handle open at the end, opens and closes paired otherwise.
	if reg.TotalOpens() != reg.TotalCloses()+1 {
		t.Fatalf("leak: opens=%d closes=%d", reg.TotalOpens(), reg.TotalCloses())
	}
}

func TestRouteSuspendResumeContinuous(t *testing.T) {
	c := setupRoutingTest(t, 48000)
	defer teardownRoutingTest(t)
	route := c.AudioDeviceRoute()
	route.SetFadeDuration(0)
	p, err := c.NewPlayerF32(newConstF32Source(0.5, 60, 48000))
	if err != nil {
		t.Fatal(err)
	}
	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}

	if err := audio.SuspendForTesting(); err != nil {
		t.Fatalf("suspend: %v", err)
	}
	// Suspend stops the device rather than pausing the player: the player still reports playing
	// (it will resume when the device resumes), matching the pre-routing behavior.
	if !p.IsPlaying() {
		t.Fatalf("player should remain in the playing state while suspended")
	}
	if err := audio.ResumeForTesting(); err != nil {
		t.Fatalf("resume: %v", err)
	}
	// The player remains tracked and can still be driven after resume.
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
}
