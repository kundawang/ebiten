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
	"io"
	"math"
	"slices"
	"sync"
)

// VirtualDeviceRegistry is the test double for deviceOpener. It models a set of pluggable output
// devices with controllable capabilities and failure modes, and captures every frame the routed
// players produce so tests can compare them against an audio-frame reference.
type VirtualDeviceRegistry struct {
	mu sync.Mutex

	devices map[string]*VirtualDevice

	// openErrFor maps a device ID to the error its next OpenDevice call returns.
	openErrFor map[string]error

	changeCb func()

	// openHandles counts open handles by device ID for leak checks.
	openHandles map[string]int

	// totalOpens and totalCloses count lifecycle events across all devices.
	totalOpens  int
	totalCloses int
}

// VirtualDevice is one entry of a VirtualDeviceRegistry.
type VirtualDevice struct {
	info DeviceInfo

	// captured holds every float32 stereo frame the device's players produced, interleaved.
	captured []float32

	// handles are the device's open handles.
	handles []*virtualHandle

	// openBlocks, when > 0, makes the next OpenDevice block until releaseOpen is called, to test
	// concurrent switches and readiness.
	openBlocks  int
	releaseOpen chan struct{}

	// openStarted is closed by OpenDevice once it has reached a blocked open, so a test can wait
	// deterministically instead of sleeping.
	openStarted chan struct{}
}

// NewVirtualDeviceRegistry installs a fresh registry as the test opener and returns it. The
// registry starts with one available stereo 48kHz default device. Tests must call
// ResetVirtualDeviceRegistryForTesting (or ResetContextForTesting) when finished.
func NewVirtualDeviceRegistryForTesting() *VirtualDeviceRegistry {
	reg := &VirtualDeviceRegistry{
		devices:     map[string]*VirtualDevice{},
		openErrFor:  map[string]error{},
		openHandles: map[string]int{},
	}
	reg.AddDevice(VirtualDeviceSpecForTesting{
		ID:            "speaker",
		Name:          "Speaker",
		Kind:          DeviceKindSpeaker,
		IsDefault:     true,
		SampleRate:    48000,
		ChannelCounts: []int{2},
	})
	deviceOpenerForTesting = reg
	return reg
}

// ResetVirtualDeviceRegistryForTesting removes the test opener so later contexts use production
// behavior again.
func ResetVirtualDeviceRegistryForTesting() {
	deviceOpenerForTesting = nil
}

// CurrentVirtualDeviceRegistryForTesting returns the installed registry, or nil.
func CurrentVirtualDeviceRegistryForTesting() *VirtualDeviceRegistry {
	reg, _ := deviceOpenerForTesting.(*VirtualDeviceRegistry)
	return reg
}

// VirtualDeviceSpecForTesting describes a device to add to a registry.
type VirtualDeviceSpecForTesting struct {
	ID            string
	Name          string
	Kind          DeviceKind
	IsDefault     bool
	IsUnavailable bool
	Denied        bool
	SampleRate    int
	ChannelCounts []int
	// MinRate and MaxRate, when set, override SampleRate for the capability range.
	MinRate int
	MaxRate int
}

// AddDevice adds or replaces a device. Adding a device fires the change callback.
func (r *VirtualDeviceRegistry) AddDevice(spec VirtualDeviceSpecForTesting) {
	r.mu.Lock()
	minRate, maxRate := spec.SampleRate, spec.SampleRate
	if spec.MinRate > 0 {
		minRate = spec.MinRate
	}
	if spec.MaxRate > 0 {
		maxRate = spec.MaxRate
	}
	ch := spec.ChannelCounts
	if len(ch) == 0 {
		ch = []int{2}
	}
	// A device is available by default: specs opt out explicitly with IsAvailable=false (for a
	// docked or disconnected device). Permission denial is tracked separately.
	available := !spec.IsUnavailable
	d := &VirtualDevice{
		info: DeviceInfo{
			ID:               spec.ID,
			Name:             spec.Name,
			Kind:             spec.Kind,
			IsDefault:        spec.IsDefault,
			IsAvailable:      available,
			PermissionDenied: spec.Denied,
			Capabilities: DeviceCapabilities{
				MinSampleRate:   minRate,
				MaxSampleRate:   maxRate,
				ChannelCounts:   slices.Clone(ch),
				CanChangeVolume: true,
			},
		},
	}
	r.devices[spec.ID] = d
	cb := r.changeCb
	r.mu.Unlock()
	if cb != nil {
		cb()
	}
}

// RemoveDevice removes a device and fires the change callback.
func (r *VirtualDeviceRegistry) RemoveDevice(id string) {
	r.mu.Lock()
	delete(r.devices, id)
	cb := r.changeCb
	r.mu.Unlock()
	if cb != nil {
		cb()
	}
}

// SetDefault makes the given device the default one and fires the change callback.
func (r *VirtualDeviceRegistry) SetDefault(id string) {
	r.mu.Lock()
	for k, d := range r.devices {
		d.info.IsDefault = k == id
	}
	cb := r.changeCb
	r.mu.Unlock()
	if cb != nil {
		cb()
	}
}

// SetAvailable marks a device available or unavailable and fires the change callback.
func (r *VirtualDeviceRegistry) SetAvailable(id string, available bool) {
	r.mu.Lock()
	if d, ok := r.devices[id]; ok {
		d.info.IsAvailable = available
	}
	cb := r.changeCb
	r.mu.Unlock()
	if cb != nil {
		cb()
	}
}

// SetOpenError makes the next OpenDevice for id return err.
func (r *VirtualDeviceRegistry) SetOpenError(id string, err error) {
	r.mu.Lock()
	r.openErrFor[id] = err
	r.mu.Unlock()
}

// BlockNextOpen makes the next OpenDevice for id block until ReleaseBlockedOpen is called.
func (r *VirtualDeviceRegistry) BlockNextOpen(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if d, ok := r.devices[id]; ok {
		d.openBlocks++
		d.releaseOpen = make(chan struct{})
		d.openStarted = make(chan struct{})
	}
}

// WaitForBlockedOpen blocks until an OpenDevice for id has reached its blocking point.
func (r *VirtualDeviceRegistry) WaitForBlockedOpen(id string) {
	r.mu.Lock()
	ch := r.devices[id].openStarted
	r.mu.Unlock()
	if ch != nil {
		<-ch
	}
}

// ReleaseBlockedOpen unblocks an OpenDevice waiting on id.
func (r *VirtualDeviceRegistry) ReleaseBlockedOpen(id string) {
	r.mu.Lock()
	d := r.devices[id]
	ch := d.releaseOpen
	r.mu.Unlock()
	if ch != nil {
		close(ch)
	}
}

// OpenCount reports how many handles are currently open for id.
func (r *VirtualDeviceRegistry) OpenCount(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.openHandles[id]
}

// TotalOpens reports the cumulative number of successful OpenDevice calls.
func (r *VirtualDeviceRegistry) TotalOpens() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.totalOpens
}

// TotalCloses reports the cumulative number of handle Close calls.
func (r *VirtualDeviceRegistry) TotalCloses() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.totalCloses
}

// CapturedFrames returns a copy of the float32 frames captured on a device.
func (r *VirtualDeviceRegistry) CapturedFrames(id string) []float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	d := r.devices[id]
	if d == nil {
		return nil
	}
	return slices.Clone(d.captured)
}

// deviceForID returns the device entry without copying its capture buffer.
func (r *VirtualDeviceRegistry) deviceForID(id string) *VirtualDevice {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.devices[id]
}

// EnumerateDevices implements deviceOpener.
func (r *VirtualDeviceRegistry) EnumerateDevices() ([]DeviceInfo, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.devices))
	for id := range r.devices {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	infos := make([]DeviceInfo, 0, len(ids))
	for _, id := range ids {
		infos = append(infos, r.devices[id].info)
	}
	return infos, nil
}

// OpenDevice implements deviceOpener.
func (r *VirtualDeviceRegistry) OpenDevice(id string, sampleRate, channelCount int) (deviceHandle, <-chan struct{}, error) {
	r.mu.Lock()
	if err := r.openErrFor[id]; err != nil {
		delete(r.openErrFor, id)
		r.mu.Unlock()
		return nil, nil, err
	}
	d := r.devices[id]
	if d == nil {
		r.mu.Unlock()
		return nil, nil, ErrAudioDeviceNotFound
	}
	if d.info.PermissionDenied {
		r.mu.Unlock()
		return nil, nil, ErrAudioDevicePermissionDenied
	}
	if !d.info.IsAvailable {
		r.mu.Unlock()
		return nil, nil, ErrNoAudioDevice
	}
	if !d.info.Capabilities.supportsAnyRateFor(channelCount) {
		r.mu.Unlock()
		return nil, nil, ErrAudioDeviceIncompatible
	}
	blockCh := d.releaseOpen
	startedCh := d.openStarted
	blocks := d.openBlocks
	if blocks > 0 {
		d.openBlocks--
	}
	r.mu.Unlock()

	if blocks > 0 && blockCh != nil {
		if startedCh != nil {
			close(startedCh)
		}
		<-blockCh
	}

	r.mu.Lock()
	h := &virtualHandle{
		reg:      r,
		device:   d,
		deviceID: d.info.ID,
		rate:     r.openRate(d, sampleRate),
	}
	d.handles = append(d.handles, h)
	r.openHandles[id]++
	r.totalOpens++
	ready := make(chan struct{})
	close(ready)
	r.mu.Unlock()
	return h, ready, nil
}

// openRate picks the rate the device runs at: its fixed rate when it cannot run at the context
// rate, otherwise the context rate.
func (r *VirtualDeviceRegistry) openRate(d *VirtualDevice, contextRate int) int {
	caps := d.info.Capabilities
	if caps.supports(contextRate, 2) {
		return contextRate
	}
	return clamp(contextRate, caps.MinSampleRate, caps.MaxSampleRate)
}

// OnDevicesChanged implements deviceOpener.
func (r *VirtualDeviceRegistry) OnDevicesChanged(f func()) {
	r.mu.Lock()
	r.changeCb = f
	r.mu.Unlock()
}

// capture records frames produced on a device.
func (r *VirtualDeviceRegistry) capture(d *VirtualDevice, frames []float32) {
	r.mu.Lock()
	d.captured = append(d.captured, frames...)
	r.mu.Unlock()
}

// virtualHandle is one open virtual device.
type virtualHandle struct {
	reg      *VirtualDeviceRegistry
	device   *VirtualDevice
	deviceID string
	rate     int
	closed   bool
	mu       sync.Mutex
}

func (h *virtualHandle) NewPlayer(r io.Reader) routedDevicePlayer {
	return &virtualPlayer{handle: h, src: r}
}

func (h *virtualHandle) Suspend() error { return nil }
func (h *virtualHandle) Resume() error  { return nil }
func (h *virtualHandle) Err() error     { return nil }

func (h *virtualHandle) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return nil
	}
	h.closed = true
	h.reg.mu.Lock()
	h.reg.openHandles[h.device.info.ID]--
	h.reg.totalCloses++
	h.reg.mu.Unlock()
	return nil
}

// drainFades pulls frames from the routed players until their fades complete, capturing them on
// the device. It implements the optional pull interface the router uses in tests.
func (h *virtualHandle) drainFades(players []*playerImpl, phase fadePhase, frames int) {
	if frames == 0 {
		return
	}
	buf := make([]byte, 4096)
	tmp := make([]float32, 4096/4)
	// A bound on the number of empty polls guards against spinning forever when a player cannot
	// produce data (a paused or finished player whose fade is already complete from its side).
	const maxEmptyPolls = 64
	emptyPolls := 0
	for emptyPolls < maxEmptyPolls {
		allDone := true
		progress := false
		for _, p := range players {
			p.m.Lock()
			rp := p.routed
			gain := p.gain
			p.m.Unlock()
			if rp == nil || gain == nil {
				continue
			}
			if rp.device == nil || rp.device.info.ID != h.deviceID {
				continue
			}
			if !gain.done(phase, frames) {
				allDone = false
				n, _ := rp.stream.Read(buf)
				if n > 0 {
					progress = true
					for i := range n / 4 {
						bits := uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 |
							uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24
						tmp[i] = math.Float32frombits(bits)
					}
					h.reg.capture(h.device, tmp[:n/4])
				}
			}
		}
		if allDone {
			return
		}
		if progress {
			emptyPolls = 0
		} else {
			emptyPolls++
		}
	}
}

// virtualPlayer is a device player on a virtual handle. It is pull-based: Play only marks state;
// reads happen through the routed stream (and drainFades).
type virtualPlayer struct {
	handle  *virtualHandle
	src     io.Reader
	playing bool
	volume  float64
	mu      sync.Mutex
}

func (p *virtualPlayer) Play() {
	p.mu.Lock()
	p.playing = true
	p.mu.Unlock()
}

func (p *virtualPlayer) Pause() {
	p.mu.Lock()
	p.playing = false
	p.mu.Unlock()
}

func (p *virtualPlayer) PauseAndStopReading() { p.Pause() }

func (p *virtualPlayer) IsPlaying() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.playing
}

func (p *virtualPlayer) Volume() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.volume
}

func (p *virtualPlayer) SetVolume(v float64) {
	p.mu.Lock()
	p.volume = v
	p.mu.Unlock()
}

func (p *virtualPlayer) BufferedSize() int { return 0 }
func (p *virtualPlayer) Err() error        { return nil }
func (p *virtualPlayer) SetBufferSize(int) {}

func (p *virtualPlayer) Seek(offset int64, whence int) (int64, error) {
	if s, ok := p.src.(io.Seeker); ok {
		return s.Seek(offset, whence)
	}
	return 0, nil
}

func (p *virtualPlayer) Close() error { return nil }
