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
	"sync"

	"github.com/ebitengine/oto/v3"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/vmaudio"
)

// systemDeviceOpener is the production deviceOpener.
//
// Oto (v3.5.x) owns one process-wide context and does not enumerate or reopen output devices, so a
// production process exposes a single logical output device: the one Oto opened. The opener still
// models it explicitly so the rest of the router has one code path, and so a future Oto version
// can add real enumeration behind this interface without changing the public API.
//
// When the first OpenDevice call reuses the context's already-created driver, no second Oto
// context is created. Opening a *different* ID fails with ErrAudioDeviceNotFound, since Oto cannot
// move the running context to another device in this version.
type systemDeviceOpener struct {
	vmGuest bool

	mu          sync.Mutex
	changeCb    func()
	openedID    string
	openHandles map[*systemDeviceHandle]struct{}
}

func newSystemDeviceOpener(vmGuest bool) *systemDeviceOpener {
	return &systemDeviceOpener{
		vmGuest:     vmGuest,
		openHandles: map[*systemDeviceHandle]struct{}{},
	}
}

// defaultDeviceID is the stable ID of the single output device the production opener exposes.
const defaultDeviceID = "default"

func (o *systemDeviceOpener) EnumerateDevices() ([]DeviceInfo, error) {
	kind := DeviceKindSpeaker
	if o.vmGuest {
		kind = DeviceKindVirtual
	}
	return []DeviceInfo{{
		ID:          defaultDeviceID,
		Name:        "Default",
		Kind:        kind,
		IsDefault:   true,
		IsAvailable: true,
		Capabilities: DeviceCapabilities{
			// Oto opens the device at the rate and channel count the context asks for. Report the
			// context's own format as supported; the router never needs to resample in production.
			MinSampleRate:   0,
			MaxSampleRate:   0,
			ChannelCounts:   []int{channelCount},
			CanChangeVolume: true,
		},
	}}, nil
}

func (o *systemDeviceOpener) OpenDevice(id string, sampleRate, channelCount int) (deviceHandle, <-chan struct{}, error) {
	// Only the single default device exists in the production opener.
	if id != "" && id != defaultDeviceID {
		return nil, nil, ErrAudioDeviceNotFound
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	// Reuse the driver created by the context's before-update hook, if any. This is the normal
	// path: routing is adopted after the device exists.
	if existing := o.existingDriver(); existing != nil {
		handle := &systemDeviceHandle{opener: o, driver: existing, adopted: true}
		o.openHandles[handle] = struct{}{}
		o.openedID = defaultDeviceID
		ready := make(chan struct{})
		close(ready)
		return handle, ready, nil
	}

	// No driver yet: create the device now. Oto permits one context per process.
	d, ready, err := newDriver(sampleRate, o.vmGuest)
	if err != nil {
		return nil, nil, err
	}
	handle := &systemDeviceHandle{opener: o, driver: d}
	o.openHandles[handle] = struct{}{}
	o.openedID = defaultDeviceID
	return handle, ready, nil
}

// existingDriver returns the context's current driver, or nil if none was created.
func (o *systemDeviceOpener) existingDriver() driver {
	return currentFactoryDriver()
}

func (o *systemDeviceOpener) OnDevicesChanged(f func()) {
	o.mu.Lock()
	o.changeCb = f
	o.mu.Unlock()
}

// systemDeviceHandle wraps a production driver and tracks open handles for symmetry with the
// virtual registry used in tests.
type systemDeviceHandle struct {
	opener  *systemDeviceOpener
	driver  driver
	adopted bool
	closed  bool
	mu      sync.Mutex
}

func (h *systemDeviceHandle) Suspend() error { return h.driver.Suspend() }
func (h *systemDeviceHandle) Resume() error  { return h.driver.Resume() }
func (h *systemDeviceHandle) Err() error     { return h.driver.Err() }

func (h *systemDeviceHandle) NewPlayer(r io.Reader) routedDevicePlayer {
	return &systemDevicePlayer{player: h.driver.NewPlayer(r)}
}

func (h *systemDeviceHandle) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return nil
	}
	h.closed = true

	h.opener.mu.Lock()
	delete(h.opener.openHandles, h)
	h.opener.mu.Unlock()

	// An adopted driver is owned by the context and lives until process exit, exactly as before
	// routing existed. A driver the opener created itself is released here. Oto's Close is a no-op
	// in v3.5.x, but vmaudio players hold resources that Close releases.
	if !h.adopted {
		if closer, ok := h.driver.(io.Closer); ok {
			_ = closer.Close()
		}
	}
	return nil
}

// systemDevicePlayer adapts the driver player to routedDevicePlayer.
type systemDevicePlayer struct {
	player player
}

func (p *systemDevicePlayer) Pause()                 { p.player.Pause() }
func (p *systemDevicePlayer) PauseAndStopReading()   { p.player.PauseAndStopReading() }
func (p *systemDevicePlayer) Play()                  { p.player.Play() }
func (p *systemDevicePlayer) IsPlaying() bool        { return p.player.IsPlaying() }
func (p *systemDevicePlayer) Volume() float64        { return p.player.Volume() }
func (p *systemDevicePlayer) SetVolume(v float64)    { p.player.SetVolume(v) }
func (p *systemDevicePlayer) BufferedSize() int      { return p.player.BufferedSize() }
func (p *systemDevicePlayer) Err() error             { return p.player.Err() }
func (p *systemDevicePlayer) SetBufferSize(size int) { p.player.SetBufferSize(size) }
func (p *systemDevicePlayer) Seek(offset int64, whence int) (int64, error) {
	return p.player.Seek(offset, whence)
}
func (p *systemDevicePlayer) Close() error {
	if closer, ok := p.player.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// currentFactoryDriver returns the live driver of the process's audio context, if any. The opener
// uses this to adopt the device the context created before routing was enabled.
func currentFactoryDriver() driver {
	theContextLock.Lock()
	c := theContext
	theContextLock.Unlock()
	if c == nil {
		return nil
	}
	return c.playerFactory.currentDriver()
}

// Compile-time assertions that the production types satisfy the routing interfaces.
var (
	_ deviceOpener = (*systemDeviceOpener)(nil)
	_ deviceHandle = (*systemDeviceHandle)(nil)
	_              = vmaudio.NewContext
	_              = oto.FormatFloat32LE
)
