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
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
)

// DeviceKind classifies an audio output device by where it sends sound.
type DeviceKind int

const (
	// DeviceKindUnknown is the kind of a device whose kind cannot be determined.
	DeviceKindUnknown DeviceKind = iota

	// DeviceKindSpeaker is a built-in or external loudspeaker.
	DeviceKindSpeaker

	// DeviceKindHeadphone is a wired headphone or headset.
	DeviceKindHeadphone

	// DeviceKindBluetooth is a Bluetooth audio device, such as Bluetooth headphones or an earpiece.
	DeviceKindBluetooth

	// DeviceKindUSB is a USB audio device.
	DeviceKindUSB

	// DeviceKindHDMI is an HDMI or DisplayPort audio output.
	DeviceKindHDMI

	// DeviceKindVirtual is a virtual audio device, for example a loopback or a virtualization device.
	DeviceKindVirtual
)

// String returns a human-readable name of the device kind.
func (k DeviceKind) String() string {
	switch k {
	case DeviceKindSpeaker:
		return "speaker"
	case DeviceKindHeadphone:
		return "headphone"
	case DeviceKindBluetooth:
		return "bluetooth"
	case DeviceKindUSB:
		return "usb"
	case DeviceKindHDMI:
		return "hdmi"
	case DeviceKindVirtual:
		return "virtual"
	default:
		return "unknown"
	}
}

// DeviceCapabilities describes what an audio output device can do.
//
// All fields describe the hardware's own format. A device whose SampleRate differs from the
// context's sample rate is still usable: the context resamples every player's stream for it.
type DeviceCapabilities struct {
	// MinSampleRate and MaxSampleRate bound the sample rates, in per-channel samples per second,
	// the device accepts. A device with a single fixed rate reports the same value in both fields.
	MinSampleRate int
	MaxSampleRate int

	// ChannelCounts lists the channel counts (1 for mono, 2 for stereo) the device accepts.
	ChannelCounts []int

	// CanChangeVolume reports whether the device has a hardware or system volume control.
	CanChangeVolume bool
}

// supports reports whether a device accepts a stream at the given sample rate and channel count.
func (c DeviceCapabilities) supports(sampleRate, channelCount int) bool {
	if sampleRate < c.MinSampleRate || sampleRate > c.MaxSampleRate {
		return false
	}
	return slices.Contains(c.ChannelCounts, channelCount)
}

// supportsAnyRateFor reports whether the channel count is supported at some rate the device
// accepts. Zero rate bounds mean the device follows the context's own rate (the production Oto
// device), which always matches.
func (c DeviceCapabilities) supportsAnyRateFor(channelCount int) bool {
	rateOK := (c.MinSampleRate == 0 && c.MaxSampleRate == 0) ||
		(c.MinSampleRate > 0 && c.MaxSampleRate >= c.MinSampleRate)
	return rateOK && slices.Contains(c.ChannelCounts, channelCount)
}

// AudioDevice describes an audio output device the context can route players to.
//
// AudioDevice is a snapshot: the fields do not change after it is returned. Use
// [AudioDeviceRoute.AvailableDevices] again to observe changes.
type AudioDevice struct {
	// ID is the stable identifier of the device within this process.
	// It is unique among the devices reported at any point, but a reconnected device can get a new ID.
	ID string

	// Name is the human-readable name of the device.
	Name string

	// Kind classifies the device.
	Kind DeviceKind

	// IsDefault reports whether the device is the system's default output device.
	IsDefault bool

	// IsAvailable reports whether the device is connected and can be opened.
	IsAvailable bool

	// Capabilities describes the formats the device accepts.
	Capabilities DeviceCapabilities
}

// AudioDeviceRouteState is the state of an [AudioDeviceRoute].
type AudioDeviceRouteState int

const (
	// RouteStateActive means a device is open and players are routed to it.
	RouteStateActive AudioDeviceRouteState = iota

	// RouteStateSwitching means a device switch, including its fade-out and fade-in, is in progress.
	RouteStateSwitching

	// RouteStateUnavailable means no device is open. Players keep their state and resume on the
	// next available device.
	RouteStateUnavailable
)

// String returns a human-readable name of the route state.
func (s AudioDeviceRouteState) String() string {
	switch s {
	case RouteStateActive:
		return "active"
	case RouteStateSwitching:
		return "switching"
	case RouteStateUnavailable:
		return "unavailable"
	default:
		return "unknown"
	}
}

// AudioDeviceRouteSnapshot is an immutable point-in-time view of an [AudioDeviceRoute].
type AudioDeviceRouteSnapshot struct {
	// State is the current route state.
	State AudioDeviceRouteState

	// CurrentID is the open device's ID, or empty when no device is open.
	CurrentID string

	// TargetID is the device a switch in progress is moving to, or empty when no switch is running.
	TargetID string

	// PendingID is the device selected with [AudioDeviceRoute.SwitchDevice] that has not been
	// opened yet, for example because no device is available. It is empty when nothing is pending.
	PendingID string

	// Generation increments whenever this snapshot's content would change.
	Generation uint64
}

// AudioDeviceRoute manages which output device an [audio.Context] plays through.
//
// The route is optional: until any of its methods is called, the context behaves exactly as it
// did before routing existed. Once used, the route owns the audio device and keeps every player's
// position, loop state and volume continuous across switches.
type AudioDeviceRoute struct {
	router *deviceRouter
}

// AvailableDevices returns the devices the context knows about, including unavailable ones.
// The result is sorted by ID so that callers get a deterministic order.
func (r *AudioDeviceRoute) AvailableDevices() []AudioDevice {
	router := r.router.factory.ensureRouter()
	return router.availableDevices()
}

// CurrentDevice returns the open device, or nil when no device is open.
func (r *AudioDeviceRoute) CurrentDevice() *AudioDevice {
	router := r.router.factory.ensureRouter()
	return router.currentDevice()
}

// SwitchDevice moves the route to the device with the given ID.
//
// Switching fades every playing player out on the current device, reopens it on the target device,
// and fades it back in. Paused players are moved silently. Position, loop state and volume are
// preserved.
//
// The empty ID selects the system's default device.
//
// SwitchDevice blocks until the switch, including the fades, finishes. A switch to the device
// already in use is a no-op.
func (r *AudioDeviceRoute) SwitchDevice(id string) error {
	router := r.router.factory.ensureRouter()
	return router.switchDevice(id)
}

// Snapshot returns the current route state.
func (r *AudioDeviceRoute) Snapshot() AudioDeviceRouteSnapshot {
	router := r.router.factory.ensureRouter()
	return router.snapshot()
}

// OnDeviceChange registers a function called after the set of devices, the default device, or the
// open device changes. The callback must not block: it must not call back into the route in a way
// that waits for a switch. Registering replaces the previous callback; passing nil removes it.
//
// The callback runs on an arbitrary goroutine.
func (r *AudioDeviceRoute) OnDeviceChange(f func(snapshot AudioDeviceRouteSnapshot)) {
	router := r.router.factory.ensureRouter()
	router.setOnDeviceChange(f)
}

// SetFadeDuration sets the duration of the fade-out and the fade-in performed during a switch.
// A zero or negative duration makes switches instant. The default is 30 milliseconds.
func (r *AudioDeviceRoute) SetFadeDuration(d time.Duration) {
	router := r.router.factory.ensureRouter()
	router.setFadeDuration(d)
}

// Reopen rebuilds the output device after it was shut down, reattaching every surviving player and
// resuming pending playback. It returns an error when the route is not closed or no device can be
// opened.
func (r *AudioDeviceRoute) Reopen() error {
	return r.router.reopen()
}

// Close shuts the route down and releases its output device. The context normally owns the route
// for the lifetime of the process, so applications rarely need to call it; it exists for tests
// and for hosts that must tear audio down explicitly. Repeated calls return ErrAudioDeviceClosed.
func (r *AudioDeviceRoute) Close() error {
	return r.router.close()
}

// Audio routing errors. Callers should test these with [errors.Is].
var (
	// ErrNoAudioDevice means no output device exists or none is currently available.
	ErrNoAudioDevice = errors.New("audio: no audio device available")

	// ErrAudioDevicePermissionDenied means the system refused access to an audio device.
	ErrAudioDevicePermissionDenied = errors.New("audio: audio device permission denied")

	// ErrAudioDeviceNotFound means a device ID does not match any known device.
	ErrAudioDeviceNotFound = errors.New("audio: audio device not found")

	// ErrAudioDeviceIncompatible means a device cannot play the context's format.
	ErrAudioDeviceIncompatible = errors.New("audio: audio device format is incompatible")

	// ErrAudioDeviceSwitchInProgress means a switch was requested while another one was running.
	ErrAudioDeviceSwitchInProgress = errors.New("audio: an audio device switch is already in progress")

	// ErrAudioDeviceClosed means a route operation was attempted after the route was closed.
	ErrAudioDeviceClosed = errors.New("audio: audio device route is closed")
)

// deviceOpener creates and enumerates output devices. The production implementation wraps Oto;
// tests replace it with a virtual device registry.
type deviceOpener interface {
	// EnumerateDevices returns the devices currently known to the system.
	EnumerateDevices() ([]DeviceInfo, error)

	// OpenDevice opens the device with the given ID. The empty ID opens the default device.
	// The returned ready channel is closed once asynchronous initialization has finished; it may be
	// already closed for a device that is ready immediately.
	OpenDevice(id string, sampleRate, channelCount int) (deviceHandle, <-chan struct{}, error)

	// OnDevicesChanged registers a callback the system invokes when devices are added or removed,
	// or when the default device changes. Passing nil unregisters the callback.
	OnDevicesChanged(func())
}

// DeviceInfo is the opener's description of a device.
type DeviceInfo struct {
	ID               string
	Name             string
	Kind             DeviceKind
	IsDefault        bool
	IsAvailable      bool
	PermissionDenied bool
	Capabilities     DeviceCapabilities
}

func (i DeviceInfo) toAudioDevice() AudioDevice {
	channels := make([]int, len(i.Capabilities.ChannelCounts))
	copy(channels, i.Capabilities.ChannelCounts)
	caps := i.Capabilities
	caps.ChannelCounts = channels
	return AudioDevice{
		ID:           i.ID,
		Name:         i.Name,
		Kind:         i.Kind,
		IsDefault:    i.IsDefault,
		IsAvailable:  i.IsAvailable,
		Capabilities: caps,
	}
}

// deviceHandle is one open output device. NewPlayer returns a player with an explicit Close so the
// router can release old device players. It lists the driver methods explicitly because its
// NewPlayer returns the narrower routedDevicePlayer type rather than driver's player.
type deviceHandle interface {
	NewPlayer(io.Reader) routedDevicePlayer
	Suspend() error
	Resume() error
	Err() error
	Close() error
}

// routedDevicePlayer is the device-level player interface plus resource release.
type routedDevicePlayer interface {
	player
	io.Closer
}

// deviceOpenerForTesting can be replaced by a test's virtual device registry.
var deviceOpenerForTesting deviceOpener

// newDeviceOpener returns the opener that backs production contexts.
func newDeviceOpener(vmGuest bool) deviceOpener {
	if deviceOpenerForTesting != nil {
		return deviceOpenerForTesting
	}
	return newSystemDeviceOpener(vmGuest)
}

// defaultRouteFadeDuration is the default length of each fade side.
const defaultRouteFadeDuration = 30 * time.Millisecond

// deviceRouter owns the routing state. Exactly one exists per context once routing is used.
//
// Locking: switchMu serializes whole device transitions. mu protects the mutable state and is
// never held while opening a device or moving players, both of which can block. Methods that need
// mu take it only for short critical sections.
type deviceRouter struct {
	factory    *playerFactory
	sampleRate int
	channelNum int
	opener     deviceOpener
	fadeDur    time.Duration
	vmGuest    bool

	// switchMu serializes every device transition: explicit switches, hot-plug reactions and
	// bootstrapping. A second transition while one runs fails with ErrAudioDeviceSwitchInProgress.
	switchMu sync.Mutex

	mu sync.Mutex

	// devices is the last enumeration, keyed by ID.
	devices map[string]*routeDevice

	// knownPlayers tracks every player that has taken part in routing, including paused ones, so a
	// device switch moves players that are not currently in the context's playing set.
	knownPlayers map[*playerImpl]struct{}

	// active is the open device, or nil.
	active *routeDevice

	// pendingID is an explicit or default selection that could not be opened yet.
	pendingID string

	// switching reports that a transition is running.
	switching bool

	// generation bumps on every state change.
	generation uint64

	// closed makes every later operation fail with ErrAudioDeviceClosed.
	closed bool

	onChange func(AudioDeviceRouteSnapshot)
}

// routeDevice is the router's per-device state.
type routeDevice struct {
	info DeviceInfo

	// handle is non-nil while the device is open. At most one device is open at a time.
	handle deviceHandle
	ready  <-chan struct{}

	// explicit marks a device chosen by an explicit SwitchDevice call. The default-device
	// follow-up moves only implicit routes.
	explicit bool

	// openCount tracks OpenDevice/Close pairing for leak checks in tests.
	openCount int
}

func (f *playerFactory) ensureRouter() *deviceRouter {
	f.m.Lock()
	defer f.m.Unlock()
	if f.router != nil {
		return f.router
	}

	router := &deviceRouter{
		factory:    f,
		sampleRate: f.sampleRate,
		channelNum: channelCount,
		fadeDur:    defaultRouteFadeDuration,
		devices:    map[string]*routeDevice{},
		vmGuest:    f.vmGuest,
	}
	f.router = router
	router.initLocked()
	return router
}

// initLocked enumerates devices, adopts an already-created driver as the default device's handle,
// or opens the default device for the first time. f.m is held; router.mu is not.
func (r *deviceRouter) initLocked() {
	r.opener = newDeviceOpener(r.vmGuest)
	r.opener.OnDevicesChanged(r.onDevicesChanged)

	infos, err := r.opener.EnumerateDevices()
	if err != nil {
		// Enumeration failing is not fatal: hot-plug callbacks or a later switch can retry.
		return
	}
	for i := range infos {
		info := infos[i]
		r.devices[info.ID] = &routeDevice{info: info}
	}

	// Adopt the driver the context created before routing was used. A test's virtual registry
	// owns its own handles, so when one is installed the router always opens through the opener.
	if existing := r.factory.driver; existing != nil && deviceOpenerForTesting == nil {
		def := r.defaultOrSyntheticLocked()
		// A plain driver's NewPlayer returns the wider player type. Wrap it so the router gets a
		// uniform deviceHandle with a Close method.
		handle := legacyDriverHandle{driver: existing}
		def.handle = handle
		def.openCount = 1
		ready := make(chan struct{})
		close(ready)
		def.ready = ready
		r.active = def
		return
	}

	// No driver yet: open the default device. Opening here is safe: the factory lock is held but
	// openDevice never calls back into the factory.
	if target := r.preferredTargetLocked(""); target != nil {
		if err := r.openDevice(target); err == nil {
			r.active = target
		}
	}
	r.generation = 1
}

// defaultOrSyntheticLocked returns the default device, creating a synthetic entry when the opener
// reported none (for example an opener with a single unnamed device).
func (r *deviceRouter) defaultOrSyntheticLocked() *routeDevice {
	for _, d := range r.devices {
		if d.info.IsDefault {
			return d
		}
	}
	const id = "default"
	d := &routeDevice{
		info: DeviceInfo{
			ID:          id,
			Name:        "Default",
			Kind:        DeviceKindUnknown,
			IsDefault:   true,
			IsAvailable: true,
			Capabilities: DeviceCapabilities{
				MinSampleRate:   r.sampleRate,
				MaxSampleRate:   r.sampleRate,
				ChannelCounts:   []int{r.channelNum},
				CanChangeVolume: true,
			},
		},
	}
	r.devices[id] = d
	return d
}

// preferredTargetLocked resolves an ID to an available device. Empty ID means default, then any
// available device in ID order.
func (r *deviceRouter) preferredTargetLocked(id string) *routeDevice {
	if id == "" {
		for _, d := range r.devices {
			if d.info.IsDefault && d.info.IsAvailable {
				return d
			}
		}
		ids := make([]string, 0, len(r.devices))
		for k, d := range r.devices {
			if d.info.IsAvailable {
				ids = append(ids, k)
			}
		}
		slices.Sort(ids)
		if len(ids) > 0 {
			return r.devices[ids[0]]
		}
		return nil
	}
	return r.devices[id]
}

func (r *deviceRouter) setOnDeviceChange(f func(AudioDeviceRouteSnapshot)) {
	r.mu.Lock()
	r.onChange = f
	r.mu.Unlock()
}

func (r *deviceRouter) setFadeDuration(d time.Duration) {
	r.mu.Lock()
	r.fadeDur = d
	r.mu.Unlock()
}

func (r *deviceRouter) availableDevices() []AudioDevice {
	r.mu.Lock()
	defer r.mu.Unlock()

	ids := make([]string, 0, len(r.devices))
	for id := range r.devices {
		ids = append(ids, id)
	}
	slices.Sort(ids)

	devs := make([]AudioDevice, 0, len(ids))
	for _, id := range ids {
		devs = append(devs, r.devices[id].info.toAudioDevice())
	}
	return devs
}

func (r *deviceRouter) currentDevice() *AudioDevice {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active == nil {
		return nil
	}
	d := r.active.info.toAudioDevice()
	return &d
}

func (r *deviceRouter) snapshot() AudioDeviceRouteSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	state := RouteStateActive
	if r.switching {
		state = RouteStateSwitching
	} else if r.active == nil {
		state = RouteStateUnavailable
	}
	snap := AudioDeviceRouteSnapshot{
		State:      state,
		PendingID:  r.pendingID,
		Generation: r.generation,
	}
	if r.active != nil {
		snap.CurrentID = r.active.info.ID
	}
	if r.switching {
		snap.TargetID = r.pendingID
	}
	return snap
}

// fireCallback invokes the change callback without holding mu.
func (r *deviceRouter) fireCallback() {
	r.mu.Lock()
	cb := r.onChange
	snap := r.snapshotLocked()
	r.mu.Unlock()
	if cb != nil {
		cb(snap)
	}
}

func (r *deviceRouter) snapshotLocked() AudioDeviceRouteSnapshot {
	state := RouteStateActive
	if r.switching {
		state = RouteStateSwitching
	} else if r.active == nil {
		state = RouteStateUnavailable
	}
	snap := AudioDeviceRouteSnapshot{
		State:      state,
		PendingID:  r.pendingID,
		Generation: r.generation,
	}
	if r.active != nil {
		snap.CurrentID = r.active.info.ID
	}
	if r.switching {
		snap.TargetID = r.pendingID
	}
	return snap
}

// onDevicesChanged is the opener's hot-plug callback. It re-enumerates and reacts deterministically.
func (r *deviceRouter) onDevicesChanged() {
	if !r.beginSwitch() {
		// A switch is already in progress: the event is reconsidered when that switch finishes,
		// since every transition re-enumerates the device set first.
		return
	}
	defer r.finishSwitch()

	infos, err := r.opener.EnumerateDevices()
	if err != nil {
		return
	}

	r.mu.Lock()
	r.refreshDevicesLocked(infos)
	active := r.active
	explicitRoute := r.pendingID != ""
	r.mu.Unlock()

	// Fixed reaction order so behavior never depends on map iteration:
	// 1. The active device vanished: move to the pending selection, then default, then any device.
	// 2. An implicit route and a new default: follow the new default.
	// 3. Nothing is open: open a pending selection, else the default.
	if active != nil && !active.info.IsAvailable {
		r.replaceAfterRemoval(active)
		return
	}

	r.mu.Lock()
	var newDefault *routeDevice
	for _, d := range r.devices {
		if d.info.IsDefault && d.info.IsAvailable {
			newDefault = d
			break
		}
	}
	follow := active != nil && newDefault != nil && active != newDefault &&
		!active.explicit && !explicitRoute
	r.mu.Unlock()

	if follow {
		_ = r.migrate(active, newDefault)
		return
	}

	// A pending explicit selection whose device just became available wins over the running
	// default device: the user asked for it and the hot-plug event is the moment it can open.
	// When the active device is itself an explicit selection, stay put: a later default-device
	// change must never override an explicit choice.
	r.mu.Lock()
	pending := r.preferredTargetLocked(r.pendingID)
	activeExplicit := active != nil && active.explicit
	if pending != nil && active != nil && pending != active && pending.info.IsAvailable &&
		!activeExplicit && r.pendingID != "" {
		r.mu.Unlock()
		_ = r.migrate(active, pending)
		return
	}
	r.mu.Unlock()

	if active == nil {
		r.mu.Lock()
		target := r.preferredTargetLocked(r.pendingID)
		if target == nil {
			target = r.preferredTargetLocked("")
		}
		r.mu.Unlock()
		if target != nil {
			_ = r.migrate(nil, target)
		}
	}
}

// refreshDevicesLocked updates the device map while preserving open handles and explicit marks.
func (r *deviceRouter) refreshDevicesLocked(infos []DeviceInfo) {
	next := make(map[string]*routeDevice, len(infos))
	for i := range infos {
		info := infos[i]
		if old, ok := r.devices[info.ID]; ok {
			old.info = info
			next[info.ID] = old
		} else {
			next[info.ID] = &routeDevice{info: info}
		}
	}
	// A device that disappeared but is still open keeps its entry marked unavailable until the
	// migration closes its handle.
	for id, old := range r.devices {
		if _, ok := next[id]; !ok && old.handle != nil {
			old.info.IsAvailable = false
			next[id] = old
		}
	}
	r.devices = next
	r.generation++
}

// beginSwitch acquires the serialized transition without blocking. It reports false when one is
// already running or the route is closed, so a concurrent switch fails immediately with
// ErrAudioDeviceSwitchInProgress instead of queueing behind the running one.
func (r *deviceRouter) beginSwitch() bool {
	if !r.switchMu.TryLock() {
		return false
	}
	r.mu.Lock()
	if r.closed || r.switching {
		r.mu.Unlock()
		r.switchMu.Unlock()
		return false
	}
	r.switching = true
	r.generation++
	r.mu.Unlock()
	return true
}

func (r *deviceRouter) finishSwitch() {
	r.mu.Lock()
	r.switching = false
	r.generation++
	r.mu.Unlock()
	r.fireCallback()
	r.switchMu.Unlock()
}

// switchDevice implements SwitchDevice.
func (r *deviceRouter) switchDevice(id string) error {
	if !r.beginSwitch() {
		r.mu.Lock()
		closed := r.closed
		r.mu.Unlock()
		if closed {
			return ErrAudioDeviceClosed
		}
		return ErrAudioDeviceSwitchInProgress
	}
	defer r.finishSwitch()

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ErrAudioDeviceClosed
	}

	// Re-enumerate so an ID added by hot-plug is known before validation.
	if infos, err := r.opener.EnumerateDevices(); err == nil {
		r.refreshDevicesLocked(infos)
	}

	// Resolve the target. Empty ID means the default device.
	target := r.preferredTargetLocked(id)
	if target == nil {
		// Distinguish an unknown ID from a missing default.
		if id != "" {
			r.mu.Unlock()
			return fmt.Errorf("%w: %q", ErrAudioDeviceNotFound, id)
		}
		r.mu.Unlock()
		return fmt.Errorf("%w: no default device", ErrNoAudioDevice)
	}

	if target.info.PermissionDenied {
		r.mu.Unlock()
		return fmt.Errorf("%w: %q", ErrAudioDevicePermissionDenied, id)
	}
	if !target.info.IsAvailable {
		// Remember the selection so a later hot-plug event opens the device automatically.
		r.pendingID = target.info.ID
		target.explicit = id != ""
		r.generation++
		r.mu.Unlock()
		return fmt.Errorf("%w: %q", ErrNoAudioDevice, id)
	}
	caps := target.info.Capabilities
	// The router resamples to any rate in the device's range, so only a channel-count mismatch is
	// truly incompatible.
	if !caps.supportsAnyRateFor(r.channelNum) {
		r.mu.Unlock()
		return fmt.Errorf("%w: device %q does not support %d channels",
			ErrAudioDeviceIncompatible, id, r.channelNum)
	}

	// Switching to the active device is a no-op, but clears any pending selection.
	if r.active != nil && r.active.info.ID == target.info.ID {
		r.pendingID = ""
		target.explicit = id != ""
		r.generation++
		r.mu.Unlock()
		return nil
	}

	old := r.active
	target.explicit = id != ""
	r.pendingID = target.info.ID
	r.generation++
	r.mu.Unlock()

	if err := r.migrate(old, target); err != nil {
		return err
	}

	r.mu.Lock()
	r.pendingID = ""
	r.generation++
	r.mu.Unlock()
	return nil
}

// replaceAfterRemoval moves off a device that disappeared. It never fails the route: when no
// replacement can be opened, the context enters the unavailable state and keeps the players.
func (r *deviceRouter) replaceAfterRemoval(active *routeDevice) {
	r.mu.Lock()
	target := r.preferredTargetLocked(r.pendingID)
	if target == nil || target == active {
		target = r.preferredTargetLocked("")
	}
	if target == active {
		target = nil
	}
	r.mu.Unlock()

	if target == nil {
		// Nothing to move to: detach the players without a new device and go unavailable.
		players := r.factory.routedPlayers()
		for _, p := range players {
			r.detachWithoutReplacement(p)
		}
		r.closeDevice(active)
		r.mu.Lock()
		if r.active == active {
			r.active = nil
		}
		r.generation++
		r.mu.Unlock()
		return
	}
	_ = r.migrate(active, target)
}

// migrate performs the full transition. The target is validated by the caller. On any failure the
// old device stays active and the players keep playing.
//
// migrate runs with a transition already begun (beginSwitch) and without r.mu held.
func (r *deviceRouter) migrate(old *routeDevice, target *routeDevice) error {
	if old != nil && old == target {
		return nil
	}

	// Open the target before touching any player, so a failure leaves the running route untouched.
	if target.handle == nil {
		if err := r.openDevice(target); err != nil {
			return err
		}
	}

	// Deterministic player order: each player gets a stable routeSeq at creation, so every
	// migration applies the same sequence regardless of map iteration.
	players := r.factory.routedPlayers()
	fadeFrames := r.fadeFrames()

	// Phase 1: fade out and pause on the old device.
	if old != nil {
		for _, p := range players {
			r.beginFadeOut(p, fadeFrames)
		}
		r.driveFades(players, fadePhaseOut, fadeFrames, old)
		for _, p := range players {
			r.detachFromOldDevice(p)
		}
	}

	// Phase 2: make the target the active device before binding, so a player created or a pending
	// play processed concurrently attaches to the new device rather than the one being torn down.
	r.mu.Lock()
	r.active = target
	r.mu.Unlock()

	// Bind every player to the new device and start its fade-in. A player already on the target
	// (a concurrent attach won the race) is left alone.
	for _, p := range players {
		if err := r.attachToNewDevice(p, target, fadeFrames); err != nil {
			// Reattach the player to the old device without a fade, so playback continues.
			if old != nil {
				r.reattachToOldDevice(p, old)
			}
			return err
		}
	}

	r.mu.Lock()
	r.pendingID = ""
	r.generation++
	r.mu.Unlock()

	if old != nil {
		r.closeDevice(old)
	}

	// Fade-in completes as the new device consumes frames. For a pull-based (virtual) device the
	// router drives the frames itself; a real device's audio clock finishes the ramp on its own,
	// and the gain stream returns to unity once the ramp length has been read.
	r.driveFades(players, fadePhaseIn, fadeFrames, target)

	r.mu.Lock()
	r.generation++
	r.mu.Unlock()
	return nil
}

// fadeFrames converts the configured fade duration to whole frames at the context rate.
func (r *deviceRouter) fadeFrames() int {
	r.mu.Lock()
	d := r.fadeDur
	rate := r.sampleRate
	r.mu.Unlock()
	if d <= 0 {
		return 0
	}
	// Integer math avoids the truncation a float conversion would cause (for example 64 frames at
	// 48kHz becoming 63). Round to the nearest whole frame.
	n := (int64(d)*int64(rate) + int64(time.Second)/2) / int64(time.Second)
	return int(max(0, n))
}

// openDevice opens a device and stores its handle, classifying the failure.
func (r *deviceRouter) openDevice(d *routeDevice) error {
	handle, ready, err := r.opener.OpenDevice(d.info.ID, r.sampleRate, r.channelNum)
	if err != nil {
		return classifyOpenError(d, err)
	}
	if handle == nil {
		return fmt.Errorf("%w: %q", ErrNoAudioDevice, d.info.ID)
	}
	if ready != nil {
		<-ready
	}
	d.handle = handle
	d.ready = ready
	d.openCount++
	return nil
}

func classifyOpenError(d *routeDevice, err error) error {
	switch {
	case errors.Is(err, ErrAudioDevicePermissionDenied):
		return fmt.Errorf("%w: %q: %v", ErrAudioDevicePermissionDenied, d.info.ID, err)
	case errors.Is(err, ErrNoAudioDevice):
		return fmt.Errorf("%w: %q: %v", ErrNoAudioDevice, d.info.ID, err)
	case errors.Is(err, ErrAudioDeviceIncompatible):
		return fmt.Errorf("%w: %q: %v", ErrAudioDeviceIncompatible, d.info.ID, err)
	default:
		return fmt.Errorf("audio: cannot open device %q: %w", d.info.ID, err)
	}
}

// closeDevice closes a device handle once its players have been moved off it.
func (r *deviceRouter) closeDevice(d *routeDevice) {
	if d.handle == nil {
		return
	}
	_ = d.handle.Close()
	d.handle = nil
	d.ready = nil
}

// close shuts the route down. It is idempotent: the first call releases the active device, later
// calls return ErrAudioDeviceClosed so a caller still using the route gets a context error.
func (r *deviceRouter) close() error {
	r.switchMu.Lock()
	defer r.switchMu.Unlock()

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ErrAudioDeviceClosed
	}
	r.closed = true
	active := r.active
	r.active = nil
	r.generation++
	r.mu.Unlock()

	if active != nil {
		r.closeDevice(active)
	}
	r.opener.OnDevicesChanged(nil)
	return nil
}

// reopen clears a closed route and opens a device again, rebuilding the device after a shutdown.
// It fails if the route is not closed or no device can be opened.
func (r *deviceRouter) reopen() error {
	r.switchMu.Lock()
	defer r.switchMu.Unlock()

	r.mu.Lock()
	if !r.closed {
		r.mu.Unlock()
		return fmt.Errorf("audio: route is not closed")
	}
	r.closed = false
	r.mu.Unlock()

	r.opener.OnDevicesChanged(r.onDevicesChanged)

	infos, err := r.opener.EnumerateDevices()
	if err != nil {
		return err
	}
	r.mu.Lock()
	r.refreshDevicesLocked(infos)
	target := r.preferredTargetLocked(r.pendingID)
	if target == nil {
		target = r.preferredTargetLocked("")
	}
	r.mu.Unlock()
	if target == nil {
		return ErrNoAudioDevice
	}
	if err := r.openDevice(target); err != nil {
		return err
	}
	r.mu.Lock()
	r.active = target
	r.generation++
	r.mu.Unlock()

	// Reattach every surviving player so playback resumes where it stopped.
	for _, p := range r.factory.routedPlayers() {
		p.m.Lock()
		wasPending := p.pendingPlay
		if err := r.bindPlayerLocked(p, target, 0); err != nil {
			p.m.Unlock()
			return err
		}
		if wasPending {
			p.playAndStartStopwatch()
			p.pendingPlay = false
			p.context.addPlayingPlayer(p)
		}
		p.m.Unlock()
	}
	return nil
}

// legacyDriverHandle adapts a pre-routing driver (the test dummy) to deviceHandle. Its players do
// not implement io.Closer, so Close is a no-op.
type legacyDriverHandle struct {
	driver driver
}

func (h legacyDriverHandle) NewPlayer(r io.Reader) routedDevicePlayer {
	return legacyPlayerHandle{player: h.driver.NewPlayer(r)}
}

func (h legacyDriverHandle) Suspend() error { return h.driver.Suspend() }
func (h legacyDriverHandle) Resume() error  { return h.driver.Resume() }
func (h legacyDriverHandle) Err() error     { return h.driver.Err() }
func (h legacyDriverHandle) Close() error   { return nil }

// legacyPlayerHandle adapts a pre-routing player that has no Close to routedDevicePlayer.
type legacyPlayerHandle struct {
	player
}

func (legacyPlayerHandle) Close() error { return nil }

// routedStream is what a routed player hands to its device player. From the device's side it is a
// plain float32 stereo stream at the device's rate; internally it chains
//
//	timeStream -> gainStream -> resampler (when rates differ)
//
// and keeps enough accounting to report the buffered size in time-stream bytes, which is what the
// position logic expects.
type routedStream struct {
	inner      io.Reader // the device-facing reader (resampler or gain)
	gain       *gainStream
	resampling *convert.Resampling // nil when the device runs at the context rate

	// readMu serializes reads: the device's own pull and a migration's drain can otherwise read
	// the resampler concurrently, and the resampler keeps mutable interpolation state.
	readMu sync.Mutex
}

// newRoutedStream builds the stream chain for one player on one device. src is the player's
// timeStream; deviceRate is the rate the device actually runs at (resampling is inserted when it
// differs from the context rate).
func newRoutedStream(src io.Reader, contextRate, deviceRate int, gain *gainStream) *routedStream {
	rs := &routedStream{gain: gain}

	var inner io.Reader = gain
	if deviceRate != contextRate {
		r := convert.NewResampling(gain, -1, contextRate, deviceRate, bitDepthInBytesFloat32)
		rs.resampling = r
		inner = r
	}
	rs.inner = inner
	return rs
}

func (s *routedStream) Read(p []byte) (int, error) {
	s.readMu.Lock()
	defer s.readMu.Unlock()
	return s.inner.Read(p)
}

// Seek resets every stage and seeks the underlying time stream.
func (s *routedStream) Seek(offset int64, whence int) (int64, error) {
	s.readMu.Lock()
	defer s.readMu.Unlock()
	seeker, ok := s.gain.source().(io.Seeker)
	if !ok {
		return 0, fmt.Errorf("audio: routed stream source must be io.Seeker: %w", errors.ErrUnsupported)
	}
	n, err := seeker.Seek(offset, whence)
	if err != nil {
		return n, err
	}
	s.gain.reset()
	if s.resampling != nil {
		if _, err := s.resampling.Seek(0, io.SeekStart); err != nil {
			return n, err
		}
	}
	return n, nil
}

// bufferedSourceBytes reports how many time-stream bytes are buffered between the device player's
// playback position and the time stream's read position.
//
// The stream position already counts every frame read from the time stream, including frames
// handed to the device player. Those handed-off frames must not be reported as buffered again, or
// the position would cancel itself out. Only frames still held inside the processing chain count:
// frames the resampler pulled but has not emitted, and frames the device player has not played.
func (s *routedStream) bufferedSourceBytes(deviceBufferedBytes, deviceFrameSize, deviceRate, contextRate int) int {
	// Take the read lock so the resampler's interpolation state cannot change underneath us.
	s.readMu.Lock()
	defer s.readMu.Unlock()

	// Bytes the device still holds, translated back to context-rate frames. Integer division keeps
	// the rounding error below one frame even when the rates do not divide evenly.
	deviceFrames := int64(deviceBufferedBytes / deviceFrameSize)
	srcFramesFromDevice := deviceFrames * int64(contextRate) / int64(deviceRate)

	// Frames the resampler has pulled from the gain stage but not emitted yet.
	var pending int64
	if s.resampling != nil {
		pending = s.resampling.BufferedSourceFrames()
	}
	total := pending + srcFramesFromDevice
	return int(total * 8)
}

// fadePhase identifies which side of a switch a gain stream is executing.
type fadePhase int

const (
	fadePhaseNone fadePhase = iota
	fadePhaseOut
	fadePhaseIn
)

// gainStream applies a linear gain ramp to float32 stereo frames while leaving every other byte
// untouched. Fades are driven by the frames Read consumes, so they are sample-accurate and
// independent of wall-clock timing.
type gainStream struct {
	src io.Reader
	mu  sync.Mutex

	phase       fadePhase
	frame       int64
	totalFrames int64

	// partial holds a sub-frame remainder from a short destination buffer.
	partial []byte
}

func newGainStream(src io.Reader) *gainStream {
	return &gainStream{src: src}
}

func (g *gainStream) source() io.Reader { return g.src }

// startFade begins a fade. A zero total makes the stream pass data unchanged at unity gain.
func (g *gainStream) startFade(phase fadePhase, totalFrames int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.phase = phase
	g.frame = 0
	g.totalFrames = int64(totalFrames)
	if totalFrames == 0 {
		g.phase = fadePhaseNone
	}
}

func (g *gainStream) reset() {
	g.mu.Lock()
	g.phase = fadePhaseNone
	g.frame = 0
	g.partial = g.partial[:0]
	g.mu.Unlock()
}

// abortFade stops any ramp immediately. Frames read afterwards pass through at unity. It is used
// when a player is paused or closed while a switch is driving its fade.
func (g *gainStream) abortFade() {
	g.mu.Lock()
	g.phase = fadePhaseNone
	g.mu.Unlock()
}

// done reports whether a fade in the given phase has been fully consumed.
func (g *gainStream) done(phase fadePhase, totalFrames int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if totalFrames == 0 {
		return true
	}
	return g.phase != phase || g.frame >= g.totalFrames
}

func (g *gainStream) Read(p []byte) (int, error) {
	g.mu.Lock()
	phase := g.phase
	total := g.totalFrames
	frame := g.frame
	g.mu.Unlock()

	if phase == fadePhaseNone || total == 0 {
		return g.src.Read(p)
	}

	frameSize := 8

	// Finish a partial frame left by a previous short buffer first.
	if len(g.partial) > 0 {
		need := frameSize - len(g.partial)
		tmp := make([]byte, frameSize)
		copy(tmp, g.partial)
		_, err := io.ReadFull(g.src, tmp[len(g.partial):])
		if len(p) < need {
			// Unlikely: caller shrank its buffer. Return what fits and keep the rest pending.
			m := copy(p, tmp[:len(p)])
			g.partial = append(g.partial[:0], tmp[m:]...)
			return m, err
		}
		copy(p[:frameSize], tmp)
		g.applyFrame(p[:frameSize], frame, phase, total)
		frame++
		g.partial = g.partial[:0]
		g.advance(frame)
		if err != nil {
			return frameSize, err
		}
		p = p[frameSize:]
		if len(p) == 0 {
			return frameSize, nil
		}
	}

	frames := len(p) / frameSize
	head := frames * frameSize
	n, err := g.src.Read(p[:head])
	gotFrames := n / frameSize
	for i := range gotFrames {
		g.applyFrame(p[i*frameSize:(i+1)*frameSize], frame+int64(i), phase, total)
	}
	if rem := n % frameSize; rem > 0 {
		g.partial = append(g.partial[:0], p[gotFrames*frameSize:n]...)
	}
	frame += int64(gotFrames)
	g.advance(frame)

	if n == 0 && err == nil {
		return 0, nil
	}
	return gotFrames*frameSize + n%frameSize, err
}

// advance records the frame counter and ends a completed ramp. Fade-out rests at the last gain
// (the old player is paused immediately afterwards); fade-in returns to unity once complete.
func (g *gainStream) advance(frame int64) {
	g.mu.Lock()
	g.frame = frame
	if g.totalFrames > 0 && frame >= g.totalFrames {
		if g.phase == fadePhaseIn {
			g.phase = fadePhaseNone
		}
	}
	g.mu.Unlock()
}

// applyFrame scales one stereo float32 frame. gain is linear in (0, 1]: fading out from 1 down to
// 1/total over the ramp's frames, fading in from 1/total up to 1. The first frame is never exact
// silence or full scale, so a total-length ramp spans exactly total frames on both sides.
func (g *gainStream) applyFrame(frame []byte, idx int64, phase fadePhase, total int64) {
	t := float64(idx+1) / float64(total)
	if t > 1 {
		t = 1
	}
	var gain float32
	if phase == fadePhaseOut {
		gain = float32(1 - t)
	} else {
		gain = float32(t)
	}
	for ch := range 2 {
		b := frame[ch*4 : ch*4+4]
		bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		v := math.Float32frombits(bits) * gain
		bits = math.Float32bits(v)
		b[0] = byte(bits)
		b[1] = byte(bits >> 8)
		b[2] = byte(bits >> 16)
		b[3] = byte(bits >> 24)
	}
}

// routedPlayer wraps the device-level player and the routed stream it reads.
type routedPlayer struct {
	player routedDevicePlayer
	stream *routedStream

	// device is the route device this player is bound to.
	device *routeDevice

	deviceRate  int
	frameSize   int
	contextRate int
}

func (p *routedPlayer) Pause()                 { p.player.Pause() }
func (p *routedPlayer) PauseAndStopReading()   { p.player.PauseAndStopReading() }
func (p *routedPlayer) Play()                  { p.player.Play() }
func (p *routedPlayer) IsPlaying() bool        { return p.player.IsPlaying() }
func (p *routedPlayer) Volume() float64        { return p.player.Volume() }
func (p *routedPlayer) SetVolume(v float64)    { p.player.SetVolume(v) }
func (p *routedPlayer) Err() error             { return p.player.Err() }
func (p *routedPlayer) SetBufferSize(size int) { p.player.SetBufferSize(size) }
func (p *routedPlayer) Close() error           { return p.player.Close() }

func (p *routedPlayer) Seek(offset int64, whence int) (int64, error) {
	return p.stream.Seek(offset, whence)
}

// BufferedSize translates the device's buffered bytes back into time-stream (source) bytes, so
// that playerImpl's position math keeps working when rates or channel counts differ.
func (p *routedPlayer) BufferedSize() int {
	return p.stream.bufferedSourceBytes(p.player.BufferedSize(), p.frameSize, p.deviceRate, p.contextRate)
}
