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
	"fmt"
	"io"
	"sort"
)

// routedPlayers returns every playerImpl that has taken part in routing (it has a gain stream),
// in the deterministic order assigned at creation. Closed players are excluded.
func (f *playerFactory) routedPlayers() []*playerImpl {
	f.m.Lock()
	router := f.router
	f.m.Unlock()
	if router == nil {
		return nil
	}

	// The context owns the live players. Playing players are always tracked; paused players are
	// not, so also include routed players remembered by the router itself.
	c := CurrentContext()
	var out []*playerImpl
	seen := map[*playerImpl]struct{}{}

	c.m.Lock()
	for p := range c.playingPlayers {
		// Include every live playing player, including one created before routing existed
		// (gain == nil): the migration promotes it onto a routed device rather than leaving it on
		// the device that is about to be closed.
		if !p.closed {
			out = append(out, p)
			seen[p] = struct{}{}
		}
	}
	c.m.Unlock()

	router.mu.Lock()
	for p := range router.knownPlayers {
		if _, ok := seen[p]; ok {
			continue
		}
		// Read closed without p.m: closed is set only under p.m and this is a best-effort liveness
		// filter; a player closed concurrently is harmless if briefly included, since every bind
		// path re-checks the state under the lock.
		if !p.closed {
			out = append(out, p)
		}
	}
	router.mu.Unlock()

	sort.Slice(out, func(i, j int) bool { return out[i].routeSeq < out[j].routeSeq })
	return out
}

// attachPlayer creates the device player for p on the router's active device. It is called from
// ensurePlayer while p.m is held, so it calls the lock-free bindPlayerLocked directly.
func (r *deviceRouter) attachPlayer(p *playerImpl) error {
	r.mu.Lock()
	active := r.active
	r.mu.Unlock()
	if active == nil {
		// No device open: the player stays without an underlying player, exactly like the
		// pre-routing behavior before the first device exists. A later switch, hot-plug event or
		// sweep retries the attachment.
		return nil
	}
	return r.bindPlayerLocked(p, active, 0)
}

// bindPlayer builds the stream chain for p on d and creates its device player. It takes p.m
// itself, so it is safe from both the migration path and the sweep's attach path. A previous
// routed player is released defensively, so a concurrent attachment cannot leak a device player.
func (r *deviceRouter) bindPlayer(p *playerImpl, d *routeDevice, fadeInFrames int) error {
	p.m.Lock()
	defer p.m.Unlock()
	return r.bindPlayerLocked(p, d, fadeInFrames)
}

// bindPlayerLocked is the body of bindPlayer. The caller must hold p.m.
func (r *deviceRouter) bindPlayerLocked(p *playerImpl, d *routeDevice, fadeInFrames int) error {
	if p.routed != nil {
		p.routed.player.PauseAndStopReading()
		_ = p.routed.player.Close()
		p.routed = nil
		p.player = nil
	}

	if p.gain == nil {
		p.gain = newGainStream(p.stream)
	}

	rate := r.deviceRate(d)
	rs := newRoutedStream(p.stream, r.sampleRate, rate, p.gain)
	dp := d.handle.NewPlayer(rs)
	if dp == nil {
		return fmt.Errorf("audio: device %q failed to create a player", d.info.ID)
	}

	rp := &routedPlayer{
		player:      dp,
		stream:      rs,
		device:      d,
		deviceRate:  rate,
		frameSize:   8,
		contextRate: r.sampleRate,
	}
	if p.initBufferSize != 0 {
		dp.SetBufferSize(p.initBufferSize)
		p.initBufferSize = 0
	}
	dp.SetVolume(p.volume)

	p.routed = rp
	p.player = rp
	if fadeInFrames > 0 {
		p.gain.startFade(fadePhaseIn, fadeInFrames)
	} else {
		p.gain.startFade(fadePhaseNone, 0)
	}

	r.mu.Lock()
	if r.knownPlayers == nil {
		r.knownPlayers = map[*playerImpl]struct{}{}
	}
	r.knownPlayers[p] = struct{}{}
	r.mu.Unlock()
	return nil
}

// deviceRate returns the rate the device runs at. The opener opens every device at the context
// rate; a virtual device used in tests may declare a different rate, which drives the resampler.
func (r *deviceRouter) deviceRate(d *routeDevice) int {
	if d.info.Capabilities.MinSampleRate > 0 &&
		!d.info.Capabilities.supports(r.sampleRate, r.channelNum) {
		// Pick the supported rate closest to the context rate within the device's range.
		rate := clamp(r.sampleRate, d.info.Capabilities.MinSampleRate, d.info.Capabilities.MaxSampleRate)
		return rate
	}
	return r.sampleRate
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// beginFadeOut starts the fade-out for a playing player. Paused players have nothing playing and
// are moved without a fade.
func (r *deviceRouter) beginFadeOut(p *playerImpl, frames int) {
	p.m.Lock()
	defer p.m.Unlock()
	if p.routed == nil || p.gain == nil {
		return
	}
	if p.isPlaying() {
		p.gain.startFade(fadePhaseOut, frames)
	}
}

// driveFades pulls frames from the routed device players until every relevant fade has completed.
//
// Real devices consume frames on their own audio clock and cannot be driven by the router. For
// them the gain ramp finishes asynchronously and driveFades only waits on the virtual/pull-based
// handles the tests use. A pull handle exposes drainForTesting; anything else returns immediately.
func (r *deviceRouter) driveFades(players []*playerImpl, phase fadePhase, frames int, d *routeDevice) {
	if frames == 0 {
		return
	}
	puller, ok := d.handle.(interface {
		drainFades(players []*playerImpl, phase fadePhase, frames int)
	})
	if !ok {
		return
	}
	puller.drainFades(players, phase, frames)
}

// detachFromOldDevice pauses and releases the player's old device player after fade-out. It also
// handles a player created before routing existed, whose device player is a plain driver player
// without a routed wrapper.
func (r *deviceRouter) detachFromOldDevice(p *playerImpl) {
	p.m.Lock()
	defer p.m.Unlock()
	if p.player == nil {
		return
	}
	wasPlaying := p.isPlaying()
	p.player.PauseAndStopReading()
	if closer, ok := p.player.(io.Closer); ok {
		_ = closer.Close()
	}
	p.player = nil
	p.routed = nil
	if wasPlaying {
		// Keep the player tracked as playing: Play state is continuous across the switch.
		p.pendingPlay = true
	}
}

// detachWithoutReplacement releases a player when its device vanished and no replacement exists.
// The player becomes pending: it restarts automatically when a device becomes available. Until
// then IsPlaying reports false, since nothing is actually producing sound.
func (r *deviceRouter) detachWithoutReplacement(p *playerImpl) {
	p.m.Lock()
	defer p.m.Unlock()
	if p.player == nil {
		return
	}
	p.player.PauseAndStopReading()
	if closer, ok := p.player.(io.Closer); ok {
		_ = closer.Close()
	}
	p.player = nil
	p.routed = nil
	// Mark pending without staying in the playing set: with no device there is nothing playing,
	// but the request survives and is restarted on the next successful attachment.
	p.pendingPlay = true
	p.context.removePlayingPlayer(p)
}

// attachToNewDevice binds the player to the new device and restarts playback with a fade-in.
func (r *deviceRouter) attachToNewDevice(p *playerImpl, d *routeDevice, fadeInFrames int) error {
	p.m.Lock()
	// A concurrent attachment (for example the sweep starting a pending play) may have already
	// bound this player to the new active device. Nothing to migrate in that case.
	if p.routed != nil && p.routed.device == d {
		p.m.Unlock()
		return nil
	}
	wasPlaying := p.pendingPlay || p.isPlaying()
	p.m.Unlock()

	if err := r.bindPlayer(p, d, fadeInFrames); err != nil {
		return err
	}

	p.m.Lock()
	defer p.m.Unlock()
	if wasPlaying {
		p.playAndStartStopwatch()
		p.pendingPlay = false
		p.context.addPlayingPlayer(p)
	}
	return nil
}

// reattachToOldDevice restores a player onto the old device after a failed new-device attach.
func (r *deviceRouter) reattachToOldDevice(p *playerImpl, d *routeDevice) {
	if err := r.bindPlayer(p, d, 0); err != nil {
		return
	}
	p.m.Lock()
	if p.pendingPlay {
		p.playAndStartStopwatch()
		p.pendingPlay = false
	}
	p.m.Unlock()
}

// onPlayerClosed removes a player from the router's known set. It is called from playerImpl.Close.
func (r *deviceRouter) onPlayerClosed(p *playerImpl) {
	r.mu.Lock()
	delete(r.knownPlayers, p)
	r.mu.Unlock()
}

// routeSuspend and routeResume forward suspend/resume to the active device. They keep the same
// signatures and nil-safety as the factory's pre-routing calls.
func (r *deviceRouter) routeSuspend() error {
	r.mu.Lock()
	active := r.active
	r.mu.Unlock()
	if active == nil {
		return nil
	}
	return active.handle.Suspend()
}

func (r *deviceRouter) routeResume() error {
	r.mu.Lock()
	active := r.active
	r.mu.Unlock()
	if active == nil {
		return nil
	}
	return active.handle.Resume()
}

func (r *deviceRouter) routeError() error {
	r.mu.Lock()
	active := r.active
	r.mu.Unlock()
	if active == nil {
		return nil
	}
	return active.handle.Err()
}
