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
	"math"
	"runtime"

	"github.com/hajimehoshi/ebiten/v2/audio/internal/convert"
)

// ChannelMixMatrix maps a source's channels onto the audio device's output channels.
//
// The matrix has one row per output channel and one column per source channel; each
// coefficient At(out, in) is the gain applied to source channel in when it is summed into
// output channel out. The same output channel receives the weighted sum of every source
// channel. A zero coefficient mutes a contribution, a negative one inverts the phase, and a
// coefficient larger than one amplifies it.
//
// The audio device always has two output channels, so matrices applied to a [Player] must have
// two output channels. The number of source channels declares how many interleaved channels
// the player's stream carries: mono (1), stereo (2), or surround (up to eight).
//
// A matrix additionally carries a scalar gain applied after the matrix and an
// auto-normalization option: when enabled, each output row is divided by the sum of the
// absolute values of its coefficients, so that downmixing several full-scale channels cannot
// exceed full scale.
//
// A ChannelMixMatrix is not safe for concurrent use itself. Pass it to a [Player] or a
// [Context] through SetChannelMixMatrix; the player keeps its own immutable snapshot and the
// matrix can be reused afterward.
type ChannelMixMatrix struct {
	sourceChannels int
	outputChannels int
	coeffs         []float64
	gain           float64
	autoNormalize  bool
}

// NewChannelMixMatrix creates a zero-filled matrix mapping sourceChannels input channels onto
// outputChannels output channels. Every coefficient starts at zero, so every output channel is
// silent until coefficients are set with [ChannelMixMatrix.Set].
//
// NewChannelMixMatrix panics when a channel count is not positive or is larger than eight.
func NewChannelMixMatrix(sourceChannels, outputChannels int) *ChannelMixMatrix {
	if sourceChannels <= 0 || sourceChannels > convert.MaxChannels {
		panic(fmt.Sprintf("audio: source channel count must be in the range [1, %d] but was %d", convert.MaxChannels, sourceChannels))
	}
	if outputChannels <= 0 || outputChannels > convert.MaxChannels {
		panic(fmt.Sprintf("audio: output channel count must be in the range [1, %d] but was %d", convert.MaxChannels, outputChannels))
	}
	return &ChannelMixMatrix{
		sourceChannels: sourceChannels,
		outputChannels: outputChannels,
		coeffs:         make([]float64, sourceChannels*outputChannels),
		gain:           1,
	}
}

// SourceChannels returns the number of source channels the matrix maps from.
func (m *ChannelMixMatrix) SourceChannels() int {
	return m.sourceChannels
}

// OutputChannels returns the number of output channels the matrix maps onto.
func (m *ChannelMixMatrix) OutputChannels() int {
	return m.outputChannels
}

// At returns the coefficient from source channel in to output channel out.
// At returns zero for an out-of-range channel index, instead of panicking.
func (m *ChannelMixMatrix) At(out, in int) float64 {
	if out < 0 || out >= m.outputChannels || in < 0 || in >= m.sourceChannels {
		return 0
	}
	return m.coeffs[out*m.sourceChannels+in]
}

// Set sets the coefficient from source channel in to output channel out.
//
// A NaN or infinite coefficient is rejected: such a value could only produce NaN or infinite
// samples, so Set returns an error and leaves the coefficient unchanged.
//
// Set returns an error for an out-of-range channel index.
func (m *ChannelMixMatrix) Set(out, in int, coeff float64) error {
	if out < 0 || out >= m.outputChannels {
		return fmt.Errorf("audio: output channel index %d is out of range for %d output channels", out, m.outputChannels)
	}
	if in < 0 || in >= m.sourceChannels {
		return fmt.Errorf("audio: source channel index %d is out of range for %d source channels", in, m.sourceChannels)
	}
	if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
		return fmt.Errorf("audio: mix coefficient must be finite but was %v", coeff)
	}
	m.coeffs[out*m.sourceChannels+in] = coeff
	return nil
}

// Gain returns the scalar gain applied after the matrix.
func (m *ChannelMixMatrix) Gain() float64 {
	return m.gain
}

// SetGain sets the scalar gain applied after the matrix.
//
// A NaN or infinite gain is rejected: SetGain returns an error and leaves the gain unchanged.
// The gain is not clamped: an extreme but finite gain is accepted, and the player clamps the
// resulting samples to the representable range instead of overflowing.
func (m *ChannelMixMatrix) SetGain(gain float64) error {
	if math.IsNaN(gain) || math.IsInf(gain, 0) {
		return fmt.Errorf("audio: mix gain must be finite but was %v", gain)
	}
	m.gain = gain
	return nil
}

// AutoNormalize reports whether each output row is divided by the sum of the absolute values
// of its coefficients.
func (m *ChannelMixMatrix) AutoNormalize() bool {
	return m.autoNormalize
}

// SetAutoNormalize enables or disables auto-normalization.
func (m *ChannelMixMatrix) SetAutoNormalize(autoNormalize bool) {
	m.autoNormalize = autoNormalize
}

// toMixState converts the matrix into the immutable state the mixing stream consumes.
// The returned coefficient slice is a copy, so later mutations of the matrix never reach a
// running player.
func (m *ChannelMixMatrix) toMixState() convert.MixState {
	coeffs := make([]float64, len(m.coeffs))
	copy(coeffs, m.coeffs)
	return convert.MixState{
		SourceChannels: m.sourceChannels,
		OutputChannels: m.outputChannels,
		Coeffs:         coeffs,
		Gain:           m.gain,
		AutoNormalize:  m.autoNormalize,
	}
}

// matrixFromMixState is the inverse of toMixState, used by the getters to return snapshots of
// the states stored on a player or context.
func matrixFromMixState(s convert.MixState) *ChannelMixMatrix {
	coeffs := make([]float64, len(s.Coeffs))
	copy(coeffs, s.Coeffs)
	return &ChannelMixMatrix{
		sourceChannels: s.SourceChannels,
		outputChannels: s.OutputChannels,
		coeffs:         coeffs,
		gain:           s.Gain,
		autoNormalize:  s.AutoNormalize,
	}
}

// defaultMasterMixState is the context's master configuration: the identity matrix with unit
// gain and no normalization, which leaves every player's output untouched.
func defaultMasterMixState() convert.MixState {
	return convert.DefaultMixState(channelCount)
}

func (c *Context) loadMasterMixState() convert.MixState {
	if s := c.masterMixState.Load(); s != nil {
		return *s
	}
	return defaultMasterMixState()
}

func (c *Context) storeMasterMixState(s convert.MixState) {
	c.masterMixState.Store(&s)
}

// SetChannelMixMatrix sets the context's master channel mixing matrix, which is applied to
// every player after that player's own matrix.
//
// The matrix must have two output channels, matching the audio device. A nil matrix restores
// the default identity master matrix.
//
// The change takes effect at the next audio frame of every player, including players already
// playing; no audio is dropped and playback timing is unaffected.
//
// SetChannelMixMatrix returns an error when the matrix does not have two output channels.
func (c *Context) SetChannelMixMatrix(matrix *ChannelMixMatrix) error {
	if matrix == nil {
		c.storeMasterMixState(defaultMasterMixState())
		return c.applyMasterMixState()
	}
	if matrix.outputChannels != channelCount {
		return fmt.Errorf("audio: a context mix matrix must have %d output channels but had %d", channelCount, matrix.outputChannels)
	}
	c.storeMasterMixState(matrix.toMixState())
	return c.applyMasterMixState()
}

// ChannelMixMatrix returns a copy of the context's master channel mixing matrix.
func (c *Context) ChannelMixMatrix() *ChannelMixMatrix {
	return matrixFromMixState(c.loadMasterMixState())
}

// SetChannelMixGain sets the scalar gain of the context's master matrix, leaving the matrix
// coefficients and the auto-normalization setting untouched.
//
// A NaN or infinite gain is rejected with an error and leaves the current gain unchanged.
func (c *Context) SetChannelMixGain(gain float64) error {
	if math.IsNaN(gain) || math.IsInf(gain, 0) {
		return fmt.Errorf("audio: mix gain must be finite but was %v", gain)
	}
	s := c.loadMasterMixState()
	s.Gain = gain
	c.storeMasterMixState(s)
	return c.applyMasterMixState()
}

// ChannelMixGain returns the scalar gain of the context's master matrix.
func (c *Context) ChannelMixGain() float64 {
	return c.loadMasterMixState().Gain
}

// SetChannelMixAutoNormalize enables or disables auto-normalization of the context's master
// matrix, leaving the coefficients and the gain untouched.
func (c *Context) SetChannelMixAutoNormalize(autoNormalize bool) error {
	s := c.loadMasterMixState()
	s.AutoNormalize = autoNormalize
	c.storeMasterMixState(s)
	return c.applyMasterMixState()
}

// ChannelMixAutoNormalize reports whether auto-normalization is enabled on the context's
// master matrix.
func (c *Context) ChannelMixAutoNormalize() bool {
	return c.loadMasterMixState().AutoNormalize
}

// applyMasterMixState makes the current master state reach the mixer of every existing player.
// Players whose stream has not been created yet pick the state up lazily.
//
// The player list is snapshotted without holding the context lock while touching players, so
// a player method cannot deadlock against the context.
func (c *Context) applyMasterMixState() error {
	c.m.Lock()
	players := make([]*playerImpl, 0, len(c.allPlayers))
	for p := range c.allPlayers {
		players = append(players, p)
	}
	c.m.Unlock()

	for _, p := range players {
		if err := p.updateMixerForMasterState(); err != nil {
			return err
		}
	}
	return nil
}

// SetChannelMixMatrix sets this player's channel mixing matrix.
//
// The matrix maps the player's source channels onto the device's two output channels, so it
// must have two output channels. Its source channel count declares the stream's layout: the
// first matrix set on a player fixes that count, and later matrices must match it.
//
// A nil matrix removes the player's own matrix; the context's master matrix still applies.
// When neither the player nor the context carries a non-default configuration, a stereo
// stream passes through untouched, matching the behavior of players without a matrix.
//
// The change takes effect at the next audio frame without dropping samples or affecting
// playback timing.
//
// SetChannelMixMatrix returns an error when the matrix has the wrong output dimension, or
// when its source channel count differs from the count already configured for the player.
func (p *Player) SetChannelMixMatrix(matrix *ChannelMixMatrix) error {
	defer runtime.KeepAlive(p)
	return p.p.setChannelMixMatrix(matrix)
}

// ChannelMixMatrix returns a copy of the player's own channel mixing matrix, or nil when the
// player has none. The context's master matrix is not included.
func (p *Player) ChannelMixMatrix() *ChannelMixMatrix {
	defer runtime.KeepAlive(p)
	if s := p.p.playerMixState.Load(); s != nil {
		return matrixFromMixState(*s)
	}
	return nil
}

// SetChannelMixGain sets the scalar gain of the player's own matrix, leaving the coefficients
// and the auto-normalization setting untouched. When the player has no matrix, a default
// identity matrix is created to carry the gain.
//
// A NaN or infinite gain is rejected with an error and leaves the current gain unchanged.
func (p *Player) SetChannelMixGain(gain float64) error {
	defer runtime.KeepAlive(p)
	if math.IsNaN(gain) || math.IsInf(gain, 0) {
		return fmt.Errorf("audio: mix gain must be finite but was %v", gain)
	}
	p.p.m.Lock()
	defer p.p.m.Unlock()

	s := p.p.currentOrIdentityPlayerMixState()
	s.Gain = gain
	p.p.playerMixState.Store(&s)
	return p.p.updateMixerLocked()
}

// ChannelMixGain returns the scalar gain of the player's own matrix, which is one when the
// player has no matrix.
func (p *Player) ChannelMixGain() float64 {
	defer runtime.KeepAlive(p)
	if s := p.p.playerMixState.Load(); s != nil {
		return s.Gain
	}
	return 1
}

// SetChannelMixAutoNormalize enables or disables auto-normalization of the player's own
// matrix. When the player has no matrix, a default identity matrix is created to carry the
// setting.
func (p *Player) SetChannelMixAutoNormalize(autoNormalize bool) error {
	defer runtime.KeepAlive(p)
	p.p.m.Lock()
	defer p.p.m.Unlock()

	s := p.p.currentOrIdentityPlayerMixState()
	s.AutoNormalize = autoNormalize
	p.p.playerMixState.Store(&s)
	return p.p.updateMixerLocked()
}

// ChannelMixAutoNormalize reports whether auto-normalization is enabled on the player's own
// matrix, which is false when the player has no matrix.
func (p *Player) ChannelMixAutoNormalize() bool {
	defer runtime.KeepAlive(p)
	if s := p.p.playerMixState.Load(); s != nil {
		return s.AutoNormalize
	}
	return false
}

func (p *playerImpl) setChannelMixMatrix(matrix *ChannelMixMatrix) error {
	p.m.Lock()
	defer p.m.Unlock()

	if matrix == nil {
		p.playerMixState.Store(nil)
		return p.updateMixerLocked()
	}
	if matrix.outputChannels != channelCount {
		return fmt.Errorf("audio: a player mix matrix must have %d output channels but had %d", channelCount, matrix.outputChannels)
	}
	if p.sourceChannels != 0 && matrix.sourceChannels != p.sourceChannels {
		return fmt.Errorf("audio: player source channel count is already %d but the matrix had %d", p.sourceChannels, matrix.sourceChannels)
	}

	s := matrix.toMixState()
	p.sourceChannels = s.SourceChannels
	p.playerMixState.Store(&s)
	return p.updateMixerLocked()
}

// currentOrIdentityPlayerMixState returns a mutable copy of the player's state, or the
// identity state for the player's source channel count when none is set.
// It must be called with p.m locked.
func (p *playerImpl) currentOrIdentityPlayerMixState() convert.MixState {
	if s := p.playerMixState.Load(); s != nil {
		return *s
	}
	if p.sourceChannels == 0 {
		p.sourceChannels = channelCount
	}
	return convert.DefaultMixState(p.sourceChannels)
}

// updateMixerForMasterState applies a master state change. It creates the mixer for a stream
// that was created before the context carried a non-default master configuration.
func (p *playerImpl) updateMixerForMasterState() error {
	p.m.Lock()
	defer p.m.Unlock()
	return p.updateMixerLocked()
}

// updateMixerLocked creates or refreshes the channel mixer according to the current player and
// master states. It must be called with p.m locked.
//
// A stereo stream without a player matrix stays on the legacy path while the master state is
// the default, so an unconfigured player produces byte-identical output to the old behavior.
func (p *playerImpl) updateMixerLocked() error {
	if p.closed {
		return nil
	}

	playerState := p.playerMixState.Load()
	masterState := p.context.loadMasterMixState()
	if p.sourceChannels == 0 {
		p.sourceChannels = channelCount
	}

	needsMixer := playerState != nil || !masterState.IsDefault()
	if !needsMixer {
		return nil
	}

	if p.stream == nil {
		// The stream is created lazily; ensureStream installs the mixer then.
		return nil
	}

	if p.mixer != nil {
		// The mixer reads the atomic state pointers directly. Nothing to rebuild; the next
		// Read picks the new states up at a frame boundary.
		return nil
	}

	if playerState != nil && playerState.SourceChannels != p.sourceChannels {
		return fmt.Errorf("audio: mix matrix has %d source channels but the player has %d", playerState.SourceChannels, p.sourceChannels)
	}

	mixer := convert.NewChannelMixer(p.stream.reader(), p.sourceChannels, p.bitDepthInBytesRaw, &p.playerMixState, &p.context.masterMixState)
	p.stream.setReader(mixer)
	p.mixer = mixer
	return nil
}
