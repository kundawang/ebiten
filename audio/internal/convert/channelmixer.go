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

package convert

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"

	"github.com/hajimehoshi/ebiten/v2/internal/mathutil"
)

const (
	// outputChannelCount is the number of channels of the device, and therefore of every
	// ChannelMixer's output.
	outputChannelCount = 2
	// outputFrameSize is the byte size of one output frame: two 32bit float samples.
	outputFrameSize = outputChannelCount * 4

	// MaxChannels is the maximum number of source channels a ChannelMixer accepts.
	MaxChannels = 8
)

// Bit depth of a ChannelMixer's source in bytes per sample per channel.
const (
	BitDepthInBytesInt16   = 2
	BitDepthInBytesFloat32 = 4
)

// MixState is an immutable snapshot of a channel mixing configuration: a matrix mapping
// SourceChannels input channels to OutputChannels output channels, an additional scalar gain,
// and whether the matrix is auto-normalized.
//
// Coefficients are stored row-major: the coefficient from source channel in to output channel
// out is at Coeffs[out*SourceChannels+in].
//
// A MixState is safe to copy; its Coeffs slice must never be mutated after the state is stored
// in an atomic pointer.
type MixState struct {
	SourceChannels int
	// OutputChannels is always 2 for states used with a ChannelMixer.
	OutputChannels int
	Coeffs         []float64
	Gain           float64
	AutoNormalize  bool
}

// DefaultMixState returns the identity mixing state for the given channel count. For a stereo
// source this is the untouched passthrough; for fewer channels it maps the first channels
// one-to-one and leaves the remaining output channels silent.
func DefaultMixState(channels int) MixState {
	if channels <= 0 {
		panic(fmt.Sprintf("convert: channel count must be positive but was %d", channels))
	}
	coeffs := make([]float64, outputChannelCount*channels)
	for out := 0; out < outputChannelCount && out < channels; out++ {
		coeffs[out*channels+out] = 1
	}
	return MixState{
		SourceChannels: channels,
		OutputChannels: outputChannelCount,
		Coeffs:         coeffs,
		Gain:           1,
	}
}

// IsDefault reports whether s is the untouched passthrough configuration: identity matrix,
// unit gain, and no auto-normalization.
func (s MixState) IsDefault() bool {
	if s.AutoNormalize || s.Gain != 1 || s.SourceChannels != outputChannelCount || s.OutputChannels != outputChannelCount {
		return false
	}
	identity := DefaultMixState(outputChannelCount)
	if len(s.Coeffs) != len(identity.Coeffs) {
		return false
	}
	for i, c := range s.Coeffs {
		if c != identity.Coeffs[i] {
			return false
		}
	}
	return true
}

// scaledRow returns row out of the matrix with auto-normalization and gain applied.
//
// Auto-normalization divides a row by the sum of the absolute values of its coefficients, so
// that downmixing several full-scale channels cannot exceed full scale. A row whose
// coefficients sum to zero (an all-zero row, or coefficients canceling each other) is kept as
// is: normalizing it would mean dividing by zero.
func scaledRow(s MixState, out int) []float64 {
	row := s.Coeffs[out*s.SourceChannels : (out+1)*s.SourceChannels]
	scaled := make([]float64, len(row))
	var sum float64
	if s.AutoNormalize {
		for _, c := range row {
			sum += math.Abs(c)
		}
	}
	for i, c := range row {
		v := c
		if s.AutoNormalize && sum != 0 {
			v = c / sum
		}
		scaled[i] = v * s.Gain
	}
	return scaled
}

// CombineMixStates combines the per-player state and the context's master state into the
// effective 2×SourceChannels matrix applied to a source.
//
// The player matrix runs first (source channels → stereo), then the master matrix (stereo →
// stereo). Each matrix's own auto-normalization and gain are applied before the two are
// multiplied, so the result equals applying them one after the other to the stream.
//
// A nil player state means the identity player matrix for a stereo source.
func CombineMixStates(player *MixState, master MixState) MixState {
	if player == nil {
		identity := DefaultMixState(outputChannelCount)
		player = &identity
	}

	playerRows := make([][]float64, outputChannelCount)
	for out := range playerRows {
		playerRows[out] = scaledRow(*player, out)
	}
	masterRows := make([][]float64, outputChannelCount)
	for out := range masterRows {
		masterRows[out] = scaledRow(master, out)
	}

	n := player.SourceChannels
	coeffs := make([]float64, outputChannelCount*n)
	for out := 0; out < outputChannelCount; out++ {
		for in := 0; in < n; in++ {
			var v float64
			for k := 0; k < outputChannelCount; k++ {
				v += masterRows[out][k] * playerRows[k][in]
			}
			coeffs[out*n+in] = v
		}
	}

	return MixState{
		SourceChannels: n,
		OutputChannels: outputChannelCount,
		Coeffs:         coeffs,
		// Normalization and gain are already folded into the coefficients.
		Gain:          1,
		AutoNormalize: false,
	}
}

// ChannelMixer reads an interleaved PCM source with sourceChannels channels and writes an
// interleaved 32bit float little-endian stereo stream whose samples are the source samples
// multiplied by the mixing matrix.
//
// The source is either 32bit float little-endian (bitDepthInBytes = 4) or signed 16bit
// little-endian (bitDepthInBytes = 2). The output always has two channels, matching the audio
// device.
//
// Read and Seek take the same mutex; the mixing states can be replaced concurrently with Read.
type ChannelMixer struct {
	source          io.Reader
	sourceChannels  int
	bitDepthInBytes int

	// player and master are the two states combined into the effective matrix. player may be
	// nil, meaning the identity player matrix.
	player *atomic.Pointer[MixState]
	master *atomic.Pointer[MixState]

	// effective caches the combined state, recomputed only when an input state's pointer
	// changed since the last computation.
	effective     atomic.Pointer[MixState]
	lastPlayerPtr *MixState
	lastMasterPtr *MixState
	cacheMu       sync.Mutex

	// eof reports that the source returned io.EOF. Buffered bytes can still be converted.
	eof bool
	// buf holds source bytes read but not converted yet. A short read can leave a partial
	// source frame pending.
	buf []byte

	m sync.Mutex
}

// NewChannelMixer creates a ChannelMixer over source.
//
// sourceChannels is the source's number of interleaved channels. bitDepthInBytes is
// BitDepthInBytesFloat32 or BitDepthInBytesInt16. player is the initial per-player mixing
// state (nil for the identity) and master is the context's master state; both must have the
// dimensions implied by sourceChannels and the stereo output.
func NewChannelMixer(source io.Reader, sourceChannels, bitDepthInBytes int, player *atomic.Pointer[MixState], master *atomic.Pointer[MixState]) *ChannelMixer {
	if sourceChannels <= 0 || sourceChannels > MaxChannels {
		panic(fmt.Sprintf("convert: source channel count must be in the range [1, %d] but was %d", MaxChannels, sourceChannels))
	}
	if bitDepthInBytes != BitDepthInBytesFloat32 && bitDepthInBytes != BitDepthInBytesInt16 {
		panic(fmt.Sprintf("convert: bit depth must be %d or %d bytes but was %d", BitDepthInBytesInt16, BitDepthInBytesFloat32, bitDepthInBytes))
	}
	if master == nil {
		panic("convert: master mixing state must not be nil")
	}
	return &ChannelMixer{
		source:          source,
		sourceChannels:  sourceChannels,
		bitDepthInBytes: bitDepthInBytes,
		player:          player,
		master:          master,
	}
}

func (m *ChannelMixer) sourceFrameSize() int {
	return m.sourceChannels * m.bitDepthInBytes
}

// state returns the effective mixing state, recomputing it only when an input state changed.
// States are immutable and always replaced by a fresh allocation, so pointer identity is a
// reliable change marker.
func (m *ChannelMixer) state() MixState {
	var playerState *MixState
	if m.player != nil {
		playerState = m.player.Load()
	}
	masterState := m.master.Load()

	m.cacheMu.Lock()
	defer m.cacheMu.Unlock()
	if eff := m.effective.Load(); eff != nil && playerState == m.lastPlayerPtr && masterState == m.lastMasterPtr {
		return *eff
	}

	eff := CombineMixStates(playerState, *masterState)
	m.effective.Store(&eff)
	m.lastPlayerPtr = playerState
	m.lastMasterPtr = masterState
	return eff
}

// Read implements io.Reader.
func (m *ChannelMixer) Read(b []byte) (int, error) {
	m.m.Lock()
	defer m.m.Unlock()

	frameSize := m.sourceFrameSize()
	if m.eof && len(m.buf) < frameSize {
		return 0, io.EOF
	}
	if len(b) == 0 {
		return 0, nil
	}

	// Read source bytes until at least one source frame is buffered, or the source is
	// exhausted. A source that keeps returning (0, nil) must not make Read spin forever.
	limit := max(len(b)/outputFrameSize, 1) * frameSize
	var readErr error
	for len(m.buf) < limit && !m.eof {
		origLen := len(m.buf)
		if cap(m.buf) < limit {
			m.buf = append(m.buf, make([]byte, limit-origLen)...)
		}

		n, err := m.source.Read(m.buf[origLen:limit])
		m.buf = m.buf[:origLen+n]
		if err != nil && err != io.EOF {
			readErr = err
			break
		}
		if err == io.EOF {
			m.eof = true
		}
		if len(m.buf) >= frameSize || n == 0 {
			break
		}
	}

	if len(b) < outputFrameSize {
		if readErr != nil {
			return 0, readErr
		}
		if m.eof && len(m.buf) < frameSize {
			return 0, io.EOF
		}
		return 0, io.ErrShortBuffer
	}

	frames := min(len(m.buf)/frameSize, len(b)/outputFrameSize)
	state := m.state()

	for f := 0; f < frames; f++ {
		var in [MaxChannels]float64
		for ch := 0; ch < m.sourceChannels; ch++ {
			in[ch] = m.readSample(f*frameSize + ch*m.bitDepthInBytes)
		}

		for out := 0; out < outputChannelCount; out++ {
			var v float64
			for ch := 0; ch < m.sourceChannels; ch++ {
				c := state.Coeffs[out*m.sourceChannels+ch]
				sample := in[ch]
				// A non-finite source sample must not poison the matrix math. Treat it as
				// silence for this frame instead of producing NaN or Inf on the device.
				if math.IsNaN(sample) || math.IsInf(sample, 0) {
					sample = 0
				}
				v += c * sample
			}
			// Clamp to the float range so that no overflow, including from extreme gains,
			// reaches the device as Inf.
			if v > 1 {
				v = 1
			} else if v < -1 {
				v = -1
			} else if math.IsNaN(v) {
				v = 0
			}
			bits := math.Float32bits(float32(v))
			pos := f*outputFrameSize + out*4
			b[pos] = byte(bits)
			b[pos+1] = byte(bits >> 8)
			b[pos+2] = byte(bits >> 16)
			b[pos+3] = byte(bits >> 24)
		}
	}

	// Keep the trailing partial frame for the next read.
	used := frames * frameSize
	copy(m.buf, m.buf[used:])
	m.buf = m.buf[:len(m.buf)-used]

	n := frames * outputFrameSize
	if readErr != nil {
		return n, readErr
	}
	if m.eof {
		return n, io.EOF
	}
	return n, nil
}

// readSample decodes one input sample at byte offset off. The caller guarantees off is in
// bounds of a whole buffered source frame.
func (m *ChannelMixer) readSample(off int) float64 {
	switch m.bitDepthInBytes {
	case BitDepthInBytesFloat32:
		bits := uint32(m.buf[off]) | uint32(m.buf[off+1])<<8 | uint32(m.buf[off+2])<<16 | uint32(m.buf[off+3])<<24
		return float64(math.Float32frombits(bits))
	case BitDepthInBytesInt16:
		v := int16(m.buf[off]) | int16(m.buf[off+1])<<8
		return float64(v) / (1 << 15)
	default:
		panic("convert: invalid bit depth")
	}
}

// Seek implements io.Seeker.
//
// Offsets are in the output's bytes (stereo float32 frames). The corresponding source
// position is derived from the fixed frame-size ratio, so seeking is sample-accurate and
// introduces no extra latency. Seeking a non-seekable source returns an error wrapping
// errors.ErrUnsupported.
func (m *ChannelMixer) Seek(offset int64, whence int) (int64, error) {
	s, ok := m.source.(io.Seeker)
	if !ok {
		return 0, fmt.Errorf("convert: source must be io.Seeker: %w", errors.ErrUnsupported)
	}

	m.m.Lock()
	defer m.m.Unlock()

	frameSize := int64(m.sourceFrameSize())

	// outputBase is the output-position base for whence, after rounding to a frame.
	var outputBase int64
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		cur, err := s.Seek(0, io.SeekCurrent)
		if err != nil {
			return 0, err
		}
		// The buffered bytes have already been consumed from the source's point of view.
		b, ok := mathutil.Mul((cur-int64(len(m.buf)))/frameSize, outputFrameSize)
		if !ok {
			return 0, fmt.Errorf("convert: position overflows int64")
		}
		outputBase = b
		// A query does not seek and keeps the buffered bytes.
		if offset == 0 {
			return outputBase, nil
		}
	case io.SeekEnd:
		cur, err := s.Seek(0, io.SeekCurrent)
		if err != nil {
			return 0, err
		}
		end, err := s.Seek(0, io.SeekEnd)
		if err != nil {
			return 0, err
		}
		// Undo the probe so that a rejected or adjusted seek leaves the source where it was.
		if _, err := s.Seek(cur, io.SeekStart); err != nil {
			return 0, err
		}
		b, ok := mathutil.Mul(end/frameSize, outputFrameSize)
		if !ok {
			return 0, fmt.Errorf("convert: position overflows int64")
		}
		outputBase = b
	default:
		return 0, fmt.Errorf("convert: whence must be io.SeekStart, io.SeekCurrent, or io.SeekEnd but was %d", whence)
	}

	if _, ok := mathutil.AddForSeek(outputBase, offset); !ok {
		return 0, fmt.Errorf("convert: invalid seek position")
	}

	// Round the requested output position down to a whole output frame, then translate it
	// into source units.
	outputFrames := mathutil.FloorDiv(outputBase+offset, outputFrameSize)
	srcPos, ok := mathutil.Mul(outputFrames, frameSize)
	if !ok {
		return 0, fmt.Errorf("convert: position overflows int64")
	}

	if whence == io.SeekCurrent {
		srcPos -= int64(len(m.buf))
	}

	if _, err := s.Seek(srcPos, io.SeekStart); err != nil {
		return 0, err
	}

	m.buf = m.buf[:0]
	m.eof = false

	outPos, ok := mathutil.Mul(outputFrames, outputFrameSize)
	if !ok {
		return 0, fmt.Errorf("convert: position overflows int64")
	}
	return outPos, nil
}
