// Copyright 2017 The Ebiten Authors
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
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// InfiniteLoop represents a looped stream which never ends as long as its source has data to loop.
type InfiniteLoop struct {
	src     io.ReadSeeker
	lstart  int64
	llength int64

	// pos is the position of src. This is ahead of the position of the data returned so far by len(extra).
	pos int64

	bitDepthInBytes int
	bytesPerSample  int

	// extra holds source bytes buffered until a destination can receive a whole sample.
	extra []byte

	// afterLoop is data after the loop.
	afterLoop []byte

	// partialAfterLoop is the part of afterLoop read before a source error. The source position is right after it.
	partialAfterLoop []byte

	// blending represents whether the legacy blending of the loop start and afterLoop is active.
	blending bool

	// legacyBlend reports whether this loop was created by one of the NewInfiniteLoop* functions.
	// Such a loop blends the loop start with all the data after the loop end it reads (at most 256 frames).
	legacyBlend bool

	// fadeFrames is the number of frames of the configured crossfade. It is used when legacyBlend is false.
	fadeFrames int64

	// curve computes the crossfade gains. It is only used when legacyBlend is false.
	curve LoopCrossfadeCurve

	// tailCaptured reports that the configured crossfade has already tried to read the data after the
	// loop end. It keeps a source with no tail from being read again at every loop end.
	tailCaptured bool

	noBlendForTesting bool
}

// LoopCrossfadeCurve returns the gains applied to the loop-start data and to the data after the loop end
// when they are crossfaded.
//
// t is the normalized position in the crossfade window: 0 is the beginning of the loop and 1 is the end
// of the window. The first return value is the gain of the loop-start data and the second is the gain of
// the data after the loop end. A gain is clamped to [0, 1], and non-finite gains are handled as a
// bounded degradation: they are replaced by the linear gains at the same position.
//
// LoopCrossfadeLinear and LoopCrossfadeEqualPower are typical curves.
type LoopCrossfadeCurve func(t float64) (loopStartGain, afterLoopGain float64)

// LoopCrossfadeLinear is a linear crossfade curve.
//
// The gains of the loop-start data and the data after the loop end sum to 1.
func LoopCrossfadeLinear(t float64) (loopStartGain, afterLoopGain float64) {
	return t, 1 - t
}

// LoopCrossfadeEqualPower is an equal-power crossfade curve.
//
// The squared gains of the loop-start data and the data after the loop end sum to 1, so that the
// perceived loudness stays constant during the crossfade.
func LoopCrossfadeEqualPower(t float64) (loopStartGain, afterLoopGain float64) {
	return math.Sincos(t * math.Pi / 2)
}

// LoopCrossfadePlan describes a crossfade at a loop joint.
//
// A crossfade mixes the first FadeFrames frames of every loop repetition with the source data that
// follows the loop end (the tail). At the loop start the output is the tail, and the output ramps to
// the loop data as the window advances. The tail is read once when the loop end is first reached and is
// reused for every repetition, so a crossfade does not read the source again per repetition.
//
// The crossfade does not change the loop's length or its positions: Length and Seek behave exactly as
// they do without a crossfade. A failed Seek leaves the crossfade state unchanged. The tail is
// captured when the loop end is first reached, so data read before that first pass cannot be
// crossfaded. After the capture, the crossfade is a function of the position alone: a read inside the
// window, including one right after a Seek, gets the same output for the same source data.
//
// If the source ends in the middle of the tail, the trailing partial frame is dropped and the window
// is shortened to the whole frames that were read. If the source has no data after the loop end, the
// loop is not crossfaded.
//
// IntroLength and LoopLength are in bytes, like the arguments of NewInfiniteLoopWithIntro, and are
// rounded down to whole frames. One frame is one sample per channel: 4 bytes in the 16-bit integer
// format and 8 bytes in the 32-bit float format.
type LoopCrossfadePlan struct {
	// IntroLength is the length of the intro in bytes. It is rounded down to a whole frame and must be
	// non-negative.
	IntroLength int64

	// LoopLength is the length of the loop in bytes. It is rounded down to a whole frame and must be
	// at least one frame.
	LoopLength int64

	// FadeFrames is the length of the crossfade in frames. Zero disables the crossfade, leaving the
	// loop data untouched. A negative value is an error. A value longer than the loop is shortened to
	// the loop length, since a longer window would never end before the loop repeats. The window is
	// also capped to 8 MiB of buffered source data.
	FadeFrames int64

	// Curve computes the crossfade gains. If Curve is nil, LoopCrossfadeLinear is used.
	Curve LoopCrossfadeCurve
}

// NewInfiniteLoop creates a new infinite loop stream with a source stream and length in bytes.
//
// src is a signed 16bit integer little endian stream, 2 channels (stereo).
//
// If the loop's total length is exactly the same as src's length, you might hear noises around the loop joint.
// This noise can be heard especially when src is decoded from a lossy compression format like Ogg/Vorbis and MP3.
// In this case, try to add more (about 0.1[s]) data to src after the loop end.
// If src has data after the loop end, an InfiniteLoop uses part of the data to blend with the loop start
// to make the loop joint smooth.
func NewInfiniteLoop(src io.ReadSeeker, length int64) *InfiniteLoop {
	return newInfiniteLoopWithIntro(src, 0, length, bitDepthInBytesInt16)
}

// NewInfiniteLoopF32 creates a new infinite loop stream with a source stream and length in bytes.
//
// src is a 32bit float little endian stream, 2 channels (stereo).
//
// If the loop's total length is exactly the same as src's length, you might hear noises around the loop joint.
// This noise can be heard especially when src is decoded from a lossy compression format like Ogg/Vorbis and MP3.
// In this case, try to add more (about 0.1[s]) data to src after the loop end.
// If src has data after the loop end, an InfiniteLoop uses part of the data to blend with the loop start
// to make the loop joint smooth.
func NewInfiniteLoopF32(src io.ReadSeeker, length int64) *InfiniteLoop {
	return newInfiniteLoopWithIntro(src, 0, length, bitDepthInBytesFloat32)
}

// NewInfiniteLoopWithIntro creates a new infinite loop stream with an intro part.
// NewInfiniteLoopWithIntro accepts a source stream src, introLength in bytes and loopLength in bytes.
//
// src is a signed 16bit integer little endian stream, 2 channels (stereo).
//
// If the loop's total length is exactly the same as src's length, you might hear noises around the loop joint.
// This noise can be heard especially when src is decoded from a lossy compression format like Ogg/Vorbis and MP3.
// In this case, try to add more (about 0.1[s]) data to src after the loop end.
// If src has data after the loop end, an InfiniteLoop uses part of the data to blend with the loop start
// to make the loop joint smooth.
func NewInfiniteLoopWithIntro(src io.ReadSeeker, introLength int64, loopLength int64) *InfiniteLoop {
	return newInfiniteLoopWithIntro(src, introLength, loopLength, bitDepthInBytesInt16)
}

// NewInfiniteLoopWithIntroF32 creates a new infinite loop stream with an intro part.
// NewInfiniteLoopWithIntroF32 accepts a source stream src, introLength in bytes and loopLength in bytes.
//
// src is a 32bit float little endian stream, 2 channels (stereo).
//
// If the loop's total length is exactly the same as src's length, you might hear noises around the loop joint.
// This noise can be heard especially when src is decoded from a lossy compression format like Ogg/Vorbis and MP3.
// In this case, try to add more (about 0.1[s]) data to src after the loop end.
// If src has data after the loop end, an InfiniteLoop uses part of the data to blend with the loop start
// to make the loop joint smooth.
func NewInfiniteLoopWithIntroF32(src io.ReadSeeker, introLength int64, loopLength int64) *InfiniteLoop {
	return newInfiniteLoopWithIntro(src, introLength, loopLength, bitDepthInBytesFloat32)
}

func newInfiniteLoopWithIntro(src io.ReadSeeker, introLength int64, loopLength int64, bitDepthInBytes int) *InfiniteLoop {
	bytesPerSample := bitDepthInBytes * channelCount
	lstart := introLength / int64(bytesPerSample) * int64(bytesPerSample)
	llength := loopLength / int64(bytesPerSample) * int64(bytesPerSample)
	if llength <= 0 {
		panic(fmt.Sprintf("audio: loop length must be a positive multiple of %d bytes but was %d", bytesPerSample, loopLength))
	}
	return &InfiniteLoop{
		src:             src,
		lstart:          lstart,
		llength:         llength,
		pos:             -1,
		bitDepthInBytes: bitDepthInBytes,
		bytesPerSample:  bytesPerSample,
		legacyBlend:     true,
	}
}

// NewInfiniteLoopWithCrossfade creates a new infinite loop stream with a source stream and a crossfade plan.
//
// src is a signed 16bit integer little endian stream, 2 channels (stereo).
//
// The plan defines the intro and loop regions and the crossfade at the loop joint. Unlike the
// NewInfiniteLoop* functions, this function does not panic on an invalid plan: it returns an error
// describing the invalid field. See LoopCrossfadePlan for the crossfade semantics.
func NewInfiniteLoopWithCrossfade(src io.ReadSeeker, plan LoopCrossfadePlan) (*InfiniteLoop, error) {
	return newInfiniteLoopWithCrossfade(src, plan, bitDepthInBytesInt16)
}

// NewInfiniteLoopWithCrossfadeF32 creates a new infinite loop stream with a source stream and a crossfade plan.
//
// src is a 32bit float little endian stream, 2 channels (stereo).
//
// The plan defines the intro and loop regions and the crossfade at the loop joint. Unlike the
// NewInfiniteLoop* functions, this function does not panic on an invalid plan: it returns an error
// describing the invalid field. See LoopCrossfadePlan for the crossfade semantics.
func NewInfiniteLoopWithCrossfadeF32(src io.ReadSeeker, plan LoopCrossfadePlan) (*InfiniteLoop, error) {
	return newInfiniteLoopWithCrossfade(src, plan, bitDepthInBytesFloat32)
}

// maxLoopCrossfadeBytes bounds the tail data a crossfade buffers. A longer crossfade is shortened to
// this many bytes (and to the loop length) so that a very long fade does not allocate unbounded memory.
const maxLoopCrossfadeBytes = 8 << 20

func newInfiniteLoopWithCrossfade(src io.ReadSeeker, plan LoopCrossfadePlan, bitDepthInBytes int) (*InfiniteLoop, error) {
	bytesPerSample := bitDepthInBytes * channelCount
	if plan.IntroLength < 0 {
		return nil, fmt.Errorf("audio: crossfade intro length must be >= 0 but was %d", plan.IntroLength)
	}
	if plan.LoopLength <= 0 {
		return nil, fmt.Errorf("audio: crossfade loop length must be positive but was %d", plan.LoopLength)
	}
	if plan.FadeFrames < 0 {
		return nil, fmt.Errorf("audio: crossfade frame count must be >= 0 but was %d", plan.FadeFrames)
	}

	lstart := plan.IntroLength / int64(bytesPerSample) * int64(bytesPerSample)
	llength := plan.LoopLength / int64(bytesPerSample) * int64(bytesPerSample)
	if llength <= 0 {
		return nil, fmt.Errorf("audio: crossfade loop length must be at least one frame (%d bytes) but was %d", bytesPerSample, plan.LoopLength)
	}
	if lstart > math.MaxInt64-llength {
		return nil, fmt.Errorf("audio: crossfade intro length %d and loop length %d overflow the stream length", plan.IntroLength, plan.LoopLength)
	}

	// A window longer than the loop would never end before the loop repeats, and an unbounded one
	// could allocate an arbitrarily large buffer. Shorten it rather than failing, so that a slightly
	// too long fade is still usable.
	fadeFrames := min(plan.FadeFrames, llength/int64(bytesPerSample), maxLoopCrossfadeBytes/int64(bytesPerSample))
	curve := plan.Curve
	if curve == nil {
		curve = LoopCrossfadeLinear
	}

	return &InfiniteLoop{
		src:             src,
		lstart:          lstart,
		llength:         llength,
		pos:             -1,
		bitDepthInBytes: bitDepthInBytes,
		bytesPerSample:  bytesPerSample,
		fadeFrames:      fadeFrames,
		curve:           curve,
	}, nil
}

func (i *InfiniteLoop) length() int64 {
	return i.lstart + i.llength
}

// Length returns the length in bytes of one pass through the loop, that is the intro length plus the
// loop length.
//
// A crossfade does not change Length: it only shapes the samples around the loop joint.
func (i *InfiniteLoop) Length() int64 {
	return i.length()
}

func (i *InfiniteLoop) ensurePos() error {
	if i.pos >= 0 {
		return nil
	}
	pos, err := i.src.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if pos >= i.length() {
		return fmt.Errorf("audio: stream position must be less than the specified length")
	}
	i.pos = pos
	return nil
}

func (i *InfiniteLoop) blendRate(pos int64) float32 {
	if pos < i.lstart {
		return 0
	}
	if pos >= i.lstart+int64(len(i.afterLoop)) {
		return 0
	}
	p := (pos - i.lstart) / int64(i.bytesPerSample)
	l := len(i.afterLoop) / i.bytesPerSample
	if l == 0 {
		return 0
	}
	return 1 - float32(p)/float32(l)
}

// afterLoopBytes returns the number of bytes after the loop end to buffer for the crossfade.
//
// It returns 0 when there is nothing to buffer: either the loop uses the legacy blending, whose
// amount is fixed, or the crossfade is disabled.
func (i *InfiniteLoop) afterLoopBytes() int64 {
	if i.legacyBlend {
		return min(int64(256*i.bytesPerSample), i.length())
	}
	if i.fadeFrames <= 0 {
		return 0
	}
	return i.fadeFrames * int64(i.bytesPerSample)
}

// captureAfterLoop reads the data after the loop end once, so that later repetitions can crossfade
// with it. It does nothing when the data was already captured or when the loop needs none.
//
// The read is bounded by the planned crossfade length and breaks on a source that makes no progress,
// so a faulty or endless source cannot spin here.
func (i *InfiniteLoop) captureAfterLoop() error {
	buflen := i.afterLoopBytes()
	if buflen <= 0 {
		return nil
	}
	// The configured crossfade reads the source once at the first loop end. If the source has no
	// tail then, there is nothing to crossfade with, and reading it again at every loop end would
	// repeat the same failing or empty read forever.
	if !i.legacyBlend && i.tailCaptured {
		return nil
	}
	if len(i.afterLoop) == 0 {
		buf := make([]byte, buflen)
		pos := copy(buf, i.partialAfterLoop)
		i.partialAfterLoop = nil
		for pos < len(buf) {
			nn, err := i.src.Read(buf[pos:])
			pos += nn
			if err != nil && err != io.EOF {
				i.partialAfterLoop = buf[:pos]
				return fmt.Errorf("audio: reading the data after the loop: %w", err)
			}
			// Break on EOF, and also when no progress is made so that a source returning
			// (0, nil) does not spin here forever.
			if err != nil || nn == 0 {
				break
			}
		}

		// A configured crossfade needs whole frames. A trailing partial frame is dropped, which is a
		// bounded degradation when the source ends in the middle of a frame.
		if !i.legacyBlend {
			pos -= pos % i.bytesPerSample
		}
		i.afterLoop = buf[:pos]
	}
	if !i.legacyBlend {
		i.tailCaptured = true
		// A configured crossfade is a function of the position and the captured tail, so it needs no
		// separate on/off state: a successful Seek keeps it, and a failed one leaves everything as is.
		return nil
	}
	// The legacy blending is turned on once the data after the loop end is available. A successful
	// Seek clears it, and the next loop end turns it back on without reading the source again.
	if len(i.afterLoop) > 0 {
		i.blending = true
	}
	return nil
}

// crossfadeGains returns the gains applied to the loop data and to the data after the loop end for
// the given frame in the crossfade window. frame must be within the effective window.
func (i *InfiniteLoop) crossfadeGains(frame int64) (loopGain, afterLoopGain float64) {
	frames := i.fadeFrames
	// A short source shortens the window rather than leaving the tail silent, so that the gains still
	// reach the loop data.
	if tailFrames := int64(len(i.afterLoop)) / int64(i.bytesPerSample); tailFrames < frames {
		frames = tailFrames
	}
	if frames <= 0 || frame < 0 || frame >= frames {
		return 1, 0
	}
	t := float64(frame) / float64(frames)
	loopGain, afterLoopGain = i.curve(t)
	// A curve that produces a non-finite gain is a bounded degradation: use the linear curve there.
	if !isFinite(loopGain) || !isFinite(afterLoopGain) {
		loopGain, afterLoopGain = LoopCrossfadeLinear(t)
	}
	return clamp01(loopGain), clamp01(afterLoopGain)
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func clamp01(v float64) float64 {
	return min(max(v, 0), 1)
}

// crossfade applies the configured crossfade to b, whose first byte is at stream position bpos.
//
// The crossfade works per channel value, so that a stream whose first read starts in the middle of a
// frame is still mixed consistently: the gain of a channel value is taken from the frame containing
// it, and the value is mixed with the tail bytes at the same offset from the loop end. Channel values
// that straddle two source values, which can only happen when src is positioned inside a frame before
// the loop is created, are left raw.
func (i *InfiniteLoop) crossfade(b []byte, bpos int64) {
	if i.noBlendForTesting || i.fadeFrames <= 0 || len(i.afterLoop) == 0 {
		return
	}
	if bpos+int64(len(b)) <= i.lstart || bpos >= i.lstart+int64(len(i.afterLoop)) {
		return
	}
	if len(b)%i.bitDepthInBytes != 0 {
		panic(fmt.Sprintf("audio: len(b) (%d) must be a multiple of bit depth %d [bytes]", len(b), i.bitDepthInBytes))
	}

	bitsPerChannel := i.bitDepthInBytes * 8
	var (
		lastFrame     int64 = -1
		loopGain      float64
		afterLoopGain float64
	)
	for idx := range len(b) / i.bitDepthInBytes {
		abspos := bpos + int64(idx)*int64(i.bitDepthInBytes)
		rel := abspos - i.lstart
		if rel < 0 {
			continue
		}
		// A value not aligned with the loop start would straddle two source values, so it cannot be
		// mixed with one whole value from the tail.
		if rel%int64(i.bitDepthInBytes) != 0 {
			continue
		}
		// Stop once the tail ends. A source that ends in the middle of a frame drops the trailing
		// partial value, so the last whole value starts strictly before the end.
		if rel+int64(i.bitDepthInBytes) > int64(len(i.afterLoop)) {
			break
		}

		frame := rel / int64(i.bytesPerSample)
		if frame != lastFrame {
			lastFrame = frame
			loopGain, afterLoopGain = i.crossfadeGains(frame)
		}
		if afterLoopGain == 0 {
			continue
		}

		off := idx * i.bitDepthInBytes
		switch bitsPerChannel {
		case 16:
			orig := int16(binary.LittleEndian.Uint16(b[off:]))
			afterLoop := int16(binary.LittleEndian.Uint16(i.afterLoop[rel:]))
			newVal := int16(float64(afterLoop)*afterLoopGain + float64(orig)*loopGain)
			binary.LittleEndian.PutUint16(b[off:], uint16(newVal))
		case 32:
			orig := math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
			afterLoop := math.Float32frombits(binary.LittleEndian.Uint32(i.afterLoop[rel:]))
			newVal := float32(float64(afterLoop)*afterLoopGain + float64(orig)*loopGain)
			binary.LittleEndian.PutUint32(b[off:], math.Float32bits(newVal))
		default:
			panic("not reached")
		}
	}
}

// Read is an implementation of ReadSeeker's Read.
//
// If the source ends before the loop, the loop has nothing to repeat and Read returns [io.EOF].
// If the source ends inside the loop, the loop is shortened to end there.
//
// Read returns only whole samples (4 bytes each in the 16-bit integer format, 8 bytes each in the 32-bit float format).
// For a non-empty buffer shorter than one sample, Read returns [io.ErrShortBuffer], or [io.EOF] once the stream has ended.
func (i *InfiniteLoop) Read(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	if len(b) < i.bytesPerSample {
		return 0, i.readShortBuffer()
	}

	if err := i.ensurePos(); err != nil {
		return 0, err
	}

	// When the source or the loop reaches its end, go back to the loop start and read again so that
	// this doesn't return (0, nil). A retry either returns data, stops at the loop start, or grows
	// the remainder, which is shorter than one sample, so the retries end.
	for rewinds := 0; ; rewinds++ {
		n, err := i.read(b)
		if err != nil && err != io.EOF {
			return n, err
		}
		atEnd := i.pos == i.length() || err == io.EOF
		if !atEnd {
			return n, nil
		}
		if n == 0 && (i.pos == i.lstart || rewinds > 0) {
			// The loop has no data to repeat.
			return 0, io.EOF
		}
		if err := i.rewind(); err != nil {
			return n, err
		}
		if n > 0 {
			return n, nil
		}
	}
}

// readShortBuffer buffers one sample to distinguish EOF from a short destination buffer.
func (i *InfiniteLoop) readShortBuffer() error {
	if err := i.ensurePos(); err != nil {
		return err
	}
	for rewinds := 0; ; {
		if len(i.extra) >= i.bytesPerSample {
			return io.ErrShortBuffer
		}
		var buf [bitDepthInBytesFloat32 * channelCount]byte
		size := min(int64(i.bytesPerSample-len(i.extra)), i.length()-i.pos)
		n, err := i.src.Read(buf[:size])
		i.extra = append(i.extra, buf[:n]...)
		i.pos += int64(n)
		if err != nil && err != io.EOF {
			return err
		}
		if len(i.extra) >= i.bytesPerSample {
			return io.ErrShortBuffer
		}
		if err == io.EOF || i.pos == i.length() {
			if i.pos == i.lstart || rewinds > 0 {
				return io.EOF
			}
			// Capture the data after the loop even when only short reads passed the loop end, so
			// that a configured crossfade does not depend on the destination buffer size. The
			// legacy blending keeps its historical timing and captures only on a full read.
			if !i.legacyBlend && i.pos == i.length() {
				if err := i.captureAfterLoop(); err != nil {
					return err
				}
			}
			if err := i.rewind(); err != nil {
				return err
			}
			rewinds++
			continue
		}
		if n == 0 {
			return io.ErrShortBuffer
		}
	}
}

// read reads data at the current position, and returns [io.EOF] when the source reaches its end.
func (i *InfiniteLoop) read(b []byte) (int, error) {
	extralen := len(i.extra)
	if i.pos+int64(len(b))-int64(extralen) > i.length() {
		b = b[:i.length()-i.pos+int64(extralen)]
	}

	copy(b, i.extra)
	i.extra = i.extra[:0]

	// Keep reading until one sample is available so that a source returning less than one sample
	// at a time doesn't make Read return (0, nil). A sample already in extra is returned without
	// reading, as the space left in b can be shorter than a sample.
	var n int
	var err error
	for extralen+n < i.bytesPerSample {
		var nn int
		nn, err = i.src.Read(b[extralen+n:])
		n += nn
		if err != nil || nn == 0 {
			break
		}
	}
	i.pos += int64(n)
	n += extralen
	if i.pos > i.length() {
		panic(fmt.Sprintf("audio: position %d exceeds length %d at (*InfiniteLoop).Read", i.pos, i.length()))
	}

	// bpos is the stream position of b[0], which must be calculated before the remainder is removed from b.
	bpos := i.pos - int64(n)

	// Save the remainder part to extra. This will be used at the next Read.
	if rem := n % i.bytesPerSample; rem != 0 {
		i.extra = append(i.extra, b[n-rem:n]...)
		b = b[:n-rem]
		n = n - rem
	}

	// Blend afterLoop and the loop start to reduce noises (#1888).
	// Ideally, afterLoop and the loop start should be identical, but they can have very slight differences.
	if !i.noBlendForTesting {
		if i.legacyBlend {
			if i.blending {
				i.blendAfterLoopLegacy(b[:n], bpos)
			}
		} else {
			// A configured crossfade is a function of the position and the captured tail, so it does
			// not depend on whether the loop was just rewound or sought.
			i.crossfade(b[:n], bpos)
		}
	}

	if err != nil && err != io.EOF {
		return n, err
	}

	// Read the afterLoop part if necessary.
	if i.pos == i.length() && err == nil {
		if err := i.captureAfterLoop(); err != nil {
			return n, err
		}
	}

	return n, err
}

// blendAfterLoopLegacy blends the loop data in b with the data after the loop end the way the
// NewInfiniteLoop* constructors have always done: a linear ramp whose length is the whole tail.
func (i *InfiniteLoop) blendAfterLoopLegacy(b []byte, bpos int64) {
	if i.pos < i.lstart || bpos >= i.lstart+int64(len(i.afterLoop)) {
		return
	}
	if len(b)%i.bitDepthInBytes != 0 {
		panic(fmt.Sprintf("audio: len(b) (%d) must be a multiple of bit depth %d [bytes]", len(b), i.bitDepthInBytes))
	}
	for idx := range len(b) / i.bitDepthInBytes {
		abspos := bpos + int64(idx)*int64(i.bitDepthInBytes)
		rate := i.blendRate(abspos)
		if rate == 0 {
			continue
		}

		relpos := abspos - i.lstart
		switch i.bitDepthInBytes {
		case 2:
			afterLoop := int16(i.afterLoop[relpos]) | (int16(i.afterLoop[relpos+1]) << 8)
			orig := int16(b[2*idx]) | (int16(b[2*idx+1]) << 8)
			newVal := int16(float32(afterLoop)*rate + float32(orig)*(1-rate))
			b[2*idx] = byte(newVal)
			b[2*idx+1] = byte(newVal >> 8)
		case 4:
			afterLoop := math.Float32frombits(uint32(i.afterLoop[relpos]) | (uint32(i.afterLoop[relpos+1]) << 8) | (uint32(i.afterLoop[relpos+2]) << 16) | (uint32(i.afterLoop[relpos+3]) << 24))
			orig := math.Float32frombits(uint32(b[4*idx]) | (uint32(b[4*idx+1]) << 8) | (uint32(b[4*idx+2]) << 16) | (uint32(b[4*idx+3]) << 24))
			newVal := float32(afterLoop*rate + orig*(1-rate))
			newValBits := math.Float32bits(newVal)
			b[4*idx] = byte(newValBits)
			b[4*idx+1] = byte(newValBits >> 8)
			b[4*idx+2] = byte(newValBits >> 16)
			b[4*idx+3] = byte(newValBits >> 24)
		default:
			panic("not reached")
		}
	}
}

// rewind moves the position back to the loop start.
func (i *InfiniteLoop) rewind() error {
	// Ignore the new position returned by Seek since the source position might not match the position
	// managed by this.
	if _, err := i.src.Seek(i.lstart, io.SeekStart); err != nil {
		return err
	}
	i.pos = i.lstart
	i.extra = i.extra[:0]
	i.partialAfterLoop = nil
	return nil
}

// Seek is an implementation of ReadSeeker's Seek.
//
// whence must be [io.SeekStart] or [io.SeekCurrent] since an [InfiniteLoop] has no end.
// Seek returns an error wrapping [errors.ErrUnsupported] for [io.SeekEnd].
//
// The returned position can differ from the requested one with a nil error: a position beyond the loop end is folded
// into the loop, and a position in the middle of a sample is rounded down to a sample boundary.
func (i *InfiniteLoop) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart, io.SeekCurrent:
	case io.SeekEnd:
		return 0, fmt.Errorf("audio: InfiniteLoop has no end: %w", errors.ErrUnsupported)
	default:
		return 0, fmt.Errorf("audio: whence must be io.SeekStart or io.SeekCurrent for InfiniteLoop but was %d", whence)
	}

	if err := i.ensurePos(); err != nil {
		return 0, err
	}

	// A query does not seek src, so that the data read next and its blending with afterLoop are kept.
	if whence == io.SeekCurrent && offset == 0 {
		return i.pos - int64(len(i.extra)), nil
	}

	var next int64
	switch whence {
	case io.SeekStart:
		next = offset
	case io.SeekCurrent:
		next = i.pos - int64(len(i.extra)) + offset
	}
	if next < 0 {
		return 0, fmt.Errorf("audio: position must be >= 0 but was %d", next)
	}
	// A position in the middle of a sample is rounded down to a sample boundary, as reading from there
	// would return bytes straddling two samples.
	next = next / int64(i.bytesPerSample) * int64(i.bytesPerSample)
	if next > i.lstart {
		next = ((next - i.lstart) % i.llength) + i.lstart
	}
	// Ignore the new position returned by Seek since the source position might not match the position
	// managed by this.
	if _, err := i.src.Seek(next, io.SeekStart); err != nil {
		return 0, err
	}
	// A configured crossfade is a function of the position and the captured tail, so it stays active
	// after a seek. The legacy blending is cleared, as it has always been, and turns on again at the
	// next loop end.
	if i.legacyBlend {
		i.blending = false
	}
	i.pos = next
	i.extra = i.extra[:0]
	i.partialAfterLoop = nil
	return i.pos, nil
}
