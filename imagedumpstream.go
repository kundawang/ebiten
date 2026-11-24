// Copyright 2026 The Ebiten Authors
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

package ebiten

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ImageDumpEncoding is the encoding used for frames written by an [ImageDumpStream].
type ImageDumpEncoding int

const (
	// ImageDumpEncodingPNG encodes frames as PNG files.
	// Pixels are interpreted as premultiplied-alpha RGBA, the same layout
	// produced by [Image.ReadPixels].
	ImageDumpEncodingPNG ImageDumpEncoding = iota

	// ImageDumpEncodingRawRGBA writes frames without compression as a
	// [RawRGBAFrameHeaderSize]-byte header followed by the raw pixels.
	// Raw dumps avoid the CPU cost and latency of PNG encoding and are
	// intended for pipelines that decode frames offline.
	ImageDumpEncodingRawRGBA
)

// RawRGBAFrameMagic is the magic number stored at the beginning of every
// raw-RGBA frame file, in little-endian byte order.
const RawRGBAFrameMagic uint32 = 0x45524741 // "AGRE"

// RawRGBAFrameVersion is the version of the raw-RGBA frame file format.
const RawRGBAFrameVersion uint32 = 1

// RawRGBAFrameHeaderSize is the fixed size in bytes of a raw-RGBA frame header.
// The header is followed by 4 * width * height premultiplied-alpha RGBA bytes.
const RawRGBAFrameHeaderSize = 32

// ManifestFileName is the name of the manifest file written inside every
// [ImageDumpStream] directory.
const ManifestFileName = "manifest.json"

// Sentinel errors reported by [ImageDumpStream].
var (
	// ErrImageDumpStreamClosed is returned when submitting or flushing a
	// stream that has already been closed, and when Close is called more
	// than once.
	ErrImageDumpStreamClosed = errors.New("ebiten: the image dump stream is already closed")

	// ErrImageDumpQueueFull is returned by [ImageDumpStream.Submit] when the
	// bounded queue has no free slot and the frame is dropped without
	// blocking the producer. A zero-capacity queue always returns this.
	ErrImageDumpQueueFull = errors.New("ebiten: the image dump stream queue is full")

	// ErrImageDumpFrameTooLarge is returned by [ImageDumpStream.Submit] when
	// a frame's encoded size would exceed [ImageDumpStreamOptions.MaxFrameBytes].
	ErrImageDumpFrameTooLarge = errors.New("ebiten: the dumped frame is too large")
)

// ImageDumpStreamOptions configures an [ImageDumpStream].
type ImageDumpStreamOptions struct {
	// Dir is the directory frames and the manifest are written to.
	// It is created together with all its parents when the stream is created.
	// Dir must not be empty.
	Dir string

	// Encoding selects PNG or raw-RGBA output. The zero value is
	// [ImageDumpEncodingPNG].
	Encoding ImageDumpEncoding

	// QueueCapacity is the maximum number of frames that may be buffered
	// while waiting for the disk-writing goroutine.
	// Submit never blocks: when the queue is full, newly submitted frames
	// are dropped and counted as such. QueueCapacity must be non-negative;
	// a zero capacity makes every Submit drop its frame.
	QueueCapacity int

	// MaxFrames limits the number of frame files retained in Dir.
	// Zero means no limit. When the limit is exceeded, the oldest frames
	// are deleted first.
	MaxFrames int

	// MaxTotalBytes limits the total size in bytes of retained frame files.
	// Zero means no limit. When the limit is exceeded, the oldest frames
	// are deleted first. A single frame larger than the limit is written
	// and then removed immediately.
	MaxTotalBytes int64

	// MaxFrameBytes rejects frames whose on-disk size would exceed this
	// value before any file is created. For raw frames the size is the
	// header plus the pixel data; for PNG frames the pixel data size is
	// checked before encoding. Zero means no limit.
	MaxFrameBytes int64

	// FilenamePrefix is the prefix of every frame file name. It must be a
	// plain file name component: path separators and "." / ".." are
	// rejected. The empty value defaults to "frame".
	FilenamePrefix string
}

// ImageDumpStats is a point-in-time snapshot of an [ImageDumpStream]'s counters.
type ImageDumpStats struct {
	// Submitted is the total number of frames accepted into the queue.
	Submitted uint64

	// Written is the number of frames successfully written to disk.
	Written uint64

	// Dropped is the number of frames that were never written: frames
	// rejected because the queue was full, frames discarded on rotation of
	// an over-limit directory, and frames lost when the writer crashed.
	Dropped uint64

	// Errors is the number of encoding or file-system failures.
	Errors uint64

	// QueueDepth is the number of frames and barriers currently buffered,
	// including the frame being written.
	QueueDepth int
}

// ImageDumpStream asynchronously writes rendered frames to a directory.
//
// Frames are submitted from the rendering goroutine through [ImageDumpStream.Submit]
// (or [ImageDumpStream.SubmitImage]), which copies the pixel data and hands it
// to a bounded queue. A single dedicated goroutine performs all encoding and
// disk I/O, so a producer never waits on the disk: when the queue is full the
// frame is dropped and reported instead of blocking.
//
// Every accepted frame is written as its own file and recorded, together with
// its frame number, bounds and timestamps, in a JSON manifest named
// [ManifestFileName]. Old frames are rotated out by frame count or total size.
//
// ImageDumpStream is safe for concurrent use.
type ImageDumpStream struct {
	dir      string
	encoding ImageDumpEncoding
	prefix   string

	maxFrames     int
	maxTotalBytes int64
	maxFrameBytes int64

	jobs chan imageDumpJob
	done chan struct{}

	// closeMu guards closed; only the first Close passes the closed check.
	closeMu sync.Mutex
	closed  bool

	// errMu protects pendingErr, lastErr and errLog, all of which are
	// touched by producers (queue-full drops) as well as the writer.
	// pendingErr holds failures the next Flush will return; lastErr keeps the
	// most recent failure for Close after Flush has drained pendingErr.
	// Queue-full backpressure is reported only through Submit's return value
	// and the counters, never through pendingErr or lastErr.
	errMu      sync.Mutex
	pendingErr error
	lastErr    error

	// All counters are touched concurrently by producers and the worker.
	submitted  atomic.Uint64
	written    atomic.Uint64
	dropped    atomic.Uint64
	errCount   atomic.Uint64
	queueDepth atomic.Int64

	// errLog is a bounded ring of failure records mirrored into the manifest.
	errLog []manifestError

	// The worker goroutine exclusively owns the fields below.
	retained []manifestFrame
	fileSeq  uint64
}

type imageDumpJobKind int

const (
	imageDumpJobFrame imageDumpJobKind = iota
	imageDumpJobBarrier
	imageDumpJobClose
)

type imageDumpJob struct {
	kind imageDumpJobKind

	// Frame fields.
	frameNo  int64
	bounds   image.Rectangle
	pixels   []byte
	queuedAt time.Time

	// Barrier fields.
	result chan error
}

// manifestFrame is one frame record inside the manifest file.
type manifestFrame struct {
	Sequence  uint64          `json:"sequence"`
	Frame     int64           `json:"frame"`
	File      string          `json:"file"`
	Encoding  string          `json:"encoding"`
	Width     int             `json:"width"`
	Height    int             `json:"height"`
	Bounds    image.Rectangle `json:"bounds"`
	SizeBytes int64           `json:"size_bytes"`
	QueuedAt  time.Time       `json:"queued_at"`
	WrittenAt time.Time       `json:"written_at"`
}

type manifestError struct {
	At    time.Time `json:"at"`
	Frame int64     `json:"frame"`
	Op    string    `json:"op"`
	Err   string    `json:"error"`
}

type manifestFile struct {
	Version   int             `json:"version"`
	Encoding  string          `json:"encoding"`
	Directory string          `json:"directory"`
	Stats     manifestStats   `json:"stats"`
	Frames    []manifestFrame `json:"frames"`
	Errors    []manifestError `json:"errors,omitempty"`
}

type manifestStats struct {
	Submitted uint64 `json:"submitted"`
	Written   uint64 `json:"written"`
	Dropped   uint64 `json:"dropped"`
	Errors    uint64 `json:"errors"`
}

const imageDumpManifestVersion = 1

// maxImageDumpErrorLog bounds the error ring embedded in the manifest. The
// atomic counters keep the full lifetime tallies regardless.
const maxImageDumpErrorLog = 128

// writerFactory wraps a freshly created temp file into the writeCloser used
// for the actual write. It is indirected only so tests can inject
// short-writing and panicking writers. The returned closer replaces the
// original *os.File, which the caller never touches directly.
type writerFactory func(f *os.File) writeCloser

type writeCloser interface {
	Write(p []byte) (int, error)
	Close() error
}

func defaultWriterFactory(f *os.File) writeCloser {
	return f
}

// NewImageDumpStream creates a stream that writes frames into options.Dir.
//
// The directory and all its parents are created immediately, so directory
// creation errors surface here rather than on the first submitted frame.
func NewImageDumpStream(options ImageDumpStreamOptions) (*ImageDumpStream, error) {
	return newImageDumpStream(options, defaultWriterFactory)
}

func newImageDumpStream(options ImageDumpStreamOptions, factory writerFactory) (*ImageDumpStream, error) {
	if options.Dir == "" {
		return nil, errors.New("ebiten: Dir must not be empty at NewImageDumpStream")
	}
	if options.QueueCapacity < 0 {
		return nil, errors.New("ebiten: QueueCapacity must not be negative at NewImageDumpStream")
	}
	if options.MaxFrames < 0 {
		return nil, errors.New("ebiten: MaxFrames must not be negative at NewImageDumpStream")
	}
	if options.MaxTotalBytes < 0 {
		return nil, errors.New("ebiten: MaxTotalBytes must not be negative at NewImageDumpStream")
	}
	if options.MaxFrameBytes < 0 {
		return nil, errors.New("ebiten: MaxFrameBytes must not be negative at NewImageDumpStream")
	}
	switch options.Encoding {
	case ImageDumpEncodingPNG, ImageDumpEncodingRawRGBA:
	default:
		return nil, fmt.Errorf("ebiten: invalid image dump encoding: %d", options.Encoding)
	}

	prefix := options.FilenamePrefix
	if prefix == "" {
		prefix = "frame"
	}
	if err := validateFilenamePrefix(prefix); err != nil {
		return nil, err
	}

	// MkdirAll also lets the worker recover when the output directory is
	// renamed or removed underneath a running stream.
	if err := os.MkdirAll(options.Dir, 0o755); err != nil {
		return nil, fmt.Errorf("ebiten: cannot create the image dump directory: %w", err)
	}
	dir, err := filepath.Abs(options.Dir)
	if err != nil {
		return nil, fmt.Errorf("ebiten: cannot resolve the image dump directory: %w", err)
	}

	s := &ImageDumpStream{
		dir:           dir,
		encoding:      options.Encoding,
		prefix:        prefix,
		maxFrames:     options.MaxFrames,
		maxTotalBytes: options.MaxTotalBytes,
		maxFrameBytes: options.MaxFrameBytes,
		jobs:          make(chan imageDumpJob, options.QueueCapacity),
		done:          make(chan struct{}),
	}
	go s.consume(factory)
	return s, nil
}

func validateFilenamePrefix(prefix string) error {
	if prefix == "." || prefix == ".." || filepath.IsAbs(prefix) {
		return fmt.Errorf("ebiten: invalid image dump filename prefix: %q", prefix)
	}
	if strings.ContainsAny(prefix, `/\`) {
		return fmt.Errorf("ebiten: invalid image dump filename prefix: %q", prefix)
	}
	// Reject anything filepath would treat as a traversal after cleaning.
	if filepath.Clean(prefix) != prefix {
		return fmt.Errorf("ebiten: invalid image dump filename prefix: %q", prefix)
	}
	return nil
}

// Submit enqueues a frame for asynchronous writing and never blocks on disk.
//
// pixels holds premultiplied-alpha RGBA bytes, 4 bytes per pixel, matching the
// layout of [Image.ReadPixels]. Its contents are copied before Submit returns,
// so the caller may reuse the buffer immediately. bounds is the logical bounds
// of the frame and may have a non-zero Min; its size must match len(pixels).
// frameNo is an arbitrary caller-chosen frame number stored verbatim;
// duplicate frame numbers are allowed. When at is the zero time, the current
// time is used.
//
// Submit returns [ErrImageDumpQueueFull] (without blocking) when the bounded
// queue is full, [ErrImageDumpFrameTooLarge] when the frame exceeds the
// configured size limit, and [ErrImageDumpStreamClosed] after Close.
func (s *ImageDumpStream) Submit(frameNo int64, bounds image.Rectangle, pixels []byte, at time.Time) error {
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return fmt.Errorf("ebiten: invalid frame bounds %v at Submit", bounds)
	}
	if got, want := len(pixels), 4*w*h; got != want {
		return fmt.Errorf("ebiten: len(pixels) must be %d for bounds %v but %d at Submit", want, bounds, got)
	}
	if s.maxFrameBytes > 0 {
		size := int64(len(pixels))
		if s.encoding == ImageDumpEncodingRawRGBA {
			size += RawRGBAFrameHeaderSize
		}
		if size > s.maxFrameBytes {
			return fmt.Errorf("%w: %d bytes", ErrImageDumpFrameTooLarge, size)
		}
	}
	if at.IsZero() {
		at = time.Now()
	}

	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return ErrImageDumpStreamClosed
	}
	s.closeMu.Unlock()

	// Copy the pixels: the producer owns the original buffer and may reuse
	// it as soon as Submit returns.
	copied := make([]byte, len(pixels))
	copy(copied, pixels)

	job := imageDumpJob{
		kind:     imageDumpJobFrame,
		frameNo:  frameNo,
		bounds:   bounds,
		pixels:   copied,
		queuedAt: at,
	}

	// A non-blocking send keeps the rendering goroutine off the disk.
	// A zero-capacity queue never buffers anything, so submissions to it
	// reject unconditionally; relying on the select would make the result
	// depend on whether the worker happens to be parked on the channel.
	if cap(s.jobs) == 0 {
		s.recordDrop(frameNo, at)
		return ErrImageDumpQueueFull
	}
	select {
	case s.jobs <- job:
		s.queueDepth.Add(1)
		return nil
	default:
		s.recordDrop(frameNo, at)
		return ErrImageDumpQueueFull
	}
}

// SubmitImage reads img's pixels and enqueues the resulting frame.
//
// The pixel read-back happens synchronously on the calling goroutine and may
// be slow, as documented on [Image.ReadPixels]; all encoding and disk I/O
// remain asynchronous. The same restrictions as ReadPixels apply, including
// that the game loop must be running.
func (s *ImageDumpStream) SubmitImage(img *Image, frameNo int64, at time.Time) error {
	b := img.Bounds()
	pixels := make([]byte, 4*b.Dx()*b.Dy())
	img.ReadPixels(pixels)
	return s.Submit(frameNo, b, pixels, at)
}

// Flush blocks until every frame submitted before the call has been written,
// rotated and recorded in the manifest, or until one of them fails.
//
// Flush returns and clears the first error accumulated since the previous
// Flush; counters and the manifest still record every failure. After an error
// the stream remains usable for later frames.
func (s *ImageDumpStream) Flush() error {
	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return ErrImageDumpStreamClosed
	}
	s.closeMu.Unlock()

	result := make(chan error, 1)
	s.jobs <- imageDumpJob{kind: imageDumpJobBarrier, result: result}
	s.queueDepth.Add(1)
	return <-result
}

// Close flushes all queued frames, stops the writer goroutine and returns the
// first error that has not yet been consumed by Flush, if any. It is safe to
// call concurrently; calls after the first one return [ErrImageDumpStreamClosed].
func (s *ImageDumpStream) Close() error {
	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return ErrImageDumpStreamClosed
	}
	s.closed = true
	s.closeMu.Unlock()

	// All producers are now rejected, so the queue only drains. Sending in a
	// separate goroutine guarantees Close returns even when the worker has
	// died and nobody is draining the channel anymore: failConsumer takes the
	// close job once, and a dead worker leaves the send to be garbage
	// collected with the channel. The select on done unblocks that sender
	// immediately once the worker exits for any reason.
	go func() {
		select {
		case s.jobs <- imageDumpJob{kind: imageDumpJobClose}:
		case <-s.done:
		}
	}()
	<-s.done

	s.errMu.Lock()
	err := s.lastErr
	s.errMu.Unlock()
	if err != nil {
		return err
	}
	return nil
}

// Stats returns a snapshot of the stream's counters.
func (s *ImageDumpStream) Stats() ImageDumpStats {
	return ImageDumpStats{
		Submitted:  s.submitted.Load(),
		Written:    s.written.Load(),
		Dropped:    s.dropped.Load(),
		Errors:     s.errCount.Load(),
		QueueDepth: int(s.queueDepth.Load()),
	}
}

func (s *ImageDumpStream) pushPendingError(err error) {
	s.errMu.Lock()
	if s.pendingErr == nil {
		s.pendingErr = err
	}
	s.lastErr = err
	s.errMu.Unlock()
}

func (s *ImageDumpStream) consume(factory writerFactory) {
	// A crashing writer must never leak the goroutine, temp files or a
	// hanging Close: recover, fail every waiter and mark the stream dead.
	defer func() {
		if r := recover(); r != nil {
			s.failConsumer(fmt.Errorf("ebiten: the image dump writer panicked: %v", r))
		}
		close(s.done)
	}()

	for job := range s.jobs {
		switch job.kind {
		case imageDumpJobFrame:
			s.processFrame(job, factory)
			s.queueDepth.Add(-1)
		case imageDumpJobBarrier:
			// Re-publish the manifest so drops recorded by producers become
			// visible even when no frame follows them.
			if err := s.writeManifest(defaultWriterFactory); err != nil {
				s.errCount.Add(1)
				s.errMu.Lock()
				if len(s.errLog) >= maxImageDumpErrorLog {
					s.errLog = s.errLog[1:]
				}
				s.errLog = append(s.errLog, manifestError{
					At:    time.Now(),
					Frame: 0,
					Op:    "manifest",
					Err:   err.Error(),
				})
				s.errMu.Unlock()
				s.pushPendingError(err)
			}
			s.errMu.Lock()
			err := s.pendingErr
			s.pendingErr = nil
			s.errMu.Unlock()
			job.result <- err
			s.queueDepth.Add(-1)
		case imageDumpJobClose:
			return
		}
	}
}

// failConsumer drains everything still queued after a panic so no producer
// or Close ever blocks, marks the stream as failed and closed so later
// operations fail fast instead of sending into a channel nobody reads.
func (s *ImageDumpStream) failConsumer(err error) {
	// The job that triggered the panic was already received from the channel,
	// so consume's normal queueDepth.Add(-1) for it never ran. Compensate
	// once before draining the jobs still buffered; clamp at zero in case the
	// panic originated from a path with no frame in flight.
	if s.queueDepth.Load() > 0 {
		s.queueDepth.Add(-1)
	}
	s.pushPendingError(err)
	for {
		select {
		case job := <-s.jobs:
			switch job.kind {
			case imageDumpJobFrame:
				s.dropped.Add(1)
				s.queueDepth.Add(-1)
			case imageDumpJobBarrier:
				job.result <- err
				s.queueDepth.Add(-1)
			case imageDumpJobClose:
				s.markClosed()
				return
			}
		default:
			s.markClosed()
			return
		}
	}
}

// markClosed flips the closed flag so Submit and Flush reject new work.
func (s *ImageDumpStream) markClosed() {
	s.closeMu.Lock()
	s.closed = true
	s.closeMu.Unlock()
}

func (s *ImageDumpStream) processFrame(job imageDumpJob, factory writerFactory) {
	s.submitted.Add(1)

	name, size, err := s.writeFrame(job, factory)
	if err != nil {
		s.recordWorkerError(job.frameNo, "write", err)
		return
	}

	s.fileSeq++
	s.retained = append(s.retained, manifestFrame{
		Sequence:  s.fileSeq,
		Frame:     job.frameNo,
		File:      name,
		Encoding:  s.encodingName(),
		Width:     job.bounds.Dx(),
		Height:    job.bounds.Dy(),
		Bounds:    job.bounds,
		SizeBytes: size,
		QueuedAt:  job.queuedAt,
		WrittenAt: time.Now(),
	})
	s.written.Add(1)

	s.rotate()
	if err := s.writeManifest(factory); err != nil {
		s.recordWorkerError(job.frameNo, "manifest", err)
	}
}

func (s *ImageDumpStream) encodingName() string {
	if s.encoding == ImageDumpEncodingRawRGBA {
		return "raw-rgba"
	}
	return "png"
}

func (s *ImageDumpStream) frameExtension() string {
	if s.encoding == ImageDumpEncodingRawRGBA {
		return ".rgba"
	}
	return ".png"
}

func (s *ImageDumpStream) writeFrame(job imageDumpJob, factory writerFactory) (string, int64, error) {
	var data []byte
	var err error
	if s.encoding == ImageDumpEncodingRawRGBA {
		data, err = encodeRawRGBA(job.bounds, job.pixels)
	} else {
		data, err = encodePNG(job.bounds, job.pixels)
	}
	if err != nil {
		return "", 0, err
	}

	// Recover the directory if it was renamed or removed out from under us.
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return "", 0, err
	}

	// A fixed-width zero-padded sequence sorts lexicographically in write
	// order on every platform and stays unique even when frame numbers repeat.
	name := fmt.Sprintf("%s_%020d%s", s.prefix, s.fileSeq+1, s.frameExtension())
	path := filepath.Join(s.dir, name)

	tmp, err := os.CreateTemp(s.dir, ".frame-*.tmp")
	if err != nil {
		return "", 0, err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			// Best effort: never leave partial files behind.
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	// The factory wraps the already-open temp file; the default factory
	// returns it unchanged, tests inject failing or panicking writers.
	wc := factory(tmp)
	if err := writeAll(wc, data); err != nil {
		_ = wc.Close()
		return "", 0, err
	}
	if err := wc.Close(); err != nil {
		return "", 0, err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return "", 0, err
	}
	cleanup = false
	return name, int64(len(data)), nil
}

func writeAll(w writeCloser, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n <= 0 {
			return errors.New("ebiten: the image dump writer made no progress")
		}
		if n > len(data) {
			return errors.New("ebiten: the image dump writer returned an invalid byte count")
		}
		data = data[n:]
	}
	return nil
}

func encodePNG(bounds image.Rectangle, pixels []byte) ([]byte, error) {
	img := &image.RGBA{
		Pix:    pixels,
		Stride: 4 * bounds.Dx(),
		// A non-zero Min is preserved: image/png stores the actual bounds.
		Rect: bounds,
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("ebiten: PNG encoding failed: %w", err)
	}
	return buf.Bytes(), nil
}

func encodeRawRGBA(bounds image.Rectangle, pixels []byte) ([]byte, error) {
	buf := make([]byte, RawRGBAFrameHeaderSize+len(pixels))
	binary.LittleEndian.PutUint32(buf[0:4], RawRGBAFrameMagic)
	binary.LittleEndian.PutUint32(buf[4:8], RawRGBAFrameVersion)
	binary.LittleEndian.PutUint32(buf[8:12], uint32(bounds.Dx()))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(bounds.Dy()))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(int32(bounds.Min.X)))
	binary.LittleEndian.PutUint32(buf[20:24], uint32(int32(bounds.Min.Y)))
	// Bytes 24..31 are reserved and stay zero.
	copy(buf[RawRGBAFrameHeaderSize:], pixels)
	return buf, nil
}

// rotate deletes the oldest retained frames until both retention limits hold.
func (s *ImageDumpStream) rotate() {
	if s.maxFrames == 0 && s.maxTotalBytes == 0 {
		return
	}

	var total int64
	for _, f := range s.retained {
		total += f.SizeBytes
	}

	for len(s.retained) > 0 {
		if (s.maxFrames <= 0 || len(s.retained) <= s.maxFrames) &&
			(s.maxTotalBytes <= 0 || total <= s.maxTotalBytes) {
			break
		}

		oldest := s.retained[0]
		path := filepath.Join(s.dir, oldest.File)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			// A removal failure is observable but must not kill the worker;
			// bookkeeping still advances so limits cannot wedge the stream.
			s.recordWorkerError(oldest.Frame, "rotate", err)
		}
		total -= oldest.SizeBytes
		if total < 0 {
			total = 0
		}
		s.retained = s.retained[1:]
		s.dropped.Add(1)
	}
}

func (s *ImageDumpStream) writeManifest(factory writerFactory) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}

	frames := s.retained
	if frames == nil {
		frames = []manifestFrame{}
	}
	s.errMu.Lock()
	errs := append([]manifestError(nil), s.errLog...)
	s.errMu.Unlock()
	if errs == nil {
		errs = []manifestError{}
	}
	mf := manifestFile{
		Version:   imageDumpManifestVersion,
		Encoding:  s.encodingName(),
		Directory: s.dir,
		Stats: manifestStats{
			Submitted: s.submitted.Load(),
			Written:   s.written.Load(),
			Dropped:   s.dropped.Load(),
			Errors:    s.errCount.Load(),
		},
		Frames: frames,
		Errors: errs,
	}
	data, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	path := filepath.Join(s.dir, ManifestFileName)
	tmp, err := os.CreateTemp(s.dir, ".manifest-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	wc := factory(tmp)
	if err := writeAll(wc, data); err != nil {
		_ = wc.Close()
		return err
	}
	if err := wc.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	cleanup = false
	return nil
}

// recordWorkerError is only called from the writer goroutine.
func (s *ImageDumpStream) recordWorkerError(frame int64, op string, err error) {
	s.errCount.Add(1)
	s.errMu.Lock()
	if len(s.errLog) >= maxImageDumpErrorLog {
		// Bound the in-memory error ring; the counters keep the full tally.
		s.errLog = s.errLog[1:]
	}
	s.errLog = append(s.errLog, manifestError{
		At:    time.Now(),
		Frame: frame,
		Op:    op,
		Err:   err.Error(),
	})
	s.errMu.Unlock()
	// Persist the failure immediately so a later crash still leaves evidence.
	if writeErr := s.writeManifest(defaultWriterFactory); writeErr != nil {
		// The manifest is best effort once the primary write already failed.
		_ = writeErr
	}
	s.pushPendingError(err)
}

// recordDrop accounts for a frame rejected by a full queue. It may run from
// any producer goroutine. Queue-full is ordinary backpressure: it is counted
// and logged, but deliberately not pushed to pendingErr so Flush stays green.
func (s *ImageDumpStream) recordDrop(frame int64, at time.Time) {
	s.dropped.Add(1)
	s.errMu.Lock()
	if len(s.errLog) >= maxImageDumpErrorLog {
		s.errLog = s.errLog[1:]
	}
	s.errLog = append(s.errLog, manifestError{
		At:    at,
		Frame: frame,
		Op:    "drop",
		Err:   ErrImageDumpQueueFull.Error(),
	})
	s.errMu.Unlock()
}
