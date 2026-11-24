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

package ebiten_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
)

// checkerboardImage builds a premultiplied-alpha RGBA checkerboard whose outer
// ring is fully transparent, exercising non-trivial alpha and a content edge.
func checkerboardImage(w, h, cell int, min image.Point) (*image.RGBA, image.Rectangle) {
	bounds := image.Rectangle{Min: min, Max: min.Add(image.Pt(w, h))}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			off := 4 * (y*w + x)
			// A one-pixel transparent border on every side.
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				continue
			}
			if ((x/cell)+(y/cell))%2 == 0 {
				img.Pix[off+0] = 0xff
				img.Pix[off+1] = 0x00
				img.Pix[off+2] = 0x00
				img.Pix[off+3] = 0xff
			} else {
				// Semi-transparent green: premultiplied values.
				img.Pix[off+0] = 0x00
				img.Pix[off+1] = 0x80
				img.Pix[off+2] = 0x00
				img.Pix[off+3] = 0x80
			}
		}
	}
	return img, bounds
}

func readManifest(t *testing.T, dir string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, ebiten.ManifestFileName))
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("manifest is not valid JSON: %v\n%s", err, data)
	}
	return m
}

func manifestFrames(t *testing.T, dir string) []map[string]any {
	t.Helper()
	m := readManifest(t, dir)
	raw, ok := m["frames"].([]any)
	if !ok {
		t.Fatalf("manifest has no frames array: %v", m)
	}
	frames := make([]map[string]any, 0, len(raw))
	for _, r := range raw {
		frames = append(frames, r.(map[string]any))
	}
	return frames
}

func listFrameFiles(t *testing.T, dir string, ext string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	var names []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ext) {
			names = append(names, e.Name())
		}
	}
	return names
}

func assertNoTempFiles(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".frame-") || strings.HasPrefix(e.Name(), ".manifest-") {
			t.Errorf("temporary file leaked: %s", e.Name())
		}
	}
}

// assertGoroutinesSettled fails if the goroutine count does not return close to
// the baseline after Close, i.e. if the worker or a Close waiter leaked.
func assertGoroutinesSettled(t *testing.T, baseline int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if runtime.NumGoroutine() <= baseline+1 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("goroutine leak: baseline %d, now %d\n%s", baseline, runtime.NumGoroutine(), debug.Stack())
}

func TestImageDumpStreamPNGPixelsMatchSynchronousEncode(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(12, 10, 3, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		Encoding:      ebiten.ImageDumpEncodingPNG,
		QueueCapacity: 4,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}

	if err := s.Submit(7, bounds, img.Pix, time.Unix(1000, 0)); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	if len(files) != 1 {
		t.Fatalf("want 1 PNG file, got %d: %v", len(files), files)
	}
	data, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), files[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	got, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decoded PNG is invalid: %v", err)
	}
	if got.Bounds() != bounds {
		t.Errorf("bounds: got %v, want %v", got.Bounds(), bounds)
	}

	// Reference: the exact synchronous encoding of the same pixels.
	var want bytes.Buffer
	if err := png.Encode(&want, img); err != nil {
		t.Fatalf("reference encode: %v", err)
	}
	if !bytes.Equal(data, want.Bytes()) {
		t.Errorf("stream PNG bytes differ from the synchronous png.Encode output")
	}

	// Pixel-for-pixel comparison including the transparent border. The
	// standard PNG decoder may choose NRGBA for images with alpha, so compare
	// through the generic image.Image API against the straight-alpha values
	// corresponding to the premultiplied input.
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			off := 4 * (y*bounds.Dx() + x)
			pr, pg, pb, pa := got.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			wantColor := color.NRGBAModel.Convert(color.RGBA{
				R: img.Pix[off+0],
				G: img.Pix[off+1],
				B: img.Pix[off+2],
				A: img.Pix[off+3],
			}).(color.NRGBA)
			wr, wg, wb, wa := wantColor.RGBA()
			if pr != wr || pg != wg || pb != wb || pa != wa {
				t.Fatalf("pixel (%d,%d): got %v,%v,%v,%v want %v,%v,%v,%v",
					x, y, pr, pg, pb, pa, wr, wg, wb, wa)
			}
		}
	}

	frames := manifestFrames(t, s.ImageDumpStreamDirForTesting())
	if len(frames) != 1 {
		t.Fatalf("manifest frames: got %d, want 1", len(frames))
	}
	if int64(frames[0]["frame"].(float64)) != 7 {
		t.Errorf("manifest frame number: got %v, want 7", frames[0]["frame"])
	}
	if frames[0]["file"] != files[0] {
		t.Errorf("manifest file %v does not match disk file %v", frames[0]["file"], files[0])
	}
	if frames[0]["encoding"] != "png" {
		t.Errorf("encoding: got %v", frames[0]["encoding"])
	}
	if int(frames[0]["width"].(float64)) != 12 || int(frames[0]["height"].(float64)) != 10 {
		t.Errorf("manifest size: %vx%v", frames[0]["width"], frames[0]["height"])
	}

	st := s.Stats()
	if st.Submitted != 1 || st.Written != 1 || st.Errors != 0 || st.Dropped != 0 {
		t.Errorf("stats after one frame: %+v", st)
	}
	if st.QueueDepth != 0 {
		t.Errorf("queue depth after flush: %d, want 0", st.QueueDepth)
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	assertNoTempFiles(t, s.ImageDumpStreamDirForTesting())
}

func TestImageDumpStreamNonZeroBounds(t *testing.T) {
	dir := t.TempDir()
	// A sub-image-like frame whose origin is far from zero.
	min := image.Pt(-15, 27)
	img, bounds := checkerboardImage(8, 6, 2, min)

	// The standard library's PNG decoder always normalizes the origin to
	// (0,0); use the raw format, which stores the bounds verbatim, for the
	// full round trip. The PNG path below checks the manifest record.
	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "raw"),
		Encoding:      ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	rawFiles := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".rgba")
	if len(rawFiles) != 1 {
		t.Fatalf("want 1 raw file, got %v", rawFiles)
	}
	data, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), rawFiles[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	hdr := data[:ebiten.RawRGBAFrameHeaderSize]
	if got := int32(binary.LittleEndian.Uint32(hdr[16:20])); got != int32(min.X) {
		t.Errorf("raw header Min.X: got %d, want %d", got, min.X)
	}
	if got := int32(binary.LittleEndian.Uint32(hdr[20:24])); got != int32(min.Y) {
		t.Errorf("raw header Min.Y: got %d, want %d", got, min.Y)
	}
	if !bytes.Equal(data[ebiten.RawRGBAFrameHeaderSize:], img.Pix) {
		t.Error("raw pixels differ from the submitted pixels")
	}
	frames := manifestFrames(t, s.ImageDumpStreamDirForTesting())
	b := frames[0]["bounds"].(map[string]any)
	minRec := b["Min"].(map[string]any)
	if int(minRec["X"].(float64)) != min.X || int(minRec["Y"].(float64)) != min.Y {
		t.Errorf("manifest bounds Min: got %v, want %v", b, min)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// PNG keeps the non-zero bounds only in the manifest; the pixels remain
	// correct relative to the normalized origin.
	s2, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "png"),
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s2.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s2.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	pngFiles := listFrameFiles(t, s2.ImageDumpStreamDirForTesting(), ".png")
	pngData, err := os.ReadFile(filepath.Join(s2.ImageDumpStreamDirForTesting(), pngFiles[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	decoded, err := png.Decode(bytes.NewReader(pngData))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Bounds().Dx() != bounds.Dx() || decoded.Bounds().Dy() != bounds.Dy() {
		t.Errorf("decoded size %v, want %vx%v", decoded.Bounds(), bounds.Dx(), bounds.Dy())
	}
	frames2 := manifestFrames(t, s2.ImageDumpStreamDirForTesting())
	b2 := frames2[0]["bounds"].(map[string]any)
	minRec2 := b2["Min"].(map[string]any)
	if int(minRec2["X"].(float64)) != min.X || int(minRec2["Y"].(float64)) != min.Y {
		t.Errorf("PNG manifest lost the non-zero origin: %v", b2)
	}
	if err := s2.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamTransparentPixelsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(6, 6, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		Encoding:      ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}

	if err := s.Submit(0, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".rgba")
	if len(files) != 1 {
		t.Fatalf("want 1 raw file, got %v", files)
	}
	data, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), files[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(data) != ebiten.RawRGBAFrameHeaderSize+len(img.Pix) {
		t.Fatalf("raw size: got %d, want %d", len(data), ebiten.RawRGBAFrameHeaderSize+len(img.Pix))
	}
	if magic := binary.LittleEndian.Uint32(data[0:4]); magic != ebiten.RawRGBAFrameMagic {
		t.Errorf("magic: %x", magic)
	}
	if ver := binary.LittleEndian.Uint32(data[4:8]); ver != ebiten.RawRGBAFrameVersion {
		t.Errorf("version: %d", ver)
	}
	if w := binary.LittleEndian.Uint32(data[8:12]); int(w) != bounds.Dx() {
		t.Errorf("header width: %d", w)
	}
	if h := binary.LittleEndian.Uint32(data[12:16]); int(h) != bounds.Dy() {
		t.Errorf("header height: %d", h)
	}
	if !bytes.Equal(data[ebiten.RawRGBAFrameHeaderSize:], img.Pix) {
		t.Errorf("raw pixels differ from submitted pixels (transparent edge not preserved)")
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamDuplicateFrameNumbers(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 8,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}

	// The same frame number three times; file names must stay unique and
	// manifest order must follow submission order.
	for range 3 {
		if err := s.Submit(42, bounds, img.Pix, time.Time{}); err != nil {
			t.Fatalf("Submit: %v", err)
		}
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	if len(files) != 3 {
		t.Fatalf("want 3 files for duplicate frame numbers, got %v", files)
	}
	seen := map[string]bool{}
	for _, f := range files {
		if seen[f] {
			t.Fatalf("duplicate file name: %s", f)
		}
		seen[f] = true
	}
	// File names must sort in submission order on every platform.
	if files[0] >= files[1] || files[1] >= files[2] {
		t.Errorf("file names not monotonically ordered: %v", files)
	}
	frames := manifestFrames(t, s.ImageDumpStreamDirForTesting())
	for i, f := range frames {
		if int64(f["frame"].(float64)) != 42 {
			t.Errorf("frame %d record number: got %v", i, f["frame"])
		}
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamRotationByCountAndSize(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(10, 10, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 16,
		MaxFrames:     2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}

	for i := range 5 {
		if err := s.Submit(int64(i), bounds, img.Pix, time.Time{}); err != nil {
			t.Fatalf("Submit %d: %v", i, err)
		}
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	if len(files) != 2 {
		t.Fatalf("MaxFrames=2: want 2 retained files, got %d: %v", len(files), files)
	}
	frames := manifestFrames(t, s.ImageDumpStreamDirForTesting())
	if len(frames) != 2 {
		t.Fatalf("manifest should retain 2 frames, got %d", len(frames))
	}
	if int64(frames[0]["frame"].(float64)) != 3 || int64(frames[1]["frame"].(float64)) != 4 {
		t.Errorf("rotation kept wrong frames: %v and %v, want 3 and 4",
			frames[0]["frame"], frames[1]["frame"])
	}
	st := s.Stats()
	if st.Written != 5 || st.Dropped != 3 {
		t.Errorf("rotation stats: %+v, want Written=5 Dropped=3", st)
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Size-based rotation: one frame above the limit is written then evicted.
	dir2 := t.TempDir()
	s2, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir2, "out"),
		Encoding:      ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity: 4,
		MaxTotalBytes: 1,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s2.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s2.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if got := listFrameFiles(t, s2.ImageDumpStreamDirForTesting(), ".rgba"); len(got) != 0 {
		t.Errorf("over-limit frame should be rotated out, retained: %v", got)
	}
	if st2 := s2.Stats(); st2.Written != 1 || st2.Dropped != 1 {
		t.Errorf("size rotation stats: %+v, want Written=1 Dropped=1", st2)
	}
	if err := s2.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	assertNoTempFiles(t, s2.ImageDumpStreamDirForTesting())
}

func TestImageDumpStreamZeroCapacityDropsEverything(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 0,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	baseline := runtime.NumGoroutine()

	for i := range 10 {
		err := s.Submit(int64(i), bounds, img.Pix, time.Time{})
		if !errors.Is(err, ebiten.ErrImageDumpQueueFull) {
			t.Fatalf("Submit %d: got %v, want ErrImageDumpQueueFull", i, err)
		}
	}
	st := s.Stats()
	if st.Submitted != 0 || st.Dropped != 10 || st.Written != 0 {
		t.Errorf("zero-capacity stats: %+v", st)
	}
	// Queue-full is backpressure observed through Submit and the counters;
	// Flush itself has no queued frame to fail on and stays green.
	if err := s.Flush(); err != nil {
		t.Errorf("Flush with no accepted frames must succeed, got %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Errorf("second Flush should clear the error, got %v", err)
	}
	// The manifest records the drops even though nothing was written.
	m := readManifest(t, s.ImageDumpStreamDirForTesting())
	if int(m["stats"].(map[string]any)["dropped"].(float64)) != 10 {
		t.Errorf("manifest dropped counter: %v", m["stats"])
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png"); len(got) != 0 {
		t.Errorf("zero-capacity stream wrote files: %v", got)
	}
	assertGoroutinesSettled(t, baseline)
}

// failingWriter simulates a disk that accepts only a limited number of
// successful writes (the first one may be a short write) and then fails. It
// wraps the real temp file so Close releases the descriptor.
type failingWriter struct {
	file       *os.File
	remaining  int
	shortFirst bool
}

func (f *failingWriter) Write(p []byte) (int, error) {
	if f.remaining <= 0 {
		return 0, errors.New("simulated disk full")
	}
	if f.shortFirst {
		f.shortFirst = false
		n := 1
		if len(p) == 0 {
			n = 0
		}
		f.remaining--
		return n, nil
	}
	f.remaining--
	return len(p), nil
}

func (f *failingWriter) Close() error {
	return f.file.Close()
}

func TestImageDumpStreamShortWriteAndWriteErrors(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(8, 8, 2, image.Pt(0, 0))

	var mu sync.Mutex
	failures := 0
	s, err := ebiten.NewImageDumpStreamForTesting(
		ebiten.ImageDumpStreamOptions{
			Dir:           filepath.Join(dir, "out"),
			QueueCapacity: 8,
		},
		func(f *os.File) ebiten.WriteCloserForTesting {
			mu.Lock()
			defer mu.Unlock()
			failures++
			// Fail every writer: the first with a short write that makes no
			// further progress, the rest with hard errors.
			if failures == 1 {
				return &failingWriter{file: f, remaining: 1, shortFirst: true}
			}
			return &failingWriter{file: f, remaining: 0}
		},
	)
	if err != nil {
		t.Fatalf("NewImageDumpStreamForTesting: %v", err)
	}

	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	// Flush must return the encoding/write error rather than hiding it.
	if err := s.Flush(); err == nil {
		t.Errorf("Flush must report the short-write failure")
	} else if errors.Is(err, ebiten.ErrImageDumpQueueFull) {
		t.Errorf("Flush returned a queue error instead of a write error: %v", err)
	}
	if st := s.Stats(); st.Errors == 0 || st.Written != 0 {
		t.Errorf("stats after failed write: %+v", st)
	}
	// No partial frame or temp file may survive.
	if got := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png"); len(got) != 0 {
		t.Errorf("failed write left frame files: %v", got)
	}
	assertNoTempFiles(t, s.ImageDumpStreamDirForTesting())

	// The stream stays usable after a failure.
	if err := s.Submit(2, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("second Submit: %v", err)
	}
	if err := s.Flush(); err == nil {
		t.Errorf("Flush must keep reporting write failures")
	}
	if err := s.Close(); err == nil {
		t.Errorf("Close must surface the last write error")
	}
	assertNoTempFiles(t, s.ImageDumpStreamDirForTesting())

	// The manifest captures the failure records.
	m := readManifest(t, s.ImageDumpStreamDirForTesting())
	if int(m["stats"].(map[string]any)["errors"].(float64)) == 0 {
		t.Errorf("manifest error counter is zero after write failures")
	}
	for _, e := range m["errors"].([]any) {
		rec := e.(map[string]any)
		if rec["op"] != "write" && rec["op"] != "manifest" {
			t.Errorf("unexpected error record op: %v", rec)
		}
	}
}

func TestImageDumpStreamDiskFullOnRealFilesystem(t *testing.T) {
	// A read-only directory makes CreateTemp fail with a real OS error.
	dir := t.TempDir()
	ro := filepath.Join(dir, "readonly")
	if err := os.Mkdir(ro, 0o555); err != nil {
		t.Fatalf("Mkdir: %v", err)
	}
	// Root can write anywhere; skip where the permission bit is meaningless.
	if os.Geteuid() == 0 {
		t.Skip("read-only directories do not restrict root")
	}
	t.Cleanup(func() { _ = os.Chmod(ro, 0o755) })

	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           ro,
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err == nil {
		t.Errorf("Flush must fail when the directory is not writable")
	}
	if st := s.Stats(); st.Errors == 0 {
		t.Errorf("disk-full stats: %+v", st)
	}
	if err := s.Close(); err == nil {
		t.Errorf("Close must surface the disk error")
	}
}

func TestImageDumpStreamInvalidOptionsAndFrames(t *testing.T) {
	dir := t.TempDir()

	if _, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{}); err == nil {
		t.Error("empty Dir must be rejected")
	}
	if _, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir: filepath.Join(dir, "a"), QueueCapacity: -1,
	}); err == nil {
		t.Error("negative QueueCapacity must be rejected")
	}
	if _, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir: filepath.Join(dir, "b"), Encoding: ebiten.ImageDumpEncoding(99),
	}); err == nil {
		t.Error("invalid encoding must be rejected")
	}
	for _, prefix := range []string{"..", ".", "a/b", `a\b`, "/abs"} {
		if _, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
			Dir: filepath.Join(dir, "p"), FilenamePrefix: prefix,
		}); err == nil {
			t.Errorf("illegal prefix %q must be rejected", prefix)
		}
	}

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "ok"),
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, image.Rect(0, 0, 4, 0), make([]byte, 0), time.Time{}); err == nil {
		t.Error("empty bounds must be rejected")
	}
	if err := s.Submit(1, image.Rect(0, 0, 2, 2), make([]byte, 7), time.Time{}); err == nil {
		t.Error("pixel length mismatch must be rejected")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Operations after Close.
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("Submit after Close: got %v", err)
	}
	if err := s.Flush(); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("Flush after Close: got %v", err)
	}
	if err := s.Close(); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("double Close: got %v, want ErrImageDumpStreamClosed", err)
	}
}

func TestImageDumpStreamOversizedFrame(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	rawSize := int64(ebiten.RawRGBAFrameHeaderSize + 4*4*4)

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "raw"),
		Encoding:      ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity: 2,
		MaxFrameBytes: rawSize - 1,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	err = s.Submit(1, bounds, img.Pix, time.Time{})
	if !errors.Is(err, ebiten.ErrImageDumpFrameTooLarge) {
		t.Errorf("raw Submit: got %v, want ErrImageDumpFrameTooLarge", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".rgba"); len(got) != 0 {
		t.Errorf("oversized raw frame created files: %v", got)
	}

	// PNG variant measures the pixel payload before encoding.
	s2, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "png"),
		Encoding:      ebiten.ImageDumpEncodingPNG,
		QueueCapacity: 2,
		MaxFrameBytes: int64(len(img.Pix)) - 1,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s2.Submit(1, bounds, img.Pix, time.Time{}); !errors.Is(err, ebiten.ErrImageDumpFrameTooLarge) {
		t.Errorf("png Submit: got %v", err)
	}
	if err := s2.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamConcurrentSubmit(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(6, 6, 2, image.Pt(0, 0))

	const goroutines = 16
	const perGoroutine = 25

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 7,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	baseline := runtime.NumGoroutine()

	var wg sync.WaitGroup
	var fullCount atomic.Int64
	for g := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range perGoroutine {
				err := s.Submit(int64(g*perGoroutine+i), bounds, img.Pix, time.Time{})
				if errors.Is(err, ebiten.ErrImageDumpQueueFull) {
					fullCount.Add(1)
				} else if err != nil {
					t.Errorf("Submit: %v", err)
				}
			}
		}()
	}
	wg.Wait()
	// Queue-full backpressure must not make Flush fail: no disk error
	// occurred, only bounded drops.
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	st := s.Stats()
	total := int64(goroutines * perGoroutine)
	if int64(st.Written)+int64(st.Dropped) != total {
		t.Errorf("accounting: written %d + dropped %d != %d", st.Written, st.Dropped, total)
	}
	if int64(st.Dropped) != fullCount.Load() {
		t.Errorf("drop counter %d differs from observed full-queue errors %d", st.Dropped, fullCount.Load())
	}
	if st.Errors != 0 {
		t.Errorf("unexpected errors: %+v", st)
	}
	frames := manifestFrames(t, s.ImageDumpStreamDirForTesting())
	if len(frames) != int(st.Written) {
		t.Errorf("manifest frame count %d != written %d", len(frames), st.Written)
	}
	// Sequences in the manifest must be gap-free and ordered.
	for i, f := range frames {
		if uint64(f["sequence"].(float64)) != uint64(i+1) {
			t.Fatalf("manifest sequence gap at %d: %v", i, f["sequence"])
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	assertGoroutinesSettled(t, baseline)
}

// atomic counter is provided by sync/atomic above.

func TestImageDumpStreamDirectoryRenameAndRecreate(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out")
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           out,
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Rename the whole output directory while the stream is open.
	renamed := filepath.Join(dir, "renamed")
	if err := os.Rename(out, renamed); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("original dir still present after rename")
	}

	// The next write must recreate the directory and succeed.
	if err := s.Submit(2, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit after rename: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush after rename: %v", err)
	}
	if got := listFrameFiles(t, out, ".png"); len(got) != 1 {
		t.Errorf("recreated dir contents: %v", got)
	}
	if _, err := os.Stat(filepath.Join(out, ebiten.ManifestFileName)); err != nil {
		t.Errorf("manifest not recreated: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// panickingWriter panics on first use to simulate a consumer crash. It wraps
// the real temp file so Close still releases the descriptor.
type panickingWriter struct {
	file *os.File
}

func (p panickingWriter) Write([]byte) (int, error) { panic("simulated writer crash") }
func (p panickingWriter) Close() error              { return p.file.Close() }

func TestImageDumpStreamConsumerCrash(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	crashOnce := sync.Once{}
	s, err := ebiten.NewImageDumpStreamForTesting(
		ebiten.ImageDumpStreamOptions{
			Dir:           filepath.Join(dir, "out"),
			QueueCapacity: 4,
		},
		func(f *os.File) ebiten.WriteCloserForTesting {
			// Only the frame writer crashes; manifest writes keep working.
			if strings.HasPrefix(filepath.Base(f.Name()), ".frame-") {
				crashOnce.Do(func() {
					panic("boom")
				})
			}
			return f
		},
	)
	if err != nil {
		t.Fatalf("NewImageDumpStreamForTesting: %v", err)
	}
	baseline := runtime.NumGoroutine()

	// Fill the queue so several frames are in flight at crash time.
	for i := range 6 {
		_ = s.Submit(int64(i), bounds, img.Pix, time.Time{})
	}

	// Close must return rather than hang, and the goroutine must not leak.
	done := make(chan error, 1)
	go func() { done <- s.Close() }()
	select {
	case err := <-done:
		if err == nil {
			t.Error("Close should report the consumer crash")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close hung after the consumer crashed")
	}
	assertGoroutinesSettled(t, baseline)
	assertNoTempFiles(t, s.ImageDumpStreamDirForTesting())
	if depth := s.Stats().QueueDepth; depth != 0 {
		t.Errorf("queue depth after crash and Close: got %d, want 0", depth)
	}

	// Every later operation fails fast with the closed error.
	if err := s.Submit(99, bounds, img.Pix, time.Time{}); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("Submit after crash: got %v", err)
	}
	if err := s.Flush(); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("Flush after crash: got %v", err)
	}
	if err := s.Close(); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("second Close after crash: got %v", err)
	}
}

// TestImageDumpStreamFlushAfterCrashWithoutClose verifies a dead worker cannot
// make Flush hang even when the caller never invokes Close.
func TestImageDumpStreamFlushAfterCrashWithoutClose(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	baseline := runtime.NumGoroutine()

	s, err := ebiten.NewImageDumpStreamForTesting(
		ebiten.ImageDumpStreamOptions{
			Dir:           filepath.Join(dir, "out"),
			QueueCapacity: 2,
		},
		func(f *os.File) ebiten.WriteCloserForTesting {
			if strings.HasPrefix(filepath.Base(f.Name()), ".frame-") {
				panic("boom")
			}
			return f
		},
	)
	if err != nil {
		t.Fatalf("NewImageDumpStreamForTesting: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	// Give the worker time to crash, then Flush must fail fast rather than
	// block forever sending into an unread channel.
	deadline := time.Now().Add(5 * time.Second)
	var flushErr error
	for time.Now().Before(deadline) {
		flushErr = s.Flush()
		if errors.Is(flushErr, ebiten.ErrImageDumpStreamClosed) {
			break
		}
		// Early attempts may still observe the in-flight panic error; the
		// important contract is that Flush always returns, never hangs.
		time.Sleep(5 * time.Millisecond)
	}
	if !errors.Is(flushErr, ebiten.ErrImageDumpStreamClosed) {
		t.Fatalf("Flush after an unclosed crash must return ErrImageDumpStreamClosed, got %v", flushErr)
	}
	// Clean up without hanging; Close is idempotent on a dead stream.
	if err := s.Close(); !errors.Is(err, ebiten.ErrImageDumpStreamClosed) {
		t.Errorf("Close after crash: got %v", err)
	}
	assertGoroutinesSettled(t, baseline)
}

func TestImageDumpStreamFlushSemantics(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 4,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}

	// A Flush with nothing queued succeeds and returns promptly.
	if err := s.Flush(); err != nil {
		t.Fatalf("empty Flush: %v", err)
	}

	// Frames submitted before Flush are all on disk when it returns; a frame
	// submitted afterwards is not accidentally included.
	for i := range 3 {
		if err := s.Submit(int64(i), bounds, img.Pix, time.Time{}); err != nil {
			t.Fatalf("Submit %d: %v", i, err)
		}
	}
	flushDone := make(chan error, 1)
	go func() { flushDone <- s.Flush() }()
	select {
	case err := <-flushDone:
		if err != nil {
			t.Fatalf("Flush: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Flush hung")
	}
	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	if len(files) != 3 {
		t.Fatalf("after Flush want 3 files, got %v", files)
	}

	if err := s.Submit(3, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	// Closing flushes the trailing frame as well.
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png"); len(got) != 4 {
		t.Errorf("Close should have flushed the last frame, got %d", len(got))
	}
	assertNoTempFiles(t, s.ImageDumpStreamDirForTesting())
}

func TestImageDumpStreamFilenameAndManifestStability(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:            filepath.Join(dir, "out"),
		QueueCapacity:  4,
		FilenamePrefix: "gpu_regression",
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	for i := range 3 {
		if err := s.Submit(int64(i), bounds, img.Pix, time.Unix(int64(1000+i), 0)); err != nil {
			t.Fatalf("Submit: %v", err)
		}
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// File names: prefix, fixed-width sequence, extension. No platform
	// separators or variable-length components.
	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	wantNames := []string{
		"gpu_regression_00000000000000000001.png",
		"gpu_regression_00000000000000000002.png",
		"gpu_regression_00000000000000000003.png",
	}
	for i, want := range wantNames {
		if files[i] != want {
			t.Errorf("file %d: got %q, want %q", i, files[i], want)
		}
	}

	// Manifest round-trips with a stable schema and deterministic field set.
	first, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), ebiten.ManifestFileName))
	if err != nil {
		t.Fatalf("ReadFile manifest: %v", err)
	}
	var m1, m2 map[string]any
	if err := json.Unmarshal(first, &m1); err != nil {
		t.Fatalf("manifest JSON: %v", err)
	}
	// Re-flush rewrites the same content (no nondeterministic keys/ordering
	// beyond json.MarshalIndent's struct-driven order).
	if err := s.Flush(); err != nil {
		t.Fatalf("second Flush: %v", err)
	}
	second, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), ebiten.ManifestFileName))
	if err != nil {
		t.Fatalf("ReadFile manifest: %v", err)
	}
	if err := json.Unmarshal(second, &m2); err != nil {
		t.Fatalf("manifest JSON: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Errorf("manifest rewrite is not stable:\n%s\nvs\n%s", first, second)
	}
	if m1["version"].(float64) != 1 || m1["encoding"] != "png" {
		t.Errorf("manifest header: %v", m1)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamDoesNotMutateInput(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	original := bytes.Clone(img.Pix)

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		QueueCapacity: 1,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	// Mutate the caller's buffer immediately; the queued copy must be safe.
	for i := range img.Pix {
		img.Pix[i] = 0xaa
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".png")
	data, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), files[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	decoded, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Compare the rendered color at every pixel with the original
	// premultiplied RGBA values, independent of the decoder's concrete type.
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			off := 4 * (y*bounds.Dx() + x)
			pr, pg, pb, pa := decoded.At(x, y).RGBA()
			wantColor := color.NRGBAModel.Convert(color.RGBA{
				R: original[off+0],
				G: original[off+1],
				B: original[off+2],
				A: original[off+3],
			}).(color.NRGBA)
			wr, wg, wb, wa := wantColor.RGBA()
			if pr != wr || pg != wg || pb != wb || pa != wa {
				t.Fatalf("pixel (%d,%d) mismatch: stream wrote mutated buffer", x, y)
			}
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// Compile-time guard that the test helper types satisfy the real interface.
var (
	_ ebiten.WriteCloserForTesting = (*failingWriter)(nil)
	_ ebiten.WriteCloserForTesting = panickingWriter{}
)

func TestImageDumpStreamRawFrameHeaderLayout(t *testing.T) {
	dir := t.TempDir()
	min := image.Pt(3, -5)
	img, bounds := checkerboardImage(5, 7, 2, min)

	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:           filepath.Join(dir, "out"),
		Encoding:      ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity: 2,
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(8, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".rgba")
	data, err := os.ReadFile(filepath.Join(s.ImageDumpStreamDirForTesting(), files[0]))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	hdr := data[:ebiten.RawRGBAFrameHeaderSize]
	var minY int32 = -5
	neg5 := uint32(minY)
	checks := []struct {
		name string
		got  uint32
		want uint32
	}{
		{"magic", binary.LittleEndian.Uint32(hdr[0:4]), ebiten.RawRGBAFrameMagic},
		{"version", binary.LittleEndian.Uint32(hdr[4:8]), ebiten.RawRGBAFrameVersion},
		{"width", binary.LittleEndian.Uint32(hdr[8:12]), 5},
		{"height", binary.LittleEndian.Uint32(hdr[12:16]), 7},
		{"minX", binary.LittleEndian.Uint32(hdr[16:20]), uint32(int32(3))},
		{"minY", binary.LittleEndian.Uint32(hdr[20:24]), neg5},
		{"reserved0", binary.LittleEndian.Uint32(hdr[24:28]), 0},
		{"reserved1", binary.LittleEndian.Uint32(hdr[28:32]), 0},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, c.got, c.want)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestImageDumpStreamPrefixAppliesToAllFiles(t *testing.T) {
	dir := t.TempDir()
	img, bounds := checkerboardImage(4, 4, 2, image.Pt(0, 0))
	s, err := ebiten.NewImageDumpStream(ebiten.ImageDumpStreamOptions{
		Dir:            filepath.Join(dir, "out"),
		Encoding:       ebiten.ImageDumpEncodingRawRGBA,
		QueueCapacity:  2,
		FilenamePrefix: "shot",
	})
	if err != nil {
		t.Fatalf("NewImageDumpStream: %v", err)
	}
	if err := s.Submit(1, bounds, img.Pix, time.Time{}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	files := listFrameFiles(t, s.ImageDumpStreamDirForTesting(), ".rgba")
	if len(files) != 1 || !strings.HasPrefix(files[0], "shot_") {
		t.Errorf("custom prefix not applied: %v", files)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
