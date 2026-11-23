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

package ebiten_test

import (
	"image"
	"image/color"
	"strings"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
)

// newTestImage creates an image with deterministic, distinguishable pixels.
func newTestImage(w, h int, opaque bool) *ebiten.Image {
	img := ebiten.NewImage(w, h)
	pix := make([]byte, 4*w*h)
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			idx := 4 * (j*w + i)
			pix[idx] = byte(0x40 + i*7)
			pix[idx+1] = byte(0x80 + j*5)
			pix[idx+2] = byte(0xc0 - i*3)
			if opaque {
				pix[idx+3] = 0xff
			} else {
				pix[idx+3] = byte(0x80 + (i+j)*2)
			}
		}
	}
	img.WritePixels(pix)
	return img
}

// batchItemToDrawImageOptions converts DrawImageBatchOptions to the equivalent DrawImageOptions.
func batchItemToDrawImageOptions(o *ebiten.DrawImageBatchOptions) *ebiten.DrawImageOptions {
	if o == nil {
		return nil
	}
	return &ebiten.DrawImageOptions{
		GeoM:           o.GeoM,
		ColorScale:     o.ColorScale,
		ColorM:         o.ColorM,
		CompositeMode:  o.CompositeMode,
		Blend:          o.Blend,
		Filter:         o.Filter,
		DisableMipmaps: o.DisableMipmaps,
	}
}

// drawItemSequentially draws one batch item with the regular DrawImage API.
// A clip is emulated by drawing on a sub-image of the destination, which clips
// to the same rectangle in the same coordinate system.
func drawItemSequentially(dst, src *ebiten.Image, o *ebiten.DrawImageBatchOptions) {
	op := batchItemToDrawImageOptions(o)
	if o != nil && !o.Clip.Empty() {
		clip := o.Clip.Intersect(dst.Bounds())
		if clip.Empty() {
			return
		}
		sub := dst.SubImage(clip).(*ebiten.Image)
		sub.DrawImage(src, op)
		return
	}
	dst.DrawImage(src, op)
}

// mixedBatchItems returns a set of items covering sub-images, non-integer
// coordinates, negative scales, premultiplied-alpha color scales, color
// matrices, composite modes, blends, filters, mipmaps, degenerate matrices,
// and clips.
func mixedBatchItems(srcA, srcB, tileA, tileB *ebiten.Image) []struct {
	img *ebiten.Image
	op  *ebiten.DrawImageBatchOptions
} {
	var items []struct {
		img *ebiten.Image
		op  *ebiten.DrawImageBatchOptions
	}
	add := func(img *ebiten.Image, op *ebiten.DrawImageBatchOptions) {
		items = append(items, struct {
			img *ebiten.Image
			op  *ebiten.DrawImageBatchOptions
		}{img: img, op: op})
	}

	// A plain draw.
	add(srcA, nil)

	// A non-integer translation.
	op := &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(10.5, 20.25)
	add(srcA, op)

	// A negative scale with a non-integer translation.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Scale(-1.5, 1.25)
	op.GeoM.Translate(40.5, 3.75)
	add(srcA, op)

	// A sub-image with a premultiplied-alpha color scale.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(5, 40)
	op.ColorScale.Scale(0.5, 0.75, 1.0, 0.5)
	add(tileA, op)

	// A rotation with the linear filter.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(-8, -8)
	op.GeoM.Rotate(0.3)
	op.GeoM.Translate(40, 24)
	op.Filter = ebiten.FilterLinear
	add(tileB, op)

	// A composite mode.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(8, 8)
	op.CompositeMode = ebiten.CompositeModeLighter
	add(srcB, op)

	// A custom blend.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(44, 44)
	op.Blend = ebiten.BlendCopy
	add(srcB, op)

	// A degenerate matrix, which draws nothing.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Scale(0, 1)
	add(srcA, op)

	// Shrinks with the linear filter, with and without mipmaps.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Scale(0.4, 0.4)
	op.GeoM.Translate(48, 0)
	op.Filter = ebiten.FilterLinear
	add(srcA, op)

	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Scale(0.15, 0.15)
	op.GeoM.Translate(56, 8)
	op.Filter = ebiten.FilterLinear
	add(srcA, op)

	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Scale(0.4, 0.4)
	op.GeoM.Translate(48, 16)
	op.Filter = ebiten.FilterLinear
	op.DisableMipmaps = true
	add(srcA, op)

	// A color matrix, which is never merged with other items.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(0, 48)
	op.ColorM.Translate(0.2, 0, 0, 0)
	add(srcB, op)

	// A clipped draw.
	op = &ebiten.DrawImageBatchOptions{}
	op.GeoM.Translate(3, 3)
	op.Clip = image.Rect(4, 4, 12, 12)
	add(srcA, op)

	// A draw clipped out completely, which draws nothing.
	op = &ebiten.DrawImageBatchOptions{}
	op.Clip = image.Rect(1000, 1000, 2000, 2000)
	add(srcA, op)

	return items
}

// comparePixels compares two pixel buffers of a size×size image, allowing a
// small per-channel error: GPU rasterization can differ by one least
// significant bit even between two identical sequential draws on this
// environment, as existing tests like TestImagePixels also assume.
func comparePixels(t *testing.T, pix0, pix1 []byte, size int) {
	t.Helper()
	for i := 0; i < len(pix0); i += 4 {
		for ch := 0; ch < 4; ch++ {
			d := int(pix0[i+ch]) - int(pix1[i+ch])
			if d < -1 || d > 1 {
				p := i / 4
				t.Fatalf("pixel at (%d, %d), channel %d: got %d, want %d", p%size, p/size, ch, pix1[i+ch], pix0[i+ch])
			}
		}
	}
}

func TestDrawImageBatchPixels(t *testing.T) {
	srcA := newTestImage(16, 16, false)
	srcB := newTestImage(8, 8, true)
	tileset := newTestImage(32, 16, false)
	tileA := tileset.SubImage(image.Rect(0, 0, 16, 16)).(*ebiten.Image)
	tileB := tileset.SubImage(image.Rect(16, 0, 32, 16)).(*ebiten.Image)

	items := mixedBatchItems(srcA, srcB, tileA, tileB)

	const size = 64
	dst0 := ebiten.NewImage(size, size)
	dst1 := ebiten.NewImage(size, size)

	// Draw the items sequentially with DrawImage.
	for _, it := range items {
		drawItemSequentially(dst0, it.img, it.op)
	}

	// Draw the same items with a batch.
	var batch ebiten.DrawImageBatch
	for _, it := range items {
		batch.Add(it.img, it.op)
	}
	if got, want := batch.Len(), len(items); got != want {
		t.Fatalf("batch.Len(): got %d, want %d", got, want)
	}
	dst1.DrawImageBatch(&batch)

	pix0 := make([]byte, 4*size*size)
	pix1 := make([]byte, 4*size*size)
	dst0.ReadPixels(pix0)
	dst1.ReadPixels(pix1)
	comparePixels(t, pix0, pix1, size)
}

func TestDrawImageBatchEmpty(t *testing.T) {
	dst := ebiten.NewImage(16, 16)

	var batch ebiten.DrawImageBatch
	if got := batch.Len(); got != 0 {
		t.Fatalf("batch.Len(): got %d, want 0", got)
	}

	// Drawing an empty batch is a no-op.
	dst.DrawImageBatch(&batch)

	// Resetting an empty batch is fine.
	batch.Reset()

	// A batch becomes empty again after Reset.
	src := newTestImage(8, 8, true)
	batch.Add(src, nil)
	batch.Reset()
	if got := batch.Len(); got != 0 {
		t.Fatalf("batch.Len() after Reset: got %d, want 0", got)
	}
	dst.DrawImageBatch(&batch)

	// Nothing should have been drawn.
	pix := make([]byte, 4*16*16)
	dst.ReadPixels(pix)
	for i, p := range pix {
		if p != 0 {
			t.Fatalf("pix[%d]: got %d, want 0", i, p)
		}
	}
}

func TestDrawImageBatchNilBatch(t *testing.T) {
	dst := ebiten.NewImage(16, 16)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("DrawImageBatch with a nil batch should panic")
		}
	}()
	dst.DrawImageBatch(nil)
}

func TestDrawImageBatchAddPanics(t *testing.T) {
	src := newTestImage(8, 8, true)

	// A nil source image.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Add with a nil image should panic")
			}
		}()
		var batch ebiten.DrawImageBatch
		batch.Add(nil, nil)
	}()

	// A disposed source image.
	func() {
		disposed := ebiten.NewImage(8, 8)
		disposed.Dispose()
		defer func() {
			if r := recover(); r == nil {
				t.Error("Add with a disposed image should panic")
			}
		}()
		var batch ebiten.DrawImageBatch
		batch.Add(disposed, nil)
	}()

	// An invalid filter.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Add with an invalid filter should panic")
			}
		}()
		var batch ebiten.DrawImageBatch
		batch.Add(src, &ebiten.DrawImageBatchOptions{Filter: ebiten.Filter(100)})
	}()

	// An invalid composite mode.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Add with an invalid composite mode should panic")
			}
		}()
		var batch ebiten.DrawImageBatch
		batch.Add(src, &ebiten.DrawImageBatchOptions{CompositeMode: ebiten.CompositeMode(100)})
	}()
}

func TestDrawImageBatchDisposedSource(t *testing.T) {
	dst := ebiten.NewImage(16, 16)
	srcs := make([]*ebiten.Image, 3)
	for i := range srcs {
		srcs[i] = newTestImage(8, 8, true)
	}

	var batch ebiten.DrawImageBatch
	for _, src := range srcs {
		batch.Add(src, nil)
	}

	// Dispose a source image after it was added.
	srcs[1].Dispose()

	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("DrawImageBatch with a disposed source image should panic")
			}
			msg, ok := r.(string)
			if !ok {
				t.Fatalf("the panic value should be a string, but was %T", r)
			}
			if !strings.Contains(msg, "index 1") {
				t.Fatalf("the panic message should contain the item index, but was: %s", msg)
			}
		}()
		dst.DrawImageBatch(&batch)
	}()

	// The failure must not leave a half-drawn batch: nothing should be drawn.
	pix := make([]byte, 4*16*16)
	dst.ReadPixels(pix)
	for i, p := range pix {
		if p != 0 {
			t.Fatalf("pix[%d]: got %d, want 0 (a failed batch must not be drawn partially)", i, p)
		}
	}
}

func TestDrawImageBatchDisposedDestination(t *testing.T) {
	src := newTestImage(8, 8, true)
	var batch ebiten.DrawImageBatch
	batch.Add(src, nil)

	dst := ebiten.NewImage(16, 16)
	dst.Dispose()

	// Drawing on a disposed destination does nothing and should not panic,
	// as long as no source image is disposed.
	dst.DrawImageBatch(&batch)
}

func TestDrawImageBatchSelfDraw(t *testing.T) {
	dst := newTestImage(16, 16, true)

	var batch ebiten.DrawImageBatch
	batch.Add(dst, nil)

	// A self-draw panics in the same way as DrawImage.
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("a self-draw should panic")
		}
	}()
	dst.DrawImageBatch(&batch)
}

// flushCommandsForTest flushes the pending graphics commands by reading back a
// pixel of a fresh image, so that the number of commands created by subsequent
// draws can be counted independently of earlier draws.
func flushCommandsForTest() {
	img := ebiten.NewImage(1, 1)
	// Allocate the image on the GPU; otherwise the read-back completes without
	// submitting the pending commands.
	img.Fill(color.White)
	img.ReadPixels(make([]byte, 4))
	img.Deallocate()
}

func TestDrawImageBatchCommandCount(t *testing.T) {
	src := newTestImage(8, 8, true)
	dst := ebiten.NewImage(64, 64)

	const n = 100

	newBatch := func(itemOptions func(i int) *ebiten.DrawImageBatchOptions) *ebiten.DrawImageBatch {
		var batch ebiten.DrawImageBatch
		for i := 0; i < n; i++ {
			batch.Add(src, itemOptions(i))
		}
		return &batch
	}

	translatedOptions := func(i int) *ebiten.DrawImageBatchOptions {
		op := &ebiten.DrawImageBatchOptions{}
		op.GeoM.Translate(float64(i%8), float64(i/8))
		return op
	}

	tests := []struct {
		name string
		draw func()
		want int64
	}{
		{
			name: "compatible items are merged into one command",
			draw: func() {
				dst.DrawImageBatch(newBatch(translatedOptions))
			},
			want: 1,
		},
		{
			name: "sequential DrawImage calls merge the same way",
			draw: func() {
				for i := 0; i < n; i++ {
					op := &ebiten.DrawImageOptions{}
					op.GeoM.Translate(float64(i%8), float64(i/8))
					dst.DrawImage(src, op)
				}
			},
			want: 1,
		},
		{
			name: "alternating blends are not reordered",
			draw: func() {
				dst.DrawImageBatch(newBatch(func(i int) *ebiten.DrawImageBatchOptions {
					op := translatedOptions(i)
					if i%2 == 1 {
						op.Blend = ebiten.BlendLighter
					}
					return op
				}))
			},
			want: n,
		},
		{
			name: "consecutive compatible runs are merged",
			draw: func() {
				dst.DrawImageBatch(newBatch(func(i int) *ebiten.DrawImageBatchOptions {
					op := translatedOptions(i)
					if i >= n/2 {
						op.Blend = ebiten.BlendLighter
					}
					return op
				}))
			},
			want: 2,
		},
		{
			name: "alternating clips merge at the command queue",
			draw: func() {
				dst.DrawImageBatch(newBatch(func(i int) *ebiten.DrawImageBatchOptions {
					op := translatedOptions(i)
					if i%2 == 0 {
						op.Clip = image.Rect(0, 0, 32, 32)
					} else {
						op.Clip = image.Rect(32, 32, 64, 64)
					}
					return op
				}))
			},
			want: 1,
		},
		{
			name: "alternating filters split commands",
			draw: func() {
				dst.DrawImageBatch(newBatch(func(i int) *ebiten.DrawImageBatchOptions {
					op := translatedOptions(i)
					if i%2 == 1 {
						op.Filter = ebiten.FilterLinear
						op.DisableMipmaps = true
					}
					return op
				}))
			},
			want: n,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Warm up the atlas state (e.g. the destination's isolation from the
			// source's backend), so that one-time internal draws are not counted.
			tc.draw()
			flushCommandsForTest()
			ebiten.ResetNumDrawTrianglesCommandsForTesting()
			tc.draw()
			if got := ebiten.NumDrawTrianglesCommandsForTesting(); got != tc.want {
				t.Errorf("number of draw-triangles commands: got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestDrawImageBatchCommandCountSplit(t *testing.T) {
	// Force the batch to split a run of compatible items into multiple internal
	// draw calls by limiting the number of vertices per draw call.
	restore := ebiten.SetDrawImageBatchMaxVertexCountForTesting(8)
	defer restore()

	src := newTestImage(8, 8, true)
	dst := ebiten.NewImage(64, 64)

	var batch ebiten.DrawImageBatch
	for i := 0; i < 5; i++ {
		op := &ebiten.DrawImageBatchOptions{}
		op.GeoM.Translate(float64(i*8), 0)
		batch.Add(src, op)
	}

	// Warm up the atlas state before counting commands.
	dst.DrawImageBatch(&batch)
	flushCommandsForTest()
	ebiten.ResetNumDrawTrianglesCommandsForTesting()
	dst.DrawImageBatch(&batch)
	// 5 quads with at most 2 quads (8 vertices) per internal draw call: 3 enqueues.
	if got, want := ebiten.NumDrawTrianglesEnqueuesForTesting(), int64(3); got != want {
		t.Errorf("number of draw-triangles enqueues: got %d, want %d", got, want)
	}
	// The split pieces are compatible, so the command queue merges them back
	// into one command.
	if got, want := ebiten.NumDrawTrianglesCommandsForTesting(), int64(1); got != want {
		t.Errorf("number of draw-triangles commands: got %d, want %d", got, want)
	}

	// All the items must be drawn even when the run is split.
	for i := 0; i < 5; i++ {
		if got := dst.At(i*8, 0).(color.RGBA); got.A == 0 {
			t.Errorf("dst.At(%d, 0) should be drawn, but was transparent", i*8)
		}
	}
}

func TestDrawImageBatchHuge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	src := newTestImage(4, 4, true)
	const size = 256
	dst := ebiten.NewImage(size, size)

	const n = 100000
	var batch ebiten.DrawImageBatch
	for i := 0; i < n; i++ {
		op := &ebiten.DrawImageBatchOptions{}
		op.GeoM.Translate(float64(i%64)*4, float64((i/64)%64)*4)
		batch.Add(src, op)
	}
	if got := batch.Len(); got != n {
		t.Fatalf("batch.Len(): got %d, want %d", got, n)
	}

	// Warm up the atlas state with a single draw before counting commands.
	var warmup ebiten.DrawImageBatch
	warmup.Add(src, nil)
	dst.DrawImageBatch(&warmup)
	flushCommandsForTest()
	ebiten.ResetNumDrawTrianglesCommandsForTesting()
	dst.DrawImageBatch(&batch)
	if got := ebiten.NumDrawTrianglesCommandsForTesting(); got != 1 {
		t.Errorf("number of draw-triangles commands: got %d, want 1", got)
	}

	// Spot-check that the draws happened.
	if got := dst.At(0, 0).(color.RGBA); got.A == 0 {
		t.Error("dst.At(0, 0) should be drawn, but was transparent")
	}
	if got := dst.At(63*4, 63*4).(color.RGBA); got.A == 0 {
		t.Error("dst.At(252, 252) should be drawn, but was transparent")
	}
}

func TestDrawImageBatchConcurrent(t *testing.T) {
	src := newTestImage(4, 4, true)
	const size = 64
	dst := ebiten.NewImage(size, size)

	// Independent batches can be submitted from multiple goroutines. Each
	// goroutine draws to its own disjoint column so that the result does not
	// depend on the submission order.
	const goroutines = 4
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			var batch ebiten.DrawImageBatch
			for k := 0; k < 64; k++ {
				op := &ebiten.DrawImageBatchOptions{}
				op.GeoM.Translate(float64(g*16+(k%4)*4), float64((k/4)*4))
				batch.Add(src, op)
			}
			dst.DrawImageBatch(&batch)
		}(g)
	}
	wg.Wait()

	// Every position in each goroutine's column should be drawn.
	for g := 0; g < goroutines; g++ {
		if got := dst.At(g*16, 0).(color.RGBA); got.A == 0 {
			t.Errorf("dst.At(%d, 0) should be drawn, but was transparent", g*16)
		}
		if got := dst.At(g*16+12, 60).(color.RGBA); got.A == 0 {
			t.Errorf("dst.At(%d, 60) should be drawn, but was transparent", g*16+12)
		}
	}
}

func TestDrawImageBatchReuse(t *testing.T) {
	srcA := newTestImage(16, 16, false)
	srcB := newTestImage(8, 8, true)
	tileset := newTestImage(32, 16, false)
	tileA := tileset.SubImage(image.Rect(0, 0, 16, 16)).(*ebiten.Image)
	tileB := tileset.SubImage(image.Rect(16, 0, 32, 16)).(*ebiten.Image)

	items := mixedBatchItems(srcA, srcB, tileA, tileB)

	const size = 64
	dst0 := ebiten.NewImage(size, size)
	dst1 := ebiten.NewImage(size, size)

	// Reuse the same batch for multiple frames.
	var batch ebiten.DrawImageBatch
	for range 3 {
		batch.Reset()
		for _, it := range items {
			batch.Add(it.img, it.op)
		}
		dst0.DrawImageBatch(&batch)
	}
	for range 3 {
		for _, it := range items {
			drawItemSequentially(dst1, it.img, it.op)
		}
	}

	pix0 := make([]byte, 4*size*size)
	pix1 := make([]byte, 4*size*size)
	dst0.ReadPixels(pix0)
	dst1.ReadPixels(pix1)
	comparePixels(t, pix0, pix1, size)
}

func BenchmarkDrawImageSequential(b *testing.B) {
	const n = 1000
	src := ebiten.NewImage(16, 16)
	dst := ebiten.NewImage(640, 640)
	op := &ebiten.DrawImageOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for k := 0; k < n; k++ {
			op.GeoM.Reset()
			op.GeoM.Translate(float64(k%40)*16, float64(k/40)*16)
			dst.DrawImage(src, op)
		}
	}
}

func BenchmarkDrawImageBatch(b *testing.B) {
	const n = 1000
	src := ebiten.NewImage(16, 16)
	dst := ebiten.NewImage(640, 640)
	var batch ebiten.DrawImageBatch
	op := &ebiten.DrawImageBatchOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch.Reset()
		for k := 0; k < n; k++ {
			op.GeoM.Reset()
			op.GeoM.Translate(float64(k%40)*16, float64(k/40)*16)
			batch.Add(src, op)
		}
		dst.DrawImageBatch(&batch)
	}
}
