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
	"bytes"
	"fmt"
	"image"
	"image/color"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
)

// newCheckerboardImage returns an image with a deterministic checkerboard pattern.
func newCheckerboardImage(bounds image.Rectangle, cell int) *ebiten.Image {
	img := ebiten.NewImageWithOptions(bounds, nil)
	w, h := bounds.Dx(), bounds.Dy()
	pix := make([]byte, 4*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := 4 * (y*w + x)
			if (x/cell+y/cell)%2 == 0 {
				pix[i] = 0xff
				pix[i+1] = 0x00
				pix[i+2] = 0x00
			} else {
				pix[i] = 0x00
				pix[i+1] = 0x00
				pix[i+2] = 0xff
			}
			pix[i+3] = 0xff
		}
	}
	img.WritePixels(pix)
	return img
}

// newTransparentEdgeImage returns an image with a transparent 1-pixel border and an opaque interior.
func newTransparentEdgeImage(w, h int) *ebiten.Image {
	img := ebiten.NewImage(w, h)
	pix := make([]byte, 4*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := 4 * (y*w + x)
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				continue
			}
			pix[i] = byte(x * 0xff / (w - 1))
			pix[i+1] = byte(y * 0xff / (h - 1))
			pix[i+2] = 0x80
			pix[i+3] = 0xff
		}
	}
	img.WritePixels(pix)
	return img
}

// readRegionPixels reads the pixels of img's whole bounds.
func readRegionPixels(img *ebiten.Image) []byte {
	b := img.Bounds()
	pix := make([]byte, 4*b.Dx()*b.Dy())
	img.ReadPixels(pix)
	return pix
}

// drawRegionReference draws src's region r onto dst without the cache,
// and returns the destination's pixels.
func drawRegionReference(dst, src *ebiten.Image, r image.Rectangle, op *ebiten.DrawImageOptions) []byte {
	sub := src.SubImage(r).(*ebiten.Image)
	dst.DrawImage(sub, op)
	return readRegionPixels(dst)
}

func TestImageRegionCacheSubImagePixelCompatibility(t *testing.T) {
	srcs := map[string]*ebiten.Image{
		"checkerboard":    newCheckerboardImage(image.Rect(0, 0, 64, 64), 8),
		"transparentEdge": newTransparentEdgeImage(32, 32),
		"nonZeroBounds":   newCheckerboardImage(image.Rect(5, 7, 5+48, 7+48), 4),
	}
	regions := map[string][]image.Rectangle{
		"checkerboard": {
			image.Rect(0, 0, 8, 8),
			image.Rect(8, 8, 24, 16),
			image.Rect(7, 7, 9, 9),
		},
		"transparentEdge": {
			image.Rect(0, 0, 4, 4),
			image.Rect(28, 28, 32, 32),
		},
		"nonZeroBounds": {
			image.Rect(5, 7, 13, 15),
			image.Rect(0, 0, 20, 20),   // partially out of bounds, negative side
			image.Rect(40, 40, 60, 60), // partially out of bounds, positive side
		},
	}

	for name, src := range srcs {
		for _, r := range regions[name] {
			t.Run(fmt.Sprintf("%s/%v", name, r), func(t *testing.T) {
				cache := ebiten.NewImageRegionCache(nil)
				got := cache.SubImage(src, r, ebiten.FilterNearest, false)
				if got == nil {
					t.Fatal("SubImage returned nil")
				}

				want := src.SubImage(r).(*ebiten.Image)
				if got.Bounds() != want.Bounds() {
					t.Fatalf("bounds: got %v, want %v", got.Bounds(), want.Bounds())
				}

				gotPix := readRegionPixels(got)
				wantPix := readRegionPixels(want)
				if !bytes.Equal(gotPix, wantPix) {
					t.Errorf("pixels differ from the no-cache reference")
				}

				// A second lookup must return the same cached image.
				got2 := cache.SubImage(src, r, ebiten.FilterNearest, false)
				if got2 != got {
					t.Errorf("the second lookup did not return the cached image")
				}
				if stats := cache.Stats(); stats.Hits != 1 || stats.Misses != 1 {
					t.Errorf("stats: got %v, want 1 hit and 1 miss", stats)
				}
			})
		}
	}
}

func TestImageRegionCacheDrawImagePixelCompatibility(t *testing.T) {
	src := newCheckerboardImage(image.Rect(0, 0, 64, 64), 4)
	r := image.Rect(4, 4, 36, 36)

	for _, filter := range []ebiten.Filter{ebiten.FilterNearest, ebiten.FilterLinear} {
		for _, disableMipmaps := range []bool{false, true} {
			name := fmt.Sprintf("filter=%d/disableMipmaps=%v", filter, disableMipmaps)
			t.Run(name, func(t *testing.T) {
				cache := ebiten.NewImageRegionCache(nil)

				op := &ebiten.DrawImageOptions{
					Filter:         filter,
					DisableMipmaps: disableMipmaps,
				}
				op.GeoM.Scale(1.5, 1.5)

				// Compare on the same destination: linear filtering can differ by one
				// between two distinct destination images due to their atlas placement,
				// which is unrelated to the cache.
				dst := ebiten.NewImage(64, 64)
				want := drawRegionReference(dst, src, r, op)

				dst.Clear()
				cache.DrawImage(dst, src, r, op)
				got := readRegionPixels(dst)

				if !bytes.Equal(got, want) {
					t.Errorf("pixels differ from the no-cache reference")
				}
			})
		}
	}
}

func TestImageRegionCacheReadPixelsPixelCompatibility(t *testing.T) {
	src := newTransparentEdgeImage(32, 32)
	cache := ebiten.NewImageRegionCache(nil)

	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 32, 32),
		image.Rect(1, 1, 31, 31),
		image.Rect(0, 0, 1, 1),
	} {
		ir := r.Intersect(src.Bounds())
		got := make([]byte, 4*ir.Dx()*ir.Dy())
		cache.ReadPixels(src, r, got)

		want := readRegionPixels(src.SubImage(r).(*ebiten.Image))
		if !bytes.Equal(got, want) {
			t.Errorf("region %v: pixels differ from the no-cache reference", r)
		}
	}
}

func TestImageRegionCacheInvalidationOnWrite(t *testing.T) {
	src := ebiten.NewImage(16, 16)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 16, 16)

	// Continuous writes: after each write, the cache must not serve stale pixels.
	for i := 0; i < 4; i++ {
		clr := color.RGBA{R: byte(16 * i), G: 0x01, B: 0x02, A: 0xff}
		src.Fill(clr)

		sub := cache.SubImage(src, r, ebiten.FilterNearest, false)
		got := readRegionPixels(sub)
		for j := 0; j < len(got); j += 4 {
			if got[j] != clr.R || got[j+1] != clr.G || got[j+2] != clr.B || got[j+3] != clr.A {
				t.Fatalf("write %d: got stale pixel (%d, %d, %d, %d), want %v", i, got[j], got[j+1], got[j+2], got[j+3], clr)
			}
		}
	}

	// Every lookup after a write is a miss: only the first lookup hits nothing.
	stats := cache.Stats()
	if stats.Misses != 4 {
		t.Errorf("misses: got %d, want 4 (one per source version)", stats.Misses)
	}
	if stats.Hits != 0 {
		t.Errorf("hits: got %d, want 0", stats.Hits)
	}

	// Without a write, repeated lookups hit.
	sub := cache.SubImage(src, r, ebiten.FilterNearest, false)
	_ = readRegionPixels(sub)
	if stats := cache.Stats(); stats.Hits != 1 {
		t.Errorf("hits after a stable lookup: got %d, want 1", stats.Hits)
	}

	// A write through a sub-image must also invalidate.
	sub2 := src.SubImage(image.Rect(0, 0, 8, 8)).(*ebiten.Image)
	sub2.Fill(color.RGBA{B: 0xff, A: 0xff})
	cache.SubImage(src, r, ebiten.FilterNearest, false)
	if stats := cache.Stats(); stats.Misses != 5 {
		t.Errorf("misses after a sub-image write: got %d, want 5", stats.Misses)
	}
}

func TestImageRegionCacheOverlappingRegions(t *testing.T) {
	src := newCheckerboardImage(image.Rect(0, 0, 32, 32), 2)
	cache := ebiten.NewImageRegionCache(nil)

	r0 := image.Rect(0, 0, 16, 16)
	r1 := image.Rect(8, 8, 24, 24)
	sub0 := cache.SubImage(src, r0, ebiten.FilterNearest, false)
	sub1 := cache.SubImage(src, r1, ebiten.FilterNearest, false)
	if sub0 == sub1 {
		t.Fatal("overlapping regions must have distinct entries")
	}
	if stats := cache.Stats(); stats.Entries != 2 {
		t.Fatalf("entries: got %d, want 2", stats.Entries)
	}

	// The overlapping pixel must read identically through both regions.
	if got, want := sub0.At(8, 8), src.At(8, 8); got != want {
		t.Errorf("sub0.At(8, 8): got %v, want %v", got, want)
	}
	if got, want := sub1.At(8, 8), src.At(8, 8); got != want {
		t.Errorf("sub1.At(8, 8): got %v, want %v", got, want)
	}
}

func TestImageRegionCacheEmptyAndOutOfBounds(t *testing.T) {
	src := ebiten.NewImage(16, 16)
	src.Fill(color.RGBA{G: 0xff, A: 0xff})
	cache := ebiten.NewImageRegionCache(nil)

	regions := []image.Rectangle{
		image.Rect(32, 32, 48, 48),     // fully out of bounds
		image.Rect(-32, -32, -16, -16), // fully out of bounds, negative
		image.Rect(4, 4, 4, 4),         // zero area
		// image.Rect normalizes its arguments, so construct an inverted rectangle literally.
		image.Rectangle{Min: image.Pt(8, 8), Max: image.Pt(4, 4)}, // inverted
	}
	for _, r := range regions {
		sub := cache.SubImage(src, r, ebiten.FilterNearest, false)
		if sub == nil {
			t.Errorf("region %v: got nil, want an empty image", r)
			continue
		}
		if !sub.Bounds().Empty() {
			t.Errorf("region %v: got non-empty bounds %v", r, sub.Bounds())
		}
		// Drawing an empty region must be a no-op.
		dst := ebiten.NewImage(8, 8)
		cache.DrawImage(dst, src, r, nil)
	}
	if stats := cache.Stats(); stats.Entries != 0 {
		t.Errorf("empty regions must not be cached, got %d entries", stats.Entries)
	}

	// An oversized region is intersected with the bounds.
	sub := cache.SubImage(src, image.Rect(-100, -100, 100, 100), ebiten.FilterNearest, false)
	if got, want := sub.Bounds(), image.Rect(0, 0, 16, 16); got != want {
		t.Errorf("oversized region: got bounds %v, want %v", got, want)
	}
	if got, want := sub.At(3, 5), (color.RGBA{G: 0xff, A: 0xff}); got != want {
		t.Errorf("oversized region: got %v, want %v", got, want)
	}
}

func TestImageRegionCacheDisposeAndDeallocate(t *testing.T) {
	t.Run("Deallocate", func(t *testing.T) {
		src := ebiten.NewImage(16, 16)
		src.Fill(color.RGBA{R: 0xff, A: 0xff})
		cache := ebiten.NewImageRegionCache(nil)
		r := image.Rect(0, 0, 16, 16)

		cache.SubImage(src, r, ebiten.FilterNearest, false)
		src.Deallocate()
		// Deallocate clears the image. The cache must not serve stale pixels.
		src.Fill(color.RGBA{B: 0xff, A: 0xff})

		sub := cache.SubImage(src, r, ebiten.FilterNearest, false)
		if got, want := sub.At(0, 0), (color.RGBA{B: 0xff, A: 0xff}); got != want {
			t.Errorf("after Deallocate: got %v, want %v", got, want)
		}
		if stats := cache.Stats(); stats.Misses != 2 {
			t.Errorf("misses: got %d, want 2 (the entry must be invalidated)", stats.Misses)
		}
	})

	t.Run("Dispose", func(t *testing.T) {
		src := ebiten.NewImage(16, 16)
		cache := ebiten.NewImageRegionCache(nil)
		cache.SubImage(src, image.Rect(0, 0, 8, 8), ebiten.FilterNearest, false)

		src.Dispose()
		// Repeated releases must be stable.
		src.Dispose()

		if got := cache.SubImage(src, image.Rect(0, 0, 8, 8), ebiten.FilterNearest, false); got != nil {
			t.Errorf("SubImage on a disposed source: got %v, want nil", got)
		}
		// Invalidate on a disposed image must be a no-op.
		cache.Invalidate(src)
		cache.Invalidate(src)
		if stats := cache.Stats(); stats.Entries != 0 {
			t.Errorf("entries after Invalidate: got %d, want 0", stats.Entries)
		}
	})
}

func TestImageRegionCacheRecycle(t *testing.T) {
	src := newCheckerboardImage(image.Rect(0, 0, 32, 32), 4)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 16, 16)

	// Mix cached regions and recyclable sub-images.
	cached := cache.SubImage(src, r, ebiten.FilterNearest, false)
	rec := src.RecyclableSubImage(r)
	if got, want := rec.At(4, 4), cached.At(4, 4); got != want {
		t.Errorf("recyclable sub-image: got %v, want %v", got, want)
	}
	rec.Recycle()

	// The cache must be unaffected by the recycle.
	if got := cache.SubImage(src, r, ebiten.FilterNearest, false); got != cached {
		t.Errorf("the cached region changed after recycling another image")
	}
	if stats := cache.Stats(); stats.Hits != 1 {
		t.Errorf("hits: got %d, want 1", stats.Hits)
	}
}

func TestImageRegionCacheMaxEntries(t *testing.T) {
	src := newCheckerboardImage(image.Rect(0, 0, 64, 64), 8)
	cache := ebiten.NewImageRegionCache(&ebiten.ImageRegionCacheOptions{MaxEntries: 4})

	var subs []*ebiten.Image
	for i := 0; i < 10; i++ {
		r := image.Rect(i*4, 0, i*4+4, 4)
		subs = append(subs, cache.SubImage(src, r, ebiten.FilterNearest, false))
	}

	stats := cache.Stats()
	if stats.Entries > 4 {
		t.Errorf("entries: got %d, want at most 4", stats.Entries)
	}
	if stats.Evictions == 0 {
		t.Errorf("evictions: got 0, want some evictions")
	}

	// Evicted or not, every returned region must still read correct pixels:
	// entries share the source's pixels and never hold copies.
	for i, sub := range subs {
		want := src.At(i*4+1, 1)
		if got := sub.At(i*4+1, 1); got != want {
			t.Errorf("region %d: got %v, want %v", i, got, want)
		}
	}
}

func TestImageRegionCacheConcurrentReadOnly(t *testing.T) {
	src := newCheckerboardImage(image.Rect(0, 0, 64, 64), 8)
	cache := ebiten.NewImageRegionCache(nil)
	regions := []image.Rectangle{
		image.Rect(0, 0, 16, 16),
		image.Rect(16, 16, 32, 32),
		image.Rect(8, 8, 40, 40),
	}

	// Warm the cache up so the goroutines mostly hit.
	for _, r := range regions {
		cache.SubImage(src, r, ebiten.FilterNearest, false)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			r := regions[g%len(regions)]
			ir := r.Intersect(src.Bounds())
			pix := make([]byte, 4*ir.Dx()*ir.Dy())
			for i := 0; i < 32; i++ {
				sub := cache.SubImage(src, r, ebiten.FilterNearest, false)
				if sub == nil {
					t.Error("SubImage returned nil")
					return
				}
				cache.ReadPixels(src, r, pix)
				_ = cache.Stats()
			}
		}(g)
	}
	wg.Wait()

	stats := cache.Stats()
	if stats.Hits == 0 {
		t.Errorf("concurrent reads: got no hits")
	}
	if stats.Entries != len(regions) {
		t.Errorf("entries: got %d, want %d", stats.Entries, len(regions))
	}
}

func TestImageRegionCacheStats(t *testing.T) {
	src := ebiten.NewImage(16, 16)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 8, 8)

	if stats := cache.Stats(); stats.HitRate() != 0 {
		t.Errorf("hit rate without lookups: got %f, want 0", stats.HitRate())
	}

	cache.SubImage(src, r, ebiten.FilterNearest, false)
	cache.SubImage(src, r, ebiten.FilterNearest, false)
	cache.SubImage(src, r, ebiten.FilterNearest, false)

	stats := cache.Stats()
	if stats.Hits != 2 || stats.Misses != 1 {
		t.Fatalf("got %d hits and %d misses, want 2 hits and 1 miss", stats.Hits, stats.Misses)
	}
	if got, want := stats.HitRate(), 2.0/3.0; got != want {
		t.Errorf("hit rate: got %f, want %f", got, want)
	}

	// Different drawing states are different entries.
	cache.SubImage(src, r, ebiten.FilterLinear, false)
	cache.SubImage(src, r, ebiten.FilterLinear, true)
	if stats := cache.Stats(); stats.Entries != 3 {
		t.Errorf("entries: got %d, want 3", stats.Entries)
	}

	cache.Reset()
	if stats := cache.Stats(); stats.Entries != 0 || stats.Hits != 0 || stats.Misses != 0 {
		t.Errorf("after Reset: got %+v, want all zero", stats)
	}
}

func TestImageRegionCacheInvalidate(t *testing.T) {
	src := ebiten.NewImage(16, 16)
	other := ebiten.NewImage(16, 16)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 8, 8)

	cache.SubImage(src, r, ebiten.FilterNearest, false)
	cache.SubImage(other, r, ebiten.FilterNearest, false)

	// Invalidating through a sub-image must invalidate the original's entries.
	cache.Invalidate(src.SubImage(image.Rect(0, 0, 4, 4)).(*ebiten.Image))
	if stats := cache.Stats(); stats.Entries != 1 {
		t.Errorf("entries after Invalidate: got %d, want 1", stats.Entries)
	}

	cache.Invalidate(other)
	if stats := cache.Stats(); stats.Entries != 0 {
		t.Errorf("entries after Invalidate: got %d, want 0", stats.Entries)
	}
}

func BenchmarkImageRegionCacheSubImage(b *testing.B) {
	src := newCheckerboardImage(image.Rect(0, 0, 2048, 2048), 64)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.SubImage(src, r, ebiten.FilterNearest, false)
	}
	b.StopTimer()
	stats := cache.Stats()
	b.Logf("hit rate: %.4f (%d hits, %d misses)", stats.HitRate(), stats.Hits, stats.Misses)
}

func BenchmarkImageRegionCacheSubImageUncachedReference(b *testing.B) {
	src := newCheckerboardImage(image.Rect(0, 0, 2048, 2048), 64)
	r := image.Rect(0, 0, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sub := src.RecyclableSubImage(r)
		sub.Recycle()
	}
}

func BenchmarkImageRegionCacheDrawImage(b *testing.B) {
	src := newCheckerboardImage(image.Rect(0, 0, 2048, 2048), 64)
	dst := ebiten.NewImage(256, 256)
	cache := ebiten.NewImageRegionCache(nil)
	r := image.Rect(0, 0, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.DrawImage(dst, src, r, nil)
	}
	b.StopTimer()
	stats := cache.Stats()
	b.Logf("hit rate: %.4f (%d hits, %d misses)", stats.HitRate(), stats.Hits, stats.Misses)
}

func BenchmarkImageRegionCacheDrawImageUncachedReference(b *testing.B) {
	src := newCheckerboardImage(image.Rect(0, 0, 2048, 2048), 64)
	dst := ebiten.NewImage(256, 256)
	r := image.Rect(0, 0, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sub := src.RecyclableSubImage(r)
		dst.DrawImage(sub, nil)
		sub.Recycle()
	}
}
