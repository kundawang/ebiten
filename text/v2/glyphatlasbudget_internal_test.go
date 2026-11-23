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

package text

import (
	"image/color"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
)

// BenchmarkImageCacheBudgetedHits measures the steady-state cost of a
// budgeted glyph lookup that always hits.
func BenchmarkImageCacheBudgetedHits(b *testing.B) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 64})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)
	for k := range 32 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(b, 2, 2), true
		})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.getOrCreate(i%32, func() (*ebiten.Image, bool) {
			b.Fatal("unexpected miss")
			return nil, false
		})
	}
}

// newTestGlyphImage creates a w*h opaque image and registers cleanup so
// tests never leak GPU storage.
func newTestGlyphImage(tb testing.TB, w, h int) *ebiten.Image {
	tb.Helper()
	img := ebiten.NewImage(w, h)
	img.Fill(color.RGBA{R: uint8(w), G: uint8(h), B: 1, A: 255})
	tb.Cleanup(img.Deallocate)
	return img
}

func TestImageCacheStandaloneSoftLimit(t *testing.T) {
	c := newImageCache[int](2)
	var creates int
	mk := func(k int) *ebiten.Image {
		creates++
		// 2x2 => 16 bytes each.
		return newTestGlyphImage(t, 2, 2)
	}

	// Tick 0: fill beyond the soft limit; entries are non-expiring within
	// cacheLifetime so they survive the immediate sweep.
	for k := range 5 {
		c.getOrCreateAt(k, 0, func() (*ebiten.Image, bool) { return mk(k), true })
	}
	if got := c.len(); got != 5 {
		t.Fatalf("entries before aging: got %d, want 5", got)
	}

	// Advance past cacheLifetime and add one more: old entries must be swept.
	c.getOrCreateAt(99, cacheLifetime+1, func() (*ebiten.Image, bool) { return mk(99), true })
	if got := c.len(); got > 3 {
		t.Errorf("entries after soft-limit sweep: got %d, want <= 3", got)
	}
}

func TestImageCacheHitIdentityAndStats(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 8})
	c := newImageCache[int](128)
	c.setBudget(budget, 1)

	first := c.getOrCreate(1, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 2, 2), true
	})
	again := c.getOrCreate(1, func() (*ebiten.Image, bool) {
		t.Fatal("create must not run on a hit")
		return nil, false
	})
	if again != first {
		t.Fatal("a cache hit changed the image identity")
	}
	stats := budget.Stats()
	if stats.Hits != 1 || stats.Misses != 1 {
		t.Errorf("stats after one miss/one hit: %+v", stats)
	}
	if stats.CachedGlyphs != 1 || stats.CachedBytes != 16 {
		t.Errorf("occupancy: %+v", stats)
	}
}

func TestBudgetGlyphCountEvictionIsLRU(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 3})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	imgs := map[int]*ebiten.Image{}
	for k := range 5 {
		imgs[k] = c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	stats := budget.Stats()
	if stats.CachedGlyphs != 3 {
		t.Errorf("cached glyphs: got %d, want 3", stats.CachedGlyphs)
	}
	if stats.Evictions != 2 {
		t.Errorf("evictions: got %d, want 2", stats.Evictions)
	}
	// The three most recently used keys survive.
	for _, k := range []int{2, 3, 4} {
		if _, ok := c.values[k]; !ok {
			t.Errorf("key %d was evicted but should have been retained", k)
		}
	}
	// Re-reading an evicted key regenerates it and pushes another entry out.
	regenerated := c.getOrCreate(0, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	if regenerated == imgs[0] {
		t.Error("an evicted glyph was served without regeneration")
	}
	if _, ok := c.values[0]; !ok {
		t.Error("regenerated glyph was not cached")
	}
}

func TestBudgetByteEviction(t *testing.T) {
	// Each image is 2x2 => 16 bytes. Budget holds two of them.
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxBytes: 32})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	for k := range 4 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 2, 2), true
		})
	}
	stats := budget.Stats()
	if stats.CachedBytes > 32 {
		t.Errorf("cached bytes %d exceed the 32-byte budget", stats.CachedBytes)
	}
	if stats.CachedGlyphs != 2 {
		t.Errorf("cached glyphs: got %d, want 2", stats.CachedGlyphs)
	}
}

func TestBudgetOversizedGlyphBypassesCache(t *testing.T) {
	// A 4x4 image costs 64 bytes, larger than the 32-byte budget.
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxBytes: 32})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	big := c.getOrCreate(1, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 4, 4), true
	})
	if big == nil {
		t.Fatal("oversized glyph did not render")
	}
	if c.len() != 0 {
		t.Errorf("oversized glyph was cached: %d entries", c.len())
	}
	stats := budget.Stats()
	if stats.Oversized != 1 {
		t.Errorf("oversized counter: got %d, want 1", stats.Oversized)
	}
	// A conforming glyph still caches normally and nothing was displaced.
	small := c.getOrCreate(2, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	if small == nil || c.len() != 1 {
		t.Fatal("conforming glyph failed to cache after an oversized one")
	}
}

func TestBudgetPinnedFamilySurvivesEviction(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 2})
	hot := newImageCache[int](1024)
	hot.setBudget(budget, 7)
	cold := newImageCache[int](1024)
	cold.setBudget(budget, 8)

	// Populate the hot family first so it sits at the LRU front.
	for k := range 2 {
		hot.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	budget.PinFamily(7)

	// Cold glyphs would normally evict the older hot ones.
	for k := range 3 {
		cold.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	if hot.len() != 2 {
		t.Errorf("pinned family lost entries: %d remain", hot.len())
	}
	if cold.len() != 0 {
		t.Errorf("cold entries should all have been evicted, %d remain", cold.len())
	}

	// After unpinning, filling past the limit evicts the oldest hot entries.
	budget.UnpinFamily(7)
	for k := range 3 {
		cold.getOrCreate(100+k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	if hot.len() != 0 {
		t.Errorf("hot entries survived after the family was unpinned: %d remain", hot.len())
	}
}

func TestBudgetLeaseBlocksEviction(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 1})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	first := c.getOrCreate(0, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	img, lease := c.acquire(0)
	if img != first || lease == nil {
		t.Fatal("acquire did not return the cached image and lease")
	}

	// Insertions while the lease is held cannot evict key 0. The budget
	// stays temporarily over its limit instead of retrying forever.
	for k := 1; k <= 4; k++ {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	if _, ok := c.values[0]; !ok {
		t.Fatal("leased glyph was evicted while in flight")
	}
	// Releasing the lease and adding one more entry allows eviction again.
	c.release(lease)
	c.getOrCreate(10, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	if _, ok := c.values[0]; ok {
		t.Error("leased glyph remained cached after its lease was released")
	}
}

func TestBudgetClearRespectsLeasesAndPins(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	for k := range 4 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	_, lease := c.acquire(1)
	budget.Clear()
	if c.len() != 1 {
		t.Errorf("Clear removed the leased entry: %d remain", c.len())
	}
	c.release(lease)
	budget.Clear()
	if c.len() != 0 {
		t.Errorf("Clear after lease release left %d entries", c.len())
	}

	// Pinned entries survive Clear.
	for k := range 3 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	budget.PinFamily(1)
	budget.Clear()
	if c.len() != 3 {
		t.Errorf("Clear removed pinned entries: %d remain", c.len())
	}
}

func TestBudgetSetLimitsShrinks(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 10})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)
	for k := range 6 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	budget.SetLimits(0, 2, 0)
	if got := budget.Stats().CachedGlyphs; got != 2 {
		t.Errorf("after SetLimits: %d cached, want 2", got)
	}
	if c.len() != 2 {
		t.Errorf("cache still holds %d entries after tightening the limit", c.len())
	}
}

func TestBudgetReattachMigratesEntries(t *testing.T) {
	c := newImageCache[int](1024)
	for k := range 3 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 10})
	c.setBudget(budget, 1)
	if got := budget.Stats().CachedGlyphs; got != 3 {
		t.Errorf("migrated %d entries, want 3", got)
	}
	// Detaching keeps the glyphs but drops the budget view.
	c.setBudget(nil, 0)
	if c.len() != 3 {
		t.Errorf("detaching the budget dropped %d cached glyphs", 3-c.len())
	}
	if got := budget.Stats().CachedGlyphs; got != 0 {
		t.Errorf("detached budget still reports %d glyphs", got)
	}
}

func TestBudgetCloseIsIdempotentAndDetaches(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)
	c.getOrCreate(0, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	budget.Close()
	budget.Close()
	if c.len() != 1 {
		t.Error("Close must not drop the cache's own glyph images")
	}
	if c.attached {
		t.Error("cache remained attached after Close")
	}
}

func TestBudgetRepeatedEvictionIsStable(t *testing.T) {
	// A one-glyph budget evicts on essentially every insertion. Repeatedly
	// cycling keys must not corrupt the cache, double-count bytes, or leave
	// dangling entries.
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 1})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	var previous *ebiten.Image
	for round := range 50 {
		for k := range 4 {
			img := c.getOrCreate(k+(round*4), func() (*ebiten.Image, bool) {
				return newTestGlyphImage(t, 2, 2), true
			})
			if img == nil {
				t.Fatal("eviction cycle produced a nil image")
			}
			previous = img
		}
	}
	stats := budget.Stats()
	if stats.CachedGlyphs != 1 {
		t.Errorf("cached glyphs %d, want 1", stats.CachedGlyphs)
	}
	if stats.CachedBytes != 16 {
		t.Errorf("cached bytes %d, want 16", stats.CachedBytes)
	}
	if stats.Evictions == 0 {
		t.Error("no evictions were recorded")
	}
	// An evicted image must remain usable when the caller still holds it;
	// reading it must not panic.
	if previous == nil {
		t.Fatal("no image was produced")
	}
	_ = previous.Bounds()
}

func TestBudgetStatsResetAndPeak(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 4})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)
	for k := range 6 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	stats := budget.Stats()
	if stats.PeakGlyphs < 4 {
		t.Errorf("peak glyphs %d, want >= 4", stats.PeakGlyphs)
	}
	if stats.Evictions == 0 {
		t.Error("evictions were not recorded")
	}
	budget.ResetStats()
	reset := budget.Stats()
	if reset.Evictions != 0 || reset.Hits != 0 || reset.Misses != 0 {
		t.Errorf("ResetStats left cumulative counters: %+v", reset)
	}
	if reset.CachedGlyphs == 0 {
		t.Error("ResetStats must preserve the current occupancy")
	}
}

func TestBudgetIdleTickEviction(t *testing.T) {
	// A glyph-count bound makes insertions trigger sweeps; the idle-time
	// bound then decides which LRU entries are old enough to remove.
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 2, MaxIdleTicks: 10})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	// Keys 0 and 1 at tick 100 fill the budget.
	c.getOrCreateAt(0, 100, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	c.getOrCreateAt(1, 100, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	// Refresh key 0 at tick 105.
	c.getOrCreateAt(0, 105, func() (*ebiten.Image, bool) {
		t.Fatal("key 0 should still be cached")
		return nil, false
	})
	// Insertions at tick 116 exceed MaxGlyphs. Key 1 has been idle for 16
	// ticks and must go; key 0, idle for 11 ticks but refreshed later, is
	// younger and stays.
	c.getOrCreateAt(2, 116, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	if _, ok := c.values[1]; ok {
		t.Error("idle glyph was not evicted after MaxIdleTicks")
	}
	if _, ok := c.values[0]; !ok {
		t.Error("the younger glyph was evicted before the idle one")
	}

	// A SetLimits sweep also enforces the idle bound without an insertion.
	budget.SetLimits(16, 16, 10)
	if _, ok := c.values[0]; !ok {
		t.Error("SetLimits evicted a glyph still within the idle window")
	}
}

// TestBudgetMemoryPressureWithPinsStaysBoundedEventually models a memory
// pressure event while fonts are pinned: the budget may transiently exceed
// its limit (pinned and leased entries cannot be removed) but must
// converge as soon as the pins are released, without unbounded retries.
func TestBudgetMemoryPressureWithPinsStaysBoundedEventually(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 2})
	c := newImageCache[int](1024)
	c.setBudget(budget, 3)

	budget.PinFamily(3)
	// Insert far beyond the limit; every entry is pinned, so the cache
	// temporarily exceeds the bound instead of looping or panicking.
	for k := range 10 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	if c.len() != 10 {
		t.Fatalf("pinned entries were evicted under pressure: %d remain", c.len())
	}

	// A second family sharing the budget cannot displace pinned entries,
	// and its own entries are themselves evictable, so the scan terminates.
	other := newImageCache[int](1024)
	other.setBudget(budget, 4)
	for k := range 6 {
		other.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}

	// Once the pin is dropped, a single insertion converges the whole
	// budget back to the limit.
	budget.UnpinFamily(3)
	c.getOrCreate(999, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 1, 1), true
	})
	if got := budget.Stats().CachedGlyphs; got > 2 {
		t.Errorf("budget did not converge after pressure: %d cached with limit 2", got)
	}
}

// TestEvictedImageStaysUsableByCaller verifies that dropping a cache entry
// never produces a dangling reference: an image handed to the caller before
// eviction remains readable afterwards, and the cache can serve a freshly
// regenerated replacement.
func TestEvictedImageStaysUsableByCaller(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxGlyphs: 1})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	held := c.getOrCreate(0, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 2, 2), true
	})
	// Evict key 0.
	c.getOrCreate(1, func() (*ebiten.Image, bool) {
		return newTestGlyphImage(t, 2, 2), true
	})
	if _, ok := c.values[0]; ok {
		t.Fatal("key 0 was not evicted")
	}
	// The caller's reference must still be a valid, readable image.
	b := held.Bounds()
	pix := make([]byte, 4*b.Dx()*b.Dy())
	held.ReadPixels(pix)
	if pix[3] == 0 {
		t.Error("the held evicted image lost its pixels")
	}
}

// TestZeroConfigBudgetOnlyCountsLookups confirms that a budget with no
// bounds never evicts but still provides observability.
func TestZeroConfigBudgetOnlyCountsLookups(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)
	for k := range 20 {
		c.getOrCreate(k, func() (*ebiten.Image, bool) {
			return newTestGlyphImage(t, 1, 1), true
		})
	}
	if c.len() != 20 {
		t.Errorf("zero-config budget evicted entries: %d remain", c.len())
	}
	stats := budget.Stats()
	if stats.Evictions != 0 {
		t.Errorf("zero-config budget recorded %d evictions", stats.Evictions)
	}
	if stats.Misses != 20 {
		t.Errorf("misses: got %d, want 20", stats.Misses)
	}
}

func TestImageCacheConcurrentAccess(t *testing.T) {
	budget := NewGlyphAtlasBudget(GlyphAtlasBudgetOptions{MaxBytes: 4 << 10})
	c := newImageCache[int](1024)
	c.setBudget(budget, 1)

	const goroutines = 16
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for k := range 200 {
				key := (k + seed) % 24
				img := c.getOrCreate(key, func() (*ebiten.Image, bool) {
					return newTestGlyphImage(t, 2, 2), true
				})
				if img == nil {
					t.Error("nil image under concurrent access")
					return
				}
			}
		}(g)
	}
	wg.Wait()

	stats := budget.Stats()
	if stats.Hits+stats.Misses+stats.Oversized == 0 {
		t.Error("no lookups were recorded")
	}
	if stats.CachedBytes > 4<<10 {
		t.Errorf("cache exceeded byte budget under concurrency: %d", stats.CachedBytes)
	}
}
