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

package ebiten

import (
	"image"
	"sync"
	"sync/atomic"
)

// defaultMaxRegionCacheEntries is the default maximum number of entries in an ImageRegionCache.
const defaultMaxRegionCacheEntries = 1024

// ImageRegionCacheOptions represents options for NewImageRegionCache.
type ImageRegionCacheOptions struct {
	// MaxEntries is the maximum number of cached regions.
	// When the cache is full, stale entries are removed first, and then
	// the least recently used entries are evicted.
	// If MaxEntries is not positive, a default value is used.
	MaxEntries int
}

// regionCacheKey identifies a cached region.
// The source pixel version is part of the key, so an entry becomes unreachable
// as soon as the source image's pixels change.
type regionCacheKey struct {
	// src is the original image. A sub-image and its original share entries.
	src *Image

	// version is the pixel version of src when the entry was created.
	version uint64

	// region is the region in src's coordinate space, already intersected with src's bounds.
	region image.Rectangle

	// filter and disableMipmaps are the drawing states the region is used with.
	filter         Filter
	disableMipmaps bool
}

// regionCacheEntry is a cached region.
type regionCacheEntry struct {
	// img is a sub-image of the source sharing its pixels.
	// Since no pixels are copied, a cached entry never holds stale pixels and
	// never owns a separate texture.
	img *Image

	// lastUsed is the logical clock value of the last access, for LRU eviction.
	lastUsed atomic.Int64
}

// ImageRegionCacheStats reports statistics of an ImageRegionCache.
type ImageRegionCacheStats struct {
	// Hits is the number of lookups served from the cache.
	Hits int64

	// Misses is the number of lookups that created a new entry.
	Misses int64

	// Evictions is the number of entries removed to satisfy the memory limit.
	Evictions int64

	// Entries is the current number of entries in the cache.
	Entries int
}

// HitRate returns the ratio of Hits to the total number of lookups.
// HitRate returns 0 when no lookup was performed.
func (s ImageRegionCacheStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// ImageRegionCache caches reusable regions of source images.
//
// A typical use case is a map editor that draws small viewports of the same large image
// every frame: instead of establishing a new sub-image and its region state on every
// access, the cache returns a reusable region keyed by the source image's pixel version,
// the region, the filter, and the mipmap option.
//
// Cached regions share the pixels with the source image, so drawing or reading a cached
// region is pixel-compatible with using [Image.SubImage], [Image.DrawImage], and
// [Image.ReadPixels] directly, and a cached region never holds stale pixels.
// Entries are invalidated automatically when the source image's pixels change, and can be
// invalidated explicitly with [ImageRegionCache.Invalidate].
//
// The cache holds no GPU resources of its own: entries are sub-images that share the
// source's textures. Deallocating the source or switching the graphics backend therefore
// requires no action on the cache; the pixel version changes and the entries are recreated
// lazily on the next access.
//
// An ImageRegionCache is safe for concurrent use by multiple goroutines,
// as long as the images used with it are used in a concurrent-safe way
// (e.g. read-only accesses like SubImage and ReadPixels).
type ImageRegionCache struct {
	// mu protects entries.
	// Read-only lookups only require a read lock, so concurrent read-only use scales.
	mu      sync.RWMutex
	entries map[regionCacheKey]*regionCacheEntry

	maxEntries int

	// clock is a logical clock for LRU eviction.
	clock atomic.Int64

	hits      atomic.Int64
	misses    atomic.Int64
	evictions atomic.Int64
}

// NewImageRegionCache returns a new ImageRegionCache with the given options.
// If options is nil, the default setting is used.
func NewImageRegionCache(options *ImageRegionCacheOptions) *ImageRegionCache {
	maxEntries := defaultMaxRegionCacheEntries
	if options != nil && options.MaxEntries > 0 {
		maxEntries = options.MaxEntries
	}
	return &ImageRegionCache{
		entries:    map[regionCacheKey]*regionCacheEntry{},
		maxEntries: maxEntries,
	}
}

// normalizeRegion resolves src to its original image and intersects r with src's bounds,
// following the same rules as [Image.SubImage].
func normalizeRegion(src *Image, r image.Rectangle) (*Image, image.Rectangle) {
	orig := src
	if orig.isSubImage() {
		orig = orig.original
	}
	r = r.Intersect(src.Bounds())
	// Need to check Empty explicitly. See the standard image package implementations.
	if r.Empty() {
		r = image.Rectangle{}
	}
	return orig, r
}

// SubImage returns a cached sub-image of src for the region r.
//
// The region is intersected with src's bounds, following [Image.SubImage].
// filter and disableMipmaps are the drawing states the region is intended to be used with;
// regions used with different states are cached as separate entries.
//
// The returned image shares pixels with src and must not be disposed.
// If src is disposed, SubImage returns nil.
func (c *ImageRegionCache) SubImage(src *Image, r image.Rectangle, filter Filter, disableMipmaps bool) *Image {
	if src.isDisposed() {
		return nil
	}

	orig, r := normalizeRegion(src, r)

	// Do not cache empty regions. There is nothing to reuse.
	if r.Empty() {
		if img, ok := orig.SubImage(r).(*Image); ok {
			return img
		}
		return nil
	}

	key := regionCacheKey{
		src:            orig,
		version:        orig.pixelsVersion(),
		region:         r,
		filter:         filter,
		disableMipmaps: disableMipmaps,
	}

	clock := c.clock.Add(1)

	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if ok {
		entry.lastUsed.Store(clock)
		c.hits.Add(1)
		return entry.img
	}

	// Create the sub-image outside the write lock. Image.SubImage has its own mutex.
	img, _ := orig.SubImage(r).(*Image)
	if img == nil {
		// The source was disposed concurrently. Do not cache a nil image.
		return nil
	}

	c.mu.Lock()
	if entry, ok := c.entries[key]; ok {
		// Another goroutine won the race.
		c.mu.Unlock()
		entry.lastUsed.Store(clock)
		c.hits.Add(1)
		return entry.img
	}
	c.evictLocked()
	entry = &regionCacheEntry{img: img}
	entry.lastUsed.Store(clock)
	c.entries[key] = entry
	c.mu.Unlock()

	c.misses.Add(1)
	return img
}

// evictLocked makes room for a new entry.
// Stale entries, whose source pixel version no longer matches, are removed first.
// If the cache is still full, the least recently used entry is evicted.
// c.mu must be held for writing.
func (c *ImageRegionCache) evictLocked() {
	if len(c.entries) < c.maxEntries {
		return
	}

	// Remove stale entries first. An entry is stale when the source's pixels changed
	// or the source was disposed after the entry was created.
	for k := range c.entries {
		if k.src.isDisposed() || k.src.pixelsVersion() != k.version {
			delete(c.entries, k)
			c.evictions.Add(1)
		}
	}
	if len(c.entries) < c.maxEntries {
		return
	}

	// Evict the least recently used entry.
	var oldestKey regionCacheKey
	var oldestClock int64 = -1
	for k, e := range c.entries {
		if used := e.lastUsed.Load(); oldestClock < 0 || used < oldestClock {
			oldestKey = k
			oldestClock = used
		}
	}
	delete(c.entries, oldestKey)
	c.evictions.Add(1)
}

// DrawImage draws the region r of src onto dst, reusing a cached region.
//
// The result is pixel-compatible with dst.DrawImage(src.SubImage(r), options).
// The filter and the mipmap option are taken from options.
// When src is disposed, DrawImage panics, like [Image.DrawImage].
// When dst is disposed and src is not, DrawImage does nothing, like [Image.DrawImage].
func (c *ImageRegionCache) DrawImage(dst, src *Image, r image.Rectangle, options *DrawImageOptions) {
	var filter Filter
	var disableMipmaps bool
	if options != nil {
		filter = options.Filter
		disableMipmaps = options.DisableMipmaps
	}

	sub := c.SubImage(src, r, filter, disableMipmaps)
	if sub == nil {
		panic("ebiten: the given image to DrawImage must not be disposed")
	}
	if sub.Bounds().Empty() {
		return
	}
	dst.DrawImage(sub, options)
}

// ReadPixels reads the pixels of the region r of src into pixels, reusing a cached region.
//
// The result is pixel-compatible with src.SubImage(r).ReadPixels(pixels):
// len(pixels) must be 4 times the width times the height of r intersected with src's bounds.
// When src is disposed, ReadPixels panics, like [Image.ReadPixels].
func (c *ImageRegionCache) ReadPixels(src *Image, r image.Rectangle, pixels []byte) {
	// ReadPixels does not depend on the drawing state, so the zero values are used for the key.
	sub := c.SubImage(src, r, FilterNearest, false)
	if sub == nil {
		panic("ebiten: the image is already disposed")
	}
	sub.ReadPixels(pixels)
}

// Invalidate removes all the cached regions of src.
// If src is a sub-image, the regions of its original image are removed.
// Invalidate is safe to call on a disposed image and does nothing in that case.
func (c *ImageRegionCache) Invalidate(src *Image) {
	orig := src
	if orig.isSubImage() {
		orig = orig.original
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for k := range c.entries {
		if k.src == orig {
			delete(c.entries, k)
		}
	}
}

// Reset removes all the cached regions and resets the statistics.
func (c *ImageRegionCache) Reset() {
	c.mu.Lock()
	clear(c.entries)
	c.mu.Unlock()

	c.hits.Store(0)
	c.misses.Store(0)
	c.evictions.Store(0)
}

// Stats returns the current statistics of the cache.
func (c *ImageRegionCache) Stats() ImageRegionCacheStats {
	c.mu.RLock()
	entries := len(c.entries)
	c.mu.RUnlock()
	return ImageRegionCacheStats{
		Hits:      c.hits.Load(),
		Misses:    c.misses.Load(),
		Evictions: c.evictions.Load(),
		Entries:   entries,
	}
}
