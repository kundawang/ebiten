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
	"container/list"
	"sync"
)

// GlyphAtlasBudget limits the glyph images retained by the glyph caches.
//
// A glyph cache rasterizes each glyph into an [github.com/hajimehoshi/ebiten/v2.Image].
// Without a budget, a cache only trims its entries when their count exceeds a
// soft limit and enough ticks have elapsed; an application that switches
// between many fonts, sizes, or color emoji can accumulate far more GPU
// memory than it reuses. A GlyphAtlasBudget enforces hard bounds on the
// retained bytes, the retained glyph count, and the age of retained glyphs,
// across every cache attached to it.
//
// Eviction removes an image from its cache but never frees it synchronously:
// Ebitengine releases the underlying GPU storage through its runtime cleanup
// once no queued draw references the image anymore. This keeps eviction safe
// for a draw already in flight. Regenerating an evicted glyph produces the
// same pixels as the original.
//
// The budget covers the rasterized outline, bitmap, and COLR color glyph
// images of [GoTextFaceSource] and [GoXFace]. SVG glyph images use a
// separate, independently bounded cache and are not counted here.
//
// The zero value is not usable; create one with [NewGlyphAtlasBudget].
//
// A budget is safe for concurrent use.
type GlyphAtlasBudget struct {
	mu sync.Mutex

	// Configuration. Zero means the corresponding bound is disabled.
	maxBytes     int64
	maxGlyphs    int
	maxIdleTicks int64

	// clock is the budget's own monotonic logical clock. It advances once
	// per cache access, so LRU order is deterministic even when the
	// Ebitengine tick does not advance (e.g. in tests).
	clock uint64

	// Registry of every attached image cache. The value is the cache's
	// victim sink, invoked when an externally-triggered sweep (SetLimits,
	// Clear, UnpinFamily) selects its entries.
	caches map[int64]victimSink

	// deferredVictims holds entries selected by a sweep that ran while an
	// owning cache's mutex was held (a miss-driven register). They are
	// dispatched after the budget mutex is released, so no cache lock is
	// ever taken under the budget mutex.
	deferredVictims []*cacheEntry

	// nextCacheID hands out registry ids.
	nextCacheID int64

	// Global LRU list across all caches. Each element is *lruEntry.
	lru list.List

	// Aggregate counters.
	totalBytes int64

	// Pinned families: family id -> pin count. Entries of a pinned family
	// are exempt from eviction while the count is positive.
	pins map[uint64]int

	// Statistics.
	stats GlyphAtlasStats

	// closed makes repeated [GlyphAtlasBudget.Close] calls harmless.
	closed bool
}

// victimSink receives cache entries selected by a sweep that runs without
// the owning cache's mutex (SetLimits, Clear, UnpinFamily). Implementations
// remove the entries from their own map/list and release GPU storage under
// their own lock.
type victimSink interface {
	applyBudgetVictims([]*cacheEntry)
	detachForBudgetClose()
}

// GlyphAtlasBudgetOptions configures a [GlyphAtlasBudget].
//
// The zero value is valid: every bound is disabled, in which case the
// attached caches keep their built-in soft-limit behavior.
type GlyphAtlasBudgetOptions struct {
	// MaxBytes is the maximum total number of pixel bytes retained across
	// all attached caches. Each glyph image costs 4 bytes per pixel
	// (premultiplied RGBA). Zero disables the byte bound.
	//
	// A glyph larger than MaxBytes on its own is never cached; it is still
	// rendered correctly on every use.
	MaxBytes int64

	// MaxGlyphs is the maximum number of glyph images retained across all
	// attached caches. Zero disables the glyph-count bound.
	MaxGlyphs int

	// MaxIdleTicks is the maximum number of Ebitengine ticks during which a
	// glyph may stay cached without being used. A glyph used on every frame
	// never ages out. Zero disables the time-based bound.
	//
	// When the Ebitengine tick is unavailable (e.g. before a game starts),
	// the budget's own access clock drives LRU ordering instead.
	MaxIdleTicks int64
}

// GlyphAtlasStats is a point-in-time snapshot of a [GlyphAtlasBudget].
//
// The fields are safe to log or expose for observability.
type GlyphAtlasStats struct {
	// CachedGlyphs is the number of glyph images currently retained.
	CachedGlyphs int

	// CachedBytes is the sum of the pixel bytes of the retained glyph images.
	CachedBytes int64

	// PeakGlyphs is the largest CachedGlyphs value observed since the budget
	// was created or last reset.
	PeakGlyphs int

	// PeakBytes is the largest CachedBytes value observed since the budget
	// was created or last reset.
	PeakBytes int64

	// Hits is the number of cache lookups served without rasterizing.
	Hits int64

	// Misses is the number of cache lookups that rasterized a glyph.
	Misses int64

	// Evictions is the number of glyph images removed to satisfy a bound.
	Evictions int64

	// Oversized is the number of glyphs that were rendered but not cached
	// because they exceeded MaxBytes.
	Oversized int64

	// PinnedFamilies is the number of font families currently pinned.
	PinnedFamilies int
}

// HitRate returns Hits/(Hits+Misses), or 0 when no lookups happened.
func (s GlyphAtlasStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// NewGlyphAtlasBudget creates a [GlyphAtlasBudget] with the given bounds.
//
// Attach the budget to caches through [GoTextFaceSource.SetGlyphAtlasBudget]
// or [GoXFace.SetGlyphAtlasBudget].
func NewGlyphAtlasBudget(options GlyphAtlasBudgetOptions) *GlyphAtlasBudget {
	if options.MaxBytes < 0 {
		panic("text: MaxBytes must not be negative at NewGlyphAtlasBudget")
	}
	if options.MaxGlyphs < 0 {
		panic("text: MaxGlyphs must not be negative at NewGlyphAtlasBudget")
	}
	if options.MaxIdleTicks < 0 {
		panic("text: MaxIdleTicks must not be negative at NewGlyphAtlasBudget")
	}
	return &GlyphAtlasBudget{
		maxBytes:     options.MaxBytes,
		maxGlyphs:    options.MaxGlyphs,
		maxIdleTicks: options.MaxIdleTicks,
		caches:       map[int64]victimSink{},
		pins:         map[uint64]int{},
	}
}

// SetLimits updates the budget's bounds.
//
// Tightening a limit evicts entries synchronously until the new bounds hold.
// A zero value disables the corresponding bound, matching
// [GlyphAtlasBudgetOptions].
//
// SetLimits is safe to call concurrently with rendering.
func (b *GlyphAtlasBudget) SetLimits(maxBytes int64, maxGlyphs int, maxIdleTicks int64) {
	if maxBytes < 0 || maxGlyphs < 0 || maxIdleTicks < 0 {
		panic("text: budget limits must not be negative at SetLimits")
	}
	b.mu.Lock()
	b.maxBytes = maxBytes
	b.maxGlyphs = maxGlyphs
	b.maxIdleTicks = maxIdleTicks
	victims := b.collectVictimsLocked(nil)
	b.mu.Unlock()
	b.dispatch(victims)
}

// Limits returns the current bounds. A zero value means the bound is disabled.
func (b *GlyphAtlasBudget) Limits() (maxBytes int64, maxGlyphs int, maxIdleTicks int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.maxBytes, b.maxGlyphs, b.maxIdleTicks
}

// Stats returns a copy of the current statistics.
func (b *GlyphAtlasBudget) Stats() GlyphAtlasStats {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.stats
	s.CachedGlyphs = b.lru.Len()
	s.CachedBytes = b.totalBytes
	s.PinnedFamilies = len(b.pins)
	return s
}

// ResetStats zeroes the cumulative counters and peak watermarks. The current
// occupancy (CachedGlyphs, CachedBytes) is not reset.
func (b *GlyphAtlasBudget) ResetStats() {
	b.mu.Lock()
	b.stats = GlyphAtlasStats{
		CachedGlyphs:   b.lru.Len(),
		CachedBytes:    b.totalBytes,
		PinnedFamilies: len(b.pins),
	}
	b.mu.Unlock()
}

// PinFamily marks every cache of the given font family as hot. Pinned
// entries are never evicted while at least one pin on the family remains.
//
// The family id is the value reported by [GoTextFaceSource.ID] or
// [GoXFace.ID]. Pinning an unknown family is harmless and takes effect if a
// cache of that family is attached later.
func (b *GlyphAtlasBudget) PinFamily(familyID uint64) {
	b.mu.Lock()
	b.pins[familyID]++
	b.mu.Unlock()
}

// UnpinFamily removes one pin from the font family.
//
// UnpinFamily panics if the family is not pinned.
func (b *GlyphAtlasBudget) UnpinFamily(familyID uint64) {
	b.mu.Lock()
	n, ok := b.pins[familyID]
	if !ok {
		b.mu.Unlock()
		panic("text: the font family is not pinned at UnpinFamily")
	}
	if n == 1 {
		delete(b.pins, familyID)
	} else {
		b.pins[familyID] = n - 1
	}
	victims := b.collectVictimsLocked(nil)
	b.mu.Unlock()
	b.dispatch(victims)
}

// Clear removes every evictable glyph image from every attached cache.
// Glyphs of pinned families are retained; call [GlyphAtlasBudget.UnpinFamily]
// first to remove them too. Subsequent rendering regenerates any needed
// glyph, producing pixels identical to the removed ones.
//
// Images still referenced by an in-flight draw stay alive until the draw
// completes; only the cache's reference is dropped.
func (b *GlyphAtlasBudget) Clear() {
	b.mu.Lock()
	victims := b.collectVictimsLocked(func(e *lruEntry) bool {
		return !b.isPinnedLocked(e) && e.leases == 0
	})
	b.mu.Unlock()
	b.dispatch(victims)
}

// Close detaches every cache from the budget. The caches keep their current
// glyph images and revert to their built-in soft limits; nothing is freed
// synchronously. After Close the budget must not be reused: attaching a
// cache to a closed budget panics.
func (b *GlyphAtlasBudget) Close() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	caches := make([]victimSink, 0, len(b.caches))
	for id, c := range b.caches {
		caches = append(caches, c)
		delete(b.caches, id)
	}
	b.lru.Init()
	b.totalBytes = 0
	b.pins = map[uint64]int{}
	b.mu.Unlock()

	for _, c := range caches {
		c.detachForBudgetClose()
	}
}

// attach registers a cache and returns its registry id.
func (b *GlyphAtlasBudget) attach(c victimSink, familyID uint64) int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		panic("text: the glyph atlas budget has already been closed")
	}
	id := b.nextCacheID
	b.nextCacheID++
	b.caches[id] = c
	return id
}

// detach unregisters a cache. The cache has already cleared the lru
// pointers of its entries and passes them for aggregate cleanup.
func (b *GlyphAtlasBudget) detach(c victimSink, entries []*lruEntry) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, e := range entries {
		b.removeEntryLocked(e, false)
	}
	for id, registered := range b.caches {
		if registered == c {
			delete(b.caches, id)
			break
		}
	}
}

// recordHit / recordMiss update the lookup counters.
func (b *GlyphAtlasBudget) recordHit() {
	b.mu.Lock()
	b.stats.Hits++
	b.mu.Unlock()
}

func (b *GlyphAtlasBudget) recordMiss() {
	b.mu.Lock()
	b.stats.Misses++
	b.mu.Unlock()
}

// isPinnedLocked reports whether the entry's family currently has a pin.
func (b *GlyphAtlasBudget) isPinnedLocked(e *lruEntry) bool {
	return b.pins[e.familyID] > 0
}

// lruEntry is the budget's per-glyph bookkeeping. One exists per cached
// glyph while a budget is attached.
type lruEntry struct {
	// budget is the owning budget, used by release.
	budget *GlyphAtlasBudget

	// sink receives the owning cache entry when a sweep selects it.
	sink victimSink

	// element is the owning cache entry.
	element *cacheEntry

	// familyID is the font family for pinning.
	familyID uint64

	// bytes is the glyph image's pixel byte cost.
	bytes int64

	// tick is the Ebitengine tick at the last access, or infTick when the
	// entry never expires by tick.
	tick int64

	// atime is the budget-clock timestamp of the last access, for LRU.
	atime uint64

	// leases is the number of in-flight users. A leased entry is never
	// evicted; the budget skips it and continues scanning.
	leases int

	// onLRU reports whether the entry is still linked into the budget LRU.
	onLRU bool

	// lruElement is the entry's position in the global LRU list.
	lruElement *list.Element
}

// acquireLease pins an entry for the duration of a draw. The caller must
// call releaseLease exactly once per successful acquisition.
func (b *GlyphAtlasBudget) acquireLease(e *lruEntry) {
	b.mu.Lock()
	e.leases++
	b.mu.Unlock()
}

func (b *GlyphAtlasBudget) releaseLease(e *lruEntry) {
	b.mu.Lock()
	if e.leases <= 0 {
		b.mu.Unlock()
		panic("text: releasing a glyph image lease that is not held")
	}
	e.leases--
	b.mu.Unlock()
}

// release implements glyphLease.
func (e *lruEntry) release() {
	e.budget.releaseLease(e)
}

// admit reports whether a glyph of the given byte cost may enter the cache.
// An oversized glyph is rejected up front so it can never displace reusable
// glyphs.
func (b *GlyphAtlasBudget) admit(byteCost int64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.maxBytes > 0 && byteCost > b.maxBytes {
		b.stats.Oversized++
		return false
	}
	return true
}

// register adds a freshly created entry to the global LRU and runs the
// eviction policy. Entries selected for removal are queued in the budget
// and returned for the caller to apply; flushDeferredVictims dispatches the
// cross-cache remainder after the budget mutex has been released. The
// caller holds the owning cache's mutex.
func (b *GlyphAtlasBudget) register(e *lruEntry, nowTick int64, localVictims []*cacheEntry) []*cacheEntry {
	b.mu.Lock()
	b.clock++
	e.atime = b.clock
	if nowTick != infTick {
		e.tick = nowTick
	}
	e.lruElement = b.lru.PushBack(e)
	e.onLRU = true
	b.totalBytes += e.bytes
	victims := b.collectVictimsLocked(nil)
	if n := b.lru.Len(); n > b.stats.PeakGlyphs {
		b.stats.PeakGlyphs = n
	}
	if b.totalBytes > b.stats.PeakBytes {
		b.stats.PeakBytes = b.totalBytes
	}
	// Split victims into entries owned by the calling cache (applied
	// immediately by the caller under its own lock) and entries owned by
	// other caches (deferred until the budget mutex is released).
	caller := e.element.owner
	for _, ent := range victims {
		if ent.owner == caller {
			localVictims = append(localVictims, ent)
		} else {
			b.deferredVictims = append(b.deferredVictims, ent)
		}
	}
	b.mu.Unlock()
	return localVictims
}

// flushDeferredVictims dispatches eviction victims queued by register calls
// on other caches. It must be called without the budget mutex; each victim
// sink takes its own cache lock. Victims owned by the calling cache are
// excluded since that cache applies them inline.
func (b *GlyphAtlasBudget) flushDeferredVictims(except victimSink) {
	b.mu.Lock()
	queued := b.deferredVictims
	b.deferredVictims = nil
	b.mu.Unlock()
	if len(queued) == 0 {
		return
	}
	bySink := map[victimSink][]*cacheEntry{}
	for _, ent := range queued {
		if ent.owner == except {
			continue
		}
		bySink[ent.owner] = append(bySink[ent.owner], ent)
	}
	for sink, batch := range bySink {
		sink.applyBudgetVictims(batch)
	}
}

// touch moves an existing entry to the LRU back and refreshes its timestamps.
func (b *GlyphAtlasBudget) touch(e *lruEntry, nowTick int64) {
	b.mu.Lock()
	b.clock++
	e.atime = b.clock
	if nowTick != infTick {
		e.tick = nowTick
	}
	if e.onLRU {
		b.lru.MoveToBack(e.lruElement)
	}
	b.mu.Unlock()
}

// removeEntryLocked unlinks an entry from the budget LRU and counters. The
// caller holds b.mu. When countEviction is true the Evictions statistic
// advances; cache-detach paths pass false.
func (b *GlyphAtlasBudget) removeEntryLocked(e *lruEntry, countEviction bool) {
	if !e.onLRU {
		return
	}
	b.lru.Remove(e.lruElement)
	e.lruElement = nil
	e.onLRU = false
	b.totalBytes -= e.bytes
	if countEviction {
		b.stats.Evictions++
	}
}

// collectVictimsLocked scans the LRU front to back and unlinks every entry
// that must leave. The optional predicate, when non-nil, adds an extra
// removal condition (used by Clear). Pinned or leased entries are skipped.
// The scan always terminates, so a fully-pinned working set cannot cause
// unbounded retries. The caller holds b.mu.
func (b *GlyphAtlasBudget) collectVictimsLocked(extra func(*lruEntry) bool) []*cacheEntry {
	if extra == nil && b.maxBytes == 0 && b.maxGlyphs == 0 && b.maxIdleTicks == 0 {
		return nil
	}
	var victims []*cacheEntry
	var next *list.Element
	for el := b.lru.Front(); el != nil; el = next {
		next = el.Next()
		e := el.Value.(*lruEntry)

		// With no explicit predicate, stop once the hard bounds hold and
		// no remaining entry can be old enough. The LRU order means the
		// youngest entries are at the back, so a non-expired front entry
		// still allows older entries later in the list to be scanned when
		// the hard bounds are exceeded.
		if extra == nil && b.withinLimitsLocked() && !b.maxIdleTicksExceededLocked(e) {
			return victims
		}
		if b.isPinnedLocked(e) || e.leases > 0 {
			continue
		}
		if extra != nil && !extra(e) {
			continue
		}
		b.removeEntryLocked(e, true)
		victims = append(victims, e.element)
	}
	return victims
}

// dispatch hands each victim batch to the cache that owns it. Each cache
// applies the batch under its own mutex, so no cache lock is ever taken
// under the budget mutex.
func (b *GlyphAtlasBudget) dispatch(victims []*cacheEntry) {
	if len(victims) == 0 {
		return
	}
	bySink := map[victimSink][]*cacheEntry{}
	for _, ent := range victims {
		if ent.owner == nil {
			continue
		}
		bySink[ent.owner] = append(bySink[ent.owner], ent)
	}
	for sink, batch := range bySink {
		sink.applyBudgetVictims(batch)
	}
}

// withinLimitsLocked reports whether the aggregate occupancy satisfies all
// enabled hard bounds. The caller holds b.mu.
func (b *GlyphAtlasBudget) withinLimitsLocked() bool {
	if b.maxBytes > 0 && b.totalBytes > b.maxBytes {
		return false
	}
	if b.maxGlyphs > 0 && b.lru.Len() > b.maxGlyphs {
		return false
	}
	return true
}

// maxIdleTicksExceededLocked reports whether an entry is older than the
// time-based bound. Non-expiring entries and entries without a tick
// timestamp never age out by time.
func (b *GlyphAtlasBudget) maxIdleTicksExceededLocked(e *lruEntry) bool {
	if b.maxIdleTicks == 0 || e.tick == 0 || e.tick == infTick {
		return false
	}
	return currentTick()-e.tick > b.maxIdleTicks
}
