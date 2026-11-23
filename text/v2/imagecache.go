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

	"github.com/hajimehoshi/ebiten/v2"
)

// cacheEntry is one glyph image held by an [imageCache].
type cacheEntry struct {
	key any

	// owner is the cache that created the entry. The budget uses it to
	// route eviction victims back to the right cache, since a sweep
	// triggered by one cache's insertion can select entries from another.
	owner victimSink

	// image is the rasterized glyph. It is never nil for a live entry.
	image *ebiten.Image

	// byteCost is 4*w*h, captured at creation so the budget does not have
	// to re-read the image bounds.
	byteCost int64

	// atime is the last Ebitengine tick at which the entry was used, or
	// infTick for entries that must not expire by the legacy soft limit.
	atime int64

	// lru is the entry's bookkeeping inside its [GlyphAtlasBudget], or nil
	// when the cache has no budget.
	lru *lruEntry

	// element is the entry's position in the cache's own LRU list, used for
	// soft-limit sweeps and for budget-driven removal.
	element *list.Element
}

// imageCache is an LRU cache of rasterized glyph images.
//
// Without a budget it preserves the historical behavior of [cache]: a
// count-based soft limit plus a tick-based lifetime. With a budget it also
// participates in the budget's global byte, count, and idle-time eviction.
//
// An imageCache is safe for concurrent use.
type imageCache[K comparable] struct {
	mu sync.Mutex

	// values is the lookup map.
	values map[K]*cacheEntry

	// order is the per-cache LRU list, back = most recently used.
	order list.List

	// softLimit is the legacy entry-count soft limit.
	softLimit int

	// budget, when non-nil, owns the hard eviction policy.
	budget *GlyphAtlasBudget

	// budgetID is the registry id handed out by the budget.
	budgetID int64

	// familyID identifies the owning font for budget pinning.
	familyID uint64

	// attached reports whether the cache is currently registered.
	attached bool
}

// newImageCache creates a standalone image cache with the given legacy
// entry-count soft limit.
func newImageCache[K comparable](softLimit int) *imageCache[K] {
	return &imageCache[K]{
		softLimit: softLimit,
	}
}

// applyBudgetVictims implements victimSink. The budget has already unlinked
// these entries from its LRU and counters; the cache removes them from its
// own map/list under its mutex. The *ebiten.Image objects are not released
// here: Ebitengine frees their GPU storage through its runtime cleanup once
// no queued draw references them, which keeps eviction safe for in-flight
// rendering (a queued DrawImage may still need the atlas region).
func (c *imageCache[K]) applyBudgetVictims(victims []*cacheEntry) {
	c.mu.Lock()
	c.applyVictimsLocked(victims)
	c.mu.Unlock()
}

// detachForBudgetClose implements victimSink. It clears the per-entry
// budget pointers; the retained glyph images stay usable in standalone mode.
func (c *imageCache[K]) detachForBudgetClose() {
	c.mu.Lock()
	c.attached = false
	for el := c.order.Front(); el != nil; el = el.Next() {
		el.Value.(*cacheEntry).lru = nil
	}
	c.mu.Unlock()
}

// setBudget attaches or detaches a budget. Passing nil restores standalone
// mode. Existing entries are migrated into the budget or dropped back to
// standalone bookkeeping; no glyph image is released by reconfiguration
// alone.
func (c *imageCache[K]) setBudget(budget *GlyphAtlasBudget, familyID uint64) {
	c.mu.Lock()

	if c.budget != nil && c.attached {
		entries := make([]*lruEntry, 0, c.order.Len())
		for el := c.order.Front(); el != nil; el = el.Next() {
			if le := el.Value.(*cacheEntry).lru; le != nil {
				entries = append(entries, le)
			}
		}
		c.budget.detach(c, entries)
		c.attached = false
		for el := c.order.Front(); el != nil; el = el.Next() {
			el.Value.(*cacheEntry).lru = nil
		}
	}
	c.budget = budget
	c.familyID = familyID
	if budget == nil {
		c.mu.Unlock()
		return
	}
	id := budget.attach(c, familyID)
	c.budgetID = id
	c.attached = true

	// Migrate existing entries so reconfiguration never silently loses the
	// current working set. Register oldest-first so the budget LRU preserves
	// recency order. Eviction during migration comes back as victims and is
	// applied to this cache before the lock is released.
	now := currentTick()
	var victims []*cacheEntry
	for el := c.order.Front(); el != nil; el = el.Next() {
		ent := el.Value.(*cacheEntry)
		le := &lruEntry{
			budget:   budget,
			sink:     c,
			element:  ent,
			familyID: familyID,
			bytes:    ent.byteCost,
			tick:     ent.atime,
		}
		ent.lru = le
		victims = budget.register(le, now, victims)
	}
	c.applyVictimsLocked(victims)
	c.mu.Unlock()
	budget.flushDeferredVictims(c)
}

// getOrCreate returns the cached image for key, creating it on a miss.
// create returns (nil, false) when the glyph has no image (control chars,
// zero-area glyphs); such results are never cached or billed.
func (c *imageCache[K]) getOrCreate(key K, create func() (*ebiten.Image, bool)) *ebiten.Image {
	return c.getOrCreateAt(key, currentTick(), create)
}

func (c *imageCache[K]) getOrCreateAt(key K, now int64, create func() (*ebiten.Image, bool)) *ebiten.Image {
	c.mu.Lock()

	if ent, ok := c.values[key]; ok {
		// Cache hit: refresh recency.
		if ent.atime != infTick {
			ent.atime = now
		}
		c.order.MoveToBack(ent.element)
		if ent.lru != nil {
			c.budget.touch(ent.lru, now)
		}
		img := ent.image
		c.mu.Unlock()
		if c.budget != nil {
			c.budget.recordHit()
		}
		return img
	}

	img, canExpire := create()
	if img == nil {
		c.mu.Unlock()
		return nil
	}

	byteCost := imageByteCost(img)

	// Oversized glyphs bypass the cache entirely. They still render, but
	// must never displace reusable entries or loop through eviction.
	if c.budget != nil && !c.budget.admit(byteCost) {
		c.mu.Unlock()
		return img
	}

	if c.values == nil {
		c.values = map[K]*cacheEntry{}
	}
	ent := &cacheEntry{
		owner:    c,
		key:      key,
		image:    img,
		byteCost: byteCost,
		atime:    now,
	}
	if !canExpire {
		ent.atime = infTick
	}
	ent.element = c.order.PushBack(ent)
	c.values[key] = ent

	if c.budget != nil {
		le := &lruEntry{
			budget:   c.budget,
			sink:     c,
			element:  ent,
			familyID: c.familyID,
			bytes:    byteCost,
			tick:     ent.atime,
		}
		ent.lru = le
		// register decides which entries (possibly including the new one)
		// must leave the cache to satisfy the bounds.
		victims := c.budget.register(le, now, nil)
		c.applyVictimsLocked(victims)
		c.budget.recordMiss()
	}

	// Legacy soft-limit sweep, kept for standalone caches and as a first
	// line of defense. With a budget the hard bounds inside register() run
	// immediately afterwards and supersede this.
	c.softLimitSweepLocked(now)

	c.mu.Unlock()
	if c.budget != nil {
		// Evictions that targeted other caches are applied now, outside both
		// the budget mutex and this cache's mutex.
		c.budget.flushDeferredVictims(c)
	}
	return img
}

// acquire returns the image for key together with its lease, or (nil, nil)
// on a miss. The caller must release the lease once the image is no longer
// needed by an in-flight draw. On a hit the lookup statistics are updated.
func (c *imageCache[K]) acquire(key K) (*ebiten.Image, *lruEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	ent, ok := c.values[key]
	if !ok {
		return nil, nil
	}
	now := currentTick()
	if ent.atime != infTick {
		ent.atime = now
	}
	c.order.MoveToBack(ent.element)
	var lease *lruEntry
	if ent.lru != nil {
		lease = ent.lru
		c.budget.touch(lease, now)
		c.budget.acquireLease(lease)
		c.budget.recordHit()
	}
	return ent.image, lease
}

// release ends a lease obtained from [imageCache.acquire]. A nil lease is
// a no-op so standalone caches need no special-casing at call sites.
func (c *imageCache[K]) release(lease *lruEntry) {
	if lease == nil {
		return
	}
	c.budget.releaseLease(lease)
}

// softLimitSweepLocked drops entries older than cacheLifetime once the
// count exceeds the soft limit. The caller holds c.mu.
func (c *imageCache[K]) softLimitSweepLocked(now int64) {
	if len(c.values) <= c.softLimit {
		return
	}
	var next *list.Element
	for el := c.order.Front(); el != nil; el = next {
		next = el.Next()
		ent := el.Value.(*cacheEntry)
		if ent.atime >= now-cacheLifetime {
			continue
		}
		c.deleteEntryLocked(ent)
	}
}

// deleteEntryLocked removes an entry from the cache's own structures. When
// a budget is attached the budget LRU bookkeeping is updated too. The
// *ebiten.Image itself is left for Ebitengine's runtime cleanup so a draw
// already queued against it stays valid. The caller holds c.mu.
func (c *imageCache[K]) deleteEntryLocked(ent *cacheEntry) {
	delete(c.values, ent.key.(K))
	c.order.Remove(ent.element)
	if ent.lru != nil && c.budget != nil {
		c.budget.mu.Lock()
		c.budget.removeEntryLocked(ent.lru, true)
		c.budget.mu.Unlock()
	}
	ent.lru = nil
}

// applyVictimsLocked removes entries chosen by the budget. The budget lock
// is no longer held. Victims whose entry is currently leased cannot appear
// here because the budget skips leased entries.
func (c *imageCache[K]) applyVictimsLocked(victims []*cacheEntry) {
	for _, ent := range victims {
		if _, ok := c.values[ent.key.(K)]; !ok {
			continue
		}
		delete(c.values, ent.key.(K))
		c.order.Remove(ent.element)
		ent.lru = nil
	}
}

// clear removes every entry that is not currently leased. Caches without a
// budget drop everything.
func (c *imageCache[K]) clear() {
	c.mu.Lock()
	var next *list.Element
	for el := c.order.Front(); el != nil; el = next {
		next = el.Next()
		ent := el.Value.(*cacheEntry)
		if ent.lru != nil && ent.lru.leases > 0 {
			continue
		}
		c.deleteEntryLocked(ent)
	}
	c.mu.Unlock()
}

// len reports the number of retained entries.
func (c *imageCache[K]) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.values)
}

func imageByteCost(img *ebiten.Image) int64 {
	b := img.Bounds()
	return int64(4) * int64(b.Dx()) * int64(b.Dy())
}

func currentTick() int64 {
	return ebiten.Tick()
}
