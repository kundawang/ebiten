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
	"fmt"
	"slices"
	"sync"

	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
)

// TextGlyphWarmupPlan is a deduplicated plan to pre-rasterize the glyphs of a
// text before the text is first drawn, so that the first [Draw] of a dialog
// or a document doesn't pay the glyph rasterization cost on its first frame.
//
// A plan is built from a text and a [Face]. The face's configuration at
// construction time — its source, size, direction, and variations — defines
// the cache identity of the warmed glyphs: executing the plan populates
// exactly the cache entries that a later [Draw] of the same text with an
// equally configured face looks up. The plan reuses the same caches as Draw
// (the shape-output cache, the render-data cache, and the per-size glyph
// image cache), so warmed entries are shared, not duplicated.
//
// # Key semantics
//
//   - Deduplication: the plan contains one task per unique rune in the text
//     (line breaks excluded). Repeated text collapses to a single task, and
//     submitting two plans for the same text and face is harmless — the
//     shared cache turns the second execution into cache hits.
//   - Missing glyphs: a rune the face cannot render warms the same fallback
//     (.notdef) entry Draw would create, or no entry at all. Either way Draw
//     behaves identically whether or not the plan ran.
//   - Color glyphs (SVG, bitmap, COLR) and [MultiFace] fallback go through
//     the same realization path as Draw, so their cache keys match exactly.
//   - Context-dependent forms (ligatures, contextual alternates) are shaped
//     per rune during warmup. Their shaped-in-context glyph images are not
//     warmed and are rasterized on demand by Draw, as before.
//   - Device rebuild: cache keys are device-independent and Ebitengine
//     restores images after a GPU context loss, so warmed entries stay valid
//     across a device rebuild.
//   - An unfinished plan never affects immediate drawing: runes that were
//     not warmed are simply rasterized by Draw on demand.
//
// # Execution
//
// A plan can be executed synchronously in chunks with [TextGlyphWarmupPlan.Run]
// (e.g. a few glyphs per frame), or asynchronously on a background worker
// with [TextGlyphWarmupPlan.Start]. Both can be mixed; each task is claimed
// exactly once. Execution is concurrent-safe with Draw and with other plans.
//
// A plan must not be used with a face that is mutated while the plan is
// being executed, just like Draw.
type TextGlyphWarmupPlan struct {
	face       Face
	variations int
	runes      []rune

	// total is the number of tasks at construction. It never changes, so
	// progress reporting stays stable even after the plan releases runes.
	total int

	mu        sync.Mutex
	next      int // index of the next unclaimed task
	completed int // tasks whose warmup has finished
	started   bool
	canceled  bool
	err       error

	doneCh        chan struct{}
	terminateOnce sync.Once
}

// NewTextGlyphWarmupPlan builds a warmup plan for text drawn with face.
//
// Tasks are ordered by priority: runes that occur more frequently in text
// come first, with ties broken by first occurrence. Warming the most frequent
// glyphs first maximizes the cache hit rate when the plan is executed only
// partially.
//
// An empty text or a nil face yields an already-terminated empty plan. A face
// that cannot even report its metrics yields an empty plan with a non-nil
// [TextGlyphWarmupPlan.Err].
func NewTextGlyphWarmupPlan(text string, face Face) (p *TextGlyphWarmupPlan) {
	p = &TextGlyphWarmupPlan{
		face:   face,
		doneCh: make(chan struct{}),
	}
	defer func() {
		// A face whose metrics cannot be read is unusable. Turn the plan
		// into a stable, already-terminated empty plan with an error
		// instead of panicking.
		if v := recover(); v != nil {
			p.err = fmt.Errorf("text: cannot build a glyph warmup plan: %v", v)
			p.runes = nil
			p.total = 0
			p.terminate()
		}
	}()

	if face == nil || text == "" {
		p.terminate()
		return p
	}

	p.variations = glyphVariationCount(face)
	p.runes = dedupRunesByPriority(text)
	p.total = len(p.runes)
	if p.total == 0 {
		p.terminate()
	}
	return p
}

// dedupRunesByPriority returns the unique non-line-break runes of text,
// ordered by descending frequency and then by first occurrence.
func dedupRunesByPriority(text string) []rune {
	type runeStat struct {
		freq  int
		first int
	}
	stats := map[rune]runeStat{}
	var order []rune
	for _, r := range text {
		if textutil.IsLineBreak(r) {
			continue
		}
		if s, ok := stats[r]; ok {
			s.freq++
			stats[r] = s
			continue
		}
		stats[r] = runeStat{freq: 1, first: len(order)}
		order = append(order, r)
	}
	slices.SortStableFunc(order, func(a, b rune) int {
		sa, sb := stats[a], stats[b]
		if sa.freq != sb.freq {
			return sb.freq - sa.freq
		}
		return sa.first - sb.first
	})
	return order
}

// Len returns the total number of deduplicated glyph tasks in the plan.
func (p *TextGlyphWarmupPlan) Len() int {
	return p.total
}

// Progress returns the number of completed tasks and the total number of
// tasks. Progress is monotonic: the done count never decreases.
func (p *TextGlyphWarmupPlan) Progress() (done, total int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.completed, p.total
}

// Run executes up to n pending tasks synchronously and returns how many tasks
// were executed. n <= 0 is a stable no-op and returns 0.
//
// Run is useful to spread warmup over multiple frames. Run can be called
// concurrently with other Run calls and with a worker started by
// [TextGlyphWarmupPlan.Start]; each task is executed exactly once.
func (p *TextGlyphWarmupPlan) Run(n int) int {
	if n <= 0 {
		return 0
	}
	return p.runTasks(n)
}

// Start launches a background worker that executes all the pending tasks and
// returns immediately. Use [TextGlyphWarmupPlan.Wait],
// [TextGlyphWarmupPlan.Progress], or [TextGlyphWarmupPlan.Cancel] to observe
// or stop the worker.
//
// Calling Start more than once, after a cancellation, or on a terminated plan
// is a no-op, so duplicate submissions never spawn extra workers.
func (p *TextGlyphWarmupPlan) Start() {
	p.mu.Lock()
	if p.started || p.canceled || p.err != nil || len(p.runes) == 0 {
		p.mu.Unlock()
		return
	}
	p.started = true
	p.mu.Unlock()

	go func() {
		p.runTasks(-1)
		p.terminate()
	}()
}

// Cancel stops the plan. Pending tasks are dropped; a task already in flight
// runs to completion. Cancel is idempotent and concurrent-safe. A canceled
// plan terminates: [TextGlyphWarmupPlan.Wait] unblocks and reports false.
func (p *TextGlyphWarmupPlan) Cancel() {
	p.mu.Lock()
	if p.canceled {
		p.mu.Unlock()
		return
	}
	p.canceled = true
	started := p.started
	p.mu.Unlock()

	// If no worker is running, nobody else will terminate the plan.
	if !started {
		p.terminate()
	}
}

// Canceled reports whether [TextGlyphWarmupPlan.Cancel] was called.
func (p *TextGlyphWarmupPlan) Canceled() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.canceled
}

// Wait blocks until the plan terminates — by completing, by cancellation, or
// by a worker error — and reports whether every task completed successfully.
//
// A plan that is never executed (via [TextGlyphWarmupPlan.Run] or
// [TextGlyphWarmupPlan.Start]) nor canceled never terminates, so Wait blocks
// indefinitely for such a plan.
func (p *TextGlyphWarmupPlan) Wait() bool {
	<-p.doneCh
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err == nil && !p.canceled && p.completed == p.total
}

// Err returns the first error that terminated the plan, if any. A plan
// terminates with an error when the face fails during warmup (e.g. the face
// became invalid); pending tasks are then dropped and the plan terminates.
func (p *TextGlyphWarmupPlan) Err() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err
}

// runTasks claims and executes up to n tasks (all remaining tasks if n < 0)
// and returns how many tasks were executed.
func (p *TextGlyphWarmupPlan) runTasks(n int) int {
	var executed int
	for n < 0 || executed < n {
		p.mu.Lock()
		if p.canceled || p.err != nil || p.next >= len(p.runes) {
			p.mu.Unlock()
			return executed
		}
		r := p.runes[p.next]
		// Claim the task before executing it so that concurrent executors
		// never run the same task twice.
		p.next++
		face := p.face
		variations := p.variations
		p.mu.Unlock()

		if !p.warm(face, r, variations) {
			p.terminate()
			return executed
		}
		executed++

		p.mu.Lock()
		p.completed++
		all := p.completed == len(p.runes)
		p.mu.Unlock()
		if all {
			p.terminate()
			return executed
		}
	}
	return executed
}

// warm executes one task and reports whether execution can continue. A panic
// from a broken face is recovered, recorded as the plan's error, and stops
// the plan instead of crashing the caller or leaking a worker goroutine.
func (p *TextGlyphWarmupPlan) warm(face Face, r rune, variations int) (ok bool) {
	defer func() {
		if v := recover(); v != nil {
			p.mu.Lock()
			if p.err == nil {
				p.err = fmt.Errorf("text: glyph warmup failed for rune U+%04X: %v", r, v)
			}
			p.mu.Unlock()
			ok = false
		}
	}()
	warmFaceTextVariations(face, string(r), variations)
	return true
}

// terminate releases the plan's references and unblocks Wait. It is
// idempotent.
func (p *TextGlyphWarmupPlan) terminate() {
	p.terminateOnce.Do(func() {
		p.mu.Lock()
		p.face = nil
		p.runes = nil
		p.mu.Unlock()
		close(p.doneCh)
	})
}

// warmFaceTextVariations rasterizes every glyph of s with face at every
// subpixel variation, populating the shared glyph caches. This is the same
// work [CacheGlyphs] does, parameterized by the variation count so a
// [TextGlyphWarmupPlan] can compute it once per plan.
func warmFaceTextVariations(face Face, s string, variations int) {
	var x, y float64
	var buf []LazyGlyph
	// Create all the possible variations (#2528).
	for range variations {
		buf = appendLazyGlyphs(buf[:0], s, face, x, y, nil, nil)
		// Realize each glyph to populate the underlying image cache.
		for _, g := range buf {
			g.Image()
		}

		if face.direction().isHorizontal() {
			x += 1.0 / float64(variations)
		} else {
			y += 1.0 / float64(variations)
		}
	}
}
