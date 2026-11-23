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

package mipmap

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"
	"sync"
)

// Errors returned by prefetch plan computation and execution. Errors returned by
// NewPrefetchPlan and PrefetchExecutor.Execute wrap one of these sentinels (or a
// generator error) together with the contextual values, so errors.Is works with them.
var (
	// ErrPrefetchEmptyViewports indicates that a prefetch request had no viewports at all.
	ErrPrefetchEmptyViewports = errors.New("mipmap: prefetch: at least one viewport is required")

	// ErrPrefetchInvalidBounds indicates that the source bounds of a prefetch request were empty.
	ErrPrefetchInvalidBounds = errors.New("mipmap: prefetch: source bounds must be non-empty")

	// ErrPrefetchInvalidSize indicates that the mipmap image has a non-positive size.
	ErrPrefetchInvalidSize = errors.New("mipmap: prefetch: image size must be positive")

	// ErrPrefetchTooLarge indicates that the estimated memory for the prefetched levels
	// exceeds the plan's limit.
	ErrPrefetchTooLarge = errors.New("mipmap: prefetch: estimated memory exceeds the limit")

	// ErrPrefetchStalePlan indicates that the mipmap was deallocated (e.g. after a device
	// rebuild) between plan creation and execution.
	ErrPrefetchStalePlan = errors.New("mipmap: prefetch: plan is stale because the image was deallocated")

	// ErrPrefetchCanceled indicates that the plan or its execution context was canceled.
	ErrPrefetchCanceled = errors.New("mipmap: prefetch: plan was canceled")

	// ErrPrefetchGeneration indicates that generating a mipmap level failed.
	ErrPrefetchGeneration = errors.New("mipmap: prefetch: mipmap level generation failed")
)

// defaultPrefetchMaxTotalBytes is the default upper bound of the estimated memory for the
// levels a single prefetch plan generates.
const defaultPrefetchMaxTotalBytes = 512 << 20

// PrefetchFilter represents the sampling filter that will be used when the image is drawn.
// Mipmaps are only used with the linear filter, so other filters produce empty plans.
type PrefetchFilter int

const (
	// PrefetchFilterNearest represents nearest-neighbor sampling. No mipmap is used.
	PrefetchFilterNearest PrefetchFilter = iota

	// PrefetchFilterLinear represents linear sampling. Mipmaps are used when minifying.
	PrefetchFilterLinear
)

// PrefetchMatrix is an affine transform from source (image) coordinates to destination
// (viewport) coordinates:
//
//	x' = A*x + B*y + Tx
//	y' = C*x + D*y + Ty
type PrefetchMatrix struct {
	A, B, C, D, Tx, Ty float64
}

// PrefetchScale returns a PrefetchMatrix that scales by (sx, sy).
func PrefetchScale(sx, sy float64) PrefetchMatrix {
	return PrefetchMatrix{A: sx, D: sy}
}

// det returns the determinant of the linear part.
func (m PrefetchMatrix) det() float64 {
	return m.A*m.D - m.B*m.C
}

// apply transforms a point from source to destination coordinates.
func (m PrefetchMatrix) apply(x, y float64) (float64, float64) {
	return m.A*x + m.B*y + m.Tx, m.C*x + m.D*y + m.Ty
}

// invert transforms a point from destination to source coordinates.
// The caller must ensure that the determinant is non-zero.
func (m PrefetchMatrix) invert(x, y float64) (float64, float64) {
	det := m.det()
	x -= m.Tx
	y -= m.Ty
	return (m.D*x - m.B*y) / det, (-m.C*x + m.A*y) / det
}

// PrefetchRequest describes one predicted draw for which mipmap levels are prefetched.
type PrefetchRequest struct {
	// Bounds is the source region in the original image's coordinates.
	Bounds image.Rectangle

	// Matrix transforms source coordinates to destination coordinates.
	Matrix PrefetchMatrix

	// Filter is the sampling filter used when drawing.
	Filter PrefetchFilter

	// Viewports are the visible regions in destination coordinates. The prefetched region
	// is the part of the drawn image that intersects at least one viewport.
	Viewports []image.Rectangle

	// Priority orders jobs across plans executed together. Higher values run first.
	Priority int

	// MaxTotalBytes limits the total estimated memory of the generated levels.
	// A non-positive value uses a default limit.
	MaxTotalBytes int64

	// ReclaimOnCancel makes the executor deallocate the levels this plan generated when
	// the plan is canceled before completion.
	ReclaimOnCancel bool
}

// PrefetchJob is one unit of prefetch work: a mipmap level and the region of that level,
// in the level image's coordinates, that the predicted draw will sample.
type PrefetchJob struct {
	Level    int
	Region   image.Rectangle
	Priority int
}

// PrefetchPlanState is the state of a PrefetchPlan.
type PrefetchPlanState int

const (
	// PrefetchPlanPending indicates that the plan was created but not executed yet.
	PrefetchPlanPending PrefetchPlanState = iota

	// PrefetchPlanRunning indicates that an executor is executing the plan.
	PrefetchPlanRunning

	// PrefetchPlanCompleted indicates that all the plan's jobs finished. Executing a
	// completed plan again is a safe no-op.
	PrefetchPlanCompleted

	// PrefetchPlanFailed indicates that execution failed. See PrefetchPlan.Err.
	PrefetchPlanFailed

	// PrefetchPlanCanceled indicates that the plan was canceled before completion.
	PrefetchPlanCanceled
)

// PrefetchPlan is a precomputed set of mipmap levels and regions to generate ahead of
// drawing, so that the levels are ready when a rapidly zooming draw needs them.
//
// A plan is created by NewPrefetchPlan and executed by PrefetchExecutor. Plan computation
// is deterministic: the same request on the same image always yields the same jobs.
type PrefetchPlan struct {
	m               *Mipmap
	jobs            []PrefetchJob
	epoch           uint64
	reclaimOnCancel bool

	mu        sync.Mutex
	state     PrefetchPlanState
	done      chan struct{} // non-nil while running; closed when the plan leaves the running state
	generated []int         // levels generated by this plan, for reclamation
	err       error
	canceled  bool
}

// NewPrefetchPlan computes the mipmap levels and regions required to draw the source
// bounds of m with the given transform, filter, and viewports, and returns a plan to
// generate them ahead of time.
//
// The computed level matches the level DrawTriangles would select for the same transform,
// so executing the plan never changes what drawing samples. Requests that need no mipmap
// (nearest filter, magnification, degenerate or invisible transforms, levels exceeding
// maxLevelImageSize) return a valid plan with no jobs instead of an error.
func (m *Mipmap) NewPrefetchPlan(req *PrefetchRequest) (*PrefetchPlan, error) {
	if req == nil {
		return nil, fmt.Errorf("%w: request must not be nil", ErrPrefetchInvalidBounds)
	}
	if m.width <= 0 || m.height <= 0 {
		return nil, fmt.Errorf("%w: got %dx%d", ErrPrefetchInvalidSize, m.width, m.height)
	}
	if req.Bounds.Empty() {
		return nil, fmt.Errorf("%w: got %v", ErrPrefetchInvalidBounds, req.Bounds)
	}
	if len(req.Viewports) == 0 {
		return nil, ErrPrefetchEmptyViewports
	}

	plan := &PrefetchPlan{
		m:               m,
		epoch:           m.epoch,
		reclaimOnCancel: req.ReclaimOnCancel,
		state:           PrefetchPlanPending,
	}

	// Mipmaps are only used with the linear filter on mipmapped image types.
	if req.Filter != PrefetchFilterLinear || !canUseMipmap(m.imageType) {
		return plan, nil
	}

	det := req.Matrix.det()
	if det == 0 || math.IsNaN(det) {
		// A degenerate transform draws nothing.
		return plan, nil
	}

	level := prefetchLevel(req.Bounds, req.Matrix)
	if level == 0 {
		return plan, nil
	}

	srcRegion := prefetchSourceRegion(req.Bounds, req.Matrix, req.Viewports)
	if srcRegion.Empty() {
		return plan, nil
	}

	// Mirror (*Mipmap).level: a level image is generated only when every lower level fits
	// within maxLevelImageSize. Level sizes are non-increasing, so if level 1 exceeds the
	// limit, no level can be generated (#1399). This is a safe skip, not an error.
	if sizeForLevel(m.width, 1) > maxLevelImageSize || sizeForLevel(m.height, 1) > maxLevelImageSize {
		return plan, nil
	}

	var jobs []PrefetchJob
	var estBytes int64
	for l := 1; l <= level; l++ {
		w := sizeForLevel(m.width, l)
		h := sizeForLevel(m.height, l)
		if w == 0 || h == 0 {
			break
		}
		r := m.regionForLevel(srcRegion, l)
		if r.Empty() {
			continue
		}
		// Generation always produces the whole level image, so estimate the full level
		// size even though only a region of it might be sampled.
		estBytes += int64(w) * int64(h) * 4
		jobs = append(jobs, PrefetchJob{
			Level:    l,
			Region:   r,
			Priority: req.Priority,
		})
	}

	limit := req.MaxTotalBytes
	if limit <= 0 {
		limit = defaultPrefetchMaxTotalBytes
	}
	if estBytes > limit {
		return nil, fmt.Errorf("%w: estimated %d bytes for %d level(s) of a %dx%d image, limit is %d bytes",
			ErrPrefetchTooLarge, estBytes, len(jobs), m.width, m.height, limit)
	}

	plan.jobs = jobs
	return plan, nil
}

// prefetchLevel returns the mipmap level that DrawTriangles would select for a quad of the
// given source bounds transformed by mat: the minimum (most detailed) level over the quad's
// two axis edges and its diagonal, which are the edges of the two triangles forming the quad.
func prefetchLevel(bounds image.Rectangle, mat PrefetchMatrix) int {
	w := float32(bounds.Dx())
	h := float32(bounds.Dy())

	// Transformed edge vectors. Rotation is handled naturally: only the lengths matter.
	exx := float32(mat.A) * w
	exy := float32(mat.C) * w
	eyx := float32(mat.B) * h
	eyy := float32(mat.D) * h

	level := math.MaxInt32
	for _, e := range [][4]float32{
		{exx, exy, w, 0},             // X axis edge
		{eyx, eyy, 0, h},             // Y axis edge
		{exx + eyx, exy + eyy, w, h}, // diagonal
	} {
		l := mipmapLevelFromDistance(0, 0, e[0], e[1], 0, 0, e[2], e[3])
		level = min(level, l)
	}
	return level
}

// prefetchSourceRegion returns the region of the source image, in the original image's
// coordinates, that is visible through at least one viewport when drawn with mat.
// The result is deterministic for any affine transform, including rotations: the visible
// destination region is mapped back by the inverse transform and its bounding box is taken.
func prefetchSourceRegion(bounds image.Rectangle, mat PrefetchMatrix, viewports []image.Rectangle) image.Rectangle {
	// Forward-transform the bounds corners to destination coordinates and take their
	// bounding box. Outward rounding keeps every potentially visible pixel.
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, c := range [][2]float64{
		{float64(bounds.Min.X), float64(bounds.Min.Y)},
		{float64(bounds.Max.X), float64(bounds.Min.Y)},
		{float64(bounds.Min.X), float64(bounds.Max.Y)},
		{float64(bounds.Max.X), float64(bounds.Max.Y)},
	} {
		dx, dy := mat.apply(c[0], c[1])
		if math.IsNaN(dx) || math.IsNaN(dy) || math.IsInf(dx, 0) || math.IsInf(dy, 0) {
			return image.Rectangle{}
		}
		minX = math.Min(minX, dx)
		minY = math.Min(minY, dy)
		maxX = math.Max(maxX, dx)
		maxY = math.Max(maxY, dy)
	}
	dst := image.Rect(int(math.Floor(minX)), int(math.Floor(minY)), int(math.Ceil(maxX)), int(math.Ceil(maxY)))

	var vis image.Rectangle
	for _, v := range viewports {
		if v.Empty() {
			continue
		}
		vis = vis.Union(v.Intersect(dst))
	}
	if vis.Empty() {
		return image.Rectangle{}
	}

	// Map the visible destination region back to source coordinates.
	sMinX, sMinY := math.Inf(1), math.Inf(1)
	sMaxX, sMaxY := math.Inf(-1), math.Inf(-1)
	for _, c := range [][2]float64{
		{float64(vis.Min.X), float64(vis.Min.Y)},
		{float64(vis.Max.X), float64(vis.Min.Y)},
		{float64(vis.Min.X), float64(vis.Max.Y)},
		{float64(vis.Max.X), float64(vis.Max.Y)},
	} {
		sx, sy := mat.invert(c[0], c[1])
		sMinX = math.Min(sMinX, sx)
		sMinY = math.Min(sMinY, sy)
		sMaxX = math.Max(sMaxX, sx)
		sMaxY = math.Max(sMaxY, sy)
	}
	r := image.Rect(int(math.Floor(sMinX)), int(math.Floor(sMinY)), int(math.Ceil(sMaxX)), int(math.Ceil(sMaxY)))
	return r.Intersect(bounds)
}

// Jobs returns a copy of the plan's jobs in computation order (ascending level).
func (p *PrefetchPlan) Jobs() []PrefetchJob {
	return slices.Clone(p.jobs)
}

// Mipmap returns the mipmap this plan belongs to.
func (p *PrefetchPlan) Mipmap() *Mipmap {
	return p.m
}

// State returns the plan's current state.
func (p *PrefetchPlan) State() PrefetchPlanState {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.state
}

// Err returns the error that made the plan fail or be canceled, if any.
func (p *PrefetchPlan) Err() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err
}

// Cancel cancels the plan. Pending jobs are skipped by the currently running or next
// execution. If the plan was created with ReclaimOnCancel and is canceled during
// execution, the levels the plan generated are deallocated. Canceling a completed plan is
// a safe no-op.
func (p *PrefetchPlan) Cancel() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.state == PrefetchPlanCompleted || p.state == PrefetchPlanFailed {
		return
	}
	p.canceled = true
}

// begin transitions the plan to the running state. It reports whether the caller became
// the plan's executor. A completed plan is a safe no-op (duplicate completion); a plan
// that is already running makes the caller wait for the in-flight execution instead.
func (p *PrefetchPlan) begin() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	switch p.state {
	case PrefetchPlanCompleted:
		return false
	case PrefetchPlanRunning:
		done := p.done
		p.mu.Unlock()
		<-done
		p.mu.Lock()
		return false
	}
	p.state = PrefetchPlanRunning
	p.canceled = false
	p.err = nil
	p.done = make(chan struct{})
	return true
}

// finish records the final state and error and wakes up waiters.
func (p *PrefetchPlan) finish(state PrefetchPlanState, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.state = state
	if p.err == nil {
		p.err = err
	}
	if p.done != nil {
		close(p.done)
		p.done = nil
	}
}

// fail records a job or validation error without changing the state yet.
func (p *PrefetchPlan) recordErr(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.err == nil {
		p.err = err
	}
}

// isCanceled reports whether Cancel was called on the plan.
func (p *PrefetchPlan) isCanceled() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.canceled
}

// noteGenerated records that the plan generated the given level.
func (p *PrefetchPlan) noteGenerated(level int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.generated = append(p.generated, level)
}

// reclaimGenerated deallocates the levels the plan generated and clears the record.
func (p *PrefetchPlan) reclaimGenerated() {
	p.mu.Lock()
	levels := p.generated
	p.generated = nil
	reclaim := p.reclaimOnCancel
	p.mu.Unlock()
	if !reclaim {
		return
	}
	prefetchGenMu.Lock()
	defer prefetchGenMu.Unlock()
	for _, level := range levels {
		p.m.deallocateLevelImage(level)
	}
}

// HasCleanLevel reports whether the given mipmap level exists and is up to date.
func (m *Mipmap) HasCleanLevel(level int) bool {
	img, ok := m.imgs[level]
	return ok && img.img != nil && !img.dirty
}

// PrefetchLevel generates the given mipmap level synchronously, reusing the existing
// generation flow. Levels that already exist and are clean are skipped. It returns an
// error wrapping ErrPrefetchGeneration when the level cannot be generated (e.g. its size
// exceeds maxLevelImageSize).
func (m *Mipmap) PrefetchLevel(level int) error {
	if m.HasCleanLevel(level) {
		return nil
	}
	if m.level(level) == nil {
		return fmt.Errorf("%w: level %d of a %dx%d image is not available", ErrPrefetchGeneration, level, m.width, m.height)
	}
	return nil
}

// deallocateLevelImage deallocates a single level image. The level is regenerated on
// demand by the regular generation flow.
func (m *Mipmap) deallocateLevelImage(level int) {
	img, ok := m.imgs[level]
	if !ok || img.img == nil {
		return
	}
	img.img.Deallocate()
	delete(m.imgs, level)
}

// prefetchGenMu serializes the default level generation and reclamation across executors,
// as the atlas package is not safe for concurrent use. Custom generators set via
// PrefetchExecutor.Generate are not serialized and must be safe for bounded concurrent
// use instead.
var prefetchGenMu sync.Mutex

// PrefetchExecutor executes prefetch plans with cancellation, priority ordering, and
// bounded concurrency.
//
// Jobs of all the plans passed to one Execute call are dispatched in a deterministic
// order: higher priority first, then lower level, then plan and job order. At most
// MaxConcurrent jobs run at once. The default generator is serialized internally (see
// prefetchGenMu), so concurrency bounds apply to custom generators; the default path is
// safe to call from the game's goroutine.
type PrefetchExecutor struct {
	// MaxConcurrent is the maximum number of jobs generated concurrently.
	// A non-positive value means 1.
	MaxConcurrent int

	// Generate generates one job. When nil, the level is generated synchronously via
	// the mipmap's regular generation flow. A custom generator must be safe for up to
	// MaxConcurrent concurrent invocations.
	Generate func(plan *PrefetchPlan, job PrefetchJob) error
}

// Execute runs the given plans' jobs and returns the first error by dispatch order, if
// any. A nil context is treated as context.Background().
//
// Plans that are already completed are skipped (duplicate completion is a safe no-op).
// Stale plans (the mipmap was deallocated after the plan was created) fail with an error
// wrapping ErrPrefetchStalePlan. Canceling ctx cancels all the plans being executed and
// returns an error wrapping ErrPrefetchCanceled.
func (e *PrefetchExecutor) Execute(ctx context.Context, plans ...*PrefetchPlan) error {
	if ctx == nil {
		ctx = context.Background()
	}
	maxConcurrent := e.MaxConcurrent
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}

	type workItem struct {
		plan   *PrefetchPlan
		jobIdx int
		seq    int
	}

	var items []workItem
	var running []*PrefetchPlan
	var errMu sync.Mutex
	var firstErr error
	errSeq := math.MaxInt32
	recordErr := func(err error, seq int) {
		if err == nil {
			return
		}
		errMu.Lock()
		defer errMu.Unlock()
		if seq >= errSeq {
			return
		}
		firstErr = err
		errSeq = seq
	}
	hasErr := func() bool {
		errMu.Lock()
		defer errMu.Unlock()
		return firstErr != nil
	}

	seq := 0
	seen := map[*PrefetchPlan]bool{}
	for _, p := range plans {
		if p == nil || seen[p] {
			// Duplicate plans in one call are duplicate requests: skip them.
			continue
		}
		seen[p] = true
		if p.epoch != p.m.epoch {
			err := fmt.Errorf("%w: plan created at epoch %d, current epoch is %d", ErrPrefetchStalePlan, p.epoch, p.m.epoch)
			p.recordErr(err)
			p.finish(PrefetchPlanFailed, err)
			recordErr(err, seq)
			seq++
			continue
		}
		if !p.begin() {
			// Already completed, or waited for an in-flight execution.
			continue
		}
		running = append(running, p)
		for i := range p.jobs {
			items = append(items, workItem{plan: p, jobIdx: i, seq: seq})
			seq++
		}
	}

	slices.SortStableFunc(items, func(a, b workItem) int {
		ja, jb := a.plan.jobs[a.jobIdx], b.plan.jobs[b.jobIdx]
		if c := jb.Priority - ja.Priority; c != 0 {
			return c
		}
		if c := ja.Level - jb.Level; c != 0 {
			return c
		}
		return a.seq - b.seq
	})

	sem := make(chan struct{}, maxConcurrent)
	var wg sync.WaitGroup

dispatch:
	for _, it := range items {
		if err := ctx.Err(); err != nil {
			recordErr(fmt.Errorf("%w: %v", ErrPrefetchCanceled, err), it.seq)
			break
		}
		if hasErr() {
			break
		}
		p := it.plan
		if p.isCanceled() {
			continue
		}
		if p.epoch != p.m.epoch {
			err := fmt.Errorf("%w: plan created at epoch %d, current epoch is %d", ErrPrefetchStalePlan, p.epoch, p.m.epoch)
			p.recordErr(err)
			recordErr(err, it.seq)
			continue
		}
		job := p.jobs[it.jobIdx]

		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			recordErr(fmt.Errorf("%w: %v", ErrPrefetchCanceled, ctx.Err()), it.seq)
			break dispatch
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			// Skip levels that already exist and are clean: partially generated chains
			// and duplicate requests complete without duplicate work.
			if e.Generate == nil {
				prefetchGenMu.Lock()
				defer prefetchGenMu.Unlock()
				if p.m.HasCleanLevel(job.Level) {
					return
				}
				if err := p.m.PrefetchLevel(job.Level); err != nil {
					p.recordErr(err)
					recordErr(err, it.seq)
					return
				}
			} else {
				prefetchGenMu.Lock()
				clean := p.m.HasCleanLevel(job.Level)
				prefetchGenMu.Unlock()
				if clean {
					return
				}
				if err := e.Generate(p, job); err != nil {
					err = fmt.Errorf("mipmap: prefetch: generating level %d: %w", job.Level, err)
					p.recordErr(err)
					recordErr(err, it.seq)
					return
				}
			}
			p.noteGenerated(job.Level)
		}()
	}
	wg.Wait()

	ctxCanceled := ctx.Err() != nil
	for _, p := range running {
		p.mu.Lock()
		canceled := p.canceled || ctxCanceled
		failed := p.err != nil && !errors.Is(p.err, ErrPrefetchCanceled)
		p.mu.Unlock()
		switch {
		case canceled:
			p.recordErr(ErrPrefetchCanceled)
			p.reclaimGenerated()
			p.finish(PrefetchPlanCanceled, p.Err())
		case failed:
			p.finish(PrefetchPlanFailed, p.Err())
		default:
			p.finish(PrefetchPlanCompleted, nil)
		}
	}

	errMu.Lock()
	defer errMu.Unlock()
	return firstErr
}
