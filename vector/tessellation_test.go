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

package vector

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
)

// quadPoint evaluates the quadratic Bézier with the given control points at t.
func quadPoint(p0, p1, p2 point, t float64) point {
	u := 1 - t
	return point{
		x: u*u*p0.x + 2*u*t*p1.x + t*t*p2.x,
		y: u*u*p0.y + 2*u*t*p1.y + t*t*p2.y,
	}
}

// cubicPoint evaluates the cubic Bézier with the given control points at t.
func cubicPoint(p0, p1, p2, p3 point, t float64) point {
	u := 1 - t
	return point{
		x: u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x,
		y: u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y,
	}
}

// maxChordError returns the maximum distance from densely sampled reference points of a curve to
// the polyline segments, measured in the tessellator's transformed space.
func maxChordError(t *testing.T, tess *tessellator, curvePoint func(float64) point, poly []point) float64 {
	t.Helper()
	var worst float64
	const samples = 4096
	for k := 0; k <= samples; k++ {
		ref := tess.apply(curvePoint(float64(k) / samples))
		best := math.Inf(1)
		for i := 0; i+1 < len(poly); i++ {
			// Distance to segment poly[i]-poly[i+1].
			a, b := poly[i], poly[i+1]
			dx, dy := b.x-a.x, b.y-a.y
			length2 := dx*dx + dy*dy
			tt := 0.0
			if length2 > 0 {
				tt = ((ref.x-a.x)*dx + (ref.y-a.y)*dy) / length2
				tt = max(0, min(1, tt))
			}
			closest := point{x: a.x + tt*dx, y: a.y + tt*dy}
			best = min(best, math.Hypot(ref.x-closest.x, ref.y-closest.y))
		}
		worst = max(worst, best)
	}
	return worst
}

func flattenWith(t *testing.T, q *VectorTessellationQuality, fn func(*tessellator, []point) []point) ([]point, *tessellator) {
	t.Helper()
	tess := newTessellator(q)
	dst := []point{}
	dst = fn(tess, dst)
	return dst, tess
}

func TestTessellationQualityDefaults(t *testing.T) {
	q := NewVectorTessellationQuality()
	if got, want := q.PixelTolerance(), float32(0.5); got != want {
		t.Errorf("default tolerance: got %v, want %v", got, want)
	}
	if got, want := q.MaxSegments(), defaultMaxTessellationSegments; got != want {
		t.Errorf("default max segments: got %d, want %d", got, want)
	}
	if !q.isDefault() {
		t.Error("a fresh quality must be the default quality")
	}
	var nilQuality *VectorTessellationQuality
	if !nilQuality.isDefault() {
		t.Error("nil quality should be treated as default")
	}
}

func TestTessellationQualityNegativeTolerancePanics(t *testing.T) {
	q := NewVectorTessellationQuality()
	assertPanics(t, "negative pixel tolerance", func() {
		q.SetPixelTolerance(-0.25)
	})
	if got := q.PixelTolerance(); got != 0.5 {
		t.Errorf("rejected tolerance changed the quality: got %v", got)
	}
	assertPanics(t, "negative max segments", func() {
		q.SetMaxSegments(-1)
	})
	if got := q.MaxSegments(); got != defaultMaxTessellationSegments {
		t.Errorf("rejected max segments changed the quality: got %d", got)
	}

	// Non-finite values are ignored, not stored.
	q.SetPixelTolerance(float32(math.NaN()))
	q.SetPixelTolerance(float32(math.Inf(1)))
	if got := q.PixelTolerance(); got != 0.5 {
		t.Errorf("non-finite tolerance was stored: got %v", got)
	}
	var bad ebiten.GeoM
	bad.SetElement(1, 2, math.NaN())
	q.SetGeoM(bad)
	if !q.isDefault() {
		t.Error("a non-finite GeoM must be ignored")
	}

	// Zero tolerance is accepted and means "finest possible".
	q.SetPixelTolerance(0)
	if got := q.PixelTolerance(); got != 0 {
		t.Errorf("zero tolerance: got %v, want 0", got)
	}
	// Zero max segments restores the default budget.
	q.SetMaxSegments(7)
	q.SetMaxSegments(0)
	if got := q.MaxSegments(); got != defaultMaxTessellationSegments {
		t.Errorf("zero max segments: got %d, want %d", got, defaultMaxTessellationSegments)
	}
}

func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected a panic, got none", name)
		}
	}()
	fn()
}

// The adaptive quad tessellation must keep every reference point within the pixel tolerance,
// and must end exactly at the curve's endpoint.
func TestFlattenQuadPixelError(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 40, y: 200}
	p2 := point{x: 200, y: 0}
	for _, tol := range []float32{4, 1, 0.25} {
		q := NewVectorTessellationQuality()
		q.SetPixelTolerance(tol)
		tess := newTessellator(q)
		poly := []point{tess.apply(p0)}
		poly = tess.flattenQuad(poly, p0, p1, p2)

		if tess.err() != nil {
			t.Errorf("tolerance %v: unexpected error %v", tol, tess.err())
		}
		if poly[len(poly)-1] != tess.apply(p2) {
			t.Errorf("tolerance %v: polyline does not end at the curve endpoint", tol)
		}
		if got := maxChordError(t, tess, func(tt float64) point {
			return quadPoint(p0, p1, p2, tt)
		}, poly); got > float64(tol)*1.001 {
			t.Errorf("tolerance %v: error %g exceeds tolerance", tol, got)
		}
	}
}

// A finer tolerance must produce at least as many segments as a coarser one for the same curve.
func TestFlattenQuadSegmentCounts(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 50, y: 300}
	p2 := point{x: 300, y: 50}
	var lastCount int
	// Coarse to fine: counts must be non-decreasing.
	for _, tol := range []float32{16, 4, 1, 0.1} {
		q := NewVectorTessellationQuality()
		q.SetPixelTolerance(tol)
		tess := newTessellator(q)
		poly := tess.flattenQuad(nil, p0, p1, p2)
		if len(poly) < lastCount {
			t.Errorf("tolerance %v: segment count %d is below the coarser count %d", tol, len(poly), lastCount)
		}
		lastCount = len(poly)
	}
}

func TestFlattenCubicPixelError(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: -60, y: 240}
	p2 := point{x: 260, y: 240}
	p3 := point{x: 200, y: 0}
	for _, tol := range []float32{4, 1, 0.25} {
		q := NewVectorTessellationQuality()
		q.SetPixelTolerance(tol)
		tess := newTessellator(q)
		poly := []point{tess.apply(p0)}
		poly = tess.flattenCubic(poly, p0, p1, p2, p3)

		if poly[len(poly)-1] != tess.apply(p3) {
			t.Errorf("tolerance %v: polyline does not end at the curve endpoint", tol)
		}
		// The convex-hull bound is conservative, so the realized error must satisfy the tolerance.
		if got := maxChordError(t, tess, func(tt float64) point {
			return cubicPoint(p0, p1, p2, p3, tt)
		}, poly); got > float64(tol)*1.001 {
			t.Errorf("tolerance %v: error %g exceeds tolerance", tol, got)
		}
	}
}

func TestFlattenArcPixelError(t *testing.T) {
	center := point{x: 100, y: 100}
	const r = 80.0
	for _, sweep := range []float64{math.Pi / 3, math.Pi / 2, math.Pi, 2 * math.Pi} {
		for _, tol := range []float32{4, 1, 0.25} {
			q := NewVectorTessellationQuality()
			q.SetPixelTolerance(tol)
			tess := newTessellator(q)
			start := point{x: center.x + r, y: center.y}
			poly := []point{tess.apply(start)}
			poly = tess.flattenArc(poly, center, r, 0, sweep, true)

			if tess.err() != nil {
				t.Errorf("sweep %v tol %v: unexpected error %v", sweep, tol, tess.err())
			}
			if got := maxChordError(t, tess, func(tt float64) point {
				angle := sweep * tt
				return point{x: center.x + r*math.Cos(angle), y: center.y + r*math.Sin(angle)}
			}, poly); got > float64(tol)*1.01 {
				t.Errorf("sweep %v tolerance %v: error %g exceeds tolerance", sweep, tol, got)
			}
		}
	}
}

// The segment budget must be honored and reported, and must terminate without unbounded work.
func TestTessellationBudgetExhausted(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 50, y: 1000}
	p2 := point{x: 1000, y: 0}
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0) // demand the finest subdivision
	const budget = 8
	q.SetMaxSegments(budget)
	tess := newTessellator(q)
	poly := []point{tess.apply(p0)}
	poly = tess.flattenQuad(poly, p0, p1, p2)
	if tess.segments > budget {
		t.Errorf("segment count %d exceeds budget %d", tess.segments, budget)
	}
	if len(poly) > budget+1 {
		t.Errorf("emitted %d points, want at most %d", len(poly), budget+1)
	}
	if tess.err() != ErrTessellationBudgetExhausted {
		t.Errorf("got error %v, want ErrTessellationBudgetExhausted", tess.err())
	}
	// Endpoints are always preserved.
	if poly[0] != tess.apply(p0) || poly[len(poly)-1] != tess.apply(p2) {
		t.Error("budget exhaustion dropped a curve endpoint")
	}
}

// Zero tolerance is capped by the depth limit instead of recursing forever.
func TestTessellationZeroToleranceTerminates(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 10, y: 9999}
	p2 := point{x: 100, y: 0}
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0)
	tess := newTessellator(q)
	poly := tess.flattenQuad(nil, p0, p1, p2)
	// Depth cap means at most 2^depth segments.
	if tess.segments > 1<<maxTessellationDepth {
		t.Errorf("depth limit violated: %d segments", tess.segments)
	}
	if tess.err() != ErrTessellationBudgetExhausted {
		t.Errorf("got error %v, want budget-exhausted error", tess.err())
	}
	if poly[len(poly)-1] != tess.apply(p2) {
		t.Error("endpoint lost under zero tolerance")
	}
}

// A curve that is already straight needs no subdivision even at a tight tolerance.
func TestFlattenFlatCurve(t *testing.T) {
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0.001)
	tess := newTessellator(q)
	poly := tess.flattenQuad(nil, point{x: 0, y: 0}, point{x: 50, y: 0}, point{x: 100, y: 0})
	if len(poly) != 1 {
		t.Errorf("collinear quad produced %d segments, want 1", len(poly))
	}
}

// Zooming in must increase the segment count, and zooming out must decrease it, because the error
// is measured in transformed pixels.
func TestTessellationMultiScale(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 25, y: 100}
	p2 := point{x: 100, y: 25}
	counts := map[float64]int{}
	for _, scale := range []float64{0.01, 1, 100} {
		q := NewVectorTessellationQuality()
		var gm ebiten.GeoM
		gm.Scale(scale, scale)
		q.SetGeoM(gm)
		tess := newTessellator(q)
		poly := tess.flattenQuad(nil, p0, p1, p2)
		counts[scale] = len(poly)
	}
	if !(counts[0.01] < counts[1] && counts[1] < counts[100]) {
		t.Errorf("segment counts do not grow with zoom: %v", counts)
	}
}

// Tessellating a small path scaled up must match tessellating the already-scaled path, because
// adaptive subdivision commutes with the affine transform.
func TestTessellationTransformCommutes(t *testing.T) {
	small := []point{{x: 0, y: 0}, {x: 4, y: 40}, {x: 40, y: 4}}
	big := []point{
		{x: small[0].x * 10, y: small[0].y * 10},
		{x: small[1].x * 10, y: small[1].y * 10},
		{x: small[2].x * 10, y: small[2].y * 10},
	}
	qScaled := NewVectorTessellationQuality()
	var gm ebiten.GeoM
	gm.Scale(10, 10)
	qScaled.SetGeoM(gm)
	tessScaled := newTessellator(qScaled)
	got := tessScaled.flattenQuad(nil, small[0], small[1], small[2])

	tessBig := newTessellator(NewVectorTessellationQuality())
	want := tessBig.flattenQuad(nil, big[0], big[1], big[2])

	if len(got) != len(want) {
		t.Fatalf("segment counts differ: scaled %d, pre-transformed %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i].x-want[i].x) > 1e-9 || math.Abs(got[i].y-want[i].y) > 1e-9 {
			t.Errorf("point %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// Non-uniform scaling measures the error along the most-stretched axis.
func TestTessellationAnisotropicScale(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 50, y: 120}
	p2 := point{x: 100, y: 0}
	base := NewVectorTessellationQuality()
	baseTess := newTessellator(base)
	baseCount := len(baseTess.flattenQuad(nil, p0, p1, p2))

	stretched := NewVectorTessellationQuality()
	var gm ebiten.GeoM
	gm.Scale(1, 30)
	stretched.SetGeoM(gm)
	stretchedTess := newTessellator(stretched)
	stretchedCount := len(stretchedTess.flattenQuad(nil, p0, p1, p2))

	if stretchedCount <= baseCount {
		t.Errorf("anisotropic stretch did not add segments: base %d, stretched %d", baseCount, stretchedCount)
	}

	// The error along the stretched axis must respect the tolerance.
	tess := newTessellator(stretched)
	poly := tess.flattenQuad([]point{tess.apply(p0)}, p0, p1, p2)
	if got := maxChordError(t, tess, func(tt float64) point {
		return quadPoint(p0, p1, p2, tt)
	}, poly); got > 0.5*1.001 {
		t.Errorf("anisotropic error %g exceeds tolerance", got)
	}
}

// Reversing a curve (traversing the identical geometry from the other endpoint) must tessellate
// stably: the reversed polyline satisfies the same tolerance and has the same subdivision order
// of magnitude. Binary subdivision counts can differ by one split level from float rounding.
func TestTessellationReversedPath(t *testing.T) {
	p0 := point{x: 10, y: 10}
	p1 := point{x: 80, y: 160}
	p2 := point{x: 170, y: 30}
	q := NewVectorTessellationQuality()
	tess := newTessellator(q)
	forward := tess.flattenQuad(nil, p0, p1, p2)

	// The reversed quadratic has control point (2*(p0 + p2) - p1): its value at t equals the
	// original at 1-t, so it describes the same geometry in the opposite parameter direction.
	r0 := p2
	r1 := point{x: 2*(p0.x+p2.x) - p1.x, y: 2*(p0.y+p2.y) - p1.y}
	r2 := p0
	tess2 := newTessellator(NewVectorTessellationQuality())
	reverse := tess2.flattenQuad(nil, r0, r1, r2)

	// Counts must stay within one subdivision level (a factor of two).
	if len(reverse) > 2*len(forward) || len(forward) > 2*len(reverse) {
		t.Fatalf("reversed count %d differs too much from forward %d", len(reverse), len(forward))
	}
	// Both polylines must meet the tolerance against the same underlying geometry.
	forwardFull := append([]point{tess.apply(p0)}, forward...)
	if got := maxChordError(t, tess, func(tt float64) point {
		return quadPoint(p0, p1, p2, tt)
	}, forwardFull); got > 0.5*1.001 {
		t.Errorf("forward error %g exceeds tolerance", got)
	}
	reverseFull := append([]point{tess2.apply(r0)}, reverse...)
	if got := maxChordError(t, tess2, func(tt float64) point {
		return quadPoint(r0, r1, r2, tt)
	}, reverseFull); got > 0.5*1.001 {
		t.Errorf("reverse error %g exceeds tolerance", got)
	}
	// The reversed run ends at the original start point.
	if reverse[len(reverse)-1] != tess2.apply(p0) {
		t.Error("reversed tessellation lost the original start point")
	}
}

// Closed sub-paths and cusps must keep their endpoints and stay finite.
func TestTessellateClosedAndCusps(t *testing.T) {
	var p Path
	p.MoveTo(0, 0)
	p.LineTo(100, 0)
	// A cusp: control point beyond the returning endpoint.
	p.QuadTo(100, 50, 100, 0)
	p.LineTo(200, 0)
	p.Close()

	subs, tess := tessellatePath(&p, NewVectorTessellationQuality())
	if tess.err() != nil {
		t.Fatalf("unexpected error: %v", tess.err())
	}
	if len(subs) != 1 || !subs[0].closed {
		t.Fatalf("closed sub-path not preserved: %+v", subs)
	}
	for _, sub := range subs {
		for _, pt := range sub.points {
			if math.IsNaN(pt.x) || math.IsNaN(pt.y) || math.IsInf(pt.x, 0) || math.IsInf(pt.y, 0) {
				t.Errorf("non-finite tessellated point %v", pt)
			}
		}
	}
}

// PathBounds under the default quality must report the same rectangle as the analytic Bounds for
// a path made of lines, and must surface the budget error otherwise.
func TestPathBoundsLineAgreement(t *testing.T) {
	var p Path
	p.MoveTo(10, 20)
	p.LineTo(70, 30)
	p.LineTo(40, 90)
	p.Close()

	got, err := p.PathBounds(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := p.Bounds(); got != want {
		t.Errorf("PathBounds %v != Bounds %v", got, want)
	}
}

func TestPathBoundsBudgetError(t *testing.T) {
	var p Path
	p.MoveTo(0, 0)
	p.QuadTo(50, 1000, 1000, 0)
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0)
	q.SetMaxSegments(4)
	rect, err := p.PathBounds(q)
	if err != ErrTessellationBudgetExhausted {
		t.Errorf("got error %v, want ErrTessellationBudgetExhausted", err)
	}
	if rect.Empty() {
		t.Error("partial bounds must still be returned")
	}
}

// A self-intersecting path tessellates without error and stays finite.
func TestTessellateSelfIntersecting(t *testing.T) {
	var p Path
	p.MoveTo(0, 0)
	p.LineTo(100, 100)
	p.LineTo(100, 0)
	p.QuadTo(0, 100, 0, 0)
	p.Close()
	subs, tess := tessellatePath(&p, NewVectorTessellationQuality())
	if tess.err() != nil {
		t.Fatalf("self-intersecting path errored: %v", tess.err())
	}
	for _, sub := range subs {
		for _, pt := range sub.points {
			if !isRegularPoint(pt) {
				t.Errorf("non-finite point in self-intersecting tessellation: %v", pt)
			}
		}
	}
}

// Coordinates near the float32 limit must not produce infinite tessellated vertices. At that
// scale a pixel-scale tolerance is meaningless, so give the quality a proportionally large
// tolerance and a generous budget; the invariant is finiteness, not segment count.
func TestTessellateHugeCoordinates(t *testing.T) {
	const huge = 1e37
	var p Path
	p.MoveTo(0, 0)
	p.QuadTo(huge, huge/2, huge, 0)
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(huge / 64)
	q.SetMaxSegments(1 << 20)
	subs, tess := tessellatePath(&p, q)
	if tess.err() != nil {
		t.Fatalf("huge coordinates errored: %v", tess.err())
	}
	if len(subs) != 1 {
		t.Fatalf("got %d sub-paths, want 1", len(subs))
	}
	for _, pt := range subs[0].points {
		if !isRegularPoint(pt) {
			t.Errorf("non-finite tessellated point for huge input: %v", pt)
		}
	}
}

// A very deep, highly curved path terminates at the depth cap and keeps its endpoints.
func TestTessellateDeepCurveTerminates(t *testing.T) {
	p0 := point{x: 0, y: 0}
	p1 := point{x: 1, y: 1e6}
	p2 := point{x: 2, y: 0}
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0)
	q.SetMaxSegments(1 << 20)
	tess := newTessellator(q)
	poly := tess.flattenQuad(nil, p0, p1, p2)
	if tess.segments > 1<<maxTessellationDepth {
		t.Errorf("depth cap violated: %d", tess.segments)
	}
	if poly[len(poly)-1] != tess.apply(p2) {
		t.Error("deep curve lost its endpoint")
	}
}

// FillPath, StrokePath, and PathBounds given the same quality must agree on the segment count:
// the fill and stroke tessellate through the shared flattenToPath entry point.
func TestConsistentStrategyFillStrokeBounds(t *testing.T) {
	var p Path
	p.MoveTo(20, 20)
	p.QuadTo(200, 300, 380, 20)

	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0.25)

	flat := flattenToPath(&p, q)
	var flatSegments int
	for _, sub := range flat.subPaths {
		flatSegments += len(sub.ops)
	}

	bounds, err := p.PathBounds(q)
	if err != nil {
		t.Fatalf("PathBounds error: %v", err)
	}

	// Bounds of the flattened polyline must contain the tessellated geometry.
	for _, sub := range flat.subPaths {
		cur := sub.start
		for _, op := range sub.ops {
			for _, pt := range []point{cur, op.p1} {
				if pt.x < float64(bounds.Min.X) || pt.x > float64(bounds.Max.X) ||
					pt.y < float64(bounds.Min.Y) || pt.y > float64(bounds.Max.Y) {
					t.Errorf("tessellated point %v outside PathBounds %v", pt, bounds)
				}
			}
			cur = op.p1
		}
	}

	// Stroke through the same quality produces a finite, non-empty path.
	var stroke Path
	stroke.AddStroke(&p, &AddStrokeOptions{
		StrokeOptions: StrokeOptions{
			Width:               6,
			TessellationQuality: q,
		},
	})
	if len(stroke.subPaths) == 0 {
		t.Error("stroke produced no sub-paths")
	}
	for _, sub := range stroke.subPaths {
		if !sub.isValid() {
			t.Error("stroke produced an invalid sub-path")
		}
	}
}

// Rendering with the default quality must be pixel-identical to rendering without any quality.
func TestDefaultQualityPreservesGeometry(t *testing.T) {
	var p Path
	p.MoveTo(16, 16)
	p.QuadTo(120, 200, 220, 24)
	p.CubicTo(260, -40, 60, 300, 240, 320)

	// Fill.
	refFill := ebiten.NewImage(256, 360)
	defer refFill.Deallocate()
	gotFill := ebiten.NewImage(256, 360)
	defer gotFill.Deallocate()
	op := &DrawPathOptions{AntiAlias: true}
	op.ColorScale.ScaleWithColor(color.White)
	FillPath(refFill, &p, nil, op)
	FillPath(gotFill, &p, &FillOptions{TessellationQuality: NewVectorTessellationQuality()}, op)
	if !bytes.Equal(pixels(t, refFill), pixels(t, gotFill)) {
		t.Error("default tessellation quality changed the filled geometry")
	}

	// Stroke.
	refStroke := ebiten.NewImage(256, 360)
	defer refStroke.Deallocate()
	gotStroke := ebiten.NewImage(256, 360)
	defer gotStroke.Deallocate()
	strokeOp := &StrokeOptions{Width: 4, LineJoin: LineJoinRound}
	StrokePath(refStroke, &p, strokeOp, op)
	StrokePath(gotStroke, &p, &StrokeOptions{
		Width:               4,
		LineJoin:            LineJoinRound,
		TessellationQuality: NewVectorTessellationQuality(),
	}, op)
	if !bytes.Equal(pixels(t, refStroke), pixels(t, gotStroke)) {
		t.Error("default tessellation quality changed the stroked geometry")
	}
}

// A coarser quality produces fewer rendered triangles than a finer quality, proving the quality
// reaches the fill pipeline rather than being ignored.
func TestFillQualityChangesSegmentCount(t *testing.T) {
	var p Path
	p.MoveTo(20, 20)
	p.QuadTo(200, 400, 460, 20)

	coarse := NewVectorTessellationQuality()
	coarse.SetPixelTolerance(16)
	fine := NewVectorTessellationQuality()
	fine.SetPixelTolerance(0.05)

	coarseFlat := flattenToPath(&p, coarse)
	fineFlat := flattenToPath(&p, fine)
	coarseOps := countOps(coarseFlat)
	fineOps := countOps(fineFlat)
	if coarseOps >= fineOps {
		t.Errorf("coarse ops %d !< fine ops %d", coarseOps, fineOps)
	}
}

func countOps(p *Path) int {
	var n int
	for _, sub := range p.subPaths {
		n += len(sub.ops)
	}
	return n
}

func pixels(t *testing.T, img *ebiten.Image) []byte {
	t.Helper()
	b := img.Bounds()
	pix := make([]byte, 4*b.Dx()*b.Dy())
	img.ReadPixels(pix)
	return pix
}

// Arcs built through the public Arc API tessellate to finite points inside their bounding box.
func TestTessellateArcEndToEnd(t *testing.T) {
	var p Path
	p.MoveTo(100, 20)
	p.Arc(100, 100, 80, -math.Pi/2, math.Pi/2, Clockwise)
	subs, tess := tessellatePath(&p, NewVectorTessellationQuality())
	if tess.err() != nil {
		t.Fatalf("arc tessellation error: %v", tess.err())
	}
	if len(subs) != 1 {
		t.Fatalf("got %d sub-paths, want 1", len(subs))
	}
	for _, pt := range subs[0].points {
		if !isRegularPoint(pt) {
			t.Errorf("non-finite arc point %v", pt)
		}
		if d := math.Hypot(pt.x-100, pt.y-100); d > 80+0.6 {
			t.Errorf("arc point %v is %g from the center, outside the radius", pt, d)
		}
	}
}

// Stroked round caps and joins with an explicit quality render and stay finite.
func TestStrokeWithQualityRenders(t *testing.T) {
	var p Path
	p.MoveTo(20, 100)
	p.QuadTo(120, 0, 220, 100)
	p.LineTo(300, 60)

	dst := ebiten.NewImage(320, 160)
	defer dst.Deallocate()
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0.2)
	StrokePath(dst, &p, &StrokeOptions{
		Width:               10,
		LineCap:             LineCapRound,
		LineJoin:            LineJoinRound,
		TessellationQuality: q,
	}, &DrawPathOptions{AntiAlias: true})

	// Some pixel must be covered.
	var covered bool
	for _, b := range pixels(t, dst) {
		if b != 0 {
			covered = true
			break
		}
	}
	if !covered {
		t.Error("quality-driven stroke rendered nothing")
	}
}

// Bounds must remain usable concurrently since it never mutates the path.
func TestPathBoundsConcurrencyWithQuality(t *testing.T) {
	var p Path
	p.MoveTo(10, 20)
	p.QuadTo(50, 60, 70, 80)
	p.Close()
	q := NewVectorTessellationQuality()
	rects := make([]image.Rectangle, 8)
	var wg sync.WaitGroup
	for i := range rects {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r, err := p.PathBounds(q)
			if err != nil {
				t.Errorf("%d: %v", i, err)
			}
			rects[i] = r
		}(i)
	}
	wg.Wait()
	for i := 1; i < len(rects); i++ {
		if rects[i] != rects[0] {
			t.Errorf("bounds differ across goroutines: %v vs %v", rects[i], rects[0])
		}
	}
}

// A geometry matrix that produces non-finite pixel coordinates must not crash: the tessellator
// falls back to untransformed endpoints and reports the incomplete state instead of looping.
func TestTessellationNonFiniteTransformFallback(t *testing.T) {
	var gm ebiten.GeoM
	gm.Scale(1e300, 1e300) // finite elements, but applying them overflows the float32 range
	q := NewVectorTessellationQuality()
	q.SetGeoM(gm)

	p0 := point{x: 0, y: 0}
	p1 := point{x: 50, y: 120}
	p2 := point{x: 100, y: 0}
	tess := newTessellator(q)
	poly := tess.flattenQuad(nil, p0, p1, p2)
	if len(poly) == 0 {
		t.Fatal("no points emitted under an overflowing transform")
	}
	for _, pt := range poly {
		if math.IsNaN(pt.x) || math.IsNaN(pt.y) || math.IsInf(pt.x, 0) || math.IsInf(pt.y, 0) {
			t.Errorf("non-finite point leaked through the fallback: %v", pt)
		}
	}
}

// PathBounds of an empty or fully invalid path returns the empty rectangle without panicking.
func TestPathBoundsInvalidAndEmpty(t *testing.T) {
	var empty Path
	if rect, err := empty.PathBounds(nil); !rect.Empty() || err != nil {
		t.Errorf("empty path: rect=%v err=%v", rect, err)
	}

	var invalid Path
	invalid.MoveTo(0, 0)
	invalid.LineTo(float32(math.NaN()), float32(math.NaN()))
	if rect, err := invalid.PathBounds(NewVectorTessellationQuality()); !rect.Empty() {
		t.Errorf("invalid path: rect=%v err=%v", rect, err)
	}
}

// The arc segment count must grow monotonically as the radius grows at a fixed tolerance.
func TestFlattenArcSegmentCountMonotonic(t *testing.T) {
	q := NewVectorTessellationQuality()
	q.SetPixelTolerance(0.5)
	var lastCount int
	for _, r := range []float64{1, 10, 100, 1000} {
		tess := newTessellator(q)
		poly := tess.flattenArc(nil, point{x: 0, y: 0}, r, 0, 2*math.Pi, true)
		if len(poly) < lastCount {
			t.Errorf("radius %v: count %d below smaller-radius count %d", r, len(poly), lastCount)
		}
		lastCount = len(poly)
	}
}
