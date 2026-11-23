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
	"errors"
	"image"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	// defaultPixelTessellationTolerance is the default maximum distance, in pixels,
	// between a tessellated line segment and the true curve.
	// It matches the allowance used by the deprecated flat-path tessellation.
	defaultPixelTessellationTolerance = 0.5

	// defaultMaxTessellationSegments is the default budget for the number of line segments
	// produced for one tessellated path.
	defaultMaxTessellationSegments = 1 << 16

	// maxTessellationDepth bounds adaptive subdivision independently of the segment budget,
	// so a zero tolerance or a pathological curve cannot cause unbounded recursion.
	maxTessellationDepth = 32
)

// ErrTessellationBudgetExhausted reports that a tessellation reached its segment budget or its
// subdivision-depth limit while the current approximation was still coarser than the requested
// pixel tolerance.
//
// The returned geometry is still valid and terminates at the curve endpoints: further subdivision
// was simply stopped. Rendering functions such as [FillPath] and [StrokePath] clamp silently and
// do not return this error; [Path.PathBounds] returns it together with the partial bounds.
var ErrTessellationBudgetExhausted = errors.New("vector: tessellation segment budget exhausted before the pixel tolerance was reached")

// VectorTessellationQuality determines how curves are adaptively approximated with line segments.
//
// The same quality value can be passed to [FillOptions], [StrokeOptions], and [Path.PathBounds]
// so that filling, stroking, and bounds computation use one consistent subdivision strategy.
//
// The subdivision error is measured in pixels after applying the quality's geometry matrix,
// so the same path tessellated at different zoom levels gets an appropriate number of segments:
// a curve shown small uses few segments, while the same curve shown large uses more.
//
// A zero VectorTessellationQuality is not a valid quality. Use [NewVectorTessellationQuality].
type VectorTessellationQuality struct {
	// pixelTolerance is the maximum permitted deviation, in transformed pixels, between a
	// tessellated chord and the curve it approximates. A zero value requests the finest
	// subdivision possible until the segment budget or the depth limit is reached.
	pixelTolerance float64

	// maxSegments bounds the number of line segments produced for one path. A zero value means
	// defaultMaxTessellationSegments.
	maxSegments int

	// geoM maps path coordinates to the pixel space in which the error is measured.
	geoM ebiten.GeoM
}

// NewVectorTessellationQuality returns a tessellation quality with the default pixel tolerance
// (0.5 pixels), the default segment budget, and an identity geometry matrix.
func NewVectorTessellationQuality() *VectorTessellationQuality {
	return &VectorTessellationQuality{
		pixelTolerance: defaultPixelTessellationTolerance,
		maxSegments:    defaultMaxTessellationSegments,
	}
}

// SetPixelTolerance sets the maximum permitted deviation, in transformed pixels, between a
// tessellated chord and the curve it approximates.
//
// A zero tolerance requests the finest subdivision possible until the segment budget or the
// subdivision-depth limit is reached. A negative finite tolerance causes a panic.
// An infinite or NaN tolerance is ignored and leaves the previous value unchanged.
func (q *VectorTessellationQuality) SetPixelTolerance(tolerance float32) {
	t := float64(tolerance)
	if !isRegular(t) {
		return
	}
	if t < 0 {
		panic("vector: tessellation pixel tolerance must be non-negative")
	}
	q.pixelTolerance = t
}

// PixelTolerance returns the current pixel tolerance.
func (q *VectorTessellationQuality) PixelTolerance() float32 {
	return float32(q.pixelTolerance)
}

// SetMaxSegments sets the maximum number of line segments one tessellated path may produce.
//
// When the budget is reached, subdivision stops and the remaining curves keep their current
// chords; [Path.PathBounds] reports [ErrTessellationBudgetExhausted] in that case.
// A zero value restores the default budget. A negative value causes a panic.
func (q *VectorTessellationQuality) SetMaxSegments(maxSegments int) {
	if maxSegments < 0 {
		panic("vector: tessellation max segments must be non-negative")
	}
	if maxSegments == 0 {
		maxSegments = defaultMaxTessellationSegments
	}
	q.maxSegments = maxSegments
}

// MaxSegments returns the current segment budget.
func (q *VectorTessellationQuality) MaxSegments() int {
	if q.maxSegments == 0 {
		return defaultMaxTessellationSegments
	}
	return q.maxSegments
}

// SetGeoM sets the geometry matrix that maps path coordinates to the pixel space in which the
// tessellation error is measured.
//
// If any matrix element is infinite or NaN, the call is ignored and the previous matrix is kept.
func (q *VectorTessellationQuality) SetGeoM(geoM ebiten.GeoM) {
	for i := range 2 {
		for j := range 3 {
			if !isRegular(geoM.Element(i, j)) {
				return
			}
		}
	}
	q.geoM = geoM
}

// GeoM returns the geometry matrix used for error measurement.
func (q *VectorTessellationQuality) GeoM() ebiten.GeoM {
	return q.geoM
}

// isDefault reports whether q is equivalent to the quality produced by
// NewVectorTessellationQuality. The default quality renders through the standard pipeline,
// so passing it keeps the geometry of existing paths byte-identical.
func (q *VectorTessellationQuality) isDefault() bool {
	if q == nil {
		return true
	}
	if q.pixelTolerance != defaultPixelTessellationTolerance {
		return false
	}
	if q.MaxSegments() != defaultMaxTessellationSegments {
		return false
	}
	for i := range 2 {
		for j := range 3 {
			want := 0.0
			if i == j {
				want = 1
			}
			if q.geoM.Element(i, j) != want {
				return false
			}
		}
	}
	return true
}

// tessellator flattens curves into line segments under one quality.
// It never recurses: subdivision uses an explicit stack, and both a segment budget and a fixed
// depth limit guarantee termination.
type tessellator struct {
	q          *VectorTessellationQuality
	a, b, c, d float64
	tx, ty     float64
	identity   bool

	// segments counts the reserved line segments. It never exceeds the quality's budget.
	segments int

	// incomplete is set whenever a primitive stopped subdividing while its error was still
	// above the tolerance.
	incomplete bool
}

func newTessellator(q *VectorTessellationQuality) *tessellator {
	if q == nil {
		q = NewVectorTessellationQuality()
	}
	t := &tessellator{
		q:        q,
		identity: true,
	}
	t.a = q.geoM.Element(0, 0)
	t.b = q.geoM.Element(0, 1)
	t.tx = q.geoM.Element(0, 2)
	t.c = q.geoM.Element(1, 0)
	t.d = q.geoM.Element(1, 1)
	t.ty = q.geoM.Element(1, 2)
	t.identity = t.a == 1 && t.b == 0 && t.tx == 0 && t.c == 0 && t.d == 1 && t.ty == 0
	return t
}

func (t *tessellator) apply(p point) point {
	if t.identity {
		return p
	}
	return point{
		x: t.a*p.x + t.b*p.y + t.tx,
		y: t.c*p.x + t.d*p.y + t.ty,
	}
}

func (t *tessellator) tolerance() float64 {
	return t.q.pixelTolerance
}

// reserve charges n segments against the budget. It reports whether the reservation fit.
// A failed reservation always marks the tessellation incomplete.
func (t *tessellator) reserve(n int) bool {
	if t.segments+n <= t.q.MaxSegments() {
		t.segments += n
		return true
	}
	t.incomplete = true
	return false
}

// err returns the error state after flattening.
func (t *tessellator) err() error {
	if t.incomplete {
		return ErrTessellationBudgetExhausted
	}
	return nil
}

// distanceToLine returns the perpendicular distance from p to the infinite line through a and b.
// The direction is normalized first with math.Hypot, which avoids overflow even when the
// coordinate differences are close to the float32 range.
func distanceToLine(p, a, b point) float64 {
	dx := b.x - a.x
	dy := b.y - a.y
	length := math.Hypot(dx, dy)
	if length == 0 {
		return math.Hypot(p.x-a.x, p.y-a.y)
	}
	if !isRegular(length) {
		return math.Inf(1)
	}
	// Unit perpendicular of (dx, dy) is (-dy/length, dx/length).
	return math.Abs(-(p.x-a.x)*dy/length + (p.y-a.y)*dx/length)
}

// quadPixelError returns the exact maximum distance, in transformed pixels, between the quadratic
// Bézier curve and its chord. A quadratic Bézier is a parabola whose greatest offset from the
// chord line is half the offset of its control point, reached at t = 0.5.
func (t *tessellator) quadPixelError(p0, p1, p2 point) float64 {
	q0 := t.apply(p0)
	q1 := t.apply(p1)
	q2 := t.apply(p2)
	if !isRegularPoint(q0) || !isRegularPoint(q1) || !isRegularPoint(q2) {
		return math.Inf(1)
	}
	if q0 == q2 {
		// A degenerate chord: the curve leaves and returns to the same point.
		return 0.5 * math.Hypot(q1.x-q0.x, q1.y-q0.y)
	}
	return 0.5 * distanceToLine(q1, q0, q2)
}

// cubicPixelError returns a rigorous upper bound on the distance, in transformed pixels, between
// a cubic Bézier curve and its chord. A Bézier curve stays within the convex hull of its control
// points, so the greatest deviation cannot exceed the greatest distance of an inner control point
// from the chord line.
func (t *tessellator) cubicPixelError(p0, p1, p2, p3 point) float64 {
	q0 := t.apply(p0)
	q1 := t.apply(p1)
	q2 := t.apply(p2)
	q3 := t.apply(p3)
	if !isRegularPoint(q0) || !isRegularPoint(q1) || !isRegularPoint(q2) || !isRegularPoint(q3) {
		return math.Inf(1)
	}
	if q0 == q3 {
		return max(math.Hypot(q1.x-q0.x, q1.y-q0.y), math.Hypot(q2.x-q0.x, q2.y-q0.y))
	}
	return max(distanceToLine(q1, q0, q3), distanceToLine(q2, q0, q3))
}

// emit appends the transformed endpoint p to the polyline dst. When the transformed point is not
// finite, the untransformed endpoint is emitted and the tessellation is marked incomplete.
func (t *tessellator) emit(dst []point, p point) []point {
	q := t.apply(p)
	if !isRegularPoint(q) {
		t.incomplete = true
		q = p
	}
	return append(dst, q)
}

// flattenLine appends the endpoint of a line segment.
func (t *tessellator) flattenLine(dst []point, p0, p1 point) []point {
	if !t.reserve(1) {
		// The budget is exhausted: the chord is still emitted, so the polyline keeps its endpoints.
		return t.emit(dst, p1)
	}
	return t.emit(dst, p1)
}

type quadFrame struct {
	p0, p1, p2 point
	depth      int
}

// flattenQuad adaptively appends the endpoints of a quadratic Bézier curve, excluding its start.
func (t *tessellator) flattenQuad(dst []point, p0, p1, p2 point) []point {
	if !t.reserve(1) {
		return t.emit(dst, p2)
	}
	stack := []quadFrame{{p0: p0, p1: p1, p2: p2}}
	for len(stack) > 0 {
		fr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		error := t.quadPixelError(fr.p0, fr.p1, fr.p2)
		if error <= t.tolerance() || fr.depth >= maxTessellationDepth || !isRegular(error) {
			if error > t.tolerance() {
				t.incomplete = true
			}
			dst = t.emit(dst, fr.p2)
			continue
		}

		// Split the quadratic at t = 0.5 by De Casteljau's algorithm.
		// Splitting in path coordinates is equivalent to splitting after the affine transform,
		// which keeps zoom-dependent subdivision independent of where the split is done.
		p01 := point{x: (fr.p0.x + fr.p1.x) / 2, y: (fr.p0.y + fr.p1.y) / 2}
		p12 := point{x: (fr.p1.x + fr.p2.x) / 2, y: (fr.p1.y + fr.p2.y) / 2}
		p012 := point{x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2}
		if !t.reserve(1) {
			// Splitting would exceed the budget: keep the chord.
			dst = t.emit(dst, fr.p2)
			continue
		}
		stack = append(stack,
			quadFrame{p0: p012, p1: p12, p2: fr.p2, depth: fr.depth + 1},
			quadFrame{p0: fr.p0, p1: p01, p2: p012, depth: fr.depth + 1},
		)
	}
	return dst
}

type cubicFrame struct {
	p0, p1, p2, p3 point
	depth          int
}

// flattenCubic adaptively appends the endpoints of a cubic Bézier curve, excluding its start.
func (t *tessellator) flattenCubic(dst []point, p0, p1, p2, p3 point) []point {
	if !t.reserve(1) {
		return t.emit(dst, p3)
	}
	stack := []cubicFrame{{p0: p0, p1: p1, p2: p2, p3: p3}}
	for len(stack) > 0 {
		fr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		error := t.cubicPixelError(fr.p0, fr.p1, fr.p2, fr.p3)
		if error <= t.tolerance() || fr.depth >= maxTessellationDepth || !isRegular(error) {
			if error > t.tolerance() {
				t.incomplete = true
			}
			dst = t.emit(dst, fr.p3)
			continue
		}

		// Split the cubic at t = 0.5 by De Casteljau's algorithm.
		p01 := point{x: (fr.p0.x + fr.p1.x) / 2, y: (fr.p0.y + fr.p1.y) / 2}
		p12 := point{x: (fr.p1.x + fr.p2.x) / 2, y: (fr.p1.y + fr.p2.y) / 2}
		p23 := point{x: (fr.p2.x + fr.p3.x) / 2, y: (fr.p2.y + fr.p3.y) / 2}
		p012 := point{x: (p01.x + p12.x) / 2, y: (p01.y + p12.y) / 2}
		p123 := point{x: (p12.x + p23.x) / 2, y: (p12.y + p23.y) / 2}
		p0123 := point{x: (p012.x + p123.x) / 2, y: (p012.y + p123.y) / 2}
		if !t.reserve(1) {
			dst = t.emit(dst, fr.p3)
			continue
		}
		stack = append(stack,
			cubicFrame{p0: p0123, p1: p123, p2: p23, p3: fr.p3, depth: fr.depth + 1},
			cubicFrame{p0: fr.p0, p1: p01, p2: p012, p3: p0123, depth: fr.depth + 1},
		)
	}
	return dst
}

// flattenArc appends points along a circular arc, excluding its start.
//
// The segment count is derived analytically: a chord of a circle with radius r subtending an
// angle Δθ sags by r(1 - cos(Δθ/2)), so the smallest segment count whose sagitta stays within
// the pixel tolerance is computed directly, without recursion. Non-uniform scaling is handled by
// measuring the transformed radii and sampling the transformed curve.
func (t *tessellator) flattenArc(dst []point, center point, radius float64, startAngle, endAngle float64, clockwise bool) []point {
	if !(radius > 0) || !isRegular(radius) {
		t.incomplete = true
		return dst
	}

	sweep := endAngle - startAngle
	if clockwise {
		if sweep < 0 {
			sweep += 2 * math.Pi
		}
	} else {
		sweep = -sweep
		if sweep < 0 {
			sweep += 2 * math.Pi
		}
	}
	if sweep <= 0 || !isRegular(sweep) {
		return dst
	}

	// The transformed axes of the circle give the ellipse radii in pixel space.
	rx := math.Hypot(t.a*radius, t.c*radius)
	ry := math.Hypot(t.b*radius, t.d*radius)
	if !isRegular(rx) || !isRegular(ry) || rx == 0 || ry == 0 {
		t.incomplete = true
		return dst
	}
	// The largest radius produces the largest sagitta, so it drives the segment count.
	rMax := max(rx, ry)

	// Solve r(1 - cos(Δθ/2)) <= tolerance for Δθ, with a safe fallback for tolerance >= rMax,
	// where acos would leave its [0, 1] domain.
	n := 1
	if t.tolerance() < rMax {
		delta := 2 * math.Acos(1-t.tolerance()/rMax)
		if !(delta > 0) || !isRegular(delta) {
			delta = sweep
		}
		n = int(math.Ceil(sweep / delta))
		if n < 1 {
			n = 1
		}
	}

	if !t.reserve(n) {
		// Clamp the segment count to the remaining budget.
		n = t.q.MaxSegments() - t.segments
		if n < 1 {
			n = 1
		}
		t.segments += n
		t.incomplete = true
	}

	ct := t.apply(center)
	if !isRegularPoint(ct) {
		t.incomplete = true
		return dst
	}
	for i := 1; i <= n; i++ {
		angle := startAngle
		if clockwise {
			angle += sweep * float64(i) / float64(n)
		} else {
			angle -= sweep * float64(i) / float64(n)
		}
		sin, cos := math.Sincos(angle)
		// Sample in path coordinates and transform, so anisotropic scales trace the exact ellipse.
		p := point{
			x: center.x + radius*cos,
			y: center.y + radius*sin,
		}
		dst = t.emit(dst, p)
	}
	return dst
}

// tessellatedSubPath is a flattened sub-path: a polyline and whether it is closed.
type tessellatedSubPath struct {
	points []point
	closed bool
}

// tessellatePath flattens every valid sub-path of p into polylines under q.
// Invalid sub-paths are skipped, matching the behavior of [Path.Bounds] and the fill pipeline.
func tessellatePath(p *Path, q *VectorTessellationQuality) ([]tessellatedSubPath, *tessellator) {
	t := newTessellator(q)
	result := make([]tessellatedSubPath, 0, len(p.subPaths))
	var points []point
	for i := range p.subPaths {
		subPath := &p.subPaths[i]
		if !subPath.isValid() || len(subPath.ops) == 0 {
			continue
		}
		points = points[:0]
		points = t.emit(points, subPath.start)
		cur := subPath.start
		for _, op := range subPath.ops {
			switch op.typ {
			case opTypeLineTo:
				points = t.flattenLine(points, cur, op.p1)
				cur = op.p1
			case opTypeQuadTo:
				points = t.flattenQuad(points, cur, op.p1, op.p2)
				cur = op.p2
			}
		}
		result = append(result, tessellatedSubPath{
			points: append([]point(nil), points...),
			closed: subPath.closed,
		})
	}
	return result, t
}

// flattenToPath returns a new path containing only MoveTo, LineTo, and Close operations that
// approximates p under q. It is the shared entry point of FillPath, StrokePath, and PathBounds.
func flattenToPath(p *Path, q *VectorTessellationQuality) *Path {
	flat := &Path{}
	subs, _ := tessellatePath(p, q)
	for _, sub := range subs {
		if len(sub.points) == 0 {
			continue
		}
		flat.moveTo(sub.points[0].x, sub.points[0].y)
		for _, pt := range sub.points[1:] {
			flat.lineTo(pt.x, pt.y)
		}
		if sub.closed {
			flat.Close()
		}
	}
	return flat
}

// PathBounds returns the minimum bounding rectangle of the path after tessellating its curves
// into line segments under q.
//
// Unlike the analytic [Path.Bounds], PathBounds measures the geometry that is actually rendered
// after adaptive subdivision, in the pixel space defined by q's geometry matrix.
//
// If q is nil, the default quality is used. PathBounds returns [ErrTessellationBudgetExhausted]
// together with still-valid partial bounds when subdivision was stopped by the segment budget or
// the depth limit before reaching the requested tolerance.
func (p *Path) PathBounds(q *VectorTessellationQuality) (image.Rectangle, error) {
	subs, t := tessellatePath(p, q)
	if len(subs) == 0 {
		return image.Rectangle{}, t.err()
	}

	totalMinX := math.MaxInt
	totalMinY := math.MaxInt
	totalMaxX := math.MinInt
	totalMaxY := math.MinInt
	for _, sub := range subs {
		for _, pt := range sub.points {
			totalMinX = min(totalMinX, floor(pt.x))
			totalMinY = min(totalMinY, floor(pt.y))
			totalMaxX = max(totalMaxX, ceil(pt.x))
			totalMaxY = max(totalMaxY, ceil(pt.y))
		}
	}
	return image.Rect(totalMinX, totalMinY, totalMaxX, totalMaxY), t.err()
}
