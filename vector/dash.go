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
	"fmt"
	"math"
	"sync"
)

const (
	// maxDashArrayLen is the maximum number of entries accepted by [NewStrokeDashPlan].
	// A longer array is rejected instead of letting pattern iteration grow without bound.
	maxDashArrayLen = 1024

	// maxDashIntervals bounds the number of drawn intervals collected for one sub-path.
	// A pattern repeated more often than this is truncated: the geometry stays bounded
	// and renderable instead of consuming memory without limit. The limit is far above
	// what is visible at a pixel scale.
	maxDashIntervals = 1 << 12

	// maxDashSteps bounds the pattern walk itself. It is separate from maxDashIntervals
	// because a pattern made mostly of gaps can advance many times without drawing, and
	// the walk must still terminate.
	maxDashSteps = 1 << 16

	// dashInverseIterations is the safety bound for arc-length inversion. The safeguarded
	// Newton iteration normally converges in a few steps; the bound keeps pathological
	// curves strictly bounded even if the derivative is unusable.
	dashInverseIterations = 24

	// dashInverseEpsilon is the parameter bracket width at which arc-length inversion stops.
	// It is far below float32 parameter resolution.
	dashInverseEpsilon = 1e-12

	// arcLengthDepth bounds adaptive quadrature subdivision. 24 halvings shrink a panel to
	// 2^-24 of the curve, far below float32 resolution, so subdivision always terminates.
	arcLengthDepth = 24

	// arcLengthRelTol is the relative convergence threshold for adaptive arc-length quadrature.
	arcLengthRelTol = 1e-10

	// maxArcLengthPanels bounds the number of panels one arc-length integral may evaluate.
	// Nearly cusped curves concentrate the integrand; once the budget is spent the current
	// estimate is accepted, so the error stays bounded and the work stays finite.
	maxArcLengthPanels = 1 << 12
)

// ErrInvalidStrokeDashPlan reports that a dash array or offset passed to
// [NewStrokeDashPlan] is invalid: the array is empty or longer than 1024 entries, an
// entry or the offset is NaN or infinite, an entry is negative, or every entry is zero.
// Errors returned by [NewStrokeDashPlan] wrap this value, so they can be tested with
// [errors.Is].
var ErrInvalidStrokeDashPlan = errors.New("vector: invalid stroke dash plan")

func newStrokeDashPlanError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidStrokeDashPlan, fmt.Sprintf(format, args...))
}

// StrokeDashPlan is a reusable plan that renders a stroke as a dashed line whose dashes
// and gaps are measured against the true arc length of each sub-path.
//
// A plan is built once with [NewStrokeDashPlan] and can be reused across
// [Path.AddStroke] calls, paths, and frames, including from multiple goroutines: a plan
// is immutable after construction.
//
// The dash array alternates drawn and gap lengths, starting with a drawn entry. Odd
// indices are gaps, even indices are drawn, and trailing entries keep their parity. The
// pattern repeats along every source sub-path after shifting it by the plan's offset,
// following the same rule as the SVG stroke-dasharray and stroke-dashoffset attributes.
//
// Dashes are measured in the source path's coordinate space before
// [AddStrokeOptions.GeoM] is applied, so a geometry matrix scales dash lengths together
// with the rest of the stroke.
//
// Lines use their exact length. Quadratic Bézier curves use adaptive Gauss-Legendre
// arc-length integration and safeguarded inversion, and cubics and circular arcs inherit
// the same treatment because [Path.CubicTo] and [Path.Arc] decompose their input into
// quadratics when the path is built. Dash boundaries are therefore placed within the
// integration tolerance, independently of the path's shape.
//
// Degenerate input is bounded rather than rejected at stroke time:
//   - A zero-length entry occupies no arc length and keeps its parity, so it merges with
//     a neighbouring drawn entry instead of drawing a cap.
//   - An array with no positive entry is rejected by [NewStrokeDashPlan]; a pattern that
//     draws nothing over a path adds no geometry.
//   - A pattern repeated more than an internal budget allows is truncated.
//   - A sub-path whose arc length cannot be measured (non-finite coordinates) is skipped.
//
// On a closed sub-path, drawn intervals meet across the closure, so the pattern is
// evaluated continuously around the loop and a dash crossing the seam is emitted once,
// open, with caps at its real ends. A dash that covers the whole loop is emitted closed.
type StrokeDashPlan struct {
	// entries is the validated pattern in arc-length units. Every entry is finite and
	// non-negative; at least one entry is positive.
	entries []float64

	// prefix has len(entries)+1 entries, with prefix[0] == 0 and prefix[len(entries)]
	// equal to period. It maps a phase to its pattern entry with a binary search.
	prefix []float64

	// offset is the arc-length distance by which the pattern is shifted.
	// Negative values are legal and resolved with floored modular arithmetic.
	offset float64

	// period is the sum of entries, always positive and finite.
	period float64

	// hasGap reports whether at least one gap entry is positive. When it is false the
	// pattern is drawn continuously, so a dashed stroke is the solid stroke.
	hasGap bool

	// hasDraw reports whether at least one drawn entry is positive. When it is false the
	// pattern draws nothing at all.
	hasDraw bool
}

// NewStrokeDashPlan creates a [StrokeDashPlan] from a dash array and an offset.
//
// dashes alternates drawn and gap arc lengths, starting with a drawn entry. offset
// shifts the pattern against the start of every sub-path, exactly like
// stroke-dashoffset, and may be negative.
//
// NewStrokeDashPlan returns an error wrapping [ErrInvalidStrokeDashPlan] when dashes is
// empty, when it contains a negative, NaN, or infinite entry, when every entry is zero,
// when it is longer than 1024 entries, or when offset is NaN or infinite. The returned
// plan does not retain dashes, so the caller may reuse or modify the slice afterwards.
func NewStrokeDashPlan(dashes []float32, offset float32) (*StrokeDashPlan, error) {
	if !isRegular(float64(offset)) {
		return nil, newStrokeDashPlanError("offset must be finite, got %v", offset)
	}
	if len(dashes) == 0 {
		return nil, newStrokeDashPlanError("dash array must not be empty")
	}
	if len(dashes) > maxDashArrayLen {
		return nil, newStrokeDashPlanError("dash array has %d entries; the limit is %d", len(dashes), maxDashArrayLen)
	}

	entries := make([]float64, len(dashes))
	prefix := make([]float64, len(dashes)+1)
	var period float64
	var hasGap, hasDraw bool
	for i, d := range dashes {
		v := float64(d)
		if !isRegular(v) {
			return nil, newStrokeDashPlanError("dash length at index %d must be finite, got %v", i, d)
		}
		if v < 0 {
			return nil, newStrokeDashPlanError("dash length at index %d must be non-negative, got %v", i, d)
		}
		entries[i] = v
		period += v
		prefix[i+1] = period
		if v > 0 {
			if i%2 == 0 {
				hasDraw = true
			} else {
				hasGap = true
			}
		}
	}
	if !isRegular(period) || period <= 0 {
		return nil, newStrokeDashPlanError("dash array must contain at least one positive length")
	}

	return &StrokeDashPlan{
		entries: entries,
		prefix:  prefix,
		offset:  float64(offset),
		period:  period,
		hasGap:  hasGap,
		hasDraw: hasDraw,
	}, nil
}

// IsContinuous reports whether the pattern has no gap, so that a dashed stroke is
// geometrically the same as a solid stroke.
//
// Such a plan still applies its offset, but since no entry is a gap every arc length is
// drawn. [Path.AddStroke] uses this to take the solid stroke path directly.
func (p *StrokeDashPlan) IsContinuous() bool {
	return !p.hasGap
}

// DrawsNothing reports whether the pattern has no drawn entry, so that a dashed stroke
// of any path adds no geometry.
func (p *StrokeDashPlan) DrawsNothing() bool {
	return !p.hasDraw
}

// Dashes returns a copy of the plan's dash array in arc-length units.
func (p *StrokeDashPlan) Dashes() []float32 {
	dashes := make([]float32, len(p.entries))
	for i, e := range p.entries {
		dashes[i] = float32(e)
	}
	return dashes
}

// Offset returns the plan's arc-length offset.
func (p *StrokeDashPlan) Offset() float32 {
	return float32(p.offset)
}

// Period returns the arc-length length of one full repetition of the pattern.
func (p *StrokeDashPlan) Period() float32 {
	return float32(p.period)
}

// DashSegment is one drawn interval of a [StrokeDashPlan] laid over a sub-path, expressed
// in the sub-path's cumulative arc length.
//
// Parity is the index in the plan's dash array of the drawn entry the interval came from;
// it is always even.
//
// For a closed sub-path, an interval that crosses the closure is reported once, with a
// non-positive Start that counts back from the end of the sub-path. For example a dash
// crossing the seam and covering the last 20 units and the first 30 is reported as
// Start = -20 and End = 30. A crossing dash is not closed: its real ends sit in the gaps
// on either side of the seam. Only a dash that covers the whole loop is closed at both
// ends, and it is reported with Start = 0 and End = total.
type DashSegment struct {
	// Start is the arc length at which the drawn interval begins, or a non-positive
	// value counted back from the end of a closed sub-path.
	Start float32

	// End is the arc length at which the drawn interval ends.
	End float32

	// Parity is the plan dash-array index of the drawn entry, always even.
	Parity int

	// ClosedStart reports that the interval starts at the sub-path's own start.
	ClosedStart bool

	// ClosedEnd reports that the interval ends at the sub-path's own end.
	ClosedEnd bool
}

// DashSegments returns the drawn intervals the plan produces over one sub-path of the
// given total arc length, in order of increasing arc length.
//
// total is the length of the sub-path. closed must match whether that sub-path is closed;
// a closed sub-path whose drawn head and tail meet is reported as a single interval
// crossing the closure, as described on [DashSegment].
//
// DashSegments performs no curve geometry and needs no source path, so callers can inspect
// the plan's phase, parity, and closed head and tail before stroking. It is safe to call
// concurrently.
func (p *StrokeDashPlan) DashSegments(total float32, closed bool) []DashSegment {
	t := float64(total)
	if !(t > 0) || !isRegular(t) {
		return nil
	}

	var b dashBuilder
	b.collectOnIntervals(t, p)
	if len(b.on) == 0 {
		return nil
	}

	segments := make([]DashSegment, 0, len(b.on))
	first, last := b.on[0], b.on[len(b.on)-1]
	wrap := closed && first.from == 0 && last.to == t

	if wrap && len(b.on) == 1 {
		// A single dash covers the whole closed loop.
		return append(segments, DashSegment{
			Start:       0,
			End:         float32(t),
			Parity:      p.parityAt(0),
			ClosedStart: true,
			ClosedEnd:   true,
		})
	}

	start := 0
	end := len(b.on)
	if wrap {
		// The tail and head are one dash crossing the seam; report it once, counted back
		// from the end so Start <= End. It is open at both real ends, so neither closed
		// flag is set. The separate tail interval is not reported again.
		segments = append(segments, DashSegment{
			Start:  float32(last.from - t),
			End:    float32(first.to),
			Parity: p.parityAt(last.from),
		})
		start = 1
		end = len(b.on) - 1
	}
	for i := start; i < end; i++ {
		in := b.on[i]
		segments = append(segments, DashSegment{
			Start:       float32(in.from),
			End:         float32(in.to),
			Parity:      p.parityAt(in.from),
			ClosedStart: closed && !wrap && i == 0,
			ClosedEnd:   closed && !wrap && i == len(b.on)-1,
		})
	}
	return segments
}

// parityAt returns the plan dash-array index of the entry covering arc length s.
func (p *StrokeDashPlan) parityAt(s float64) int {
	idx, _ := p.patternEntryAt(p.phaseAt(s))
	return idx
}

// phaseAt returns the pattern phase at arc length s, resolved into [0, period).
// The floored modulus makes the result independent of the traversal direction for
// negative offsets and for s measured past one full period.
func (p *StrokeDashPlan) phaseAt(s float64) float64 {
	ph := math.Mod(s-p.offset, p.period)
	if ph < 0 {
		ph += p.period
	}
	return ph
}

// patternEntryAt maps a phase to its pattern index and reports whether that entry is drawn.
// A phase that lands exactly on an entry boundary belongs to the following entry, as with
// half-open intervals; zero-length entries are skipped the same way.
func (p *StrokeDashPlan) patternEntryAt(phase float64) (index int, on bool) {
	// Find the first entry whose end is beyond phase.
	lo, hi := 0, len(p.entries)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if phase < p.prefix[mid+1] {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	if lo >= len(p.entries) {
		// Only rounding can reach here; the last entry owns the remainder.
		lo = len(p.entries) - 1
	}
	return lo, lo%2 == 0
}

// dashInterval is a drawn arc-length range measured against the total length of a
// normalized sub-path.
type dashInterval struct {
	from, to float64
}

// dashBuilder accumulates the dashed geometry of one source path. Its buffers are reused
// across sub-paths and, through dashBuilderPool, across [Path.AddStroke] calls.
type dashBuilder struct {
	lengths []float64
	cum     []float64
	on      []dashInterval
	scratch subPath
}

// dashBuilderPool reuses dash builders so that repeated dashed strokes do not allocate.
// A builder holds no Path, so it can be pooled; the dashed source path itself is owned by
// the caller and dropped after the stroke is built.
var dashBuilderPool = sync.Pool{
	New: func() any {
		return &dashBuilder{}
	},
}

// buildDashedPath appends the dashed sub-paths of src to dst.
//
// Every source sub-path is normalized first, so dashing only ever sees non-degenerate
// lines and quadratic curves (cusps become pairs of lines). Invalid sub-paths and
// zero-length sub-paths produce no output, which is a bounded degeneration.
func (b *dashBuilder) buildDashedPath(dst *Path, src *Path, plan *StrokeDashPlan) {
	for i := range src.subPaths {
		normalizeSubPath(&b.scratch, &src.subPaths[i])
		if len(b.scratch.ops) == 0 || b.scratch.invalid {
			continue
		}
		b.appendDashedSubPath(dst, &b.scratch, plan)
	}
}

// appendDashedSubPath computes the arc-length table of one normalized sub-path, walks the
// dash pattern over it, and materializes the drawn intervals.
func (b *dashBuilder) appendDashedSubPath(dst *Path, sp *subPath, plan *StrokeDashPlan) {
	n := len(sp.ops)
	b.lengths = growFloat64s(b.lengths, n)[:n]
	b.cum = growFloat64s(b.cum, n+1)[:n+1]

	var total float64
	cur := sp.start
	for i, op := range sp.ops {
		b.cum[i] = total
		var l float64
		switch op.typ {
		case opTypeLineTo:
			l = lineLength(cur, op.p1)
		case opTypeQuadTo:
			l = quadArcLength(cur, op.p1, op.p2)
		}
		if !isRegular(l) || l < 0 {
			// A length that cannot be placed in arc-length space makes the whole
			// sub-path unmeasurable; skip it as a bounded degeneration.
			return
		}
		if l == 0 {
			// Normalization keeps a cusp as a pair of lines, so a quadratic can only
			// collapse when tiny curves meet at a join. It contributes no arc length and
			// must not be selectable as an operation to emit.
			b.lengths[i] = 0
			cur = sp.endAtOp(i)
			continue
		}
		b.lengths[i] = l
		total += l
		cur = sp.endAtOp(i)
	}
	b.cum[n] = total
	if !isRegular(total) || total <= 0 {
		return
	}

	b.on = b.on[:0]
	b.collectOnIntervals(total, plan)
	if len(b.on) == 0 {
		return
	}

	// For a closed sub-path, the dash touching the start and the dash touching the end
	// are two halves of one dash that crosses the closure: the pattern continues from the
	// end of the sub-path back to its start. Emitting them as one sub-path puts caps only
	// at the dash's real ends instead of at the seam, which has no gap.
	start, end := 0, len(b.on)
	if sp.closed && b.on[0].from == 0 && b.on[len(b.on)-1].to == total {
		if len(b.on) == 1 {
			// The dash covers the whole loop, so it is a genuinely closed dash.
			b.emitRange(dst, sp, 0, total, false)
			dst.Close()
			return
		}
		b.emitRange(dst, sp, b.on[len(b.on)-1].from, total, false)
		b.emitRange(dst, sp, 0, b.on[0].to, true)
		start, end = 1, len(b.on)-1
	}
	for _, in := range b.on[start:end] {
		b.emitRange(dst, sp, in.from, in.to, false)
	}
}

// collectOnIntervals walks the periodic pattern over [0, total] and records the drawn
// ranges. The walk advances by pattern transitions, so the iteration count is bounded by
// the number of transitions actually crossed. Both the number of steps and the number of
// recorded intervals are capped, which truncates a pathologically dense pattern instead
// of subdividing without limit.
func (b *dashBuilder) collectOnIntervals(total float64, plan *StrokeDashPlan) {
	s := 0.0
	steps := 0
	for s < total {
		phase := plan.phaseAt(s)
		idx, on := plan.patternEntryAt(phase)

		next := s + (plan.prefix[idx+1] - phase)
		if !isRegular(next) || next > total {
			next = total
		}
		if next <= s {
			// A subnormal remaining length cannot advance s by a representable step.
			// Move one ULP forward so the walk terminates.
			next = math.Nextafter(s, total)
		}
		if on && next > s {
			// Coalesce touching drawn intervals. This happens when the pattern turns a
			// zero-length gap into nothing, and it keeps a fully drawn path as a single
			// sub-path instead of a chain of caps.
			if len(b.on) > 0 && b.on[len(b.on)-1].to == s {
				b.on[len(b.on)-1].to = next
			} else {
				b.on = append(b.on, dashInterval{from: s, to: next})
			}
		}
		s = next
		steps++
		if steps >= maxDashSteps || len(b.on) >= maxDashIntervals {
			// Bounded degeneration for a tiny pattern on a long path.
			return
		}
	}
}

// emitRange appends the geometry of the global arc-length range [from, to].
// When connect is true it extends the current sub-path (the second half of a dash that
// wraps across a closure); otherwise it starts a new sub-path at the range start.
func (b *dashBuilder) emitRange(dst *Path, sp *subPath, from, to float64, connect bool) {
	if !(to > from) {
		return
	}

	first := locateOp(b.cum, from)
	last := locateOp(b.cum, to)

	started := false
	for i := first; i <= last; i++ {
		op := sp.ops[i]
		opStart := b.cum[i]
		opLen := b.lengths[i]
		localFrom := max(from-opStart, 0)
		localTo := min(to-opStart, opLen)
		if localTo <= localFrom {
			continue
		}
		p0 := sp.startAtOp(i)

		switch op.typ {
		case opTypeLineTo:
			// Use the original endpoints for whole-argument pieces so that an undashed
			// path reproduces the solid outline bit for bit.
			a := p0
			if localFrom > 0 {
				a = pointAtLineLength(p0, op.p1, opLen, localFrom)
			}
			if !started {
				if connect {
					// A wrapping dash resumes from its first half; emit a point so the
					// joining line is explicit. AddStroke drops a no-op zero-length line.
					dst.lineTo(a.x, a.y)
				} else {
					dst.moveTo(a.x, a.y)
				}
				started = true
			} else if localFrom > 0 {
				dst.lineTo(a.x, a.y)
			}
			c := op.p1
			if localTo < opLen {
				c = pointAtLineLength(p0, op.p1, opLen, localTo)
			}
			dst.lineTo(c.x, c.y)
		case opTypeQuadTo:
			t0 := 0.0
			if localFrom > 0 {
				t0 = quadTForLength(p0, op.p1, op.p2, opLen, localFrom)
			}
			t1 := 1.0
			if localTo < opLen {
				t1 = quadTForLength(p0, op.p1, op.p2, opLen, localTo)
			}
			if !(t1 > t0) {
				// Inversion is monotone, so this only happens at rounding limits; skipping
				// keeps the emitted sub-curve non-degenerate.
				continue
			}
			q0, q1, q2 := subQuad(p0, op.p1, op.p2, t0, t1)
			if !started {
				if connect {
					dst.lineTo(q0.x, q0.y)
				} else {
					dst.moveTo(q0.x, q0.y)
				}
				started = true
			} else if t0 > 0 {
				dst.lineTo(q0.x, q0.y)
			}
			dst.quadTo(q1.x, q1.y, q2.x, q2.y)
		}
	}
}

// locateOp returns the index of the operation whose cumulative length range contains s.
// cum has n+1 entries with cum[n] the total length. Zero-length operations share their
// cumulative value with the next operation, so the search skips them; the result is
// always the first operation with a positive remaining length, in [0, n-1].
func locateOp(cum []float64, s float64) int {
	// Binary search: largest i with cum[i] <= s.
	lo, hi := 0, len(cum)-2
	for lo < hi {
		mid := int(uint(lo+hi+1) >> 1)
		if cum[mid] <= s {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	// Skip zero-length operations at the boundary so that emitting starts on an
	// operation that can actually contribute geometry.
	for lo < len(cum)-2 && cum[lo] == cum[lo+1] && s >= cum[lo+1] {
		lo++
	}
	return lo
}

func growFloat64s(s []float64, n int) []float64 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float64, n)
}

// --- Arc-length geometry -----------------------------------------------------

func lineLength(p0, p1 point) float64 {
	return math.Hypot(p1.x-p0.x, p1.y-p0.y)
}

func pointAtLineLength(p0, p1 point, length, s float64) point {
	if length == 0 {
		return p0
	}
	u := s / length
	return point{
		x: p0.x + (p1.x-p0.x)*u,
		y: p0.y + (p1.y-p0.y)*u,
	}
}

// gaussLegendre8 holds the 8-point Gauss-Legendre nodes and weights on [-1, 1].
// A single panel integrates polynomials up to degree 15 exactly; adaptive subdivision
// recovers high accuracy for the square-root speed of curved paths.
var gaussLegendre8 = struct {
	nodes   [8]float64
	weights [8]float64
}{
	nodes: [8]float64{
		-0.9602898564975363,
		-0.7966664774136267,
		-0.5255324099163290,
		-0.1834346424956498,
		0.1834346424956498,
		0.5255324099163290,
		0.7966664774136267,
		0.9602898564975363,
	},
	weights: [8]float64{
		0.1012285362903763,
		0.2223810344533745,
		0.3137066458778873,
		0.3626837833783620,
		0.3626837833783620,
		0.3137066458778873,
		0.2223810344533745,
		0.1012285362903763,
	},
}

// quadSpeed returns |B'(t)| for the quadratic curve with start p0, control p1, end p2.
func quadSpeed(p0, p1, p2 point, t float64) float64 {
	// B'(t) = 2(1-t)(p1-p0) + 2t(p2-p1)
	dx := 2 * ((1-t)*(p1.x-p0.x) + t*(p2.x-p1.x))
	dy := 2 * ((1-t)*(p1.y-p0.y) + t*(p2.y-p1.y))
	return math.Hypot(dx, dy)
}

// gl8Panel integrates the quadratic speed over t in [a, b] with one Gauss-Legendre panel.
func gl8Panel(p0, p1, p2 point, a, b float64) float64 {
	mid, half := (a+b)/2, (b-a)/2
	var sum float64
	for i := range gaussLegendre8.nodes {
		t := mid + half*gaussLegendre8.nodes[i]
		sum += gaussLegendre8.weights[i] * quadSpeed(p0, p1, p2, t)
	}
	return half * sum
}

// quadArcLengthBetween integrates the quadratic speed over t in [a, b].
//
// It uses an explicit stack instead of recursion so that a nearly cusped curve cannot
// exhaust the goroutine stack, and a panel budget so that a concentrated integrand
// cannot consume unbounded time. When the budget is spent the current estimate is
// accepted; the error is bounded and the result is finite.
func quadArcLengthBetween(p0, p1, p2 point, a, b float64) float64 {
	if !(b > a) {
		return 0
	}
	type panel struct {
		a, b  float64
		whole float64
		depth int
	}
	whole := gl8Panel(p0, p1, p2, a, b)
	if !isRegular(whole) {
		return math.Inf(1)
	}
	stack := []panel{{a: a, b: b, whole: whole}}
	sum := 0.0
	budget := maxArcLengthPanels
	for len(stack) > 0 {
		fr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if fr.depth >= arcLengthDepth || budget <= 0 {
			sum += fr.whole
			continue
		}
		budget--

		mid := (fr.a + fr.b) / 2
		left := gl8Panel(p0, p1, p2, fr.a, mid)
		right := gl8Panel(p0, p1, p2, mid, fr.b)
		children := left + right
		if !isRegular(children) {
			return math.Inf(1)
		}
		if math.Abs(fr.whole-children) <= arcLengthRelTol*math.Abs(children) {
			// The children use twice as many nodes, so they are the more accurate estimate.
			sum += children
			continue
		}
		stack = append(stack,
			panel{a: mid, b: fr.b, whole: right, depth: fr.depth + 1},
			panel{a: fr.a, b: mid, whole: left, depth: fr.depth + 1},
		)
	}
	return sum
}

// quadArcLength integrates the quadratic curve length over t in [0, 1].
func quadArcLength(p0, p1, p2 point) float64 {
	return quadArcLengthBetween(p0, p1, p2, 0, 1)
}

// quadTForLength inverts the arc-length function of a quadratic curve: it returns the
// parameter t at which the curve has covered target of its total length.
//
// The iteration is a safeguarded Newton method: each step keeps a bracket around the
// solution and falls back to bisection whenever the Newton step leaves it or the speed is
// unusable. The bracket and the iteration count guarantee termination, and the method
// converges for nearly straight curves (tiny curvature) as well as for curve ends.
func quadTForLength(p0, p1, p2 point, length, target float64) float64 {
	if target <= 0 {
		return 0
	}
	if target >= length {
		return 1
	}

	lo, hi := 0.0, 1.0
	t := min(max(target/length, 0), 1)
	for range dashInverseIterations {
		f := quadArcLengthBetween(p0, p1, p2, 0, t) - target
		if !isRegular(f) {
			// Fall back to bisection for the rest of the budget.
			f = 0
		}
		if f > 0 {
			hi = t
		} else {
			lo = t
		}
		if hi-lo <= dashInverseEpsilon {
			break
		}
		next := t
		if d := quadSpeed(p0, p1, p2, t); d > 0 && isRegular(d) {
			next = t - f/d
		}
		if !isRegular(next) || !(next > lo && next < hi) {
			next = (lo + hi) / 2
		}
		if next == t {
			break
		}
		t = next
	}
	return (lo + hi) / 2
}

// quadAt evaluates a quadratic Bézier curve at parameter t.
func quadAt(p0, p1, p2 point, t float64) point {
	u := 1 - t
	return point{
		x: u*u*p0.x + 2*u*t*p1.x + t*t*p2.x,
		y: u*u*p0.y + 2*u*t*p1.y + t*t*p2.y,
	}
}

// subQuad returns the control points of the portion of a quadratic curve covering
// t in [t0, t1], reparameterized to [0, 1]. Whole-interval requests return the original
// points to avoid needless rounding.
func subQuad(p0, p1, p2 point, t0, t1 float64) (q0, q1, q2 point) {
	if t0 == 0 && t1 == 1 {
		return p0, p1, p2
	}
	if t0 == t1 {
		q := quadAt(p0, p1, p2, t0)
		return q, q, q
	}
	// Split at t0 by De Casteljau's algorithm; the right piece is (a012, a12, p2).
	a01 := lerpPoint(p0, p1, t0)
	a12 := lerpPoint(p1, p2, t0)
	a012 := lerpPoint(a01, a12, t0)

	// Split the right piece at the normalized parameter u = (t1-t0)/(1-t0).
	rp0, rp1, rp2 := a012, a12, p2
	u := (t1 - t0) / (1 - t0)
	b01 := lerpPoint(rp0, rp1, u)
	b12 := lerpPoint(rp1, rp2, u)
	b012 := lerpPoint(b01, b12, u)
	return rp0, b01, b012
}

func lerpPoint(a, b point, t float64) point {
	return point{
		x: a.x + (b.x-a.x)*t,
		y: a.y + (b.y-a.y)*t,
	}
}
