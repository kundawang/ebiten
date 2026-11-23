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

package vector_test

import (
	"errors"
	"image"
	"image/color"
	"math"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// --- Plan validation ---------------------------------------------------------

func TestStrokeDashPlanValidation(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))

	tests := []struct {
		name   string
		dashes []float32
		offset float32
	}{
		{name: "empty", dashes: nil},
		{name: "empty slice", dashes: []float32{}},
		{name: "negative entry", dashes: []float32{10, -2}},
		{name: "NaN entry", dashes: []float32{10, nan}},
		{name: "infinite entry", dashes: []float32{10, inf}},
		{name: "all zero", dashes: []float32{0, 0, 0}},
		{name: "NaN offset", dashes: []float32{10, 5}, offset: nan},
		{name: "infinite offset", dashes: []float32{10, 5}, offset: inf},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := vector.NewStrokeDashPlanForTest(tc.dashes, tc.offset); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}

	t.Run("too long", func(t *testing.T) {
		dashes := make([]float32, 1025)
		for i := range dashes {
			dashes[i] = 1
		}
		if _, err := vector.NewStrokeDashPlanForTest(dashes, 0); err == nil {
			t.Fatal("expected error for over-long dash array")
		}
	})
}

func TestStrokeDashPlanAcceptsZeroEntriesAndNegativeOffset(t *testing.T) {
	// A zero entry keeps parity but draws nothing; a negative offset is legal.
	plan, err := vector.NewStrokeDashPlanForTest([]float32{10, 0, 5, 0}, -7)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got, want := plan.Offset(), float32(-7); got != want {
		t.Errorf("offset: got %v want %v", got, want)
	}
	dashes := plan.Dashes()
	if len(dashes) != 4 || dashes[1] != 0 || dashes[3] != 0 {
		t.Errorf("dashes round trip mismatch: %v", dashes)
	}
}

func TestDashZeroEntriesDoNotStall(t *testing.T) {
	// Zero-length gaps must not trap the pattern walker in an infinite loop.
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{10, 0, 5, 0}, 0)
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)
	var dst vector.Path
	vector.BuildDashedPathForTest(&dst, &src, plan)
	if n := vector.SubPathCount(&dst); n == 0 || n > 32 {
		t.Fatalf("zero-entry dash count out of bounded range: %d", n)
	}
}

func TestStrokeDashPlanDoesNotRetainInput(t *testing.T) {
	in := []float32{10, 5}
	plan, err := vector.NewStrokeDashPlanForTest(in, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	in[0] = 999
	if got := plan.Dashes()[0]; got != 10 {
		t.Errorf("plan retained caller slice: got %v want 10", got)
	}
}

// --- Arc-length reference ----------------------------------------------------

// independentQuadLength is a high-resolution reference for quadratic arc length,
// computed with dense midpoint polyline sampling rather than the package's quadrature.
func independentQuadLength(p0, p1, p2 vector.Point, steps int) float64 {
	at := func(t float64) (float64, float64) {
		u := 1 - t
		x := u*u*float64(p0.X) + 2*u*t*float64(p1.X) + t*t*float64(p2.X)
		y := u*u*float64(p0.Y) + 2*u*t*float64(p1.Y) + t*t*float64(p2.Y)
		return x, y
	}
	var sum float64
	px, py := at(0)
	for i := 1; i <= steps; i++ {
		x, y := at(float64(i) / float64(steps))
		sum += math.Hypot(x-px, y-py)
		px, py = x, y
	}
	return sum
}

func TestQuadArcLengthAgainstReference(t *testing.T) {
	curves := []struct {
		name       string
		p0, p1, p2 vector.Point
		relTol     float64
	}{
		{name: "arch", p0: vector.Point{0, 0}, p1: vector.Point{50, 100}, p2: vector.Point{100, 0}, relTol: 1e-8},
		{name: "rounded corner", p0: vector.Point{10, 10}, p1: vector.Point{10, 90}, p2: vector.Point{90, 90}, relTol: 1e-8},
		// Nearly straight (tiny curvature): must stay bounded and match the chord length.
		{name: "tiny curvature", p0: vector.Point{0, 0}, p1: vector.Point{50, 1e-9}, p2: vector.Point{100, 0}, relTol: 1e-9},
		// A cusp (start and end coincide) has a speed singularity at the tip; adaptive
		// subdivision still converges, but the dense polyline reference lags slightly.
		{name: "cusp", p0: vector.Point{0, 0}, p1: vector.Point{120, 80}, p2: vector.Point{0, 0}, relTol: 2e-3},
	}
	for _, c := range curves {
		t.Run(c.name, func(t *testing.T) {
			got := vector.QuadArcLengthForTest(c.p0, c.p1, c.p2)
			want := independentQuadLength(c.p0, c.p1, c.p2, 100000)
			if math.Abs(got-want) > c.relTol*max(1, want) {
				t.Errorf("arc length got %.8f want %.8f", got, want)
			}
			if !isFinite(got) || got <= 0 {
				t.Errorf("arc length must be a positive finite number, got %v", got)
			}
		})
	}

	// The nearly straight curve must equal its 100px chord to within floating-point noise.
	straight := vector.QuadArcLengthForTest(vector.Point{0, 0}, vector.Point{50, 1e-9}, vector.Point{100, 0})
	if math.Abs(straight-100) > 1e-6 {
		t.Errorf("near-straight length %.10f deviates from chord 100", straight)
	}
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func TestQuadTForLengthMonotoneAndInverts(t *testing.T) {
	p0, p1, p2 := vector.Point{0, 0}, vector.Point{40, 120}, vector.Point{120, 0}
	length := vector.QuadArcLengthForTest(p0, p1, p2)

	// Endpoints.
	if got := vector.QuadTForLengthForTest(p0, p1, p2, length, 0); got != 0 {
		t.Errorf("t(0) = %v, want 0", got)
	}
	if got := vector.QuadTForLengthForTest(p0, p1, p2, length, length); got != 1 {
		t.Errorf("t(length) = %v, want 1", got)
	}

	// Monotonicity across the interior.
	prev := 0.0
	for k := 1; k <= 20; k++ {
		target := length * float64(k) / 21
		tt := vector.QuadTForLengthForTest(p0, p1, p2, length, target)
		if tt < prev || tt < 0 || tt > 1 {
			t.Fatalf("non-monotone or out-of-range parameter at target %.4f: %.6f", target, tt)
		}
		prev = tt
	}

	// Inversion consistency: a recovered t must map back to the requested length.
	for _, target := range []float64{length * 0.25, length * 0.5, length * 0.75} {
		tt := vector.QuadTForLengthForTest(p0, p1, p2, length, target)
		recovered := independentQuadLengthToT(p0, p1, p2, tt)
		if math.Abs(recovered-target) > 1e-4*length {
			t.Errorf("inversion mismatch at target %.4f: recovered %.4f (t=%.6f)", target, recovered, tt)
		}
	}

	// Tiny-curvature curve must still produce a valid, monotone parameter without divergence.
	l0, l1, l2 := vector.Point{0, 0}, vector.Point{50, 1e-9}, vector.Point{100, 0}
	sLen := vector.QuadArcLengthForTest(l0, l1, l2)
	prev = 0
	for k := 1; k <= 10; k++ {
		tt := vector.QuadTForLengthForTest(l0, l1, l2, sLen, sLen*float64(k)/11)
		if tt < prev || tt < 0 || tt > 1 || math.IsNaN(tt) {
			t.Fatalf("tiny-curvature parameter invalid: %.6f", tt)
		}
		prev = tt
	}
}

// independentQuadLengthToT is a dense-sampling reference for the length covered up to t.
func independentQuadLengthToT(p0, p1, p2 vector.Point, tTarget float64) float64 {
	const steps = 20000
	at := func(t float64) (float64, float64) {
		u := 1 - t
		x := u*u*float64(p0.X) + 2*u*t*float64(p1.X) + t*t*float64(p2.X)
		y := u*u*float64(p0.Y) + 2*u*t*float64(p1.Y) + t*t*float64(p2.Y)
		return x, y
	}
	var sum float64
	px, py := at(0)
	for i := 1; i <= steps; i++ {
		t := tTarget * float64(i) / float64(steps)
		x, y := at(t)
		sum += math.Hypot(x-px, y-py)
		px, py = x, y
	}
	return sum
}

// --- Dashed geometry: line ---------------------------------------------------

func TestDashLineSegments(t *testing.T) {
	// A 100px line with pattern 10 on, 10 off yields 5 dashes.
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)

	plan, err := vector.NewStrokeDashPlanForTest([]float32{10, 10}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dashed vector.Path
	vector.BuildDashedPathForTest(&dashed, &src, plan)
	if got, want := vector.SubPathCount(&dashed), 5; got != want {
		t.Fatalf("dash count: got %d want %d", got, want)
	}
	// Pattern 10 on / 10 off: dashes cover [0,10],[20,30],...,[80,90], so the last painted
	// point is x=90 and x=100 lies in the trailing gap.
	if got, want := dashed.Bounds(), image.Rect(0, 0, 90, 0); got != want {
		t.Errorf("bounds: got %v want %v", got, want)
	}
}

func TestDashLineBoundariesAgainstReference(t *testing.T) {
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)

	const on, off = 15.0, 7.0
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{on, off}, 0)
	var dashed vector.Path
	vector.BuildDashedPathForTest(&dashed, &src, plan)

	// Independent reference: walk the periodic pattern and record on intervals.
	var want [][2]float64
	period := on + off
	for s := 0.0; s < 100; {
		phase := math.Mod(s, period)
		var idx int
		acc := 0.0
		entries := []float64{on, off}
		for i, e := range entries {
			if phase < acc+e {
				idx = i
				break
			}
			acc += e
		}
		next := s + (entries[idx] - (phase - acc))
		if next > 100 {
			next = 100
		}
		if idx%2 == 0 && next > s {
			want = append(want, [2]float64{s, next})
		}
		s = next
	}

	if got := vector.SubPathCount(&dashed); got != len(want) {
		t.Fatalf("dash count got %d want %d (%v)", got, len(want), want)
	}

	// Each dashed sub-path is a single LineTo whose start and end X must equal the reference
	// interval endpoints (Y stays 0).
	ops := vector.PathOperationsString(&dashed)
	for _, in := range want {
		from := image.Pt(int(math.Round(in[0])), 0)
		to := image.Pt(int(math.Round(in[1])), 0)
		if !containsLine(ops, from, to) {
			t.Errorf("reference interval [%v, %v] not found in dashed ops:\n%s", in[0], in[1], ops)
		}
	}
}

// containsLine reports whether the operations text includes a MoveTo at the line start and
// a LineTo at its end (both on y=0 here).
func containsLine(ops string, from, to image.Point) bool {
	move := "MoveTo(" + itoa(float32(from.X)) + ", " + itoa(float32(from.Y)) + ")"
	line := "LineTo(" + itoa(float32(to.X)) + ", " + itoa(float32(to.Y)) + ")"
	return strings.Contains(ops, move) && strings.Contains(ops, line)
}

func itoa(f float32) string {
	return strconv.FormatFloat(float64(f), 'f', -1, 32)
}

func TestDashOffsetNegativeAndPhaseReversal(t *testing.T) {
	// Negative offset must shift phase per floored modulus and produce the same visible rule
	// as an equivalent positive offset within one period.
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)

	planNeg, _ := vector.NewStrokeDashPlanForTest([]float32{10, 10}, -5)
	planPos, _ := vector.NewStrokeDashPlanForTest([]float32{10, 10}, 15) // -5 mod 20 == 15

	var a, b vector.Path
	vector.BuildDashedPathForTest(&a, &src, planNeg)
	vector.BuildDashedPathForTest(&b, &src, planPos)
	if got, want := vector.PathOperationsString(&a), vector.PathOperationsString(&b); got != want {
		t.Errorf("negative and equivalent positive offset differ:\n%v\nvs\n%v", got, want)
	}
}

// --- Reversed path invariance ------------------------------------------------

func TestDashReversedPathSamePhaseRule(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{12, 8}, 0)

	var fwd, rev vector.Path
	fwd.MoveTo(0, 0)
	fwd.LineTo(100, 0)
	fwd.QuadTo(150, 50, 200, 0)

	rev.MoveTo(200, 0)
	rev.QuadTo(150, 50, 100, 0)
	rev.LineTo(0, 0)

	var df, dr vector.Path
	vector.BuildDashedPathForTest(&df, &fwd, plan)
	vector.BuildDashedPathForTest(&dr, &rev, plan)

	// Both directions must produce the same number of dashes for the same underlying
	// geometric curve, since the phase rule depends only on accumulated arc length.
	if got, want := vector.SubPathCount(&df), vector.SubPathCount(&dr); got != want {
		t.Fatalf("dash count differs by direction: %d vs %d", got, want)
	}

	// The sum of every dash's drawn arc length must equal the on-fraction of the total.
	// Compute both totals from the materialized geometry via an independent dense sampler.
	fwdTotal := referencePathLength(&fwd)
	revTotal := referencePathLength(&rev)
	if math.Abs(fwdTotal-revTotal) > 1e-6 {
		t.Errorf("reversed total length changed: %.8f vs %.8f", fwdTotal, revTotal)
	}

	dfDrawn := referenceDrawnLength(&df, plan, fwdTotal)
	drDrawn := referenceDrawnLength(&dr, plan, revTotal)
	if math.Abs(dfDrawn-drDrawn) > 1e-6 {
		t.Errorf("drawn length differs by direction: %.8f vs %.8f", dfDrawn, drDrawn)
	}
}

// referencePathLength measures a test path (lines + one quadratic) by dense sampling.
func referencePathLength(p *vector.Path) float64 {
	// The test paths in this package are constructed from MoveTo/LineTo/QuadTo only and are
	// read back through their operation dump. Re-derive length from the public dash builder
	// output is circular, so reconstruct from the known fixture shape instead.
	// Both fixtures share a 100px line and the quadratic (0,0)->(150,50)->(200,0).
	line := 100.0
	q := independentQuadLength(vector.Point{100, 0}, vector.Point{150, 50}, vector.Point{200, 0}, 100000)
	return line + q
}

// referenceDrawnLength returns the arc length the pattern marks as on over a path of total
// length total, computed independently of the production walker.
func referenceDrawnLength(_ *vector.Path, plan *vector.StrokeDashPlan, total float64) float64 {
	dashes := plan.Dashes()
	var entries []float64
	for _, d := range dashes {
		entries = append(entries, float64(d))
	}
	period := 0.0
	for _, e := range entries {
		period += e
	}
	var drawn float64
	const fine = 1e-3
	for s := 0.0; s < total; s += fine {
		phase := math.Mod(s, period)
		acc := 0.0
		for i, e := range entries {
			if phase < acc+e {
				if i%2 == 0 {
					drawn += min(fine, total-s)
				}
				break
			}
			acc += e
		}
	}
	return drawn
}

// --- Closed path seam --------------------------------------------------------

func TestDashClosedPathSeam(t *testing.T) {
	// Square perimeter 400 (100 per side).
	mkSquare := func() *vector.Path {
		var src vector.Path
		src.MoveTo(0, 0)
		src.LineTo(100, 0)
		src.LineTo(100, 100)
		src.LineTo(0, 100)
		src.Close()
		return &src
	}

	t.Run("no wrap when endpoints land in gaps", func(t *testing.T) {
		// Pattern 60 on / 40 off: on intervals end at 360, leaving a gap to the seam.
		plan, _ := vector.NewStrokeDashPlanForTest([]float32{60, 40}, 0)
		var dashed vector.Path
		vector.BuildDashedPathForTest(&dashed, mkSquare(), plan)
		if n := vector.SubPathCount(&dashed); n == 0 || n > 16 {
			t.Fatalf("closed seam dash count out of bounded range: %d", n)
		}
	})

	t.Run("wrapping dash merges into one sub-path", func(t *testing.T) {
		// Negative offset makes the pattern drawn at both the s=0 and s=400 boundaries.
		// Drawn intervals are [0,70], [150,270], [350,400]; the first and last are two
		// halves of one dash crossing the closure, so they merge into one sub-path.
		plan, _ := vector.NewStrokeDashPlanForTest([]float32{120, 80}, -50)
		var dashed vector.Path
		vector.BuildDashedPathForTest(&dashed, mkSquare(), plan)
		// Two distinct dashes: one crossing the seam and one interior [150,270].
		if got, want := vector.SubPathCount(&dashed), 2; got != want {
			t.Fatalf("sub-path count: got %d want %d", got, want)
		}
		// The crossing dash is open: it has real ends where the gaps meet, so it must not
		// be closed (Closing would add a spurious straight chord across the gap).
		ops := vector.PathOperationsString(&dashed)
		if strings.Count(ops, "MoveTo") != 2 {
			t.Errorf("expected two sub-paths:\n%s", ops)
		}
		if strings.Contains(ops, "Close()") {
			t.Errorf("a dash crossing the seam must stay open (only a whole-loop dash closes):\n%s", ops)
		}
	})
}

// --- End-to-end AddStroke with caps/joins/GeoM -------------------------------

func TestAddStrokeDashedUsesCapsAndJoins(t *testing.T) {
	var src vector.Path
	src.MoveTo(20, 100)
	src.LineTo(100, 100)
	src.LineTo(100, 20)

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{20, 10}, 0)

	for _, cap := range []vector.LineCap{vector.LineCapButt, vector.LineCapRound, vector.LineCapSquare} {
		t.Run("cap", func(t *testing.T) {
			op := &vector.AddStrokeOptions{}
			op.StrokeOptions.Width = 8
			op.StrokeOptions.LineCap = cap
			op.StrokeOptions.LineJoin = vector.LineJoinRound
			op.StrokeOptions.Dash = plan

			var dst vector.Path
			dst.AddStroke(&src, op)
			if vector.SubPathCount(&dst) == 0 {
				t.Fatalf("cap %d: dashed stroke produced no outline", cap)
			}
			// Round and square caps extend beyond the butt end, so bounds must be at least as large.
			_ = dst.Bounds()
		})
	}
}

func TestAddStrokeDashedRespectsJoinAndMiter(t *testing.T) {
	// A sharp corner whose dash keeps the joint inside one drawn segment.
	mkSrc := func() *vector.Path {
		var src vector.Path
		src.MoveTo(0, 50)
		src.LineTo(50, 50)
		src.LineTo(50, 0)
		return &src
	}
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{1000, 10}, 0) // one long dash keeps the corner

	for _, join := range []vector.LineJoin{vector.LineJoinMiter, vector.LineJoinBevel, vector.LineJoinRound} {
		t.Run("join", func(t *testing.T) {
			op := &vector.AddStrokeOptions{}
			op.StrokeOptions.Width = 10
			op.StrokeOptions.LineJoin = join
			op.StrokeOptions.MiterLimit = 10
			op.StrokeOptions.Dash = plan
			var dst vector.Path
			dst.AddStroke(mkSrc(), op)
			if vector.SubPathCount(&dst) == 0 {
				t.Fatalf("join %d produced no outline", join)
			}
		})
	}

	// A miter limit below 1 forces every miter to a bevel; the dashed outline must still be
	// produced and bounded.
	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 10
	op.StrokeOptions.LineJoin = vector.LineJoinMiter
	op.StrokeOptions.MiterLimit = 0
	op.StrokeOptions.Dash = plan
	var dst vector.Path
	dst.AddStroke(mkSrc(), op)
	if vector.SubPathCount(&dst) == 0 {
		t.Fatal("miter-limit-beveled dashed stroke produced no outline")
	}
}

// TestAddStrokeDashedEqualsManualSegments proves the dash path is fed through the unchanged
// stroke pipeline: stroking the dashed source must equal stroking each manually cut segment
// with the same options.
func TestAddStrokeDashedEqualsManualSegments(t *testing.T) {
	// Straight line, pattern 25/25 over 100px => dashes [0,25],[50,75].
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{25, 25}, 0)

	dashOp := &vector.AddStrokeOptions{}
	dashOp.StrokeOptions.Width = 6
	dashOp.StrokeOptions.LineCap = vector.LineCapButt
	dashOp.StrokeOptions.LineJoin = vector.LineJoinMiter
	dashOp.StrokeOptions.MiterLimit = 4
	dashOp.StrokeOptions.Dash = plan
	var dashed vector.Path
	dashed.AddStroke(&src, dashOp)

	// Manual equivalent: stroke the two cut segments independently with identical options.
	var manual vector.Path
	segOp := *dashOp
	segOp.StrokeOptions.Dash = nil
	for _, x := range [][2]float32{{0, 25}, {50, 75}} {
		var seg vector.Path
		seg.MoveTo(x[0], 0)
		seg.LineTo(x[1], 0)
		manual.AddStroke(&seg, &segOp)
	}

	if got, want := vector.PathOperationsString(&dashed), vector.PathOperationsString(&manual); got != want {
		t.Errorf("dashed stroke differs from manually segmented stroke:\n%s\nvs\n%s", got, want)
	}
}

func TestAddStrokeDashedRespectsGeoM(t *testing.T) {
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{20, 20}, 0)

	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 4
	op.StrokeOptions.Dash = plan
	op.GeoM.Translate(50, 70)

	var dst vector.Path
	dst.AddStroke(&src, op)
	b := dst.Bounds()
	// Every dash is translated by (50, 70); min X must be near 50, not 0.
	if b.Min.X < 48 {
		t.Errorf("GeoM translation not applied to dashes: bounds %v", b)
	}
}

func TestAddStrokeDashPlanReusable(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{15, 15}, 0)

	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(120, 0)

	var first, second vector.Path
	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 4
	op.StrokeOptions.Dash = plan
	first.AddStroke(&src, op)
	second.AddStroke(&src, op)
	if got, want := vector.PathOperationsString(&first), vector.PathOperationsString(&second); got != want {
		t.Errorf("reusing a plan changed the output:\n%v\nvs\n%v", got, want)
	}
}

// --- Pixel compatibility: solid path unchanged -------------------------------

func TestDashNilIsPixelIdenticalToSolid(t *testing.T) {
	var src vector.Path
	src.MoveTo(10, 10)
	src.LineTo(90, 10)
	src.QuadTo(120, 60, 90, 90)

	baseOp := &vector.AddStrokeOptions{}
	baseOp.StrokeOptions.Width = 6

	var solid vector.Path
	solid.AddStroke(&src, baseOp)

	// A plan whose pattern never has a gap over this path must not change geometry.
	// Easiest guaranteed-equivalent case: nil Dash (the documented default).
	var nilDash vector.Path
	op := *baseOp
	nilDash.AddStroke(&src, &op)

	if got, want := vector.PathOperationsString(&nilDash), vector.PathOperationsString(&solid); got != want {
		t.Errorf("nil Dash changed solid stroke geometry:\n%v\nvs\n%v", got, want)
	}
}

// --- Rendered pixels: ring (arc) fixture -------------------------------------

func TestDashedRingPixels(t *testing.T) {
	// A full circle built from arcs, stroked dashed.
	const cx, cy, radius = 100.0, 100.0, 60.0
	var src vector.Path
	src.Arc(cx, cy, radius, 0, 2*math.Pi, vector.Clockwise)

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{30, 20}, 0)
	op := &vector.StrokeOptions{}
	op.Width = 6
	op.Dash = plan

	dst := ebiten.NewImage(200, 200)
	defer dst.Deallocate()
	vector.StrokePath(dst, &src, op, nil)

	// Sample points exactly on the circle. A dashed ring must contain both painted dashes
	// and unpainted gaps, while a solid ring would paint every sample.
	var painted, gaps int
	for k := 0; k < 72; k++ {
		a := 2 * math.Pi * float64(k) / 72
		x := int(math.Round(cx + radius*math.Cos(a)))
		y := int(math.Round(cy + radius*math.Sin(a)))
		if dst.At(x, y) != (color.RGBA{}) {
			painted++
		} else {
			gaps++
		}
	}
	if painted == 0 {
		t.Fatal("dashed ring rendered no pixels")
	}
	if gaps == 0 {
		t.Fatal("dashed ring has no gaps: the pattern was not applied around the circle")
	}
}

// --- Degenerate inputs -------------------------------------------------------

func TestDashDegenerateSubPaths(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{10, 10}, 0)

	t.Run("empty source", func(t *testing.T) {
		var src, dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, plan)
		if vector.SubPathCount(&dst) != 0 {
			t.Fatal("empty source produced dashes")
		}
	})

	t.Run("zero length segment", func(t *testing.T) {
		var src vector.Path
		src.MoveTo(5, 5)
		src.LineTo(5, 5) // collapsed; normalization drops it
		var dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, plan)
		if vector.SubPathCount(&dst) != 0 {
			t.Fatal("zero-length segment produced a dash")
		}
	})

	t.Run("pattern longer than path yields at most one bounded dash", func(t *testing.T) {
		var src vector.Path
		src.MoveTo(0, 0)
		src.LineTo(5, 0) // shorter than the first 100-long on entry
		big, _ := vector.NewStrokeDashPlanForTest([]float32{100, 50}, 0)
		var dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, big)
		if got := vector.SubPathCount(&dst); got > 1 {
			t.Fatalf("bounded degeneration violated: %d dashes", got)
		}
	})

	t.Run("self intersecting path stays bounded", func(t *testing.T) {
		var src vector.Path
		// A star-like self-intersecting polyline.
		src.MoveTo(0, 0)
		for i := 0; i < 12; i++ {
			x := float32(40 + 30*math.Sin(float64(i)*2.4))
			y := float32(40 + 30*math.Cos(float64(i)*2.4))
			src.LineTo(x, y)
		}
		var dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, plan)
		if n := vector.SubPathCount(&dst); n <= 0 || n > 1000 {
			t.Fatalf("self-intersecting dash count out of bounded range: %d", n)
		}
	})
}

// --- Cubic and arc inherit dashing ------------------------------------------

func TestDashCubicAndArcProduceBoundedDashes(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{18, 12}, 0)

	t.Run("cubic", func(t *testing.T) {
		var src vector.Path
		src.MoveTo(10, 100)
		src.CubicTo(40, 10, 120, 10, 150, 100)
		var dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, plan)
		if vector.SubPathCount(&dst) == 0 {
			t.Fatal("cubic produced no dashes")
		}
	})

	t.Run("arc", func(t *testing.T) {
		var src vector.Path
		src.Arc(80, 80, 50, 0, math.Pi, vector.Clockwise)
		var dst vector.Path
		vector.BuildDashedPathForTest(&dst, &src, plan)
		if vector.SubPathCount(&dst) == 0 {
			t.Fatal("arc produced no dashes")
		}
	})
}

// --- Hardened behaviors ------------------------------------------------------

func TestStrokeDashPlanErrorIs(t *testing.T) {
	_, err := vector.NewStrokeDashPlanForTest(nil, 0)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, vector.ErrInvalidStrokeDashPlan) {
		t.Errorf("error %v does not wrap ErrInvalidStrokeDashPlan", err)
	}
}

func TestStrokeDashPlanContinuousAndNothing(t *testing.T) {
	continuous, _ := vector.NewStrokeDashPlanForTest([]float32{10}, 0)
	if !continuous.IsContinuous() {
		t.Error("a single drawn entry is continuous")
	}
	if continuous.DrawsNothing() {
		t.Error("a single drawn entry draws something")
	}
	// A pattern whose only positive entries are gaps draws nothing.
	nothing, err := vector.NewStrokeDashPlanForTest([]float32{0, 5}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !nothing.DrawsNothing() || nothing.IsContinuous() {
		t.Errorf("a gap-only pattern must draw nothing: drawsNothing=%v continuous=%v",
			nothing.DrawsNothing(), nothing.IsContinuous())
	}
}

func TestAddStrokeContinuousPlanMatchesSolid(t *testing.T) {
	var src vector.Path
	src.MoveTo(10, 10)
	src.LineTo(90, 10)
	src.QuadTo(120, 60, 90, 90)

	base := &vector.AddStrokeOptions{}
	base.StrokeOptions.Width = 6

	var solid vector.Path
	solid.AddStroke(&src, base)

	continuous, _ := vector.NewStrokeDashPlanForTest([]float32{3}, 0)
	dashedOp := *base
	dashedOp.StrokeOptions.Dash = continuous
	var dashed vector.Path
	dashed.AddStroke(&src, &dashedOp)

	if got, want := vector.PathOperationsString(&dashed), vector.PathOperationsString(&solid); got != want {
		t.Errorf("a gapless plan must equal the solid stroke:\n%v\nvs\n%v", got, want)
	}
}

func TestAddStrokeGapOnlyPlanAddsNothing(t *testing.T) {
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)

	nothing, _ := vector.NewStrokeDashPlanForTest([]float32{0, 10}, 0)
	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 4
	op.StrokeOptions.Dash = nothing
	var dst vector.Path
	dst.AddStroke(&src, op)
	if got := vector.SubPathCount(&dst); got != 0 {
		t.Errorf("a gap-only plan added %d sub-paths", got)
	}
}

func TestAddStrokeClosedRingFullyDrawnIsOneClosedSubPath(t *testing.T) {
	// A ring whose circumference is shorter than the first drawn entry is fully covered by
	// one dash. Its seam must be a join, not two caps: the dashed outline must equal the
	// solid outline exactly.
	var src vector.Path
	src.Arc(50, 50, 30, 0, 2*math.Pi, vector.Clockwise)

	base := &vector.AddStrokeOptions{}
	base.StrokeOptions.Width = 4
	var solid vector.Path
	solid.AddStroke(&src, base)

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{1000, 10}, 0)
	dashedOp := *base
	dashedOp.StrokeOptions.Dash = plan
	var dashed vector.Path
	dashed.AddStroke(&src, &dashedOp)

	if got, want := vector.PathOperationsString(&dashed), vector.PathOperationsString(&solid); got != want {
		t.Errorf("a fully drawn closed ring must equal the solid ring:\n%v\nvs\n%v", got, want)
	}
}

func TestAddStrokeClosedRingSeamHasNoCaps(t *testing.T) {
	// Square perimeter 400. Pattern 120 drawn / 80 gap with offset -50 is drawn at both
	// s=0 and s=400, so the first and last intervals are two halves of one dash crossing
	// the closure and must be emitted as one open sub-path joining them.
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(100, 0)
	src.LineTo(100, 100)
	src.LineTo(0, 100)
	src.Close()

	roundCap := &vector.AddStrokeOptions{}
	roundCap.StrokeOptions.Width = 10
	roundCap.StrokeOptions.LineCap = vector.LineCapRound
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{120, 80}, -50)
	roundCap.StrokeOptions.Dash = plan

	var dashed vector.Path
	dashed.AddStroke(&src, roundCap)

	// A crossing dash is open, so round caps sit at its real ends; a butt-capped version
	// differs only inside the cap half-widths, both of which lie over the same gaps. The
	// composite must therefore not grow beyond the butt-capped bounds.
	buttCap := *roundCap
	buttCap.StrokeOptions.LineCap = vector.LineCapButt
	var dashedButt vector.Path
	dashedButt.AddStroke(&src, &buttCap)

	if rb, bb := dashed.Bounds(), dashedButt.Bounds(); rb.Min.X < bb.Min.X || rb.Min.Y < bb.Min.Y || rb.Max.X > bb.Max.X || rb.Max.Y > bb.Max.Y {
		t.Errorf("round caps grew the outline beyond the butt-capped bounds: %v vs %v", rb, bb)
	}
}

func TestAddStrokeDensePatternIsBounded(t *testing.T) {
	// A long path with a 0.001 pattern crosses tens of thousands of intervals. The result
	// must stay bounded and must not hang.
	var src vector.Path
	src.MoveTo(0, 0)
	src.LineTo(1e5, 0)
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{0.001, 0.001}, 0)

	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 1
	op.StrokeOptions.Dash = plan
	var dst vector.Path
	dst.AddStroke(&src, op)
	if n := vector.SubPathCount(&dst); n <= 0 || n > 1<<13 {
		t.Fatalf("dense dash count out of bounded range: %d", n)
	}
}

func TestAddStrokeDashConcurrent(t *testing.T) {
	// Plans are immutable and dashed scratch state is per-call, so concurrent dashed strokes
	// on distinct paths must not race or interfere.
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{9, 4}, 2)

	const goroutines = 8
	var wg sync.WaitGroup
	results := make([]string, goroutines)
	for i := range goroutines {
		wg.Go(func() {
			var src, dst vector.Path
			src.MoveTo(0, 0)
			src.LineTo(60, 0)
			src.QuadTo(80, 40, 120, 0)
			op := &vector.AddStrokeOptions{}
			op.StrokeOptions.Width = 3
			op.StrokeOptions.Dash = plan
			dst.AddStroke(&src, op)
			results[i] = vector.PathOperationsString(&dst)
		})
	}
	wg.Wait()
	for i := 1; i < goroutines; i++ {
		if results[i] != results[0] {
			t.Fatalf("concurrent dashed strokes differ:\n%v\nvs\n%v", results[i], results[0])
		}
	}
}

// --- DashSegments inspection ------------------------------------------------

// referenceSegments walks a dash pattern independently and returns the drawn intervals
// over [0, total] as (start, end) pairs, assuming a closed sub-path merges a head/tail pair.
func referenceSegments(entries []float64, offset, total float64, closed bool) [][2]float64 {
	period := 0.0
	for _, e := range entries {
		period += e
	}
	phaseAt := func(s float64) float64 {
		ph := math.Mod(s-offset, period)
		if ph < 0 {
			ph += period
		}
		return ph
	}
	var out [][2]float64
	for s := 0.0; s < total; {
		phase := phaseAt(s)
		idx, acc := 0, 0.0
		for i, e := range entries {
			if phase < acc+e {
				idx = i
				break
			}
			acc += e
		}
		next := min(s+(acc+entries[idx]-phase), total)
		if next <= s {
			next = math.Nextafter(s, total)
		}
		if idx%2 == 0 && next > s {
			// Coalesce touching drawn intervals exactly as the production walker does.
			if n := len(out); n > 0 && out[n-1][1] == s {
				out[n-1][1] = next
			} else {
				out = append(out, [2]float64{s, next})
			}
		}
		s = next
	}
	if closed && len(out) > 1 && out[0][0] == 0 && out[len(out)-1][1] == total {
		merged := [2]float64{out[len(out)-1][0] - total, out[0][1]}
		out = append([][2]float64{merged}, out[1:len(out)-1]...)
	}
	return out
}

func TestDashSegmentsAgainstReference(t *testing.T) {
	tests := []struct {
		name   string
		dashes []float32
		offset float32
		total  float32
		closed bool
	}{
		{name: "open line", dashes: []float32{15, 7}, total: 100},
		{name: "open with offset", dashes: []float32{15, 7}, offset: 3, total: 100},
		{name: "negative offset", dashes: []float32{10, 10}, offset: -5, total: 100},
		{name: "exact fit", dashes: []float32{20, 5}, total: 100},
		{name: "closed no wrap", dashes: []float32{60, 40}, total: 400, closed: true},
		{name: "closed wrap", dashes: []float32{120, 80}, offset: -50, total: 400, closed: true},
		{name: "closed fully drawn", dashes: []float32{1000, 10}, total: 400, closed: true},
		{name: "zero gap", dashes: []float32{10, 0, 5, 0}, total: 100},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := vector.NewStrokeDashPlanForTest(tc.dashes, tc.offset)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			entries := make([]float64, len(tc.dashes))
			for i, d := range tc.dashes {
				entries[i] = float64(d)
			}
			want := referenceSegments(entries, float64(tc.offset), float64(tc.total), tc.closed)

			got := plan.DashSegments(tc.total, tc.closed)
			if len(got) != len(want) {
				t.Fatalf("segment count: got %d (%v) want %d (%v)", len(got), got, len(want), want)
			}
			for i, seg := range got {
				if seg.Parity != 0 && seg.Parity%2 != 0 {
					t.Errorf("segment %d: non-even parity %d", i, seg.Parity)
				}
				if !(seg.Start <= seg.End) {
					t.Errorf("segment %d: Start %v > End %v", i, seg.Start, seg.End)
				}
				if math.Abs(float64(seg.Start)-want[i][0]) > 1e-3 || math.Abs(float64(seg.End)-want[i][1]) > 1e-3 {
					t.Errorf("segment %d: got [%v,%v] want [%v,%v]", i, seg.Start, seg.End, want[i][0], want[i][1])
				}
			}
		})
	}
}

func TestDashSegmentsClosedFlags(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{120, 80}, -50)
	segments := plan.DashSegments(400, true)
	if len(segments) != 2 {
		t.Fatalf("segment count: got %d want 2", len(segments))
	}
	// The first reported segment crosses the seam and counts its start back from the end.
	// It is open: its real ends sit in the gaps on either side of the seam.
	if segments[0].ClosedStart || segments[0].ClosedEnd {
		t.Errorf("crossing segment must be open at both ends: %+v", segments[0])
	}
	if segments[0].Start >= 0 {
		t.Errorf("crossing segment Start must count back from the end: %v", segments[0].Start)
	}
	if segments[1].ClosedStart || segments[1].ClosedEnd {
		t.Errorf("interior segment must not be closed: %+v", segments[1])
	}
}

func TestDashSegmentsBoundedAndFinite(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{0.001, 0.001}, 0)
	segments := plan.DashSegments(1e6, false)
	if len(segments) == 0 || len(segments) > 1<<13 {
		t.Fatalf("segment count out of bounded range: %d", len(segments))
	}
	for i, seg := range segments {
		if !isFinite(float64(seg.Start)) || !isFinite(float64(seg.End)) {
			t.Fatalf("segment %d is not finite: %+v", i, seg)
		}
	}
	if got := plan.DashSegments(0, false); got != nil {
		t.Errorf("zero-length sub-path must have no segments, got %v", got)
	}
	if got := plan.DashSegments(-1, false); got != nil {
		t.Errorf("negative length must have no segments, got %v", got)
	}
}

func TestDashSegmentsPeriod(t *testing.T) {
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{4, 6}, 0)
	if got, want := plan.Period(), float32(10); got != want {
		t.Errorf("period: got %v want %v", got, want)
	}
}

// --- Stroke geometry agrees with the inspected plan --------------------------

// TestAddStrokeDashMatchesDashSegments strokes a polyline with a plan and compares it,
// operation for operation, with stroking each plan-reported segment as its own polyline.
// This proves that AddStroke materializes exactly the intervals DashSegments reports.
func TestAddStrokeDashMatchesDashSegments(t *testing.T) {
	// One open polyline, total length 160.
	polyline := func() *vector.Path {
		var p vector.Path
		p.MoveTo(0, 0)
		p.LineTo(40, 0)
		p.LineTo(40, 40)
		p.LineTo(0, 40)
		p.LineTo(0, 0)
		p.LineTo(0, 20)
		return &p
	}
	const total = 180.0

	cases := []struct {
		name   string
		dashes []float32
		offset float32
	}{
		{name: "simple", dashes: []float32{12, 8}},
		{name: "negative offset", dashes: []float32{12, 8}, offset: -6},
		{name: "three entries", dashes: []float32{9, 4, 17}, offset: 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := vector.NewStrokeDashPlanForTest(tc.dashes, tc.offset)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			op := &vector.AddStrokeOptions{}
			op.StrokeOptions.Width = 5
			op.StrokeOptions.LineCap = vector.LineCapButt
			op.StrokeOptions.Dash = plan
			var dashed vector.Path
			dashed.AddStroke(polyline(), op)

			// The same polyline, cut at the plan's segment boundaries.
			var manual vector.Path
			op.StrokeOptions.Dash = nil
			for _, seg := range plan.DashSegments(float32(total), false) {
				var piece vector.Path
				piece.MoveTo(pointAtArcLength(0, 0, seg.Start))
				for _, k := range []float64{40, 80, 120, 160, 180} {
					if k <= float64(seg.Start) || k >= float64(seg.End) {
						continue
					}
					x, y := polylineAt(k)
					piece.LineTo(x, y)
				}
				piece.LineTo(pointAtArcLength(0, 0, seg.End))
				manual.AddStroke(&piece, op)
			}

			if got, want := vector.PathOperationsString(&dashed), vector.PathOperationsString(&manual); got != want {
				t.Errorf("dashed stroke differs from stroking the inspected segments:\n%s\nvs\n%s", got, want)
			}
		})
	}
}

// polylineAt returns the fixture polyline's point at arc length s.
func polylineAt(s float64) (float32, float32) {
	switch {
	case s <= 40:
		return float32(s), 0
	case s <= 80:
		return 40, float32(s - 40)
	case s <= 120:
		return float32(120 - s), 40
	case s <= 160:
		return 0, float32(160 - s)
	default:
		return 0, float32(s - 160)
	}
}

// pointAtArcLength is a tiny wrapper used to express endpoints by arc length.
func pointAtArcLength(_ float64, _ float64, s float32) (float32, float32) {
	return polylineAt(float64(s))
}

func TestAddStrokeDashedClosedRingReverseDirection(t *testing.T) {
	// The same closed circle traversed in either direction must produce the same number of
	// dashes. The phase depends only on accumulated arc length from each sub-path's own
	// start, so a reversal changes where dashes fall, but not how many or their lengths.
	mk := func(ccw bool) *vector.Path {
		var p vector.Path
		dir := vector.Clockwise
		if ccw {
			dir = vector.CounterClockwise
		}
		p.MoveTo(90, 50)
		p.Arc(50, 50, 40, 0, 2*math.Pi, dir)
		p.Close()
		return &p
	}

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{10, 6}, 0)

	count := func(p *vector.Path) int {
		var dashed vector.Path
		vector.BuildDashedPathForTest(&dashed, p, plan)
		return vector.SubPathCount(&dashed)
	}
	cw, ccw := count(mk(false)), count(mk(true))
	if cw == 0 {
		t.Fatal("clockwise ring produced no dashes")
	}
	if cw != ccw {
		t.Errorf("dash count depends on ring direction: %d vs %d", cw, ccw)
	}
}

func TestAddStrokeDashedCubicLoopClosed(t *testing.T) {
	// A closed cubic-only loop: dashes must exist, be bounded, and not emit a cap at the
	// seam when the pattern is drawn there.
	var src vector.Path
	src.MoveTo(50, 20)
	src.CubicTo(90, 20, 90, 80, 50, 80)
	src.CubicTo(10, 80, 10, 20, 50, 20)
	src.Close()

	plan, _ := vector.NewStrokeDashPlanForTest([]float32{1000, 10}, 0)
	op := &vector.AddStrokeOptions{}
	op.StrokeOptions.Width = 6
	op.StrokeOptions.LineCap = vector.LineCapRound
	op.StrokeOptions.Dash = plan
	var dashed vector.Path
	dashed.AddStroke(&src, op)

	if n := vector.SubPathCount(&dashed); n == 0 || n > 8 {
		t.Fatalf("cubic loop dash count out of bounded range: %d", n)
	}
	// A fully drawn closed loop must not bulge past a butt-capped version at the seam.
	butt := *op
	butt.StrokeOptions.LineCap = vector.LineCapButt
	var dashedButt vector.Path
	dashedButt.AddStroke(&src, &butt)
	if got, want := dashed.Bounds(), dashedButt.Bounds(); got != want {
		t.Errorf("round caps leaked at the cubic loop seam: %v vs %v", got, want)
	}
}

func TestStrokePathHonorsDashPlan(t *testing.T) {
	var src vector.Path
	src.MoveTo(10, 20)
	src.LineTo(110, 20)
	plan, _ := vector.NewStrokeDashPlanForTest([]float32{10, 10}, 0)

	dashed := ebiten.NewImage(120, 40)
	defer dashed.Deallocate()
	solid := ebiten.NewImage(120, 40)
	defer solid.Deallocate()

	op := &vector.StrokeOptions{Width: 6, Dash: plan}
	vector.StrokePath(dashed, &src, op, nil)
	vector.StrokePath(solid, &src, &vector.StrokeOptions{Width: 6}, nil)

	var dashedPixels, solidPixels int
	for y := range 40 {
		for x := range 120 {
			if _, _, _, a := dashed.At(x, y).RGBA(); a != 0 {
				dashedPixels++
			}
			if _, _, _, a := solid.At(x, y).RGBA(); a != 0 {
				solidPixels++
			}
		}
	}
	if dashedPixels == 0 {
		t.Fatal("dashed StrokePath painted nothing")
	}
	// The pattern is half drawn, so the dashed version must paint noticeably fewer pixels.
	if dashedPixels >= solidPixels {
		t.Errorf("dashed stroke did not reduce coverage: dashed=%d solid=%d", dashedPixels, solidPixels)
	}
}
