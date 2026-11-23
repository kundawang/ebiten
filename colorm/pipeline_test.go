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

package colorm_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"math"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/colorm"
)

// Reference implementations. These rebuild every stage's ColorM using only
// the public colorm API, so the pipeline's composition can be checked
// against sequential ColorM computation.

func refExposure(ev float64) colorm.ColorM {
	var m colorm.ColorM
	v := math.Exp2(ev)
	m.Scale(v, v, v, 1)
	return m
}

// refTemperatureRGB is Tanner Helland's black-body approximation, the
// reference for the temperature stage.
func refTemperatureRGB(kelvin float64) (r, g, b float64) {
	t := kelvin / 100
	if t <= 66 {
		r = 255
	} else {
		r = 329.698727446 * math.Pow(t-60, -0.1332047592)
	}
	if t <= 66 {
		g = 99.4708025861*math.Log(t) - 161.1195681661
	} else {
		g = 288.1221695283 * math.Pow(t-60, -0.0755148492)
	}
	switch {
	case t >= 66:
		b = 255
	case t <= 19:
		b = 0
	default:
		b = 138.5177312231*math.Log(t-10) - 305.0447927307
	}
	cl := func(v float64) float64 {
		return min(max(v, 0), 255) / 255
	}
	return cl(r), cl(g), cl(b)
}

func refTemperature(kelvin float64) colorm.ColorM {
	nr, ng, nb := refTemperatureRGB(6500)
	tr, tg, tb := refTemperatureRGB(kelvin)
	var m colorm.ColorM
	m.Scale(tr/nr, tg/ng, tb/nb, 1)
	return m
}

// refColorBlindSimulations mirrors the well-known simulation matrices.
var refColorBlindSimulations = [3][3][3]float64{
	{ // protanopia
		{0.567, 0.433, 0},
		{0.558, 0.442, 0},
		{0, 0.242, 0.758},
	},
	{ // deuteranopia
		{0.625, 0.375, 0},
		{0.7, 0.3, 0},
		{0, 0.3, 0.7},
	},
	{ // tritanopia
		{0.95, 0.05, 0},
		{0, 0.433, 0.567},
		{0, 0.475, 0.525},
	},
}

func refColorBlind(t colorm.ColorBlindType, correct bool, severity float64) colorm.ColorM {
	s := refColorBlindSimulations[t]
	var base [3][3]float64
	if !correct {
		base = s
	} else {
		e := [3][3]float64{
			{0, 0, 0},
			{0.7, 1, 0},
			{0.7, 0, 1},
		}
		// base = I + E*(I-S)
		for i := range 3 {
			for j := range 3 {
				var v float64
				for k := range 3 {
					iks := -s[k][j]
					if k == j {
						iks++
					}
					v += e[i][k] * iks
				}
				if i == j {
					v++
				}
				base[i][j] = v
			}
		}
	}
	var m colorm.ColorM
	for i := range 3 {
		for j := range 3 {
			ident := 0.0
			if i == j {
				ident = 1
			}
			m.SetElement(i, j, ident+severity*(base[i][j]-ident))
		}
	}
	return m
}

// refEvalCurve is an independent piecewise-linear interpolation.
func refEvalCurve(pts []colorm.CurvePoint, x float64) float64 {
	if x <= pts[0].X {
		return pts[0].Y
	}
	if x >= pts[len(pts)-1].X {
		return pts[len(pts)-1].Y
	}
	for i := 1; i < len(pts); i++ {
		if x <= pts[i].X {
			t := (x - pts[i-1].X) / (pts[i].X - pts[i-1].X)
			return pts[i-1].Y*(1-t) + pts[i].Y*t
		}
	}
	return pts[len(pts)-1].Y
}

// refFitCurve is an independent least-squares affine fit of the curve.
func refFitCurve(pts []colorm.CurvePoint) (a, b float64) {
	const n = 64
	var sx, sy, sxx, sxy float64
	for i := range n {
		x := float64(i) / (n - 1)
		y := refEvalCurve(pts, x)
		sx += x
		sy += y
		sxx += x * x
		sxy += x * y
	}
	a = (n*sxy - sx*sy) / (n*sxx - sx*sx)
	b = (sy - a*sx) / n
	return a, b
}

func refCurve(pts []colorm.CurvePoint) colorm.ColorM {
	a, b := refFitCurve(pts)
	var m colorm.ColorM
	m.Scale(a, a, a, 1)
	m.Translate(b, b, b, 0)
	return m
}

// refStageColorM rebuilds one stage's ColorM from a snapshot.
func refStageColorM(t *testing.T, s colorm.Stage) colorm.ColorM {
	t.Helper()
	switch s.Kind {
	case colorm.StageExposure:
		return refExposure(s.EV)
	case colorm.StageColorTemperature:
		return refTemperature(s.Kelvin)
	case colorm.StageCurve:
		return refCurve(s.Points)
	case colorm.StageColorBlind:
		return refColorBlind(s.BlindType, s.Correct, s.Severity)
	case colorm.StageMatrix:
		return *s.Matrix
	}
	t.Fatalf("unknown stage kind: %d", s.Kind)
	return colorm.ColorM{}
}

// refPipelineColorM concatenates the enabled stages in order, the way a
// caller would combine ColorMs by hand.
func refPipelineColorM(t *testing.T, stages []colorm.Stage) colorm.ColorM {
	t.Helper()
	var m colorm.ColorM
	for _, s := range stages {
		if s.Disabled {
			continue
		}
		m.Concat(refStageColorM(t, s))
	}
	return m
}

// identityApply applies the identity ColorM to clr.
func identityApply(clr color.Color) color.Color {
	var m colorm.ColorM
	return m.Apply(clr)
}

func assertColorMClose(t *testing.T, got, want colorm.ColorM, delta float64) {
	t.Helper()
	for i := range colorm.Dim - 1 {
		for j := range colorm.Dim {
			g, w := got.Element(i, j), want.Element(i, j)
			if math.Abs(g-w) > delta {
				t.Errorf("Element(%d, %d) = %f, want %f", i, j, g, w)
			}
		}
	}
}

func assertColorClose(t *testing.T, got, want color.Color, delta uint32) {
	t.Helper()
	r0, g0, b0, a0 := got.RGBA()
	r1, g1, b1, a1 := want.RGBA()
	if absDiffU32(r0, r1) > delta || absDiffU32(g0, g1) > delta ||
		absDiffU32(b0, b1) > delta || absDiffU32(a0, a1) > delta {
		t.Errorf("got {%d, %d, %d, %d}, want {%d, %d, %d, %d}", r0, g0, b0, a0, r1, g1, b1, a1)
	}
}

// Fixtures: a grayscale ramp, skin tones, transparent edges and bright
// values for extreme exposure.
var (
	grayRamp = []color.Color{
		color.NRGBA{0x10, 0x10, 0x10, 0xff},
		color.NRGBA{0x40, 0x40, 0x40, 0xff},
		color.NRGBA{0x80, 0x80, 0x80, 0xff},
		color.NRGBA{0xc0, 0xc0, 0xc0, 0xff},
		color.NRGBA{0xf0, 0xf0, 0xf0, 0xff},
	}
	skinTones = []color.Color{
		color.NRGBA{0xff, 0xe0, 0xc4, 0xff},
		color.NRGBA{0xe0, 0xac, 0x69, 0xff},
		color.NRGBA{0x8d, 0x55, 0x24, 0xff},
	}
	transparentEdges = []color.Color{
		color.NRGBA{0xff, 0x00, 0x00, 0x80},
		color.NRGBA{0x00, 0x80, 0xff, 0x40},
		color.NRGBA{0x33, 0x66, 0x99, 0x01},
		color.NRGBA{0x00, 0x00, 0x00, 0x00},
		color.RGBA{0x40, 0x80, 0xc0, 0xf0},
	}
	extremeExposure = []color.Color{
		color.NRGBA{0xe0, 0xe0, 0xe0, 0xff},
		color.NRGBA{0xff, 0xff, 0xff, 0xff},
		color.NRGBA{0xff, 0x80, 0x40, 0xff},
	}
	allFixtures = func() []color.Color {
		out := append([]color.Color{}, grayRamp...)
		out = append(out, skinTones...)
		out = append(out, transparentEdges...)
		return out
	}
)

func TestPipelineEmptyIsIdentity(t *testing.T) {
	var p colorm.Pipeline
	if got := p.Len(); got != 0 {
		t.Errorf("Len() = %d, want 0", got)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	if !c.IsIdentity() {
		t.Errorf("IsIdentity() = false, want true")
	}
	assertColorMClose(t, c.ColorM(), colorm.ColorM{}, 0)
	for _, f := range allFixtures() {
		assertColorClose(t, c.Apply(f), identityApply(f), 0x101)
	}
}

func TestPipelineOrder(t *testing.T) {
	exposure := colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 1.5}
	// A curve with a non-zero fitted offset does not commute with exposure.
	curve := colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{
		{X: 0, Y: 0.1}, {X: 0.5, Y: 0.5}, {X: 1, Y: 1},
	}}

	p0, err := colorm.NewPipeline(exposure, curve)
	if err != nil {
		t.Fatal(err)
	}
	p1, err := colorm.NewPipeline(curve, exposure)
	if err != nil {
		t.Fatal(err)
	}
	c0, err := p0.Compile()
	if err != nil {
		t.Fatal(err)
	}
	c1, err := p1.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// The compiled matrices must match sequential ColorM concatenation in
	// the same order.
	assertColorMClose(t, c0.ColorM(), refPipelineColorM(t, p0.Snapshot()), 1e-4)
	assertColorMClose(t, c1.ColorM(), refPipelineColorM(t, p1.Snapshot()), 1e-4)

	// And the two orders must actually differ.
	same := true
	for i := range colorm.Dim - 1 {
		for j := range colorm.Dim {
			m0, m1 := c0.ColorM(), c1.ColorM()
			if math.Abs(m0.Element(i, j)-m1.Element(i, j)) > 1e-4 {
				same = false
			}
		}
	}
	if same {
		t.Errorf("different stage orders produced the same matrix")
	}
}

func TestPipelineSequentialColorM(t *testing.T) {
	var mat colorm.ColorM
	mat.ChangeHSV(0.3, 1.1, 0.95)
	stages := []colorm.Stage{
		{Name: "exposure", Kind: colorm.StageExposure, EV: -0.7},
		{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 5500},
		{Name: "colorblind", Kind: colorm.StageColorBlind, BlindType: colorm.Deuteranopia, Severity: 0.6, Correct: true},
		{Name: "grade", Kind: colorm.StageMatrix, Matrix: &mat},
	}
	p, err := colorm.NewPipeline(stages...)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}

	ref := refPipelineColorM(t, p.Snapshot())
	assertColorMClose(t, c.ColorM(), ref, 1e-4)

	for _, f := range allFixtures() {
		assertColorClose(t, c.Apply(f), ref.Apply(f), 0x200)
		cm := c.ColorM()
		assertColorClose(t, cm.Apply(f), ref.Apply(f), 0x200)
	}
}

func TestPipelineCurve(t *testing.T) {
	pts := []colorm.CurvePoint{{X: 0, Y: 0}, {X: 0.25, Y: 0.18}, {X: 0.75, Y: 0.82}, {X: 1, Y: 1}}
	p, err := colorm.NewPipeline(colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: pts})
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// Apply must use the exact piecewise-linear curve.
	for _, f := range grayRamp {
		v, _, _, _ := f.RGBA()
		x := float64(v) / 0xffff
		want := uint16(refEvalCurve(pts, x) * 0xffff)
		got := c.Apply(f)
		r, g, b, a := got.RGBA()
		if absDiffU32(r, uint32(want)) > 0x200 || absDiffU32(g, uint32(want)) > 0x200 || absDiffU32(b, uint32(want)) > 0x200 {
			t.Errorf("Apply(%v) = {%x, %x, %x, %x}, want gray %x", f, r, g, b, a, want)
		}
	}

	// The compiled ColorM must be the least-squares fit of the curve.
	assertColorMClose(t, c.ColorM(), refCurve(pts), 1e-4)

	// The fit must stay close to the exact curve for this gentle S-curve.
	a, b := refFitCurve(pts)
	for i := range 65 {
		x := float64(i) / 64
		if d := math.Abs(refEvalCurve(pts, x) - (a*x + b)); d > 0.06 {
			t.Errorf("fit error at x=%f: %f, want <= 0.06", x, d)
		}
	}
}

func TestPipelineCurveIdentity(t *testing.T) {
	p, err := colorm.NewPipeline(colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{
		{X: 0, Y: 0}, {X: 1, Y: 1},
	}})
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, c.ColorM(), colorm.ColorM{}, 1e-4)
	for _, f := range allFixtures() {
		assertColorClose(t, c.Apply(f), identityApply(f), 0x101)
	}
}

func TestPipelineExtremeExposure(t *testing.T) {
	// +8 stops followed by -8 stops must round-trip: composition is
	// unclamped, so highlights are not clipped mid-chain.
	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "up", Kind: colorm.StageExposure, EV: 8},
		colorm.Stage{Name: "down", Kind: colorm.StageExposure, EV: -8},
	)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	ref := refPipelineColorM(t, p.Snapshot())
	assertColorMClose(t, c.ColorM(), ref, 1e-4)
	assertColorMClose(t, c.ColorM(), colorm.ColorM{}, 1e-4)
	for _, f := range extremeExposure {
		assertColorClose(t, c.Apply(f), ref.Apply(f), 0x200)
		assertColorClose(t, c.Apply(f), identityApply(f), 0x400)
	}

	// A single large positive exposure clamps once at the output and must
	// not produce NaN or wrap around.
	p2, err := colorm.NewPipeline(colorm.Stage{Name: "up", Kind: colorm.StageExposure, EV: 8})
	if err != nil {
		t.Fatal(err)
	}
	c2, err := p2.Compile()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range extremeExposure {
		got := c2.Apply(f)
		r, g, b, a := got.RGBA()
		for _, v := range []uint32{r, g, b, a} {
			if v > 0xffff {
				t.Errorf("Apply(%v) produced out-of-range value %x", f, v)
			}
		}
		assertColorClose(t, got, applyColorM(refExposure(8), f), 0x200)
	}
}

// applyColorM is a helper to call the pointer-receiver Apply on a ColorM
// returned by value.
func applyColorM(m colorm.ColorM, clr color.Color) color.Color {
	return m.Apply(clr)
}

func TestPipelineErrors(t *testing.T) {
	var mat colorm.ColorM
	nanMat := colorm.ColorM{}
	nanMat.SetElement(0, 0, math.NaN())

	cases := []struct {
		name  string
		stage colorm.Stage
		want  error
		field string
	}{
		{"empty name", colorm.Stage{Kind: colorm.StageExposure, EV: 1}, colorm.ErrEmptyStage, "name"},
		{"zero value", colorm.Stage{}, colorm.ErrEmptyStage, "name"},
		{"nan ev", colorm.Stage{Name: "x", Kind: colorm.StageExposure, EV: math.NaN()}, colorm.ErrNonFinite, "ev"},
		{"inf ev", colorm.Stage{Name: "x", Kind: colorm.StageExposure, EV: math.Inf(1)}, colorm.ErrNonFinite, "ev"},
		{"zero kelvin", colorm.Stage{Name: "x", Kind: colorm.StageColorTemperature}, colorm.ErrEmptyStage, "kelvin"},
		{"nan kelvin", colorm.Stage{Name: "x", Kind: colorm.StageColorTemperature, Kelvin: math.NaN()}, colorm.ErrNonFinite, "kelvin"},
		{"empty curve", colorm.Stage{Name: "x", Kind: colorm.StageCurve}, colorm.ErrEmptyStage, "points"},
		{"one point curve", colorm.Stage{Name: "x", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{{0, 0}}}, colorm.ErrEmptyStage, "points"},
		{"nan point", colorm.Stage{Name: "x", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{{0, 0}, {math.NaN(), 1}}}, colorm.ErrNonFinite, "points[1].x"},
		{"inf point y", colorm.Stage{Name: "x", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{{0, 0}, {1, math.Inf(-1)}}}, colorm.ErrNonFinite, "points[1].y"},
		{"duplicate point x", colorm.Stage{Name: "x", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{{0, 0}, {0, 1}}}, colorm.ErrInvalidParameter, "points[1].x"},
		{"nan severity", colorm.Stage{Name: "x", Kind: colorm.StageColorBlind, Severity: math.NaN()}, colorm.ErrNonFinite, "severity"},
		{"invalid blind type", colorm.Stage{Name: "x", Kind: colorm.StageColorBlind, BlindType: colorm.ColorBlindType(9)}, colorm.ErrInvalidParameter, "blindtype"},
		{"nil matrix", colorm.Stage{Name: "x", Kind: colorm.StageMatrix}, colorm.ErrEmptyStage, "matrix"},
		{"nan matrix", colorm.Stage{Name: "x", Kind: colorm.StageMatrix, Matrix: &nanMat}, colorm.ErrNonFinite, "matrix.body[0]"},
		{"invalid kind", colorm.Stage{Name: "x", Kind: colorm.StageKind(42)}, colorm.ErrInvalidParameter, "kind"},
		{"ok matrix", colorm.Stage{Name: "x", Kind: colorm.StageMatrix, Matrix: &mat}, nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var p colorm.Pipeline
			err := p.Add(tc.stage)
			if tc.want == nil {
				if err != nil {
					t.Fatalf("Add() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("Add() = %v, want %v", err, tc.want)
			}
			var e *colorm.Error
			if !errors.As(err, &e) {
				t.Fatalf("error %v is not a *colorm.Error", err)
			}
			if e.Op != "Add" || e.Stage != tc.stage.Name || e.Field != tc.field {
				t.Errorf("error context = (%q, %q, %q), want (%q, %q, %q)", e.Op, e.Stage, e.Field, "Add", tc.stage.Name, tc.field)
			}
			// A failed Add must not change the pipeline.
			if got := p.Len(); got != 0 {
				t.Errorf("Len() = %d after failed Add, want 0", got)
			}
		})
	}
}

func TestPipelineDuplicateName(t *testing.T) {
	var p colorm.Pipeline
	s := colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 1}
	if err := p.Add(s); err != nil {
		t.Fatal(err)
	}
	if err := p.Add(s); !errors.Is(err, colorm.ErrDuplicateName) {
		t.Errorf("Add() = %v, want ErrDuplicateName", err)
	}
	// Duplicates within a single Add call must fail atomically.
	if err := p.Add(
		colorm.Stage{Name: "a", Kind: colorm.StageExposure, EV: 1},
		colorm.Stage{Name: "a", Kind: colorm.StageExposure, EV: 2},
	); !errors.Is(err, colorm.ErrDuplicateName) {
		t.Errorf("Add() = %v, want ErrDuplicateName", err)
	}
	if got := p.Len(); got != 1 {
		t.Errorf("Len() = %d, want 1", got)
	}
}

func TestPipelineTooManyStages(t *testing.T) {
	var p colorm.Pipeline
	stages := make([]colorm.Stage, colorm.MaxStages)
	for i := range stages {
		stages[i] = colorm.Stage{Name: fmt.Sprintf("s%d", i), Kind: colorm.StageExposure, EV: 0.1}
	}
	if err := p.Add(stages...); err != nil {
		t.Fatal(err)
	}
	if err := p.Add(colorm.Stage{Name: "overflow", Kind: colorm.StageExposure, EV: 1}); !errors.Is(err, colorm.ErrTooManyStages) {
		t.Errorf("Add() = %v, want ErrTooManyStages", err)
	}
	if got := p.Len(); got != colorm.MaxStages {
		t.Errorf("Len() = %d, want %d", got, colorm.MaxStages)
	}

	// A single oversized Add must fail atomically.
	var q colorm.Pipeline
	if err := q.Add(append(stages, colorm.Stage{Name: "overflow", Kind: colorm.StageExposure})...); !errors.Is(err, colorm.ErrTooManyStages) {
		t.Errorf("Add() = %v, want ErrTooManyStages", err)
	}
	if got := q.Len(); got != 0 {
		t.Errorf("Len() = %d, want 0", got)
	}
}

func TestPipelineCyclicMerge(t *testing.T) {
	mk := func(name string) *colorm.Pipeline {
		p, err := colorm.NewPipeline(colorm.Stage{Name: name, Kind: colorm.StageExposure, EV: 1})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	a, b := mk("a"), mk("b")
	if err := a.Merge(a); !errors.Is(err, colorm.ErrCyclicReference) {
		t.Errorf("self Merge() = %v, want ErrCyclicReference", err)
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	if err := b.Merge(a); !errors.Is(err, colorm.ErrCyclicReference) {
		t.Errorf("cyclic Merge() = %v, want ErrCyclicReference", err)
	}
	// Transitive: c merged into b (already merged into a) must not be
	// mergeable back.
	c := mk("c")
	if err := b.Merge(c); err != nil {
		t.Fatal(err)
	}
	if err := c.Merge(b); !errors.Is(err, colorm.ErrCyclicReference) {
		t.Errorf("transitive cyclic Merge() = %v, want ErrCyclicReference", err)
	}
	if err := a.Merge(nil); !errors.Is(err, colorm.ErrInvalidParameter) {
		t.Errorf("nil Merge() = %v, want ErrInvalidParameter", err)
	}
}

func TestPipelineMerge(t *testing.T) {
	a, err := colorm.NewPipeline(colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 1})
	if err != nil {
		t.Fatal(err)
	}
	b, err := colorm.NewPipeline(colorm.Stage{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 5000})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Merge(b); err != nil {
		t.Fatal(err)
	}
	if got := a.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2", got)
	}
	// Merging copies: changing b afterwards must not affect a.
	if err := b.SetDisabled("whitebalance", true); err != nil {
		t.Fatal(err)
	}
	c, err := a.Compile()
	if err != nil {
		t.Fatal(err)
	}
	ref := colorm.ColorM{}
	ref.Concat(refExposure(1))
	ref.Concat(refTemperature(5000))
	assertColorMClose(t, c.ColorM(), ref, 1e-4)
	// Duplicate names across merged pipelines fail.
	d, err := colorm.NewPipeline(colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Merge(d); !errors.Is(err, colorm.ErrDuplicateName) {
		t.Errorf("Merge() = %v, want ErrDuplicateName", err)
	}
}

func TestPipelineDisableEnable(t *testing.T) {
	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 1},
		colorm.Stage{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 5000},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetDisabled("whitebalance", true); err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, c.ColorM(), refExposure(1), 1e-4)
	if got := p.Len(); got != 2 {
		t.Errorf("Len() = %d, want 2 (disabled stages are kept)", got)
	}
	if err := p.SetDisabled("whitebalance", false); err != nil {
		t.Fatal(err)
	}
	c, err = p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, c.ColorM(), refPipelineColorM(t, p.Snapshot()), 1e-4)
	if err := p.SetDisabled("nosuch", true); !errors.Is(err, colorm.ErrStageNotFound) {
		t.Errorf("SetDisabled() = %v, want ErrStageNotFound", err)
	}
	if !p.Remove("exposure") {
		t.Errorf("Remove() = false, want true")
	}
	if p.Remove("exposure") {
		t.Errorf("Remove() = true, want false")
	}
}

func TestPipelineSnapshotIsolation(t *testing.T) {
	pts := []colorm.CurvePoint{{X: 0, Y: 0}, {X: 1, Y: 1}}
	mat := colorm.ColorM{}
	mat.Scale(2, 2, 2, 1)
	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: pts},
		colorm.Stage{Name: "matrix", Kind: colorm.StageMatrix, Matrix: &mat},
	)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	before := c.ColorM()

	// Mutating the caller's inputs after Add must not affect the pipeline.
	pts[1].Y = 0
	mat.Reset()

	// Mutating snapshots must not affect the pipeline.
	snap := p.Snapshot()
	snap[0].Points[0].Y = 1
	snap[1].Matrix.Reset()
	s, ok := p.Stage("curve")
	if !ok {
		t.Fatal("Stage(curve) not found")
	}
	s.Points[0].Y = 1

	// Mutating the pipeline after Compile must not affect the compiled
	// pipeline.
	if err := p.SetDisabled("matrix", true); err != nil {
		t.Fatal(err)
	}
	if err := p.Add(colorm.Stage{Name: "extra", Kind: colorm.StageExposure, EV: 3}); err != nil {
		t.Fatal(err)
	}

	assertColorMClose(t, c.ColorM(), before, 0)
	ref := colorm.ColorM{}
	ref.Concat(refCurve([]colorm.CurvePoint{{X: 0, Y: 0}, {X: 1, Y: 1}}))
	m2 := colorm.ColorM{}
	m2.Scale(2, 2, 2, 1)
	ref.Concat(m2)
	assertColorMClose(t, c.ColorM(), ref, 1e-4)
}

func TestPipelineNormalization(t *testing.T) {
	// Finite but out-of-range values are clamped, not rejected.
	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 100},
		colorm.Stage{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 10},
		colorm.Stage{Name: "colorblind", Kind: colorm.StageColorBlind, Severity: 7},
		colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{
			{X: 2, Y: 1.5}, {X: -1, Y: -0.5},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	snap := p.Snapshot()
	if got := snap[0].EV; got != 16 {
		t.Errorf("EV = %f, want 16 (clamped)", got)
	}
	if got := snap[1].Kelvin; got != 1000 {
		t.Errorf("Kelvin = %f, want 1000 (clamped)", got)
	}
	if got := snap[2].Severity; got != 1 {
		t.Errorf("Severity = %f, want 1 (clamped)", got)
	}
	// Curve points are clamped and sorted.
	wantPts := []colorm.CurvePoint{{X: 0, Y: 0}, {X: 1, Y: 1}}
	if snap[3].Points[0] != wantPts[0] || snap[3].Points[1] != wantPts[1] {
		t.Errorf("Points = %v, want %v", snap[3].Points, wantPts)
	}
	// The compiled result must be finite even at the clamped extremes.
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	for i := range colorm.Dim - 1 {
		for j := range colorm.Dim {
			cm := c.ColorM()
			if v := cm.Element(i, j); math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Element(%d, %d) = %f, want finite", i, j, v)
			}
		}
	}
	for _, f := range allFixtures() {
		r, g, b, a := c.Apply(f).RGBA()
		for _, v := range []uint32{r, g, b, a} {
			if v > 0xffff {
				t.Errorf("Apply(%v) produced out-of-range value %x", f, v)
			}
		}
	}
}

func assertStagesEqual(t *testing.T, got, want []colorm.Stage) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(stages) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.Name != w.Name || g.Kind != w.Kind || g.Disabled != w.Disabled ||
			g.EV != w.EV || g.Kelvin != w.Kelvin || g.BlindType != w.BlindType ||
			g.Severity != w.Severity || g.Correct != w.Correct {
			t.Errorf("stage %d = %+v, want %+v", i, g, w)
			continue
		}
		if len(g.Points) != len(w.Points) {
			t.Errorf("stage %d: len(Points) = %d, want %d", i, len(g.Points), len(w.Points))
		} else {
			for j := range g.Points {
				if g.Points[j] != w.Points[j] {
					t.Errorf("stage %d: Points[%d] = %v, want %v", i, j, g.Points[j], w.Points[j])
				}
			}
		}
		if (g.Matrix == nil) != (w.Matrix == nil) {
			t.Errorf("stage %d: Matrix nil mismatch", i)
		} else if g.Matrix != nil {
			assertColorMClose(t, *g.Matrix, *w.Matrix, 0)
		}
	}
}

func TestPipelineSerializationRoundTrip(t *testing.T) {
	mat := colorm.ColorM{}
	mat.ChangeHSV(0.5, 0.8, 1.2)
	mat.Translate(0.01, -0.02, 0.03, 0)
	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: -1.25},
		colorm.Stage{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 7200},
		colorm.Stage{Name: "curve", Kind: colorm.StageCurve, Points: []colorm.CurvePoint{
			{X: 0, Y: 0.05}, {X: 0.3, Y: 0.28}, {X: 1, Y: 0.95},
		}},
		colorm.Stage{Name: "colorblind", Kind: colorm.StageColorBlind, BlindType: colorm.Protanopia, Severity: 0.4, Correct: true},
		colorm.Stage{Name: "grade", Kind: colorm.StageMatrix, Matrix: &mat, Disabled: true},
	)
	if err != nil {
		t.Fatal(err)
	}

	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var q colorm.Pipeline
	if err := json.Unmarshal(data, &q); err != nil {
		t.Fatal(err)
	}
	assertStagesEqual(t, q.Snapshot(), p.Snapshot())

	// The compiled results must be identical.
	cp, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	cq, err := q.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, cq.ColorM(), cp.ColorM(), 0)
	for _, f := range allFixtures() {
		assertColorClose(t, cq.Apply(f), cp.Apply(f), 0)
	}

	// A round trip must be stable.
	data2, err := json.Marshal(&q)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(data2) {
		t.Errorf("second marshal differs:\n%s\n%s", data, data2)
	}
}

func TestPipelineSerializationErrors(t *testing.T) {
	var p colorm.Pipeline
	if err := p.Add(colorm.Stage{Name: "keep", Kind: colorm.StageExposure, EV: 1}); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		data string
		want error
	}{
		{"bad json", `{`, nil},
		{"bad version", `{"version": 2, "stages": []}`, colorm.ErrUnsupportedVersion},
		{"missing version", `{"stages": []}`, colorm.ErrUnsupportedVersion},
		{"unknown kind", `{"version": 1, "stages": [{"name": "x", "kind": "neon"}]}`, colorm.ErrInvalidParameter},
		{"duplicate names", `{"version": 1, "stages": [{"name": "x", "kind": "exposure", "ev": 1}, {"name": "x", "kind": "exposure", "ev": 2}]}`, colorm.ErrDuplicateName},
		{"empty stage", `{"version": 1, "stages": [{"name": "x", "kind": "curve"}]}`, colorm.ErrEmptyStage},
		{"bad matrix length", `{"version": 1, "stages": [{"name": "x", "kind": "matrix", "matrix": [1, 2]}]}`, colorm.ErrInvalidParameter},
		{"unknown blind type", `{"version": 1, "stages": [{"name": "x", "kind": "colorblind", "blindType": "achromatopsia"}]}`, colorm.ErrInvalidParameter},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := json.Unmarshal([]byte(tc.data), &p)
			if err == nil {
				t.Fatal("Unmarshal() = nil, want error")
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Errorf("Unmarshal() = %v, want %v", err, tc.want)
			}
			// A failed Unmarshal must leave the pipeline unchanged.
			if got := p.Len(); got != 1 {
				t.Errorf("Len() = %d after failed Unmarshal, want 1", got)
			}
		})
	}
}

func TestPipelineConcurrency(t *testing.T) {
	var p colorm.Pipeline
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 8; i++ {
				name := fmt.Sprintf("g%d-s%d", g, i)
				if err := p.Add(colorm.Stage{Name: name, Kind: colorm.StageExposure, EV: float64(i%5) - 2}); err != nil {
					t.Errorf("Add() = %v", err)
					return
				}
				if err := p.SetDisabled(name, i%2 == 0); err != nil {
					t.Errorf("SetDisabled() = %v", err)
					return
				}
				c, err := p.Compile()
				if err != nil {
					t.Errorf("Compile() = %v", err)
					return
				}
				_ = p.Snapshot()
				if _, err := json.Marshal(&p); err != nil {
					t.Errorf("Marshal() = %v", err)
					return
				}
				// The result must never contain NaN, whatever the
				// interleaving of concurrent writes was.
				for _, f := range allFixtures() {
					r, g, b, a := c.Apply(f).RGBA()
					for _, v := range []uint32{r, g, b, a} {
						if v > 0xffff {
							t.Errorf("Apply(%v) produced out-of-range value %x", f, v)
							return
						}
					}
				}
				for i := range colorm.Dim - 1 {
					for j := range colorm.Dim {
						cm := c.ColorM()
						if v := cm.Element(i, j); math.IsNaN(v) || math.IsInf(v, 0) {
							t.Errorf("Element(%d, %d) = %f, want finite", i, j, v)
							return
						}
					}
				}
			}
		}(g)
	}
	wg.Wait()
	if got := p.Len(); got != 8*8 {
		t.Errorf("Len() = %d, want %d", got, 8*8)
	}
}

func TestPipelineColorBlindValues(t *testing.T) {
	// Spot-check the simulation against the reference matrix: protanopia
	// maps pure red to (0.567, 0.558, 0).
	p, err := colorm.NewPipeline(colorm.Stage{
		Name: "sim", Kind: colorm.StageColorBlind, BlindType: colorm.Protanopia, Severity: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	got := c.Apply(color.NRGBA{0xff, 0x00, 0x00, 0xff})
	r, g, b, _ := got.RGBA()
	if d := absDiffU32(r, uint32(math.Round(0.567*0xffff))); d > 0x200 {
		t.Errorf("r = %x, want ~%x", r, uint32(math.Round(0.567*0xffff)))
	}
	if d := absDiffU32(g, uint32(math.Round(0.558*0xffff))); d > 0x200 {
		t.Errorf("g = %x, want ~%x", g, uint32(math.Round(0.558*0xffff)))
	}
	if d := absDiffU32(b, 0); d > 0x200 {
		t.Errorf("b = %x, want ~0", b)
	}

	// Severity 0 is identity, and a full simulation is singular while a
	// zero-severity stage stays invertible.
	p0, err := colorm.NewPipeline(colorm.Stage{
		Name: "sim", Kind: colorm.StageColorBlind, BlindType: colorm.Protanopia, Severity: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	c0, err := p0.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, c0.ColorM(), colorm.ColorM{}, 1e-4)
	if !c0.IsInvertible() {
		t.Errorf("IsInvertible() = false, want true")
	}

	// A singular stage makes the whole pipeline non-invertible.
	zero := colorm.ColorM{}
	zero.Scale(1, 1, 0, 1)
	pz, err := colorm.NewPipeline(
		colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: 1},
		colorm.Stage{Name: "singular", Kind: colorm.StageMatrix, Matrix: &zero},
	)
	if err != nil {
		t.Fatal(err)
	}
	cz, err := pz.Compile()
	if err != nil {
		t.Fatal(err)
	}
	if cz.IsInvertible() {
		t.Errorf("IsInvertible() = true for a singular stage, want false")
	}
}

func TestPipelineTemperatureDirection(t *testing.T) {
	// 6500K is identity; warmer temperatures boost red over blue.
	p, err := colorm.NewPipeline(colorm.Stage{Name: "wb", Kind: colorm.StageColorTemperature, Kelvin: 6500})
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}
	assertColorMClose(t, c.ColorM(), colorm.ColorM{}, 1e-3)

	warm, err := colorm.NewPipeline(colorm.Stage{Name: "wb", Kind: colorm.StageColorTemperature, Kelvin: 3000})
	if err != nil {
		t.Fatal(err)
	}
	cw, err := warm.Compile()
	if err != nil {
		t.Fatal(err)
	}
	cwm := cw.ColorM()
	rs := cwm.Element(0, 0)
	bs := cwm.Element(2, 2)
	if rs <= bs {
		t.Errorf("warm temperature: red scale %f <= blue scale %f, want red > blue", rs, bs)
	}
	assertColorMClose(t, cw.ColorM(), refTemperature(3000), 1e-4)
}

func TestPipelineDrawImage(t *testing.T) {
	const w, h = 16, 16
	src := ebiten.NewImage(w, h)
	for y := range h {
		for x := range w {
			src.Set(x, y, color.NRGBA{
				R: uint8(x * 16),
				G: uint8(y * 16),
				B: uint8((x + y) * 8),
				A: uint8(0x40 + (x+y)*8),
			})
		}
	}

	p, err := colorm.NewPipeline(
		colorm.Stage{Name: "exposure", Kind: colorm.StageExposure, EV: -0.5},
		colorm.Stage{Name: "whitebalance", Kind: colorm.StageColorTemperature, Kelvin: 5500},
		colorm.Stage{Name: "colorblind", Kind: colorm.StageColorBlind, BlindType: colorm.Tritanopia, Severity: 0.3},
	)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.Compile()
	if err != nil {
		t.Fatal(err)
	}

	dst0 := ebiten.NewImage(w, h)
	c.DrawImage(dst0, src, nil)

	// The DrawImage path must match the equivalent sequential ColorM.
	dst1 := ebiten.NewImage(w, h)
	colorm.DrawImage(dst1, src, refPipelineColorM(t, p.Snapshot()), nil)

	for y := range h {
		for x := range w {
			c0 := dst0.At(x, y).(color.RGBA)
			c1 := dst1.At(x, y).(color.RGBA)
			if !sameColors(c0, c1, 2) {
				t.Errorf("pixel (%d, %d) = %v, want %v", x, y, c0, c1)
			}
		}
	}
}
