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

package colorm

import (
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"math"
	"slices"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
)

// This file implements ColorPipeline, a composable post-processing chain on
// top of ColorM. The semantics of the pipeline are defined as follows.
//
// # Color space and alpha
//
// Stages operate on straight-alpha (r, g, b, a) values in the range [0, 1],
// exactly like ColorM: an Image's pixels are alpha-premultiplied, so colors
// are un-multiplied before the transform and multiplied again afterwards.
// No sRGB <-> linear conversion is performed: the linear algebra is applied
// to the (gamma-encoded) values as stored, which matches ColorM's behavior.
// Exposure, color temperature, curve and color-blind stages never modify
// alpha; only a StageMatrix stage can.
//
// # Clamping
//
// Composition is unclamped linear algebra. Values are clamped to [0, 1] only
// once, at the final output (ColorM.Apply and the shader both clamp their
// results). Intermediate results may exceed [0, 1], so e.g. an exposure of
// +8 stops followed by -8 stops round-trips highlights losslessly.
//
// # Curves
//
// A ColorM can only express affine transforms, so a Curve stage is
// approximated by the least-squares affine fit of its control points when
// compiled to a ColorM. CompiledPipeline.Apply evaluates the exact
// piecewise-linear curve instead; its input is clamped to [0, 1] per stage.
// For pipelines without Curve stages, Apply and ColorM().Apply agree up to
// float32 rounding.
//
// # Inversion
//
// Whether the compiled ColorM is invertible depends on the stages. Exposure,
// temperature and non-degenerate curve stages are invertible; a StageMatrix
// with a zero scale is not, and one singular stage makes the whole chain
// singular. Use CompiledPipeline.IsInvertible before calling Invert on the
// result of ColorM(), which panics for singular matrices.
//
// # Duplicates and ordering
//
// Stage names must be unique within a pipeline. Stages of the same kind may
// appear multiple times; they are applied in the order they were added.
//
// # Serialization
//
// Pipeline implements json.Marshaler and json.Unmarshaler. The format is
// versioned (FormatVersion); unmarshaling a different version fails with
// ErrUnsupportedVersion. Serialized data is the normalized parameter
// snapshot, so a round-trip preserves the compiled result exactly.
//
// # Immutability and concurrency
//
// Add and Compile take deep copies of their inputs: mutating a Stage, its
// curve points or its matrix afterwards never affects the pipeline or an
// already compiled pipeline. A Pipeline is safe for concurrent use; a
// CompiledPipeline is immutable.

// FormatVersion is the version of the pipeline serialization format.
const FormatVersion = 1

// MaxStages is the maximum number of stages a pipeline can hold.
// Chains are bounded so that a compiled pipeline stays a single ColorM and
// parameter validation stays cheap.
const MaxStages = 64

// minEV and maxEV bound an exposure value in stops. Finite values outside
// the range are clamped (bounded degradation) instead of producing huge or
// denormal scale factors.
const (
	minEV = -16
	maxEV = 16
)

// minKelvin and maxKelvin bound color temperature values.
const (
	minKelvin = 1000
	maxKelvin = 40000
)

// neutralKelvin is the color temperature that maps to the identity transform.
const neutralKelvin = 6500

// Sentinel errors. Use errors.Is to match them; an *Error wraps them with
// the operation, stage name and field that caused the failure.
var (
	// ErrEmptyStage indicates a stage without a name or without the
	// parameters its kind requires (e.g. a curve without points).
	ErrEmptyStage = errors.New("colorm: empty stage")
	// ErrDuplicateName indicates two stages with the same name.
	ErrDuplicateName = errors.New("colorm: duplicate stage name")
	// ErrNonFinite indicates a NaN or infinite parameter.
	ErrNonFinite = errors.New("colorm: non-finite parameter")
	// ErrCyclicReference indicates a merge that would make a pipeline
	// depend on itself.
	ErrCyclicReference = errors.New("colorm: cyclic pipeline reference")
	// ErrTooManyStages indicates a chain longer than MaxStages.
	ErrTooManyStages = errors.New("colorm: too many stages")
	// ErrStageNotFound indicates a lookup by an unknown stage name.
	ErrStageNotFound = errors.New("colorm: stage not found")
	// ErrInvalidParameter indicates a parameter outside its valid domain
	// (e.g. an unknown stage kind).
	ErrInvalidParameter = errors.New("colorm: invalid parameter")
	// ErrUnsupportedVersion indicates a serialized pipeline with an
	// unsupported format version.
	ErrUnsupportedVersion = errors.New("colorm: unsupported pipeline format version")
)

// Error describes a pipeline failure with field context.
type Error struct {
	// Op is the operation that failed, e.g. "Add" or "Compile".
	Op string
	// Stage is the name of the stage involved, if any.
	Stage string
	// Field is the parameter involved, e.g. "ev" or "points[2].y".
	Field string
	// Err is the underlying sentinel error.
	Err error
}

// Error returns a string representation of the error.
func (e *Error) Error() string {
	s := "colorm: " + e.Op
	if e.Stage != "" {
		s += fmt.Sprintf(": stage %q", e.Stage)
	}
	if e.Field != "" {
		s += ": field " + e.Field
	}
	return s + ": " + e.Err.Error()
}

// Unwrap returns the underlying sentinel error.
func (e *Error) Unwrap() error {
	return e.Err
}

// StageKind identifies the type of a pipeline stage.
type StageKind int

const (
	// StageExposure scales linear RGB by 2^EV stops. Alpha is untouched.
	StageExposure StageKind = iota
	// StageColorTemperature applies a white-balance shift towards the
	// given color temperature in kelvin. neutralKelvin (6500K) is identity.
	StageColorTemperature
	// StageCurve applies a per-channel tone curve. See the package-level
	// semantics above for the affine approximation used in ColorM form.
	StageCurve
	// StageColorBlind simulates a color-vision deficiency or applies a
	// daltonization correction for it.
	StageColorBlind
	// StageMatrix applies an arbitrary ColorM.
	StageMatrix
)

// ColorBlindType identifies a color-vision deficiency.
type ColorBlindType int

const (
	// Protanopia is red-blindness.
	Protanopia ColorBlindType = iota
	// Deuteranopia is green-blindness.
	Deuteranopia
	// Tritanopia is blue-blindness.
	Tritanopia
)

// CurvePoint is a control point of a tone curve. X and Y are clamped to
// [0, 1]. Points are sorted by X when the stage is added.
type CurvePoint struct {
	X float64
	Y float64
}

// Stage is a single named step of a Pipeline. The fields used depend on
// Kind; the others are ignored. The zero value is invalid: a stage needs at
// least a name and the parameters its kind requires.
type Stage struct {
	// Name uniquely identifies the stage within a pipeline.
	Name string
	// Kind selects the transform.
	Kind StageKind
	// Disabled excludes the stage from compilation without removing it.
	Disabled bool

	// EV is the exposure in stops for StageExposure. Finite values are
	// clamped to [-16, 16].
	EV float64
	// Kelvin is the target color temperature for StageColorTemperature.
	// Finite values are clamped to [1000, 40000].
	Kelvin float64
	// Points are the control points for StageCurve. At least two points
	// with distinct X values are required.
	Points []CurvePoint
	// BlindType is the deficiency for StageColorBlind.
	BlindType ColorBlindType
	// Severity blends StageColorBlind between identity (0) and the full
	// transform (1). Finite values are clamped to [0, 1].
	Severity float64
	// Correct makes StageColorBlind apply a daltonization correction
	// instead of simulating the deficiency.
	Correct bool
	// Matrix is the transform for StageMatrix. It is copied when the
	// stage is added, so later changes to it do not affect the pipeline.
	Matrix *ColorM
}

// clone returns a deep copy of s.
func (s Stage) clone() Stage {
	c := s
	if s.Points != nil {
		c.Points = slices.Clone(s.Points)
	}
	if s.Matrix != nil {
		m := cloneColorM(s.Matrix)
		c.Matrix = &m
	}
	return c
}

// cloneColorM returns a copy of m that shares no state with it.
func cloneColorM(m *ColorM) ColorM {
	var body [16]float32
	var tr [4]float32
	m.ReadElements(body[:], tr[:])
	var c ColorM
	for i := range Dim - 1 {
		for j := range Dim - 1 {
			c.SetElement(i, j, float64(body[i+4*j]))
		}
		c.SetElement(i, Dim-1, float64(tr[i]))
	}
	return c
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func clamp(v, lo, hi float64) float64 {
	return min(max(v, lo), hi)
}

// normalize validates s and normalizes its parameters in place: finite but
// out-of-range values are clamped and curve points are sorted. Non-finite
// values and missing required parameters are errors.
func (s *Stage) normalize(op string) error {
	err := func(field string, sentinel error) error {
		return &Error{Op: op, Stage: s.Name, Field: field, Err: sentinel}
	}
	if s.Name == "" {
		return err("name", ErrEmptyStage)
	}
	switch s.Kind {
	case StageExposure:
		if !isFinite(s.EV) {
			return err("ev", ErrNonFinite)
		}
		s.EV = clamp(s.EV, minEV, maxEV)
	case StageColorTemperature:
		if !isFinite(s.Kelvin) {
			return err("kelvin", ErrNonFinite)
		}
		if s.Kelvin == 0 {
			return err("kelvin", ErrEmptyStage)
		}
		s.Kelvin = clamp(s.Kelvin, minKelvin, maxKelvin)
	case StageCurve:
		if len(s.Points) < 2 {
			return err("points", ErrEmptyStage)
		}
		pts := slices.Clone(s.Points)
		for i, p := range pts {
			if !isFinite(p.X) {
				return err(fmt.Sprintf("points[%d].x", i), ErrNonFinite)
			}
			if !isFinite(p.Y) {
				return err(fmt.Sprintf("points[%d].y", i), ErrNonFinite)
			}
			pts[i].X = clamp(p.X, 0, 1)
			pts[i].Y = clamp(p.Y, 0, 1)
		}
		slices.SortFunc(pts, func(a, b CurvePoint) int {
			switch {
			case a.X < b.X:
				return -1
			case a.X > b.X:
				return 1
			}
			return 0
		})
		for i := 1; i < len(pts); i++ {
			if pts[i].X == pts[i-1].X {
				return err(fmt.Sprintf("points[%d].x", i), ErrInvalidParameter)
			}
		}
		s.Points = pts
	case StageColorBlind:
		if s.BlindType < Protanopia || s.BlindType > Tritanopia {
			return err("blindtype", ErrInvalidParameter)
		}
		if !isFinite(s.Severity) {
			return err("severity", ErrNonFinite)
		}
		s.Severity = clamp(s.Severity, 0, 1)
	case StageMatrix:
		if s.Matrix == nil {
			return err("matrix", ErrEmptyStage)
		}
		var body [16]float32
		var tr [4]float32
		s.Matrix.ReadElements(body[:], tr[:])
		for i, v := range body {
			if !isFinite(float64(v)) {
				return err(fmt.Sprintf("matrix.body[%d]", i), ErrNonFinite)
			}
		}
		for i, v := range tr {
			if !isFinite(float64(v)) {
				return err(fmt.Sprintf("matrix.translation[%d]", i), ErrNonFinite)
			}
		}
	default:
		return err("kind", ErrInvalidParameter)
	}
	return nil
}

// Pipeline is an ordered, composable chain of color stages. The zero value
// is an empty pipeline, which compiles to the identity transform.
//
// A Pipeline is safe for concurrent use.
type Pipeline struct {
	mu     sync.RWMutex
	stages []Stage
	// refs tracks this pipeline and every pipeline merged into it, to
	// detect cyclic merges. Stages are always copied, never shared.
	refs map[*Pipeline]struct{}
}

// NewPipeline returns a pipeline with the given stages, equivalent to
// calling Add on an empty pipeline.
func NewPipeline(stages ...Stage) (*Pipeline, error) {
	p := &Pipeline{}
	if err := p.Add(stages...); err != nil {
		return nil, err
	}
	return p, nil
}

// Add appends stages to the pipeline. Each stage is validated, normalized
// and deep-copied; either all stages are added or none are.
func (p *Pipeline) Add(stages ...Stage) error {
	norm := make([]Stage, len(stages))
	for i, s := range stages {
		if err := s.normalize("Add"); err != nil {
			return err
		}
		norm[i] = s.clone()
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	seen := make(map[string]struct{}, len(p.stages)+len(norm))
	for _, s := range p.stages {
		seen[s.Name] = struct{}{}
	}
	for _, s := range norm {
		if _, ok := seen[s.Name]; ok {
			return &Error{Op: "Add", Stage: s.Name, Field: "name", Err: ErrDuplicateName}
		}
		seen[s.Name] = struct{}{}
	}
	if len(p.stages)+len(norm) > MaxStages {
		return &Error{Op: "Add", Err: ErrTooManyStages}
	}
	p.stages = append(p.stages, norm...)
	return nil
}

// Merge appends snapshots of every stage of each other pipeline to p, in
// order. Merging copies stages; later changes to the other pipelines do not
// affect p. A merge that would make a pipeline (transitively) contain
// itself fails with ErrCyclicReference.
func (p *Pipeline) Merge(others ...*Pipeline) error {
	// Snapshot the sources before locking p, so that two pipelines merging
	// each other concurrently cannot deadlock.
	type snapshot struct {
		src    *Pipeline
		stages []Stage
		refs   map[*Pipeline]struct{}
	}
	snaps := make([]snapshot, 0, len(others))
	for _, o := range others {
		if o == nil {
			return &Error{Op: "Merge", Err: ErrInvalidParameter}
		}
		o.mu.RLock()
		s := snapshot{src: o, stages: make([]Stage, len(o.stages)), refs: make(map[*Pipeline]struct{}, len(o.refs)+1)}
		for i, st := range o.stages {
			s.stages[i] = st.clone()
		}
		for r := range o.refs {
			s.refs[r] = struct{}{}
		}
		s.refs[o] = struct{}{}
		o.mu.RUnlock()
		snaps = append(snaps, s)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.refs == nil {
		p.refs = map[*Pipeline]struct{}{p: {}}
	}
	names := make(map[string]struct{}, len(p.stages))
	for _, s := range p.stages {
		names[s.Name] = struct{}{}
	}
	total := len(p.stages)
	for _, s := range snaps {
		if _, ok := s.refs[p]; ok {
			return &Error{Op: "Merge", Err: ErrCyclicReference}
		}
		for _, st := range s.stages {
			if _, ok := names[st.Name]; ok {
				return &Error{Op: "Merge", Stage: st.Name, Field: "name", Err: ErrDuplicateName}
			}
			names[st.Name] = struct{}{}
		}
		total += len(s.stages)
	}
	if total > MaxStages {
		return &Error{Op: "Merge", Err: ErrTooManyStages}
	}
	for _, s := range snaps {
		p.stages = append(p.stages, s.stages...)
		for r := range s.refs {
			p.refs[r] = struct{}{}
		}
	}
	return nil
}

// Remove deletes the stage with the given name. It reports whether the
// stage existed.
func (p *Pipeline) Remove(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, s := range p.stages {
		if s.Name == name {
			p.stages = slices.Delete(p.stages, i, i+1)
			return true
		}
	}
	return false
}

// SetDisabled enables or disables the stage with the given name.
func (p *Pipeline) SetDisabled(name string, disabled bool) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, s := range p.stages {
		if s.Name == name {
			p.stages[i].Disabled = disabled
			return nil
		}
	}
	return &Error{Op: "SetDisabled", Stage: name, Field: "name", Err: ErrStageNotFound}
}

// Stage returns a snapshot of the stage with the given name. Mutating the
// returned stage does not affect the pipeline.
func (p *Pipeline) Stage(name string) (Stage, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, s := range p.stages {
		if s.Name == name {
			return s.clone(), true
		}
	}
	return Stage{}, false
}

// Snapshot returns deep copies of all stages in order, including disabled
// ones. Mutating the result does not affect the pipeline.
func (p *Pipeline) Snapshot() []Stage {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]Stage, len(p.stages))
	for i, s := range p.stages {
		out[i] = s.clone()
	}
	return out
}

// Len returns the number of stages, including disabled ones.
func (p *Pipeline) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.stages)
}

// compiledStage is an immutable, normalized stage with its affine form.
type compiledStage struct {
	stage Stage
	m     ColorM
	body  [16]float32
	tr    [4]float32
	// curve is non-nil for curve stages and holds the exact control points.
	curve []CurvePoint
}

// CompiledPipeline is the immutable result of compiling a Pipeline. It is
// safe for concurrent use.
type CompiledPipeline struct {
	stages []compiledStage
	colorM ColorM
}

// Compile snapshots the pipeline's enabled stages and compiles them into an
// immutable CompiledPipeline. Later changes to the pipeline do not affect
// the result. An empty pipeline (or one with all stages disabled) compiles
// to the identity transform.
func (p *Pipeline) Compile() (*CompiledPipeline, error) {
	p.mu.RLock()
	stages := make([]Stage, len(p.stages))
	for i, s := range p.stages {
		stages[i] = s.clone()
	}
	p.mu.RUnlock()

	c := &CompiledPipeline{}
	var m ColorM
	for _, s := range stages {
		if s.Disabled {
			continue
		}
		cs, err := compileStage(s)
		if err != nil {
			return nil, err
		}
		m.Concat(cs.m)
		c.stages = append(c.stages, cs)
	}
	c.colorM = m
	return c, nil
}

// compileStage computes the affine form of a single normalized stage.
func compileStage(s Stage) (compiledStage, error) {
	cs := compiledStage{stage: s.clone()}
	m := ColorM{}
	switch s.Kind {
	case StageExposure:
		v := math.Exp2(s.EV)
		m.Scale(v, v, v, 1)
	case StageColorTemperature:
		r, g, b := temperatureScale(s.Kelvin)
		m.Scale(r, g, b, 1)
	case StageCurve:
		a, b := fitCurve(s.Points)
		m.Scale(a, a, a, 1)
		m.Translate(b, b, b, 0)
		cs.curve = slices.Clone(s.Points)
	case StageColorBlind:
		cb := colorBlindMatrix(s.BlindType, s.Correct, s.Severity)
		for i := range 3 {
			for j := range 3 {
				m.SetElement(i, j, cb[i][j])
			}
		}
	case StageMatrix:
		m = cloneColorM(s.Matrix)
	default:
		return compiledStage{}, &Error{Op: "Compile", Stage: s.Name, Field: "kind", Err: ErrInvalidParameter}
	}
	cs.m = m
	m.ReadElements(cs.body[:], cs.tr[:])
	return cs, nil
}

// ColorM returns the whole pipeline as a single color matrix, ready for
// DrawImage or DrawTriangles. Curve stages contribute their least-squares
// affine approximation.
func (c *CompiledPipeline) ColorM() ColorM {
	return c.colorM
}

// IsIdentity reports whether the pipeline compiles to the identity
// transform, e.g. when it has no enabled stages.
func (c *CompiledPipeline) IsIdentity() bool {
	return len(c.stages) == 0
}

// IsInvertible reports whether the compiled color matrix is invertible.
func (c *CompiledPipeline) IsInvertible() bool {
	return c.colorM.IsInvertible()
}

// Stages returns snapshots of the compiled (enabled) stages in order.
func (c *CompiledPipeline) Stages() []Stage {
	out := make([]Stage, len(c.stages))
	for i, s := range c.stages {
		out[i] = s.stage.clone()
	}
	return out
}

// Apply applies the exact pipeline to clr: matrix stages compose without
// intermediate clamping, curve stages use their exact piecewise-linear form,
// and the result is clamped to [0, 1] once at the end.
//
// Apply panics if clr is nil.
func (c *CompiledPipeline) Apply(clr color.Color) color.Color {
	if clr == nil {
		panic("colorm: the given color to Apply must not be nil")
	}
	r, g, b, a := clr.RGBA()
	var rf, gf, bf, af float64
	if a > 0 {
		af = float64(a) / 0xffff
		rf = float64(r) / float64(a)
		gf = float64(g) / float64(a)
		bf = float64(b) / float64(a)
	}
	for _, s := range c.stages {
		if s.curve != nil {
			rf = evalCurve(s.curve, clamp(rf, 0, 1))
			gf = evalCurve(s.curve, clamp(gf, 0, 1))
			bf = evalCurve(s.curve, clamp(bf, 0, 1))
			continue
		}
		eb, et := &s.body, &s.tr
		rf, gf, bf, af =
			float64(eb[0])*rf+float64(eb[4])*gf+float64(eb[8])*bf+float64(eb[12])*af+float64(et[0]),
			float64(eb[1])*rf+float64(eb[5])*gf+float64(eb[9])*bf+float64(eb[13])*af+float64(et[1]),
			float64(eb[2])*rf+float64(eb[6])*gf+float64(eb[10])*bf+float64(eb[14])*af+float64(et[2]),
			float64(eb[3])*rf+float64(eb[7])*gf+float64(eb[11])*bf+float64(eb[15])*af+float64(et[3])
	}
	return color.NRGBA64{
		R: uint16(clamp(rf, 0, 1) * 0xffff),
		G: uint16(clamp(gf, 0, 1) * 0xffff),
		B: uint16(clamp(bf, 0, 1) * 0xffff),
		A: uint16(clamp(af, 0, 1) * 0xffff),
	}
}

// DrawImage draws src onto dst with the compiled pipeline. It is shorthand
// for colorm.DrawImage with the pipeline's ColorM and reuses that path.
func (c *CompiledPipeline) DrawImage(dst, src *ebiten.Image, op *DrawImageOptions) {
	DrawImage(dst, src, c.colorM, op)
}

// temperatureRGB approximates the black-body color of the given temperature
// in kelvin, with each channel in [0, 1]. It uses Tanner Helland's
// approximation; kelvin is expected to be pre-clamped to [1000, 40000].
func temperatureRGB(kelvin float64) (r, g, b float64) {
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
	return clamp(r, 0, 255) / 255, clamp(g, 0, 255) / 255, clamp(b, 0, 255) / 255
}

// temperatureScale returns the RGB scale factors that shift white balance
// towards the given temperature, normalized so that neutralKelvin is
// identity.
func temperatureScale(kelvin float64) (r, g, b float64) {
	nr, ng, nb := temperatureRGB(neutralKelvin)
	tr, tg, tb := temperatureRGB(kelvin)
	return tr / nr, tg / ng, tb / nb
}

// colorBlindSimulations are the RGB simulation matrices for each deficiency
// (Vienot-Brettel style, row-major: out row = matrix row . in rgb).
var colorBlindSimulations = [3][3][3]float64{
	Protanopia: {
		{0.567, 0.433, 0},
		{0.558, 0.442, 0},
		{0, 0.242, 0.758},
	},
	Deuteranopia: {
		{0.625, 0.375, 0},
		{0.7, 0.3, 0},
		{0, 0.3, 0.7},
	},
	Tritanopia: {
		{0.95, 0.05, 0},
		{0, 0.433, 0.567},
		{0, 0.475, 0.525},
	},
}

// daltonizeError redistributes the simulation error onto the remaining
// channels (Fidaner-style daltonization).
var daltonizeError = [3][3]float64{
	{0, 0, 0},
	{0.7, 1, 0},
	{0.7, 0, 1},
}

// colorBlindMatrix returns the 3x3 RGB matrix for a color-blind stage:
// identity blended by severity with either the simulation matrix or its
// daltonization correction I + E*(I-S).
func colorBlindMatrix(t ColorBlindType, correct bool, severity float64) [3][3]float64 {
	s := colorBlindSimulations[t]
	var base [3][3]float64
	if !correct {
		base = s
	} else {
		// C = I + E*(I-S), where E redistributes the simulation error.
		for i := range 3 {
			for j := range 3 {
				var v float64
				for k := range 3 {
					iks := -s[k][j]
					if k == j {
						iks += 1
					}
					v += daltonizeError[i][k] * iks
				}
				base[i][j] = v
				if i == j {
					base[i][j] += 1
				}
			}
		}
	}
	var m [3][3]float64
	for i := range 3 {
		for j := range 3 {
			var ident float64
			if i == j {
				ident = 1
			}
			m[i][j] = ident + severity*(base[i][j]-ident)
		}
	}
	return m
}

// evalCurve evaluates the piecewise-linear curve through pts at x.
// pts must be sorted by X with at least two points.
func evalCurve(pts []CurvePoint, x float64) float64 {
	if x <= pts[0].X {
		return pts[0].Y
	}
	last := pts[len(pts)-1]
	if x >= last.X {
		return last.Y
	}
	for i := 1; i < len(pts); i++ {
		if x <= pts[i].X {
			p0, p1 := pts[i-1], pts[i]
			t := (x - p0.X) / (p1.X - p0.X)
			return p0.Y + t*(p1.Y-p0.Y)
		}
	}
	return last.Y
}

// fitCurve returns the least-squares affine fit y = a*x + b of the curve
// through pts over [0, 1].
func fitCurve(pts []CurvePoint) (a, b float64) {
	const n = 64
	var sx, sy, sxx, sxy float64
	for i := range n {
		x := float64(i) / (n - 1)
		y := evalCurve(pts, x)
		sx += x
		sy += y
		sxx += x * x
		sxy += x * y
	}
	d := float64(n)*sxx - sx*sx
	a = (float64(n)*sxy - sx*sy) / d
	b = (sy - a*sx) / float64(n)
	return a, b
}

// stageKindNames map stage kinds to their serialization names.
var stageKindNames = map[StageKind]string{
	StageExposure:         "exposure",
	StageColorTemperature: "temperature",
	StageCurve:            "curve",
	StageColorBlind:       "colorblind",
	StageMatrix:           "matrix",
}

var colorBlindTypeNames = map[ColorBlindType]string{
	Protanopia:   "protanopia",
	Deuteranopia: "deuteranopia",
	Tritanopia:   "tritanopia",
}

// stageJSON is the serialized form of a Stage. Matrix holds the 20 elements
// of the ColorM in row-major (i, j) order as reported by Element.
type stageJSON struct {
	Name      string       `json:"name"`
	Kind      string       `json:"kind"`
	Disabled  bool         `json:"disabled,omitempty"`
	EV        float64      `json:"ev,omitempty"`
	Kelvin    float64      `json:"kelvin,omitempty"`
	Points    []CurvePoint `json:"points,omitempty"`
	BlindType string       `json:"blindType,omitempty"`
	Severity  float64      `json:"severity,omitempty"`
	Correct   bool         `json:"correct,omitempty"`
	Matrix    []float64    `json:"matrix,omitempty"`
}

type pipelineJSON struct {
	Version int         `json:"version"`
	Stages  []stageJSON `json:"stages"`
}

// MarshalJSON implements json.Marshaler. The output is a normalized,
// versioned snapshot of the pipeline.
func (p *Pipeline) MarshalJSON() ([]byte, error) {
	stages := p.Snapshot()
	pj := pipelineJSON{Version: FormatVersion, Stages: make([]stageJSON, 0, len(stages))}
	for _, s := range stages {
		sj := stageJSON{
			Name:     s.Name,
			Kind:     stageKindNames[s.Kind],
			Disabled: s.Disabled,
			EV:       s.EV,
			Kelvin:   s.Kelvin,
			Points:   s.Points,
			Severity: s.Severity,
			Correct:  s.Correct,
		}
		if s.Kind == StageColorBlind {
			sj.BlindType = colorBlindTypeNames[s.BlindType]
		}
		if s.Kind == StageMatrix && s.Matrix != nil {
			sj.Matrix = make([]float64, 0, Dim*(Dim-1))
			for i := range Dim - 1 {
				for j := range Dim {
					sj.Matrix = append(sj.Matrix, s.Matrix.Element(i, j))
				}
			}
		}
		pj.Stages = append(pj.Stages, sj)
	}
	return json.Marshal(pj)
}

// UnmarshalJSON implements json.Unmarshaler. The pipeline's content is
// replaced only if the whole document is valid; otherwise the pipeline is
// left unchanged and an error is returned.
func (p *Pipeline) UnmarshalJSON(data []byte) error {
	var pj pipelineJSON
	if err := json.Unmarshal(data, &pj); err != nil {
		return &Error{Op: "UnmarshalJSON", Err: err}
	}
	if pj.Version != FormatVersion {
		return &Error{Op: "UnmarshalJSON", Field: "version", Err: ErrUnsupportedVersion}
	}
	kinds := make(map[string]StageKind, len(stageKindNames))
	for k, n := range stageKindNames {
		kinds[n] = k
	}
	blindTypes := make(map[string]ColorBlindType, len(colorBlindTypeNames))
	for t, n := range colorBlindTypeNames {
		blindTypes[n] = t
	}
	stages := make([]Stage, 0, len(pj.Stages))
	for _, sj := range pj.Stages {
		kind, ok := kinds[sj.Kind]
		if !ok {
			return &Error{Op: "UnmarshalJSON", Stage: sj.Name, Field: "kind", Err: ErrInvalidParameter}
		}
		s := Stage{
			Name:     sj.Name,
			Kind:     kind,
			Disabled: sj.Disabled,
			EV:       sj.EV,
			Kelvin:   sj.Kelvin,
			Points:   sj.Points,
			Severity: sj.Severity,
			Correct:  sj.Correct,
		}
		if kind == StageColorBlind {
			t, ok := blindTypes[sj.BlindType]
			if !ok {
				return &Error{Op: "UnmarshalJSON", Stage: sj.Name, Field: "blindType", Err: ErrInvalidParameter}
			}
			s.BlindType = t
		}
		if kind == StageMatrix {
			if len(sj.Matrix) != Dim*(Dim-1) {
				return &Error{Op: "UnmarshalJSON", Stage: sj.Name, Field: "matrix", Err: ErrInvalidParameter}
			}
			m := ColorM{}
			for i := range Dim - 1 {
				for j := range Dim {
					m.SetElement(i, j, sj.Matrix[i*Dim+j])
				}
			}
			s.Matrix = &m
		}
		stages = append(stages, s)
	}

	// Reuse Add's validation by building a fresh pipeline and swapping.
	np := &Pipeline{}
	if err := np.Add(stages...); err != nil {
		if e, ok := err.(*Error); ok {
			e.Op = "UnmarshalJSON"
		}
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.stages = np.stages
	return nil
}
