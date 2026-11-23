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

package ebiten_test

import (
	"fmt"
	"image/color"
	"math"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/legacyshader"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/glsl"
)

// reflectionFixtureShader is a fixture with scalar, vector, and array uniforms, an unused
// uniform, and a texture read.
const reflectionFixtureShader = `//kage:unit pixels

package main

var Factor float
var Color vec4
var Weights [4]float
var Count int
var Flag bool
var UnusedVec vec3

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	c := Color * Factor
	if Flag {
		c *= vec4(Weights[0], Weights[1], Weights[2], Weights[3])
	}
	return c + vec4(float(Count)) * imageSrc0At(src0Pos)
}
`

func newReflectionFixtureShader(t *testing.T) *ebiten.Shader {
	t.Helper()
	s, err := ebiten.NewShader([]byte(reflectionFixtureShader))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestShaderReflectionPlan(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}

	type wantUniform struct {
		name       string
		kind       ebiten.ShaderValueKind
		offset     int
		dwordCount int
		used       bool
	}
	want := []wantUniform{
		{name: "Factor", kind: ebiten.ShaderValueKindFloat, offset: 0, dwordCount: 1, used: true},
		{name: "Color", kind: ebiten.ShaderValueKindVec4, offset: 1, dwordCount: 4, used: true},
		{name: "Weights", kind: ebiten.ShaderValueKindArray, offset: 5, dwordCount: 4, used: true},
		{name: "Count", kind: ebiten.ShaderValueKindInt, offset: 9, dwordCount: 1, used: true},
		{name: "Flag", kind: ebiten.ShaderValueKindBool, offset: 10, dwordCount: 1, used: true},
		{name: "UnusedVec", kind: ebiten.ShaderValueKindVec3, offset: 11, dwordCount: 3, used: false},
	}
	if got, want := len(uniforms), len(want); got != want {
		t.Fatalf("len(uniforms): got: %d, want: %d", got, want)
	}
	for i, w := range want {
		u := uniforms[i]
		if u.Name != w.name {
			t.Errorf("uniforms[%d].Name: got: %s, want: %s", i, u.Name, w.name)
		}
		if u.Type.Kind != w.kind {
			t.Errorf("uniforms[%d].Type.Kind: got: %s, want: %s", i, u.Type.Kind, w.kind)
		}
		if u.Offset != w.offset {
			t.Errorf("uniforms[%d].Offset: got: %d, want: %d", i, u.Offset, w.offset)
		}
		if u.DwordCount != w.dwordCount {
			t.Errorf("uniforms[%d].DwordCount: got: %d, want: %d", i, u.DwordCount, w.dwordCount)
		}
		if u.Internal {
			t.Errorf("uniforms[%d].Internal: got: true, want: false", i)
		}
		// User uniforms are reachable only from the fragment stage.
		if u.UsedInVertex {
			t.Errorf("uniforms[%d].UsedInVertex: got: true, want: false", i)
		}
		if u.UsedInFragment != w.used {
			t.Errorf("uniforms[%d].UsedInFragment: got: %t, want: %t", i, u.UsedInFragment, w.used)
		}
		if got, want := len(u.Defaults), u.DwordCount; got != want {
			t.Errorf("len(uniforms[%d].Defaults): got: %d, want: %d", i, got, want)
		}
		for _, d := range u.Defaults {
			if d != 0 {
				t.Errorf("uniforms[%d].Defaults must be zero-filled, got: %v", i, u.Defaults)
				break
			}
		}
	}

	// The array uniform must report its element type and length.
	w := uniforms[2]
	if w.Type.Element == nil {
		t.Fatal("Weights.Type.Element must not be nil")
	}
	if got, want := w.Type.Element.Kind, ebiten.ShaderValueKindFloat; got != want {
		t.Errorf("Weights.Type.Element.Kind: got: %s, want: %s", got, want)
	}
	if got, want := w.Type.ArrayLength, 4; got != want {
		t.Errorf("Weights.Type.ArrayLength: got: %d, want: %d", got, want)
	}

	if got, want := mustUniformDwordCount(t, plan), 14; got != want {
		t.Errorf("plan.UniformDwordCount(): got: %d, want: %d", got, want)
	}
}

func mustUniformDwordCount(t *testing.T, plan *ebiten.ShaderReflectionPlan) int {
	t.Helper()
	n, err := plan.UniformDwordCount()
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func TestShaderReflectionPlanInternalUniforms(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	internal, err := plan.InternalUniforms()
	if err != nil {
		t.Fatal(err)
	}

	// The engine's reserved uniforms come first in the program's uniform block.
	want := []struct {
		name       string
		offset     int
		dwordCount int
	}{
		{name: "__imageDstTextureSize", offset: 0, dwordCount: 2},
		{name: "__imageSrcTextureSizes", offset: 2, dwordCount: 8},
		{name: "__imageDstRegionOrigin", offset: 10, dwordCount: 2},
		{name: "__imageDstRegionSize", offset: 12, dwordCount: 2},
		{name: "__imageSrcRegionOrigins", offset: 14, dwordCount: 8},
		{name: "__imageSrcRegionSizes", offset: 22, dwordCount: 8},
		{name: "__projectionMatrix", offset: 30, dwordCount: 16},
	}
	if got, want := len(internal), len(want); got != want {
		t.Fatalf("len(internal): got: %d, want: %d", got, want)
	}
	for i, w := range want {
		u := internal[i]
		if u.Name != w.name {
			t.Errorf("internal[%d].Name: got: %s, want: %s", i, u.Name, w.name)
		}
		if u.Offset != w.offset {
			t.Errorf("internal[%d].Offset: got: %d, want: %d", i, u.Offset, w.offset)
		}
		if u.DwordCount != w.dwordCount {
			t.Errorf("internal[%d].DwordCount: got: %d, want: %d", i, u.DwordCount, w.dwordCount)
		}
		if !u.Internal {
			t.Errorf("internal[%d].Internal: got: false, want: true", i)
		}
	}

	// __projectionMatrix is used by the engine's vertex entry point, and the source image
	// regions are used by the fixture's fragment stage through imageSrc0At.
	if got := internal[6]; !got.UsedInVertex {
		t.Errorf("__projectionMatrix must be used in the vertex stage, got: %+v", got)
	}
	if got := internal[4]; !got.UsedInFragment {
		t.Errorf("__imageSrcRegionOrigins must be used in the fragment stage, got: %+v", got)
	}
	if got := internal[5]; !got.UsedInFragment {
		t.Errorf("__imageSrcRegionSizes must be used in the fragment stage, got: %+v", got)
	}
}

func TestShaderReflectionPlanTextures(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	textures, err := plan.Textures()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := len(textures), graphics.ShaderSrcImageCount; got != want {
		t.Fatalf("len(textures): got: %d, want: %d", got, want)
	}
	for i, tex := range textures {
		if got, want := tex.Slot, i; got != want {
			t.Errorf("textures[%d].Slot: got: %d, want: %d", i, got, want)
		}
		if tex.UsedInVertex {
			t.Errorf("textures[%d].UsedInVertex: got: true, want: false", i)
		}
		// The fixture reads only the 0th source image.
		if got, want := tex.UsedInFragment, i == 0; got != want {
			t.Errorf("textures[%d].UsedInFragment: got: %t, want: %t", i, got, want)
		}
	}
}

func TestShaderReflectionPlanUniform(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	u, err := plan.Uniform("Color")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := u.Type.Kind, ebiten.ShaderValueKindVec4; got != want {
		t.Errorf("u.Type.Kind: got: %s, want: %s", got, want)
	}
	if got, want := u.Offset, 1; got != want {
		t.Errorf("u.Offset: got: %d, want: %d", got, want)
	}

	if _, err := plan.Uniform("NoSuchUniform"); err == nil {
		t.Error("plan.Uniform must return an error for an unknown name")
	}
	// Internal uniforms are not user uniforms.
	if _, err := plan.Uniform("__projectionMatrix"); err == nil {
		t.Error("plan.Uniform must return an error for an internal uniform name")
	}
}

// TestShaderReflectionPlanBindingBytes compares the plan's offsets and default values with the
// actual uniform block encoding produced for a draw call.
func TestShaderReflectionPlanBindingBytes(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	values := map[string]any{
		"Factor":    2.0,
		"Color":     []float32{1, 0.5, 0.25, 1},
		"Weights":   []float32{4, 3, 2, 1},
		"Count":     3,
		"Flag":      true,
		"UnusedVec": []float32{7, 8, 9},
	}
	want := []uint32{
		math.Float32bits(2),    // Factor
		math.Float32bits(1),    // Color
		math.Float32bits(0.5),  //
		math.Float32bits(0.25), //
		math.Float32bits(1),    //
		math.Float32bits(4),    // Weights
		math.Float32bits(3),    //
		math.Float32bits(2),    //
		math.Float32bits(1),    //
		3,                      // Count
		1,                      // Flag
		math.Float32bits(7),    // UnusedVec
		math.Float32bits(8),    //
		math.Float32bits(9),    //
	}

	got := s.AppendUniformsForTesting(nil, values)
	if len(got) != len(want) {
		t.Fatalf("len(got): got: %d, want: %d", len(got), len(want))
	}
	if got, want := mustUniformDwordCount(t, plan), len(want); got != want {
		t.Fatalf("plan.UniformDwordCount(): got: %d, want: %d", got, want)
	}

	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}
	for _, u := range uniforms {
		for i := range u.DwordCount {
			if got, want := got[u.Offset+i], want[u.Offset+i]; got != want {
				t.Errorf("the encoded value of %s at offset %d: got: %d, want: %d", u.Name, u.Offset+i, got, want)
			}
		}
	}

	// With no values given, the encoding must equal the concatenation of the plan's default
	// values.
	got = s.AppendUniformsForTesting(nil, nil)
	var defaults []uint32
	for _, u := range uniforms {
		defaults = append(defaults, u.Defaults...)
	}
	if len(got) != len(defaults) {
		t.Fatalf("len(got): got: %d, want: %d", len(got), len(defaults))
	}
	for i := range got {
		if got[i] != defaults[i] {
			t.Errorf("the default encoding at %d: got: %d, want: %d", i, got[i], defaults[i])
		}
	}
}

// TestShaderReflectionPlanCompilerIR compares the plan with the compiler's intermediate
// representation for the same source.
func TestShaderReflectionPlanCompilerIR(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	ir, _, err := legacyshader.CompileShader([]byte(reflectionFixtureShader))
	if err != nil {
		t.Fatal(err)
	}

	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}

	// The user uniforms must be the IR's uniforms without the preserved ones, in the same
	// order and with the same layout.
	irNames := ir.UniformNames[graphics.PreservedUniformVariablesCount:]
	irTypes := ir.Uniforms[graphics.PreservedUniformVariablesCount:]
	if got, want := len(uniforms), len(irNames); got != want {
		t.Fatalf("len(uniforms): got: %d, want: %d", got, want)
	}
	offset := 0
	for i, name := range irNames {
		u := uniforms[i]
		if u.Name != name {
			t.Errorf("uniforms[%d].Name: got: %s, want: %s", i, u.Name, name)
		}
		if got, want := u.Offset, offset; got != want {
			t.Errorf("uniforms[%d].Offset: got: %d, want: %d", i, got, want)
		}
		if got, want := u.DwordCount, irTypes[i].DwordCount(); got != want {
			t.Errorf("uniforms[%d].DwordCount: got: %d, want: %d", i, got, want)
		}
		offset += irTypes[i].DwordCount()
	}
}

// TestShaderReflectionPlanGLSL compares the plan with the GLSL sources the OpenGL backend
// generates for the same program. The GLSL backend declares the uniforms as U0, U1, ... in the
// program's uniform order — the internal uniforms followed by the user uniforms — and the
// textures as T0, T1, ... by their slots, so the plan's order and types must match the generated
// declarations of every GLSL flavor.
func TestShaderReflectionPlanGLSL(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	ir, _, err := legacyshader.CompileShader([]byte(reflectionFixtureShader))
	if err != nil {
		t.Fatal(err)
	}

	internal, err := plan.InternalUniforms()
	if err != nil {
		t.Fatal(err)
	}
	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}
	textures, err := plan.Textures()
	if err != nil {
		t.Fatal(err)
	}

	// The expected declarations in the plan's order: the internal uniforms U0-U6, then the
	// user uniforms U7-U12.
	wantDecls := []string{
		"uniform vec2 U0;",
		"uniform vec2 U1[4];",
		"uniform vec2 U2;",
		"uniform vec2 U3;",
		"uniform vec2 U4[4];",
		"uniform vec2 U5[4];",
		"uniform mat4 U6;",
		"uniform float U7;",    // Factor
		"uniform vec4 U8;",     // Color
		"uniform float U9[4];", // Weights
		"uniform int U10;",     // Count
		"uniform bool U11;",    // Flag
		"uniform vec3 U12;",    // UnusedVec: declared even though it is unused
	}
	if got, want := len(internal)+len(uniforms), len(wantDecls); got != want {
		t.Fatalf("len(internal)+len(uniforms): got: %d, want: %d", got, want)
	}

	for _, version := range []glsl.GLSLVersion{glsl.GLSLVersionDefault, glsl.GLSLVersionES300} {
		vs, fs := glsl.Compile(ir, version, graphics.ProjectionMatrixUniformVariableIndex)
		src := vs + fs
		for _, decl := range wantDecls {
			if !strings.Contains(src, decl) {
				t.Errorf("the GLSL source (version %d) must contain %q", version, decl)
			}
		}
		for _, tex := range textures {
			if decl := fmt.Sprintf("uniform sampler2D T%d;", tex.Slot); !strings.Contains(src, decl) {
				t.Errorf("the GLSL source (version %d) must contain %q", version, decl)
			}
		}
	}
}

// TestShaderReflectionPlanAliases uses the deprecated builtin aliases in the texel unit. The
// aliases refer to the engine's internal uniforms through different source-level names, but the
// plan must list each underlying uniform exactly once.
func TestShaderReflectionPlanAliases(t *testing.T) {
	// The default unit is the texel unit, in which imageDstTextureSize and imageSrcTextureSize
	// are deprecated aliases of the engine's internal uniforms.
	src := `package main

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	s := imageDstTextureSize() + imageSrcTextureSize()
	return vec4(s.x, s.y, 0, 1)
}
`
	s, err := ebiten.NewShader([]byte(src))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	internal, err := plan.InternalUniforms()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(internal), graphics.PreservedUniformVariablesCount; got != want {
		t.Fatalf("len(internal): got: %d, want: %d", got, want)
	}

	seen := map[string]int{}
	for _, u := range internal {
		seen[u.Name]++
	}
	for name, n := range seen {
		if n != 1 {
			t.Errorf("the internal uniform %s must be listed exactly once, but is listed %d times", name, n)
		}
	}

	// The uniforms behind the aliases must be marked as used in the fragment stage.
	used := map[string]bool{}
	for _, u := range internal {
		used[u.Name] = u.UsedInFragment
	}
	if !used["__imageDstTextureSize"] {
		t.Error("__imageDstTextureSize must be used in the fragment stage")
	}
	if !used["__imageSrcTextureSizes"] {
		t.Error("__imageSrcTextureSizes must be used in the fragment stage")
	}
}

func TestShaderReflectionPlanBadShader(t *testing.T) {
	cases := []struct {
		name string
		src  string
	}{
		{
			name: "not a shader",
			src:  `this is not a shader`,
		},
		{
			name: "duplicate uniform declaration",
			src: `//kage:unit pixels

package main

var U float
var U float

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return vec4(U)
}
`,
		},
		{
			name: "too large array",
			src: `//kage:unit pixels

package main

var U [4097]float

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return vec4(U[0])
}
`,
		},
		{
			name: "unknown uniform type",
			src: `//kage:unit pixels

package main

var U foo

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return dstPos
}
`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s, err := ebiten.NewShader([]byte(c.src))
			if err == nil {
				t.Fatal("NewShader must return an error but does not")
			}
			if s != nil {
				t.Error("NewShader must return a nil shader on an error")
			}
		})
	}
}

func TestShaderReflectionPlanDispose(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	s.Dispose()
	// Disposing twice must be a no-op.
	s.Dispose()

	if _, err := s.ReflectionPlan(); err == nil {
		t.Error("ReflectionPlan must return an error for a disposed shader")
	}
	if _, err := plan.Uniforms(); err == nil {
		t.Error("Uniforms must return an error after the shader is disposed")
	}
	if _, err := plan.Uniform("Factor"); err == nil {
		t.Error("Uniform must return an error after the shader is disposed")
	}
	if _, err := plan.InternalUniforms(); err == nil {
		t.Error("InternalUniforms must return an error after the shader is disposed")
	}
	if _, err := plan.Textures(); err == nil {
		t.Error("Textures must return an error after the shader is disposed")
	}
	if _, err := plan.UniformDwordCount(); err == nil {
		t.Error("UniformDwordCount must return an error after the shader is disposed")
	}
}

func TestShaderReflectionPlanImmutable(t *testing.T) {
	s := newReflectionFixtureShader(t)
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	// Mutate everything the queries return.
	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}
	uniforms[0].Name = "Mutated"
	uniforms[0].Offset = -1
	uniforms[0].Defaults[0] = 0xffffffff
	uniforms[2].Type.ArrayLength = 999
	*uniforms[2].Type.Element = ebiten.ShaderTypeInfo{Kind: ebiten.ShaderValueKindMat4}

	textures, err := plan.Textures()
	if err != nil {
		t.Fatal(err)
	}
	textures[0].Slot = 99
	textures[0].UsedInFragment = false

	// Re-query: the plan must be unaffected.
	uniforms2, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := uniforms2[0].Name, "Factor"; got != want {
		t.Errorf("uniforms2[0].Name: got: %s, want: %s", got, want)
	}
	if got, want := uniforms2[0].Offset, 0; got != want {
		t.Errorf("uniforms2[0].Offset: got: %d, want: %d", got, want)
	}
	if got, want := uniforms2[0].Defaults[0], uint32(0); got != want {
		t.Errorf("uniforms2[0].Defaults[0]: got: %d, want: %d", got, want)
	}
	if got, want := uniforms2[2].Type.ArrayLength, 4; got != want {
		t.Errorf("uniforms2[2].Type.ArrayLength: got: %d, want: %d", got, want)
	}
	if got, want := uniforms2[2].Type.Element.Kind, ebiten.ShaderValueKindFloat; got != want {
		t.Errorf("uniforms2[2].Type.Element.Kind: got: %s, want: %s", got, want)
	}

	textures2, err := plan.Textures()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := textures2[0].Slot, 0; got != want {
		t.Errorf("textures2[0].Slot: got: %d, want: %d", got, want)
	}
	if !textures2[0].UsedInFragment {
		t.Error("textures2[0].UsedInFragment: got: false, want: true")
	}
}

// TestShaderReflectionPlanDraw draws with uniforms discovered through the plan, exercising the
// graphics backend (OpenGL, or its software rasterizer under xvfb) end to end.
func TestShaderReflectionPlanDraw(t *testing.T) {
	src := `//kage:unit pixels

package main

var Tint vec4

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return Tint
}
`
	s, err := ebiten.NewShader([]byte(src))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := s.ReflectionPlan()
	if err != nil {
		t.Fatal(err)
	}

	// Discover the uniform through the plan instead of hard-coding its name.
	uniforms, err := plan.Uniforms()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(uniforms), 1; got != want {
		t.Fatalf("len(uniforms): got: %d, want: %d", got, want)
	}
	u := uniforms[0]
	if got, want := u.Type.Kind, ebiten.ShaderValueKindVec4; got != want {
		t.Fatalf("u.Type.Kind: got: %s, want: %s", got, want)
	}

	dst := ebiten.NewImage(1, 1)
	op := &ebiten.DrawRectShaderOptions{}
	op.Uniforms = map[string]any{
		u.Name: []float32{1, 0, 0, 1},
	}
	dst.DrawRectShader(1, 1, s, op)

	if got, want := dst.At(0, 0).(color.RGBA), (color.RGBA{R: 0xff, A: 0xff}); got != want {
		t.Errorf("dst.At(0, 0): got: %v, want: %v", got, want)
	}
}
