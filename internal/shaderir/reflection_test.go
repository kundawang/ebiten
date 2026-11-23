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

package shaderir_test

import (
	"go/constant"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/shader"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir"
)

func uniformExpr(index int) shaderir.Expr {
	return shaderir.Expr{
		Type:  shaderir.UniformVariable,
		Index: index,
	}
}

func texelAtExpr(index int) shaderir.Expr {
	return shaderir.Expr{
		Type:        shaderir.BuiltinFuncExpr,
		BuiltinFunc: shaderir.TexelAt,
		Exprs: []shaderir.Expr{
			{
				Type:  shaderir.TextureVariable,
				Index: index,
			},
			{
				Type:  shaderir.NumberExpr,
				Const: constant.MakeInt64(0),
			},
		},
	}
}

func emptyProgram() *shaderir.Program {
	return &shaderir.Program{
		VertexFunc: shaderir.VertexFunc{
			Block: &shaderir.Block{},
		},
		FragmentFunc: shaderir.FragmentFunc{
			Block: &shaderir.Block{},
		},
	}
}

func TestReflectionUniforms(t *testing.T) {
	p := emptyProgram()
	p.UniformNames = []string{"B", "I", "F", "V2", "V3", "V4", "IV4", "M2", "M3", "M4"}
	p.Uniforms = []shaderir.Type{
		{Main: shaderir.Bool},
		{Main: shaderir.Int},
		{Main: shaderir.Float},
		{Main: shaderir.Vec2},
		{Main: shaderir.Vec3},
		{Main: shaderir.Vec4},
		{Main: shaderir.IVec4},
		{Main: shaderir.Mat2},
		{Main: shaderir.Mat3},
		{Main: shaderir.Mat4},
	}
	// F is used only in the vertex stage, and B only in the fragment stage. The other uniforms
	// are unused.
	p.VertexFunc.Block.Stmts = []shaderir.Stmt{exprStmt(uniformExpr(2))}
	p.FragmentFunc.Block.Stmts = []shaderir.Stmt{exprStmt(uniformExpr(0))}

	r, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	wantOffsets := []int{0, 1, 2, 3, 5, 8, 12, 16, 20, 29}
	wantDwords := []int{1, 1, 1, 2, 3, 4, 4, 4, 9, 16}
	if got, want := len(r.Uniforms), len(p.UniformNames); got != want {
		t.Fatalf("len(r.Uniforms): got: %d, want: %d", got, want)
	}
	for i, u := range r.Uniforms {
		if got, want := u.Name, p.UniformNames[i]; got != want {
			t.Errorf("r.Uniforms[%d].Name: got: %s, want: %s", i, got, want)
		}
		if got, want := u.Offset, wantOffsets[i]; got != want {
			t.Errorf("r.Uniforms[%d].Offset: got: %d, want: %d", i, got, want)
		}
		if got, want := u.DwordCount, wantDwords[i]; got != want {
			t.Errorf("r.Uniforms[%d].DwordCount: got: %d, want: %d", i, got, want)
		}
	}
	if got, want := r.UniformDwordCount, 45; got != want {
		t.Errorf("r.UniformDwordCount: got: %d, want: %d", got, want)
	}

	// Check the per-stage usage.
	if !r.Uniforms[2].UsedInVertex || r.Uniforms[2].UsedInFragment {
		t.Errorf("F must be used only in the vertex stage, got: %+v", r.Uniforms[2])
	}
	if r.Uniforms[0].UsedInVertex || !r.Uniforms[0].UsedInFragment {
		t.Errorf("B must be used only in the fragment stage, got: %+v", r.Uniforms[0])
	}
	for _, i := range []int{1, 3, 4, 5, 6, 7, 8, 9} {
		if r.Uniforms[i].UsedInVertex || r.Uniforms[i].UsedInFragment {
			t.Errorf("r.Uniforms[%d] must be unused, got: %+v", i, r.Uniforms[i])
		}
	}
}

func TestReflectionArrays(t *testing.T) {
	p := emptyProgram()
	p.UniformNames = []string{"A", "M", "N"}
	p.Uniforms = []shaderir.Type{
		{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Vec2}}, Length: 4},
		{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Mat4}}, Length: 3},
		{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Float}}, Length: 3}}, Length: 2},
	}

	r, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	wantOffsets := []int{0, 8, 56}
	wantDwords := []int{8, 48, 6}
	for i, u := range r.Uniforms {
		if got, want := u.Offset, wantOffsets[i]; got != want {
			t.Errorf("r.Uniforms[%d].Offset: got: %d, want: %d", i, got, want)
		}
		if got, want := u.DwordCount, wantDwords[i]; got != want {
			t.Errorf("r.Uniforms[%d].DwordCount: got: %d, want: %d", i, got, want)
		}
	}
	if got, want := r.UniformDwordCount, 62; got != want {
		t.Errorf("r.UniformDwordCount: got: %d, want: %d", got, want)
	}

	// The array lengths must be preserved.
	if got, want := r.Uniforms[0].Type.Length, 4; got != want {
		t.Errorf("r.Uniforms[0].Type.Length: got: %d, want: %d", got, want)
	}
	if got, want := r.Uniforms[2].Type.Sub[0].Length, 3; got != want {
		t.Errorf("r.Uniforms[2].Type.Sub[0].Length: got: %d, want: %d", got, want)
	}
}

func TestReflectionTextures(t *testing.T) {
	p := emptyProgram()
	p.TextureCount = 3
	p.VertexFunc.Block.Stmts = []shaderir.Stmt{exprStmt(texelAtExpr(1))}
	p.FragmentFunc.Block.Stmts = []shaderir.Stmt{
		exprStmt(texelAtExpr(0)),
		exprStmt(texelAtExpr(2)),
	}

	r, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := len(r.Textures), 3; got != want {
		t.Fatalf("len(r.Textures): got: %d, want: %d", got, want)
	}
	wantVertex := []bool{false, true, false}
	wantFragment := []bool{true, false, true}
	for i, tex := range r.Textures {
		if got, want := tex.Index, i; got != want {
			t.Errorf("r.Textures[%d].Index: got: %d, want: %d", i, got, want)
		}
		if got, want := tex.UsedInVertex, wantVertex[i]; got != want {
			t.Errorf("r.Textures[%d].UsedInVertex: got: %t, want: %t", i, got, want)
		}
		if got, want := tex.UsedInFragment, wantFragment[i]; got != want {
			t.Errorf("r.Textures[%d].UsedInFragment: got: %t, want: %t", i, got, want)
		}
	}
}

func TestReflectionStruct(t *testing.T) {
	p := emptyProgram()
	p.UniformNames = []string{"S", "F"}
	p.Uniforms = []shaderir.Type{
		{Main: shaderir.Struct, Sub: []shaderir.Type{
			{Main: shaderir.Float},
			{Main: shaderir.Vec4},
			{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Int}}, Length: 2},
		}},
		{Main: shaderir.Float},
	}

	r, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	// The struct must be expanded: 1 (float) + 4 (vec4) + 2 ([2]int) dwords.
	if got, want := r.Uniforms[0].DwordCount, 7; got != want {
		t.Errorf("r.Uniforms[0].DwordCount: got: %d, want: %d", got, want)
	}
	if got, want := r.Uniforms[1].Offset, 7; got != want {
		t.Errorf("r.Uniforms[1].Offset: got: %d, want: %d", got, want)
	}
	if got, want := r.UniformDwordCount, 8; got != want {
		t.Errorf("r.UniformDwordCount: got: %d, want: %d", got, want)
	}
}

func TestReflectionUsageThroughFunctions(t *testing.T) {
	p := emptyProgram()
	p.UniformNames = []string{"U0", "U1"}
	p.Uniforms = []shaderir.Type{
		{Main: shaderir.Float},
		{Main: shaderir.Float},
	}
	p.Funcs = []shaderir.Func{
		{
			Index: 0,
			Block: &shaderir.Block{
				Stmts: []shaderir.Stmt{exprStmt(uniformExpr(0))},
			},
		},
		{
			Index: 1,
			Block: &shaderir.Block{
				Stmts: []shaderir.Stmt{exprStmt(shaderir.Expr{
					Type:  shaderir.FunctionExpr,
					Index: 0,
				})},
			},
		},
		{
			Index: 2,
			Block: &shaderir.Block{
				Stmts: []shaderir.Stmt{exprStmt(uniformExpr(1))},
			},
		},
	}
	// The vertex stage calls F2, and the fragment stage calls F1, which calls F0.
	p.VertexFunc.Block.Stmts = []shaderir.Stmt{exprStmt(shaderir.Expr{
		Type:  shaderir.FunctionExpr,
		Index: 2,
	})}
	p.FragmentFunc.Block.Stmts = []shaderir.Stmt{exprStmt(shaderir.Expr{
		Type:  shaderir.FunctionExpr,
		Index: 1,
	})}

	r, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	if r.Uniforms[0].UsedInVertex || !r.Uniforms[0].UsedInFragment {
		t.Errorf("U0 must be used only in the fragment stage, got: %+v", r.Uniforms[0])
	}
	if !r.Uniforms[1].UsedInVertex || r.Uniforms[1].UsedInFragment {
		t.Errorf("U1 must be used only in the vertex stage, got: %+v", r.Uniforms[1])
	}
}

func TestReflectionCompiledShader(t *testing.T) {
	// A fixture compiled by the real compiler, with scalar and array uniforms, an unused
	// uniform, and a uniform reachable only through a user-defined function.
	src := `package main

var F float
var A [4]vec2
var Unused int

func helper() float {
	return F + A[1].x
}

func Vertex(position vec4, texCoord vec2, color vec4) (vec4, vec2, vec4) {
	return position, texCoord, color
}

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return vec4(helper())
}
`
	ir, err := shader.Compile([]byte(src), "Vertex", "Fragment", 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := ir.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := len(r.Uniforms), 3; got != want {
		t.Fatalf("len(r.Uniforms): got: %d, want: %d", got, want)
	}

	// F is a scalar at offset 0, and A is a [4]vec2 at offset 1.
	if got, want := r.Uniforms[0].Name, "F"; got != want {
		t.Errorf("r.Uniforms[0].Name: got: %s, want: %s", got, want)
	}
	if got, want := r.Uniforms[0].DwordCount, 1; got != want {
		t.Errorf("r.Uniforms[0].DwordCount: got: %d, want: %d", got, want)
	}
	if got, want := r.Uniforms[1].Offset, 1; got != want {
		t.Errorf("r.Uniforms[1].Offset: got: %d, want: %d", got, want)
	}
	if got, want := r.Uniforms[1].DwordCount, 8; got != want {
		t.Errorf("r.Uniforms[1].DwordCount: got: %d, want: %d", got, want)
	}
	if got, want := r.Uniforms[1].Type.Length, 4; got != want {
		t.Errorf("r.Uniforms[1].Type.Length: got: %d, want: %d", got, want)
	}

	// F and A are reachable from the fragment stage through helper. Unused is listed but
	// marked as unused.
	for _, i := range []int{0, 1} {
		if r.Uniforms[i].UsedInVertex || !r.Uniforms[i].UsedInFragment {
			t.Errorf("r.Uniforms[%d] must be used only in the fragment stage, got: %+v", i, r.Uniforms[i])
		}
	}
	if got, want := r.Uniforms[2].Name, "Unused"; got != want {
		t.Errorf("r.Uniforms[2].Name: got: %s, want: %s", got, want)
	}
	if r.Uniforms[2].UsedInVertex || r.Uniforms[2].UsedInFragment {
		t.Errorf("Unused must be unused, got: %+v", r.Uniforms[2])
	}
}

func TestReflectionErrors(t *testing.T) {
	programWithUniforms := func(names []string, types []shaderir.Type) *shaderir.Program {
		p := emptyProgram()
		p.UniformNames = names
		p.Uniforms = types
		return p
	}

	missingVertex := emptyProgram()
	missingVertex.VertexFunc.Block = nil

	missingFragment := emptyProgram()
	missingFragment.FragmentFunc.Block = nil

	negativeTextureCount := emptyProgram()
	negativeTextureCount.TextureCount = -1

	cases := []struct {
		name    string
		program *shaderir.Program
		err     string
	}{
		{
			name:    "nil program",
			program: nil,
			err:     "nil",
		},
		{
			name:    "missing vertex entry point",
			program: missingVertex,
			err:     "entry point",
		},
		{
			name:    "missing fragment entry point",
			program: missingFragment,
			err:     "entry point",
		},
		{
			name:    "mismatched uniform names and types",
			program: programWithUniforms([]string{"U"}, nil),
			err:     "1 uniform names but 0 uniform types",
		},
		{
			name: "duplicate uniform names",
			program: programWithUniforms(
				[]string{"U", "U"},
				[]shaderir.Type{{Main: shaderir.Float}, {Main: shaderir.Float}},
			),
			err: "duplicate uniform variable: U",
		},
		{
			name: "zero-length array",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Float}}, Length: 0}},
			),
			err: "uniform variable U",
		},
		{
			name: "negative-length array",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Float}}, Length: -3}},
			),
			err: "must be positive",
		},
		{
			name: "too large array",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Float}}, Length: shaderir.MaxReflectionArrayLength + 1}},
			),
			err: "exceeds the limit",
		},
		{
			name: "array without an element type",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Array, Length: 1}},
			),
			err: "exactly one element type",
		},
		{
			name: "array with two element types",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Float}, {Main: shaderir.Float}}, Length: 1}},
			),
			err: "exactly one element type",
		},
		{
			name: "texture uniform",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.Texture}},
			),
			err: "a texture cannot be a uniform variable",
		},
		{
			name: "none type uniform",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.None}},
			),
			err: "unknown type",
		},
		{
			name: "unknown type uniform",
			program: programWithUniforms(
				[]string{"U"},
				[]shaderir.Type{{Main: shaderir.BasicType(255)}},
			),
			err: "unknown type",
		},
		{
			name:    "negative texture count",
			program: negativeTextureCount,
			err:     "negative texture count",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r, err := c.program.Reflection()
			if err == nil {
				t.Fatalf("Reflection must return an error but does not")
			}
			if r != nil {
				t.Errorf("Reflection must return a nil reflection on an error")
			}
			if !strings.Contains(err.Error(), c.err) {
				t.Errorf("the error must contain %q, got: %v", c.err, err)
			}
		})
	}
}

func TestReflectionDoesNotShareState(t *testing.T) {
	p := emptyProgram()
	p.UniformNames = []string{"A"}
	p.Uniforms = []shaderir.Type{
		{Main: shaderir.Array, Sub: []shaderir.Type{{Main: shaderir.Vec2}}, Length: 4},
	}

	r0, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}

	// Mutating the reflection must affect neither the program nor other reflections.
	r0.Uniforms[0].Name = "Mutated"
	r0.Uniforms[0].Type.Main = shaderir.Mat4
	r0.Uniforms[0].Type.Sub[0].Main = shaderir.Mat4
	r0.Uniforms[0].Type.Length = 1
	r0.Uniforms[0].Offset = -1
	r0.Textures = append(r0.Textures, shaderir.TextureReflection{Index: 99})

	r1, err := p.Reflection()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := r1.Uniforms[0].Name, "A"; got != want {
		t.Errorf("r1.Uniforms[0].Name: got: %s, want: %s", got, want)
	}
	if got, want := r1.Uniforms[0].Type.Main, shaderir.Array; got != want {
		t.Errorf("r1.Uniforms[0].Type.Main: got: %v, want: %v", got, want)
	}
	if got, want := r1.Uniforms[0].Type.Sub[0].Main, shaderir.Vec2; got != want {
		t.Errorf("r1.Uniforms[0].Type.Sub[0].Main: got: %v, want: %v", got, want)
	}
	if got, want := r1.Uniforms[0].Type.Length, 4; got != want {
		t.Errorf("r1.Uniforms[0].Type.Length: got: %d, want: %d", got, want)
	}
	if got, want := r1.Uniforms[0].Offset, 0; got != want {
		t.Errorf("r1.Uniforms[0].Offset: got: %d, want: %d", got, want)
	}
	if got, want := len(r1.Textures), 0; got != want {
		t.Errorf("len(r1.Textures): got: %d, want: %d", got, want)
	}

	// The program itself must be intact.
	if got, want := p.Uniforms[0].Sub[0].Main, shaderir.Vec2; got != want {
		t.Errorf("p.Uniforms[0].Sub[0].Main: got: %v, want: %v", got, want)
	}
}
