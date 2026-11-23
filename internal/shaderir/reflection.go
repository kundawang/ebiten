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

package shaderir

import (
	"fmt"
	"math"
)

// MaxReflectionArrayLength is the maximum array length [Program.Reflection] accepts for a uniform
// variable. This mirrors the compiler's limit: the backends emit code for each array element, so a
// huge length makes code generation take an enormous amount of time and memory.
const MaxReflectionArrayLength = 4096

// UniformReflection describes a single uniform variable of a compiled shader program.
type UniformReflection struct {
	// Name is the name of the uniform variable in the source.
	Name string

	// Type is the type of the uniform variable. Type is a deep copy: modifying it does not affect
	// the program the reflection was derived from.
	Type Type

	// Offset is the offset of the variable's first element in the program's uniform block, in
	// 32-bit words (dwords).
	Offset int

	// DwordCount is the number of 32-bit words (dwords) the variable occupies in the uniform
	// block.
	DwordCount int

	// UsedInVertex reports whether the vertex entry point can reach the variable, including
	// through user-defined functions.
	UsedInVertex bool

	// UsedInFragment reports whether the fragment entry point can reach the variable, including
	// through user-defined functions.
	UsedInFragment bool
}

// TextureReflection describes a single texture binding of a compiled shader program.
type TextureReflection struct {
	// Index is the texture's binding slot.
	Index int

	// UsedInVertex reports whether the vertex entry point can read the texture.
	UsedInVertex bool

	// UsedInFragment reports whether the fragment entry point can read the texture.
	UsedInFragment bool
}

// Reflection is a snapshot of a compiled shader program's externally visible resources: its
// uniform variables and texture bindings, with their names, types, sizes, offsets, and per-stage
// usage.
//
// A Reflection is derived only from the program's intermediate representation, so it is identical
// for every graphics backend. A Reflection owns all of its data: it shares no slices with the
// program it was derived from.
type Reflection struct {
	// Uniforms lists the uniform variables in declaration order. Uniform variables that are
	// declared but never used are included.
	Uniforms []UniformReflection

	// Textures lists the texture bindings by their slots.
	Textures []TextureReflection

	// UniformDwordCount is the total size of the program's uniform block in 32-bit words
	// (dwords).
	UniformDwordCount int
}

// Reflection returns a [Reflection] of the program.
//
// Reflection returns an error if the program is incomplete or inconsistent: a missing vertex or
// fragment entry point, mismatched uniform names and types, duplicated uniform names, a negative
// texture count, or a uniform type whose binary layout cannot be determined (an unknown basic
// type, a texture, or an array with a non-positive or too large length).
func (p *Program) Reflection() (*Reflection, error) {
	if p == nil {
		return nil, fmt.Errorf("shaderir: cannot reflect a nil program")
	}
	if p.VertexFunc.Block == nil || p.FragmentFunc.Block == nil {
		return nil, fmt.Errorf("shaderir: cannot reflect a program without both vertex and fragment entry points")
	}
	if len(p.UniformNames) != len(p.Uniforms) {
		return nil, fmt.Errorf("shaderir: cannot reflect a program with %d uniform names but %d uniform types", len(p.UniformNames), len(p.Uniforms))
	}
	if p.TextureCount < 0 {
		return nil, fmt.Errorf("shaderir: cannot reflect a program with a negative texture count: %d", p.TextureCount)
	}

	var vertexUsage, fragmentUsage resourceUsage
	vertexUsage.init(len(p.Uniforms), p.TextureCount)
	fragmentUsage.init(len(p.Uniforms), p.TextureCount)
	p.collectResourceUsage(p.VertexFunc.Block, &vertexUsage)
	p.collectResourceUsage(p.FragmentFunc.Block, &fragmentUsage)

	r := &Reflection{
		Uniforms: make([]UniformReflection, 0, len(p.Uniforms)),
		Textures: make([]TextureReflection, p.TextureCount),
	}

	names := map[string]struct{}{}
	offset := 0
	for i, name := range p.UniformNames {
		if _, ok := names[name]; ok {
			return nil, fmt.Errorf("shaderir: cannot reflect a program with a duplicate uniform variable: %s", name)
		}
		names[name] = struct{}{}

		n, err := reflectionDwordCount(p.Uniforms[i])
		if err != nil {
			return nil, fmt.Errorf("shaderir: cannot reflect the uniform variable %s: %w", name, err)
		}
		r.Uniforms = append(r.Uniforms, UniformReflection{
			Name:           name,
			Type:           p.Uniforms[i].clone(),
			Offset:         offset,
			DwordCount:     n,
			UsedInVertex:   vertexUsage.uniforms[i],
			UsedInFragment: fragmentUsage.uniforms[i],
		})
		offset += n
	}
	r.UniformDwordCount = offset

	for i := range r.Textures {
		r.Textures[i] = TextureReflection{
			Index:          i,
			UsedInVertex:   vertexUsage.textures[i],
			UsedInFragment: fragmentUsage.textures[i],
		}
	}

	return r, nil
}

// clone returns a deep copy of the type.
func (t Type) clone() Type {
	c := Type{
		Main:   t.Main,
		Length: t.Length,
	}
	if t.Sub != nil {
		c.Sub = make([]Type, len(t.Sub))
		for i, s := range t.Sub {
			c.Sub[i] = s.clone()
		}
	}
	return c
}

// reflectionDwordCount returns the number of 32-bit words (dwords) a value of the type occupies in
// a uniform block. Unlike [Type.DwordCount], reflectionDwordCount expands struct types and reports
// an error for types that cannot be laid out instead of returning -1.
func reflectionDwordCount(t Type) (int, error) {
	switch t.Main {
	case Bool, Int, Float:
		return 1, nil
	case Vec2, IVec2:
		return 2, nil
	case Vec3, IVec3:
		return 3, nil
	case Vec4, IVec4:
		return 4, nil
	case Mat2:
		return 4, nil
	case Mat3:
		return 9, nil
	case Mat4:
		return 16, nil
	case Array:
		if len(t.Sub) != 1 {
			return 0, fmt.Errorf("an array type must have exactly one element type, but has %d", len(t.Sub))
		}
		if t.Length <= 0 {
			return 0, fmt.Errorf("an array length must be positive, but is %d", t.Length)
		}
		if t.Length > MaxReflectionArrayLength {
			return 0, fmt.Errorf("an array length %d exceeds the limit %d", t.Length, MaxReflectionArrayLength)
		}
		elem, err := reflectionDwordCount(t.Sub[0])
		if err != nil {
			return 0, err
		}
		if elem > math.MaxInt/t.Length {
			return 0, fmt.Errorf("the array is too large: [%d]%s", t.Length, t.Sub[0].String())
		}
		return t.Length * elem, nil
	case Struct:
		var total int
		for _, st := range t.Sub {
			n, err := reflectionDwordCount(st)
			if err != nil {
				return 0, err
			}
			if n > math.MaxInt-total {
				return 0, fmt.Errorf("the struct is too large")
			}
			total += n
		}
		return total, nil
	case Texture:
		return 0, fmt.Errorf("a texture cannot be a uniform variable")
	default:
		return 0, fmt.Errorf("unknown type: %s", t.String())
	}
}

// resourceUsage records which uniform variables and textures are reachable from an entry point.
type resourceUsage struct {
	uniforms []bool
	textures []bool
}

func (u *resourceUsage) init(uniformCount, textureCount int) {
	u.uniforms = make([]bool, uniformCount)
	u.textures = make([]bool, textureCount)
}

// collectResourceUsage marks the uniform variables and textures reachable from block, following
// calls to user-defined functions.
func (p *Program) collectResourceUsage(block *Block, usage *resourceUsage) {
	indexToFunc := map[int]*Func{}
	for i := range p.Funcs {
		indexToFunc[p.Funcs[i].Index] = &p.Funcs[i]
	}

	visitedFuncs := map[int]struct{}{}
	var f func(expr *Expr)
	f = func(expr *Expr) {
		switch expr.Type {
		case UniformVariable:
			if 0 <= expr.Index && expr.Index < len(usage.uniforms) {
				usage.uniforms[expr.Index] = true
			}
		case TextureVariable:
			if 0 <= expr.Index && expr.Index < len(usage.textures) {
				usage.textures[expr.Index] = true
			}
		case FunctionExpr:
			if _, ok := visitedFuncs[expr.Index]; ok {
				return
			}
			visitedFuncs[expr.Index] = struct{}{}
			if fn, ok := indexToFunc[expr.Index]; ok {
				walkExprs(f, fn.Block)
			}
		}
	}
	walkExprs(f, block)
}
