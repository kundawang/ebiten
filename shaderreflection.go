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

package ebiten

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/hajimehoshi/ebiten/v2/internal/shaderir"
)

// ShaderValueKind is the kind of a shader value type.
type ShaderValueKind int

const (
	ShaderValueKindBool ShaderValueKind = iota
	ShaderValueKindInt
	ShaderValueKindFloat
	ShaderValueKindVec2
	ShaderValueKindVec3
	ShaderValueKindVec4
	ShaderValueKindIVec2
	ShaderValueKindIVec3
	ShaderValueKindIVec4
	ShaderValueKindMat2
	ShaderValueKindMat3
	ShaderValueKindMat4
	ShaderValueKindArray
	ShaderValueKindStruct
)

// String returns a string representation of the kind.
func (k ShaderValueKind) String() string {
	switch k {
	case ShaderValueKindBool:
		return "bool"
	case ShaderValueKindInt:
		return "int"
	case ShaderValueKindFloat:
		return "float"
	case ShaderValueKindVec2:
		return "vec2"
	case ShaderValueKindVec3:
		return "vec3"
	case ShaderValueKindVec4:
		return "vec4"
	case ShaderValueKindIVec2:
		return "ivec2"
	case ShaderValueKindIVec3:
		return "ivec3"
	case ShaderValueKindIVec4:
		return "ivec4"
	case ShaderValueKindMat2:
		return "mat2"
	case ShaderValueKindMat3:
		return "mat3"
	case ShaderValueKindMat4:
		return "mat4"
	case ShaderValueKindArray:
		return "array"
	case ShaderValueKindStruct:
		return "struct"
	default:
		return fmt.Sprintf("ShaderValueKind(%d)", int(k))
	}
}

// ShaderTypeInfo describes the type of a uniform variable.
//
// For an array type, Element describes the element type and ArrayLength holds the number of
// elements. For a struct type, Fields holds the expanded field types in declaration order. For
// any other type, only Kind is set.
type ShaderTypeInfo struct {
	Kind        ShaderValueKind
	Element     *ShaderTypeInfo
	ArrayLength int
	Fields      []ShaderTypeInfo
}

// clone returns a deep copy of the type info.
func (t ShaderTypeInfo) clone() ShaderTypeInfo {
	c := t
	if t.Element != nil {
		e := t.Element.clone()
		c.Element = &e
	}
	if t.Fields != nil {
		c.Fields = make([]ShaderTypeInfo, len(t.Fields))
		for i, f := range t.Fields {
			c.Fields[i] = f.clone()
		}
	}
	return c
}

// shaderTypeInfoFromIR converts a shaderir.Type to a ShaderTypeInfo.
func shaderTypeInfoFromIR(t shaderir.Type) (ShaderTypeInfo, error) {
	switch t.Main {
	case shaderir.Bool:
		return ShaderTypeInfo{Kind: ShaderValueKindBool}, nil
	case shaderir.Int:
		return ShaderTypeInfo{Kind: ShaderValueKindInt}, nil
	case shaderir.Float:
		return ShaderTypeInfo{Kind: ShaderValueKindFloat}, nil
	case shaderir.Vec2:
		return ShaderTypeInfo{Kind: ShaderValueKindVec2}, nil
	case shaderir.Vec3:
		return ShaderTypeInfo{Kind: ShaderValueKindVec3}, nil
	case shaderir.Vec4:
		return ShaderTypeInfo{Kind: ShaderValueKindVec4}, nil
	case shaderir.IVec2:
		return ShaderTypeInfo{Kind: ShaderValueKindIVec2}, nil
	case shaderir.IVec3:
		return ShaderTypeInfo{Kind: ShaderValueKindIVec3}, nil
	case shaderir.IVec4:
		return ShaderTypeInfo{Kind: ShaderValueKindIVec4}, nil
	case shaderir.Mat2:
		return ShaderTypeInfo{Kind: ShaderValueKindMat2}, nil
	case shaderir.Mat3:
		return ShaderTypeInfo{Kind: ShaderValueKindMat3}, nil
	case shaderir.Mat4:
		return ShaderTypeInfo{Kind: ShaderValueKindMat4}, nil
	case shaderir.Array:
		if len(t.Sub) != 1 {
			return ShaderTypeInfo{}, fmt.Errorf("an array type must have exactly one element type, but has %d", len(t.Sub))
		}
		elem, err := shaderTypeInfoFromIR(t.Sub[0])
		if err != nil {
			return ShaderTypeInfo{}, err
		}
		return ShaderTypeInfo{
			Kind:        ShaderValueKindArray,
			Element:     &elem,
			ArrayLength: t.Length,
		}, nil
	case shaderir.Struct:
		fields := make([]ShaderTypeInfo, len(t.Sub))
		for i, st := range t.Sub {
			f, err := shaderTypeInfoFromIR(st)
			if err != nil {
				return ShaderTypeInfo{}, err
			}
			fields[i] = f
		}
		return ShaderTypeInfo{
			Kind:   ShaderValueKindStruct,
			Fields: fields,
		}, nil
	default:
		return ShaderTypeInfo{}, fmt.Errorf("unknown type: %s", t.String())
	}
}

// ShaderUniformInfo describes a single uniform variable of a compiled shader.
type ShaderUniformInfo struct {
	// Name is the name of the uniform variable in the shader source.
	Name string

	// Type is the type of the uniform variable.
	Type ShaderTypeInfo

	// Offset is the offset of the variable's first element in the uniform block, in 32-bit
	// words (dwords). For a user uniform variable, the block is the one encoded from
	// DrawRectShaderOptions.Uniforms or DrawTrianglesShaderOptions.Uniforms. For an internal
	// uniform variable, the block is the engine's whole uniform block.
	Offset int

	// DwordCount is the number of 32-bit words (dwords) the variable occupies in the uniform
	// block.
	DwordCount int

	// Defaults holds the variable's default values in the same raw 32-bit word encoding that
	// the uniform block uses. Kage does not allow initial values for uniform variables, so
	// Defaults is always zero-filled. Its length equals DwordCount.
	Defaults []uint32

	// UsedInVertex reports whether the vertex entry point can reach the variable.
	UsedInVertex bool

	// UsedInFragment reports whether the fragment entry point can reach the variable.
	UsedInFragment bool

	// Internal reports whether the variable is reserved by the engine. Internal uniform
	// variable names start with __ and cannot be set from DrawRectShaderOptions.Uniforms or
	// DrawTrianglesShaderOptions.Uniforms.
	Internal bool
}

// clone returns a deep copy of the uniform info.
func (u ShaderUniformInfo) clone() ShaderUniformInfo {
	c := u
	c.Type = u.Type.clone()
	c.Defaults = slices.Clone(u.Defaults)
	return c
}

func cloneShaderUniformInfos(us []ShaderUniformInfo) []ShaderUniformInfo {
	c := make([]ShaderUniformInfo, len(us))
	for i, u := range us {
		c[i] = u.clone()
	}
	return c
}

// ShaderTextureInfo describes a single texture binding of a compiled shader.
type ShaderTextureInfo struct {
	// Slot is the texture's binding slot, corresponding to an index of
	// DrawRectShaderOptions.Images or DrawTrianglesShaderOptions.Images.
	Slot int

	// UsedInVertex reports whether the vertex entry point can read the texture.
	UsedInVertex bool

	// UsedInFragment reports whether the fragment entry point can read the texture.
	UsedInFragment bool
}

// ShaderReflectionPlan is a read-only, precomputed view of a compiled [Shader]'s uniform
// variables and texture bindings: their names, types, array lengths, default values, binding
// offsets, and which shader stages use them.
//
// The plan is derived from the shader's intermediate representation, so its contents are
// identical for every graphics backend. The plan is a snapshot taken when
// [Shader.ReflectionPlan] is called, and every query returns copies: mutating a returned value
// never affects the plan, the shader, or other queries.
//
// A plan is tied to its shader's lifetime: once the shader is disposed, every query returns an
// error.
type ShaderReflectionPlan struct {
	shader *Shader

	uniforms          []ShaderUniformInfo
	internalUniforms  []ShaderUniformInfo
	textures          []ShaderTextureInfo
	uniformDwordCount int
}

// ReflectionPlan returns a [ShaderReflectionPlan] for the shader.
//
// ReflectionPlan returns an error if the shader is disposed.
func (s *Shader) ReflectionPlan() (*ShaderReflectionPlan, error) {
	if s.isDisposed() {
		return nil, errors.New("ebiten: cannot create a reflection plan from a disposed shader")
	}
	r, err := s.shader.Reflection()
	if err != nil {
		return nil, fmt.Errorf("ebiten: cannot create a reflection plan: %w", err)
	}

	p := &ShaderReflectionPlan{
		shader: s,
	}

	// Internal uniform variables, whose names start with __, come first in the program's uniform
	// block. User uniform variables follow, and their offsets are relative to the user uniform
	// block, which is the block DrawRectShaderOptions.Uniforms and
	// DrawTrianglesShaderOptions.Uniforms are encoded into.
	fullOffset := 0
	for _, u := range r.Uniforms {
		typ, err := shaderTypeInfoFromIR(u.Type)
		if err != nil {
			return nil, fmt.Errorf("ebiten: cannot create a reflection plan: uniform variable %s: %w", u.Name, err)
		}
		info := ShaderUniformInfo{
			Name:           u.Name,
			Type:           typ,
			DwordCount:     u.DwordCount,
			Defaults:       make([]uint32, u.DwordCount),
			UsedInVertex:   u.UsedInVertex,
			UsedInFragment: u.UsedInFragment,
		}
		if strings.HasPrefix(u.Name, "__") {
			info.Internal = true
			info.Offset = fullOffset
			p.internalUniforms = append(p.internalUniforms, info)
		} else {
			info.Offset = p.uniformDwordCount
			p.uniformDwordCount += u.DwordCount
			p.uniforms = append(p.uniforms, info)
		}
		fullOffset += u.DwordCount
	}

	for _, t := range r.Textures {
		p.textures = append(p.textures, ShaderTextureInfo{
			Slot:           t.Index,
			UsedInVertex:   t.UsedInVertex,
			UsedInFragment: t.UsedInFragment,
		})
	}

	return p, nil
}

// checkShader returns an error if the plan's shader was disposed.
func (p *ShaderReflectionPlan) checkShader() error {
	if p.shader.isDisposed() {
		return errors.New("ebiten: the shader of this reflection plan is disposed")
	}
	return nil
}

// Uniforms returns the shader's user uniform variables in declaration order. Uniform variables
// that are declared but never used are included.
//
// Uniforms returns an error if the shader was disposed.
func (p *ShaderReflectionPlan) Uniforms() ([]ShaderUniformInfo, error) {
	if err := p.checkShader(); err != nil {
		return nil, err
	}
	return cloneShaderUniformInfos(p.uniforms), nil
}

// Uniform returns the shader's user uniform variable with the given name.
//
// Uniform returns an error if the shader was disposed or has no user uniform variable with the
// name.
func (p *ShaderReflectionPlan) Uniform(name string) (ShaderUniformInfo, error) {
	if err := p.checkShader(); err != nil {
		return ShaderUniformInfo{}, err
	}
	for _, u := range p.uniforms {
		if u.Name == name {
			return u.clone(), nil
		}
	}
	return ShaderUniformInfo{}, fmt.Errorf("ebiten: the shader has no uniform variable named %q", name)
}

// InternalUniforms returns the engine's reserved uniform variables in declaration order. Their
// names start with __, and they cannot be set from DrawRectShaderOptions.Uniforms or
// DrawTrianglesShaderOptions.Uniforms.
//
// InternalUniforms returns an error if the shader was disposed.
func (p *ShaderReflectionPlan) InternalUniforms() ([]ShaderUniformInfo, error) {
	if err := p.checkShader(); err != nil {
		return nil, err
	}
	return cloneShaderUniformInfos(p.internalUniforms), nil
}

// Textures returns the shader's texture bindings by their slots.
//
// Textures returns an error if the shader was disposed.
func (p *ShaderReflectionPlan) Textures() ([]ShaderTextureInfo, error) {
	if err := p.checkShader(); err != nil {
		return nil, err
	}
	return slices.Clone(p.textures), nil
}

// UniformDwordCount returns the total size of the shader's user uniform block in 32-bit words
// (dwords). This is the number of words the shader's uniforms are encoded into from
// DrawRectShaderOptions.Uniforms or DrawTrianglesShaderOptions.Uniforms.
//
// UniformDwordCount returns an error if the shader was disposed.
func (p *ShaderReflectionPlan) UniformDwordCount() (int, error) {
	if err := p.checkShader(); err != nil {
		return 0, err
	}
	return p.uniformDwordCount, nil
}
