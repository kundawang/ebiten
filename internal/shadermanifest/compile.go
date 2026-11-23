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

package shadermanifest

import (
	"fmt"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/legacyshader"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/glsl"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/hlsl"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/msl"
)

// legacyshaderSourceID computes the Ebitengine source ID of a Kage source in either unit, matching
// shadercollector and the runtime.
func legacyshaderSourceID(source []byte) (shaderir.SourceID, error) {
	return legacyshader.CalcSourceID(source)
}

// compiledShader holds the intermediate representation and the derived layout of one Kage source.
type compiledShader struct {
	ir           *shaderir.Program
	uniforms     []Uniform
	textureCount int
}

// compileSource compiles the Kage source and extracts the uniform layout and texture count. The layout
// is the same information Ebitengine uses to lay out uniform buffers at runtime, so recording it lets the
// manifest catch a source whose uniforms changed even if its source ID were otherwise stale.
func compileSource(source []byte) (*compiledShader, error) {
	ir, _, err := legacyshader.CompileShader(source)
	if err != nil {
		return nil, err
	}

	uniforms := make([]Uniform, 0, len(ir.UniformNames))
	for i, name := range ir.UniformNames {
		if i >= len(ir.Uniforms) {
			break
		}
		t := ir.Uniforms[i]
		uniforms = append(uniforms, Uniform{
			Index: i,
			Name:  name,
			Type:  t.String(),
			Size:  t.DwordCount(),
		})
	}

	return &compiledShader{
		ir:           ir,
		uniforms:     uniforms,
		textureCount: ir.TextureCount,
	}, nil
}

// buildTextArtifacts converts the compiled shader into the text artifacts for one backend, keyed by role.
// It returns CodeCompileFailed-wrapped errors via the caller; here a nil map means an unknown backend.
func (c *compiledShader) textArtifacts(backend Backend) (map[string][]byte, error) {
	switch backend {
	case BackendGLSL:
		vs, fs := glsl.Compile(c.ir, glsl.GLSLVersionDefault, graphics.ProjectionMatrixUniformVariableIndex)
		return map[string][]byte{
			roleVertex:   []byte(vs),
			roleFragment: []byte(fs),
		}, nil
	case BackendGLSLES:
		vs, fs := glsl.Compile(c.ir, glsl.GLSLVersionES300, graphics.ProjectionMatrixUniformVariableIndex)
		return map[string][]byte{
			roleVertex:   []byte(vs),
			roleFragment: []byte(fs),
		}, nil
	case BackendHLSL:
		vs, ps, _, _ := hlsl.Compile(c.ir)
		return map[string][]byte{
			roleVertex: []byte(vs),
			rolePixel:  []byte(ps),
		}, nil
	case BackendMSL:
		s := msl.Compile(c.ir)
		return map[string][]byte{
			roleShader: []byte(s),
		}, nil
	default:
		return nil, fmt.Errorf("shadermanifest: backend %s is not a text backend that can be generated", backend)
	}
}
