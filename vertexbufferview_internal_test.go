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
	"math"
	"slices"
	"testing"
	"unsafe"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
)

func testVertices() []Vertex {
	return []Vertex{
		{DstX: 1, DstY: 2, SrcX: 3, SrcY: 4, ColorR: 0.1, ColorG: 0.2, ColorB: 0.3, ColorA: 0.4, Custom0: 5, Custom1: 6, Custom2: 7, Custom3: 8},
		{DstX: 10, DstY: 20, SrcX: 30, SrcY: 40, ColorR: 0.5, ColorG: 0.6, ColorB: 0.7, ColorA: 0.8, Custom0: 50, Custom1: 60, Custom2: 70, Custom3: 80},
		{DstX: 100, DstY: 200, SrcX: 300, SrcY: 400, ColorR: 0.9, ColorG: 1, ColorB: 1.1, ColorA: 1.2, Custom0: 500, Custom1: 600, Custom2: 700, Custom3: 800},
		{DstX: -5, DstY: -6, SrcX: -7, SrcY: -8, ColorR: 0, ColorG: 0.25, ColorB: 0.5, ColorA: 0.75, Custom0: -1, Custom1: -2, Custom2: -3, Custom3: -4},
	}
}

func vertexFloats(vertices []Vertex) []float32 {
	if len(vertices) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&vertices[0])), len(vertices)*graphics.VertexFloatCount)
}

func TestVertexBufferViewConstruction(t *testing.T) {
	vertices := testVertices()

	t.Run("slice view", func(t *testing.T) {
		v := NewVertexBufferView(vertices, 1, 2)
		if got, want := v.Len(), 2; got != want {
			t.Errorf("Len() = %d, want %d", got, want)
		}
		if got, want := v.Offset(), 1; got != want {
			t.Errorf("Offset() = %d, want %d", got, want)
		}
		if got, want := v.Stride(), graphics.VertexFloatCount; got != want {
			t.Errorf("Stride() = %d, want %d", got, want)
		}
	})

	t.Run("interleaved packed", func(t *testing.T) {
		v := NewInterleavedVertexBufferView(vertexFloats(vertices), graphics.VertexFloatCount, 0, len(vertices))
		if v.Len() != len(vertices) {
			t.Errorf("Len() = %d, want %d", v.Len(), len(vertices))
		}
	})

	t.Run("interleaved strided", func(t *testing.T) {
		const stride = 16
		data := make([]float32, len(vertices)*stride)
		for i, vtx := range vertices {
			copy(data[i*stride:i*stride+graphics.VertexFloatCount], vertexFloats([]Vertex{vtx}))
			for p := graphics.VertexFloatCount; p < stride; p++ {
				data[i*stride+p] = float32(-1000 - i*stride - p)
			}
		}
		v := NewInterleavedVertexBufferView(data, stride, 1, 2)
		if got := v.Len(); got != 2 {
			t.Fatalf("Len() = %d, want 2", got)
		}
		got := make([]float32, 2*graphics.VertexFloatCount)
		v.packInto(got)
		want := vertexFloats(vertices[1:3])
		if !slices.Equal(got, want) {
			t.Errorf("packed strided data = %v, want %v", got, want)
		}
	})

	t.Run("planar", func(t *testing.T) {
		planar := makePlanarAttributes(vertices)
		v := NewPlanarVertexBufferView(planar, 2, 2)
		if v.Len() != 2 || v.Offset() != 2 {
			t.Fatalf("Len/Offset = %d/%d, want 2/2", v.Len(), v.Offset())
		}
		got := make([]float32, 2*graphics.VertexFloatCount)
		v.packInto(got)
		if want := vertexFloats(vertices[2:]); !slices.Equal(got, want) {
			t.Errorf("packed planar data = %v, want %v", got, want)
		}
	})
}

func makePlanarAttributes(vertices []Vertex) PlanarVertexAttributes {
	var a PlanarVertexAttributes
	n := len(vertices)
	a.DstX = make([]float32, n)
	a.DstY = make([]float32, n)
	a.SrcX = make([]float32, n)
	a.SrcY = make([]float32, n)
	a.ColorR = make([]float32, n)
	a.ColorG = make([]float32, n)
	a.ColorB = make([]float32, n)
	a.ColorA = make([]float32, n)
	a.Custom0 = make([]float32, n)
	a.Custom1 = make([]float32, n)
	a.Custom2 = make([]float32, n)
	a.Custom3 = make([]float32, n)
	for i, v := range vertices {
		a.DstX[i], a.DstY[i] = v.DstX, v.DstY
		a.SrcX[i], a.SrcY[i] = v.SrcX, v.SrcY
		a.ColorR[i], a.ColorG[i], a.ColorB[i], a.ColorA[i] = v.ColorR, v.ColorG, v.ColorB, v.ColorA
		a.Custom0[i], a.Custom1[i], a.Custom2[i], a.Custom3[i] = v.Custom0, v.Custom1, v.Custom2, v.Custom3
	}
	return a
}

func TestVertexBufferViewLayoutsMatchSlice(t *testing.T) {
	vertices := testVertices()
	ref := vertexFloats(vertices)

	// Packed interleaved view over the whole buffer.
	packed := NewInterleavedVertexBufferView(vertexFloats(vertices), graphics.VertexFloatCount, 0, len(vertices))
	got := make([]float32, len(ref))
	packed.packInto(got)
	if !slices.Equal(got, ref) {
		t.Errorf("packed: got %v want %v", got, ref)
	}

	// A view with a nonzero offset must produce exactly the matching sub-slice.
	sub := NewVertexBufferView(vertices, 2, 2)
	gotSub := make([]float32, 2*graphics.VertexFloatCount)
	sub.packInto(gotSub)
	if want := vertexFloats(vertices[2:]); !slices.Equal(gotSub, want) {
		t.Errorf("offset view: got %v want %v", gotSub, want)
	}

	// Planar view.
	planar := NewPlanarVertexBufferView(makePlanarAttributes(vertices), 0, len(vertices))
	gotPlanar := make([]float32, len(ref))
	planar.packInto(gotPlanar)
	if !slices.Equal(gotPlanar, ref) {
		t.Errorf("planar: got %v want %v", gotPlanar, ref)
	}
}

func TestVertexBufferViewDrawFloatsMatchSlice(t *testing.T) {
	vertices := testVertices()
	const dox, doy, sox, soy = 1.5, 2.5, 3.5, 4.5
	const cr, cg, cb, ca = 0.9, 0.8, 0.7, 0.6

	type constructor func() vertexSource
	cases := []struct {
		name string
		makeSource
	}{
		{
			name: "slice",
			makeSource: func() vertexSource {
				return vertexSlice(vertices)
			},
		},
		{
			name: "packed",
			makeSource: func() vertexSource {
				return NewInterleavedVertexBufferView(vertexFloats(vertices), graphics.VertexFloatCount, 0, len(vertices))
			},
		},
		{
			name: "strided",
			makeSource: func() vertexSource {
				const stride = 20
				data := make([]float32, len(vertices)*stride)
				for i, vtx := range vertices {
					copy(data[i*stride:], vertexFloats([]Vertex{vtx}))
				}
				return NewInterleavedVertexBufferView(data, stride, 0, len(vertices))
			},
		},
		{
			name: "planar",
			makeSource: func() vertexSource {
				return NewPlanarVertexBufferView(makePlanarAttributes(vertices), 0, len(vertices))
			},
		},
	}

	for _, straight := range []bool{true, false} {
		want := vertexSlice(vertices).appendVertexFloatsForDraw(nil, dox, doy, sox, soy, cr, cg, cb, ca, straight)
		for _, c := range cases {
			got := c.makeSource().appendVertexFloatsForDraw(nil, dox, doy, sox, soy, cr, cg, cb, ca, straight)
			if !slices.Equal(got, want) {
				t.Errorf("draw floats (%s, straight=%t): got %v want %v", c.name, straight, got, want)
			}
		}
	}

	shaderWant := vertexSlice(vertices).appendShaderVertexFloats(nil, dox, doy, sox, soy)
	for _, c := range cases {
		got := c.makeSource().appendShaderVertexFloats(nil, dox, doy, sox, soy)
		if !slices.Equal(got, shaderWant) {
			t.Errorf("shader floats (%s): got %v want %v", c.name, got, shaderWant)
		}
	}
}

type makeSource func() vertexSource

func TestVertexBufferViewSubViewAndClone(t *testing.T) {
	vertices := testVertices()
	v := NewVertexBufferView(vertices, 1, 3)

	sub := v.SubView(1, 2)
	if sub.Offset() != 2 || sub.Len() != 2 {
		t.Fatalf("SubView offset/length = %d/%d, want 2/2", sub.Offset(), sub.Len())
	}
	got := make([]float32, 2*graphics.VertexFloatCount)
	sub.packInto(got)
	if want := vertexFloats(vertices[2:4]); !slices.Equal(got, want) {
		t.Errorf("SubView data = %v, want %v", got, want)
	}

	// Clone takes a snapshot and is unaffected by later mutations of the source.
	clone := v.Clone()
	vertices[1].DstX = 9999
	cloneData := make([]float32, 3*graphics.VertexFloatCount)
	clone.packInto(cloneData)
	if cloneData[0] == 9999 {
		t.Error("Clone must not be affected by mutations of the original buffer")
	}
}

func TestVertexBufferViewMaterializeCopies(t *testing.T) {
	vertices := testVertices()
	v := NewVertexBufferView(vertices, 0, len(vertices))
	materialized := v.materializeVertices()
	materialized[0].DstX = 12345
	if vertices[0].DstX == 12345 {
		t.Error("materializeVertices must return a copy and never alias the backing []Vertex")
	}

	s := vertexSlice(vertices)
	m2 := s.materializeVertices()
	m2[0].DstX = 23456
	if vertices[0].DstX == 23456 {
		t.Error("vertexSlice.materializeVertices must return a copy")
	}
}

func TestVertexBufferViewConstructionErrors(t *testing.T) {
	vertices := testVertices()

	t.Run("empty slice buffer", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(nil, 0, 1) })
	})
	t.Run("empty interleaved buffer", func(t *testing.T) {
		assertPanics(t, func() { NewInterleavedVertexBufferView(nil, graphics.VertexFloatCount, 0, 1) })
	})
	t.Run("empty planar buffers", func(t *testing.T) {
		assertPanics(t, func() { NewPlanarVertexBufferView(PlanarVertexAttributes{}, 0, 1) })
	})
	t.Run("negative offset", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, -1, 1) })
	})
	t.Run("zero length", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, 0, 0) })
	})
	t.Run("negative length", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, 0, -1) })
	})
	t.Run("offset beyond capacity", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, len(vertices), 1) })
	})
	t.Run("range beyond capacity", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, len(vertices)-1, 2) })
	})
	t.Run("huge length", func(t *testing.T) {
		assertPanics(t, func() { NewVertexBufferView(vertices, 0, math.MaxInt) })
	})
	t.Run("stride too small", func(t *testing.T) {
		assertPanics(t, func() {
			NewInterleavedVertexBufferView(vertexFloats(vertices), graphics.VertexFloatCount-1, 0, 1)
		})
	})
	t.Run("strided range out of bounds", func(t *testing.T) {
		const stride = 16
		// Four vertex starts would fit, but the last vertex's 12 attributes do not.
		data := make([]float32, 3*stride+graphics.VertexFloatCount-1)
		assertPanics(t, func() { NewInterleavedVertexBufferView(data, stride, 0, 4) })
	})
	t.Run("strided vertex beyond data", func(t *testing.T) {
		// One full vertex fits, a second one would start inside but not fully fit.
		data := make([]float32, graphics.VertexFloatCount+4)
		assertPanics(t, func() { NewInterleavedVertexBufferView(data, graphics.VertexFloatCount+4, 0, 2) })
	})
	t.Run("huge stride cannot reach out of range", func(t *testing.T) {
		// With a huge stride only one vertex start fits, so asking for two is
		// rejected by the capacity check without any overflowing arithmetic.
		data := make([]float32, graphics.VertexFloatCount)
		assertPanics(t, func() { NewInterleavedVertexBufferView(data, math.MaxInt/2, 0, 2) })
	})
	t.Run("planar nil attribute", func(t *testing.T) {
		attrs := makePlanarAttributes(vertices)
		attrs.ColorB = nil
		assertPanics(t, func() { NewPlanarVertexBufferView(attrs, 0, 1) })
	})
	t.Run("planar mismatched lengths", func(t *testing.T) {
		attrs := makePlanarAttributes(vertices)
		attrs.Custom3 = attrs.Custom3[:len(attrs.Custom3)-1]
		assertPanics(t, func() { NewPlanarVertexBufferView(attrs, 0, 1) })
	})
	t.Run("subview out of range", func(t *testing.T) {
		v := NewVertexBufferView(vertices, 0, 2)
		assertPanics(t, func() { v.SubView(1, 2) })
	})
}

func assertPanics(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected a panic but got none")
		}
	}()
	fn()
}

func TestVertexBufferViewClamping(t *testing.T) {
	vertices := testVertices()
	v := NewVertexBufferView(vertices, 0, len(vertices))
	clamped := v.clampedVertexSource(2)
	if clamped.vertexCount() != 2 {
		t.Errorf("clamped count = %d, want 2", clamped.vertexCount())
	}
	got := make([]float32, 2*graphics.VertexFloatCount)
	clamped.(*VertexBufferView).packInto(got)
	if want := vertexFloats(vertices[:2]); !slices.Equal(got, want) {
		t.Errorf("clamped data = %v, want %v", got, want)
	}

	s := vertexSlice(vertices)
	if c := s.clampedVertexSource(2); c.vertexCount() != 2 {
		t.Errorf("slice clamped count = %d, want 2", c.vertexCount())
	}
	if c := s.clampedVertexSource(len(vertices) + 10); c.vertexCount() != len(vertices) {
		t.Errorf("clamping beyond length must not extend: got %d, want %d", c.vertexCount(), len(vertices))
	}
}
