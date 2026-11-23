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
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
)

const viewTestSize = 64

// viewQuad returns the four vertices of an axis-aligned quad in destination and
// source coordinates, tinted with the given straight-alpha color scale.
func viewQuad(dx, dy, sx, sy, w, h float32, r, g, b, a float32) []ebiten.Vertex {
	return []ebiten.Vertex{
		{DstX: dx, DstY: dy, SrcX: sx, SrcY: sy, ColorR: r, ColorG: g, ColorB: b, ColorA: a},
		{DstX: dx + w, DstY: dy, SrcX: sx + w, SrcY: sy, ColorR: r, ColorG: g, ColorB: b, ColorA: a},
		{DstX: dx, DstY: dy + h, SrcX: sx, SrcY: sy + h, ColorR: r, ColorG: g, ColorB: b, ColorA: a},
		{DstX: dx + w, DstY: dy + h, SrcX: sx + w, SrcY: sy + h, ColorR: r, ColorG: g, ColorB: b, ColorA: a},
	}
}

var viewQuadIndices32 = []uint32{0, 1, 2, 1, 2, 3}

// transparentTextureFixture returns a texture whose right half is fully
// transparent, so tests can prove texture coordinates (and not just colors)
// survive a VertexBufferView.
func transparentTextureFixture() *ebiten.Image {
	const s = 16
	img := ebiten.NewImage(s, s)
	pix := make([]byte, 4*s*s)
	for y := range s {
		for x := range s {
			idx := 4 * (y*s + x)
			if x < s/2 {
				pix[idx] = 0x40
				pix[idx+1] = 0x80
				pix[idx+2] = 0xc0
				pix[idx+3] = 0xff
			}
		}
	}
	img.WritePixels(pix)
	return img
}

func readViewPixels(t *testing.T, img *ebiten.Image) []byte {
	t.Helper()
	b := img.Bounds()
	pix := make([]byte, 4*b.Dx()*b.Dy())
	img.ReadPixels(pix)
	return pix
}

// TestVertexBufferViewDrawTrianglesPixels compares, pixel by pixel, the result
// of drawing through a VertexBufferView against the equivalent copied-slice
// DrawTriangles32 call.
func TestVertexBufferViewDrawTrianglesPixels(t *testing.T) {
	src := transparentTextureFixture()

	// One big shared interleaved buffer holding several quads back to back.
	meshes := [][]ebiten.Vertex{
		viewQuad(4, 6, 0, 0, 20, 16, 1, 1, 1, 1),
		viewQuad(30, 10, 8, 0, 24, 20, 1, 0.5, 0.25, 1),
		viewQuad(8, 36, 0, 8, 28, 18, 0.6, 0.7, 0.8, 0.5),
	}
	var shared []ebiten.Vertex
	meshOffsets := make([]int, 0, len(meshes))
	for _, m := range meshes {
		meshOffsets = append(meshOffsets, len(shared))
		shared = append(shared, m...)
	}

	cases := []struct {
		name string
		// meshView returns a view of the mesh at meshOffsets[meshIndex] within
		// the shared buffer.
		meshView func(meshIndex int) *ebiten.VertexBufferView
	}{
		{
			name: "slice-backed",
			meshView: func(meshIndex int) *ebiten.VertexBufferView {
				return ebiten.NewVertexBufferView(shared, meshOffsets[meshIndex], len(meshes[meshIndex]))
			},
		},
		{
			name: "packed float32 buffer",
			meshView: func(meshIndex int) *ebiten.VertexBufferView {
				data := verticesToFloats(shared)
				return ebiten.NewInterleavedVertexBufferView(data, graphics.VertexFloatCount, meshOffsets[meshIndex], len(meshes[meshIndex]))
			},
		},
		{
			name: "strided float32 buffer",
			meshView: func(meshIndex int) *ebiten.VertexBufferView {
				const stride = 16
				data := make([]float32, len(shared)*stride)
				for i := range shared {
					copy(data[i*stride:i*stride+graphics.VertexFloatCount], verticesToFloats(shared[i:i+1]))
				}
				return ebiten.NewInterleavedVertexBufferView(data, stride, meshOffsets[meshIndex], len(meshes[meshIndex]))
			},
		},
		{
			name: "planar buffers",
			meshView: func(meshIndex int) *ebiten.VertexBufferView {
				return ebiten.NewPlanarVertexBufferView(planarFromVertices(shared), meshOffsets[meshIndex], len(meshes[meshIndex]))
			},
		},
		{
			name: "subview of a full-range view",
			meshView: func(meshIndex int) *ebiten.VertexBufferView {
				full := ebiten.NewVertexBufferView(shared, 0, len(shared))
				if meshIndex == 0 {
					return full.SubView(0, len(meshes[0]))
				}
				return full.SubView(meshOffsets[meshIndex], len(meshes[meshIndex]))
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref := ebiten.NewImage(viewTestSize, viewTestSize)
			got := ebiten.NewImage(viewTestSize, viewTestSize)

			for meshIndex, mesh := range meshes {
				// Reference: an independent copied slice.
				ref.DrawTriangles32(mesh, viewQuadIndices32, src, nil)

				view := tc.meshView(meshIndex)
				got.DrawTriangles32WithView(view, viewQuadIndices32, src, nil)
			}

			if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
				t.Fatalf("pixels differ between slice and view draws (%s)", tc.name)
			}
		})
	}
}

// TestVertexBufferView16BitIndicesAndOffset covers uint16 indices, a nonzero
// view offset, and index values relative to the view's first vertex.
func TestVertexBufferView16BitIndicesAndOffset(t *testing.T) {
	src := transparentTextureFixture()

	// Prefix vertices are padding to prove the view offset is honored.
	padding := viewQuad(-50, -50, 0, 0, 8, 8, 1, 1, 1, 1)
	mesh := viewQuad(12, 14, 0, 0, 30, 24, 0.9, 0.8, 0.7, 1)
	shared := append(padding, mesh...)

	ref := ebiten.NewImage(viewTestSize, viewTestSize)
	got := ebiten.NewImage(viewTestSize, viewTestSize)

	ref.DrawTriangles(mesh, []uint16{0, 1, 2, 1, 2, 3}, src, nil)

	view := ebiten.NewVertexBufferView(shared, len(padding), len(mesh))
	got.DrawTrianglesWithView(view, []uint16{0, 1, 2, 1, 2, 3}, src, nil)

	if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
		t.Fatal("16-bit indexed view draw differs from the slice reference")
	}

	// Indices reaching outside the visible range must panic, exactly like slices.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("out-of-range index against a view did not panic")
			}
		}()
		got.DrawTriangles32WithView(view, []uint32{0, 1, 4}, src, nil)
	}()
}

// TestVertexBufferViewPremultipliedAndColorM exercises the color paths that
// differ between ColorScaleModeStraightAlpha and ColorScaleModePremultipliedAlpha,
// plus a ColorM, against copied-slice references.
func TestVertexBufferViewPremultipliedAndColorM(t *testing.T) {
	src := transparentTextureFixture()
	mesh := viewQuad(6, 8, 0, 0, 40, 30, 0.8, 0.4, 0.2, 0.75)

	for _, mode := range []ebiten.ColorScaleMode{ebiten.ColorScaleModeStraightAlpha, ebiten.ColorScaleModePremultipliedAlpha} {
		ref := ebiten.NewImage(viewTestSize, viewTestSize)
		got := ebiten.NewImage(viewTestSize, viewTestSize)

		op := &ebiten.DrawTrianglesOptions{ColorScaleMode: mode}
		ref.DrawTriangles32(mesh, viewQuadIndices32, src, op)

		view := ebiten.NewVertexBufferView(append([]ebiten.Vertex{}, mesh...), 0, len(mesh))
		got.DrawTriangles32WithView(view, viewQuadIndices32, src, op)

		if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
			t.Fatalf("mode %d: view draw differs from the slice reference", mode)
		}
	}

	t.Run("ColorM", func(t *testing.T) {
		ref := ebiten.NewImage(viewTestSize, viewTestSize)
		got := ebiten.NewImage(viewTestSize, viewTestSize)

		op := &ebiten.DrawTrianglesOptions{}
		op.ColorM.Scale(0.9, 0.8, 0.7, 0.9)
		op.ColorM.Translate(0.05, 0.02, 0.01, 0)
		ref.DrawTriangles32(mesh, viewQuadIndices32, src, op)

		view := ebiten.NewInterleavedVertexBufferView(verticesToFloats(mesh), graphics.VertexFloatCount, 0, len(mesh))
		got.DrawTriangles32WithView(view, viewQuadIndices32, src, op)

		if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
			t.Fatal("ColorM view draw differs from the slice reference")
		}
	})
}

// TestVertexBufferViewRotatedGeoM proves negative/rotated geometry produced via
// per-vertex DstX/DstY values is identical through both paths.
func TestVertexBufferViewRotatedGeoM(t *testing.T) {
	src := transparentTextureFixture()

	rotated := func() []ebiten.Vertex {
		angle := math.Pi / 3
		cos, sin := float32(math.Cos(angle)), float32(math.Sin(angle))
		vs := viewQuad(0, 0, 0, 0, 24, 18, 1, 1, 1, 1)
		cx, cy := float32(32), float32(32)
		for i := range vs {
			x, y := vs[i].DstX-cx, vs[i].DstY-cy
			vs[i].DstX = x*cos - y*sin + cx
			vs[i].DstY = x*sin + y*cos + cy
		}
		return vs
	}()

	ref := ebiten.NewImage(viewTestSize, viewTestSize)
	got := ebiten.NewImage(viewTestSize, viewTestSize)
	ref.DrawTriangles32(rotated, viewQuadIndices32, src, nil)

	view := ebiten.NewPlanarVertexBufferView(planarFromVertices(rotated), 0, len(rotated))
	got.DrawTriangles32WithView(view, viewQuadIndices32, src, nil)

	if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
		t.Fatal("rotated view draw differs from the slice reference")
	}
}

// TestVertexBufferViewShaderPixels checks shader drawing, including the custom
// attributes that only exist on the shader path.
func TestVertexBufferViewShaderPixels(t *testing.T) {
	s, err := ebiten.NewShader([]byte(`//kage:unit pixels

package main

func Fragment(dstPos vec4, src0Pos vec2, color vec4, custom vec4) vec4 {
	// Encode the custom attributes into the green/blue channels so any
	// difference is visible.
	return vec4(color.r, custom.r, custom.g, color.a)
}
`))
	if err != nil {
		t.Fatal(err)
	}

	mesh := viewQuad(8, 10, 0, 0, 44, 40, 0.7, 0.6, 0.5, 0.9)
	for i := range mesh {
		mesh[i].Custom0 = 0.1 * float32(i+1)
		mesh[i].Custom1 = 0.05 * float32(i+1)
	}

	ref := ebiten.NewImage(viewTestSize, viewTestSize)
	got := ebiten.NewImage(viewTestSize, viewTestSize)

	op := &ebiten.DrawTrianglesShaderOptions{}
	op.Images[0] = transparentTextureFixture()
	ref.DrawTrianglesShader32(mesh, viewQuadIndices32, s, op)

	view := ebiten.NewVertexBufferView(append([]ebiten.Vertex{}, mesh...), 0, len(mesh))
	got.DrawTrianglesShader32WithView(view, viewQuadIndices32, s, op)

	if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
		n, maxd := pixelDeltaStats(want, have)
		t.Fatalf("shader view draw differs from the slice reference (diff bytes: %d, max delta: %d)", n, maxd)
	}

	// Same again with the 16-bit-index entry point and a strided backing buffer.
	ref16 := ebiten.NewImage(viewTestSize, viewTestSize)
	got16 := ebiten.NewImage(viewTestSize, viewTestSize)
	ref16.DrawTrianglesShader(mesh, []uint16{0, 1, 2, 1, 2, 3}, s, op)

	const stride = 20
	data := make([]float32, len(mesh)*stride)
	for i := range mesh {
		copy(data[i*stride:], verticesToFloats(mesh[i:i+1]))
	}
	strided := ebiten.NewInterleavedVertexBufferView(data, stride, 0, len(mesh))
	got16.DrawTrianglesShaderWithView(strided, []uint16{0, 1, 2, 1, 2, 3}, s, op)

	if want, have := readViewPixels(t, ref16), readViewPixels(t, got16); !pixelsEqual(want, have, 1) {
		n, maxd := pixelDeltaStats(want, have)
		t.Fatalf("16-bit shader view draw differs from the slice reference (diff bytes: %d, max delta: %d)", n, maxd)
	}
}

func pixelDeltaStats(a, b []byte) (diffCount, maxDelta int) {
	for i := range a {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		if d > maxDelta {
			maxDelta = d
		}
		if a[i] != b[i] {
			diffCount++
		}
	}
	return diffCount, maxDelta
}

// TestVertexBufferViewSnapshotIsolation mutates the backing shared buffer after
// the draw call returns and confirms the already enqueued drawing is unaffected:
// views are snapshotted into engine-owned storage at draw time.
func TestVertexBufferViewSnapshotIsolation(t *testing.T) {
	src := transparentTextureFixture()
	shared := viewQuad(10, 10, 0, 0, 30, 30, 1, 0, 0, 1)

	ref := ebiten.NewImage(viewTestSize, viewTestSize)
	got := ebiten.NewImage(viewTestSize, viewTestSize)
	ref.DrawTriangles32(shared, viewQuadIndices32, src, nil)

	view := ebiten.NewVertexBufferView(shared, 0, len(shared))
	got.DrawTriangles32WithView(view, viewQuadIndices32, src, nil)

	// Mutate after the call: vertices already copied into the command queue.
	for i := range shared {
		shared[i].DstX += 100
		shared[i].ColorG = 1
	}

	if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
		t.Fatal("mutating the shared buffer after drawing changed the enqueued view draw")
	}
}

// TestVertexBufferViewTransparentTexture proves texture coordinates are carried
// by the view: sampling the transparent half of the fixture must stay empty.
func TestVertexBufferViewTransparentTexture(t *testing.T) {
	src := transparentTextureFixture()

	// Source x starts inside the transparent right half of the fixture.
	mesh := viewQuad(4, 4, 12, 0, 4, 12, 1, 1, 1, 1)

	ref := ebiten.NewImage(viewTestSize, viewTestSize)
	got := ebiten.NewImage(viewTestSize, viewTestSize)
	ref.DrawTriangles32(mesh, viewQuadIndices32, src, nil)

	view := ebiten.NewPlanarVertexBufferView(planarFromVertices(mesh), 0, len(mesh))
	got.DrawTriangles32WithView(view, viewQuadIndices32, src, nil)

	if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
		t.Fatal("transparent-texture view draw differs from the slice reference")
	}
	var transparent color.RGBA
	if clr := got.At(6, 10).(color.RGBA); clr != transparent {
		t.Errorf("transparent region sampled through a view was not empty: %v", clr)
	}
}

// TestVertexBufferViewEmptyIndices keeps parity with DrawTriangles32's behavior
// when no indices are given.
func TestVertexBufferViewEmptyIndices(t *testing.T) {
	src := transparentTextureFixture()
	mesh := viewQuad(0, 0, 0, 0, 8, 8, 1, 1, 1, 1)
	view := ebiten.NewVertexBufferView(mesh, 0, len(mesh))

	img := ebiten.NewImage(viewTestSize, viewTestSize)
	img.DrawTriangles32WithView(view, nil, src, nil)
	img.DrawTrianglesWithView(view, []uint16{}, src, nil)
	img.DrawTrianglesShader32WithView(view, nil, mustSolidShader(t), nil)

	var transparent color.RGBA
	if clr := img.At(4, 4).(color.RGBA); clr != transparent {
		t.Errorf("drawing with no indices must be a no-op, got %v", clr)
	}
}

// TestVertexBufferViewCommandPayload compares the packed float32 payload handed
// to the graphics command queue between an offset view and the equivalent
// copied sub-slice. Both draws use the same destination image so its internal
// atlas origin is identical.
func TestVertexBufferViewCommandPayload(t *testing.T) {
	src := transparentTextureFixture()
	mesh := viewQuad(5, 7, 1, 2, 33, 21, 0.8, 0.6, 0.4, 0.9)
	for i := range mesh {
		mesh[i].Custom0 = float32(i) * 1.5
		mesh[i].Custom3 = -float32(i)
	}

	t.Run("draw offset view", func(t *testing.T) {
		img := ebiten.NewImage(viewTestSize, viewTestSize)
		img.DrawTriangles32(mesh[1:], []uint32{0, 1, 2}, src, nil)
		want := img.TmpVerticesForTesting()

		view := ebiten.NewVertexBufferView(mesh, 1, 3)
		img.DrawTriangles32WithView(view, []uint32{0, 1, 2}, src, nil)
		if have := img.TmpVerticesForTesting(); !bytesEqualF32(want, have) {
			t.Errorf("queued vertex payload differs\nwant: %v\nhave: %v", want, have)
		}
	})

	t.Run("shader offset view", func(t *testing.T) {
		shader := mustSolidShader(t)
		op := &ebiten.DrawTrianglesShaderOptions{}
		op.Images[0] = src

		img := ebiten.NewImage(viewTestSize, viewTestSize)
		img.DrawTrianglesShader32(mesh[1:], []uint32{0, 1, 2}, shader, op)
		want := img.TmpVerticesForTesting()

		view := ebiten.NewPlanarVertexBufferView(planarFromVertices(mesh), 1, 3)
		img.DrawTrianglesShader32WithView(view, []uint32{0, 1, 2}, shader, op)
		if have := img.TmpVerticesForTesting(); !bytesEqualF32(want, have) {
			t.Errorf("queued shader vertex payload differs\nwant: %v\nhave: %v", want, have)
		}
	})
}

// TestVertexBufferViewStencilPath exercises the deprecated FillRule/AntiAlias
// path, which materializes the view into a fresh []Vertex before drawing.
func TestVertexBufferViewStencilPath(t *testing.T) {
	src := transparentTextureFixture()
	mesh := viewQuad(8, 9, 0, 0, 40, 32, 1, 0.7, 0.4, 1)

	for _, op := range []*ebiten.DrawTrianglesOptions{
		{FillRule: ebiten.FillRuleNonZero},
		{FillRule: ebiten.FillRuleEvenOdd},
		{AntiAlias: true},
	} {
		ref := ebiten.NewImage(viewTestSize, viewTestSize)
		got := ebiten.NewImage(viewTestSize, viewTestSize)
		ref.DrawTriangles32(mesh, viewQuadIndices32, src, op)

		view := ebiten.NewVertexBufferView(append([]ebiten.Vertex{}, mesh...), 0, len(mesh))
		got.DrawTriangles32WithView(view, viewQuadIndices32, src, op)

		if want, have := readViewPixels(t, ref), readViewPixels(t, got); !pixelsEqual(want, have, 1) {
			t.Fatalf("stencil path (%+v) differs between slice and view", op)
		}
	}
}

// TestVertexBufferViewIndexPanics mirrors the slice path's index validation for
// views.
func TestVertexBufferViewIndexPanics(t *testing.T) {
	src := transparentTextureFixture()
	mesh := viewQuad(0, 0, 0, 0, 8, 8, 1, 1, 1, 1)
	view := ebiten.NewVertexBufferView(mesh, 0, len(mesh))
	img := ebiten.NewImage(viewTestSize, viewTestSize)

	t.Run("32-bit out of range", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("out-of-range uint32 index did not panic")
			}
		}()
		img.DrawTriangles32WithView(view, []uint32{0, 1, 4}, src, nil)
	})
	t.Run("16-bit out of range", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("out-of-range uint16 index did not panic")
			}
		}()
		img.DrawTrianglesWithView(view, []uint16{0, 1, 4}, src, nil)
	})
	t.Run("not a multiple of three", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("index count not divisible by 3 did not panic")
			}
		}()
		img.DrawTriangles32WithView(view, []uint32{0, 1}, src, nil)
	})
}

// TestVertexBufferViewSubImageOrigin verifies the destination/source origin
// adjustments (originF32) applied on the slice path are identical on the view
// path when drawing onto and from a sub-image with a nonzero bounds origin.
func TestVertexBufferViewSubImageOrigin(t *testing.T) {
	src := transparentTextureFixture()
	sub := src.SubImage(image.Rect(4, 2, 12, 10)).(*ebiten.Image)

	mesh := viewQuad(2, 2, 4, 2, 8, 8, 1, 1, 1, 1)

	canvasRef := ebiten.NewImage(32, 32)
	canvasGot := ebiten.NewImage(32, 32)
	dstRef := canvasRef.SubImage(image.Rect(8, 6, 28, 26)).(*ebiten.Image)
	dstGot := canvasGot.SubImage(image.Rect(8, 6, 28, 26)).(*ebiten.Image)

	dstRef.DrawTriangles32(mesh, viewQuadIndices32, sub, nil)
	view := ebiten.NewVertexBufferView(append([]ebiten.Vertex{}, mesh...), 0, len(mesh))
	dstGot.DrawTriangles32WithView(view, viewQuadIndices32, sub, nil)

	if want, have := readViewPixels(t, canvasRef), readViewPixels(t, canvasGot); !pixelsEqual(want, have, 1) {
		t.Fatal("sub-image origin handling differs between slice and view")
	}
}

// TestVertexBufferViewNilView checks every WithView entry point rejects a nil
// view explicitly instead of panicking with a nil-pointer dereference.
func TestVertexBufferViewNilView(t *testing.T) {
	src := transparentTextureFixture()
	shader := mustSolidShader(t)
	img := ebiten.NewImage(viewTestSize, viewTestSize)
	indices32 := []uint32{0, 1, 2}
	indices16 := []uint16{0, 1, 2}

	calls := []struct {
		name string
		fn   func()
	}{
		{"DrawTrianglesWithView", func() { img.DrawTrianglesWithView(nil, indices16, src, nil) }},
		{"DrawTriangles32WithView", func() { img.DrawTriangles32WithView(nil, indices32, src, nil) }},
		{"DrawTrianglesShaderWithView", func() {
			img.DrawTrianglesShaderWithView(nil, indices16, shader, nil)
		}},
		{"DrawTrianglesShader32WithView", func() {
			img.DrawTrianglesShader32WithView(nil, indices32, shader, nil)
		}},
	}
	for _, c := range calls {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("nil view did not panic")
				}
				if msg, ok := r.(string); !ok || msg == "" {
					t.Fatalf("nil view panic is not an explicit message: %v", r)
				}
			}()
			c.fn()
		})
	}
}

func bytesEqualF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func mustSolidShader(t *testing.T) *ebiten.Shader {
	t.Helper()
	s, err := ebiten.NewShader([]byte(`//kage:unit pixels

package main

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return color
}
`))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func verticesToFloats(vertices []ebiten.Vertex) []float32 {
	// Construct via the public planar attributes to avoid unsafe in tests.
	attrs := planarFromVertices(vertices)
	n := len(vertices)
	out := make([]float32, 0, n*graphics.VertexFloatCount)
	plans := [12][]float32{
		attrs.DstX, attrs.DstY, attrs.SrcX, attrs.SrcY,
		attrs.ColorR, attrs.ColorG, attrs.ColorB, attrs.ColorA,
		attrs.Custom0, attrs.Custom1, attrs.Custom2, attrs.Custom3,
	}
	for i := range n {
		for _, p := range plans {
			out = append(out, p[i])
		}
	}
	return out
}

func planarFromVertices(vertices []ebiten.Vertex) ebiten.PlanarVertexAttributes {
	n := len(vertices)
	a := ebiten.PlanarVertexAttributes{
		DstX: make([]float32, n), DstY: make([]float32, n),
		SrcX: make([]float32, n), SrcY: make([]float32, n),
		ColorR: make([]float32, n), ColorG: make([]float32, n),
		ColorB: make([]float32, n), ColorA: make([]float32, n),
		Custom0: make([]float32, n), Custom1: make([]float32, n),
		Custom2: make([]float32, n), Custom3: make([]float32, n),
	}
	for i, v := range vertices {
		a.DstX[i], a.DstY[i] = v.DstX, v.DstY
		a.SrcX[i], a.SrcY[i] = v.SrcX, v.SrcY
		a.ColorR[i], a.ColorG[i], a.ColorB[i], a.ColorA[i] = v.ColorR, v.ColorG, v.ColorB, v.ColorA
		a.Custom0[i], a.Custom1[i], a.Custom2[i], a.Custom3[i] = v.Custom0, v.Custom1, v.Custom2, v.Custom3
	}
	return a
}

// pixelsEqual compares premultiplied-alpha RGBA pixel buffers, tolerating the
// per-channel GPU rounding differences that the rest of the suite also accounts
// for (see sameColors). A delta of 1 means only a single least-significant byte
// may differ.
func pixelsEqual(a, b []byte, delta int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		if d > delta {
			return false
		}
	}
	return true
}
