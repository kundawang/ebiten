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
	"fmt"
	"slices"
	"unsafe"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
)

// PlanarVertexAttributes represents per-vertex attributes stored in separate,
// structure-of-arrays buffers.
//
// Every slice must be non-nil and must have the same length: each index i
// describes one vertex. A [VertexBufferView] created from a PlanarVertexAttributes
// reads these slices directly; the slices must not be modified while a draw call
// that uses the view is running.
type PlanarVertexAttributes struct {
	DstX []float32
	DstY []float32
	SrcX []float32
	SrcY []float32

	ColorR []float32
	ColorG []float32
	ColorB []float32
	ColorA []float32

	Custom0 []float32
	Custom1 []float32
	Custom2 []float32
	Custom3 []float32
}

// VertexBufferView is a read-only view over a vertex buffer that addresses a
// contiguous range of vertices with an offset and a length.
//
// A VertexBufferView lets UI code and map renderers share one large vertex
// buffer between multiple draw calls without copying sub-slices for every call.
// A view can be created either from an interleaved vertex buffer
// (a []Vertex value or a packed []float32 buffer with a stride), or from
// separate, per-attribute buffers (PlanarVertexAttributes).
//
// The view itself never modifies its backing buffers. The visible vertices are
// snapshotted into engine-owned storage when a draw call consumes the view, so
// mutating the backing buffers after a DrawTriangles*WithView call returns does
// not affect the already enqueued drawing. The backing buffers must not be
// modified concurrently with a draw call that reads the view.
//
// For example, several meshes can share one big vertex buffer and be drawn as
// separate sub-ranges:
//
//	shared := make([]ebiten.Vertex, totalVertices)
//	// ... fill shared once, and keep the per-mesh offsets and lengths.
//	view := ebiten.NewVertexBufferView(shared, meshOffset, meshLength)
//	dst.DrawTriangles32WithView(view, meshIndices, src, nil)
//
// A packed float32 buffer produced outside of Ebitengine can be used the same
// way, including padding between vertices:
//
//	view := ebiten.NewInterleavedVertexBufferView(data, strideInFloats, 0, vertexCount)
//	dst.DrawTriangles32WithView(view, indices, src, nil)
type VertexBufferView struct {
	// interleaved holds the packed float32 backing buffer for interleaved views.
	// interleaved is nil for views created from PlanarVertexAttributes.
	interleaved []float32

	// planar holds the per-attribute backing buffers for planar views.
	// Every entry is nil for interleaved views.
	planar [graphics.VertexFloatCount][]float32

	// stride is the number of float32 values between two consecutive vertices
	// in an interleaved buffer. stride is graphics.VertexFloatCount for planar views.
	stride int

	// offset is the index of the first visible vertex in the backing buffer.
	offset int

	// length is the number of visible vertices.
	length int
}

// NewVertexBufferView returns a read-only VertexBufferView over the given
// vertices, covering length vertices starting at offset.
//
// The vertices are referenced directly without copying; their memory is
// reinterpreted as packed float32 values, which is safe as [Vertex] is exactly
// graphics.VertexFloatCount (12) float32 values with no padding.
//
// NewVertexBufferView panics if vertices is empty, if offset or length is
// negative or zero where not allowed, or if offset plus length exceeds
// len(vertices).
func NewVertexBufferView(vertices []Vertex, offset, length int) *VertexBufferView {
	if len(vertices) == 0 {
		panic("ebiten: NewVertexBufferView requires a non-empty vertex buffer")
	}
	validateViewRange(offset, length, len(vertices))

	// Vertex is guaranteed to be exactly VertexFloatCount float32 values
	// (see the size assertion next to the Vertex declaration).
	data := unsafe.Slice((*float32)(unsafe.Pointer(&vertices[0])), len(vertices)*graphics.VertexFloatCount)

	return &VertexBufferView{
		interleaved: data,
		stride:      graphics.VertexFloatCount,
		offset:      offset,
		length:      length,
	}
}

// NewInterleavedVertexBufferView returns a read-only VertexBufferView over an
// interleaved float32 buffer.
//
// A vertex occupies graphics.VertexFloatCount (12) consecutive float32 values in
// the same order as the fields of [Vertex], and two consecutive vertices are
// stride float32 values apart. A stride larger than 12 leaves padding between
// vertices. The view covers length vertices starting at the vertex index offset.
//
// NewInterleavedVertexBufferView panics if data is empty, if stride is smaller
// than 12, if offset or length is negative or zero, or if the addressed range
// does not fully fit into data (including integer-overflow cases).
func NewInterleavedVertexBufferView(data []float32, stride, offset, length int) *VertexBufferView {
	if len(data) == 0 {
		panic("ebiten: NewInterleavedVertexBufferView requires a non-empty buffer")
	}
	if stride < graphics.VertexFloatCount {
		panic(fmt.Sprintf("ebiten: stride must be at least %d float32 values but was %d", graphics.VertexFloatCount, stride))
	}
	capacity := interleavedCapacity(len(data), stride)
	validateViewRange(offset, length, capacity)

	// validateViewRange already bounds offset+length-1 by the number of vertex
	// starts that fit, so the multiplication below cannot overflow. Re-check the
	// final vertex's full footprint explicitly: when the buffer tail is shorter
	// than a stride, a vertex start can fit while its 12 attributes do not.
	last := offset + length - 1
	if end := last*stride + graphics.VertexFloatCount; end > len(data) {
		panic(fmt.Sprintf("ebiten: vertex range (offset %d, length %d, stride %d) exceeds buffer length %d", offset, length, stride, len(data)))
	}

	return &VertexBufferView{
		interleaved: data,
		stride:      stride,
		offset:      offset,
		length:      length,
	}
}

// NewPlanarVertexBufferView returns a read-only VertexBufferView from separate,
// per-attribute buffers.
//
// The view covers length vertices starting at the vertex index offset. All
// attribute slices must be non-nil and have the same length.
//
// NewPlanarVertexBufferView panics if any attribute slice is nil, if the
// attribute slices have different lengths, if all of them are empty, or if the
// addressed range does not fit into the buffers.
func NewPlanarVertexBufferView(attributes PlanarVertexAttributes, offset, length int) *VertexBufferView {
	slices := [graphics.VertexFloatCount][]float32{
		attributes.DstX,
		attributes.DstY,
		attributes.SrcX,
		attributes.SrcY,
		attributes.ColorR,
		attributes.ColorG,
		attributes.ColorB,
		attributes.ColorA,
		attributes.Custom0,
		attributes.Custom1,
		attributes.Custom2,
		attributes.Custom3,
	}

	capacity := -1
	for i, s := range slices {
		if s == nil {
			panic(fmt.Sprintf("ebiten: planar vertex attribute %d must not be nil", i))
		}
		if capacity == -1 {
			capacity = len(s)
		} else if len(s) != capacity {
			panic(fmt.Sprintf("ebiten: planar vertex attribute %d has length %d but the first attribute has length %d", i, len(s), capacity))
		}
	}
	if capacity == 0 {
		panic("ebiten: NewPlanarVertexBufferView requires non-empty attribute buffers")
	}
	validateViewRange(offset, length, capacity)

	return &VertexBufferView{
		planar: slices,
		stride: graphics.VertexFloatCount,
		offset: offset,
		length: length,
	}
}

// validateViewRange panics unless [offset, offset+length) is a valid non-empty
// range within capacity. All arithmetic is overflow-safe.
func validateViewRange(offset, length, capacity int) {
	if offset < 0 {
		panic(fmt.Sprintf("ebiten: offset must not be negative but was %d", offset))
	}
	if length <= 0 {
		panic(fmt.Sprintf("ebiten: length must be positive but was %d", length))
	}
	if offset > capacity {
		panic(fmt.Sprintf("ebiten: offset %d exceeds the buffer capacity %d", offset, capacity))
	}
	if length > capacity-offset {
		panic(fmt.Sprintf("ebiten: offset %d plus length %d exceeds the buffer capacity %d", offset, length, capacity))
	}
}

// interleavedCapacity returns how many vertices with the given stride can start
// in a buffer of dataLen float32 values.
func interleavedCapacity(dataLen, stride int) int {
	if dataLen < graphics.VertexFloatCount {
		return 0
	}
	return (dataLen-graphics.VertexFloatCount)/stride + 1
}

// Len returns the number of vertices visible through the view.
func (v *VertexBufferView) Len() int {
	return v.length
}

// Offset returns the index of the first visible vertex in the backing buffer.
func (v *VertexBufferView) Offset() int {
	return v.offset
}

// Stride returns the number of float32 values between two consecutive vertices
// in the backing buffer. For planar views, Stride returns
// graphics.VertexFloatCount (12).
func (v *VertexBufferView) Stride() int {
	return v.stride
}

// SubView returns a new VertexBufferView covering length vertices of v starting
// at offset, where offset is relative to v's visible range.
//
// The returned view shares v's backing buffers. SubView panics if offset is
// negative, length is not positive, or the range exceeds v's length.
func (v *VertexBufferView) SubView(offset, length int) *VertexBufferView {
	validateViewRange(offset, length, v.length)
	cp := *v
	cp.offset = v.offset + offset
	cp.length = length
	return &cp
}

// Clone returns an independent VertexBufferView that contains a packed copy of
// v's visible vertices. Later mutations of the original backing buffers do not
// affect the returned view.
func (v *VertexBufferView) Clone() *VertexBufferView {
	data := make([]float32, v.length*graphics.VertexFloatCount)
	v.packInto(data)
	return &VertexBufferView{
		interleaved: data,
		stride:      graphics.VertexFloatCount,
		offset:      0,
		length:      v.length,
	}
}

// vertexCount implements vertexSource.
func (v *VertexBufferView) vertexCount() int {
	return v.length
}

// clampedVertexSource implements vertexSource.
func (v *VertexBufferView) clampedVertexSource(n int) vertexSource {
	if n >= v.length {
		return v
	}
	return v.SubView(0, n)
}

// materializeVertices implements vertexSource by copying the visible vertices
// into a fresh []Vertex. This is only used by the legacy stencil-buffer path,
// which mutates the returned vertices, so the caller's buffers must never be
// modified through it.
func (v *VertexBufferView) materializeVertices() []Vertex {
	vertices := make([]Vertex, v.length)
	data := unsafe.Slice((*float32)(unsafe.Pointer(&vertices[0])), v.length*graphics.VertexFloatCount)
	v.packInto(data)
	return vertices
}

// packInto writes the visible vertices as packed, untransformed
// graphics.VertexFloatCount-float records into dst, which must have capacity for
// at least length*graphics.VertexFloatCount values starting at offset 0.
func (v *VertexBufferView) packInto(dst []float32) {
	if v.interleaved != nil {
		for k := 0; k < v.length; k++ {
			base := (v.offset + k) * v.stride
			out := dst[k*graphics.VertexFloatCount : k*graphics.VertexFloatCount+graphics.VertexFloatCount]
			copy(out, v.interleaved[base:base+graphics.VertexFloatCount])
		}
		return
	}
	for k := 0; k < v.length; k++ {
		idx := v.offset + k
		out := dst[k*graphics.VertexFloatCount : k*graphics.VertexFloatCount+graphics.VertexFloatCount]
		for a := 0; a < graphics.VertexFloatCount; a++ {
			out[a] = v.planar[a][idx]
		}
	}
}

// appendVertexFloatsForDraw implements vertexSource for regular
// DrawTriangles draws. It applies the destination/source origin adjustment and
// the color-matrix/color-scale factors exactly like the []Vertex path.
//
// Each record still occupies graphics.VertexFloatCount floats: the first eight
// values are position, texture coordinates and colors, and the four custom
// values remain zero as they are ignored outside shader draws.
func (v *VertexBufferView) appendVertexFloatsForDraw(dst []float32, dox, doy, sox, soy, cr, cg, cb, ca float32, straightAlpha bool) []float32 {
	old := len(dst)
	dst = slices.Grow(dst, v.length*graphics.VertexFloatCount)[:old+v.length*graphics.VertexFloatCount]

	if v.interleaved != nil {
		for k := 0; k < v.length; k++ {
			s := v.interleaved[(v.offset+k)*v.stride:]
			out := dst[old+k*graphics.VertexFloatCount : old+k*graphics.VertexFloatCount+8]
			out[0] = s[0] - dox
			out[1] = s[1] - doy
			out[2] = s[2] - sox
			out[3] = s[3] - soy
			if straightAlpha {
				out[4] = s[4] * s[7] * cr
				out[5] = s[5] * s[7] * cg
				out[6] = s[6] * s[7] * cb
			} else {
				out[4] = s[4] * cr
				out[5] = s[5] * cg
				out[6] = s[6] * cb
			}
			out[7] = s[7] * ca
		}
		return dst
	}

	p := v.planar
	for k := 0; k < v.length; k++ {
		idx := v.offset + k
		out := dst[old+k*graphics.VertexFloatCount : old+k*graphics.VertexFloatCount+8]
		out[0] = p[0][idx] - dox
		out[1] = p[1][idx] - doy
		out[2] = p[2][idx] - sox
		out[3] = p[3][idx] - soy
		colorA := p[7][idx]
		if straightAlpha {
			out[4] = p[4][idx] * colorA * cr
			out[5] = p[5][idx] * colorA * cg
			out[6] = p[6][idx] * colorA * cb
		} else {
			out[4] = p[4][idx] * cr
			out[5] = p[5][idx] * cg
			out[6] = p[6][idx] * cb
		}
		out[7] = colorA * ca
	}
	return dst
}

// appendShaderVertexFloats implements vertexSource for DrawTrianglesShader
// draws. Only the destination and source origins are adjusted; all other
// attributes are passed through unchanged, including the custom attributes.
func (v *VertexBufferView) appendShaderVertexFloats(dst []float32, dox, doy, sox, soy float32) []float32 {
	old := len(dst)
	dst = slices.Grow(dst, v.length*graphics.VertexFloatCount)[:old+v.length*graphics.VertexFloatCount]

	if v.interleaved != nil {
		for k := 0; k < v.length; k++ {
			s := v.interleaved[(v.offset+k)*v.stride:]
			out := dst[old+k*graphics.VertexFloatCount : old+(k+1)*graphics.VertexFloatCount]
			out[0] = s[0] - dox
			out[1] = s[1] - doy
			out[2] = s[2] - sox
			out[3] = s[3] - soy
			copy(out[4:], s[4:12])
		}
		return dst
	}

	p := v.planar
	for k := 0; k < v.length; k++ {
		idx := v.offset + k
		out := dst[old+k*graphics.VertexFloatCount : old+(k+1)*graphics.VertexFloatCount]
		out[0] = p[0][idx] - dox
		out[1] = p[1][idx] - doy
		out[2] = p[2][idx] - sox
		out[3] = p[3][idx] - soy
		for a := 4; a < graphics.VertexFloatCount; a++ {
			out[a] = p[a][idx]
		}
	}
	return dst
}

// vertexSlice adapts a plain []Vertex to the vertexSource interface.
type vertexSlice []Vertex

func (s vertexSlice) vertexCount() int {
	return len(s)
}

func (s vertexSlice) clampedVertexSource(n int) vertexSource {
	if n >= len(s) {
		return s
	}
	return s[:n]
}

// materializeVertices returns a copy of the vertices. The copy is required
// because the legacy stencil-buffer path mutates the result.
func (s vertexSlice) materializeVertices() []Vertex {
	return slices.Clone(s)
}

// appendVertexFloatsForDraw mirrors the original per-vertex loops of
// DrawTriangles32, including the temporary sub-slice that removes boundary
// checks (#3103).
func (s vertexSlice) appendVertexFloatsForDraw(dst []float32, dox, doy, sox, soy, cr, cg, cb, ca float32, straightAlpha bool) []float32 {
	old := len(dst)
	dst = slices.Grow(dst, len(s)*graphics.VertexFloatCount)[:old+len(s)*graphics.VertexFloatCount]
	if straightAlpha {
		for i := range s {
			out := dst[old+i*graphics.VertexFloatCount : old+i*graphics.VertexFloatCount+8]
			out[0] = s[i].DstX - dox
			out[1] = s[i].DstY - doy
			out[2] = s[i].SrcX - sox
			out[3] = s[i].SrcY - soy
			out[4] = s[i].ColorR * s[i].ColorA * cr
			out[5] = s[i].ColorG * s[i].ColorA * cg
			out[6] = s[i].ColorB * s[i].ColorA * cb
			out[7] = s[i].ColorA * ca
		}
	} else {
		for i := range s {
			out := dst[old+i*graphics.VertexFloatCount : old+i*graphics.VertexFloatCount+8]
			out[0] = s[i].DstX - dox
			out[1] = s[i].DstY - doy
			out[2] = s[i].SrcX - sox
			out[3] = s[i].SrcY - soy
			out[4] = s[i].ColorR * cr
			out[5] = s[i].ColorG * cg
			out[6] = s[i].ColorB * cb
			out[7] = s[i].ColorA * ca
		}
	}
	return dst
}

func (s vertexSlice) appendShaderVertexFloats(dst []float32, dox, doy, sox, soy float32) []float32 {
	old := len(dst)
	dst = slices.Grow(dst, len(s)*graphics.VertexFloatCount)[:old+len(s)*graphics.VertexFloatCount]
	for i := range s {
		out := dst[old+i*graphics.VertexFloatCount : old+(i+1)*graphics.VertexFloatCount]
		out[0] = s[i].DstX - dox
		out[1] = s[i].DstY - doy
		out[2] = s[i].SrcX - sox
		out[3] = s[i].SrcY - soy
		out[4] = s[i].ColorR
		out[5] = s[i].ColorG
		out[6] = s[i].ColorB
		out[7] = s[i].ColorA
		out[8] = s[i].Custom0
		out[9] = s[i].Custom1
		out[10] = s[i].Custom2
		out[11] = s[i].Custom3
	}
	return dst
}

// vertexSource is the common input of the DrawTriangles* cores. A plain
// []Vertex is adapted via vertexSlice; a strided or planar buffer is represented
// by *VertexBufferView.
type vertexSource interface {
	vertexCount() int
	clampedVertexSource(n int) vertexSource
	materializeVertices() []Vertex
	appendVertexFloatsForDraw(dst []float32, dox, doy, sox, soy, cr, cg, cb, ca float32, straightAlpha bool) []float32
	appendShaderVertexFloats(dst []float32, dox, doy, sox, soy float32) []float32
}
