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
	"image"

	"github.com/hajimehoshi/ebiten/v2/internal/builtinshader"
	"github.com/hajimehoshi/ebiten/v2/internal/colormshader"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
)

// drawImageBatchMaxVertexCount is the maximum number of vertices in one internal draw call
// issued by (*Image).DrawImageBatch. A larger batch is split into multiple draw calls.
//
// This is a variable only so that tests can exercise the splitting path with a small value.
var drawImageBatchMaxVertexCount = graphicscommand.MaxVertexCount

// DrawImageBatchOptions represents options for a single item added to a DrawImageBatch.
//
// The fields have the same meanings as the corresponding fields of DrawImageOptions,
// except Clip, which is specific to DrawImageBatch.
type DrawImageBatchOptions struct {
	// GeoM is a geometry matrix to draw.
	// The default (zero) value is identity, which draws the image at (0, 0).
	GeoM GeoM

	// ColorScale is a scale of color.
	//
	// ColorScale is slightly different from colorm.ColorM's Scale in terms of alphas.
	// ColorScale is applied to premultiplied-alpha colors, while colorm.ColorM is applied to straight-alpha colors.
	// Thus, ColorM.Scale(r, g, b, a) is equal to ColorScale.Scale(r*a, g*a, b*a, a).
	//
	// The default (zero) value is identity, which is (1, 1, 1, 1).
	ColorScale ColorScale

	// ColorM is a color matrix to draw.
	// The default (zero) value is identity, which doesn't change any color.
	//
	// Deprecated: as of v2.5. Use ColorScale or the package colorm instead.
	ColorM ColorM

	// CompositeMode is a composite mode to draw.
	// The default (zero) value is CompositeModeCustom (Blend is used).
	//
	// Deprecated: as of v2.5. Use Blend instead.
	CompositeMode CompositeMode

	// Blend is a blending way of the source color and the destination color.
	// Blend is used only when CompositeMode is CompositeModeCustom.
	// The default (zero) value is the regular alpha blending.
	Blend Blend

	// Filter is a type of texture filter.
	// The default (zero) value is FilterNearest.
	Filter Filter

	// DisableMipmaps disables mipmaps.
	// When Filter is FilterLinear and GeoM shrinks the image, mipmaps are used by default.
	// Mipmap is useful to render a shrunk image with high quality.
	// However, mipmaps can be expensive, especially on mobiles.
	// When DisableMipmaps is true, mipmap is not used.
	// When Filter is not FilterLinear, DisableMipmaps is ignored.
	//
	// The default (zero) value is false.
	DisableMipmaps bool

	// Clip is a clipping rectangle in the destination's drawing coordinate system,
	// i.e. the same coordinate system in which GeoM operates. Only the pixels
	// inside the intersection of Clip and the destination image's bounds are drawn.
	//
	// The default (zero) value is an empty rectangle, which means no clipping.
	Clip image.Rectangle
}

// drawImageBatchItem is a single draw operation recorded in a DrawImageBatch.
type drawImageBatchItem struct {
	img     *Image
	options DrawImageBatchOptions
}

// DrawImageBatch is a reusable ordered list of image-drawing operations.
//
// A DrawImageBatch records drawing operations with Add and applies them to a
// destination image with (*Image).DrawImageBatch. Each item is equivalent to a
// DrawImage call, and the items are drawn in the order they were added. Drawing
// a batch is cheaper than calling DrawImage for each item: the batch reuses its
// internal vertex buffers across draws, and consecutive compatible items are
// merged into a single internal draw command before they reach the command queue.
//
// A DrawImageBatch is intended to be reused every frame: call Reset, add the
// frame's items with Add, and draw it with (*Image).DrawImageBatch.
//
// A DrawImageBatch must not be copied after the first use, and a single
// DrawImageBatch must not be used from multiple goroutines at the same time.
// Independent batches can be drawn from multiple goroutines concurrently; such
// draws are serialized internally in the same way as concurrent DrawImage calls.
type DrawImageBatch struct {
	items []drawImageBatchItem

	// vertices, indices and uniforms are scratch buffers reused across draws.
	// They must not be reused while a draw issued by this batch is still consuming
	// them; the internal draw calls copy them before returning, so resetting them
	// after each internal draw call is safe.
	vertices []float32
	indices  []uint32
	uniforms []uint32
}

// Reset removes all the items from the batch, keeping the allocated buffers for reuse.
//
// Reset is typically called at the beginning of each frame before adding the frame's items.
func (b *DrawImageBatch) Reset() {
	for i := range b.items {
		// Release the references to the source images so that they can be garbage collected.
		b.items[i] = drawImageBatchItem{}
	}
	b.items = b.items[:0]
}

// Len returns the number of items in the batch.
func (b *DrawImageBatch) Len() int {
	return len(b.items)
}

// Add appends a drawing operation of img to the batch.
//
// Add accepts the options. For details, see the document of DrawImageBatchOptions.
// The options are copied at this call; modifying them after this call doesn't
// affect the batch. The source image's pixels are adopted not at this call but
// when the batch is drawn, in the same way as DrawImage.
//
// When the given image img is nil or disposed, Add panics.
// When the given Filter or CompositeMode is invalid, Add panics.
func (b *DrawImageBatch) Add(img *Image, options *DrawImageBatchOptions) {
	if img == nil {
		panic("ebiten: the given image to DrawImageBatch.Add must not be nil")
	}
	if img.isDisposed() {
		panic("ebiten: the given image to DrawImageBatch.Add must not be disposed")
	}

	var o DrawImageBatchOptions
	if options != nil {
		o = *options
	}
	if o.Filter < FilterNearest || o.Filter > FilterPixelated {
		panic(fmt.Sprintf("ebiten: invalid filter: %d", o.Filter))
	}
	if o.CompositeMode != CompositeModeCustom {
		// blend panics on an invalid composite mode.
		o.CompositeMode.blend()
	}

	b.items = append(b.items, drawImageBatchItem{
		img:     img,
		options: o,
	})
}

// ensureVertices returns a slice of n floats backed by the batch's vertex buffer,
// keeping the already written contents.
func (b *DrawImageBatch) ensureVertices(n int) []float32 {
	if cap(b.vertices) < n {
		// Grow geometrically so that appending quads one by one is amortized O(1).
		vs := make([]float32, n, max(n, 2*cap(b.vertices)))
		copy(vs, b.vertices)
		b.vertices = vs
	}
	return b.vertices[:n]
}

// quadIndices returns 6*n indices for n consecutive quads backed by the batch's index buffer.
func (b *DrawImageBatch) quadIndices(n int) []uint32 {
	for len(b.indices) < 6*n {
		base := uint32(4 * (len(b.indices) / 6))
		// The same pattern as graphics.QuadIndices.
		b.indices = append(b.indices, base, base+1, base+2, base+1, base+2, base+3)
	}
	return b.indices[:6*n]
}

// DrawImageBatch draws all the items in the batch on the image i, in the order they were added.
//
// Each item is drawn exactly as if it were drawn by a DrawImage call with the
// equivalent options, including sub-images, non-integer coordinates, negative
// scales, premultiplied-alpha color scales, and mipmaps. In addition, each item
// is clipped by its Clip rectangle.
//
// Consecutive items that share the same source image, blend, filter, clip, and
// mipmap state are merged into as few internal draw commands as possible, while
// the drawing order is always preserved. Items that use a color matrix or
// mipmaps are never merged with other items, so that their results are exactly
// the same as sequential DrawImage calls.
//
// All the source images are validated before anything is drawn: if any source
// image was disposed after it was added, DrawImageBatch panics with the item's
// index and nothing is drawn. A failure never leaves a half-drawn batch.
//
// When the batch is nil, DrawImageBatch panics.
// When the batch is empty, or the image i is disposed and no source image is disposed,
// DrawImageBatch does nothing.
//
// An error in the graphics driver during the actual rendering is reported in the
// same way as an error caused by DrawImage: the game's Update returns the error.
func (i *Image) DrawImageBatch(batch *DrawImageBatch) {
	i.copyCheck()

	if batch == nil {
		panic("ebiten: the given batch to DrawImageBatch must not be nil")
	}
	if len(batch.items) == 0 {
		return
	}

	// Validate all the source images before drawing anything, so that a disposed
	// source image never leaves a half-submitted batch.
	for idx := range batch.items {
		if batch.items[idx].img.isDisposed() {
			panic(fmt.Sprintf("ebiten: the source image at index %d in the batch must not be disposed", idx))
		}
	}
	// TODO: Panic at v3 if a source image shares pixels with i, including any sub-image of the same image.
	// In v2, a self-draw that returns early or samples a mipmap level does not panic.
	if i.isDisposed() {
		return
	}

	i.invokeUsageCallbacks()
	i.updateAccessTime()

	offsetX, offsetY := i.adjustPosition(0, 0)
	dstBounds := i.adjustedBounds()

	// The state of the current run of consecutive compatible items, which is
	// flushed as one internal draw call.
	var (
		groupLen       int
		groupSrc       *bufferedImage
		groupBlend     graphicsdriver.Blend
		groupFilter    builtinshader.Filter
		groupDstRegion image.Rectangle
		groupSrcRegion image.Rectangle
	)

	flush := func() {
		if groupLen == 0 {
			return
		}
		vs := batch.vertices[:groupLen*4*graphics.VertexFloatCount]
		is := batch.quadIndices(groupLen)
		srcs := [graphics.ShaderSrcImageCount]*bufferedImage{groupSrc}
		shader := builtinShader(groupFilter, builtinshader.AddressUnsafe, false)
		i.image.DrawTriangles(srcs, vs, is, groupBlend, groupDstRegion, [graphics.ShaderSrcImageCount]image.Rectangle{groupSrcRegion}, shader.shader, nil, true)
		groupLen = 0
	}

	for idx := range batch.items {
		it := &batch.items[idx]
		o := &it.options
		img := it.img

		img.invokeUsageCallbacks()
		img.updateAccessTime()

		var blend graphicsdriver.Blend
		if o.CompositeMode == CompositeModeCustom {
			blend = o.Blend.internalBlend()
		} else {
			blend = o.CompositeMode.blend().internalBlend()
		}
		filter := builtinshader.Filter(o.Filter)

		geoM := o.GeoM
		if offsetX != 0 || offsetY != 0 {
			geoM.Translate(float64(offsetX), float64(offsetY))
		}
		a, b, c, d, tx, ty := geoM.elements32()
		det := a*d - b*c
		if det == 0 {
			// A degenerate matrix draws nothing, in the same way as DrawImage.
			continue
		}

		bounds := img.Bounds()
		sx0, sy0 := img.adjustPosition(bounds.Min.X, bounds.Min.Y)
		sx1, sy1 := img.adjustPosition(bounds.Max.X, bounds.Max.Y)

		colorm, cr, cg, cb, ca := colorMToScale(o.ColorM.affineColorM())
		cr, cg, cb, ca = o.ColorScale.apply(cr, cg, cb, ca)
		useColorM := !colorm.IsIdentity()

		dr := dstBounds
		if !o.Clip.Empty() {
			dr = dr.Intersect(o.Clip.Add(image.Pt(offsetX, offsetY)))
			if dr.Empty() {
				// The item is completely clipped out.
				continue
			}
		}

		skipMipmap := o.DisableMipmaps
		if !skipMipmap {
			skipMipmap = canSkipMipmap(det, filter)
		}

		if !useColorM && skipMipmap {
			// The item can be merged into the current run of compatible items.
			// The color scale and the geometry are recorded in the vertices, so
			// they don't have to match between merged items.
			if groupLen > 0 &&
				(img.image != groupSrc || blend != groupBlend || filter != groupFilter || dr != groupDstRegion ||
					groupLen*4 >= drawImageBatchMaxVertexCount) {
				flush()
			}
			if groupLen == 0 {
				groupSrc = img.image
				groupBlend = blend
				groupFilter = filter
				groupDstRegion = dr
				groupSrcRegion = image.Rect(sx0, sy0, sx1, sy1)
			} else {
				// The source region is used only for shader uniforms that the
				// built-in shader for DrawImage doesn't read. The union is the
				// most accurate single region for the merged items.
				groupSrcRegion = groupSrcRegion.Union(image.Rect(sx0, sy0, sx1, sy1))
			}
			vs := batch.ensureVertices((groupLen + 1) * 4 * graphics.VertexFloatCount)
			graphics.QuadVerticesFromSrcAndMatrix(vs[groupLen*4*graphics.VertexFloatCount:], float32(sx0), float32(sy0), float32(sx1), float32(sy1), a, b, c, d, tx, ty, cr, cg, cb, ca)
			groupLen++
			continue
		}

		// Items with a color matrix or mipmaps are drawn individually, so that
		// their uniforms and mipmap levels are exactly the same as a DrawImage call's.
		flush()
		vs := batch.ensureVertices(4 * graphics.VertexFloatCount)
		graphics.QuadVerticesFromSrcAndMatrix(vs, float32(sx0), float32(sy0), float32(sx1), float32(sy1), a, b, c, d, tx, ty, cr, cg, cb, ca)
		srcs := [graphics.ShaderSrcImageCount]*bufferedImage{img.image}
		shader := builtinShader(filter, builtinshader.AddressUnsafe, useColorM)
		batch.uniforms = batch.uniforms[:0]
		if useColorM {
			var body [16]float32
			var translation [4]float32
			colorm.Elements(body[:], translation[:])
			batch.uniforms = shader.appendUniforms(batch.uniforms, map[string]any{
				colormshader.UniformColorMBody:        body[:],
				colormshader.UniformColorMTranslation: translation[:],
			})
		}
		i.image.DrawTriangles(srcs, vs, graphics.QuadIndices(), blend, dr, [graphics.ShaderSrcImageCount]image.Rectangle{image.Rect(sx0, sy0, sx1, sy1)}, shader.shader, batch.uniforms, skipMipmap)
	}
	flush()
}
