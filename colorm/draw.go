// Copyright 2022 The Ebitengine Authors
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
	"fmt"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/internal/colormshader"
)

// DrawImageOptions represents options for DrawImage.
type DrawImageOptions struct {
	// GeoM is a geometry matrix to draw.
	// The default (zero) value is identity, which draws the image at (0, 0).
	GeoM ebiten.GeoM

	// ColorScale is a scale of color.
	//
	// ColorScale is slightly different from colorm.ColorM's Scale in terms of alphas.
	// ColorScale is applied to premultiplied-alpha colors, while colorm.ColorM is applied to straight-alpha colors.
	// Thus, ColorM.Scale(r, g, b, a) is equal to ColorScale.Scale(r*a, g*a, b*a, a).
	//
	// The default (zero) value is identity, which is (1, 1, 1, 1).
	ColorScale ebiten.ColorScale

	// Blend is a blending way of the source color and the destination color.
	// The default (zero) value is the regular alpha blending.
	Blend ebiten.Blend

	// Filter is a type of texture filter.
	// The default (zero) value is ebiten.FilterNearest.
	Filter ebiten.Filter

	// LUT is an optional lookup table used to apply the color transform.
	//
	// When LUT is non-nil, the colorM argument is ignored and the transform
	// is applied through the LUT instead of an exact per-pixel matrix
	// multiplication. This can be faster when the same transform is applied
	// repeatedly, at the cost of a small quantization error.
	//
	// The default (zero) value is nil, which uses the exact matrix mode.
	LUT *ColorTransformLUT
}

// DrawImage draws src onto dst.
//
// DrawImage is basically the same as ebiten.DrawImage, but with a color matrix.
//
// When the given Filter is invalid, DrawImage panics.
func DrawImage(dst, src *ebiten.Image, colorM ColorM, op *DrawImageOptions) {
	if op == nil {
		op = &DrawImageOptions{}
	}

	if src.Bounds().Empty() {
		return
	}

	opShader := &ebiten.DrawRectShaderOptions{}
	opShader.GeoM = op.GeoM
	opShader.ColorScale = op.ColorScale
	opShader.CompositeMode = ebiten.CompositeModeCustom
	opShader.Blend = op.Blend
	opShader.Images[0] = src
	if op.LUT != nil {
		// DrawRectShader requires all the source images to be the same size
		// as the rectangle, which the LUT image is not. Draw a quad with
		// DrawTrianglesShader instead; this is equivalent to DrawRectShader
		// with the GeoM and ColorScale applied to the vertices.
		snap, img := op.LUT.drawState()
		u := acquireLUTUniforms(snap)
		defer releaseLUTUniforms(u)

		w, h := src.Bounds().Dx(), src.Bounds().Dy()
		sx0, sy0 := float32(src.Bounds().Min.X), float32(src.Bounds().Min.Y)
		sx1, sy1 := float32(src.Bounds().Max.X), float32(src.Bounds().Max.Y)
		cr, cg, cb, ca := op.ColorScale.R(), op.ColorScale.G(), op.ColorScale.B(), op.ColorScale.A()
		vs := make([]ebiten.Vertex, 4)
		for i, p := range [][2]float32{{0, 0}, {float32(w), 0}, {0, float32(h)}, {float32(w), float32(h)}} {
			dx, dy := op.GeoM.Apply(float64(p[0]), float64(p[1]))
			vs[i].DstX = float32(dx)
			vs[i].DstY = float32(dy)
			vs[i].ColorR = cr
			vs[i].ColorG = cg
			vs[i].ColorB = cb
			vs[i].ColorA = ca
		}
		vs[0].SrcX, vs[0].SrcY = sx0, sy0
		vs[1].SrcX, vs[1].SrcY = sx1, sy0
		vs[2].SrcX, vs[2].SrcY = sx0, sy1
		vs[3].SrcX, vs[3].SrcY = sx1, sy1
		is := []uint16{0, 1, 2, 1, 2, 3}

		opTriangles := &ebiten.DrawTrianglesShaderOptions{}
		opTriangles.CompositeMode = ebiten.CompositeModeCustom
		opTriangles.Blend = op.Blend
		opTriangles.Uniforms = u.uniforms
		opTriangles.Images[0] = src
		opTriangles.Images[1] = img
		s := builtinShader(colormshader.Filter(op.Filter), colormshader.AddressUnsafe, true)
		dst.DrawTrianglesShader(vs, is, s, opTriangles)
		return
	}
	u := acquireUniforms(colorM)
	defer releaseUniforms(u)
	opShader.Uniforms = u.uniforms
	s := builtinShader(colormshader.Filter(op.Filter), colormshader.AddressUnsafe, false)
	dst.DrawRectShader(src.Bounds().Dx(), src.Bounds().Dy(), s, opShader)
}

// DrawTrianglesOptions represents options for DrawTriangles.
type DrawTrianglesOptions struct {
	// ColorScaleMode is the mode of color scales in vertices.
	// The default (zero) value is ebiten.ColorScaleModeStraightAlpha.
	ColorScaleMode ebiten.ColorScaleMode

	// Blend is a blending way of the source color and the destination color.
	// The default (zero) value is the regular alpha blending.
	Blend ebiten.Blend

	// Filter is a type of texture filter.
	// The default (zero) value is ebiten.FilterNearest.
	Filter ebiten.Filter

	// Address is a sampler address mode.
	// The default (zero) value is ebiten.AddressUnsafe.
	Address ebiten.Address

	// FillRule indicates the rule how an overlapped region is rendered.
	//
	// The rules FillRuleNonZero and FillRuleEvenOdd are useful when you want to render a complex polygon.
	// A complex polygon is a non-convex polygon like a concave polygon, a polygon with holes, or a self-intersecting polygon.
	// See examples/vector for actual usages.
	//
	// The default (zero) value is ebiten.FillRuleFillAll.
	//
	// Deprecated: as of v2.9.
	FillRule ebiten.FillRule

	// AntiAlias indicates whether the rendering uses anti-alias or not.
	// AntiAlias is useful especially when you pass vertices from the vector package.
	//
	// AntiAlias increases internal draw calls and might affect performance.
	// Use the build tag `ebitenginedebug` to check the number of draw calls if you care.
	//
	// The default (zero) value is false.
	//
	// Deprecated: as of v2.9.
	AntiAlias bool

	// LUT is an optional lookup table used to apply the color transform.
	//
	// When LUT is non-nil, the colorM argument is ignored and the transform
	// is applied through the LUT instead of an exact per-pixel matrix
	// multiplication. This can be faster when the same transform is applied
	// repeatedly, at the cost of a small quantization error.
	//
	// The default (zero) value is nil, which uses the exact matrix mode.
	LUT *ColorTransformLUT
}

// DrawTriangles draws triangles onto dst.
//
// DrawTriangles is basically the same as ebiten.DrawTriangles, but with a color matrix.
//
// When the given Filter, Address, or ColorScaleMode is invalid, DrawTriangles panics.
func DrawTriangles(dst *ebiten.Image, vertices []ebiten.Vertex, indices []uint16, img *ebiten.Image, colorM ColorM, op *DrawTrianglesOptions) {
	if op == nil {
		op = &DrawTrianglesOptions{}
	}

	switch op.ColorScaleMode {
	case ebiten.ColorScaleModeStraightAlpha, ebiten.ColorScaleModePremultipliedAlpha:
	default:
		panic(fmt.Sprintf("colorm: invalid color scale mode: %d", op.ColorScaleMode))
	}

	if op.ColorScaleMode == ebiten.ColorScaleModeStraightAlpha {
		vs := make([]ebiten.Vertex, len(vertices))
		copy(vs, vertices)
		for i := range vertices {
			vs[i].ColorR *= vs[i].ColorA
			vs[i].ColorG *= vs[i].ColorA
			vs[i].ColorB *= vs[i].ColorA
		}
		vertices = vs
	}

	opShader := &ebiten.DrawTrianglesShaderOptions{}
	opShader.CompositeMode = ebiten.CompositeModeCustom
	opShader.Blend = op.Blend
	opShader.FillRule = op.FillRule
	opShader.AntiAlias = op.AntiAlias
	opShader.Images[0] = img
	if op.LUT != nil {
		snap, lutImg := op.LUT.drawState()
		u := acquireLUTUniforms(snap)
		defer releaseLUTUniforms(u)
		opShader.Uniforms = u.uniforms
		opShader.Images[1] = lutImg
		s := builtinShader(colormshader.Filter(op.Filter), colormshader.Address(op.Address), true)
		dst.DrawTrianglesShader(vertices, indices, s, opShader)
		return
	}
	u := acquireUniforms(colorM)
	defer releaseUniforms(u)
	opShader.Uniforms = u.uniforms
	s := builtinShader(colormshader.Filter(op.Filter), colormshader.Address(op.Address), false)
	dst.DrawTrianglesShader(vertices, indices, s, opShader)
}
