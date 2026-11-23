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

package colorm_test

import (
	"errors"
	"fmt"
	"image/color"
	"math"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/colorm"
)

// lutTestW and lutTestH are the fixture image dimensions.
const (
	lutTestW = 16
	lutTestH = 16
)

// grayscaleFixture returns a 16x16 image covering the full 0-255 gray ramp,
// fully opaque.
func grayscaleFixture() *ebiten.Image {
	pix := make([]byte, lutTestW*lutTestH*4)
	for y := 0; y < lutTestH; y++ {
		for x := 0; x < lutTestW; x++ {
			v := byte(y*lutTestW + x)
			off := (y*lutTestW + x) * 4
			pix[off+0] = v
			pix[off+1] = v
			pix[off+2] = v
			pix[off+3] = 0xff
		}
	}
	img := ebiten.NewImage(lutTestW, lutTestH)
	img.WritePixels(pix)
	return img
}

// transparentEdgeFixture returns an image with a horizontal alpha ramp
// (including fully transparent pixels) and varying hues, in premultiplied
// alpha.
func transparentEdgeFixture() *ebiten.Image {
	pix := make([]byte, lutTestW*lutTestH*4)
	for y := 0; y < lutTestH; y++ {
		for x := 0; x < lutTestW; x++ {
			a := int(x * 17) // 0, 17, ..., 255
			// Straight-alpha color varying with y.
			sr := y * 16
			sg := 128
			sb := 255 - y*16
			off := (y*lutTestW + x) * 4
			pix[off+0] = byte(sr * a / 255)
			pix[off+1] = byte(sg * a / 255)
			pix[off+2] = byte(sb * a / 255)
			pix[off+3] = byte(a)
		}
	}
	img := ebiten.NewImage(lutTestW, lutTestH)
	img.WritePixels(pix)
	return img
}

// extremeColorFixture returns an image with saturated primaries, black,
// white, fully transparent pixels, and low-alpha edge cases. All the pixels
// are valid premultiplied-alpha colors (RGB <= alpha).
func extremeColorFixture() *ebiten.Image {
	colors := [][4]byte{
		{0xff, 0x00, 0x00, 0xff},
		{0x00, 0xff, 0x00, 0xff},
		{0x00, 0x00, 0xff, 0xff},
		{0xff, 0xff, 0xff, 0xff},
		{0x00, 0x00, 0x00, 0xff},
		{0x00, 0x00, 0x00, 0x00},
		{0x01, 0x01, 0x01, 0x01},
		{0x80, 0x00, 0x80, 0x80},
		{0x40, 0x40, 0x00, 0x40},
		{0x00, 0x80, 0x80, 0x80},
		{0x7f, 0x7f, 0x7f, 0xff},
		{0x80, 0x80, 0x80, 0xff},
		{0xc0, 0x60, 0x00, 0xc0},
		{0x04, 0x06, 0x08, 0x08},
		{0xfe, 0xfe, 0xfe, 0xff},
		{0x02, 0x02, 0x02, 0x02},
	}
	pix := make([]byte, lutTestW*lutTestH*4)
	for i := 0; i < lutTestW*lutTestH; i++ {
		c := colors[i%len(colors)]
		copy(pix[i*4:], c[:])
	}
	img := ebiten.NewImage(lutTestW, lutTestH)
	img.WritePixels(pix)
	return img
}

// lutTestColorMs returns the color matrices under test, including extreme
// coefficients that push outputs out of the [0, 1] range.
func lutTestColorMs() map[string]colorm.ColorM {
	ms := map[string]colorm.ColorM{}

	var scale colorm.ColorM
	scale.Scale(0.2, 0.4, 0.6, 0.8)
	ms["scale"] = scale

	var hue colorm.ColorM
	hue.RotateHue(1.2)
	ms["rotate-hue"] = hue

	var gray colorm.ColorM
	gray.ChangeHSV(0, 0, 1)
	ms["grayscale"] = gray

	var tr colorm.ColorM
	tr.Scale(1.2, 0.9, 0.7, 1)
	tr.Translate(0.1, -0.05, 0.2, 0)
	ms["scale-translate"] = tr

	var extreme colorm.ColorM
	extreme.Scale(4, -1, 2, 1.5)
	extreme.Translate(-0.5, 0.5, 0, 0)
	ms["extreme"] = extreme

	var combo colorm.ColorM
	combo.RotateHue(0.6)
	var combo2 colorm.ColorM
	combo2.ChangeHSV(0.3, 1.5, 0.8)
	combo.Concat(combo2)
	ms["concat"] = combo

	return ms
}

// colorDiff returns the maximum per-channel absolute difference between two
// premultiplied colors, in 8-bit units.
func colorDiff(c0, c1 color.Color) int {
	r0, g0, b0, a0 := c0.RGBA()
	r1, g1, b1, a1 := c1.RGBA()
	d := 0
	for _, dd := range []int{
		abs(int(r0>>8) - int(r1>>8)),
		abs(int(g0>>8) - int(g1>>8)),
		abs(int(b0>>8) - int(b1>>8)),
		abs(int(a0>>8) - int(a1>>8)),
	} {
		if dd > d {
			d = dd
		}
	}
	return d
}

// TestLUTDrawImageVsColorM draws the fixtures with the exact matrix path and
// with the LUT path, and reports the per-pixel error between them.
func TestLUTDrawImageVsColorM(t *testing.T) {
	fixtures := map[string]*ebiten.Image{
		"grayscale":        grayscaleFixture(),
		"transparent-edge": transparentEdgeFixture(),
		"extreme-color":    extremeColorFixture(),
	}
	for _, opts := range []*colorm.LUTOptions{
		nil,
		{Size: 64},
		{Size: 64, ColorSpace: colorm.LUTColorSpaceSRGB},
	} {
		optsDesc := "default"
		tolerance := 4
		if opts != nil {
			optsDesc = fmt.Sprintf("size%d-cs%d", opts.Size, opts.ColorSpace)
			tolerance = 12
		}
		t.Run(optsDesc, func(t *testing.T) {
			for fname, src := range fixtures {
				for mname, cm := range lutTestColorMs() {
					t.Run(fname+"/"+mname, func(t *testing.T) {
						lut, err := colorm.NewColorTransformLUTFromColorM(cm, opts)
						if err != nil {
							t.Fatal(err)
						}

						dst0 := ebiten.NewImage(lutTestW, lutTestH)
						colorm.DrawImage(dst0, src, cm, nil)
						dst1 := ebiten.NewImage(lutTestW, lutTestH)
						colorm.DrawImage(dst1, src, cm, &colorm.DrawImageOptions{LUT: lut})

						maxErr := 0
						var sumErr int
						for y := 0; y < lutTestH; y++ {
							for x := 0; x < lutTestW; x++ {
								d := colorDiff(dst0.At(x, y), dst1.At(x, y))
								if d > maxErr {
									maxErr = d
								}
								sumErr += d
							}
						}
						meanErr := float64(sumErr) / (lutTestW * lutTestH)
						t.Logf("error vs exact matrix: max=%d/255 mean=%.3f/255 (LUT memory: %d bytes)",
							maxErr, meanErr, lut.MemorySize())
						if maxErr > tolerance {
							t.Errorf("max error %d/255 exceeds tolerance %d/255", maxErr, tolerance)
						}
					})
				}
			}
		})
	}
}

// TestLUTDrawTrianglesVsColorM exercises the DrawTriangles color path with a
// LUT against the exact matrix path.
func TestLUTDrawTrianglesVsColorM(t *testing.T) {
	src := extremeColorFixture()
	cm := lutTestColorMs()["concat"]

	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		t.Fatal(err)
	}

	vs := []ebiten.Vertex{
		{DstX: 0, DstY: 0, SrcX: 0, SrcY: 0, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		{DstX: lutTestW, DstY: 0, SrcX: lutTestW, SrcY: 0, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		{DstX: 0, DstY: lutTestH, SrcX: 0, SrcY: lutTestH, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		{DstX: lutTestW, DstY: lutTestH, SrcX: lutTestW, SrcY: lutTestH, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
	}
	is := []uint16{0, 1, 2, 1, 2, 3}

	dst0 := ebiten.NewImage(lutTestW, lutTestH)
	colorm.DrawTriangles(dst0, vs, is, src, cm, nil)
	dst1 := ebiten.NewImage(lutTestW, lutTestH)
	colorm.DrawTriangles(dst1, vs, is, src, cm, &colorm.DrawTrianglesOptions{LUT: lut})

	maxErr := 0
	for y := 0; y < lutTestH; y++ {
		for x := 0; x < lutTestW; x++ {
			if d := colorDiff(dst0.At(x, y), dst1.At(x, y)); d > maxErr {
				maxErr = d
			}
		}
	}
	t.Logf("DrawTriangles error vs exact matrix: max=%d/255", maxErr)
	if maxErr > 4 {
		t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
	}
}

// TestLUTApplyVsColorMApply compares the CPU-side LUT application with
// ColorM.Apply over a dense set of colors, including transparent and extreme
// ones.
func TestLUTApplyVsColorMApply(t *testing.T) {
	colors := []color.Color{}
	for _, c := range []color.NRGBA64{
		{R: 0, G: 0, B: 0, A: 0},
		{R: 0xffff, G: 0xffff, B: 0xffff, A: 0xffff},
		{R: 0xffff, G: 0, B: 0, A: 0xffff},
		{R: 0, G: 0xffff, B: 0, A: 0xffff},
		{R: 0, G: 0, B: 0xffff, A: 0xffff},
		{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xdef0},
		{R: 0x0100, G: 0x0100, B: 0x0100, A: 0x0100},
		{R: 0x8000, G: 0x4000, B: 0xc000, A: 0x8000},
	} {
		colors = append(colors, c)
	}
	// A gray ramp and an alpha ramp.
	for i := 0; i <= 32; i++ {
		v := uint16(i * 0xffff / 32)
		colors = append(colors, color.NRGBA64{R: v, G: v, B: v, A: 0xffff})
		colors = append(colors, color.NRGBA64{R: 0x8000, G: 0x8000, B: 0x8000, A: v})
	}

	for mname, cm := range lutTestColorMs() {
		t.Run(mname, func(t *testing.T) {
			lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
			if err != nil {
				t.Fatal(err)
			}
			maxErr := 0
			var sumErr int
			for _, c := range colors {
				got := lut.Apply(c)
				want := cm.Apply(c)
				d := colorDiff(got, want)
				if d > maxErr {
					maxErr = d
				}
				sumErr += d
			}
			t.Logf("Apply error vs ColorM.Apply: max=%d/255 mean=%.3f/255 over %d colors",
				maxErr, float64(sumErr)/float64(len(colors)), len(colors))
			if maxErr > 4 {
				t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
			}
		})
	}
}

func TestLUTErrors(t *testing.T) {
	var identity colorm.ColorM

	// Non-finite coefficients.
	var nan colorm.ColorM
	nan.SetElement(0, 0, math.NaN())
	if _, err := colorm.NewColorTransformLUTFromColorM(nan, nil); !errors.Is(err, colorm.ErrNonFiniteCoefficients) {
		t.Errorf("expected ErrNonFiniteCoefficients for NaN, got %v", err)
	}
	var inf colorm.ColorM
	inf.SetElement(1, 1, math.Inf(1))
	if _, err := colorm.NewColorTransformLUTFromColorM(inf, nil); !errors.Is(err, colorm.ErrNonFiniteCoefficients) {
		t.Errorf("expected ErrNonFiniteCoefficients for +Inf, got %v", err)
	}
	var nanT colorm.ColorM
	nanT.Translate(math.NaN(), 0, 0, 0)
	if _, err := colorm.NewColorTransformLUTFromColorM(nanT, nil); !errors.Is(err, colorm.ErrNonFiniteCoefficients) {
		t.Errorf("expected ErrNonFiniteCoefficients for NaN translation, got %v", err)
	}

	// Invalid sizes.
	if _, err := colorm.NewColorTransformLUTFromColorM(identity, &colorm.LUTOptions{Size: -1}); !errors.Is(err, colorm.ErrInvalidLUTSize) {
		t.Errorf("expected ErrInvalidLUTSize for size -1, got %v", err)
	}
	if _, err := colorm.NewColorTransformLUTFromColorM(identity, &colorm.LUTOptions{Size: 1}); !errors.Is(err, colorm.ErrInvalidLUTSize) {
		t.Errorf("expected ErrInvalidLUTSize for size 1, got %v", err)
	}
	if _, err := colorm.NewColorTransformLUTFromColorM(identity, &colorm.LUTOptions{Size: colorm.MaxLUTSize + 1}); !errors.Is(err, colorm.ErrLUTTooLarge) {
		t.Errorf("expected ErrLUTTooLarge for size %d, got %v", colorm.MaxLUTSize+1, err)
	}

	// Zero size falls back to the default size (bounded degradation).
	lut, err := colorm.NewColorTransformLUTFromColorM(identity, &colorm.LUTOptions{Size: 0})
	if err != nil {
		t.Fatal(err)
	}
	if got := lut.Size(); got != colorm.DefaultLUTSize {
		t.Errorf("Size: got %d, want %d", got, colorm.DefaultLUTSize)
	}

	// Invalid color space.
	if _, err := colorm.NewColorTransformLUTFromColorM(identity, &colorm.LUTOptions{ColorSpace: colorm.LUTColorSpace(99)}); !errors.Is(err, colorm.ErrInvalidLUTColorSpace) {
		t.Errorf("expected ErrInvalidLUTColorSpace, got %v", err)
	}

	// Non-finite color scale.
	var cs ebiten.ColorScale
	cs.SetR(float32(math.NaN()))
	if _, err := colorm.NewColorTransformLUTFromColorScale(cs, nil); !errors.Is(err, colorm.ErrNonFiniteCoefficients) {
		t.Errorf("expected ErrNonFiniteCoefficients for NaN color scale, got %v", err)
	}
}

// TestLUTNoNaN checks that extreme but finite transforms never produce NaN
// and degrade in a bounded way.
func TestLUTNoNaN(t *testing.T) {
	var huge colorm.ColorM
	huge.Scale(1e30, -1e30, 0, 1)
	lut, err := colorm.NewColorTransformLUTFromColorM(huge, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []color.Color{
		color.NRGBA64{R: 0xffff, G: 0xffff, B: 0xffff, A: 0xffff},
		color.NRGBA64{A: 0},
		color.NRGBA64{R: 0x8000, G: 0x4000, B: 0x2000, A: 0xffff},
	} {
		got := lut.Apply(c).(color.NRGBA64)
		// NRGBA64 components are integers and cannot be NaN; check the values
		// are saturated as expected instead of wrapped or garbage.
		t.Logf("Apply(%v) = %v", c, got)
	}
	// 1e30 * anything in [0, 1] is far above 1, so R saturates to the max.
	got := lut.Apply(color.NRGBA64{R: 0xffff, G: 0xffff, B: 0xffff, A: 0xffff}).(color.NRGBA64)
	if got.R != 0xffff || got.G != 0 || got.B != 0 {
		t.Errorf("unexpected saturation: got %v", got)
	}

	// A zero transform (e.g. from a zero-alpha color scale) has a degenerate
	// encoding range; it must still work and stay transparent black.
	var cs ebiten.ColorScale
	cs.Scale(1, 1, 1, 0)
	zlut, err := colorm.NewColorTransformLUTFromColorScale(cs, nil)
	if err != nil {
		t.Fatal(err)
	}
	got = zlut.Apply(color.NRGBA64{R: 0xffff, G: 0xffff, B: 0xffff, A: 0xffff}).(color.NRGBA64)
	if got != (color.NRGBA64{}) {
		t.Errorf("zero transform: got %v, want transparent black", got)
	}

	// The GPU path must also survive a zero transform.
	src := extremeColorFixture()
	dst := ebiten.NewImage(lutTestW, lutTestH)
	colorm.DrawImage(dst, src, colorm.ColorM{}, &colorm.DrawImageOptions{LUT: zlut})
	for y := 0; y < lutTestH; y++ {
		for x := 0; x < lutTestW; x++ {
			if got := dst.At(x, y); got != (color.RGBA{}) {
				t.Errorf("At(%d, %d): got %v, want transparent black", x, y, got)
			}
		}
	}
}

// TestLUTConcurrentReadOnly checks that concurrent read-only use of a LUT is
// race-free. Run with -race.
func TestLUTConcurrentReadOnly(t *testing.T) {
	cm := lutTestColorMs()["concat"]
	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Prime the GPU image cache on this goroutine.
	dst := ebiten.NewImage(1, 1)
	src := ebiten.NewImage(1, 1)
	colorm.DrawImage(dst, src, cm, &colorm.DrawImageOptions{LUT: lut})

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				v := uint16((g*500 + i) * 0xffff / 4000)
				_ = lut.Apply(color.NRGBA64{R: v, G: v, B: v, A: 0xffff})
				_ = lut.Size()
				_ = lut.MemorySize()
				_ = lut.ColorM()
			}
		}(g)
	}
	wg.Wait()
}

// TestLUTDispose checks that disposing the cached GPU image (cache eviction)
// keeps the LUT usable with identical results.
func TestLUTDispose(t *testing.T) {
	src := grayscaleFixture()
	cm := lutTestColorMs()["scale-translate"]
	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		t.Fatal(err)
	}
	op := &colorm.DrawImageOptions{LUT: lut}

	dst0 := ebiten.NewImage(lutTestW, lutTestH)
	colorm.DrawImage(dst0, src, cm, op)

	lut.Dispose()

	dst1 := ebiten.NewImage(lutTestW, lutTestH)
	colorm.DrawImage(dst1, src, cm, op)

	for y := 0; y < lutTestH; y++ {
		for x := 0; x < lutTestW; x++ {
			if got, want := dst1.At(x, y), dst0.At(x, y); got != want {
				t.Fatalf("At(%d, %d): got %v after Dispose, want %v", x, y, got, want)
			}
		}
	}
}

// TestLUTConcat checks that Concat composes transforms like ColorM.Concat
// and does not modify the matrices passed to it.
func TestLUTConcat(t *testing.T) {
	cm1 := lutTestColorMs()["scale-translate"]
	cm2 := lutTestColorMs()["rotate-hue"]

	// Record the state of the caller's matrices.
	probe := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xffff}
	before1 := cm1.Apply(probe)
	before2 := cm2.Apply(probe)

	lut, err := colorm.NewColorTransformLUTFromColorM(cm1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := lut.Concat(cm2); err != nil {
		t.Fatal(err)
	}

	var want colorm.ColorM
	want.Concat(cm1)
	want.Concat(cm2)

	maxErr := 0
	for i := 0; i <= 32; i++ {
		v := uint16(i * 0xffff / 32)
		c := color.NRGBA64{R: v, G: 0xffff - v, B: v / 2, A: 0xffff}
		if d := colorDiff(lut.Apply(c), want.Apply(c)); d > maxErr {
			maxErr = d
		}
	}
	t.Logf("Concat error vs concatenated ColorM: max=%d/255", maxErr)
	if maxErr > 4 {
		t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
	}

	// The caller's matrices must be untouched.
	if got := cm1.Apply(probe); got != before1 {
		t.Errorf("cm1 was modified: got %v, want %v", got, before1)
	}
	if got := cm2.Apply(probe); got != before2 {
		t.Errorf("cm2 was modified: got %v, want %v", got, before2)
	}

	// Concat with a non-finite matrix fails and leaves the LUT unchanged.
	var nan colorm.ColorM
	nan.SetElement(0, 0, math.NaN())
	prev := lut.Apply(probe)
	if err := lut.Concat(nan); !errors.Is(err, colorm.ErrNonFiniteCoefficients) {
		t.Errorf("expected ErrNonFiniteCoefficients, got %v", err)
	}
	if got := lut.Apply(probe); got != prev {
		t.Errorf("LUT changed after failed Concat: got %v, want %v", got, prev)
	}
}

// TestLUTInvert checks that Invert matches ColorM.Invert and that a
// non-invertible transform yields a stable error without corrupting the LUT.
func TestLUTInvert(t *testing.T) {
	var cm colorm.ColorM
	cm.Scale(1.2, 0.9, 0.7, 1)
	cm.Translate(0.1, -0.05, 0.2, 0)
	cm.RotateHue(0.6)

	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := lut.Invert(); err != nil {
		t.Fatal(err)
	}

	inv := cm
	inv.Invert()

	maxErr := 0
	for i := 0; i <= 32; i++ {
		v := uint16(i * 0xffff / 32)
		c := color.NRGBA64{R: v, G: v / 2, B: 0xffff - v, A: 0xffff}
		if d := colorDiff(lut.Apply(c), inv.Apply(c)); d > maxErr {
			maxErr = d
		}
	}
	t.Logf("Invert error vs inverted ColorM: max=%d/255", maxErr)
	if maxErr > 4 {
		t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
	}

	// A round trip (invert twice) approximates the original transform.
	if err := lut.Invert(); err != nil {
		t.Fatal(err)
	}
	maxErr = 0
	for i := 0; i <= 32; i++ {
		v := uint16(i * 0xffff / 32)
		c := color.NRGBA64{R: v, G: v / 2, B: 0xffff - v, A: 0xffff}
		if d := colorDiff(lut.Apply(c), cm.Apply(c)); d > maxErr {
			maxErr = d
		}
	}
	t.Logf("double-Invert error vs original ColorM: max=%d/255", maxErr)
	if maxErr > 4 {
		t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
	}

	// Non-invertible transform: stable error, LUT unchanged.
	var z colorm.ColorM
	z.Scale(0, 1, 1, 1)
	zlut, err := colorm.NewColorTransformLUTFromColorM(z, nil)
	if err != nil {
		t.Fatal(err)
	}
	probe := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xffff}
	prev := zlut.Apply(probe)
	if err := zlut.Invert(); !errors.Is(err, colorm.ErrLUTNotInvertible) {
		t.Errorf("expected ErrLUTNotInvertible, got %v", err)
	}
	if got := zlut.Apply(probe); got != prev {
		t.Errorf("LUT changed after failed Invert: got %v, want %v", got, prev)
	}
}

// TestLUTFromColorScale checks the ColorScale conversion, including the
// premultiplied-alpha semantics.
func TestLUTFromColorScale(t *testing.T) {
	var cs ebiten.ColorScale
	cs.Scale(0.5, 0.25, 1.0, 0.75)
	lut, err := colorm.NewColorTransformLUTFromColorScale(cs, nil)
	if err != nil {
		t.Fatal(err)
	}

	// The equivalent straight-alpha transform is Scale(r/a, g/a, b/a, a).
	var cm colorm.ColorM
	cm.Scale(0.5/0.75, 0.25/0.75, 1.0/0.75, 0.75)

	maxErr := 0
	for i := 0; i <= 32; i++ {
		v := uint16(i * 0xffff / 32)
		c := color.NRGBA64{R: v, G: 0xffff - v, B: v / 2, A: 0xffff}
		if d := colorDiff(lut.Apply(c), cm.Apply(c)); d > maxErr {
			maxErr = d
		}
	}
	t.Logf("ColorScale LUT error vs equivalent ColorM: max=%d/255", maxErr)
	if maxErr > 4 {
		t.Errorf("max error %d/255 exceeds tolerance 4/255", maxErr)
	}
}

// TestLUTSRGBColorSpace reports the approximation error of the sRGB-distributed
// levels against the exact matrix, on a gray ramp where distribution matters.
func TestLUTSRGBColorSpace(t *testing.T) {
	cm := lutTestColorMs()["scale-translate"]
	for _, cs := range []colorm.LUTColorSpace{colorm.LUTColorSpaceLinear, colorm.LUTColorSpaceSRGB} {
		lut, err := colorm.NewColorTransformLUTFromColorM(cm, &colorm.LUTOptions{
			Size:       64,
			ColorSpace: cs,
		})
		if err != nil {
			t.Fatal(err)
		}
		maxErr := 0
		var sumErr int
		const n = 256
		for i := 0; i < n; i++ {
			v := uint16(i * 0xffff / (n - 1))
			c := color.NRGBA64{R: v, G: v, B: v, A: 0xffff}
			d := colorDiff(lut.Apply(c), cm.Apply(c))
			if d > maxErr {
				maxErr = d
			}
			sumErr += d
		}
		t.Logf("color space %d (size 64): gray-ramp error max=%d/255 mean=%.3f/255",
			cs, maxErr, float64(sumErr)/n)
		if maxErr > 12 {
			t.Errorf("max error %d/255 exceeds tolerance 12/255", maxErr)
		}
	}
}

// TestLUTColorMRoundTrip checks that ColorM returns an equivalent matrix
// without modifying the LUT.
func TestLUTColorMRoundTrip(t *testing.T) {
	src := lutTestColorMs()["concat"]
	lut, err := colorm.NewColorTransformLUTFromColorM(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := lut.ColorM()
	probe := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xdef0}
	if g, w := got.Apply(probe), src.Apply(probe); g != w {
		t.Errorf("round-tripped ColorM differs: got %v, want %v", g, w)
	}
	// Mutating the returned matrix must not affect the LUT.
	got.Scale(0, 0, 0, 0)
	prev := lut.Apply(probe)
	got2 := lut.ColorM()
	if g, w := got2.Apply(probe), src.Apply(probe); g != w {
		t.Errorf("LUT changed after mutating the returned ColorM: got %v, want %v", g, w)
	}
	if p := lut.Apply(probe); p != prev {
		t.Errorf("LUT.Apply changed after mutating the returned ColorM: got %v, want %v", p, prev)
	}
}

func BenchmarkDrawImageWithColorM(b *testing.B) {
	const w, h = 1920, 1080
	dst := ebiten.NewImage(w, h)
	src := ebiten.NewImage(w, h)
	src.Fill(color.White)
	cm := lutTestColorMs()["concat"]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		colorm.DrawImage(dst, src, cm, nil)
	}
	// Force the queued draws to complete.
	dst.At(0, 0)
}

func BenchmarkDrawImageWithLUT(b *testing.B) {
	const w, h = 1920, 1080
	dst := ebiten.NewImage(w, h)
	src := ebiten.NewImage(w, h)
	src.Fill(color.White)
	cm := lutTestColorMs()["concat"]
	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		b.Fatal(err)
	}
	op := &colorm.DrawImageOptions{LUT: lut}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		colorm.DrawImage(dst, src, cm, op)
	}
	// Force the queued draws to complete.
	dst.At(0, 0)
}

func BenchmarkColorMApply(b *testing.B) {
	cm := lutTestColorMs()["concat"]
	c := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xffff}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cm.Apply(c)
	}
}

func BenchmarkLUTApply(b *testing.B) {
	cm := lutTestColorMs()["concat"]
	lut, err := colorm.NewColorTransformLUTFromColorM(cm, nil)
	if err != nil {
		b.Fatal(err)
	}
	c := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x9abc, A: 0xffff}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = lut.Apply(c)
	}
}
