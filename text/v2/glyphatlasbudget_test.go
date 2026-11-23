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

package text_test

import (
	"bytes"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/math/fixed"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
)

// loadTestSource reads a font from testdata and builds a face source.
func loadTestSource(t *testing.T, name string) *text.GoTextFaceSource {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	src, err := text.NewGoTextFaceSource(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return src
}

func goRegularSource(t *testing.T) *text.GoTextFaceSource {
	t.Helper()
	src, err := text.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		t.Fatal(err)
	}
	return src
}

// renderText draws the sample on a fresh image and returns the image.
func renderText(t *testing.T, sample string, face *text.GoTextFace) *ebiten.Image {
	t.Helper()
	dst := ebiten.NewImage(256, 96)
	t.Cleanup(dst.Deallocate)
	op := &text.DrawOptions{}
	op.GeoM.Translate(4, 40)
	text.Draw(dst, sample, face, op)
	return dst
}

// TestGlyphAtlasBudgetRendersIdenticallyToNoBudget draws a multi-font,
// multi-size, color-glyph fixture with and without a tight budget and
// compares the rendered pixels. A budget that evicts and regenerates glyphs
// must produce the same image as the unbounded path.
func TestGlyphAtlasBudgetRendersIdenticallyToNoBudget(t *testing.T) {
	roboto := loadTestSource(t, "Roboto-Regular.ttf")
	mplus := loadTestSource(t, "MPLUS1p-Regular.ttf")
	colorSrc := loadTestSource(t, "chromacheck-cbdt.ttf")

	const sample = "Hello, 世界! 0123456789"

	face := func(src *text.GoTextFaceSource, size float64) *text.GoTextFace {
		return &text.GoTextFace{Source: src, Size: size}
	}

	// Reference: no budget, every glyph stays cached. The same number of
	// draws is issued as on the budgeted image so accumulated pixels match.
	ref := ebiten.NewImage(256, 160)
	t.Cleanup(ref.Deallocate)
	for range 2 {
		text.Draw(ref, sample, face(roboto, 16), &text.DrawOptions{})
		text.Draw(ref, sample, face(mplus, 22), drawOption(0, 48))
		text.Draw(ref, "color", face(colorSrc, 24), drawOption(0, 96))
	}

	// Budgeted: a deliberately tiny byte budget forces eviction and
	// regeneration on every draw.
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{
		MaxBytes:  4 * 12 * 12,
		MaxGlyphs: 8,
	})
	roboto.SetGlyphAtlasBudget(budget)
	mplus.SetGlyphAtlasBudget(budget)
	colorSrc.SetGlyphAtlasBudget(budget)
	if got := text.GlyphImageCacheBudget(roboto); got != budget {
		t.Fatal("the budget was not attached to the Roboto source")
	}

	got := ebiten.NewImage(256, 160)
	t.Cleanup(got.Deallocate)
	// Draw twice: the second pass mostly hits regenerated entries and also
	// re-renders glyphs evicted between the two passes.
	for range 2 {
		text.Draw(got, sample, face(roboto, 16), &text.DrawOptions{})
		text.Draw(got, sample, face(mplus, 22), drawOption(0, 48))
		text.Draw(got, "color", face(colorSrc, 24), drawOption(0, 96))
	}

	if !imagesEqual(t, ref, got) {
		t.Fatal("budgeted rendering differs from the unbounded reference")
	}

	stats := budget.Stats()
	if stats.Evictions == 0 {
		t.Error("a tight budget recorded no evictions")
	}
	if stats.Misses == 0 || stats.Hits == 0 {
		t.Errorf("expected both hits and misses, got %+v", stats)
	}
	if stats.PeakBytes == 0 {
		t.Error("peak byte watermark was never recorded")
	}
}

// TestGlyphAtlasBudgetHitRateAndPeakMemory exercises a bursty font-switching
// workload and checks that the budget bounds the peak and reports sane
// hit/miss counters.
func TestGlyphAtlasBudgetHitRateAndPeakMemory(t *testing.T) {
	sources := []*text.GoTextFaceSource{
		goRegularSource(t),
		loadTestSource(t, "Roboto-Regular.ttf"),
		loadTestSource(t, "MPLUS1p-Regular.ttf"),
	}
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{
		MaxBytes:  8 << 10,
		MaxGlyphs: 24,
	})
	for _, s := range sources {
		s.SetGlyphAtlasBudget(budget)
	}

	dst := ebiten.NewImage(512, 256)
	t.Cleanup(dst.Deallocate)

	// Burst: cycle through fonts and sizes so the unbudgeted path would
	// accumulate a glyph image per (font, size, glyph).
	samples := []string{"abcdef", "ghijkl", "mnopqr", "stuvwx", "yz0123"}
	for round := range 8 {
		for fi, src := range sources {
			for si := range 4 {
				face := &text.GoTextFace{
					Source: src,
					Size:   12 + float64((round+si)*2),
				}
				text.Draw(dst, samples[(round+fi+si)%len(samples)], face, drawOption(0, float64(fi*30+si*8)))
			}
		}
	}

	stats := budget.Stats()
	if stats.CachedBytes > 8<<10 {
		t.Errorf("cached bytes %d exceed the 8 KiB budget", stats.CachedBytes)
	}
	if stats.CachedGlyphs > 24 {
		t.Errorf("cached glyphs %d exceed the 24-glyph budget", stats.CachedGlyphs)
	}
	if stats.PeakBytes > 8<<10+64 {
		// A small slack covers the in-flight entry before eviction lands.
		t.Errorf("peak bytes %d far exceed the budget", stats.PeakBytes)
	}
	if stats.Hits+stats.Misses == 0 {
		t.Fatal("no lookups recorded")
	}
	t.Logf("budget stats: glyphs=%d bytes=%d peakGlyphs=%d peakBytes=%d hits=%d misses=%d evictions=%d hitRate=%.2f",
		stats.CachedGlyphs, stats.CachedBytes, stats.PeakGlyphs, stats.PeakBytes,
		stats.Hits, stats.Misses, stats.Evictions, stats.HitRate())

	// Every source must stay within the same aggregate budget.
	total := 0
	for _, s := range sources {
		total += text.GlyphImageCount(s)
	}
	if total > 24 {
		t.Errorf("aggregate glyph count %d exceeds the budget", total)
	}
}

// TestGlyphAtlasBudgetEvictionOrder verifies that repeated use keeps a hot
// font's glyphs while a cold font's are evicted first, both via natural LRU
// and via explicit family pinning.
func TestGlyphAtlasBudgetEvictionOrder(t *testing.T) {
	hot := goRegularSource(t)
	cold := loadTestSource(t, "Roboto-Regular.ttf")
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxGlyphs: 12})
	hot.SetGlyphAtlasBudget(budget)
	cold.SetGlyphAtlasBudget(budget)

	dst := ebiten.NewImage(256, 256)
	t.Cleanup(dst.Deallocate)

	// Prime both fonts.
	text.Draw(dst, "abcdefghij", &text.GoTextFace{Source: hot, Size: 16}, drawOption(0, 20))
	text.Draw(dst, "abcdefghij", &text.GoTextFace{Source: cold, Size: 16}, drawOption(0, 60))
	beforeHot := text.GlyphImageCount(hot)
	if beforeHot == 0 {
		t.Fatal("hot source cached no glyphs")
	}

	// Keep the hot font in use while churning the cold one past the limit.
	for range 6 {
		text.Draw(dst, "abcdefghij", &text.GoTextFace{Source: hot, Size: 16}, drawOption(0, 20))
		text.Draw(dst, "klmnopqrst", &text.GoTextFace{Source: cold, Size: 16 + float64(round2())}, drawOption(0, 100))
	}
	if text.GlyphImageCount(hot) == 0 {
		t.Error("the repeatedly used hot font lost all its glyphs")
	}

	// Explicit pinning makes the guarantee absolute.
	budget.PinFamily(hot.ID())
	for range 8 {
		text.Draw(dst, "uvwxyz0123", &text.GoTextFace{Source: cold, Size: 30 + float64(round2())}, drawOption(0, 140))
	}
	if text.GlyphImageCount(hot) != beforeHot {
		t.Errorf("pinned hot font changed size: %d -> %d", beforeHot, text.GlyphImageCount(hot))
	}
	budget.UnpinFamily(hot.ID())
}

// TestGlyphAtlasBudgetClearAndReuse checks explicit Clear plus regeneration,
// including the GoXFace path.
func TestGlyphAtlasBudgetClearAndReuse(t *testing.T) {
	src := goRegularSource(t)
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxBytes: 1 << 14})
	src.SetGlyphAtlasBudget(budget)

	dst := ebiten.NewImage(128, 64)
	t.Cleanup(dst.Deallocate)
	face := &text.GoTextFace{Source: src, Size: 18}
	text.Draw(dst, "Clear", face, nil)
	if text.GlyphImageCount(src) == 0 {
		t.Fatal("drawing populated no glyph cache")
	}

	before := pixelsOf(t, dst)
	budget.Clear()
	if text.GlyphImageCount(src) != 0 {
		t.Errorf("Clear left %d glyphs behind", text.GlyphImageCount(src))
	}

	dst2 := ebiten.NewImage(128, 64)
	t.Cleanup(dst2.Deallocate)
	text.Draw(dst2, "Clear", face, nil)
	if !bytes.Equal(before, pixelsOf(t, dst2)) {
		t.Error("regenerated glyphs after Clear differ from the original pixels")
	}

	// Source-level Clear has the same effect without touching the budget.
	src.ClearGlyphCache()
	if text.GlyphImageCount(src) != 0 {
		t.Error("ClearGlyphCache did not empty the source cache")
	}
}

// TestGoXFaceGlyphBudget covers the per-face GoXFace cache and its lease
// path, ensuring eviction never breaks an in-flight Draw.
func TestGoXFaceGlyphBudget(t *testing.T) {
	face := text.NewGoXFace(newGoXTestFace(t))
	// A one-glyph budget forces an eviction for every distinct glyph key.
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxGlyphs: 1})
	face.SetGlyphAtlasBudget(budget)
	if face.GlyphAtlasBudget() != budget {
		t.Fatal("budget attachment failed on GoXFace")
	}

	dst := ebiten.NewImage(128, 64)
	t.Cleanup(dst.Deallocate)
	for range 20 {
		text.Draw(dst, "abab", face, nil)
	}
	stats := budget.Stats()
	if stats.CachedGlyphs > 1 {
		t.Errorf("GoXFace cache holds %d glyphs with a 1-glyph budget", stats.CachedGlyphs)
	}
	if stats.Evictions == 0 {
		t.Error("no evictions happened under a tight GoXFace budget")
	}

	// Pixel stability: compare against an unbounded face given the same
	// number of draws on a fresh image.
	got := pixelsOf(t, dst)
	ref := ebiten.NewImage(128, 64)
	t.Cleanup(ref.Deallocate)
	refFace := text.NewGoXFace(newGoXTestFace(t))
	for range 20 {
		text.Draw(ref, "abab", refFace, nil)
	}
	if !bytes.Equal(got, pixelsOf(t, ref)) {
		t.Error("GoXFace budgeted rendering differs from the unbounded rendering")
	}

	face.ClearGlyphCache()
	if text.GoXFaceGlyphImageCount(face) != 0 {
		t.Error("ClearGlyphCache did not empty the GoXFace cache")
	}
}

// TestGlyphAtlasZeroBudgetAndOversizedFallback checks the degenerate
// configurations: a zero-valued options struct disables all bounds, and a
// budget smaller than one glyph still renders correctly without caching.
func TestGlyphAtlasZeroBudgetAndOversizedFallback(t *testing.T) {
	src := goRegularSource(t)

	// Zero options: unbounded, legacy behavior.
	unbounded := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{})
	src.SetGlyphAtlasBudget(unbounded)
	dst := ebiten.NewImage(128, 64)
	t.Cleanup(dst.Deallocate)
	for range 5 {
		text.Draw(dst, "Zero", &text.GoTextFace{Source: src, Size: 16}, nil)
	}
	if text.GlyphImageCount(src) == 0 {
		t.Fatal("unbounded budget cached nothing")
	}

	// A one-byte budget rejects every glyph from caching; rendering must
	// still succeed and never loop or leak. Both images receive the same
	// number of draws so the accumulated pixels are comparable.
	tiny := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxBytes: 1})
	src.SetGlyphAtlasBudget(tiny)
	dst2 := ebiten.NewImage(128, 64)
	t.Cleanup(dst2.Deallocate)
	for range 5 {
		text.Draw(dst2, "Zero", &text.GoTextFace{Source: src, Size: 16}, nil)
	}
	if text.GlyphImageCount(src) != 0 {
		t.Errorf("oversized glyphs were cached: %d", text.GlyphImageCount(src))
	}
	stats := tiny.Stats()
	if stats.Oversized == 0 {
		t.Error("the oversized counter was not incremented")
	}
	if !imagesEqual(t, dst, dst2) {
		t.Error("rendering with a one-byte budget differs from the unbounded render")
	}
}

// TestGlyphAtlasBudgetColorGlyphStability ensures color emoji glyphs survive
// eviction and regenerate with identical colors.
func TestGlyphAtlasBudgetColorGlyphStability(t *testing.T) {
	src := loadTestSource(t, "chromacheck-cbdt.ttf")
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxGlyphs: 1})
	src.SetGlyphAtlasBudget(budget)

	face := &text.GoTextFace{Source: src, Size: 48}
	first := text.AppendGlyphs(nil, "\uE903", face, nil)
	if len(first) != 1 || first[0].Image == nil || !first[0].Colored {
		t.Fatal("expected one colored glyph")
	}
	want := color.RGBA{R: 0x64, A: 0xff}
	if got := first[0].Image.At(2, 2); got != want {
		t.Fatalf("color glyph pixel: got %v, want %v", got, want)
	}

	// Churn the single-glyph budget and re-render: the color must come back.
	for i := range 6 {
		f := &text.GoTextFace{Source: src, Size: float64(16 + 4*i)}
		text.AppendGlyphs(nil, "\uE903", f, nil)
	}
	again := text.AppendGlyphs(nil, "\uE903", face, nil)[0]
	if got := again.Image.At(2, 2); got != want {
		t.Errorf("regenerated color glyph pixel: got %v, want %v", got, want)
	}
}

// TestGlyphAtlasBudgetVariationStability renders a variable font at several
// variation coordinates under a tight budget and checks the outlines remain
// stable across eviction and regeneration.
func TestGlyphAtlasBudgetVariationStability(t *testing.T) {
	src := loadTestSource(t, "RobotoFlex.ttf")
	budget := text.NewGlyphAtlasBudget(text.GlyphAtlasBudgetOptions{MaxGlyphs: 4})
	src.SetGlyphAtlasBudget(budget)

	tag := text.MustParseTag("wght")
	render := func(weight float32) *ebiten.Image {
		f := &text.GoTextFace{Source: src, Size: 24}
		f.SetVariation(tag, weight)
		dst := ebiten.NewImage(128, 64)
		t.Cleanup(dst.Deallocate)
		text.Draw(dst, "Ag", f, nil)
		return dst
	}
	ref400 := render(400)
	// Interleave weights to force eviction, then re-render the reference.
	for range 4 {
		render(100)
		render(700)
		render(400)
	}
	if !imagesEqual(t, ref400, render(400)) {
		t.Error("variable-font glyphs changed after eviction and regeneration")
	}
}

func drawOption(x, y float64) *text.DrawOptions {
	op := &text.DrawOptions{}
	op.GeoM.Translate(x, y)
	return op
}

func imagesEqual(t *testing.T, a, b *ebiten.Image) bool {
	t.Helper()
	return bytes.Equal(pixelsOf(t, a), pixelsOf(t, b))
}

func pixelsOf(t *testing.T, img *ebiten.Image) []byte {
	t.Helper()
	r := img.Bounds()
	pix := make([]byte, 4*r.Dx()*r.Dy())
	img.ReadPixels(pix)
	return pix
}

// round2 returns a small changing integer used to vary face sizes without
// importing math/rand into the test.
func round2() int {
	roundState++
	return roundState
}

var roundState int

// newGoXTestFace builds a simple deterministic font.Face for GoXFace tests.
func newGoXTestFace(t *testing.T) font.Face {
	t.Helper()
	return &goXBudgetTestFace{}
}

// goXBudgetTestFace renders solid 6x6 alpha masks for 'a' and 'b'.
type goXBudgetTestFace struct{}

func (f *goXBudgetTestFace) Glyph(dot fixed.Point26_6, r rune) (image.Rectangle, image.Image, image.Point, fixed.Int26_6, bool) {
	dr := image.Rect(0, 0, 6, 6)
	a := image.NewAlpha(dr)
	for y := range 6 {
		for x := range 6 {
			v := uint8(0xff)
			if r == 'a' && (x+y)%2 == 0 {
				v = 0x80
			}
			a.SetAlpha(x, y, color.Alpha{A: v})
		}
	}
	return dr, a, image.Point{}, fixed.I(6), true
}

func (f *goXBudgetTestFace) GlyphBounds(r rune) (fixed.Rectangle26_6, fixed.Int26_6, bool) {
	return fixed.R(0, 0, 6, 6), fixed.I(6), true
}

func (f *goXBudgetTestFace) GlyphAdvance(r rune) (fixed.Int26_6, bool) {
	return fixed.I(6), true
}

func (f *goXBudgetTestFace) Kern(r0, r1 rune) fixed.Int26_6 { return 0 }

func (f *goXBudgetTestFace) Metrics() font.Metrics {
	return font.Metrics{Height: fixed.I(8), Ascent: fixed.I(6), Descent: fixed.I(2)}
}

func (f *goXBudgetTestFace) Close() error { return nil }
