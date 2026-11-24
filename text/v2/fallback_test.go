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

package text

import (
	"bytes"
	"errors"
	"image"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hajimehoshi/bitmapfont/v4"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/text/language"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const fallbackTestEps = 1.0 / (1 << 6)

// fallbackFixture bundles the faces used across the fallback tests.
//
// Coverage of the bundled fonts:
//   - goregular (GoTextFace): Latin, Latin-1; no CJK/Arabic/emoji.
//   - M+ 1p (GoTextFace): Latin, CJK, kana, Hebrew; no Arabic/emoji.
//   - bitmapfont (GoXFace): Latin, CJK, kana, Arabic presentation forms,
//     Hebrew; no color emoji.
//   - chromacheck-colr (GoTextFace): a single color glyph at U+E900.
//
// Limited-face variants pin coverage to one script so per-cluster routing is
// deterministic even when two bundled fonts overlap (e.g. bitmapfont also
// ships Han glyphs).
type fallbackFixture struct {
	latin      *GoTextFace
	mplus      *GoTextFace
	bitmap     *GoXFace
	emoji      *GoTextFace
	hanOnly    *LimitedFace // M+ restricted to Han
	arabicOnly *LimitedFace // bitmapfont restricted to Arabic blocks
	emojiOnly  *LimitedFace // chromacheck restricted to its PUA glyph
}

func newFallbackFixture(t *testing.T) *fallbackFixture {
	t.Helper()

	latinSrc, err := NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		t.Fatal(err)
	}
	mplusSrc, err := NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		t.Fatal(err)
	}
	emojiBytes, err := os.ReadFile(filepath.Join("testdata", "chromacheck-colr.ttf"))
	if err != nil {
		t.Fatal(err)
	}
	emojiSrc, err := NewGoTextFaceSource(bytes.NewReader(emojiBytes))
	if err != nil {
		t.Fatal(err)
	}

	mplus := &GoTextFace{Source: mplusSrc, Size: 16}
	emoji := &GoTextFace{Source: emojiSrc, Size: 16}

	hanOnly := NewLimitedFace(mplus)
	hanOnly.AddUnicodeRange(0x2E80, 0x9FFF) // CJK radicals through Han

	arabicOnly := NewLimitedFace(NewGoXFace(bitmapfont.Face))
	arabicOnly.AddUnicodeRange(0x0600, 0x06FF)
	arabicOnly.AddUnicodeRange(0x0750, 0x077F)
	arabicOnly.AddUnicodeRange(0x08A0, 0x08FF)
	arabicOnly.AddUnicodeRange(0xFB50, 0xFDFF) // Arabic presentation forms-A
	arabicOnly.AddUnicodeRange(0xFE70, 0xFEFF) // Arabic presentation forms-B

	emojiOnly := NewLimitedFace(emoji)
	emojiOnly.AddUnicodeRange(0xE900, 0xE900)

	return &fallbackFixture{
		latin:      &GoTextFace{Source: latinSrc, Size: 16},
		mplus:      mplus,
		bitmap:     NewGoXFace(bitmapfont.Face),
		emoji:      emoji,
		hanOnly:    hanOnly,
		arabicOnly: arabicOnly,
		emojiOnly:  emojiOnly,
	}
}

// emojiClusterRune is a PUA codepoint only the chromacheck color face covers.
// It stands in for an emoji in the mixed-script fixtures.
const emojiClusterRune rune = 0xE900

// FaceKind identifies a fixture face for reference comparisons.
type FaceKind int

const (
	KindUnknown FaceKind = iota
	KindLatin
	KindMPlus
	KindBitmap
	KindEmoji
)

func (k FaceKind) String() string {
	switch k {
	case KindLatin:
		return "latin(goregular)"
	case KindMPlus:
		return "mplus"
	case KindBitmap:
		return "bitmapfont"
	case KindEmoji:
		return "emoji(chromacheck)"
	default:
		return "unknown"
	}
}

func classifyFace(fx *fallbackFixture, f Face) FaceKind {
	switch f {
	case fx.latin:
		return KindLatin
	case fx.mplus:
		return KindMPlus
	case fx.bitmap:
		return KindBitmap
	case fx.emoji:
		return KindEmoji
	}
	// Limited faces wrap a known face; compare by pointer identity against
	// the fixture's limited variants and their underlying faces.
	switch f {
	case fx.hanOnly:
		return KindMPlus
	case fx.arabicOnly:
		return KindBitmap
	case fx.emojiOnly:
		return KindEmoji
	}
	if lf, ok := f.(*LimitedFace); ok {
		switch lf.face {
		case fx.mplus:
			return KindMPlus
		case fx.bitmap:
			return KindBitmap
		case fx.emoji:
			return KindEmoji
		}
	}
	return KindUnknown
}

// resolverClusterFaces resolves each cluster's selected face by reading the
// resolver's plan and mapping chain indices to fixture kinds.
func resolverClusterFaces(t *testing.T, fx *fallbackFixture, r *TextFontFallbackResolver, line string) ([]string, []FaceKind) {
	t.Helper()
	plan := r.plan(line)
	clusters := make([]string, len(plan.clusters))
	kinds := make([]FaceKind, len(plan.clusters))
	for i, c := range plan.clusters {
		clusters[i] = line[c.start:c.end]
		kinds[i] = classifyFace(fx, r.chain[plan.faceIndex[i]])
	}
	return clusters, kinds
}

func TestNewTextFontFallbackResolverValidation(t *testing.T) {
	fx := newFallbackFixture(t)

	t.Run("nil primary", func(t *testing.T) {
		if _, err := NewTextFontFallbackResolver(nil); !errors.Is(err, ErrEmptyFallbackChain) {
			t.Errorf("err = %v, want ErrEmptyFallbackChain", err)
		}
	})

	t.Run("candidate with zero value", func(t *testing.T) {
		if _, err := NewTextFontFallbackResolver(fx.latin, FallbackCandidate{}); !errors.Is(err, ErrEmptyFallbackChain) {
			t.Errorf("err = %v, want ErrEmptyFallbackChain", err)
		}
	})

	t.Run("short script code", func(t *testing.T) {
		// ParseScript requires at least four bytes; a shorter code is invalid.
		if _, err := NewScriptFallbackCandidate("ab", language.Und, fx.mplus); err == nil {
			t.Error("expected error for too-short script code")
		}
	})

	t.Run("nil face in constructors", func(t *testing.T) {
		if _, err := NewScriptFallbackCandidate("Arab", language.Und, nil); err == nil {
			t.Error("expected error for nil script face")
		}
		if _, err := NewEmojiFallbackCandidate(nil); err == nil {
			t.Error("expected error for nil emoji face")
		}
		if _, err := NewGeneralFallbackCandidate(nil); err == nil {
			t.Error("expected error for nil general face")
		}
	})

	t.Run("direction mismatch", func(t *testing.T) {
		rtl := &GoTextFace{Source: fx.mplus.Source, Size: 16, Direction: DirectionRightToLeft}
		cand, err := NewScriptFallbackCandidate("Arab", language.Und, rtl)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := NewTextFontFallbackResolver(fx.latin, cand); err == nil {
			t.Error("expected error for direction mismatch")
		}
	})

	t.Run("duplicate face rejected", func(t *testing.T) {
		cand, err := NewGeneralFallbackCandidate(fx.mplus)
		if err != nil {
			t.Fatal(err)
		}
		cand2, err := NewScriptFallbackCandidate("Hani", language.Und, fx.mplus)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := NewTextFontFallbackResolver(fx.latin, cand, cand2); err == nil {
			t.Error("expected error for duplicate face in chain")
		}
	})

	t.Run("too many candidates", func(t *testing.T) {
		cands := make([]FallbackCandidate, maxFallbackCandidates+1)
		for i := range cands {
			lf := NewLimitedFace(fx.mplus)
			lf.AddUnicodeRange(rune(i), rune(i))
			c, err := NewGeneralFallbackCandidate(lf)
			if err != nil {
				t.Fatal(err)
			}
			cands[i] = c
		}
		if _, err := NewTextFontFallbackResolver(fx.latin, cands...); err == nil {
			t.Error("expected error for over-long fallback chain")
		}
	})

	t.Run("valid construction", func(t *testing.T) {
		arab, err := NewScriptFallbackCandidate("Arab", language.Und, fx.arabicOnly)
		if err != nil {
			t.Fatal(err)
		}
		emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
		if err != nil {
			t.Fatal(err)
		}
		r, err := NewTextFontFallbackResolver(fx.latin, arab, emoji)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if r == nil {
			t.Fatal("resolver is nil")
		}
	})
}

// TestFallbackResolverClusterSelection verifies per-cluster face assignment
// against a hand-computed reference for Latin/CJK/Arabic/emoji mixed text.
func TestFallbackResolverClusterSelection(t *testing.T) {
	fx := newFallbackFixture(t)

	arab, err := NewScriptFallbackCandidate("Arab", language.Und, fx.arabicOnly)
	if err != nil {
		t.Fatal(err)
	}
	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, arab, han, emoji)
	if err != nil {
		t.Fatal(err)
	}

	// "a中ا<emoji>": Latin stays primary, Han -> M+, Arabic -> bitmap,
	// emoji -> color face.
	line := "a中ا" + string(emojiClusterRune)

	wantFaces := []FaceKind{KindLatin, KindMPlus, KindBitmap, KindEmoji}
	wantClusters := []string{"a", "中", "ا", string(emojiClusterRune)}

	gotClusters, gotKinds := resolverClusterFaces(t, fx, r, line)
	if len(gotClusters) != len(wantClusters) {
		t.Fatalf("cluster count = %d, want %d (%q)", len(gotClusters), len(wantClusters), gotClusters)
	}
	for i := range wantClusters {
		if gotClusters[i] != wantClusters[i] {
			t.Errorf("cluster %d = %q, want %q", i, gotClusters[i], wantClusters[i])
		}
		if gotKinds[i] != wantFaces[i] {
			t.Errorf("cluster %d (%q) face = %s, want %s", i, gotClusters[i], gotKinds[i], wantFaces[i])
		}
	}
}

// TestFallbackResolverPrimaryWins ensures a rune the primary face covers is
// never routed to a matching fallback face.
func TestFallbackResolverPrimaryWins(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	catchAll, err := NewGeneralFallbackCandidate(fx.arabicOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, catchAll)
	if err != nil {
		t.Fatal(err)
	}

	line := "a"
	got := AppendLazyGlyphs(nil, line, r, nil)
	want := AppendLazyGlyphs(nil, line, fx.latin, nil)
	if len(got) != len(want) || got[0].GID != want[0].GID {
		t.Errorf("primary face was bypassed: got GID %v, want %v", lazyGIDs(got), lazyGIDs(want))
	}

	str := "Hello, world"
	if advGot, advWant := Advance(str, r), Advance(str, fx.latin); math.Abs(advGot-advWant) > fallbackTestEps {
		t.Errorf("Latin advance = %v, want %v", advGot, advWant)
	}
}

func lazyGIDs(gs []LazyGlyph) []uint32 {
	out := make([]uint32, len(gs))
	for i, g := range gs {
		out[i] = g.GID
	}
	return out
}

// TestFallbackResolverEmojiSequenceKeptWhole verifies emoji clusters are never
// split across faces.
func TestFallbackResolverEmojiSequenceKeptWhole(t *testing.T) {
	fx := newFallbackFixture(t)

	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, emoji)
	if err != nil {
		t.Fatal(err)
	}

	emojiGID := AppendLazyGlyphs(nil, string(emojiClusterRune), fx.emoji, nil)[0].GID

	for _, line := range []string{
		string(emojiClusterRune),
		"a" + string(emojiClusterRune) + "b",
		string(emojiClusterRune) + string(emojiClusterRune),
	} {
		t.Run(line, func(t *testing.T) {
			plan := r.plan(line)
			for i, c := range plan.clusters {
				sub := line[c.start:c.end]
				if bytes.ContainsRune([]byte(sub), emojiClusterRune) {
					if r.chain[plan.faceIndex[i]] != fx.emojiOnly {
						t.Errorf("emoji cluster %q routed to %s, want emoji face", sub,
							classifyFace(fx, r.chain[plan.faceIndex[i]]))
					}
				}
			}
			gs := AppendLazyGlyphs(nil, line, r, nil)
			for _, g := range gs {
				sub := line[g.StartIndexInBytes:g.EndIndexInBytes]
				if bytes.ContainsRune([]byte(sub), emojiClusterRune) && g.GID != emojiGID {
					t.Errorf("emoji glyph GID = %d, want %d", g.GID, emojiGID)
				}
			}
		})
	}
}

// TestFallbackResolverCombiningMark verifies a base + combining mark stays on
// one face.
func TestFallbackResolverCombiningMark(t *testing.T) {
	fx := newFallbackFixture(t)

	arab, err := NewScriptFallbackCandidate("Arab", language.Und, fx.arabicOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, arab)
	if err != nil {
		t.Fatal(err)
	}

	// U+0627 ALEF + U+0651 SHADDA is one grapheme cluster, absent from
	// goregular, so the whole cluster routes to the Arabic-limited face.
	const cluster = "اّ"
	if n := utf8RuneCount(cluster); n != 2 {
		t.Fatalf("test setup: %q has %d runes, want 2", cluster, n)
	}
	plan := r.plan(cluster)
	if len(plan.clusters) != 1 {
		t.Fatalf("combining-mark sequence split into %d clusters", len(plan.clusters))
	}
	if classifyFace(fx, r.chain[plan.faceIndex[0]]) != KindBitmap {
		t.Errorf("cluster routed to %s, want bitmap face", classifyFace(fx, r.chain[plan.faceIndex[0]]))
	}
	gs := AppendLazyGlyphs(nil, cluster, r, nil)
	if len(gs) == 0 {
		t.Fatal("no glyphs produced for combining-mark cluster")
	}
	// The cluster may produce multiple glyphs (base + mark), but every glyph
	// must stay within the single cluster's byte range.
	for _, g := range gs {
		if g.StartIndexInBytes < 0 || g.EndIndexInBytes > len(cluster) {
			t.Errorf("glyph range [%d,%d) escapes cluster [0,%d)", g.StartIndexInBytes, g.EndIndexInBytes, len(cluster))
		}
	}
}

func utf8RuneCount(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

// TestFallbackResolverAdvanceAndBaseline checks advance, baseline and glyph
// origins against independently computed per-face references.
func TestFallbackResolverAdvanceAndBaseline(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	arab, err := NewScriptFallbackCandidate("Arab", language.Und, fx.arabicOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, arab, emoji)
	if err != nil {
		t.Fatal(err)
	}

	const line = "ab中ا" + string(emojiClusterRune)
	clusters := []string{"a", "b", "中", "ا", string(emojiClusterRune)}
	faces := []Face{fx.latin, fx.latin, fx.hanOnly, fx.arabicOnly, fx.emojiOnly}

	var wantTotal float64
	for i, c := range clusters {
		wantTotal += faces[i].advanceAt(c, len(c))
	}
	gotTotal := r.advanceAt(line, len(line))
	if math.Abs(gotTotal-wantTotal) > fallbackTestEps {
		t.Errorf("total advance = %v, want %v", gotTotal, wantTotal)
	}

	// AdvanceAt at every cluster boundary matches the running sum; interior
	// bytes snap to the cluster's leading edge.
	var running float64
	bytePos := 0
	for i, c := range clusters {
		if got := r.advanceAt(line, bytePos); math.Abs(got-running) > fallbackTestEps {
			t.Errorf("advanceAt(byte %d, cluster %q) = %v, want %v", bytePos, c, got, running)
		}
		if len(c) > 1 {
			if got := r.advanceAt(line, bytePos+1); math.Abs(got-running) > fallbackTestEps {
				t.Errorf("interior advanceAt(byte %d) = %v, want snapped %v", bytePos+1, got, running)
			}
		}
		running += faces[i].advanceAt(c, len(c))
		bytePos += len(c)
	}
	if got := r.advanceAt(line, len(line)); math.Abs(got-wantTotal) > fallbackTestEps {
		t.Errorf("advanceAt(end) = %v, want %v", got, wantTotal)
	}

	// Glyph origins are monotonic; the last glyph ends at the total width.
	gs := AppendLazyGlyphs(nil, line, r, nil)
	prevX := math.Inf(-1)
	for _, g := range gs {
		x := g.OriginX + g.OriginOffsetX
		if x < prevX-fallbackTestEps {
			t.Errorf("glyph origin X decreased: %v < %v", x, prevX)
		}
		prevX = x
	}
	last := gs[len(gs)-1]
	endX := last.OriginX + last.OriginOffsetX + last.AdvanceX
	if math.Abs(endX-wantTotal) > fallbackTestEps {
		t.Errorf("last glyph end X = %v, want total %v", endX, wantTotal)
	}
}

// TestFallbackResolverGlyphSource verifies glyph images come from the selected
// face, including a colored emoji glyph.
func TestFallbackResolverGlyphSource(t *testing.T) {
	fx := newFallbackFixture(t)

	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, emoji)
	if err != nil {
		t.Fatal(err)
	}

	line := "a" + string(emojiClusterRune)
	gs := AppendGlyphs(nil, line, r, nil)
	if len(gs) < 2 {
		t.Fatalf("expected >=2 glyphs, got %d", len(gs))
	}

	var emojiGlyph *Glyph
	for i := range gs {
		if gs[i].StartIndexInBytes >= 1 {
			emojiGlyph = &gs[i]
			break
		}
	}
	if emojiGlyph == nil {
		t.Fatal("emoji glyph not found")
	}
	if !emojiGlyph.Colored {
		t.Error("emoji glyph must be Colored")
	}
	direct := AppendGlyphs(nil, string(emojiClusterRune), fx.emoji, nil)
	if len(direct) != 1 {
		t.Fatalf("direct emoji glyph count = %d, want 1", len(direct))
	}
	if emojiGlyph.Image == nil || direct[0].Image == nil {
		t.Fatal("emoji image is nil")
	}
	if b1, b2 := emojiGlyph.Image.Bounds(), direct[0].Image.Bounds(); b1.Dx() != b2.Dx() || b1.Dy() != b2.Dy() {
		t.Errorf("emoji image bounds = %v, want %v", b1, b2)
	}
}

// TestFallbackResolverMissingGlyphFallback ensures a cluster no face covers is
// rendered by the primary face's missing-glyph form, with an error report.
func TestFallbackResolverMissingGlyphFallback(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}

	var reported int
	r.SetErrorHandler(func(error) { reported++ })

	// U+1F600 is covered by neither Latin nor Han faces.
	const missing = "😀"
	gs := AppendLazyGlyphs(nil, "a"+missing, r, nil)
	if len(gs) == 0 {
		t.Fatal("expected glyphs even for an uncovered cluster")
	}
	if reported == 0 {
		t.Error("expected an error report for the uncovered cluster")
	}
	for _, g := range gs {
		if g.EndIndexInBytes <= g.StartIndexInBytes {
			t.Errorf("degenerate glyph range [%d,%d)", g.StartIndexInBytes, g.EndIndexInBytes)
		}
	}
}

// failingFace is a Face whose CheckFace reports an error and whose glyph
// methods optionally panic once closed, exercising the unavailable-face path.
type failingFace struct {
	face      Face
	closed    bool
	checkErr  error
	panicOnOp bool
}

func (f *failingFace) Metrics() Metrics {
	if f.panicOnOp && f.closed {
		panic("face is closed")
	}
	return f.face.Metrics()
}

func (f *failingFace) advanceAt(text string, indexInBytes int) float64 {
	if f.panicOnOp && f.closed {
		panic("face is closed")
	}
	return f.face.advanceAt(text, indexInBytes)
}

func (f *failingFace) hasGlyph(r rune) bool {
	if f.panicOnOp && f.closed {
		panic("face is closed")
	}
	return f.face.hasGlyph(r)
}

func (f *failingFace) appendLazyGlyphsForLine(glyphs []LazyGlyph, line string, indexOffset int, originX, originY float64, keepGlyph func(originX, originY float64) bool) []LazyGlyph {
	if f.panicOnOp && f.closed {
		panic("face is closed")
	}
	return f.face.appendLazyGlyphsForLine(glyphs, line, indexOffset, originX, originY, keepGlyph)
}

func (f *failingFace) appendVectorPathForLine(path *vector.Path, line string, originX, originY float64) {
	f.face.appendVectorPathForLine(path, line, originX, originY)
}

func (f *failingFace) direction() Direction { return f.face.direction() }
func (f *failingFace) private()             {}
func (f *failingFace) CheckFace() error     { return f.checkErr }

// TestFallbackResolverUnavailableFace verifies a candidate failing its health
// check is skipped, the error is reported, and layout continues.
func TestFallbackResolverUnavailableFace(t *testing.T) {
	fx := newFallbackFixture(t)

	// The broken face would be the only Han candidate. Its health check
	// fails, so 中 falls back to the primary; the error must wrap
	// ErrFaceUnavailable.
	broken := &failingFace{face: fx.hanOnly, checkErr: errors.New("resource closed")}
	han, err := NewScriptFallbackCandidate("Hani", language.Und, broken)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}

	var reports []error
	r.SetErrorHandler(func(err error) { reports = append(reports, err) })

	gs := AppendLazyGlyphs(nil, "中a", r, nil)
	if len(gs) == 0 {
		t.Fatal("layout produced no glyphs")
	}
	if len(reports) == 0 {
		t.Fatal("expected an unavailable-face error report")
	}
	sawUnavailable := false
	for _, e := range reports {
		if errors.Is(e, ErrFaceUnavailable) {
			sawUnavailable = true
		}
	}
	if !sawUnavailable {
		t.Errorf("no report wrapped ErrFaceUnavailable: %v", reports)
	}
}

// TestFallbackResolverPanickingFace verifies a panicking fallback face is
// contained and the primary face still renders the line.
func TestFallbackResolverPanickingFace(t *testing.T) {
	fx := newFallbackFixture(t)

	panicker := &failingFace{face: fx.hanOnly, closed: true, panicOnOp: true}
	han, err := NewScriptFallbackCandidate("Hani", language.Und, panicker)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}
	r.SetErrorHandler(func(error) {})

	gs := AppendLazyGlyphs(nil, "a中", r, nil)
	if len(gs) == 0 {
		t.Fatal("Latin glyphs lost when fallback face panics")
	}
	wantLatin := AppendLazyGlyphs(nil, "a", fx.latin, nil)[0].GID
	if gs[0].GID != wantLatin {
		t.Errorf("Latin GID = %d, want %d", gs[0].GID, wantLatin)
	}

	// Metrics must also survive a panicking face.
	_ = r.Metrics()

	// The full Draw path and vector path must not panic either.
	dst := ebiten.NewImage(64, 64)
	Draw(dst, "a中", r, nil)
	var p vector.Path
	AppendVectorPath(&p, "a中", r, nil)

	// advanceAt must recover the panic and still return the Latin width.
	if got := r.advanceAt("a中", len("a中")); got <= 0 {
		t.Errorf("advance after panicking face = %v, want positive Latin width", got)
	}
}

// TestFallbackResolverPanicAfterPlan verifies a face that passes the health
// check at plan time but panics while shaping a segment does not crash layout.
func TestFallbackResolverPanicAfterPlan(t *testing.T) {
	fx := newFallbackFixture(t)

	// A face whose CheckFace succeeds but whose shaping panics: simulate a
	// resource closed between plan and render.
	late := &failingFace{face: fx.hanOnly, panicOnOp: true}
	han, err := NewScriptFallbackCandidate("Hani", language.Und, late)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}
	var reports int
	r.SetErrorHandler(func(error) { reports++ })

	// Plan while healthy so the Han segment selects this face.
	plan := r.plan("a中")
	if plan.faceIndex[1] != 1 {
		t.Fatalf("expected Han to select candidate, got %d", plan.faceIndex[1])
	}
	// Then "close" it before rendering.
	late.closed = true

	gs := AppendLazyGlyphs(nil, "a中", r, nil)
	if len(gs) == 0 {
		t.Fatal("rendering produced no glyphs after late face failure")
	}
	if reports == 0 {
		t.Error("expected an error report when shaping panicked")
	}
}

// TestFallbackResolverConcurrent exercises concurrent layout through one
// resolver, checking for races and deterministic results.
func TestFallbackResolverConcurrent(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, emoji)
	if err != nil {
		t.Fatal(err)
	}

	const line = "a中b" + string(emojiClusterRune)
	want := r.advanceAt(line, len(line))

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				if got := r.advanceAt(line, len(line)); math.Abs(got-want) > fallbackTestEps {
					t.Errorf("concurrent advance = %v, want %v", got, want)
				}
				_ = AppendLazyGlyphs(nil, line, r, nil)
			}
		}()
	}
	wg.Wait()
}

// TestFallbackResolverCacheReuse verifies repeated layout of the same line is
// reproducible and plan cache keys are stable.
func TestFallbackResolverCacheReuse(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}

	const line = "a中b中c"
	first := r.plan(line)
	second := r.plan(line)
	if len(first.clusters) != len(second.clusters) {
		t.Fatalf("cluster count changed: %d vs %d", len(first.clusters), len(second.clusters))
	}
	for i := range first.clusters {
		if first.faceIndex[i] != second.faceIndex[i] ||
			first.clusters[i] != second.clusters[i] {
			t.Errorf("plan %d not reproducible: %+v vs %+v", i, first.clusters[i], second.clusters[i])
		}
	}

	g1 := AppendLazyGlyphs(nil, line, r, nil)
	g2 := AppendLazyGlyphs(nil, line, r, nil)
	if len(g1) != len(g2) {
		t.Fatalf("glyph count changed: %d vs %d", len(g1), len(g2))
	}
	for i := range g1 {
		if g1[i].GID != g2[i].GID ||
			math.Abs(g1[i].OriginX-g2[i].OriginX) > fallbackTestEps ||
			g1[i].StartIndexInBytes != g2[i].StartIndexInBytes {
			t.Errorf("glyph %d not reproducible: %+v vs %+v", i, g1[i], g2[i])
		}
	}
}

// TestFallbackResolverRTL verifies right-to-left layout: total advance parity
// with Advance(), valid glyph indices, and the leading-edge convention at the
// line endpoints.
func TestFallbackResolverRTL(t *testing.T) {
	fx := newFallbackFixture(t)

	rtlPrimary := &GoTextFace{Source: fx.mplus.Source, Size: 16, Direction: DirectionRightToLeft}
	rtlLimited := NewLimitedFace(&GoTextFace{Source: fx.mplus.Source, Size: 16, Direction: DirectionRightToLeft})
	rtlLimited.AddUnicodeRange(0x05D0, 0x05EA) // Hebrew only

	cand, err := NewGeneralFallbackCandidate(rtlLimited)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(rtlPrimary, cand)
	if err != nil {
		t.Fatal(err)
	}

	const line = "א1ב2"
	gs := AppendLazyGlyphs(nil, line, r, nil)
	if len(gs) == 0 {
		t.Fatal("no RTL glyphs")
	}
	for _, g := range gs {
		if g.StartIndexInBytes < 0 || g.EndIndexInBytes > len(line) {
			t.Errorf("RTL glyph index out of range: [%d,%d)", g.StartIndexInBytes, g.EndIndexInBytes)
		}
	}
	// AdvanceAt(end) equals Advance(): whole-line width parity.
	if got, want := AdvanceAt(line, len(line), r), Advance(line, r); math.Abs(got-want) > fallbackTestEps {
		t.Errorf("RTL AdvanceAt(total) = %v, want Advance() = %v", got, want)
	}
	// Zero is always the origin.
	if got := AdvanceAt(line, 0, r); got != 0 {
		t.Errorf("AdvanceAt(0) = %v, want 0", got)
	}
	// The RTL line grows leftward, so the total is non-negative.
	if total := Advance(line, r); total < 0 {
		t.Errorf("RTL total advance negative: %v", total)
	}
}

// TestFallbackResolverMetricsMax verifies metrics take the maximum across the
// chain so fallback glyphs fit the line box.
func TestFallbackResolverMetricsMax(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}

	got := r.Metrics()
	want := fx.latin.Metrics()
	m := fx.mplus.Metrics()
	want.HAscent = math.Max(want.HAscent, m.HAscent)
	want.HDescent = math.Max(want.HDescent, m.HDescent)
	want.HLineGap = math.Max(want.HLineGap, m.HLineGap)
	if math.Abs(got.HAscent-want.HAscent) > fallbackTestEps ||
		math.Abs(got.HDescent-want.HDescent) > fallbackTestEps ||
		math.Abs(got.HLineGap-want.HLineGap) > fallbackTestEps {
		t.Errorf("metrics = %+v, want %+v", got, want)
	}
}

// TestFallbackResolverDraw renders through the full Draw path.
func TestFallbackResolverDraw(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, emoji)
	if err != nil {
		t.Fatal(err)
	}

	dst := ebiten.NewImage(256, 64)
	Draw(dst, "ab中"+string(emojiClusterRune), r, nil)

	bounds := dst.Bounds()
	var painted int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if _, _, _, a := dst.At(x, y).RGBA(); a != 0 {
				painted++
			}
		}
	}
	if painted == 0 {
		t.Error("Draw painted nothing")
	}
}

// TestFallbackResolverVariationSelector verifies a base+VS16 cluster is routed
// as a unit and not split.
func TestFallbackResolverVariationSelector(t *testing.T) {
	fx := newFallbackFixture(t)

	// M+ covers the copyright sign; goregular does not. A general fallback to
	// M+ routes the whole base+VS16 cluster together.
	gen, err := NewGeneralFallbackCandidate(fx.mplus)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, gen)
	if err != nil {
		t.Fatal(err)
	}

	const cluster = "©️" // COPYRIGHT SIGN + VS16
	plan := r.plan(cluster)
	if len(plan.clusters) != 1 {
		t.Fatalf("variation-selector cluster split into %d pieces", len(plan.clusters))
	}
	gs := AppendLazyGlyphs(nil, cluster, r, nil)
	if len(gs) == 0 {
		t.Fatal("no glyphs")
	}
	for _, g := range gs {
		if g.StartIndexInBytes != 0 {
			t.Errorf("variation-selector cluster split: glyph starts at %d", g.StartIndexInBytes)
		}
	}
}

// TestFallbackResolverNested verifies a resolver can nest another resolver.
func TestFallbackResolverNested(t *testing.T) {
	fx := newFallbackFixture(t)

	// Inner resolver: a Latin-limited bitmap primary + Han fallback to M+.
	latinLimited := NewLimitedFace(fx.bitmap)
	latinLimited.AddUnicodeRange(0x0020, 0x007F)
	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	inner, err := NewTextFontFallbackResolver(latinLimited, han)
	if err != nil {
		t.Fatal(err)
	}
	nested, err := NewGeneralFallbackCandidate(inner)
	if err != nil {
		t.Fatal(err)
	}
	// Outer primary (goregular) covers Latin but not Han, so Han descends
	// into the nested resolver.
	outer, err := NewTextFontFallbackResolver(fx.latin, nested)
	if err != nil {
		t.Fatalf("nested resolver rejected: %v", err)
	}

	line := "a中"
	gs := AppendLazyGlyphs(nil, line, outer, nil)
	wantGID := AppendLazyGlyphs(nil, "中", fx.mplus, nil)[0].GID
	found := false
	for _, g := range gs {
		if g.GID == wantGID {
			found = true
		}
	}
	if !found {
		t.Error("nested resolver did not route Han to the M+ face")
	}
}

// TestFallbackResolverCyclicRejected verifies a resolver whose chain reaches
// back into an outer resolver's face is rejected.
func TestFallbackResolverCyclicRejected(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	// The inner resolver contains fx.hanOnly; nesting it under an outer
	// resolver that also uses fx.hanOnly forms a reachable cycle.
	inner, err := NewTextFontFallbackResolver(fx.bitmap, han)
	if err != nil {
		t.Fatal(err)
	}
	nested, err := NewGeneralFallbackCandidate(inner)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewTextFontFallbackResolver(fx.hanOnly, nested); err == nil {
		t.Error("expected cyclic resolver construction to fail")
	}
}

// TestFallbackResolverEmptyLine verifies empty input does not panic.
func TestFallbackResolverEmptyLine(t *testing.T) {
	fx := newFallbackFixture(t)
	r, err := NewTextFontFallbackResolver(fx.latin)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.advanceAt("", 0); got != 0 {
		t.Errorf("empty advance = %v, want 0", got)
	}
	if gs := AppendLazyGlyphs(nil, "", r, nil); len(gs) != 0 {
		t.Errorf("empty line produced %d glyphs", len(gs))
	}
	if w, h := Measure("multi\nline", r, 20); w <= 0 || h <= 0 {
		t.Errorf("Measure returned non-positive dimensions: %v x %v", w, h)
	}
}

// TestFallbackResolverMultiLine verifies line breaks are handled across lines,
// including distinct per-line origins under a non-zero line spacing.
func TestFallbackResolverMultiLine(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han)
	if err != nil {
		t.Fatal(err)
	}

	const text = "a中\nb中"
	var op LayoutOptions
	op.LineSpacing = 24
	gs := AppendLazyGlyphs(nil, text, r, &op)
	if len(gs) == 0 {
		t.Fatal("no glyphs across two lines")
	}
	maxEnd := 0
	var lineOrigins []float64
	for _, g := range gs {
		if g.StartIndexInBytes < 0 || g.EndIndexInBytes > len(text) {
			t.Errorf("glyph index out of range: [%d,%d) of %d", g.StartIndexInBytes, g.EndIndexInBytes, len(text))
		}
		maxEnd = max(maxEnd, g.EndIndexInBytes)
		lineOrigins = append(lineOrigins, g.OriginY)
	}
	if maxEnd != len(text) {
		t.Errorf("glyphs end at %d, want %d", maxEnd, len(text))
	}
	// With non-zero line spacing, the second line origin exceeds the first.
	if lineOrigins[len(lineOrigins)-1] <= lineOrigins[0] {
		t.Errorf("second line Y = %v should exceed first line Y = %v",
			lineOrigins[len(lineOrigins)-1], lineOrigins[0])
	}
}

// TestFallbackResolverResolveClustersAPI exercises the public per-cluster
// resolution API, checking face, metrics and advance against references.
func TestFallbackResolverResolveClustersAPI(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, emoji)
	if err != nil {
		t.Fatal(err)
	}

	const line = "a中" + string(emojiClusterRune)
	resolved := r.ResolveClusters(line)
	if len(resolved) != 3 {
		t.Fatalf("resolved %d clusters, want 3", len(resolved))
	}

	wantTexts := []string{"a", "中", string(emojiClusterRune)}
	wantKinds := []FaceKind{KindLatin, KindMPlus, KindEmoji}
	var totalAdvance float64
	for i, rc := range resolved {
		if rc.Text != wantTexts[i] {
			t.Errorf("cluster %d text = %q, want %q", i, rc.Text, wantTexts[i])
		}
		if got := classifyFace(fx, rc.Face); got != wantKinds[i] {
			t.Errorf("cluster %d face = %s, want %s", i, got, wantKinds[i])
		}
		if rc.FaceIndex != i {
			t.Errorf("cluster %d FaceIndex = %d, want %d", i, rc.FaceIndex, i)
		}
		// Byte ranges must be contiguous and cover the whole line.
		if i > 0 && rc.StartIndexInBytes != resolved[i-1].EndIndexInBytes {
			t.Errorf("cluster %d not contiguous: start %d, prev end %d",
				i, rc.StartIndexInBytes, resolved[i-1].EndIndexInBytes)
		}
		// Metrics come from the selected face.
		if rc.Metrics != rc.Face.Metrics() {
			t.Errorf("cluster %d metrics do not match its face", i)
		}
		wantAdv := rc.Face.advanceAt(rc.Text, len(rc.Text))
		if math.Abs(rc.Advance-wantAdv) > fallbackTestEps {
			t.Errorf("cluster %d advance = %v, want %v", i, rc.Advance, wantAdv)
		}
		totalAdvance += rc.Advance
	}
	if resolved[0].StartIndexInBytes != 0 || resolved[2].EndIndexInBytes != len(line) {
		t.Errorf("clusters do not cover the full line: %+v", resolved)
	}
	// The sum of per-cluster advances equals the line advance.
	if math.Abs(totalAdvance-r.advanceAt(line, len(line))) > fallbackTestEps {
		t.Errorf("sum of cluster advances = %v, want line advance %v",
			totalAdvance, r.advanceAt(line, len(line)))
	}

	// Only the first line is resolved.
	multi := r.ResolveClusters("a\nb")
	if len(multi) != 1 || multi[0].Text != "a" {
		t.Errorf("multi-line ResolveClusters = %+v, want just the first line", multi)
	}
	if r.ResolveClusters("") != nil {
		t.Error("empty input should resolve to nil")
	}
}

// TestFallbackResolverLanguageFilter verifies a language-specific candidate
// only matches when the primary face's language shares its base language.
func TestFallbackResolverLanguageFilter(t *testing.T) {
	fx := newFallbackFixture(t)

	// Two distinct Han candidates: one tagged for Japanese, one for Chinese.
	jpFace := NewLimitedFace(fx.mplus)
	jpFace.AddUnicodeRange(0x2E80, 0x9FFF)
	jp, err := NewScriptFallbackCandidate("Hani", language.Japanese, jpFace)
	if err != nil {
		t.Fatal(err)
	}
	zhFace := NewLimitedFace(fx.mplus)
	zhFace.AddUnicodeRange(0x2E80, 0x9FFF)
	zh, err := NewScriptFallbackCandidate("Hani", language.Chinese, zhFace)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Japanese primary selects Japanese candidate", func(t *testing.T) {
		jpPrimary := &GoTextFace{Source: fx.latin.Source, Size: 16, Language: language.Japanese}
		r, err := NewTextFontFallbackResolver(jpPrimary, zh, jp)
		if err != nil {
			t.Fatal(err)
		}
		plan := r.plan("中")
		// The Chinese candidate (chain index 1) must be skipped on language;
		// the Japanese candidate (index 2) wins.
		if plan.faceIndex[0] != 2 {
			t.Errorf("face index = %d, want 2 (Japanese candidate)", plan.faceIndex[0])
		}
	})

	t.Run("Und primary skips language-specific candidates", func(t *testing.T) {
		// No language on the primary: both specific candidates are filtered
		// out, so 中 falls back to the primary missing-glyph form.
		r, err := NewTextFontFallbackResolver(fx.latin, jp, zh)
		if err != nil {
			t.Fatal(err)
		}
		var reported bool
		r.SetErrorHandler(func(error) { reported = true })
		plan := r.plan("中")
		if plan.faceIndex[0] != 0 {
			t.Errorf("face index = %d, want 0 (primary fallback)", plan.faceIndex[0])
		}
		if !reported {
			t.Error("expected an uncovered-cluster report")
		}
	})
}

// TestFallbackExtendedPictographicTable locks the structure and representative
// values of the generated Extended_Pictographic range table.
func TestFallbackExtendedPictographicTable(t *testing.T) {
	// Ranges must be sorted and non-overlapping, and inclusive.
	for i := 1; i < len(extendedPictographicRanges); i++ {
		prev, cur := extendedPictographicRanges[i-1], extendedPictographicRanges[i]
		if cur.lo <= prev.hi {
			t.Fatalf("range %d overlaps or is unsorted: {%x,%x} after {%x,%x}", i, cur.lo, cur.hi, prev.lo, prev.hi)
		}
	}

	pictographic := []rune{
		0x00A9, 0x00AE, 0x2122, 0x231A, 0x2600, 0x2764,
		0x1F600, 0x1F300, 0x1FAF1, 0x2B50, 0x3030, 0x1FC00,
	}
	for _, r := range pictographic {
		if !isExtendedPictographic(r) {
			t.Errorf("isExtendedPictographic(%U) = false, want true", r)
		}
	}
	notPictographic := []rune{
		'A', '中', 0x0627, 0x200D, 0xFE0F, 0x1F1E6, // regional indicator is not ExtPict
		0x0031, 0x20E3,
	}
	for _, r := range notPictographic {
		if isExtendedPictographic(r) {
			t.Errorf("isExtendedPictographic(%U) = true, want false", r)
		}
	}
}

// TestFallbackGraphemeSegmentation verifies the UAX#29 split keeps combining
// marks, variation selectors and ZWJ emoji sequences together.
func TestFallbackGraphemeSegmentation(t *testing.T) {
	var gs graphemeSegmenter
	cases := []struct {
		line string
		want []string
	}{
		{"", nil},
		{"a", []string{"a"}},
		{"abc", []string{"a", "b", "c"}},
		{"é", []string{"é"}}, // e + combining acute
		{"أب", []string{"أ", "ب"}},
		{"😀", []string{"😀"}},
		{"😀︎", []string{"😀︎"}},
		{"❤️", []string{"❤️"}},       // heart + VS16
		{"🫱🏼‍🫲🏽", []string{"🫱🏼‍🫲🏽"}}, // ZWJ handshake
		{"#️⃣", []string{"#️⃣"}},     // keycap
		{"👨‍👩‍👧", []string{"👨‍👩‍👧"}}, // family ZWJ
		{"a😀中", []string{"a", "😀", "中"}},
	}
	for _, tc := range cases {
		var got []graphemeCluster
		got = gs.split(got, tc.line)
		gotTexts := make([]string, len(got))
		for i, c := range got {
			gotTexts[i] = tc.line[c.start:c.end]
			if c.start < 0 || c.end <= c.start || c.end > len(tc.line) {
				t.Errorf("%q: invalid cluster range [%d,%d)", tc.line, c.start, c.end)
			}
		}
		wantN := len(tc.want)
		if len(gotTexts) != wantN {
			t.Errorf("%q: got %d clusters %q, want %d %q", tc.line, len(gotTexts), gotTexts, wantN, tc.want)
			continue
		}
		for i := range tc.want {
			if gotTexts[i] != tc.want[i] {
				t.Errorf("%q cluster %d = %q, want %q", tc.line, i, gotTexts[i], tc.want[i])
			}
		}
	}
}

// TestFallbackClusterIsEmoji covers the emoji-cluster classifier including
// keycaps and flag sequences that contain no Extended_Pictographic rune.
func TestFallbackClusterIsEmoji(t *testing.T) {
	emojiClusters := []string{"😀", "❤️", "🫱🏼‍🫲🏽", "#️⃣", "©️", "🇯🇵"}
	for _, c := range emojiClusters {
		if !clusterIsEmoji(c) {
			t.Errorf("clusterIsEmoji(%q) = false, want true", c)
		}
	}
	plain := []string{"a", "中", "أ", "ّ", "123", "a b"}
	for _, c := range plain {
		if clusterIsEmoji(c) {
			t.Errorf("clusterIsEmoji(%q) = true, want false", c)
		}
	}
}

// TestFallbackPUASymbolRoutesViaUnknownScript verifies a Private-Use symbol
// (no Extended_Pictographic property, Unknown script) is still routed to an
// emoji candidate through the Unknown-script rule.
func TestFallbackPUASymbolRoutesViaUnknownScript(t *testing.T) {
	fx := newFallbackFixture(t)
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, emoji)
	if err != nil {
		t.Fatal(err)
	}
	plan := r.plan(string(emojiClusterRune))
	if plan.faceIndex[0] != 1 {
		t.Errorf("PUA cluster face index = %d, want 1 (emoji candidate)", plan.faceIndex[0])
	}
}

// TestFallbackResolverVertical verifies a vertical-direction resolver carries
// the fallback faces' vertical metrics and lays glyphs out along Y.
func TestFallbackResolverVertical(t *testing.T) {
	fx := newFallbackFixture(t)

	dir := DirectionTopToBottomAndRightToLeft
	verticalPrimary := &GoTextFace{Source: fx.mplus.Source, Size: 16, Direction: dir}
	verticalHan := NewLimitedFace(&GoTextFace{Source: fx.mplus.Source, Size: 16, Direction: dir})
	verticalHan.AddUnicodeRange(0x2E80, 0x9FFF)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, verticalHan)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(verticalPrimary, han)
	if err != nil {
		t.Fatal(err)
	}
	if r.direction() != dir {
		t.Errorf("direction = %v, want %v", r.direction(), dir)
	}

	// Vertical metrics are the max over the chain and are non-zero for a face
	// supporting vertical writing.
	m := r.Metrics()
	if m.VAscent <= 0 && m.VDescent <= 0 {
		t.Errorf("vertical metrics are all zero: %+v", m)
	}
	if m.VAscent != verticalPrimary.Metrics().VAscent ||
		m.VDescent != verticalPrimary.Metrics().VDescent {
		t.Errorf("vertical metrics not propagated: %+v vs %+v", m, verticalPrimary.Metrics())
	}

	// Vertical layout advances along Y, not X.
	gs := AppendLazyGlyphs(nil, "a中", r, nil)
	if len(gs) == 0 {
		t.Fatal("no vertical glyphs")
	}
	var sawYAdvance bool
	for _, g := range gs {
		if g.AdvanceY != 0 {
			sawYAdvance = true
		}
		if g.AdvanceX != 0 {
			t.Errorf("vertical glyph has non-zero AdvanceX = %v", g.AdvanceX)
		}
	}
	if !sawYAdvance {
		t.Error("no glyph advanced along Y in vertical layout")
	}
}

// TestFallbackResolverPlanCacheHit verifies the per-line plan is memoized: a
// second resolution returns the same cached plan pointer and identical
// assignment, which is what keeps cache keys reproducible.
func TestFallbackResolverPlanCacheHit(t *testing.T) {
	fx := newFallbackFixture(t)

	han, err := NewScriptFallbackCandidate("Hani", language.Und, fx.hanOnly)
	if err != nil {
		t.Fatal(err)
	}
	emoji, err := NewEmojiFallbackCandidate(fx.emojiOnly)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewTextFontFallbackResolver(fx.latin, han, emoji)
	if err != nil {
		t.Fatal(err)
	}

	const line = "a中" + string(emojiClusterRune)
	first := r.plan(line)
	second := r.plan(line)
	if first != second {
		t.Error("repeated plan() did not return the cached plan")
	}

	// A distinct line gets its own plan, split one cluster per rune.
	other := r.plan("different")
	if other == first {
		t.Error("different lines share a plan")
	}
	if want := utf8RuneCount("different"); len(other.clusters) != want {
		t.Errorf("different-line cluster count = %d, want %d", len(other.clusters), want)
	}
}

// TestFallbackResolverInvalidUTF8 verifies invalid byte sequences are split
// into one cluster per (replacement) rune and never crash the layout APIs.
func TestFallbackResolverInvalidUTF8(t *testing.T) {
	fx := newFallbackFixture(t)
	r, err := NewTextFontFallbackResolver(fx.latin)
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range []string{"a\xffb", "\xffZ", "Z\xff\xfeA"} {
		var cs []graphemeCluster
		cs = r.segmenter.split(cs, line)
		if len(cs) != utf8RuneCount(line) {
			t.Errorf("% x: %d clusters, want %d", line, len(cs), utf8RuneCount(line))
		}
		// The public APIs must not panic and indices stay in range.
		_ = r.advanceAt(line, len(line))
		gs := AppendLazyGlyphs(nil, line, r, nil)
		for _, g := range gs {
			if g.StartIndexInBytes < 0 || g.EndIndexInBytes > len(line) {
				t.Errorf("glyph range out of bounds for % x", line)
			}
		}
		if rc := r.ResolveClusters(line); len(rc) != len(cs) {
			t.Errorf("% x: resolved %d clusters, want %d", line, len(rc), len(cs))
		}
	}
}

// keep image import used even if a future edit removes the only direct use.
var _ = image.Rectangle{}
