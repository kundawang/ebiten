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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func newSnapshotSource(t *testing.T, data []byte) *text.GoTextFaceSource {
	t.Helper()
	src, err := text.NewGoTextFaceSource(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return src
}

func snapshotTestFontData() []byte {
	return fonts.MPlus1pRegular_ttf
}

// snapshotErrorPath returns the field path of a snapshot error, or "" if err is
// nil or not a snapshot error.
func snapshotErrorPath(err error) string {
	var se *text.TextLayoutSnapshotError
	if errors.As(err, &se) {
		return se.Path
	}
	return ""
}

func mustSnapshot(t *testing.T, s string, face text.Face, op *text.TextLayoutSnapshotOptions) *text.TextLayoutSnapshot {
	t.Helper()
	snap, err := text.NewTextLayoutSnapshot(s, face, op)
	if err != nil {
		t.Fatalf("NewTextLayoutSnapshot(%q): %v", s, err)
	}
	return snap
}

// lineStarts returns the byte offset of each line start, following the same
// line-break rules as text/v2.
func lineStarts(s string) []int {
	var starts []int
	off := 0
	for len(s) > 0 {
		starts = append(starts, off)
		idx := -1
		for i, r := range s {
			if textutil.IsLineBreak(r) {
				idx = i
				break
			}
		}
		if idx < 0 {
			break
		}
		_, size := decodeRune(s[idx:])
		step := idx + size
		if s[idx] == '\r' && step < len(s) && s[step] == '\n' {
			step++
		}
		off += step
		s = s[step:]
	}
	return starts
}

func decodeRune(s string) (rune, int) {
	return utf8.DecodeRuneInString(s)
}

// TestSnapshotMatchesLiveLayout compares a snapshot's lines and glyphs against
// AppendLazyGlyphs for mixed scripts, RTL, CJK, emoji, combining marks, and
// empty lines.
func TestSnapshotMatchesLiveLayout(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())

	for _, tc := range []struct {
		name      string
		str       string
		direction text.Direction
		align     text.Align
		secondary text.Align
		spacing   float64
	}{
		{name: "Latin", str: "Hello, world!"},
		{name: "MixedScripts", str: "Hello 世界 مرحبا שלום สวัสดี"},
		{name: "RTL", str: "مرحبا بالعالم", direction: text.DirectionRightToLeft},
		{name: "RTLCenter", str: "שלום עולם", direction: text.DirectionRightToLeft, align: text.AlignCenter},
		{name: "CJK", str: "日本語のテキストを描画する"},
		{name: "Combining", str: "áèñö"},
		{name: "Newlines", str: "first\nsecond\nthird", spacing: 30},
		{name: "EmptyLines", str: "a\n\n\nb", spacing: 24},
		{name: "LeadingEmptyLine", str: "\nabc", spacing: 24},
		{name: "TrailingNewline", str: "abc\n", spacing: 24},
		{name: "OnlyNewlines", str: "\n\n", spacing: 24},
		{name: "CRLF", str: "abc\r\ndef", spacing: 24},
		{name: "Aligned", str: "one\ntwo\nthree", align: text.AlignCenter, secondary: text.AlignCenter, spacing: 22},
		{name: "Vertical", str: "縦書き\nテスト", direction: text.DirectionTopToBottomAndRightToLeft, spacing: 30},
	} {
		t.Run(tc.name, func(t *testing.T) {
			face := &text.GoTextFace{
				Source:    src,
				Size:      24,
				Direction: tc.direction,
			}
			layoutOp := text.LayoutOptions{
				LineSpacing:    tc.spacing,
				PrimaryAlign:   tc.align,
				SecondaryAlign: tc.secondary,
			}
			var snapOp text.TextLayoutSnapshotOptions
			snapOp.LayoutOptions = layoutOp
			snap := mustSnapshot(t, tc.str, face, &snapOp)

			live := text.AppendLazyGlyphs(nil, tc.str, face, &layoutOp)
			got := snap.Glyphs()
			if len(got) != len(live) {
				t.Fatalf("glyph count: got %d, want %d", len(got), len(live))
			}
			for i := range live {
				lg := &live[i]
				sg := &got[i]
				if sg.StartIndexInBytes != lg.StartIndexInBytes ||
					sg.EndIndexInBytes != lg.EndIndexInBytes ||
					sg.GID != lg.GID ||
					sg.ImageBounds != lg.ImageBounds ||
					sg.OriginX != lg.OriginX ||
					sg.OriginY != lg.OriginY ||
					sg.OriginOffsetX != lg.OriginOffsetX ||
					sg.OriginOffsetY != lg.OriginOffsetY ||
					sg.AdvanceX != lg.AdvanceX ||
					sg.AdvanceY != lg.AdvanceY {
					t.Fatalf("glyph %d differs: got %+v, want start=%d end=%d gid=%d bounds=%v origin=(%v,%v) offset=(%v,%v) adv=(%v,%v)",
						i, sg, lg.StartIndexInBytes, lg.EndIndexInBytes, lg.GID, lg.ImageBounds,
						lg.OriginX, lg.OriginY, lg.OriginOffsetX, lg.OriginOffsetY, lg.AdvanceX, lg.AdvanceY)
				}
				if sg.HasImage != !lg.ImageBounds.Empty() {
					t.Errorf("glyph %d: HasImage = %v, want %v", i, sg.HasImage, !lg.ImageBounds.Empty())
				}
				if sg.Colored != lg.Colored() {
					t.Errorf("glyph %d: Colored = %v, want %v", i, sg.Colored, lg.Colored())
				}
			}

			// Line boundaries must match the text's lines and cover it
			// exactly: each line's break length plus content reaches the
			// next line's start.
			lines := snap.Lines()
			starts := lineStarts(tc.str)
			if len(lines) != len(starts) {
				t.Fatalf("line count: got %d, want %d (%v)", len(lines), len(starts), starts)
			}
			for i := range lines {
				if lines[i].IndexInBytes != starts[i] {
					t.Errorf("line %d: index = %d, want %d", i, lines[i].IndexInBytes, starts[i])
				}
				if lines[i].EndIndexInBytes+lines[i].LineBreakLen > len(tc.str) {
					t.Errorf("line %d: end %d + break %d exceeds text length %d", i, lines[i].EndIndexInBytes, lines[i].LineBreakLen, len(tc.str))
				}
			}

			// Every non-terminator byte must be covered by some glyph.
			covered := make([]bool, len(tc.str))
			for _, g := range got {
				for i := g.StartIndexInBytes; i < g.EndIndexInBytes; i++ {
					covered[i] = true
				}
			}
			for i := 0; i < len(tc.str); {
				r, size := decodeRune(tc.str[i:])
				if !textutil.IsLineBreak(r) && !covered[i] {
					t.Errorf("byte %d (%q) is not covered by any glyph", i, r)
				}
				i += size
			}
		})
	}
}

// TestSnapshotRecordCopies checks that the accessors return copies and that
// mutations do not leak back into the snapshot.
func TestSnapshotRecordCopies(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "abc", face, nil)

	wantLine := snap.Lines()[0]
	wantGlyph := snap.Glyphs()[0]

	lines := snap.Lines()
	lines[0].IndexInBytes = 99
	glyphs := snap.Glyphs()
	for i := range glyphs {
		glyphs[i].GID = ^uint32(0)
	}

	if got := snap.Lines()[0]; got != wantLine {
		t.Errorf("mutating Lines() changed the snapshot: got %+v, want %+v", got, wantLine)
	}
	if got := snap.Glyphs()[0]; got != wantGlyph {
		t.Errorf("mutating Glyphs() changed the snapshot: got %+v, want %+v", got, wantGlyph)
	}
	if snap.Text() != "abc" {
		t.Errorf("Text() = %q, want %q", snap.Text(), "abc")
	}
}

// TestSnapshotDrawMatchesLiveGlyphs renders text directly and through a
// snapshot, and compares the pixels.
//
// The reference under a scale or rotation is a manual drawing of the live
// glyphs rather than [text.Draw]: Ebitengine's atlas realization makes Draw
// itself differ from a manual drawing by a few edge pixels under scaling, so
// Draw is not a stable pixel reference there. Under the identity transform,
// Draw and the manual drawing agree, and both are checked.
func TestSnapshotDrawMatchesLiveGlyphs(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())

	for _, tc := range []struct {
		name  string
		str   string
		geoM  func(*ebiten.GeoM)
		align text.Align
	}{
		{name: "Plain", str: "Hello, 世界!"},
		{name: "Translated", str: "Hello, 世界!", geoM: func(g *ebiten.GeoM) { g.Translate(7, 11) }},
		{name: "Rotated", str: "Rotated text", geoM: func(g *ebiten.GeoM) { g.Rotate(math.Pi / 6) }},
		{name: "ScaledRotated", str: "Scale & rotate", geoM: func(g *ebiten.GeoM) {
			g.Scale(1.5, 1.5)
			g.Rotate(math.Pi / 4)
			g.Translate(64, 64)
		}},
		{name: "Multiline", str: "first line\nsecond line\nthird", align: text.AlignCenter},
		{name: "RTLLines", str: "مرحبا\nبالعالم", align: text.AlignEnd},
	} {
		t.Run(tc.name, func(t *testing.T) {
			face := &text.GoTextFace{Source: src, Size: 22}
			layoutOp := text.LayoutOptions{
				LineSpacing:  26,
				PrimaryAlign: tc.align,
			}
			var geoM ebiten.GeoM
			if tc.geoM != nil {
				tc.geoM(&geoM)
			}

			const w, h = 200, 160

			// Manual reference: draw the live glyphs with the same GeoM.
			want := ebiten.NewImage(w, h)
			defer want.Deallocate()
			for _, g := range text.AppendGlyphs(nil, tc.str, face, &layoutOp) {
				if g.Image == nil {
					continue
				}
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(g.X, g.Y)
				op.GeoM.Concat(geoM)
				want.DrawImage(g.Image, op)
			}

			snap := mustSnapshot(t, tc.str, face, &text.TextLayoutSnapshotOptions{LayoutOptions: layoutOp, GeoM: geoM})
			got := ebiten.NewImage(w, h)
			defer got.Deallocate()
			if err := snap.Draw(got, face, nil); err != nil {
				t.Fatal(err)
			}
			if tc.geoM != nil {
				assertImagesClose(t, got, want, "snapshot Draw")
			} else {
				assertImagesEqual(t, got, want, "snapshot Draw")
			}

			// Under the identity transform, Draw itself must match.
			if tc.geoM != nil {
				return
			}
			drawn := ebiten.NewImage(w, h)
			defer drawn.Deallocate()
			var drawOp text.DrawOptions
			drawOp.LayoutOptions = layoutOp
			text.Draw(drawn, tc.str, face, &drawOp)
			assertImagesEqual(t, got, drawn, "Draw")
		})
	}
}

// assertImagesClose compares images allowing a few edge pixels to differ by a
// small amount. Under a scale or rotation, Ebitengine's atlas placement changes
// which texels linear sampling picks at a glyph's edge; the same artifact makes
// Draw itself differ from a manual drawing, so exact equality is not a property
// of the engine there. The tolerance is far below a visible difference.
func assertImagesClose(t *testing.T, got, want *ebiten.Image, what string) {
	t.Helper()
	if got.Bounds() != want.Bounds() {
		t.Fatalf("%s: bounds: got %v, want %v", what, got.Bounds(), want.Bounds())
	}
	b := got.Bounds()
	n := 4 * b.Dx() * b.Dy()
	gotPix := make([]byte, n)
	wantPix := make([]byte, n)
	got.ReadPixels(gotPix)
	want.ReadPixels(wantPix)
	const maxChannelDelta = 64
	const maxDiffFraction = 0.005
	diffPixels, totalPixels := 0, b.Dx()*b.Dy()
	for i := 0; i < n; i += 4 {
		close := true
		for c := 0; c < 4; c++ {
			d := int(gotPix[i+c]) - int(wantPix[i+c])
			if d < 0 {
				d = -d
			}
			if d > maxChannelDelta {
				close = false
				break
			}
		}
		if !close {
			diffPixels++
		}
	}
	if float64(diffPixels) > maxDiffFraction*float64(totalPixels) {
		t.Errorf("%s differs from the reference in %d of %d pixels", what, diffPixels, totalPixels)
	}
}

func assertImagesEqual(t *testing.T, got, want *ebiten.Image, what string) {
	t.Helper()
	if got.Bounds() != want.Bounds() {
		t.Fatalf("%s: bounds: got %v, want %v", what, got.Bounds(), want.Bounds())
	}
	b := got.Bounds()
	wantPix := make([]byte, 4*b.Dx()*b.Dy())
	gotPix := make([]byte, 4*b.Dx()*b.Dy())
	got.ReadPixels(gotPix)
	want.ReadPixels(wantPix)
	if !bytes.Equal(gotPix, wantPix) {
		t.Errorf("%s differs from the reference", what)
	}
}

// TestSnapshotAppendGlyphsMatchesAppendGlyphs checks that replaying a
// snapshot's glyphs yields the same positions and images as AppendGlyphs.
func TestSnapshotAppendGlyphsMatchesAppendGlyphs(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())
	face := &text.GoTextFace{Source: src, Size: 24}
	const str = "Replay\n混合 text"

	layoutOp := text.LayoutOptions{LineSpacing: 30}
	snap := mustSnapshot(t, str, face, &text.TextLayoutSnapshotOptions{LayoutOptions: layoutOp})
	live := text.AppendGlyphs(nil, str, face, &layoutOp)
	replayed, err := snap.AppendGlyphs(nil, face)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayed) != len(live) {
		t.Fatalf("glyph count: got %d, want %d", len(replayed), len(live))
	}
	for i := range live {
		a, b := &live[i], &replayed[i]
		if a.StartIndexInBytes != b.StartIndexInBytes ||
			a.EndIndexInBytes != b.EndIndexInBytes ||
			a.GID != b.GID ||
			a.X != b.X || a.Y != b.Y ||
			a.OriginX != b.OriginX || a.OriginY != b.OriginY ||
			a.OriginOffsetX != b.OriginOffsetX || a.OriginOffsetY != b.OriginOffsetY ||
			a.AdvanceX != b.AdvanceX || a.AdvanceY != b.AdvanceY ||
			a.Colored != b.Colored {
			t.Errorf("glyph %d differs: got %+v, want %+v", i, *b, *a)
		}
		if (a.Image == nil) != (b.Image == nil) {
			t.Errorf("glyph %d: image presence differs: got %v, want %v", i, b.Image != nil, a.Image != nil)
		}
		if a.Image != nil && a.Image.Bounds() != b.Image.Bounds() {
			t.Errorf("glyph %d: image bounds: got %v, want %v", i, b.Image.Bounds(), a.Image.Bounds())
		}
	}
}

// TestSnapshotVectorPathMatchesAppend checks that vector outlines exported from
// a snapshot match AppendVectorPath for an identity GeoM.
func TestSnapshotVectorPathMatchesAppend(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())
	const str = "Vector\nパス"

	for _, direction := range []text.Direction{
		text.DirectionLeftToRight,
		text.DirectionRightToLeft,
		text.DirectionTopToBottomAndRightToLeft,
	} {
		face := &text.GoTextFace{Source: src, Size: 30, Direction: direction}
		t.Run(directionName(direction), func(t *testing.T) {
			snap := mustSnapshot(t, str, face, nil)

			var want, got vector.Path
			text.AppendVectorPath(&want, str, face, nil)
			if err := snap.AppendVectorPath(&got, face); err != nil {
				t.Fatal(err)
			}
			if pathSignature(&want) != pathSignature(&got) {
				t.Error("snapshot vector path differs from AppendVectorPath")
			}
		})
	}
}

func directionName(d text.Direction) string {
	switch d {
	case text.DirectionLeftToRight:
		return "LTR"
	case text.DirectionRightToLeft:
		return "RTL"
	case text.DirectionTopToBottomAndLeftToRight:
		return "TTB_LTR"
	case text.DirectionTopToBottomAndRightToLeft:
		return "TTB_RTL"
	}
	return "Invalid"
}

// TestSnapshotVectorPathAppliesGeoM checks that the snapshot's GeoM transforms
// its exported outlines.
func TestSnapshotVectorPathAppliesGeoM(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 40}
	const str = "GeoM"

	var geoM ebiten.GeoM
	geoM.Scale(2, 3)
	geoM.Translate(5, 7)

	snap := mustSnapshot(t, str, face, &text.TextLayoutSnapshotOptions{GeoM: geoM})
	var transformed vector.Path
	if err := snap.AppendVectorPath(&transformed, face); err != nil {
		t.Fatal(err)
	}

	var plain vector.Path
	text.AppendVectorPath(&plain, str, face, nil)
	var want vector.Path
	want.AddPath(&plain, &vector.AddPathOptions{GeoM: geoM})

	if pathSignature(&transformed) != pathSignature(&want) {
		t.Error("snapshot vector path with GeoM differs from a transformed AppendVectorPath")
	}
}

// TestSnapshotRoundTrip checks that MarshalBinary and DecodeTextLayoutSnapshot
// reproduce the snapshot's data exactly.
func TestSnapshotRoundTrip(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())
	face := &text.GoTextFace{Source: src, Size: 27, Direction: text.DirectionLeftToRight}
	const str = "Round trip 往復 🎉\nsecond line"

	layoutOp := text.LayoutOptions{LineSpacing: 33, PrimaryAlign: text.AlignEnd}
	var geoM ebiten.GeoM
	geoM.Rotate(0.3)
	geoM.Translate(3, 4)

	snap := mustSnapshot(t, str, face, &text.TextLayoutSnapshotOptions{LayoutOptions: layoutOp, GeoM: geoM})
	data, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := text.DecodeTextLayoutSnapshot(data)
	if err != nil {
		t.Fatal(err)
	}

	if decoded.Text() != snap.Text() {
		t.Errorf("text: got %q, want %q", decoded.Text(), snap.Text())
	}
	if decoded.Direction() != snap.Direction() {
		t.Errorf("direction: got %v, want %v", decoded.Direction(), snap.Direction())
	}
	if decoded.LayoutOptions() != snap.LayoutOptions() {
		t.Errorf("layout options: got %+v, want %+v", decoded.LayoutOptions(), snap.LayoutOptions())
	}
	if got, want := decoded.GeoM(), snap.GeoM(); got != want {
		t.Errorf("GeoM: got %v, want %v", got, want)
	}
	if decoded.Metrics() != snap.Metrics() {
		t.Errorf("metrics: got %+v, want %+v", decoded.Metrics(), snap.Metrics())
	}
	if got, want := decoded.Fingerprint(), snap.Fingerprint(); !got.Equal(want) {
		t.Errorf("fingerprint: got %v, want %v", got, want)
	}
	if !equalLines(decoded.Lines(), snap.Lines()) {
		t.Error("lines differ after round trip")
	}
	if !equalGlyphs(decoded.Glyphs(), snap.Glyphs()) {
		t.Error("glyphs differ after round trip")
	}
	if w0, h0 := decoded.Bounds(); true {
		w1, h1 := snap.Bounds()
		if w0 != w1 || h0 != h1 {
			t.Errorf("bounds: got (%v, %v), want (%v, %v)", w0, h0, w1, h1)
		}
	}

	// Re-encoding the decoded snapshot must be byte-identical.
	data2, err := decoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, data2) {
		t.Error("re-encoding a decoded snapshot changed the bytes")
	}

	// Decoded snapshots must replay.
	dst := ebiten.NewImage(240, 160)
	defer dst.Deallocate()
	if err := decoded.Draw(dst, face, nil); err != nil {
		t.Fatalf("Draw after decode: %v", err)
	}
}

// TestSnapshotRejectsChangedFace checks that replay against a different face is
// rejected with a field path.
func TestSnapshotRejectsChangedFace(t *testing.T) {
	srcA := newSnapshotSource(t, goregular.TTF)
	srcB := newSnapshotSource(t, snapshotTestFontData())
	faceA := &text.GoTextFace{Source: srcA, Size: 20}
	snap := mustSnapshot(t, "abc", faceA, nil)

	changedSize := *faceA
	changedSize.Size = 28
	changedDir := *faceA
	changedDir.Direction = text.DirectionRightToLeft
	otherFace := &text.GoTextFace{Source: srcB, Size: 20}
	var nilSource text.GoTextFace

	for _, tc := range []struct {
		name string
		face text.Face
	}{
		{name: "DifferentSize", face: &changedSize},
		{name: "DifferentDirection", face: &changedDir},
		{name: "DifferentFont", face: otherFace},
		{name: "NilSource", face: &nilSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dst := ebiten.NewImage(32, 32)
			defer dst.Deallocate()
			err := snap.Draw(dst, tc.face, nil)
			if err == nil {
				t.Fatal("Draw must reject a changed face")
			}
			if path := snapshotErrorPath(err); path != "face" && path != "face.source" {
				t.Errorf("error path = %q, want a face field", path)
			}
		})
	}
}

// TestSnapshotRejectsChangedVariation checks that changing a variable font's
// coordinates invalidates replay.
func TestSnapshotRejectsChangedVariation(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "RobotoFlex.ttf"))
	if err != nil {
		t.Fatal(err)
	}
	src := newSnapshotSource(t, data)
	face := &text.GoTextFace{Source: src, Size: 30}
	face.SetVariation(text.MustParseTag("wght"), 100)
	snap := mustSnapshot(t, "Variable", face, nil)

	face.SetVariation(text.MustParseTag("wght"), 900)
	dst := ebiten.NewImage(64, 64)
	defer dst.Deallocate()
	if err := snap.Draw(dst, face, nil); err == nil {
		t.Fatal("Draw must reject a face with changed variations")
	}
}

// TestSnapshotFingerprint checks the fingerprint's equality semantics.
func TestSnapshotFingerprint(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	otherSrc := newSnapshotSource(t, snapshotTestFontData())

	a := &text.GoTextFace{Source: src, Size: 20}
	b := &text.GoTextFace{Source: src, Size: 20}
	c := &text.GoTextFace{Source: src, Size: 21}
	d := &text.GoTextFace{Source: otherSrc, Size: 20}

	fpA, err := text.FaceFingerprint(a)
	if err != nil {
		t.Fatal(err)
	}
	fpB, err := text.FaceFingerprint(b)
	if err != nil {
		t.Fatal(err)
	}
	fpC, err := text.FaceFingerprint(c)
	if err != nil {
		t.Fatal(err)
	}
	fpD, err := text.FaceFingerprint(d)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := text.FaceFingerprint(nil); err == nil {
		t.Error("FaceFingerprint(nil) must fail")
	}
	if !fpA.Equal(fpB) {
		t.Error("identical faces must have equal fingerprints")
	}
	if fpA.Equal(fpC) {
		t.Error("faces of different sizes must have different fingerprints")
	}
	if fpA.Equal(fpD) {
		t.Error("faces with different fonts must have different fingerprints")
	}
	if fpA.String() == "" {
		t.Error("fingerprint String() must not be empty")
	}
}

// TestSnapshotDecodeErrors checks that corrupted, truncated, and unknown-version
// data are rejected with the right field path.
func TestSnapshotDecodeErrors(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "decode errors\nsecond", face, nil)
	valid, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	mutate := func(f func([]byte) []byte) []byte {
		return f(bytes.Clone(valid))
	}

	for _, tc := range []struct {
		name     string
		data     []byte
		wantPath string
	}{
		{name: "Empty", data: nil, wantPath: "data"},
		{name: "ShortHeader", data: valid[:8], wantPath: "data"},
		{name: "BadMagic", data: mutate(func(b []byte) []byte {
			b[0] = 'X'
			return b
		}), wantPath: "magic"},
		{name: "UnknownVersion", data: mutate(func(b []byte) []byte {
			binary.LittleEndian.PutUint32(b[8:12], 999)
			return b
		}), wantPath: "version"},
		{name: "TruncatedPayload", data: valid[:len(valid)/2], wantPath: "data"},
		{name: "CorruptChecksum", data: mutate(func(b []byte) []byte {
			b[30] ^= 0xff
			return b
		}), wantPath: "checksum"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := text.DecodeTextLayoutSnapshot(tc.data)
			if err == nil {
				t.Fatal("DecodeTextLayoutSnapshot must fail")
			}
			if got := snapshotErrorPath(err); got != tc.wantPath {
				t.Errorf("error path = %q, want %q (err: %v)", got, tc.wantPath, err)
			}
		})
	}
}

// TestSnapshotDecodeRejectsNonFinite rewrites a valid payload's line spacing to
// NaN, reseals the checksum, and checks that decoding rejects it by field path.
func TestSnapshotDecodeRejectsNonFinite(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "hostile", face, nil)
	valid, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	payload := bytes.Clone(valid[20 : len(valid)-32])
	off := 0
	// text: an 8-byte little-endian length then the bytes.
	if len(payload) < 8 {
		t.Fatal("payload is too short")
	}
	n := binary.LittleEndian.Uint64(payload[off:])
	off += 8 + int(n)
	off += 8 // direction
	off += 8 // primaryAlign
	off += 8 // secondaryAlign
	binary.LittleEndian.PutUint64(payload[off:off+8], math.Float64bits(math.NaN()))

	out := bytes.Clone(valid[:20])
	out = append(out, payload...)
	sum := sha256.Sum256(payload)
	out = append(out, sum[:]...)
	binary.LittleEndian.PutUint64(out[12:20], uint64(len(payload)))

	_, err = text.DecodeTextLayoutSnapshot(out)
	if err == nil {
		t.Fatal("decoding a non-finite line spacing must fail")
	}
	if got := snapshotErrorPath(err); got != "lineSpacing" {
		t.Errorf("error path = %q, want %q (err: %v)", got, "lineSpacing", err)
	}
}

// TestSnapshotDecodeRejectsOutOfRangeGlyph rewrites a glyph's byte range past
// the text length, reseals the checksum, and checks the field path.
func TestSnapshotDecodeRejectsOutOfRangeGlyph(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "abc", face, nil)
	valid, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	// Parse the payload enough to locate the first glyph's byte range.
	payload := bytes.Clone(valid[20 : len(valid)-32])
	off := 0
	readU64 := func() uint64 {
		v := binary.LittleEndian.Uint64(payload[off:])
		off += 8
		return v
	}
	readF64 := func() float64 {
		return math.Float64frombits(readU64())
	}
	n := binary.LittleEndian.Uint64(payload[off:])
	off += 8 + int(n) // text
	readU64()         // direction
	readU64()         // primaryAlign
	readU64()         // secondaryAlign
	readF64()         // lineSpacing
	for range 6 {
		readF64() // geoM
	}
	for range 8 {
		readF64() // metrics
	}
	readF64() // width
	readF64() // height
	off += 32 // fingerprint hash
	descLen := readU64()
	off += int(descLen) // fingerprint description
	lineCount := readU64()
	for range lineCount {
		readU64() // indexInBytes
		readU64() // endIndexInBytes
		readU64() // lineBreakLen
		readF64() // advance
		readF64() // baselineX
		readF64() // baselineY
	}
	glyphCount := readU64()
	if glyphCount == 0 {
		t.Fatal("snapshot has no glyphs")
	}
	// Overwrite the first glyph's startIndexInBytes with a huge value.
	binary.LittleEndian.PutUint64(payload[off:off+8], 1<<40)

	out := bytes.Clone(valid[:20])
	out = append(out, payload...)
	sum := sha256.Sum256(payload)
	out = append(out, sum[:]...)
	binary.LittleEndian.PutUint64(out[12:20], uint64(len(payload)))

	_, err = text.DecodeTextLayoutSnapshot(out)
	if err == nil {
		t.Fatal("decoding an out-of-range glyph must fail")
	}
	if got := snapshotErrorPath(err); got != "glyphs[0].startIndexInBytes" {
		t.Errorf("error path = %q, want %q (err: %v)", got, "glyphs[0].startIndexInBytes", err)
	}
}

// TestSnapshotEmptyText checks that an empty text produces an empty snapshot.
func TestSnapshotEmptyText(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "", face, nil)
	if len(snap.Lines()) != 0 || len(snap.Glyphs()) != 0 {
		t.Errorf("empty text produced %d lines and %d glyphs", len(snap.Lines()), len(snap.Glyphs()))
	}
	data, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := text.DecodeTextLayoutSnapshot(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Lines()) != 0 || len(decoded.Glyphs()) != 0 {
		t.Error("decoded empty snapshot is not empty")
	}
}

// TestSnapshotInvalidOptions checks that non-finite or invalid options are
// rejected with a field path rather than panicking.
func TestSnapshotInvalidOptions(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}

	var nanGeoM, infGeoM ebiten.GeoM
	nanGeoM.SetElement(0, 0, math.NaN())
	infGeoM.SetElement(1, 2, math.Inf(1))

	for _, tc := range []struct {
		name     string
		face     text.Face
		op       *text.TextLayoutSnapshotOptions
		wantPath string
	}{
		{name: "NilFace", face: nil, wantPath: "face"},
		{name: "NaNLineSpacing", face: face,
			op:       &text.TextLayoutSnapshotOptions{LayoutOptions: text.LayoutOptions{LineSpacing: math.NaN()}},
			wantPath: "options.lineSpacing"},
		{name: "InfLineSpacing", face: face,
			op:       &text.TextLayoutSnapshotOptions{LayoutOptions: text.LayoutOptions{LineSpacing: math.Inf(-1)}},
			wantPath: "options.lineSpacing"},
		{name: "InvalidPrimaryAlign", face: face,
			op:       &text.TextLayoutSnapshotOptions{LayoutOptions: text.LayoutOptions{PrimaryAlign: text.Align(99)}},
			wantPath: "options.primaryAlign"},
		{name: "InvalidSecondaryAlign", face: face,
			op:       &text.TextLayoutSnapshotOptions{LayoutOptions: text.LayoutOptions{SecondaryAlign: text.Align(99)}},
			wantPath: "options.secondaryAlign"},
		{name: "NaNGeoM", face: face,
			op:       &text.TextLayoutSnapshotOptions{GeoM: nanGeoM},
			wantPath: "options.geoM[0]"},
		{name: "InfGeoM", face: face,
			op:       &text.TextLayoutSnapshotOptions{GeoM: infGeoM},
			wantPath: "options.geoM[5]"},
		{name: "InvalidDirection", face: &text.GoTextFace{Source: src, Size: 20, Direction: text.Direction(42)},
			wantPath: "face.direction"},
		{name: "NilFaceSource", face: &text.GoTextFace{Size: 20},
			wantPath: "face.source"},
		{name: "NaNGoTextFaceSize", face: &text.GoTextFace{Source: src, Size: math.NaN()},
			wantPath: "face.size"},
		{name: "InfGoTextFaceSize", face: &text.GoTextFace{Source: src, Size: math.Inf(1)},
			wantPath: "face.size"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := text.NewTextLayoutSnapshot("abc", tc.face, tc.op)
			if err == nil {
				t.Fatal("NewTextLayoutSnapshot must fail")
			}
			if got := snapshotErrorPath(err); got != tc.wantPath {
				t.Errorf("error path = %q, want %q (err: %v)", got, tc.wantPath, err)
			}
		})
	}
}

// TestSnapshotInvalidText checks that invalid UTF-8 is rejected.
func TestSnapshotInvalidText(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	_, err := text.NewTextLayoutSnapshot("a\xffb", face, nil)
	if err == nil {
		t.Fatal("NewTextLayoutSnapshot must reject invalid UTF-8")
	}
	if got := snapshotErrorPath(err); got != "text" {
		t.Errorf("error path = %q, want %q", got, "text")
	}
}

// TestSnapshotMultiFace checks that a snapshot over a composite face replays
// and that a face whose composition changes is rejected.
func TestSnapshotMultiFace(t *testing.T) {
	latin := &text.GoTextFace{Source: newSnapshotSource(t, goregular.TTF), Size: 20}
	cjk := &text.GoTextFace{Source: newSnapshotSource(t, snapshotTestFontData()), Size: 20}
	multi, err := text.NewMultiFace(latin, cjk)
	if err != nil {
		t.Fatal(err)
	}
	const str = "Latin 日本語 mixed"
	snap := mustSnapshot(t, str, multi, nil)

	const w, h = 220, 60
	dst := ebiten.NewImage(w, h)
	defer dst.Deallocate()
	want := ebiten.NewImage(w, h)
	defer want.Deallocate()
	text.Draw(want, str, multi, nil)
	if err := snap.Draw(dst, multi, nil); err != nil {
		t.Fatal(err)
	}
	wantPix := make([]byte, 4*w*h)
	gotPix := make([]byte, 4*w*h)
	want.ReadPixels(wantPix)
	dst.ReadPixels(gotPix)
	if !bytes.Equal(gotPix, wantPix) {
		t.Error("MultiFace snapshot Draw differs from Draw")
	}

	reordered, err := text.NewMultiFace(cjk, latin)
	if err != nil {
		t.Fatal(err)
	}
	if err := snap.Draw(dst, reordered, nil); err == nil {
		t.Fatal("Draw must reject a reordered MultiFace")
	}
}

// TestSnapshotLimitedFace checks that a snapshot over a LimitedFace replays and
// is invalidated when its ranges change.
func TestSnapshotLimitedFace(t *testing.T) {
	base := &text.GoTextFace{Source: newSnapshotSource(t, goregular.TTF), Size: 20}
	limited := text.NewLimitedFace(base)
	limited.AddUnicodeRange('A', 'Z')
	limited.AddUnicodeRange('a', 'z')
	const str = "abcXYZ"
	snap := mustSnapshot(t, str, limited, nil)

	dst := ebiten.NewImage(80, 40)
	defer dst.Deallocate()
	if err := snap.Draw(dst, limited, nil); err != nil {
		t.Fatal(err)
	}

	limited.AddUnicodeRange('0', '9')
	if err := snap.Draw(dst, limited, nil); err == nil {
		t.Fatal("Draw must reject a LimitedFace whose ranges changed")
	}
}

// TestSnapshotGoXFace checks that a snapshot over a GoXFace replays. GoXFace
// has no outlines, so AppendVectorPath appends nothing but must not fail.
func TestSnapshotGoXFace(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "Roboto-Regular.ttf"))
	if err != nil {
		t.Fatal(err)
	}
	sfnt, err := opentype.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	xface, err := opentype.NewFace(sfnt, &opentype.FaceOptions{Size: 24, DPI: 72})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = xface.Close()
	}()
	face := text.NewGoXFace(xface)

	const str = "abc"
	snap := mustSnapshot(t, str, face, nil)

	const w, h = 80, 40
	dst := ebiten.NewImage(w, h)
	defer dst.Deallocate()
	want := ebiten.NewImage(w, h)
	defer want.Deallocate()
	text.Draw(want, str, face, nil)
	if err := snap.Draw(dst, face, nil); err != nil {
		t.Fatal(err)
	}
	wantPix := make([]byte, 4*w*h)
	gotPix := make([]byte, 4*w*h)
	want.ReadPixels(wantPix)
	dst.ReadPixels(gotPix)
	if !bytes.Equal(gotPix, wantPix) {
		t.Error("GoXFace snapshot Draw differs from Draw")
	}

	var path vector.Path
	if err := snap.AppendVectorPath(&path, face); err != nil {
		t.Fatal(err)
	}
	if sig := pathSignature(&path); sig != "" {
		t.Errorf("GoXFace AppendVectorPath created geometry %q, want none", sig)
	}
}

func equalLines(a, b []text.TextLayoutSnapshotLine) bool {
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

func equalGlyphs(a, b []text.TextLayoutSnapshotGlyph) bool {
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

// pathSignature flattens a path into vertices and returns them as a string,
// giving a stable comparison key without access to the path's internals.
func pathSignature(p *vector.Path) string {
	vertices, _ := p.AppendVerticesAndIndicesForFilling(nil, nil)
	var b strings.Builder
	for _, v := range vertices {
		fmt.Fprintf(&b, "%v,%v;", v.DstX, v.DstY)
	}
	return b.String()
}

// TestSnapshotClusters checks that clusters group glyphs by byte range and that
// a combining sequence shapes into one cluster.
func TestSnapshotClusters(t *testing.T) {
	src := newSnapshotSource(t, snapshotTestFontData())
	face := &text.GoTextFace{Source: src, Size: 24}

	// "a" + combining acute accent is one cluster over two bytes even though
	// it may shape into multiple glyphs.
	const str = "ábc"
	snap := mustSnapshot(t, str, face, nil)
	glyphs := snap.Glyphs()
	clusters := snap.Clusters()

	// Every glyph belongs to exactly one cluster, in order.
	var next int
	for _, c := range clusters {
		if c.GlyphStart != next {
			t.Fatalf("cluster glyph range starts at %d, want %d", c.GlyphStart, next)
		}
		if c.GlyphEnd <= c.GlyphStart {
			t.Fatalf("cluster [%d, %d) is empty", c.GlyphStart, c.GlyphEnd)
		}
		if c.StartIndexInBytes < 0 || c.EndIndexInBytes > len(str) || c.EndIndexInBytes < c.StartIndexInBytes {
			t.Fatalf("cluster byte range [%d, %d) is invalid", c.StartIndexInBytes, c.EndIndexInBytes)
		}
		for i := c.GlyphStart; i < c.GlyphEnd; i++ {
			if glyphs[i].StartIndexInBytes != c.StartIndexInBytes || glyphs[i].EndIndexInBytes != c.EndIndexInBytes {
				t.Errorf("glyph %d byte range [%d, %d) does not match its cluster [%d, %d)",
					i, glyphs[i].StartIndexInBytes, glyphs[i].EndIndexInBytes, c.StartIndexInBytes, c.EndIndexInBytes)
			}
		}
		next = c.GlyphEnd
	}
	if next != len(glyphs) {
		t.Errorf("clusters cover %d of %d glyphs", next, len(glyphs))
	}

	// The combining sequence must be a single cluster.
	found := false
	for _, c := range clusters {
		if c.StartIndexInBytes == 0 && c.EndIndexInBytes == len("á") {
			found = true
		}
	}
	if !found {
		t.Errorf("the combining sequence %q did not form one cluster: %+v", "á", clusters)
	}

	// Clusters() must return a copy.
	clusters[0].StartIndexInBytes = 999
	if snap.Clusters()[0].StartIndexInBytes == 999 {
		t.Error("mutating Clusters() changed the snapshot")
	}
}

// TestSnapshotVerifyFace checks VerifyFace's success and mismatch reporting.
func TestSnapshotVerifyFace(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	snap := mustSnapshot(t, "verify", face, nil)

	if err := snap.VerifyFace(face); err != nil {
		t.Errorf("VerifyFace with the original face: %v", err)
	}

	changed := *face
	changed.Size = 24
	err := snap.VerifyFace(&changed)
	if err == nil {
		t.Fatal("VerifyFace must reject a changed face")
	}
	if path := snapshotErrorPath(err); path == "" {
		t.Errorf("VerifyFace error has no field path: %v", err)
	}
}

// TestSnapshotColorGlyph checks that a color glyph's snapshot replay matches
// Draw and preserves the Colored flag.
func TestSnapshotColorGlyph(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "chromacheck-colr.ttf"))
	if err != nil {
		t.Fatal(err)
	}
	src := newSnapshotSource(t, data)
	face := &text.GoTextFace{Source: src, Size: 32}
	const str = ""
	snap := mustSnapshot(t, str, face, nil)

	glyphs := snap.Glyphs()
	if len(glyphs) != 1 {
		t.Fatalf("got %d glyphs, want 1", len(glyphs))
	}
	if !glyphs[0].Colored {
		t.Error("the glyph is not marked colored")
	}

	const w, h = 48, 48
	want := ebiten.NewImage(w, h)
	defer want.Deallocate()
	var drawOp text.DrawOptions
	drawOp.ColorScale.ScaleWithColor(color.Black) // must not tint a color glyph
	text.Draw(want, str, face, &drawOp)

	got := ebiten.NewImage(w, h)
	defer got.Deallocate()
	var replayOp ebiten.DrawImageOptions
	replayOp.ColorScale.ScaleWithColor(color.Black)
	if err := snap.Draw(got, face, &replayOp); err != nil {
		t.Fatal(err)
	}

	b := want.Bounds()
	wantPix := make([]byte, 4*b.Dx()*b.Dy())
	gotPix := make([]byte, 4*b.Dx()*b.Dy())
	want.ReadPixels(wantPix)
	got.ReadPixels(gotPix)
	if !bytes.Equal(gotPix, wantPix) {
		t.Error("color glyph snapshot Draw differs from Draw")
	}
}

// TestSnapshotErrorMessage checks the error's formatting and unwrapping.
func TestSnapshotErrorMessage(t *testing.T) {
	base := errors.New("base")
	err := &text.TextLayoutSnapshotError{Path: "glyphs[3].gid", Message: "bad", Err: base}
	if got, want := err.Error(), `text: snapshot: field "glyphs[3].gid": bad: base`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if !errors.Is(err, base) {
		t.Error("errors.Is must see the wrapped error")
	}
	noCause := &text.TextLayoutSnapshotError{Path: "x", Message: "y"}
	if got, want := noCause.Error(), `text: snapshot: field "x": y`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if noCause.Unwrap() != nil {
		t.Error("Unwrap() = non-nil for an error without a cause")
	}
}

// TestSnapshotFontSubstitution checks the font-replacement scenario: a face
// pointed at a different font is rejected even when the new font has similar
// metrics, and every replay entry point reports it.
func TestSnapshotFontSubstitution(t *testing.T) {
	latinData := goregular.TTF
	cjkData := snapshotTestFontData()
	if bytes.Equal(latinData, cjkData) {
		t.Fatal("the fixture fonts must differ")
	}

	original := &text.GoTextFace{Source: newSnapshotSource(t, latinData), Size: 24}
	snap := mustSnapshot(t, "Substitution", original, nil)

	substituteFace := *original
	substituteFace.Source = newSnapshotSource(t, cjkData)

	dst := ebiten.NewImage(64, 64)
	defer dst.Deallocate()
	if err := snap.Draw(dst, &substituteFace, nil); err == nil {
		t.Error("Draw must reject a substituted font")
	} else if path := snapshotErrorPath(err); path != "face" {
		t.Errorf("Draw error path = %q, want %q", path, "face")
	}
	if _, err := snap.AppendGlyphs(nil, &substituteFace); err == nil {
		t.Error("AppendGlyphs must reject a substituted font")
	}
	var path vector.Path
	if err := snap.AppendVectorPath(&path, &substituteFace); err == nil {
		t.Error("AppendVectorPath must reject a substituted font")
	}
	if err := snap.VerifyFace(&substituteFace); err == nil {
		t.Error("VerifyFace must reject a substituted font")
	}

	// The original face still replays.
	if err := snap.VerifyFace(original); err != nil {
		t.Errorf("VerifyFace with the original face: %v", err)
	}
}

// TestSnapshotDecodedDoesNotAliasInput checks that a decoded snapshot does not
// share memory with the encoded bytes.
func TestSnapshotDecodedDoesNotAliasInput(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	const str = "no alias"
	snap := mustSnapshot(t, str, face, nil)
	data, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := text.DecodeTextLayoutSnapshot(data)
	if err != nil {
		t.Fatal(err)
	}
	wantText := decoded.Text()
	wantGlyph := decoded.Glyphs()[0]

	// Overwrite the encoded buffer; the decoded snapshot must be unaffected.
	for i := range data {
		data[i] = 0xff
	}
	if got := decoded.Text(); got != wantText {
		t.Errorf("mutating the encoded bytes changed the decoded text: %q", got)
	}
	if got := decoded.Glyphs()[0]; got != wantGlyph {
		t.Errorf("mutating the encoded bytes changed the decoded glyph: %+v", got)
	}
}

// TestSnapshotSurvivesCacheEviction checks that a snapshot replays after the
// face's caches have been churned by other sizes and texts.
func TestSnapshotSurvivesCacheEviction(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	const str = "evict me"
	snap := mustSnapshot(t, str, face, nil)

	// Churn the source's caches with many sizes and texts.
	for i := 1; i <= 40; i++ {
		f := &text.GoTextFace{Source: src, Size: float64(i)}
		_ = text.AppendGlyphs(nil, strings.Repeat("x", i), f, nil)
	}

	dst := ebiten.NewImage(120, 40)
	defer dst.Deallocate()
	want := ebiten.NewImage(120, 40)
	defer want.Deallocate()
	text.Draw(want, str, face, nil)
	if err := snap.Draw(dst, face, nil); err != nil {
		t.Fatalf("Draw after cache churn: %v", err)
	}
	b := dst.Bounds()
	wantPix := make([]byte, 4*b.Dx()*b.Dy())
	gotPix := make([]byte, 4*b.Dx()*b.Dy())
	want.ReadPixels(wantPix)
	dst.ReadPixels(gotPix)
	if !bytes.Equal(gotPix, wantPix) {
		t.Error("snapshot Draw changed after cache eviction")
	}
}

// TestSnapshotConcurrentReplay checks that a snapshot is safe to read from
// multiple goroutines.
func TestSnapshotConcurrentReplay(t *testing.T) {
	src := newSnapshotSource(t, goregular.TTF)
	face := &text.GoTextFace{Source: src, Size: 20}
	const str = "concurrent replay"
	snap := mustSnapshot(t, str, face, nil)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 10 {
				_ = snap.Text()
				_ = snap.Lines()
				_ = snap.Glyphs()
				_ = snap.Clusters()
				_ = snap.VerifyFace(face)
				_, _ = snap.Bounds()
			}
		}()
	}
	wg.Wait()
}
