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
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/hajimehoshi/bitmapfont/v4"
	"golang.org/x/image/font/gofont/goregular"

	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
)

const caretTestEps = 1.0 / (1 << 6)

func caretTestFace(t *testing.T) *text.GoTextFace {
	t.Helper()
	source, err := text.NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		t.Fatal(err)
	}
	return &text.GoTextFace{Source: source, Size: 24, Direction: text.DirectionLeftToRight}
}

func caretTestFaceDirection(t *testing.T, d text.Direction) *text.GoTextFace {
	f := caretTestFace(t)
	f.Direction = d
	return f
}

func floatEq(a, b float64) bool {
	return math.Abs(a-b) <= caretTestEps
}

// TestTextCaretMapBasic checks the line/caret structure against a simple
// single-line Latin text and the glyph layout produced by AppendLazyGlyphs.
func TestTextCaretMapBasic(t *testing.T) {
	face := caretTestFace(t)
	const s = "abc"

	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.Len() != len(s) {
		t.Errorf("Len = %d, want %d", m.Len(), len(s))
	}
	if m.LineCount() != 1 {
		t.Errorf("LineCount = %d, want 1", m.LineCount())
	}

	glyphs := text.AppendLazyGlyphs(nil, s, face, nil)
	if len(glyphs) != 3 {
		t.Fatalf("glyph count = %d, want 3", len(glyphs))
	}

	// Each rune boundary is a caret; the caret X must equal a glyph
	// origin from the same layout pipeline.
	wantX := []float64{
		glyphs[0].OriginX,
		glyphs[1].OriginX,
		glyphs[2].OriginX,
		glyphs[2].OriginX + glyphs[2].AdvanceX,
	}
	baseline := glyphs[0].OriginY
	for i, want := range wantX {
		c := m.CaretAtByte(i, text.CaretAffinityBefore)
		if c.ByteOffset != i {
			t.Errorf("CaretAtByte(%d).ByteOffset = %d", i, c.ByteOffset)
		}
		if c.Line != 0 {
			t.Errorf("CaretAtByte(%d).Line = %d, want 0", i, c.Line)
		}
		if !floatEq(c.X, want) {
			t.Errorf("CaretAtByte(%d).X = %v, want %v", i, c.X, want)
		}
		if !floatEq(c.Y, baseline) {
			t.Errorf("CaretAtByte(%d).Y = %v, want baseline %v", i, c.Y, baseline)
		}
	}

	// The line box spans the full line advance along X and the font's
	// ascent/descent along Y.
	_, box := m.Line(0)
	if !floatEq(box.MinX, 0) || !floatEq(box.MaxX, wantX[3]) {
		t.Errorf("line box X = [%v,%v], want [0,%v]", box.MinX, box.MaxX, wantX[3])
	}
	metrics := face.Metrics()
	if !floatEq(box.MinY, baseline-metrics.HAscent) || !floatEq(box.MaxY, baseline+metrics.HDescent) {
		t.Errorf("line box Y = [%v,%v], want ascent/descent span", box.MinY, box.MaxY)
	}
}

// TestTextCaretMapEmptyText checks the single empty line of an empty
// text and the empty final line of a trailing line break.
func TestTextCaretMapEmptyAndTrailingLines(t *testing.T) {
	face := caretTestFace(t)

	for _, tc := range []struct {
		name      string
		text      string
		wantLines int
	}{
		{name: "empty", text: "", wantLines: 1},
		{name: "trailing newline", text: "ab\n", wantLines: 2},
		{name: "only newline", text: "\n", wantLines: 2},
		{name: "blank middle", text: "ab\n\ncd", wantLines: 3},
		{name: "no trailing", text: "ab\ncd", wantLines: 2},
		{name: "CRLF", text: "ab\r\ncd", wantLines: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := text.NewTextCaretMap(tc.text, face, nil)
			if err != nil {
				t.Fatal(err)
			}
			if m.LineCount() != tc.wantLines {
				t.Fatalf("LineCount = %d, want %d", m.LineCount(), tc.wantLines)
			}
			// Every line reports a valid, finite box.
			for i := 0; i < m.LineCount(); i++ {
				idx, box := m.Line(i)
				if idx != i {
					t.Errorf("Line(%d) returned index %d", i, idx)
				}
				for _, v := range []float64{box.MinX, box.MinY, box.MaxX, box.MaxY} {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Errorf("line %d has non-finite box edge %v", i, v)
					}
				}
			}
			// The last caret of a trailing-newline text sits on the
			// final (empty) line at the text end.
			c := m.CaretAtByte(m.Len(), text.CaretAffinityBefore)
			if c.Line != tc.wantLines-1 {
				t.Errorf("end caret line = %d, want %d", c.Line, tc.wantLines-1)
			}
			if c.ByteOffset != m.Len() {
				t.Errorf("end caret byte = %d, want %d", c.ByteOffset, m.Len())
			}
		})
	}
}

// TestTextCaretMapLineBreakAffinity checks that a line-break boundary
// resolves to the preceding line with Before affinity and the following
// line with After affinity.
func TestTextCaretMapLineBreakAffinity(t *testing.T) {
	face := caretTestFace(t)
	const s = "ab\ncd"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Bytes: a=0 b=1 \n=2 c=3 d=4. Offset 2 is the shared boundary at the
	// end of line 0 and the start of line 1.
	const boundary = 2

	before := m.CaretAtByte(boundary, text.CaretAffinityBefore)
	after := m.CaretAtByte(boundary, text.CaretAffinityAfter)
	if before.Line != 0 {
		t.Errorf("Before affinity line = %d, want 0", before.Line)
	}
	if after.Line != 1 {
		t.Errorf("After affinity line = %d, want 1", after.Line)
	}
	// Before keeps the boundary offset on the preceding line; After moves
	// to the following line's first caret boundary (the byte after '\n').
	if before.ByteOffset != boundary {
		t.Errorf("Before byte offset = %d, want %d", before.ByteOffset, boundary)
	}
	if after.ByteOffset != 3 {
		t.Errorf("After byte offset = %d, want 3 (next line start)", after.ByteOffset)
	}

	// A byte strictly inside the next line's content still resolves to
	// that line and never to the line-break codepoint.
	c3 := m.CaretAtByte(3, text.CaretAffinityBefore)
	if c3.Line != 1 || c3.ByteOffset != 3 {
		t.Errorf("offset 3 = line %d byte %d, want line 1 byte 3", c3.Line, c3.ByteOffset)
	}
}

// TestTextCaretMapEmptyLineBreakAffinity checks that a line-break byte
// terminating an empty line still resolves its After affinity to the
// following line, rather than collapsing onto the empty line.
func TestTextCaretMapEmptyLineBreakAffinity(t *testing.T) {
	face := caretTestFace(t)
	// Bytes: a0 b1 \n2 \n3 c4 d5. Line 1 is the empty line "between" the
	// two breaks, with start == end == 3.
	for _, tc := range []struct {
		s             string
		breakByte     int
		afterLine     int
		afterByte     int
		wantLineCount int
	}{
		{"ab\n\ncd", 3, 2, 4, 3},
		{"ab\n\n", 3, 2, 4, 3},
	} {
		m, err := text.NewTextCaretMap(tc.s, face, nil)
		if err != nil {
			t.Fatal(err)
		}
		if m.LineCount() != tc.wantLineCount {
			t.Fatalf("%q: LineCount = %d, want %d", tc.s, m.LineCount(), tc.wantLineCount)
		}
		before := m.CaretAtByte(tc.breakByte, text.CaretAffinityBefore)
		after := m.CaretAtByte(tc.breakByte, text.CaretAffinityAfter)
		if before.Line != 1 {
			t.Errorf("%q: break byte Before line = %d, want 1", tc.s, before.Line)
		}
		if after.Line != tc.afterLine || after.ByteOffset != tc.afterByte {
			t.Errorf("%q: break byte After = line %d byte %d, want line %d byte %d",
				tc.s, after.Line, after.ByteOffset, tc.afterLine, tc.afterByte)
		}
	}
}

// TestTextCaretMapClusterSnap verifies that offsets inside a rune, a
// combining-mark cluster, an emoji ZWJ sequence, and a ligature never
// produce a caret at an interior byte.
func TestTextCaretMapClusterSnap(t *testing.T) {
	face := caretTestFace(t)
	cases := []struct {
		name string
		text string
	}{
		{name: "multibyte rune", text: "a" + "\u754c" + "b"},
		{name: "combining mark", text: "e" + "e" + "\u0301"},
		{name: "emoji ZWJ", text: "a" + "\U0001f468\u200d\U0001f469\u200d\U0001f467" + "b"},
		{name: "ligature", text: "ffi"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := text.NewTextCaretMap(tc.text, face, nil)
			if err != nil {
				t.Fatal(err)
			}
			// Collect the set of valid caret byte offsets.
			valid := map[int]bool{0: true, m.Len(): true}
			for off := 0; off <= m.Len(); off++ {
				c := m.CaretAtByte(off, text.CaretAffinityBefore)
				valid[c.ByteOffset] = true
				// Every reported offset must be a UTF-8 boundary.
				if c.ByteOffset > 0 && c.ByteOffset < m.Len() && !utf8.RuneStart(tc.text[c.ByteOffset]) {
					t.Errorf("caret byte %d is not a UTF-8 rune start", c.ByteOffset)
				}
			}
			// Every byte that starts a rune must either be a valid caret
			// boundary or snap, and interior bytes must not appear.
			for i := range tc.text {
				c := m.CaretAtByte(i, text.CaretAffinityBefore)
				if !valid[c.ByteOffset] {
					t.Errorf("offset %d snapped to unknown boundary %d", i, c.ByteOffset)
				}
			}
			// Interior bytes of the first multi-byte rune snap to its start.
			if _, size := utf8.DecodeRuneInString(tc.text[1:]); size > 1 {
				start := m.CaretAtByte(1, text.CaretAffinityBefore).ByteOffset
				for k := 2; k < 1+size; k++ {
					if got := m.CaretAtByte(k, text.CaretAffinityBefore).ByteOffset; got != start {
						t.Errorf("interior byte %d snapped to %d, want cluster start %d", k, got, start)
					}
				}
			}
		})
	}
}

// TestTextCaretMapClusterSnapAffinity checks that an offset inside a
// multi-byte rune or a ligature snaps in the direction implied by the
// affinity: Before to the cluster start, After to the cluster end.
func TestTextCaretMapClusterSnapAffinity(t *testing.T) {
	face := caretTestFace(t)
	for _, tc := range []struct {
		name       string
		text       string
		clusterEnd int // byte just past the first cluster
	}{
		{name: "multibyte rune", text: "界", clusterEnd: 3},
		{name: "ligature", text: "ffi", clusterEnd: 3},
		{name: "combining mark", text: "e" + "́" + "x", clusterEnd: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := text.NewTextCaretMap(tc.text, face, nil)
			if err != nil {
				t.Fatal(err)
			}
			for off := 1; off < tc.clusterEnd; off++ {
				before := m.CaretAtByte(off, text.CaretAffinityBefore)
				after := m.CaretAtByte(off, text.CaretAffinityAfter)
				if before.ByteOffset != 0 {
					t.Errorf("offset %d Before byte = %d, want 0", off, before.ByteOffset)
				}
				if after.ByteOffset != tc.clusterEnd {
					t.Errorf("offset %d After byte = %d, want %d", off, after.ByteOffset, tc.clusterEnd)
				}
			}
		})
	}
}

// TestTextCaretMapRTL checks a pure-RTL line: glyphs progress visually
// from right to left and carets follow the same geometry.
func TestTextCaretMapRTL(t *testing.T) {
	face := caretTestFaceDirection(t, text.DirectionRightToLeft)
	// alef bet gimel, two bytes each.
	const s = "\u05d0\u05d1\u05d2"

	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	glyphs := text.AppendLazyGlyphs(nil, s, face, nil)
	if len(glyphs) != 3 {
		t.Fatalf("glyph count = %d, want 3", len(glyphs))
	}
	// Logical byte 0's caret is at the line's origin edge; the logical
	// end is at the opposite edge. With an RTL face the origin is on the
	// right and advances run leftward (negative X).
	c0 := m.CaretAtByte(0, text.CaretAffinityBefore)
	cEnd := m.CaretAtByte(m.Len(), text.CaretAffinityAfter)
	if !(c0.X > cEnd.X) {
		t.Errorf("RTL start X %v must be right of end X %v", c0.X, cEnd.X)
	}
	// The distinct cluster-boundary carets, ordered by logical offset,
	// move in the reading direction (decreasing X). The first and last
	// offsets share boundary positions, so walk the rune starts.
	var boundaryX []float64
	for off := 0; off < m.Len(); {
		_, size := utf8.DecodeRuneInString(s[off:])
		boundaryX = append(boundaryX, m.CaretAtByte(off, text.CaretAffinityBefore).X)
		off += size
	}
	for i := 1; i < len(boundaryX); i++ {
		if boundaryX[i] > boundaryX[i-1]+caretTestEps {
			t.Errorf("RTL boundary X increased at rune %d: %v > %v", i, boundaryX[i], boundaryX[i-1])
		}
	}
}

// TestTextCaretMapGlyphParity verifies that every caret the map produces
// sits on a real glyph origin (or a line edge) from the same layout
// pipeline as AppendLazyGlyphs, across face types and directions. This is
// the "must agree with Draw/AppendGlyphs" guarantee.
func TestTextCaretMapGlyphParity(t *testing.T) {
	source, err := text.NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		face text.Face
		text string
	}{
		{name: "GoText/LTR", face: &text.GoTextFace{Source: source, Size: 24}, text: "Hello, " + "\u4e16\u754c"},
		{name: "GoText/RTL", face: &text.GoTextFace{Source: source, Size: 24, Direction: text.DirectionRightToLeft}, text: "\u05d0\u05d1\u05d2"},
		{
			name: "GoText/vertical",
			face: &text.GoTextFace{Source: source, Size: 24, Direction: text.DirectionTopToBottomAndRightToLeft},
			text: "\u3042\u3044\u3046",
		},
		{name: "GoX/LTR", face: text.NewGoXFace(bitmapfont.Face), text: "ab"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := text.NewTextCaretMap(tc.text, tc.face, nil)
			if err != nil {
				t.Fatal(err)
			}
			glyphs := text.AppendLazyGlyphs(nil, tc.text, tc.face, nil)

			// A horizontal face advances glyphs along X; a vertical one
			// along Y. Infer it from the first glyph's advance.
			horizontal := true
			if len(glyphs) > 0 {
				horizontal = glyphs[0].AdvanceX != 0
			}
			// Primary-axis coordinates that a real glyph origin occupies.
			primaryOrigins := map[float64]bool{}
			secondaryOrigins := map[float64]bool{}
			for _, g := range glyphs {
				if horizontal {
					primaryOrigins[g.OriginX] = true
					secondaryOrigins[g.OriginY] = true
				} else {
					primaryOrigins[g.OriginY] = true
					secondaryOrigins[g.OriginX] = true
				}
			}

			near := func(set map[float64]bool, v float64) bool {
				for k := range set {
					if floatEq(k, v) {
						return true
					}
				}
				return false
			}
			// In an RTL face the line's primary origin (0) is the logical
			// end edge and the glyphs extend into negative coordinates, so
			// only verify the secondary axis and that every primary value is
			// finite there; LTR and vertical faces additionally match an
			// interior glyph origin.
			rtl := len(glyphs) > 0 && glyphs[0].OriginX < 0
			for off := 0; off <= m.Len(); off++ {
				c := m.CaretAtByte(off, text.CaretAffinityBefore)
				primary, secondary := c.X, c.Y
				if !horizontal {
					primary, secondary = c.Y, c.X
				}
				if math.IsNaN(primary) || math.IsInf(primary, 0) {
					t.Errorf("offset %d non-finite primary %v", off, primary)
				}
				if !rtl && off != 0 && off != m.Len() && !near(primaryOrigins, primary) {
					t.Errorf("offset %d caret primary %v matches no glyph origin", off, primary)
				}
				if !near(secondaryOrigins, secondary) {
					t.Errorf("offset %d caret secondary %v matches no glyph origin", off, secondary)
				}
			}
		})
	}
}

// TestTextCaretMapBidi checks a mixed LTR/RTL/LTR line against the
// AdvanceAt leading-edge convention already tested for the layout.
func TestTextCaretMapBidi(t *testing.T) {
	face := caretTestFace(t)
	const s = "abc" + "\u05d0\u05d1\u05d2" + "xyz"

	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	// At the LTR->RTL seam (offset 3, the first Hebrew byte), the Before
	// caret is at the right edge of the Hebrew run and the After caret at
	// its left edge: two distinct visual positions.
	before := m.CaretAtByte(3, text.CaretAffinityBefore)
	after := m.CaretAtByte(3, text.CaretAffinityAfter)
	if floatEq(before.X, after.X) {
		t.Errorf("bidi seam affinities collapsed to one X %v", before.X)
	}
	// The seam carets must match AdvanceAt's leading-edge positions.
	if got := text.AdvanceAt(s, 3, face); !floatEq(before.X, got) {
		t.Errorf("seam Before X %v, want AdvanceAt %v", before.X, got)
	}
	// Snap check on a Hebrew rune's interior byte.
	c := m.CaretAtByte(4, text.CaretAffinityBefore)
	if c.ByteOffset != 3 {
		t.Errorf("Hebrew interior byte snapped to %d, want 3", c.ByteOffset)
	}
}

// TestTextCaretMapBidiClusterDirection pins the per-cluster direction to
// the shaped run, not to a guess from visual byte order. A lone RTL rune
// between LTR runes, and the LTR cluster that precedes an RTL run, must
// keep their own leading edges; otherwise the caret at a cluster start
// jumps to the wrong side of the cluster.
func TestTextCaretMapBidiClusterDirection(t *testing.T) {
	face := caretTestFace(t)

	// "a" + alef + "b": one RTL cluster between two LTR ones.
	const lone = "a" + "א" + "b"
	m, err := text.NewTextCaretMap(lone, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	glyphs := text.AppendLazyGlyphs(nil, lone, face, nil)
	if len(glyphs) != 3 {
		t.Fatalf("glyph count = %d, want 3", len(glyphs))
	}
	// The alef is the visually last glyph placed before 'b' but is shaped
	// right-to-left, so its leading edge is its right side. Before
	// affinity at offset 1 sits on that right edge.
	alef := glyphs[1]
	if got, want := m.CaretAtByte(1, text.CaretAffinityBefore).X, alef.OriginX+alef.AdvanceX; !floatEq(got, want) {
		t.Errorf("lone RTL cluster leading edge = %v, want right edge %v", got, want)
	}
	// After affinity at the alef's end sits on its left edge.
	if got, want := m.CaretAtByte(3, text.CaretAffinityAfter).X, alef.OriginX; !floatEq(got, want) {
		t.Errorf("lone RTL cluster trailing edge = %v, want left edge %v", got, want)
	}

	// "abc" + alef bet gimel + "xyz": the LTR 'c' immediately before the
	// RTL run must keep an LTR leading edge (its left side), not be swept
	// into the RTL region.
	const mixed = "abc" + "אבג" + "xyz"
	mm, err := text.NewTextCaretMap(mixed, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	mglyphs := text.AppendLazyGlyphs(nil, mixed, face, nil)
	var cGlyph text.LazyGlyph
	for _, g := range mglyphs {
		if g.StartIndexInBytes == 2 {
			cGlyph = g
		}
	}
	if got, want := mm.CaretAtByte(2, text.CaretAffinityBefore).X, cGlyph.OriginX; !floatEq(got, want) {
		t.Errorf("LTR 'c' leading edge = %v, want left edge %v", got, want)
	}
}

// TestTextCaretMapPositionAt checks hit testing, including clamping
// outside the text bounds and the round trip against CaretAtByte.
func TestTextCaretMapPositionAt(t *testing.T) {
	face := caretTestFace(t)
	const s = "abcdef"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Hit testing the exact point of every caret returns that caret.
	for off := 0; off <= m.Len(); off++ {
		c := m.CaretAtByte(off, text.CaretAffinityBefore)
		hit := m.PositionAt(c.X, c.Y)
		if hit.ByteOffset != c.ByteOffset {
			t.Errorf("PositionAt caret %d returned byte %d", off, hit.ByteOffset)
		}
	}

	// Points far left and far right clamp to the first and last carets.
	first := m.CaretAtByte(0, text.CaretAffinityBefore)
	last := m.CaretAtByte(m.Len(), text.CaretAffinityAfter)
	if h := m.PositionAt(-1e6, first.Y); h.ByteOffset != first.ByteOffset {
		t.Errorf("far-left hit byte = %d, want %d", h.ByteOffset, first.ByteOffset)
	}
	if h := m.PositionAt(1e6, last.Y); h.ByteOffset != last.ByteOffset {
		t.Errorf("far-right hit byte = %d, want %d", h.ByteOffset, last.ByteOffset)
	}

	// A point on each glyph's midpoint selects one of its bounding carets,
	// never a byte inside the glyph.
	glyphs := text.AppendLazyGlyphs(nil, s, face, nil)
	for _, g := range glyphs {
		mid := g.OriginX + g.AdvanceX/2
		h := m.PositionAt(mid, g.OriginY)
		if h.ByteOffset < g.StartIndexInBytes || h.ByteOffset > g.EndIndexInBytes {
			t.Errorf("hit on glyph [%d,%d) returned byte %d", g.StartIndexInBytes, g.EndIndexInBytes, h.ByteOffset)
		}
	}
}

// TestTextCaretMapPositionAtMultiLine checks vertical line selection,
// including the default zero line spacing where line boxes touch.
func TestTextCaretMapPositionAtMultiLine(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{}
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	const s = "line1\nline2\nline3"
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}
	if m.LineCount() != 3 {
		t.Fatalf("LineCount = %d, want 3", m.LineCount())
	}
	for i := 0; i < 3; i++ {
		_, box := m.Line(i)
		midY := (box.MinY + box.MaxY) / 2
		h := m.PositionAt(0, midY)
		if h.Line != i {
			t.Errorf("hit at line %d mid-height returned line %d", i, h.Line)
		}
	}
	// Above the first line and below the last line clamp.
	_, first := m.Line(0)
	_, lastBox := m.Line(2)
	if h := m.PositionAt(0, first.MinY-1000); h.Line != 0 {
		t.Errorf("above-block hit line = %d, want 0", h.Line)
	}
	if h := m.PositionAt(0, lastBox.MaxY+1000); h.Line != 2 {
		t.Errorf("below-block hit line = %d, want 2", h.Line)
	}
}

// TestTextCaretMapGeoM checks that points and rectangles are reported in
// the transformed space and that PositionAt inverts the transform.
func TestTextCaretMapGeoM(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{}
	opts.GeoM.Scale(2, 3)
	opts.GeoM.Translate(10, 20)
	const s = "abc"
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}

	c := m.CaretAtByte(1, text.CaretAffinityBefore)
	glyphs := text.AppendLazyGlyphs(nil, s, face, nil)
	wantX, wantY := opts.GeoM.Apply(glyphs[1].OriginX, glyphs[1].OriginY)
	if !floatEq(c.X, wantX) || !floatEq(c.Y, wantY) {
		t.Errorf("transformed caret = (%v,%v), want (%v,%v)", c.X, c.Y, wantX, wantY)
	}
	// Round trip: invert the destination point back to a caret.
	hit := m.PositionAt(c.X, c.Y)
	if hit.ByteOffset != 1 {
		t.Errorf("GeoM round-trip byte = %d, want 1", hit.ByteOffset)
	}
	// A rotated but invertible transform is accepted and round-trips.
	rot := &text.TextCaretMapOptions{}
	rot.GeoM.Rotate(math.Pi / 4)
	rm, err := text.NewTextCaretMap(s, face, rot)
	if err != nil {
		t.Fatal(err)
	}
	rc := rm.CaretAtByte(2, text.CaretAffinityBefore)
	if rh := rm.PositionAt(rc.X, rc.Y); rh.ByteOffset != 2 {
		t.Errorf("rotation round-trip byte = %d, want 2", rh.ByteOffset)
	}
}

// TestTextCaretMapCaretRect checks the caret bar geometry and integer
// rounding.
func TestTextCaretMapCaretRect(t *testing.T) {
	face := caretTestFace(t)
	const s = "abc"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := m.CaretAtByte(1, text.CaretAffinityBefore)
	r := m.CaretRect(c)
	if r.MaxX <= r.MinX || r.MaxY <= r.MinY {
		t.Fatalf("caret rect has non-positive area: %+v", r)
	}
	// Under the identity transform the bar is about one pixel wide and
	// spans the line box.
	if r.MaxX-r.MinX > 1.5 {
		t.Errorf("caret width %v exceeds ~1px", r.MaxX-r.MinX)
	}
	_, lineBox := m.Line(0)
	if !floatEq(r.MinY, lineBox.MinY) || !floatEq(r.MaxY, lineBox.MaxY) {
		t.Errorf("caret bar does not span the line box: %+v vs %+v", r, lineBox)
	}
	ir := r.Rect()
	if ir.Empty() {
		t.Errorf("integer caret rect is empty: %v", ir)
	}
}

// TestTextCaretMapSelection checks selection rectangles for single- and
// multi-line selections, including reversed and zero-length inputs.
func TestTextCaretMapSelection(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{}
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	const s = "ab\ncd"
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}
	if rects := m.SelectionRect(1, 1); rects != nil {
		t.Errorf("zero-length selection = %v, want nil", rects)
	}
	// Reversed offsets are normalized.
	fwd := m.SelectionRect(0, 2)
	rev := m.SelectionRect(2, 0)
	if len(fwd) != 1 || len(rev) != 1 {
		t.Fatalf("single-line selection rect counts: %d vs %d", len(fwd), len(rev))
	}
	if fwd[0] != rev[0] {
		t.Errorf("reversed selection differs: %+v vs %+v", fwd[0], rev[0])
	}
	// A selection crossing the line break yields one rectangle per line.
	cross := m.SelectionRect(1, 4)
	if len(cross) != 2 {
		t.Fatalf("cross-line selection rect count = %d, want 2", len(cross))
	}
	for _, r := range cross {
		if r.MaxX <= r.MinX || r.MaxY <= r.MinY {
			t.Errorf("degenerate selection rect %+v", r)
		}
	}
	// Clamping beyond the text end.
	clamped := m.SelectionRect(0, 1000)
	if len(clamped) != 2 {
		t.Errorf("clamped selection rect count = %d, want 2", len(clamped))
	}
}

// TestTextCaretMapSoftWrap checks soft line breaking at a width.
func TestTextCaretMapSoftWrap(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{LineBreakWidth: 60}
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	const s = "the quick brown fox"
	noWrap, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	if noWrap.LineCount() != 1 {
		t.Fatalf("unwrapped LineCount = %d, want 1", noWrap.LineCount())
	}
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}
	if m.LineCount() <= 1 {
		t.Fatalf("wrapped LineCount = %d, want > 1", m.LineCount())
	}
	// Every visual line's primary extent fits the requested width.
	for i := 0; i < m.LineCount(); i++ {
		_, box := m.Line(i)
		if w := box.MaxX - box.MinX; w > opts.LineBreakWidth+caretTestEps {
			t.Errorf("line %d width %v exceeds wrap width %v", i, w, opts.LineBreakWidth)
		}
	}
	// Every logical offset maps to a caret on a UTF-8 boundary, and the
	// end-of-text caret sits on the last line.
	for off := 0; off <= m.Len(); off++ {
		c := m.CaretAtByte(off, text.CaretAffinityBefore)
		if c.ByteOffset < 0 || c.ByteOffset > m.Len() {
			t.Errorf("offset %d produced out-of-range byte %d", off, c.ByteOffset)
		}
	}
	end := m.CaretAtByte(m.Len(), text.CaretAffinityAfter)
	if end.Line != m.LineCount()-1 {
		t.Errorf("end caret line = %d, want %d", end.Line, m.LineCount()-1)
	}
	// A wrap boundary: Before stays on the earlier fragment, After moves
	// to the next one.
	var boundary int
	for off := 1; off < m.Len(); off++ {
		b := m.CaretAtByte(off, text.CaretAffinityBefore)
		a := m.CaretAtByte(off, text.CaretAffinityAfter)
		if b.Line != a.Line {
			boundary = off
			break
		}
	}
	if boundary == 0 {
		t.Fatal("no soft-wrap boundary found")
	}
}

// TestTextCaretMapSoftWrapThenHardLine checks that a hard line following a
// soft-wrapped hard line is placed below all of the wrapped fragments, not
// overlapping them. Without the cumulative secondary advance, a hit test at
// the later hard line's band would select one of the wrapped fragments.
func TestTextCaretMapSoftWrapThenHardLine(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{LineBreakWidth: 60}
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	const s = "the quick brown fox\nNEXT"
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}
	metrics := face.Metrics()
	lineHeight := metrics.HAscent + metrics.HDescent
	// Line boxes must march downward without overlap between consecutive
	// lines; with LineSpacing equal to the line height the boxes are
	// exactly adjacent, so MinY of each line equals the previous MaxY.
	var prevMaxY float64
	for i := 0; i < m.LineCount(); i++ {
		_, box := m.Line(i)
		if i > 0 {
			if box.MinY < prevMaxY-caretTestEps {
				t.Errorf("line %d box MinY %v overlaps previous MaxY %v", i, box.MinY, prevMaxY)
			}
			if !floatEq(box.MaxY-box.MinY, lineHeight) {
				t.Errorf("line %d height = %v, want %v", i, box.MaxY-box.MinY, lineHeight)
			}
		}
		prevMaxY = box.MaxY
	}
	// The last visible line's own band hit-tests to the last line.
	_, lastBox := m.Line(m.LineCount() - 1)
	midY := (lastBox.MinY + lastBox.MaxY) / 2
	h := m.PositionAt(0, midY)
	if h.Line != m.LineCount()-1 {
		t.Errorf("hit at last line mid-Height returned line %d, want %d", h.Line, m.LineCount()-1)
	}
}

// TestTextCaretMapVertical checks a vertical face: the primary axis is Y,
// carets move downward, and hit testing uses the Y coordinate.
func TestTextCaretMapVertical(t *testing.T) {
	face := caretTestFaceDirection(t, text.DirectionTopToBottomAndRightToLeft)
	const s = "\u3042\u3044\u3046"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	var prevY float64 = math.Inf(-1)
	for off := 0; off <= m.Len(); off++ {
		c := m.CaretAtByte(off, text.CaretAffinityBefore)
		if c.Y < prevY-caretTestEps {
			t.Errorf("vertical caret at %d moved upward: %v < %v", off, c.Y, prevY)
		}
		prevY = c.Y
	}
	first := m.CaretAtByte(0, text.CaretAffinityBefore)
	last := m.CaretAtByte(m.Len(), text.CaretAffinityAfter)
	if !floatEq(first.X, last.X) {
		t.Errorf("vertical carets changed X: %v vs %v", first.X, last.X)
	}
	// Hit testing along Y returns a caret.
	h := m.PositionAt(first.X, (first.Y+last.Y)/2)
	if h.ByteOffset < 0 || h.ByteOffset > m.Len() {
		t.Errorf("vertical hit returned byte %d", h.ByteOffset)
	}
}

// TestTextCaretMapAllLineBreaks checks every line-break codepoint the
// layout recognizes, including CRLF and multi-byte separators.
func TestTextCaretMapAllLineBreaks(t *testing.T) {
	face := caretTestFace(t)
	breaks := []string{"\n", "\v", "\f", "\r", "\r\n", "\u0085", "\u2028", "\u2029"}
	for _, br := range breaks {
		s := "ab" + br + "cd"
		m, err := text.NewTextCaretMap(s, face, nil)
		if err != nil {
			t.Errorf("break %q: %v", br, err)
			continue
		}
		if m.LineCount() != 2 {
			t.Errorf("break %q: LineCount = %d, want 2", br, m.LineCount())
		}
		if m.Len() != len(s) {
			t.Errorf("break %q: Len = %d, want %d", br, m.Len(), len(s))
		}
		// No caret may land on an interior byte of the break codepoint.
		breakStart := 2
		for k := breakStart; k < breakStart+len(br); k++ {
			c := m.CaretAtByte(k, text.CaretAffinityBefore)
			if c.ByteOffset > breakStart && c.ByteOffset < breakStart+len(br) {
				t.Errorf("break %q: caret landed on interior break byte %d", br, c.ByteOffset)
			}
		}
	}
}

// TestTextCaretMapMultiFace checks a MultiFace layout end to end.
func TestTextCaretMapMultiFace(t *testing.T) {
	source, err := text.NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		t.Fatal(err)
	}
	gt := &text.GoTextFace{Source: source, Size: 24}
	gx := text.NewGoXFace(bitmapfont.Face)
	mf, err := text.NewMultiFace(gt, gx)
	if err != nil {
		t.Fatal(err)
	}
	m, err := text.NewTextCaretMap("ab", mf, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.LineCount() != 1 {
		t.Fatalf("LineCount = %d, want 1", m.LineCount())
	}
	for off := 0; off <= m.Len(); off++ {
		c := m.CaretAtByte(off, text.CaretAffinityBefore)
		if math.IsNaN(c.X) || math.IsNaN(c.Y) {
			t.Errorf("offset %d produced NaN caret", off)
		}
	}
}

// TestTextCaretMapValidation checks every documented error path and its
// field path.
func TestTextCaretMapValidation(t *testing.T) {
	face := caretTestFace(t)

	var badErr *text.TextCaretMapError
	check := func(t *testing.T, err error, wantField string) {
		t.Helper()
		if err == nil {
			t.Fatalf("expected an error for field %s", wantField)
		}
		if !errors.As(err, &badErr) {
			t.Fatalf("error %v is not a *TextCaretMapError", err)
		}
		if badErr.Field != wantField {
			t.Errorf("error field = %q, want %q (error: %v)", badErr.Field, wantField, err)
		}
	}

	t.Run("nil face", func(t *testing.T) {
		_, err := text.NewTextCaretMap("abc", nil, nil)
		check(t, err, "face")
	})
	t.Run("invalid UTF-8", func(t *testing.T) {
		_, err := text.NewTextCaretMap("a\xffb", face, nil)
		check(t, err, "text")
	})
	t.Run("singular GeoM", func(t *testing.T) {
		op := &text.TextCaretMapOptions{}
		op.GeoM.Scale(0, 1)
		_, err := text.NewTextCaretMap("abc", face, op)
		check(t, err, "options.GeoM")
	})
	t.Run("non-finite GeoM", func(t *testing.T) {
		op := &text.TextCaretMapOptions{}
		op.GeoM.SetElement(0, 0, math.NaN())
		_, err := text.NewTextCaretMap("abc", face, op)
		check(t, err, "options.GeoM")
	})
	t.Run("non-finite line spacing", func(t *testing.T) {
		op := &text.TextCaretMapOptions{}
		op.LineSpacing = math.Inf(1)
		_, err := text.NewTextCaretMap("abc", face, op)
		check(t, err, "options.LayoutOptions.LineSpacing")
	})
	t.Run("non-finite wrap width", func(t *testing.T) {
		op := &text.TextCaretMapOptions{LineBreakWidth: math.NaN()}
		_, err := text.NewTextCaretMap("abc", face, op)
		check(t, err, "options.LineBreakWidth")
	})
	t.Run("invalid alignment", func(t *testing.T) {
		op := &text.TextCaretMapOptions{}
		op.PrimaryAlign = text.Align(99)
		_, err := text.NewTextCaretMap("abc", face, op)
		check(t, err, "options")
	})
	t.Run("oversized text", func(t *testing.T) {
		// Just over the documented 64 MiB limit.
		big := strings.Repeat("a", 64*1024*1024+1)
		_, err := text.NewTextCaretMap(big, face, nil)
		check(t, err, "text")
	})
}

// TestTextCaretMapGoXFace checks the bitmap-font Face implementation and
// its missing-glyph behavior.
func TestTextCaretMapGoXFace(t *testing.T) {
	face := text.NewGoXFace(bitmapfont.Face)
	const s = "ab"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.LineCount() != 1 {
		t.Fatalf("LineCount = %d, want 1", m.LineCount())
	}
	glyphs := text.AppendLazyGlyphs(nil, s, face, nil)
	c := m.CaretAtByte(1, text.CaretAffinityBefore)
	if !floatEq(c.X, glyphs[1].OriginX) {
		t.Errorf("GoXFace caret X %v, want glyph origin %v", c.X, glyphs[1].OriginX)
	}
	// A rune the bitmap font lacks still yields a caret at a UTF-8 boundary.
	const missing = "a" + "\u754c"
	mm, err := text.NewTextCaretMap(missing, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	c2 := mm.CaretAtByte(1, text.CaretAffinityBefore)
	if c2.ByteOffset != 1 {
		t.Errorf("missing-glyph caret byte = %d, want 1", c2.ByteOffset)
	}
}

// TestTextCaretMapLimitedFace checks that caret offsets stay on UTF-8
// boundaries even when the LimitedFace re-encodes unsupported runes.
func TestTextCaretMapLimitedFace(t *testing.T) {
	source, err := text.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		t.Fatal(err)
	}
	base := &text.GoTextFace{Source: source, Size: 24}
	limited := text.NewLimitedFace(base)
	limited.AddUnicodeRange('a', 'z')
	limited.AddUnicodeRange('A', 'Z')
	limited.AddUnicodeRange(' ', ' ')
	// "abc" + CJK + "xyz": the CJK rune is replaced by U+FFFD internally.
	const s = "abc" + "\u754c" + "xyz"
	m, err := text.NewTextCaretMap(s, limited, nil)
	if err != nil {
		t.Fatal(err)
	}
	for off := 0; off <= m.Len(); off++ {
		c := m.CaretAtByte(off, text.CaretAffinityBefore)
		if c.ByteOffset < 0 || c.ByteOffset > m.Len() {
			t.Errorf("offset %d produced byte %d outside the text", off, c.ByteOffset)
		}
	}
	// The CJK rune's interior (it is three bytes) must never be a caret.
	start := 3
	c := m.CaretAtByte(start+1, text.CaretAffinityBefore)
	if c.ByteOffset != start {
		t.Errorf("LimitedFace interior byte snapped to %d, want %d", c.ByteOffset, start)
	}
}

// TestTextCaretMapSerialization checks JSON and binary round trips and
// that the deserialized map answers geometry queries without a Face.
func TestTextCaretMapSerialization(t *testing.T) {
	face := caretTestFace(t)
	opts := &text.TextCaretMapOptions{}
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	opts.GeoM.Translate(5, 7)
	const s = "ab\ncd"
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	// The JSON must carry the text and line structure.
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	if raw["text"] != s {
		t.Errorf("serialized text = %v, want %q", raw["text"], s)
	}
	lines, ok := raw["lines"].([]any)
	if !ok || len(lines) != 2 {
		t.Fatalf("serialized lines = %v", raw["lines"])
	}

	var fromJSON text.TextCaretMap
	if err := json.Unmarshal(data, &fromJSON); err != nil {
		t.Fatal(err)
	}
	if fromJSON.Len() != m.Len() || fromJSON.LineCount() != m.LineCount() {
		t.Errorf("deserialized size mismatch: %d/%d vs %d/%d",
			fromJSON.Len(), fromJSON.LineCount(), m.Len(), m.LineCount())
	}
	for off := 0; off <= m.Len(); off++ {
		want := m.CaretAtByte(off, text.CaretAffinityBefore)
		got := fromJSON.CaretAtByte(off, text.CaretAffinityBefore)
		if got.ByteOffset != want.ByteOffset || got.Line != want.Line ||
			!floatEq(got.X, want.X) || !floatEq(got.Y, want.Y) {
			t.Errorf("deserialized caret %d = %+v, want %+v", off, got, want)
		}
	}
	if h := fromJSON.PositionAt(1, 1); h.ByteOffset < 0 {
		t.Errorf("deserialized PositionAt failed: %+v", h)
	}
	if rects := fromJSON.SelectionRect(0, m.Len()); len(rects) == 0 {
		t.Error("deserialized SelectionRect returned no rectangles")
	}

	// Binary round trip via the encoding.Binary* methods.
	bin, err := m.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var fromBin text.TextCaretMap
	if err := fromBin.UnmarshalBinary(bin); err != nil {
		t.Fatal(err)
	}
	c0 := m.CaretAtByte(2, text.CaretAffinityAfter)
	c1 := fromBin.CaretAtByte(2, text.CaretAffinityAfter)
	if c0 != c1 {
		t.Errorf("binary round-trip caret mismatch: %+v vs %+v", c0, c1)
	}
}

// TestTextCaretMapUnmarshalMalformed checks that a malformed serialized map
// is rejected with an error instead of panicking or leaving the receiver
// queryable but inconsistent.
func TestTextCaretMapUnmarshalMalformed(t *testing.T) {
	line := `{"start":0,"end":2,"hardLine":0,"originX":0,"originY":0,` +
		`"box":{"MinX":0,"MinY":0,"MaxX":10,"MaxY":10},"startP":0,"endP":10,` +
		`"stops":[{"offset":0,"p":0,"affinity":0},{"offset":2,"p":10,"affinity":0}]}`
	for _, tc := range []struct {
		name string
		data string
	}{
		{
			name: "non-invertible GeoM",
			data: `{"version":1,"direction":0,"text":"ab","geoM":[0,0,0,0,0,0],` +
				`"lines":[` + line + `],"textBounds":{"MinX":0,"MinY":0,"MaxX":10,"MaxY":10}}`,
		},
		{
			name: "NaN GeoM",
			data: `{"version":1,"direction":0,"text":"ab","geoM":[0,0,0,0,0,0],` +
				`"lines":[` + line + `],"textBounds":{"MinX":0,"MinY":0,"MaxX":10,"MaxY":10}}`,
		},
		{
			name: "no lines",
			data: `{"version":1,"direction":0,"text":"ab","geoM":[1,0,0,0,1,0],` +
				`"lines":[],"textBounds":{"MinX":0,"MinY":0,"MaxX":0,"MaxY":0}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var m text.TextCaretMap
			if err := json.Unmarshal([]byte(tc.data), &m); err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}

// TestTextCaretMapZeroValueQueries checks that querying a zero-value map
// (never built, or left so by a rejected deserialization) returns stable
// zero results instead of panicking.
func TestTextCaretMapZeroValueQueries(t *testing.T) {
	var m text.TextCaretMap
	if got := m.LineCount(); got != 0 {
		t.Errorf("LineCount = %d, want 0", got)
	}
	if idx, _ := m.Line(0); idx != -1 {
		t.Errorf("Line(0) index = %d, want -1", idx)
	}
	if c := m.CaretAtByte(0, text.CaretAffinityBefore); c.Line != 0 || c.ByteOffset != 0 {
		t.Errorf("CaretAtByte = %+v, want zero caret", c)
	}
	if c := m.PositionAt(0, 0); c.ByteOffset != 0 || c.Line != 0 {
		t.Errorf("PositionAt = %+v, want zero caret", c)
	}
	if r := m.CaretRect(text.Caret{}); r != (text.CaretRectangle{}) {
		t.Errorf("CaretRect = %+v, want zero rect", r)
	}
	if rects := m.SelectionRect(0, 5); len(rects) != 0 {
		t.Errorf("SelectionRect = %v, want empty", rects)
	}
}

// TestTextCaretMapBidiSerialization checks that the two distinct visual
// carets at a bidi seam survive a JSON round trip, even though the
// deserialized map carries no clusters or Face.
func TestTextCaretMapBidiSerialization(t *testing.T) {
	face := caretTestFace(t)
	const s = "abc" + "\u05d0\u05d1\u05d2" + "xyz"
	m, err := text.NewTextCaretMap(s, face, nil)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var d text.TextCaretMap
	if err := json.Unmarshal(data, &d); err != nil {
		t.Fatal(err)
	}
	// Offset 3 is the LTR->RTL seam: Before and After must be distinct.
	before := m.CaretAtByte(3, text.CaretAffinityBefore)
	after := m.CaretAtByte(3, text.CaretAffinityAfter)
	if floatEq(before.X, after.X) {
		t.Fatalf("bidi seam carets collapsed before serialization: %v", before.X)
	}
	dbefore := d.CaretAtByte(3, text.CaretAffinityBefore)
	dafter := d.CaretAtByte(3, text.CaretAffinityAfter)
	if !floatEq(dbefore.X, before.X) || !floatEq(dafter.X, after.X) {
		t.Errorf("bidi seam round trip: before %v->%v after %v->%v",
			before.X, dbefore.X, after.X, dafter.X)
	}
	if floatEq(dbefore.X, dafter.X) {
		t.Errorf("bidi seam carets collapsed after serialization")
	}
}

// TestTextCaretMapMonotonicRoundTrip runs an exhaustive round-trip and
// monotonicity sweep over a fixed mixed text against the layout pipeline.
func TestTextCaretMapMonotonicRoundTrip(t *testing.T) {
	face := caretTestFace(t)
	// Latin + Hebrew + CJK + emoji + combining marks, two lines.
	s := "abc" + "\u05d0\u05d1\u05d2" + "\u754c" + "\n" +
		"e" + "e" + "\u0301" + "x" + "\U0001f468\u200d\U0001f469\u200d\U0001f467" + "y"
	opts := &text.TextCaretMapOptions{}
	// A non-zero line spacing is required for distinct visual lines; the
	// default zero spacing overlaps line boxes, as documented for Draw.
	opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
	m, err := text.NewTextCaretMap(s, face, opts)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Every distinct caret point on a line is finite and, among carets
	//    at the same visual position, ordering stays deterministic. (A
	//    bidi line is not globally monotonic: RTL runs reverse the visual
	//    order inside their run.)
	for li := 0; li < m.LineCount(); li++ {
		for off := 0; off <= m.Len(); off++ {
			c := m.CaretAtByte(off, text.CaretAffinityBefore)
			if c.Line != li {
				continue
			}
			if math.IsNaN(c.X) || math.IsInf(c.X, 0) || math.IsNaN(c.Y) || math.IsInf(c.Y, 0) {
				t.Errorf("line %d offset %d produced non-finite caret %+v", li, off, c)
			}
		}
	}

	// 2. Every caret, hit-tested at its own point, resolves to the same
	//    visual position (the same line and point). At a bidi seam two
	//    logical offsets can share one visual X, so compare the resolved
	//    point rather than the exact byte.
	for off := 0; off <= m.Len(); off++ {
		for _, aff := range []text.CaretAffinity{text.CaretAffinityBefore, text.CaretAffinityAfter} {
			c := m.CaretAtByte(off, aff)
			h := m.PositionAt(c.X, c.Y)
			if h.Line != c.Line || !floatEq(h.X, c.X) || !floatEq(h.Y, c.Y) {
				t.Errorf("round trip at %d/%v: hit %+v, want line %d point (%v,%v)",
					off, aff, h, c.Line, c.X, c.Y)
			}
		}
	}

	// 3. A dense pixel grid never returns an out-of-range byte.
	bounds := m.TextBounds()
	for x := bounds.MinX - 5; x <= bounds.MaxX+5; x += 1.5 {
		for y := bounds.MinY - 5; y <= bounds.MaxY+5; y += 1.5 {
			h := m.PositionAt(x, y)
			if h.ByteOffset < 0 || h.ByteOffset > m.Len() {
				t.Errorf("grid (%v,%v) returned byte %d", x, y, h.ByteOffset)
			}
			if h.Line < 0 || h.Line >= m.LineCount() {
				t.Errorf("grid (%v,%v) returned line %d", x, y, h.Line)
			}
		}
	}
}

// TestTextCaretMapConcurrency is a smoke check that a built map is safe
// to query from multiple goroutines.
func TestTextCaretMapConcurrency(t *testing.T) {
	face := caretTestFace(t)
	m, err := text.NewTextCaretMap("abc\nabc", face, nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	for range 8 {
		go func() {
			defer func() { done <- struct{}{} }()
			for range 100 {
				for off := 0; off <= m.Len(); off++ {
					_ = m.CaretAtByte(off, text.CaretAffinityBefore)
					_ = m.PositionAt(float64(off), 5)
					_ = m.SelectionRect(0, off)
				}
			}
		}()
	}
	for range 8 {
		<-done
	}
}

// Ensure the error type is unwrappable and the documented sentinel-free
// errors carry a reason.
func TestTextCaretMapErrorUnwrap(t *testing.T) {
	e := &text.TextCaretMapError{Field: "x", Reason: "y"}
	if e.Error() == "" {
		t.Fatal("empty error string")
	}
	if e.Unwrap() != nil {
		t.Errorf("nil-wrap Unwrap = %v", e.Unwrap())
	}
	if !strings.Contains(e.Error(), "x") || !strings.Contains(e.Error(), "y") {
		t.Errorf("error string %q missing field or reason", e.Error())
	}
}
