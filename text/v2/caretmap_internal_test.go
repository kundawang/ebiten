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
	"testing"
	"unicode/utf8"

	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
)

// TestNewTextCaretMapFacePanic verifies that a panic raised by a Face
// during layout is converted into a field-path error instead of escaping.
func TestNewTextCaretMapFacePanic(t *testing.T) {
	// A GoTextFace with a nil Source panics when layout touches it.
	face := &GoTextFace{Size: 24}
	_, err := NewTextCaretMap("abc", face, nil)
	if err == nil {
		t.Fatal("expected an error from a panicking face")
	}
	var e *TextCaretMapError
	if !errors.As(err, &e) {
		t.Fatalf("want *TextCaretMapError, got %T: %v", err, err)
	}
	if e.Field != "face" {
		t.Errorf("Field = %q, want face", e.Field)
	}
	if e.Unwrap() == nil {
		t.Error("the panic cause was not wrapped")
	}
}

// TestTextCaretMapSoftWrapCoverage checks, on the internal line table, that
// soft wrapping leaves no byte owned by zero lines and no byte owned by
// more than one line, including runs of whitespace and all-space text.
func TestTextCaretMapSoftWrapCoverage(t *testing.T) {
	src, err := NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		t.Fatal(err)
	}
	face := &GoTextFace{Source: src, Size: 24}
	for _, s := range []string{"   ", " a ", "a   b", "     \n   ", "one two three four"} {
		opts := &TextCaretMapOptions{LineBreakWidth: 20}
		opts.LineSpacing = face.Metrics().HAscent + face.Metrics().HDescent
		m, err := NewTextCaretMap(s, face, opts)
		if err != nil {
			t.Fatalf("%q: %v", s, err)
		}
		if len(m.lines) < 1 {
			t.Fatalf("%q: LineCount = %d, want >= 1", s, len(m.lines))
		}
		owner := make([]int, len(s))
		for i := range owner {
			owner[i] = -1
		}
		for i, l := range m.lines {
			if l.start > l.end {
				t.Errorf("%q: line %d has start %d > end %d", s, i, l.start, l.end)
			}
			for k := l.start; k < l.end; k++ {
				if owner[k] != -1 {
					t.Errorf("%q: byte %d owned by lines %d and %d", s, k, owner[k], i)
				}
				owner[k] = i
			}
		}
		for k, o := range owner {
			if o == -1 {
				// A hard line break belongs to no line: line ranges
				// exclude the terminating break codepoint, matching how
				// forEachLine and AppendLazyGlyphs never emit a glyph for
				// it.
				r, _ := utf8.DecodeRuneInString(s[k:])
				if textutil.IsLineBreak(r) {
					continue
				}
				t.Errorf("%q: byte %d (%q) is owned by no line", s, k, s[k])
			}
		}
	}
}
