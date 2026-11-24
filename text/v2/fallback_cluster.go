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
	"sync"
	"unicode/utf8"

	"github.com/go-text/typesetting/segmenter"

	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
)

// graphemeCluster is a byte range [start, end) of one extended grapheme
// cluster (UAX #29) within a line.
type graphemeCluster struct {
	start int
	end   int
}

// graphemeSegmenter splits a line into extended grapheme clusters.
//
// The underlying segmenter keeps reusable buffers, so one instance is held
// per resolver and guarded by a mutex: a Segmenter is not safe for
// concurrent use.
type graphemeSegmenter struct {
	mu  sync.Mutex
	seg segmenter.Segmenter
}

// split appends the grapheme clusters of text to dst and returns the
// extended slice. Callers pass a single line (line breaks are trimmed
// before calling), so line-break handling is defensive only.
//
// Invalid UTF-8 does not produce an error: each invalid byte is treated as
// its own single-byte cluster, mirroring how the rest of the package renders
// invalid sequences through U+FFFD (#3284).
func (g *graphemeSegmenter) split(dst []graphemeCluster, text string) []graphemeCluster {
	g.mu.Lock()
	defer g.mu.Unlock()

	if err := g.seg.InitWithString(text); err != nil {
		// InitWithString rejects invalid UTF-8. Fall back to a rune-wise
		// split, which never fails: range turns each invalid byte into
		// U+FFFD of length one, and U+FFFD itself is a valid cluster.
		return splitRuneClusters(dst, text)
	}

	iter := g.seg.GraphemeIterator()
	for iter.Next() {
		gr := iter.Grapheme()
		if gr.OffsetInBytes >= len(text) {
			break
		}
		r, _ := utf8.DecodeRuneInString(text[gr.OffsetInBytes:])
		if textutil.IsLineBreak(r) {
			break
		}
		dst = append(dst, graphemeCluster{
			start: gr.OffsetInBytes,
			end:   gr.OffsetInBytes + gr.LengthInBytes,
		})
	}
	return dst
}

// splitRuneClusters splits text into one cluster per rune. It is the
// invalid-UTF-8 fallback of graphemeSegmenter.split.
func splitRuneClusters(dst []graphemeCluster, text string) []graphemeCluster {
	for i := range text {
		r, size := utf8.DecodeRuneInString(text[i:])
		if textutil.IsLineBreak(r) {
			break
		}
		dst = append(dst, graphemeCluster{start: i, end: i + size})
	}
	return dst
}
