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
	"encoding/json"
	"fmt"
	"image"
	"math"
	"slices"
	"unicode/utf8"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
)

// maxTextCaretMapTextLength is the maximum text length, in bytes, that
// [NewTextCaretMap] accepts. It bounds the work and memory of a single
// map so a pathological input cannot exhaust a process.
const maxTextCaretMapTextLength = 64 * 1024 * 1024

// CaretAffinity disambiguates a logical position that has more than one
// visually valid caret.
//
// Most positions — inside a uniform run — have a single visual caret.
// Affinity only matters at two kinds of boundary:
//
//   - A bidirectional level boundary, where two runs of opposite
//     directions meet. The two affinities point at the two visually
//     distinct edges on the two sides of the seam.
//   - A line break. The end of one line and the start of the next line
//     share a byte offset but live on different lines.
type CaretAffinity int

const (
	// CaretAffinityBefore is the caret on the leading edge of the
	// cluster that starts at the queried byte offset. In a
	// left-to-right run this is the left edge; in a right-to-left run,
	// the right edge; in a top-to-bottom run, the top edge.
	CaretAffinityBefore CaretAffinity = iota

	// CaretAffinityAfter is the caret on the trailing edge of the
	// cluster that ends at the queried byte offset. In a
	// left-to-right run this is the right edge; in a right-to-left run,
	// the left edge; in a top-to-bottom run, the bottom edge.
	CaretAffinityAfter
)

// Caret is a resolved caret: a logical text position plus the visual
// geometry that [TextCaretMap] computed for it.
type Caret struct {
	// ByteOffset is the caret's position as a byte offset into the text
	// passed to [NewTextCaretMap]. It always falls on a UTF-8 boundary:
	// it is never an interior byte of a rune or a glyph cluster.
	ByteOffset int

	// Affinity is the affinity used to resolve this caret at a
	// bidirectional or line boundary.
	Affinity CaretAffinity

	// Line is the zero-based index of the visual line that contains the
	// caret, in the order the lines are rendered (top to bottom for
	// horizontal text).
	Line int

	// X and Y are the caret point in destination space, i.e. the layout
	// coordinate after applying the TextCaretMapOptions.GeoM transform.
	// For horizontal faces the caret stands on the baseline at (X, Y);
	// for vertical faces it runs through (X, Y) along the primary axis.
	X, Y float64
}

// CaretRectangle is an axis-aligned rectangle in destination space, with
// floating-point edges so sub-pixel layouts are not rounded away.
type CaretRectangle struct {
	MinX, MinY, MaxX, MaxY float64
}

// Rect returns the rectangle rounded outward to an integer
// [image.Rectangle], suitable for clipping or drawing.
func (r CaretRectangle) Rect() image.Rectangle {
	return image.Rect(
		int(math.Floor(r.MinX)),
		int(math.Floor(r.MinY)),
		int(math.Ceil(r.MaxX)),
		int(math.Ceil(r.MaxY)),
	)
}

// TextCaretMapError reports an invalid argument to [NewTextCaretMap].
//
// Field names the rejected option, using a dotted path for nested
// fields, for example "options.GeoM" or "text".
type TextCaretMapError struct {
	// Field is the option path that failed validation, or "face" when
	// the face itself is invalid.
	Field string

	// Reason describes why the field was rejected.
	Reason string

	// Err wraps an underlying failure, such as a panic raised by a face.
	// It may be nil.
	Err error
}

// Error implements error.
func (e *TextCaretMapError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("text: invalid %s: %s: %v", e.Field, e.Reason, e.Err)
	}
	return fmt.Sprintf("text: invalid %s: %s", e.Field, e.Reason)
}

// Unwrap returns the wrapped error, if any.
func (e *TextCaretMapError) Unwrap() error {
	return e.Err
}

func caretMapError(field, reason string, err error) *TextCaretMapError {
	return &TextCaretMapError{Field: field, Reason: reason, Err: err}
}

// TextCaretMapOptions are options for [NewTextCaretMap].
//
// The embedded [LayoutOptions] are interpreted exactly as by [Draw] and
// [AppendGlyphs], so a map built with the same options places its carets
// on the glyphs those functions produce.
type TextCaretMapOptions struct {
	LayoutOptions

	// GeoM is the geometry transform applied to the layout, matching the
	// GeoM of [DrawOptions]. Every point and rectangle reported by the
	// map is in GeoM's destination space.
	//
	// The zero value is the identity transform. GeoM must contain only
	// finite values and must be invertible.
	GeoM ebiten.GeoM

	// LineBreakWidth, when positive, enables soft wrapping: a visual
	// line breaks once adding another cluster would push its extent past
	// this many pixels along the primary axis, measured in the
	// untransformed layout space. Breaks fall only on cluster
	// boundaries, and the whitespace at a break stays on the preceding
	// line. A run with no break opportunity (a single word, or a run
	// longer than the width) is split at the last cluster boundary that
	// fits, so no line exceeds the width unless one cluster alone does;
	// such a cluster occupies a line on its own rather than being split.
	//
	// Zero or negative values disable soft wrapping, so only explicit
	// line-break codepoints start a new line, as in [Draw].
	LineBreakWidth float64
}

// TextCaretMap is a bidirectional map between logical text positions
// (byte offsets on UTF-8 boundaries) and visual caret geometry.
//
// A map is built once from a Face, a text, and [TextCaretMapOptions],
// and is safe for concurrent read afterwards. The map captures the face
// and options at construction time; mutating them later does not affect
// the map.
//
// For hard line breaks, the glyph positions used to build the map are
// produced by the same layout pipeline as [Draw] and [AppendLazyGlyphs].
// Soft wrapping (TextCaretMapOptions.LineBreakWidth) lays out each
// wrapped sub-line independently, as [Draw] does not wrap text itself.
type TextCaretMap struct {
	text      string
	face      Face // nil after deserialization
	direction Direction
	opts      TextCaretMapOptions
	geoM      ebiten.GeoM
	invM      ebiten.GeoM
	lines     []caretLine

	// textBounds is the tight rectangle of every line box in
	// destination (GeoM) space.
	textBounds CaretRectangle
}

// hardLine is one line as forEachLine emits it, before soft wrapping.
type hardLine struct {
	// content is the line text excluding the terminating line break.
	content string
	// start is the absolute byte offset of the line's first byte.
	start   int
	originX float64
	originY float64
	glyphs  []LazyGlyph
}

// caretLine is one visual line: either a hard line between line-break
// codepoints or a soft-wrapped fragment of one.
type caretLine struct {
	// start and end are byte offsets into the source text. The range
	// excludes the line-break codepoint that terminates a hard line;
	// whitespace at a soft-wrap break stays in the preceding line's
	// range, so the ranges of consecutive visual lines tile the text
	// except for hard line-break codepoints.
	start int
	end   int

	// hardLine is the zero-based index of the hard line this visual
	// line belongs to.
	hardLine int

	// origin is the line's origin in layout space, matching the origin
	// passed to the face's appendLazyGlyphsForLine.
	originX float64
	originY float64

	// box is the line box in layout space.
	boxMinX float64
	boxMinY float64
	boxMaxX float64
	boxMaxY float64

	// startP and endP are the primary-axis coordinates of the logical
	// line start and end. They are retained explicitly so a deserialized
	// map, which carries no clusters, can still place boundary carets.
	startP float64
	endP   float64

	// clusters are the glyph clusters in visual order.
	clusters []caretCluster

	// stops is every distinct caret position on the line, ordered along
	// the primary axis. Ties at a bidi seam keep both entries, ordered
	// by affinity.
	stops []caretStop
}

// caretCluster is one logical glyph cluster: one or more shaped glyphs
// covering one byte range [start,end), positioned as one visual unit.
type caretCluster struct {
	start int
	end   int

	// e0 and e1 are the cluster's leading and trailing edges along the
	// primary axis, in layout space, with e0 <= e1.
	e0 float64
	e1 float64

	// rtl reports whether the cluster belongs to a right-to-left run.
	// It is always false for vertical faces.
	rtl bool
}

// caretStop is one visual caret position on a line.
type caretStop struct {
	// offset is the logical byte offset of this stop.
	offset int

	// p is the primary-axis coordinate in layout space.
	p float64

	// affinity is the affinity represented by this stop.
	affinity CaretAffinity
}

// NewTextCaretMap builds a [TextCaretMap] for text laid out with face
// and options.
//
// NewTextCaretMap returns a [*TextCaretMapError] for these failures:
//
//   - face is nil or fails while shaping the text (field "face"),
//   - options.GeoM contains a non-finite value or is not invertible
//     (field "options.GeoM"),
//   - options.LineSpacing is non-finite (field
//     "options.LayoutOptions.LineSpacing"),
//   - options.LineBreakWidth is non-finite (field
//     "options.LineBreakWidth"),
//   - an alignment or the face's direction is invalid (field "options"),
//   - text is longer than 64 MiB (field "text"),
//   - text contains invalid UTF-8 (field "text").
//
// A nil options is equivalent to the zero value.
func NewTextCaretMap(text string, face Face, options *TextCaretMapOptions) (m *TextCaretMap, err error) {
	if face == nil {
		return nil, caretMapError("face", "must not be nil", nil)
	}
	var opts TextCaretMapOptions
	if options != nil {
		opts = *options
	}

	if len(text) > maxTextCaretMapTextLength {
		return nil, caretMapError("text",
			fmt.Sprintf("length %d exceeds the maximum of %d bytes", len(text), maxTextCaretMapTextLength), nil)
	}
	if !utf8.ValidString(text) {
		return nil, caretMapError("text", "contains invalid UTF-8; every caret must fall on a UTF-8 boundary", nil)
	}
	if !isFinite(opts.LineSpacing) {
		return nil, caretMapError("options.LayoutOptions.LineSpacing", "must be finite", nil)
	}
	if !isFinite(opts.LineBreakWidth) {
		return nil, caretMapError("options.LineBreakWidth", "must be finite", nil)
	}
	if !geoMIsFinite(opts.GeoM) {
		return nil, caretMapError("options.GeoM", "contains a non-finite element", nil)
	}
	if !opts.GeoM.IsInvertible() {
		return nil, caretMapError("options.GeoM", "must be invertible so screen positions can be mapped back to layout space", nil)
	}
	// Validate the direction and alignments the same way forEachLine
	// does. calcAligns panics on an invalid direction or alignment.
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = caretMapError("options", "invalid direction or alignment", fmt.Errorf("%v", r))
			}
		}()
		_, _ = calcAligns(face.direction(), opts.PrimaryAlign, opts.SecondaryAlign)
	}()
	if err != nil {
		return nil, err
	}

	m = &TextCaretMap{
		text:      text,
		face:      face,
		direction: face.direction(),
		opts:      opts,
		geoM:      opts.GeoM,
	}
	m.invM = opts.GeoM
	m.invM.Invert()

	// A face may panic on an unusable source; surface that as a field
	// path error instead of crashing the caller.
	defer func() {
		if r := recover(); r != nil {
			m = nil
			err = caretMapError("face", "failed while laying out the text", fmt.Errorf("%v", r))
		}
	}()

	m.build()
	return m, nil
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func geoMIsFinite(g ebiten.GeoM) bool {
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if !isFinite(g.Element(i, j)) {
				return false
			}
		}
	}
	return true
}

// build lays out every line and constructs the per-line cluster and stop
// tables.
func (m *TextCaretMap) build() {
	horizontal := m.direction.isHorizontal()
	metrics := m.face.Metrics()
	layoutOpts := m.opts.LayoutOptions

	// First pass: shape each hard line through the same entry points the
	// rendering pipeline uses. forEachLine owns the origin and alignment
	// math, so hard-line glyph geometry is identical to Draw's.
	var hard []hardLine
	if m.text != "" {
		forEachLine(m.text, m.face, &layoutOpts, func(content string, indexOffset int, originX, originY float64) {
			// indexOffset is the absolute byte offset of this line's
			// content start.
			glyphs := m.face.appendLazyGlyphsForLine(nil, content, indexOffset, originX, originY, nil)
			hard = append(hard, hardLine{
				content: content,
				start:   indexOffset,
				originX: originX,
				originY: originY,
				glyphs:  glyphs,
			})
		})
	}

	// A text ending with a line break renders one additional empty line.
	// forEachLine does not emit it; synthesize it at the origin the next
	// iteration would have used. An empty text gets one empty line at the
	// base origin.
	if len(hard) == 0 {
		ox, oy := m.baseEmptyOrigin(metrics, layoutOpts)
		hard = append(hard, hardLine{start: len(m.text), originX: ox, originY: oy})
	} else if endsWithLineBreak(m.text) {
		last := &hard[len(hard)-1]
		ox, oy := nextSecondaryOrigin(m.direction, last.originX, last.originY, layoutOpts.LineSpacing)
		hard = append(hard, hardLine{start: len(m.text), originX: ox, originY: oy})
	}

	// Lay out the hard lines in order, assigning each visual line the next
	// slot along the secondary axis. forEachLine spaces the *hard* lines
	// one LineSpacing apart, but a soft-wrapped hard line expands into
	// several visual lines, so each hard line must be re-based onto the
	// running visual-line slot; otherwise lines of a later hard line would
	// overlap the wrapped fragments of an earlier one.
	var secondaryX, secondaryY float64
	for hi := range hard {
		hl := &hard[hi]
		if hi == 0 {
			secondaryX, secondaryY = hl.originX, hl.originY
		}
		baseX, baseY := hl.originX, hl.originY
		if horizontal {
			baseY = secondaryY
		} else {
			baseX = secondaryX
		}

		clusters := buildCaretClusters(hl.glyphs, horizontal)
		resolveClusterDirections(clusters, horizontal)

		var produced []caretLine
		if m.opts.LineBreakWidth > 0 && len(clusters) > 0 {
			produced = m.wrapHardLine(hl, clusters, metrics, layoutOpts, horizontal, hi, baseX, baseY)
		}
		if len(produced) == 0 {
			// No soft wrapping, or a hard line whose fragments were all
			// wrap-whitespace: keep one line so the bytes have a home.
			produced = []caretLine{m.newCaretLine(
				hl.start, hl.start+len(hl.content), hi,
				baseX, baseY, clusters, metrics, horizontal)}
		}
		m.lines = append(m.lines, produced...)

		for range produced {
			secondaryX, secondaryY = nextSecondaryOrigin(m.direction, secondaryX, secondaryY, layoutOpts.LineSpacing)
		}
	}

	m.textBounds = m.computeTextBounds()
}

// baseEmptyOrigin returns the origin of a single empty line for an empty
// text, reproducing forEachLine's base origin before any line spacing.
func (m *TextCaretMap) baseEmptyOrigin(metrics Metrics, opts LayoutOptions) (ox, oy float64) {
	h, v := calcAligns(m.direction, opts.PrimaryAlign, opts.SecondaryAlign)
	switch m.direction {
	case DirectionLeftToRight, DirectionRightToLeft:
		oy = metrics.HAscent
		boundaryHeight := metrics.HAscent + metrics.HDescent
		switch v {
		case verticalAlignCenter:
			oy -= boundaryHeight / 2
		case verticalAlignBottom:
			oy -= boundaryHeight
		}
		// An empty line has zero advance; every primary alignment places
		// its origin at 0 along the primary axis.
		return 0, oy
	case DirectionTopToBottomAndLeftToRight:
		ox = metrics.VDescent
		boundaryWidth := metrics.VAscent + metrics.VDescent
		switch h {
		case horizontalAlignCenter:
			ox -= boundaryWidth / 2
		case horizontalAlignRight:
			ox -= boundaryWidth
		}
		return ox, 0
	case DirectionTopToBottomAndRightToLeft:
		ox = -metrics.VAscent
		boundaryWidth := metrics.VAscent + metrics.VDescent
		switch h {
		case horizontalAlignLeft:
			ox += boundaryWidth
		case horizontalAlignCenter:
			ox += boundaryWidth / 2
		}
		return ox, 0
	}
	return 0, 0
}

// wrapHardLine partitions one hard line's clusters into soft-wrapped
// visual lines and lays each fragment out independently.
//
// The hard line was shaped as a whole, so every cluster already carries
// full-line pen coordinates. Each fragment is re-based onto its own line
// frame: the minimum primary edge of the fragment becomes the primary
// origin implied by the alignment, and every cluster edge is shifted by
// the same delta. baseX/baseY is the layout-space origin for the first
// fragment, already placed on the running visual-line slot by the caller;
// subsequent fragments advance by LineSpacing from there.
func (m *TextCaretMap) wrapHardLine(
	hl *hardLine,
	clusters []caretCluster,
	metrics Metrics,
	layoutOpts LayoutOptions,
	horizontal bool,
	hardLineIndex int,
	baseX, baseY float64,
) []caretLine {
	width := m.opts.LineBreakWidth

	// Logical order for wrapping decisions.
	logical := append([]caretCluster(nil), clusters...)
	slices.SortFunc(logical, func(a, b caretCluster) int { return a.start - b.start })

	// Greedy partition. A fragment's extent is max(e1)-min(e0) over its
	// clusters, which stays correct when bidi reverses visual order. A
	// whitespace cluster offers a break opportunity after it, and a break
	// taken there leaves the whitespace on the earlier fragment.
	type fragment struct {
		clusters []caretCluster
	}
	var fragments []fragment
	begin := 0
	for begin < len(logical) {
		end := begin
		minE0, maxE1 := logical[begin].e0, logical[begin].e1
		lastBreakable := -1
		for end < len(logical) {
			c := logical[end]
			ne0, ne1 := min(minE0, c.e0), max(maxE1, c.e1)
			if end > begin && ne1-ne0 > width {
				break
			}
			minE0, maxE1 = ne0, ne1
			end++
			if m.isWrapOpportunity(logical, end) {
				lastBreakable = end
			}
		}
		if end == len(logical) {
			fragments = append(fragments, fragment{clusters: logical[begin:]})
			break
		}
		cut := end
		if lastBreakable > begin {
			cut = lastBreakable
		}
		if cut == begin {
			// A cluster wider than the width takes a line alone.
			cut = begin + 1
		}
		fragments = append(fragments, fragment{clusters: logical[begin:cut]})
		begin = cut
	}

	h, v := calcAligns(m.direction, layoutOpts.PrimaryAlign, layoutOpts.SecondaryAlign)
	lines := make([]caretLine, 0, len(fragments))
	secondaryX, secondaryY := baseX, baseY
	for i, frag := range fragments {
		if i > 0 {
			secondaryX, secondaryY = nextSecondaryOrigin(m.direction, secondaryX, secondaryY, layoutOpts.LineSpacing)
		}

		// The fragment's extent and the origin it would have as an
		// independently aligned line.
		fragMin, fragMax := frag.clusters[0].e0, frag.clusters[0].e1
		for _, c := range frag.clusters[1:] {
			fragMin = min(fragMin, c.e0)
			fragMax = max(fragMax, c.e1)
		}
		fragWidth := fragMax - fragMin

		var primaryOrigin float64
		switch m.direction {
		case DirectionLeftToRight, DirectionRightToLeft:
			switch h {
			case horizontalAlignLeft:
				primaryOrigin = 0
			case horizontalAlignCenter:
				primaryOrigin = -fragWidth / 2
			case horizontalAlignRight:
				primaryOrigin = -fragWidth
			}
		default:
			switch v {
			case verticalAlignTop:
				primaryOrigin = 0
			case verticalAlignCenter:
				primaryOrigin = -fragWidth / 2
			case verticalAlignBottom:
				primaryOrigin = -fragWidth
			}
		}

		// Re-base every cluster from the full-line pen frame onto the
		// fragment's own frame. A cluster edge at e becomes
		// primaryOrigin + (e - fragMin), so the fragment's leftmost
		// cluster starts at the aligned primary origin and the internal
		// spacing is unchanged.
		reBased := make([]caretCluster, len(frag.clusters))
		for i, c := range frag.clusters {
			c.e0 = primaryOrigin + (c.e0 - fragMin)
			c.e1 = primaryOrigin + (c.e1 - fragMin)
			reBased[i] = c
		}

		// The fragment's byte range stays contiguous with its neighbours:
		// a break taken at whitespace leaves that whitespace at the end of
		// the earlier fragment, so every byte is owned by exactly one
		// visual line and offsets never fall into a gap.
		start := reBased[0].start
		end := reBased[len(reBased)-1].end

		// Primary coordinate uses primaryOrigin; the secondary coordinate
		// keeps forEachLine's secondary offset held in secondaryX/Y.
		var originX, originY float64
		if horizontal {
			originX, originY = primaryOrigin, secondaryY
		} else {
			originX, originY = secondaryX, primaryOrigin
		}
		lines = append(lines, m.newCaretLine(start, end, hardLineIndex, originX, originY, reBased, metrics, horizontal))
	}
	return lines
}

// isWrapOpportunity reports whether logical[:end] ends at a position
// where a soft line break is allowed, namely after a whitespace cluster.
func (m *TextCaretMap) isWrapOpportunity(logical []caretCluster, end int) bool {
	if end <= 0 || end > len(logical) {
		return false
	}
	last := logical[end-1]
	return isAllSpace(m.text[last.start:last.end])
}

// newCaretLine assembles a caretLine from clusters already positioned in
// layout space.
func (m *TextCaretMap) newCaretLine(start, end, hardLine int, originX, originY float64, clusters []caretCluster, metrics Metrics, horizontal bool) caretLine {
	line := caretLine{
		start:    start,
		end:      end,
		hardLine: hardLine,
		originX:  originX,
		originY:  originY,
		clusters: clusters,
	}

	// Secondary-axis box from the line origin and face metrics, matching
	// forEachLine's rendering region.
	if horizontal {
		line.boxMinY = originY - metrics.HAscent
		line.boxMaxY = originY + metrics.HDescent
	} else {
		line.boxMinX = originX - metrics.VDescent
		line.boxMaxX = originX + metrics.VAscent
	}

	// Primary-axis span from the clusters (zero width for an empty line).
	if len(clusters) == 0 {
		emptyP := originX
		if !horizontal {
			emptyP = originY
		}
		line.startP, line.endP = emptyP, emptyP
		if horizontal {
			line.boxMinX, line.boxMaxX = originX, originX
		} else {
			line.boxMinY, line.boxMaxY = originY, originY
		}
	} else {
		lo, hi := clusters[0].e0, clusters[0].e1
		for _, c := range clusters[1:] {
			lo = min(lo, c.e0)
			hi = max(hi, c.e1)
		}
		if horizontal {
			line.boxMinX, line.boxMaxX = lo, hi
		} else {
			line.boxMinY, line.boxMaxY = lo, hi
		}
		line.startP, line.endP = m.boundaryPrimary(&line)
	}

	line.stops = m.buildStops(&line, horizontal)
	return line
}

// boundaryPrimary returns the primary-axis coordinates of the line's
// logical start and end edges, derived only from the clusters so the
// result is identical before and after deserialization (which carries no
// Face). The start edge is the leading edge of the cluster that begins
// the line content, and the end edge is the trailing edge of the cluster
// that ends it. When a soft-wrapped trim leaves no cluster exactly at a
// boundary, the line's primary extent (lo, hi) is used for that edge.
func (m *TextCaretMap) boundaryPrimary(line *caretLine) (startP, endP float64) {
	lo, hi := line.clusters[0].e0, line.clusters[0].e1
	for _, c := range line.clusters[1:] {
		lo = min(lo, c.e0)
		hi = max(hi, c.e1)
	}
	startP, endP = lo, hi
	for i := range line.clusters {
		c := &line.clusters[i]
		if c.start == line.start {
			if c.rtl {
				startP = c.e1
			} else {
				startP = c.e0
			}
		}
		if c.end == line.end {
			if c.rtl {
				endP = c.e0
			} else {
				endP = c.e1
			}
		}
	}
	return startP, endP
}

// buildStops creates the ordered caret stops for a line.
func (m *TextCaretMap) buildStops(line *caretLine, horizontal bool) []caretStop {
	type key struct {
		offset   int
		affinity CaretAffinity
	}
	seen := map[key]bool{}
	var stops []caretStop
	add := func(offset int, p float64, affinity CaretAffinity) {
		k := key{offset, affinity}
		if seen[k] {
			return
		}
		seen[k] = true
		stops = append(stops, caretStop{offset: offset, p: p, affinity: affinity})
	}

	for _, c := range line.clusters {
		if c.rtl {
			// RTL: the logical start is at the visual right edge (e1),
			// the logical end at the visual left edge (e0).
			add(c.start, c.e1, CaretAffinityBefore)
			add(c.end, c.e0, CaretAffinityAfter)
		} else {
			add(c.start, c.e0, CaretAffinityBefore)
			add(c.end, c.e1, CaretAffinityAfter)
		}
	}

	if len(stops) == 0 {
		// Empty line: one position at the origin on the primary axis.
		add(line.start, line.startP, CaretAffinityBefore)
		add(line.start, line.startP, CaretAffinityAfter)
	} else {
		// Every line boundary needs both affinities at its geometric
		// edge. Cluster stops supply Before at each cluster start and
		// After at each cluster end, but the line start lacks an After
		// stop and the line end lacks a Before stop; add them at the
		// boundary's own position. This keeps CaretAtByte deterministic
		// at offset 0 and at the line end instead of falling back onto
		// an unrelated interior stop.
		add(line.start, line.startP, CaretAffinityAfter)
		add(line.end, line.endP, CaretAffinityBefore)
	}

	// Order along the primary axis. At equal positions, Before precedes
	// After, then smaller byte offset wins, giving a deterministic tie
	// break for hit testing.
	slices.SortFunc(stops, func(a, b caretStop) int {
		if a.p != b.p {
			return cmpFloat(a.p, b.p)
		}
		if a.affinity != b.affinity {
			return int(a.affinity) - int(b.affinity)
		}
		return a.offset - b.offset
	})
	return stops
}

func cmpFloat(a, b float64) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

// buildCaretClusters groups shaped glyphs, already in visual order, into
// logical clusters. Glyphs sharing or overlapping a byte range belong to
// one cluster (multi-glyph ligatures, ZWJ emoji sequences, marks with no
// independent advance). Each cluster's edges span the full pen travel of
// its glyphs along the primary axis, and its direction comes from the
// shaped run the glyphs belong to.
func buildCaretClusters(glyphs []LazyGlyph, horizontal bool) []caretCluster {
	var clusters []caretCluster
	for _, g := range glyphs {
		e0, e1 := glyphPrimaryEdges(g, horizontal)
		if len(clusters) > 0 {
			last := &clusters[len(clusters)-1]
			if g.StartIndexInBytes < last.end && g.EndIndexInBytes > last.start {
				last.end = max(last.end, g.EndIndexInBytes)
				last.start = min(last.start, g.StartIndexInBytes)
				last.e0 = min(last.e0, e0)
				last.e1 = max(last.e1, e1)
				continue
			}
		}
		clusters = append(clusters, caretCluster{
			start: g.StartIndexInBytes,
			end:   g.EndIndexInBytes,
			e0:    e0,
			e1:    e1,
			rtl:   g.rtl,
		})
	}
	return clusters
}

// glyphPrimaryEdges returns the leading/trailing primary-axis edges of a
// single glyph's pen segment.
func glyphPrimaryEdges(g LazyGlyph, horizontal bool) (e0, e1 float64) {
	if horizontal {
		e0, e1 = g.OriginX, g.OriginX+g.AdvanceX
	} else {
		e0, e1 = g.OriginY, g.OriginY+g.AdvanceY
	}
	if e1 < e0 {
		e0, e1 = e1, e0
	}
	return e0, e1
}

// resolveClusterDirections marks whether each cluster belongs to a
// right-to-left run.
//
// The direction is taken from the shaped runs themselves: the faces'
// appendLazyGlyphsForLine implementations record each glyph's run
// direction on [LazyGlyph.rtl], which buildCaretClusters forwards to its
// cluster. That is the same shaping result [AdvanceAt] derives positions
// from, so caret direction and advance direction cannot disagree.
//
// Vertical faces never have RTL progression, so their clusters stay LTR.
func resolveClusterDirections(clusters []caretCluster, horizontal bool) {
	if !horizontal {
		for i := range clusters {
			clusters[i].rtl = false
		}
	}
}

// nextSecondaryOrigin advances a line origin by one line spacing in the
// face's secondary direction.
func nextSecondaryOrigin(d Direction, ox, oy, lineSpacing float64) (float64, float64) {
	switch d {
	case DirectionLeftToRight, DirectionRightToLeft:
		oy += lineSpacing
	case DirectionTopToBottomAndLeftToRight:
		ox += lineSpacing
	case DirectionTopToBottomAndRightToLeft:
		ox -= lineSpacing
	}
	return ox, oy
}

// computeTextBounds transforms every line box into destination space and
// returns their union.
func (m *TextCaretMap) computeTextBounds() CaretRectangle {
	var b CaretRectangle
	first := true
	for _, l := range m.lines {
		r := m.lineBoxDest(l)
		if first {
			b = r
			first = false
			continue
		}
		b.MinX = min(b.MinX, r.MinX)
		b.MaxX = max(b.MaxX, r.MaxX)
		b.MinY = min(b.MinY, r.MinY)
		b.MaxY = max(b.MaxY, r.MaxY)
	}
	return b
}

// Len returns the length of the mapped text in bytes.
func (m *TextCaretMap) Len() int {
	return len(m.text)
}

// LineCount returns the number of visual lines. An empty text has one
// empty line, and a text ending with a line break has an empty final
// line.
func (m *TextCaretMap) LineCount() int {
	return len(m.lines)
}

// Line returns the index and box of visual line index in destination
// space. A negative index selects the first line; an index past the last
// line selects the last line, so callers never need to clamp themselves.
// On a map with no lines (a zero value), Line returns (-1, zero rect).
func (m *TextCaretMap) Line(index int) (int, CaretRectangle) {
	l := m.clampLine(index)
	if l < 0 {
		return -1, CaretRectangle{}
	}
	return l, m.lineBoxDest(m.lines[l])
}

func (m *TextCaretMap) clampLine(index int) int {
	if len(m.lines) == 0 {
		return -1
	}
	if index < 0 {
		return 0
	}
	if index >= len(m.lines) {
		return len(m.lines) - 1
	}
	return index
}

func (m *TextCaretMap) lineBoxDest(l caretLine) CaretRectangle {
	return transformRect(m.geoM, CaretRectangle{
		MinX: l.boxMinX, MinY: l.boxMinY, MaxX: l.boxMaxX, MaxY: l.boxMaxY,
	})
}

// TextBounds returns the union of all line boxes in destination space.
func (m *TextCaretMap) TextBounds() CaretRectangle {
	return m.textBounds
}

// CaretAtByte maps a logical byte offset to a visual caret.
//
// offset is clamped to [0, Len]. An offset inside a rune or a glyph
// cluster snaps to the cluster boundary in the direction implied by
// affinity. Line-break codepoints map to the line on the affinity's side
// of the break: CaretAffinityBefore stays on the preceding line,
// CaretAffinityAfter moves to the next.
func (m *TextCaretMap) CaretAtByte(offset int, affinity CaretAffinity) Caret {
	if offset < 0 {
		offset = 0
	}
	if offset > len(m.text) {
		offset = len(m.text)
	}
	if len(m.lines) == 0 {
		// A zero-value map or one left half-built by a rejected
		// deserialization has no geometry; return a stable zero caret
		// instead of indexing into an empty slice.
		return Caret{ByteOffset: offset, Affinity: affinity}
	}

	lineIdx, snapped := m.resolveOffsetLine(offset, affinity)
	line := &m.lines[lineIdx]
	stop := m.findStop(line, snapped, affinity)

	px, py := m.stopPoint(line, stop.p)
	x, y := m.geoM.Apply(px, py)
	return Caret{
		ByteOffset: stop.offset,
		Affinity:   affinity,
		Line:       lineIdx,
		X:          x,
		Y:          y,
	}
}

// resolveOffsetLine picks the visual line for an offset and returns the
// offset snapped to a valid cluster boundary on that line.
func (m *TextCaretMap) resolveOffsetLine(offset int, affinity CaretAffinity) (int, int) {
	// 1. An offset strictly inside a visual line's content range belongs
	//    to that line. This never matches a line-break byte, since a hard
	//    line's range excludes its terminating break codepoint.
	for i := range m.lines {
		l := &m.lines[i]
		if offset > l.start && offset < l.end {
			return i, m.snapOffsetOnLine(l, offset, affinity)
		}
	}
	// 2. The offset equals one or more line boundaries (a line start, a
	//    line end, a soft-wrap position, or a hard line break).
	//
	//    CaretAffinityBefore selects the line that ends at the offset;
	//    CaretAffinityAfter the line that starts at it. A hard line
	//    break sits between them, so the two affinities land on different
	//    lines.
	if affinity == CaretAffinityAfter {
		// True soft-wrap boundary: a line starts at offset and the
		// preceding line ends at the same offset, with no break byte
		// between them. Requiring the preceding line to end there keeps a
		// degenerate empty line (start == end == offset) from swallowing
		// the After caret of the break byte that terminates it.
		for i := 1; i < len(m.lines); i++ {
			if m.lines[i].start == offset && m.lines[i-1].end == offset {
				return i, offset
			}
		}
		// Hard line break: a line ends at the offset and the next line
		// starts after the break codepoint, so move to the following
		// line's own start boundary.
		for i := 0; i+1 < len(m.lines); i++ {
			if m.lines[i].end == offset {
				return i + 1, m.lines[i+1].start
			}
		}
	}
	// Before affinity, or an After offset with no starting line (the end
	// of text): prefer a line ending at the offset.
	for i := range m.lines {
		if m.lines[i].end == offset {
			return i, offset
		}
	}
	// A line-start with no ending line (offset 0) under Before affinity.
	for i := range m.lines {
		if m.lines[i].start == offset {
			return i, offset
		}
	}
	// Defensive fallback: the offset lies inside a gap that no line range
	// covers (a soft-wrap boundary whose whitespace belongs to no
	// fragment). Attach it to the nearest line by byte distance, snapping
	// to that line's start or end, rather than to an unrelated line.
	best := 0
	bestDist := math.Inf(1)
	for i := range m.lines {
		l := &m.lines[i]
		var d float64
		var snap int
		switch {
		case offset < l.start:
			d = float64(l.start - offset)
			snap = l.start
		case offset > l.end:
			d = float64(offset - l.end)
			snap = l.end
		default:
			d = 0
			snap = offset
		}
		if d < bestDist {
			bestDist = d
			best = i
			if d == 0 {
				return i, snap
			}
		}
	}
	l := &m.lines[best]
	if offset < l.start {
		return best, l.start
	}
	return best, l.end
}

// snapOffsetOnLine clamps an offset to a valid caret byte on the line,
// never landing inside a cluster. It works from the stop table, which is
// present both for freshly built and for deserialized maps (the latter
// carry no clusters).
//
// An offset inside a cluster snaps to the cluster boundary implied by
// affinity: CaretAffinityBefore snaps backward to the cluster's start
// (the leading-edge convention [AdvanceAt] uses), CaretAffinityAfter
// snaps forward to the cluster's end. This is chosen over the stops'
// logical offsets, independent of their visual order on a bidi line.
func (m *TextCaretMap) snapOffsetOnLine(l *caretLine, offset int, affinity CaretAffinity) int {
	if len(l.stops) == 0 {
		return l.start
	}
	if affinity == CaretAffinityAfter {
		// Smallest represented offset that is >= offset.
		best := 0
		found := false
		for _, s := range l.stops {
			if s.offset >= offset && (!found || s.offset < best) {
				best = s.offset
				found = true
			}
		}
		if found {
			return best
		}
		// offset is past every stop: clamp to the largest.
		maxOff := l.stops[0].offset
		for _, s := range l.stops[1:] {
			maxOff = max(maxOff, s.offset)
		}
		return maxOff
	}
	// CaretAffinityBefore: largest represented offset that is <= offset.
	best := l.stops[0].offset
	found := false
	for _, s := range l.stops {
		if s.offset <= offset && (!found || s.offset > best) {
			best = s.offset
			found = true
		}
	}
	if found {
		return best
	}
	// offset precedes every stop (only possible before the line start, as
	// the line start is always a stop): snap forward to the smallest.
	minOff := l.stops[0].offset
	for _, s := range l.stops[1:] {
		minOff = min(minOff, s.offset)
	}
	return minOff
}

// findStop returns the stop matching the offset and affinity. If the line
// has no stop at exactly that offset (for example a Caret whose Line field
// does not match its ByteOffset after a caller edits it), it returns the
// nearest stop with the requested affinity, falling back to the nearest
// stop overall. A line always has at least one stop.
func (m *TextCaretMap) findStop(l *caretLine, offset int, affinity CaretAffinity) caretStop {
	for _, s := range l.stops {
		if s.offset == offset && s.affinity == affinity {
			return s
		}
	}
	best := caretStop{offset: -1}
	found := false
	for _, s := range l.stops {
		if s.affinity != affinity {
			continue
		}
		if !found || absInt(s.offset-offset) < absInt(best.offset-offset) {
			best, found = s, true
		}
	}
	if found {
		return best
	}
	best = l.stops[0]
	for _, s := range l.stops[1:] {
		if absInt(s.offset-offset) < absInt(best.offset-offset) {
			best = s
		}
	}
	return best
}

// stopPoint converts a primary-axis stop coordinate into a layout-space
// point.
func (m *TextCaretMap) stopPoint(l *caretLine, p float64) (x, y float64) {
	if m.direction.isHorizontal() {
		return p, l.originY
	}
	return l.originX, p
}

// PositionAt maps a destination-space point (e.g. a mouse position) to
// the nearest caret.
//
// Points outside the text bounds clamp to the boundary caret: the first
// or last line along the secondary axis, and the line's start or end
// along the primary axis. PositionAt always returns a caret; degenerate
// (empty) text returns its single empty-line caret.
func (m *TextCaretMap) PositionAt(x, y float64) Caret {
	if len(m.lines) == 0 {
		return Caret{}
	}
	lx, ly := m.invM.Apply(x, y)
	horizontal := m.direction.isHorizontal()

	lineIdx := m.lineAtSecondary(lx, ly)
	line := &m.lines[lineIdx]

	if len(line.stops) <= 1 {
		s := line.stops[0]
		px, py := m.stopPoint(line, s.p)
		dx, dy := m.geoM.Apply(px, py)
		return Caret{ByteOffset: s.offset, Affinity: s.affinity, Line: lineIdx, X: dx, Y: dy}
	}

	primary := ly
	if horizontal {
		primary = lx
	}

	idx, exact := slices.BinarySearchFunc(line.stops, primary, func(s caretStop, target float64) int {
		return cmpFloat(s.p, target)
	})
	var chosen caretStop
	switch {
	case idx == 0:
		chosen = line.stops[0]
	case idx >= len(line.stops):
		chosen = line.stops[len(line.stops)-1]
	default:
		prev, next := line.stops[idx-1], line.stops[idx]
		if exact {
			chosen = next
		} else if math.Abs(primary-prev.p) <= math.Abs(next.p-primary) {
			chosen = prev
		} else {
			chosen = next
		}
	}

	px, py := m.stopPoint(line, chosen.p)
	dx, dy := m.geoM.Apply(px, py)
	return Caret{
		ByteOffset: chosen.offset,
		Affinity:   chosen.affinity,
		Line:       lineIdx,
		X:          dx,
		Y:          dy,
	}
}

// lineAtSecondary returns the visual line whose secondary-axis span
// contains the layout-space point, clamping to the first or last line
// outside the block and splitting inter-line gaps at their midpoint.
func (m *TextCaretMap) lineAtSecondary(lx, ly float64) int {
	horizontal := m.direction.isHorizontal()
	secondary := ly
	if !horizontal {
		secondary = lx
	}
	for i := range m.lines {
		l := &m.lines[i]
		lo, hi := l.boxMinY, l.boxMaxY
		if !horizontal {
			lo, hi = l.boxMinX, l.boxMaxX
		}
		if secondary >= lo && secondary <= hi {
			return i
		}
		if i < len(m.lines)-1 {
			n := &m.lines[i+1]
			nlo := n.boxMinY
			if !horizontal {
				nlo = n.boxMinX
			}
			if secondary > hi && secondary < nlo {
				if secondary-hi <= nlo-secondary {
					return i
				}
				return i + 1
			}
		}
	}
	if len(m.lines) == 1 {
		return 0
	}
	first := &m.lines[0]
	firstLo := first.boxMinY
	if !horizontal {
		firstLo = first.boxMinX
	}
	if secondary < firstLo {
		return 0
	}
	return len(m.lines) - 1
}

// CaretRect returns the rectangle occupied by the caret itself in
// destination space. For horizontal faces it is a thin bar on the
// baseline spanning the line box; for vertical faces a thin horizontal
// bar. The bar is one destination-space pixel wide (in layout space,
// scaled by the inverse of GeoM's primary-axis scale).
func (m *TextCaretMap) CaretRect(c Caret) CaretRectangle {
	idx := m.clampLine(c.Line)
	if idx < 0 {
		return CaretRectangle{}
	}
	line := m.lines[idx]
	stop := m.findStop(&line, c.ByteOffset, c.Affinity)
	px, py := m.stopPoint(&line, stop.p)

	thickness := m.caretThickness()
	horizontal := m.direction.isHorizontal()

	var r CaretRectangle
	if horizontal {
		r = CaretRectangle{
			MinX: px - thickness/2,
			MaxX: px + thickness/2,
			MinY: line.boxMinY,
			MaxY: line.boxMaxY,
		}
	} else {
		r = CaretRectangle{
			MinX: line.boxMinX,
			MaxX: line.boxMaxX,
			MinY: py - thickness/2,
			MaxY: py + thickness/2,
		}
	}
	return transformRect(m.geoM, r)
}

// caretThickness estimates one destination pixel in layout space along
// the primary axis, defaulting to one layout pixel when the scale cannot
// be derived (a rotated transform).
func (m *TextCaretMap) caretThickness() float64 {
	var ux, uy float64
	if m.direction.isHorizontal() {
		ux, uy = 1, 0
	} else {
		ux, uy = 0, 1
	}
	x0, y0 := m.geoM.Apply(0, 0)
	x1, y1 := m.geoM.Apply(ux, uy)
	length := math.Hypot(x1-x0, y1-y0)
	if length == 0 || !isFinite(length) {
		return 1
	}
	return 1 / length
}

// SelectionRect returns the rectangles covering a selection from start to
// end byte offsets in destination space. start and end are clamped to
// [0, Len] and reordered so the smaller offset is the anchor.
//
// A selection spanning multiple lines returns one rectangle per visual
// line the selection touches; within a line the rectangle spans the
// selected primary-axis extent and the full line box along the secondary
// axis. A zero-length selection returns nil.
func (m *TextCaretMap) SelectionRect(start, end int) []CaretRectangle {
	if start > end {
		start, end = end, start
	}
	start = clampInt(start, 0, len(m.text))
	end = clampInt(end, 0, len(m.text))
	if start == end {
		return nil
	}

	var rects []CaretRectangle
	horizontal := m.direction.isHorizontal()
	for i := range m.lines {
		l := &m.lines[i]
		if end <= l.start || start >= l.end {
			continue
		}
		selStart := max(start, l.start)
		selEnd := min(end, l.end)
		if selStart == selEnd {
			continue
		}
		p0 := m.primaryAtOffset(l, selStart, CaretAffinityBefore)
		p1 := m.primaryAtOffset(l, selEnd, CaretAffinityAfter)
		if p1 < p0 {
			p0, p1 = p1, p0
		}
		var r CaretRectangle
		if horizontal {
			r = CaretRectangle{MinX: p0, MaxX: p1, MinY: l.boxMinY, MaxY: l.boxMaxY}
		} else {
			r = CaretRectangle{MinX: l.boxMinX, MaxX: l.boxMaxX, MinY: p0, MaxY: p1}
		}
		rects = append(rects, transformRect(m.geoM, r))
	}
	return rects
}

// primaryAtOffset resolves a byte offset to its primary coordinate on a
// line, snapping inside clusters as needed.
func (m *TextCaretMap) primaryAtOffset(l *caretLine, offset int, affinity CaretAffinity) float64 {
	snapped := m.snapOffsetOnLine(l, offset, affinity)
	return m.findStop(l, snapped, affinity).p
}

// transformRect applies GeoM to an axis-aligned layout-space rectangle
// and returns the axis-aligned bounding box in destination space.
func transformRect(g ebiten.GeoM, r CaretRectangle) CaretRectangle {
	corners := [4][2]float64{
		{r.MinX, r.MinY},
		{r.MaxX, r.MinY},
		{r.MinX, r.MaxY},
		{r.MaxX, r.MaxY},
	}
	var out CaretRectangle
	for i, c := range corners {
		x, y := g.Apply(c[0], c[1])
		if i == 0 {
			out.MinX, out.MaxX = x, x
			out.MinY, out.MaxY = y, y
			continue
		}
		out.MinX = min(out.MinX, x)
		out.MaxX = max(out.MaxX, x)
		out.MinY = min(out.MinY, y)
		out.MaxY = max(out.MaxY, y)
	}
	return out
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// endsWithLineBreak reports whether s ends with one of the package's
// line-break codepoints (including CRLF).
func endsWithLineBreak(s string) bool {
	if len(s) >= 2 && s[len(s)-2] == '\r' && s[len(s)-1] == '\n' {
		return true
	}
	r, _ := utf8.DecodeLastRuneInString(s)
	return textutil.IsLineBreak(r)
}

func isAllSpace(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !isSpaceRune(r) {
			return false
		}
	}
	return true
}

// isSpaceRune reports whether r is a cluster-breaking whitespace rune
// used to find soft-wrap opportunities.
func isSpaceRune(r rune) bool {
	switch r {
	case ' ', '\t', 0x00A0, 0x1680, 0x202F, 0x205F, 0x3000:
		return true
	}
	if r >= 0x2000 && r <= 0x200A {
		return true
	}
	return false
}

// Serialization.

type caretMapJSON struct {
	Version   int             `json:"version"`
	Direction Direction       `json:"direction"`
	Text      string          `json:"text"`
	GeoM      [6]float64      `json:"geoM"`
	Lines     []caretLineJSON `json:"lines"`
	TextBound CaretRectangle  `json:"textBounds"`
}

type caretLineJSON struct {
	Start    int             `json:"start"`
	End      int             `json:"end"`
	HardLine int             `json:"hardLine"`
	OriginX  float64         `json:"originX"`
	OriginY  float64         `json:"originY"`
	Box      CaretRectangle  `json:"box"`
	StartP   float64         `json:"startP"`
	EndP     float64         `json:"endP"`
	Stops    []caretStopJSON `json:"stops"`
}

type caretStopJSON struct {
	Offset   int           `json:"offset"`
	P        float64       `json:"p"`
	Affinity CaretAffinity `json:"affinity"`
}

// MarshalJSON serializes the map's logical positions, line geometry, and
// stops. The serialized form is self-contained: an unmarshaled map
// answers every geometric query without the original face.
func (m *TextCaretMap) MarshalJSON() ([]byte, error) {
	j := caretMapJSON{
		Version:   1,
		Direction: m.direction,
		Text:      m.text,
		GeoM: [6]float64{
			m.geoM.Element(0, 0), m.geoM.Element(0, 1), m.geoM.Element(0, 2),
			m.geoM.Element(1, 0), m.geoM.Element(1, 1), m.geoM.Element(1, 2),
		},
		TextBound: m.textBounds,
	}
	j.Lines = make([]caretLineJSON, len(m.lines))
	for i, l := range m.lines {
		lj := caretLineJSON{
			Start:    l.start,
			End:      l.end,
			HardLine: l.hardLine,
			OriginX:  l.originX,
			OriginY:  l.originY,
			Box: CaretRectangle{
				MinX: l.boxMinX, MinY: l.boxMinY, MaxX: l.boxMaxX, MaxY: l.boxMaxY,
			},
			StartP: l.startP,
			EndP:   l.endP,
			Stops:  make([]caretStopJSON, len(l.stops)),
		}
		for k, s := range l.stops {
			lj.Stops[k] = caretStopJSON{Offset: s.offset, P: s.p, Affinity: s.affinity}
		}
		j.Lines[i] = lj
	}
	return json.Marshal(j)
}

// UnmarshalJSON restores a map previously produced by
// [TextCaretMap.MarshalJSON]. The restored map has no Face and cannot be
// re-laid-out, but supports every position, rectangle, line, and
// selection query.
func (m *TextCaretMap) UnmarshalJSON(data []byte) error {
	var j caretMapJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	if j.Version != 1 {
		return fmt.Errorf("text: unsupported TextCaretMap serialization version %d", j.Version)
	}
	var geoM ebiten.GeoM
	for k := 0; k < 6; k++ {
		geoM.SetElement(k/3, k%3, j.GeoM[k])
	}
	if !geoMIsFinite(geoM) {
		return fmt.Errorf("text: TextCaretMap serialization has a non-finite GeoM")
	}
	if !geoM.IsInvertible() {
		return fmt.Errorf("text: TextCaretMap serialization has a non-invertible GeoM")
	}
	if len(j.Lines) == 0 {
		return fmt.Errorf("text: TextCaretMap serialization has no lines")
	}
	// Validate every line before mutating the receiver so a rejected
	// payload leaves m unchanged rather than in a half-built, panicky
	// state.
	for i, lj := range j.Lines {
		if lj.Start > lj.End {
			return fmt.Errorf("text: TextCaretMap serialization line %d has start %d > end %d", i, lj.Start, lj.End)
		}
	}
	*m = TextCaretMap{
		text:       j.Text,
		direction:  j.Direction,
		geoM:       geoM,
		textBounds: j.TextBound,
	}
	m.invM = geoM
	m.invM.Invert()
	m.lines = make([]caretLine, len(j.Lines))
	for i, lj := range j.Lines {
		l := caretLine{
			start:    lj.Start,
			end:      lj.End,
			hardLine: lj.HardLine,
			originX:  lj.OriginX,
			originY:  lj.OriginY,
			boxMinX:  lj.Box.MinX,
			boxMinY:  lj.Box.MinY,
			boxMaxX:  lj.Box.MaxX,
			boxMaxY:  lj.Box.MaxY,
			startP:   lj.StartP,
			endP:     lj.EndP,
		}
		l.stops = make([]caretStop, len(lj.Stops))
		if len(l.stops) == 0 {
			// A line must have at least one stop so queries on it are
			// well-defined. Synthesize the line-start stop, as a freshly
			// built empty line has.
			l.stops = []caretStop{{offset: l.start, p: l.startP, affinity: CaretAffinityBefore}}
		}
		for k, s := range lj.Stops {
			l.stops[k] = caretStop{offset: s.Offset, p: s.P, affinity: s.Affinity}
		}
		m.lines[i] = l
	}
	return nil
}

// MarshalBinary implements encoding.BinaryMarshaler as a wrapper around
// the JSON form.
func (m *TextCaretMap) MarshalBinary() ([]byte, error) {
	return m.MarshalJSON()
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (m *TextCaretMap) UnmarshalBinary(data []byte) error {
	return m.UnmarshalJSON(data)
}

var _ error = (*TextCaretMapError)(nil)
