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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"
	"unicode/utf8"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// TextLayoutSnapshotVersion is the format version written by
// [TextLayoutSnapshot.MarshalBinary].
//
// A decoder rejects any other version with a [TextLayoutSnapshotError] whose
// Path is "version".
const TextLayoutSnapshotVersion = 1

// Limits applied when a snapshot is built or decoded. They bound the memory a
// malicious or corrupted snapshot can request.
const (
	maxTextLayoutSnapshotTextBytes        = 16 << 20
	maxTextLayoutSnapshotLines            = 1 << 20
	maxTextLayoutSnapshotGlyphs           = 1 << 20
	maxTextLayoutSnapshotDescriptionBytes = 1 << 12
	maxTextLayoutSnapshotBytes            = 64 << 20

	// snapshotEncodedLineBytes and snapshotEncodedGlyphBytes are the encoded
	// sizes of one line and one glyph record. They let the decoder reject a
	// count that cannot fit in the remaining payload before allocating.
	snapshotEncodedLineBytes  = 3*8 + 3*8 // three ints and three floats
	snapshotEncodedGlyphBytes = 4*8 + 8 + 6*8
)

// snapshotMagic is the file signature at the start of an encoded snapshot.
var snapshotMagic = [8]byte{'E', 'B', 'T', 'L', 'S', 'N', 'A', 'P'}

// TextLayoutSnapshotError reports a problem in a layout snapshot operation.
//
// Path is a dotted field path into the snapshot or the build/replay inputs,
// such as "options.geoM[0]", "lines[2].advance", or "glyphs[7].imageBounds".
// It lets a caller identify the offending field without parsing the message.
type TextLayoutSnapshotError struct {
	// Path is the field path where the problem was found.
	Path string

	// Message describes the problem.
	Message string

	// Err is the underlying error, if any.
	Err error
}

// Error implements error.
func (e *TextLayoutSnapshotError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("text: snapshot: field %q: %s: %v", e.Path, e.Message, e.Err)
	}
	return fmt.Sprintf("text: snapshot: field %q: %s", e.Path, e.Message)
}

// Unwrap returns the underlying error.
func (e *TextLayoutSnapshotError) Unwrap() error {
	return e.Err
}

func snapshotErrorf(path, format string, args ...any) error {
	return &TextLayoutSnapshotError{
		Path:    path,
		Message: fmt.Sprintf(format, args...),
	}
}

// prefixSnapshotErrorPath returns err with its path prefixed by prefix. Errors
// that are not [*TextLayoutSnapshotError] are returned unchanged.
func prefixSnapshotErrorPath(prefix string, err error) error {
	var se *TextLayoutSnapshotError
	if !errors.As(err, &se) {
		return err
	}
	cloned := *se
	cloned.Path = prefix + "." + se.Path
	return &cloned
}

// TextLayoutSnapshotOptions configures [NewTextLayoutSnapshot].
type TextLayoutSnapshotOptions struct {
	// LayoutOptions controls the alignment and the line spacing, exactly as
	// for [Draw] and [AppendGlyphs].
	LayoutOptions

	// GeoM is the transformation applied to the layout coordinates when the
	// snapshot is replayed. It corresponds to [DrawOptions.GeoM].
	//
	// GeoM does not affect glyph positions or rasterization; it only places
	// the finished layout in destination space. The zero value is identity.
	GeoM ebiten.GeoM
}

// TextLayoutSnapshotLine describes one laid-out line.
//
// The coordinates are in the layout space captured by the snapshot: the
// origin is the top-left of the rendering region before the snapshot GeoM is
// applied, matching the coordinates [AppendGlyphs] returns.
type TextLayoutSnapshotLine struct {
	// IndexInBytes is the byte offset of the first byte of the line in the
	// snapshot text.
	IndexInBytes int

	// EndIndexInBytes is the byte offset one past the line's content,
	// excluding the line terminator.
	EndIndexInBytes int

	// LineBreakLen is the byte length of the line terminator. It is 0 for a
	// last line without a terminator.
	LineBreakLen int

	// Advance is the line's visual advance along the primary direction.
	Advance float64

	// BaselineX and BaselineY are the line's baseline origin. The primary
	// coordinate advances along the text; the secondary coordinate is the
	// baseline for horizontal text, or the column center for vertical text.
	BaselineX float64
	BaselineY float64
}

// TextLayoutSnapshotGlyph describes one glyph in a snapshot.
//
// A glyph cluster that shapes into multiple glyphs produces multiple entries
// sharing byte indices. The fields mirror [Glyph].
type TextLayoutSnapshotGlyph struct {
	// StartIndexInBytes is the start byte index in the snapshot text.
	StartIndexInBytes int

	// EndIndexInBytes is the end byte index in the snapshot text.
	EndIndexInBytes int

	// GID is the glyph ID in the face's font.
	GID uint32

	// ImageBounds is the rectangle the rasterized glyph image occupies in
	// layout coordinates, the same rectangle [LazyGlyph.ImageBounds] reports
	// and the position [Glyph.Image] is drawn at by [Draw].
	ImageBounds image.Rectangle

	// HasImage reports whether the glyph produces an image. Control
	// characters and zero-area glyphs do not.
	HasImage bool

	// Colored reports whether the glyph image is colored, like an emoji,
	// rather than grayscale.
	Colored bool

	// OriginX and OriginY are the glyph's origin in layout coordinates.
	OriginX float64
	OriginY float64

	// OriginOffsetX and OriginOffsetY adjust the origin for this glyph.
	OriginOffsetX float64
	OriginOffsetY float64

	// AdvanceX and AdvanceY are the advances applied after this glyph.
	AdvanceX float64
	AdvanceY float64
}

// TextLayoutSnapshot is an immutable, serializable record of a laid-out text.
//
// A snapshot captures line boundaries, per-glyph positions, advances, and
// baselines for a text, face, and layout options without rasterizing the text
// or drawing it. It records the face's [FontFingerprint] so that replaying the
// snapshot against a changed face is rejected instead of silently drawing at
// stale positions.
//
// Replay requires the original font resource: the snapshot stores glyph IDs
// and positions, not glyph bitmaps or outlines. [TextLayoutSnapshot.Draw],
// [TextLayoutSnapshot.AppendGlyphs], and
// [TextLayoutSnapshot.AppendVectorPath] therefore take a Face and verify that
// its fingerprint matches the snapshot's before doing any work.
//
// A TextLayoutSnapshot is safe for concurrent use. Its accessors return copies;
// callers cannot mutate the snapshot or share mutable state with it.
type TextLayoutSnapshot struct {
	text        string
	direction   Direction
	layout      LayoutOptions
	geoM        ebiten.GeoM
	metrics     Metrics
	width       float64
	height      float64
	fingerprint FontFingerprint
	lines       []TextLayoutSnapshotLine
	glyphs      []TextLayoutSnapshotGlyph
}

// NewTextLayoutSnapshot lays out text with face and options and returns an
// immutable snapshot of the result.
//
// The layout arithmetic is the same as [Draw] and [AppendGlyphs], so line
// boundaries, glyph origins, advances, and baselines match the live layout
// byte for byte. The text is not rasterized.
//
// NewTextLayoutSnapshot returns a [TextLayoutSnapshotError] on invalid options
// or an over-long text.
func NewTextLayoutSnapshot(text string, face Face, options *TextLayoutSnapshotOptions) (*TextLayoutSnapshot, error) {
	if face == nil {
		return nil, snapshotErrorf("face", "no face is given")
	}
	if len(text) > maxTextLayoutSnapshotTextBytes {
		return nil, snapshotErrorf("text", "text is %d bytes, which exceeds the limit %d", len(text), maxTextLayoutSnapshotTextBytes)
	}
	if !utf8.ValidString(text) {
		return nil, snapshotErrorf("text", "text is not valid UTF-8")
	}

	var op TextLayoutSnapshotOptions
	if options != nil {
		op = *options
	}

	direction := face.direction()
	if !direction.isValid() {
		return nil, snapshotErrorf("face.direction", "invalid direction %d", direction)
	}
	if err := validateLayoutOptions("options", &op.LayoutOptions); err != nil {
		return nil, err
	}
	if err := validateGeoM("options.geoM", &op.GeoM); err != nil {
		return nil, err
	}

	fingerprint, err := face.fingerprint()
	if err != nil {
		return nil, err
	}

	metrics := face.Metrics()
	if err := validateMetrics("metrics", &metrics); err != nil {
		return nil, err
	}

	s := &TextLayoutSnapshot{
		text:        text,
		direction:   direction,
		layout:      op.LayoutOptions,
		geoM:        op.GeoM,
		metrics:     metrics,
		fingerprint: fingerprint,
	}

	lazyBuf := make([]LazyGlyph, 0, 64)
	var buildErr error
	var boundaryWidth, boundaryHeight float64
	var lines []TextLayoutSnapshotLine
	var glyphs []TextLayoutSnapshotGlyph
	layoutLines(text, face, &s.layout, func(line string, indexOffset int, originX, originY float64, info lineLayoutInfo) {
		if buildErr != nil {
			return
		}
		if len(lines) >= maxTextLayoutSnapshotLines {
			buildErr = snapshotErrorf("lines", "the text produces more than %d lines", maxTextLayoutSnapshotLines)
			return
		}
		lazyBuf = face.appendLazyGlyphsForLine(lazyBuf[:0], line, indexOffset, originX, originY, nil)
		if len(glyphs)+len(lazyBuf) > maxTextLayoutSnapshotGlyphs {
			buildErr = snapshotErrorf("glyphs", "the text produces more than %d glyphs", maxTextLayoutSnapshotGlyphs)
			return
		}
		if info.index == 0 {
			boundaryWidth = info.boundaryWidth
			boundaryHeight = info.boundaryHeight
		}
		lines = append(lines, TextLayoutSnapshotLine{
			IndexInBytes:    indexOffset,
			EndIndexInBytes: indexOffset + info.contentLen,
			LineBreakLen:    info.lineBreakLen,
			Advance:         info.advance,
			BaselineX:       info.originX,
			BaselineY:       info.originY,
		})
		for i := range lazyBuf {
			lg := &lazyBuf[i]
			glyphs = append(glyphs, TextLayoutSnapshotGlyph{
				StartIndexInBytes: lg.StartIndexInBytes,
				EndIndexInBytes:   lg.EndIndexInBytes,
				GID:               lg.GID,
				ImageBounds:       lg.ImageBounds,
				HasImage:          !lg.ImageBounds.Empty(),
				Colored:           lg.Colored(),
				OriginX:           lg.OriginX,
				OriginY:           lg.OriginY,
				OriginOffsetX:     lg.OriginOffsetX,
				OriginOffsetY:     lg.OriginOffsetY,
				AdvanceX:          lg.AdvanceX,
				AdvanceY:          lg.AdvanceY,
			})
		}
	})
	if buildErr != nil {
		return nil, buildErr
	}
	s.lines = lines
	s.glyphs = glyphs
	s.width = boundaryWidth
	s.height = boundaryHeight
	if err := s.validate(); err != nil {
		return nil, err
	}
	return s, nil
}

// validateLayoutOptions checks alignments. Invalid values are rejected rather
// than panicking as Draw does, since snapshots cross a trust boundary.
func validateLayoutOptions(path string, options *LayoutOptions) error {
	if math.IsNaN(options.LineSpacing) || math.IsInf(options.LineSpacing, 0) {
		return snapshotErrorf(path+".lineSpacing", "line spacing must be finite, but was %v", options.LineSpacing)
	}
	for _, a := range [...]struct {
		name  string
		align Align
	}{
		{"primaryAlign", options.PrimaryAlign},
		{"secondaryAlign", options.SecondaryAlign},
	} {
		switch a.align {
		case AlignStart, AlignCenter, AlignEnd:
		default:
			return snapshotErrorf(path+"."+a.name, "invalid alignment %d", a.align)
		}
	}
	return nil
}

func validateGeoM(path string, geoM *ebiten.GeoM) error {
	for i := range 2 {
		for j := range 3 {
			v := geoM.Element(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return snapshotErrorf(fmt.Sprintf("%s[%d]", path, i*3+j), "geometry matrix element must be finite, but was %v", v)
			}
		}
	}
	return nil
}

func validateMetrics(path string, m *Metrics) error {
	for _, v := range [...]struct {
		name  string
		value float64
	}{
		{"hLineGap", m.HLineGap},
		{"hAscent", m.HAscent},
		{"hDescent", m.HDescent},
		{"vLineGap", m.VLineGap},
		{"vAscent", m.VAscent},
		{"vDescent", m.VDescent},
		{"xHeight", m.XHeight},
		{"capHeight", m.CapHeight},
	} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return snapshotErrorf(path+"."+v.name, "metric must be finite, but was %v", v.value)
		}
	}
	return nil
}

// Text returns the snapshot's text.
func (s *TextLayoutSnapshot) Text() string {
	return s.text
}

// Direction returns the face direction the snapshot was laid out with.
func (s *TextLayoutSnapshot) Direction() Direction {
	return s.direction
}

// LayoutOptions returns the layout options the snapshot was laid out with.
func (s *TextLayoutSnapshot) LayoutOptions() LayoutOptions {
	return s.layout
}

// GeoM returns the transformation the snapshot was built with.
func (s *TextLayoutSnapshot) GeoM() ebiten.GeoM {
	return s.geoM
}

// Metrics returns the face metrics at snapshot time.
func (s *TextLayoutSnapshot) Metrics() Metrics {
	return s.metrics
}

// Bounds returns the size of the rendering region the snapshot describes,
// in layout coordinates and before the snapshot GeoM is applied.
func (s *TextLayoutSnapshot) Bounds() (width, height float64) {
	return s.width, s.height
}

// Fingerprint returns the fingerprint of the face the snapshot was built with.
func (s *TextLayoutSnapshot) Fingerprint() FontFingerprint {
	return s.fingerprint
}

// VerifyFace reports whether face still matches the snapshot.
//
// VerifyFace checks the face's fingerprint and then re-runs the layout,
// comparing every glyph's identity and position with the snapshot. A nil error
// means the snapshot can be replayed with face. A mismatch is a
// [TextLayoutSnapshotError] whose Path identifies the first differing field.
//
// VerifyFace does not rasterize any glyph, so it is cheap compared to
// [TextLayoutSnapshot.Draw].
func (s *TextLayoutSnapshot) VerifyFace(face Face) error {
	_, err := s.realize(face)
	return err
}

// Lines returns the snapshot's line records. The returned slice is a copy;
// modifying it does not affect the snapshot.
func (s *TextLayoutSnapshot) Lines() []TextLayoutSnapshotLine {
	return slices.Clone(s.lines)
}

// Glyphs returns the snapshot's glyph records. The returned slice is a copy;
// modifying it does not affect the snapshot.
func (s *TextLayoutSnapshot) Glyphs() []TextLayoutSnapshotGlyph {
	return slices.Clone(s.glyphs)
}

// TextLayoutSnapshotCluster is a run of glyphs that share a byte range in the
// text. A cluster can shape into multiple glyphs (for example, a ligature or a
// base letter plus a combining mark); they refer to the same bytes and appear
// consecutively in glyph order.
type TextLayoutSnapshotCluster struct {
	// StartIndexInBytes is the start byte index of the cluster in the text.
	StartIndexInBytes int

	// EndIndexInBytes is the end byte index of the cluster in the text.
	EndIndexInBytes int

	// GlyphStart and GlyphEnd delimit the cluster's glyphs in
	// [TextLayoutSnapshot.Glyphs]: the glyphs are Glyphs()[GlyphStart:GlyphEnd].
	GlyphStart int
	GlyphEnd   int

	// Advance is the cluster's total advance along the primary direction,
	// the sum of its glyphs' advances.
	Advance float64
}

// Clusters returns the snapshot's glyph clusters in visual order. The returned
// slice is a copy; modifying it does not affect the snapshot.
func (s *TextLayoutSnapshot) Clusters() []TextLayoutSnapshotCluster {
	clusters := make([]TextLayoutSnapshotCluster, 0, len(s.glyphs))
	for i := 0; i < len(s.glyphs); {
		start, end := s.glyphs[i].StartIndexInBytes, s.glyphs[i].EndIndexInBytes
		j := i + 1
		var advance float64
		advance += s.glyphs[i].AdvanceX + s.glyphs[i].AdvanceY
		for j < len(s.glyphs) && s.glyphs[j].StartIndexInBytes == start && s.glyphs[j].EndIndexInBytes == end {
			advance += s.glyphs[j].AdvanceX + s.glyphs[j].AdvanceY
			j++
		}
		clusters = append(clusters, TextLayoutSnapshotCluster{
			StartIndexInBytes: start,
			EndIndexInBytes:   end,
			GlyphStart:        i,
			GlyphEnd:          j,
			Advance:           advance,
		})
		i = j
	}
	return clusters
}

// Draw replays the snapshot onto dst with face.
//
// The snapshot's captured positions are drawn, so the result matches
// [AppendGlyphs] for the same text, face, and layout options for any
// destination image and frame. options supplies color, blending, and an
// additional transformation applied after the snapshot's own GeoM; its GeoM
// does not change glyph rasterization, matching Draw.
//
// Draw rejects a face whose fingerprint or live layout no longer matches the
// snapshot with a [TextLayoutSnapshotError].
//
// Draw is concurrent-safe.
func (s *TextLayoutSnapshot) Draw(dst *ebiten.Image, face Face, options *ebiten.DrawImageOptions) error {
	if dst == nil {
		return snapshotErrorf("dst", "no destination image is given")
	}
	lazy, err := s.realize(face)
	if err != nil {
		return err
	}

	var drawOp ebiten.DrawImageOptions
	if options != nil {
		drawOp = *options
	}
	geoM := s.geoM
	geoM.Concat(drawOp.GeoM)

	coloredOp := drawOp
	coloredOp.ColorScale.Reset()
	coloredOp.ColorScale.ScaleAlpha(drawOp.ColorScale.A())

	// Realize every glyph image in glyph order before drawing any of them,
	// mirroring AppendGlyphs and Draw's two-pass order. Realizing in glyph
	// order, regardless of visibility, keeps the atlas realization sequence
	// identical to a live layout, so a rotated or scaled replay samples the
	// same texels.
	images := make([]*ebiten.Image, len(s.glyphs))
	for i := range s.glyphs {
		sg := &s.glyphs[i]
		if !sg.HasImage || sg.ImageBounds.Empty() {
			continue
		}
		images[i] = lazy[i].Image()
	}

	for i := range s.glyphs {
		sg := &s.glyphs[i]
		img := images[i]
		if img == nil {
			continue
		}
		op := &drawOp
		if sg.Colored {
			op = &coloredOp
		}
		op.GeoM.Reset()
		op.GeoM.Translate(float64(sg.ImageBounds.Min.X), float64(sg.ImageBounds.Min.Y))
		op.GeoM.Concat(geoM)
		dst.DrawImage(img, op)
	}
	return nil
}

// AppendGlyphs replays the snapshot's glyphs onto glyphs and returns the
// extended slice, mirroring [AppendGlyphs]. The returned glyphs use the
// snapshot's captured positions and the face's glyph images.
//
// AppendGlyphs returns a [TextLayoutSnapshotError] if face does not match the
// snapshot.
func (s *TextLayoutSnapshot) AppendGlyphs(glyphs []Glyph, face Face) ([]Glyph, error) {
	lazy, err := s.realize(face)
	if err != nil {
		return glyphs, err
	}
	for i := range s.glyphs {
		sg := &s.glyphs[i]
		lg := &lazy[i]
		glyphs = append(glyphs, Glyph{
			StartIndexInBytes: sg.StartIndexInBytes,
			EndIndexInBytes:   sg.EndIndexInBytes,
			GID:               sg.GID,
			Image:             lg.Image(),
			Colored:           sg.Colored,
			X:                 float64(sg.ImageBounds.Min.X),
			Y:                 float64(sg.ImageBounds.Min.Y),
			OriginX:           sg.OriginX,
			OriginY:           sg.OriginY,
			OriginOffsetX:     sg.OriginOffsetX,
			OriginOffsetY:     sg.OriginOffsetY,
			AdvanceX:          sg.AdvanceX,
			AdvanceY:          sg.AdvanceY,
		})
	}
	return glyphs, nil
}

// AppendVectorPath replays the snapshot as vector outlines appended to path.
//
// The snapshot's GeoM is applied to the outlines. The result matches
// [AppendVectorPath] with the same face and layout options when the GeoM is
// identity. Faces without outlines, such as [GoXFace], append nothing.
//
// AppendVectorPath returns a [TextLayoutSnapshotError] if face does not match
// the snapshot.
func (s *TextLayoutSnapshot) AppendVectorPath(path *vector.Path, face Face) error {
	if path == nil {
		return snapshotErrorf("path", "no path is given")
	}
	if err := s.checkFace(face); err != nil {
		return err
	}

	if s.geoMIsIdentity() {
		for i := range s.lines {
			line := &s.lines[i]
			face.appendVectorPathForLine(path, s.text[line.IndexInBytes:line.EndIndexInBytes], line.BaselineX, line.BaselineY)
		}
		return nil
	}

	var scratch vector.Path
	for i := range s.lines {
		line := &s.lines[i]
		face.appendVectorPathForLine(&scratch, s.text[line.IndexInBytes:line.EndIndexInBytes], line.BaselineX, line.BaselineY)
	}
	path.AddPath(&scratch, &vector.AddPathOptions{GeoM: s.geoM})
	return nil
}

// geoMIsIdentity reports whether s.geoM is the identity matrix. A zero GeoM
// is identity by construction.
func (s *TextLayoutSnapshot) geoMIsIdentity() bool {
	return s.geoM.Element(0, 0) == 1 && s.geoM.Element(0, 1) == 0 && s.geoM.Element(0, 2) == 0 &&
		s.geoM.Element(1, 0) == 0 && s.geoM.Element(1, 1) == 1 && s.geoM.Element(1, 2) == 0
}

// realize verifies that face matches the snapshot and returns the glyphs the
// face lays out for the snapshot's text.
func (s *TextLayoutSnapshot) realize(face Face) ([]LazyGlyph, error) {
	if err := s.checkFace(face); err != nil {
		return nil, err
	}

	var lazy []LazyGlyph
	var scratch []LazyGlyph
	var realizeErr error
	layoutLines(s.text, face, &s.layout, func(line string, indexOffset int, originX, originY float64, info lineLayoutInfo) {
		if realizeErr != nil {
			return
		}
		scratch = face.appendLazyGlyphsForLine(scratch[:0], line, indexOffset, originX, originY, nil)
		lazy = append(lazy, scratch...)
	})
	if err := realizeErr; err != nil {
		return nil, err
	}
	if len(lazy) != len(s.glyphs) {
		return nil, snapshotErrorf("face", "the face produces %d glyphs, but the snapshot has %d", len(lazy), len(s.glyphs))
	}
	for i := range lazy {
		lg := &lazy[i]
		sg := &s.glyphs[i]
		switch {
		case lg.StartIndexInBytes != sg.StartIndexInBytes:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].startIndexInBytes", i), "the face produces %d, but the snapshot has %d", lg.StartIndexInBytes, sg.StartIndexInBytes)
		case lg.EndIndexInBytes != sg.EndIndexInBytes:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].endIndexInBytes", i), "the face produces %d, but the snapshot has %d", lg.EndIndexInBytes, sg.EndIndexInBytes)
		case lg.GID != sg.GID:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].gid", i), "the face produces %d, but the snapshot has %d", lg.GID, sg.GID)
		case lg.ImageBounds != sg.ImageBounds:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].imageBounds", i), "the face produces %v, but the snapshot has %v", lg.ImageBounds, sg.ImageBounds)
		case lg.OriginX != sg.OriginX || lg.OriginY != sg.OriginY:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].origin", i), "the face produces (%v, %v), but the snapshot has (%v, %v)", lg.OriginX, lg.OriginY, sg.OriginX, sg.OriginY)
		case lg.OriginOffsetX != sg.OriginOffsetX || lg.OriginOffsetY != sg.OriginOffsetY:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].originOffset", i), "the face produces (%v, %v), but the snapshot has (%v, %v)", lg.OriginOffsetX, lg.OriginOffsetY, sg.OriginOffsetX, sg.OriginOffsetY)
		case lg.AdvanceX != sg.AdvanceX || lg.AdvanceY != sg.AdvanceY:
			return nil, snapshotErrorf(fmt.Sprintf("glyphs[%d].advance", i), "the face produces (%v, %v), but the snapshot has (%v, %v)", lg.AdvanceX, lg.AdvanceY, sg.AdvanceX, sg.AdvanceY)
		}
	}
	return lazy, nil
}

// checkFace reports whether face matches the snapshot's recorded face.
func (s *TextLayoutSnapshot) checkFace(face Face) error {
	if face == nil {
		return snapshotErrorf("face", "no face is given")
	}
	got, err := face.fingerprint()
	if err != nil {
		return err
	}
	if !got.Equal(s.fingerprint) {
		return snapshotErrorf("face", "the face %s does not match the snapshot's face %s", got, s.fingerprint)
	}
	return nil
}

// MarshalBinary encodes the snapshot into a self-describing binary form.
//
// The encoding is versioned and checksummed. It contains only plain data:
// decoding never invokes user code and never interprets the payload as a
// program.
func (s *TextLayoutSnapshot) MarshalBinary() ([]byte, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}

	var body bytes.Buffer
	w := &snapshotWriter{w: &body}
	w.writeString("text", s.text)
	w.writeUint64("direction", uint64(s.direction))
	w.writeUint64("primaryAlign", uint64(s.layout.PrimaryAlign))
	w.writeUint64("secondaryAlign", uint64(s.layout.SecondaryAlign))
	w.writeFloat64("lineSpacing", s.layout.LineSpacing)
	for i := range 2 {
		for j := range 3 {
			w.writeFloat64(fmt.Sprintf("geoM[%d]", i*3+j), s.geoM.Element(i, j))
		}
	}
	w.writeMetrics("metrics", &s.metrics)
	w.writeFloat64("width", s.width)
	w.writeFloat64("height", s.height)
	w.writeHash("fingerprint.hash", s.fingerprint.Hash)
	w.writeString("fingerprint.description", s.fingerprint.Description)

	w.writeUint64("lines.count", uint64(len(s.lines)))
	for i := range s.lines {
		line := &s.lines[i]
		path := fmt.Sprintf("lines[%d]", i)
		w.writeUint64(path+".indexInBytes", uint64(line.IndexInBytes))
		w.writeUint64(path+".endIndexInBytes", uint64(line.EndIndexInBytes))
		w.writeUint64(path+".lineBreakLen", uint64(line.LineBreakLen))
		w.writeFloat64(path+".advance", line.Advance)
		w.writeFloat64(path+".baselineX", line.BaselineX)
		w.writeFloat64(path+".baselineY", line.BaselineY)
	}

	w.writeUint64("glyphs.count", uint64(len(s.glyphs)))
	for i := range s.glyphs {
		g := &s.glyphs[i]
		path := fmt.Sprintf("glyphs[%d]", i)
		w.writeUint64(path+".startIndexInBytes", uint64(g.StartIndexInBytes))
		w.writeUint64(path+".endIndexInBytes", uint64(g.EndIndexInBytes))
		w.writeUint64(path+".gid", uint64(g.GID))
		w.writeUint64(path+".imageBounds.minX", uint64(int64(g.ImageBounds.Min.X)))
		w.writeUint64(path+".imageBounds.minY", uint64(int64(g.ImageBounds.Min.Y)))
		w.writeUint64(path+".imageBounds.maxX", uint64(int64(g.ImageBounds.Max.X)))
		w.writeUint64(path+".imageBounds.maxY", uint64(int64(g.ImageBounds.Max.Y)))
		var flags uint64
		if g.HasImage {
			flags |= 1
		}
		if g.Colored {
			flags |= 2
		}
		w.writeUint64(path+".flags", flags)
		w.writeFloat64(path+".originX", g.OriginX)
		w.writeFloat64(path+".originY", g.OriginY)
		w.writeFloat64(path+".originOffsetX", g.OriginOffsetX)
		w.writeFloat64(path+".originOffsetY", g.OriginOffsetY)
		w.writeFloat64(path+".advanceX", g.AdvanceX)
		w.writeFloat64(path+".advanceY", g.AdvanceY)
	}
	if w.err != nil {
		return nil, w.err
	}

	payload := body.Bytes()
	if len(payload) > maxTextLayoutSnapshotBytes {
		return nil, snapshotErrorf("data", "the encoded snapshot is %d bytes, which exceeds the limit %d", len(payload), maxTextLayoutSnapshotBytes)
	}

	var out bytes.Buffer
	out.Grow(len(snapshotMagic) + 4 + 8 + len(payload) + sha256.Size)
	out.Write(snapshotMagic[:])
	var header [12]byte
	binary.LittleEndian.PutUint32(header[0:4], TextLayoutSnapshotVersion)
	binary.LittleEndian.PutUint64(header[4:12], uint64(len(payload)))
	out.Write(header[:])
	out.Write(payload)
	sum := sha256.Sum256(payload)
	out.Write(sum[:])
	return out.Bytes(), nil
}

// DecodeTextLayoutSnapshot decodes a snapshot encoded by
// [TextLayoutSnapshot.MarshalBinary].
//
// DecodeTextLayoutSnapshot rejects unknown versions, truncated data, corrupt
// checksums, non-finite numbers, and indices outside the text, returning a
// [TextLayoutSnapshotError] identifying the offending field. The decoded
// snapshot does not alias data.
func DecodeTextLayoutSnapshot(data []byte) (*TextLayoutSnapshot, error) {
	const minLen = 8 + 4 + 8 + sha256.Size
	if len(data) < minLen {
		return nil, snapshotErrorf("data", "the data is %d bytes, shorter than the %d-byte header", len(data), minLen)
	}
	if !bytes.Equal(data[:8], snapshotMagic[:]) {
		return nil, snapshotErrorf("magic", "not a text layout snapshot")
	}
	version := binary.LittleEndian.Uint32(data[8:12])
	if version != TextLayoutSnapshotVersion {
		return nil, snapshotErrorf("version", "unsupported version %d (supported: %d)", version, TextLayoutSnapshotVersion)
	}
	payloadLen := binary.LittleEndian.Uint64(data[12:20])
	if payloadLen > maxTextLayoutSnapshotBytes {
		return nil, snapshotErrorf("data", "the payload is %d bytes, which exceeds the limit %d", payloadLen, maxTextLayoutSnapshotBytes)
	}
	if uint64(len(data)-minLen) != payloadLen {
		return nil, snapshotErrorf("data", "the payload length is %d, but %d bytes are available", payloadLen, len(data)-minLen)
	}
	payload := data[20 : 20+payloadLen]
	wantSum := data[20+payloadLen:]
	gotSum := sha256.Sum256(payload)
	if !bytes.Equal(wantSum, gotSum[:]) {
		return nil, snapshotErrorf("checksum", "the payload checksum does not match")
	}

	r := &snapshotReader{data: payload}
	s := &TextLayoutSnapshot{}
	s.text = r.readString("text", maxTextLayoutSnapshotTextBytes)
	if r.err == nil && !utf8.ValidString(s.text) {
		return nil, snapshotErrorf("text", "text is not valid UTF-8")
	}
	direction := r.readUint64("direction")
	s.direction = Direction(direction)
	s.layout.PrimaryAlign = Align(r.readUint64("primaryAlign"))
	s.layout.SecondaryAlign = Align(r.readUint64("secondaryAlign"))
	s.layout.LineSpacing = r.readFloat64("lineSpacing")
	var geoM ebiten.GeoM
	for i := range 2 {
		for j := range 3 {
			geoM.SetElement(i, j, r.readFloat64(fmt.Sprintf("geoM[%d]", i*3+j)))
		}
	}
	s.geoM = geoM
	r.readMetrics("metrics", &s.metrics)
	s.width = r.readFloat64("width")
	s.height = r.readFloat64("height")
	s.fingerprint.Hash = r.readHash("fingerprint.hash")
	s.fingerprint.Description = r.readString("fingerprint.description", maxTextLayoutSnapshotDescriptionBytes)

	lineCount := r.readCount("lines.count", maxTextLayoutSnapshotLines, snapshotEncodedLineBytes)
	if r.err == nil {
		s.lines = make([]TextLayoutSnapshotLine, lineCount)
		for i := range s.lines {
			line := &s.lines[i]
			path := fmt.Sprintf("lines[%d]", i)
			line.IndexInBytes = int(r.readInt64(path + ".indexInBytes"))
			line.EndIndexInBytes = int(r.readInt64(path + ".endIndexInBytes"))
			line.LineBreakLen = int(r.readInt64(path + ".lineBreakLen"))
			line.Advance = r.readFloat64(path + ".advance")
			line.BaselineX = r.readFloat64(path + ".baselineX")
			line.BaselineY = r.readFloat64(path + ".baselineY")
		}
	}
	glyphCount := r.readCount("glyphs.count", maxTextLayoutSnapshotGlyphs, snapshotEncodedGlyphBytes)
	if r.err == nil {
		s.glyphs = make([]TextLayoutSnapshotGlyph, glyphCount)
		for i := range s.glyphs {
			g := &s.glyphs[i]
			path := fmt.Sprintf("glyphs[%d]", i)
			g.StartIndexInBytes = int(r.readInt64(path + ".startIndexInBytes"))
			g.EndIndexInBytes = int(r.readInt64(path + ".endIndexInBytes"))
			g.GID = uint32(r.readUint64(path + ".gid"))
			minX, minXOK := decodeSnapshotInt(r.readUint64(path + ".imageBounds.minX"))
			minY, minYOK := decodeSnapshotInt(r.readUint64(path + ".imageBounds.minY"))
			maxX, maxXOK := decodeSnapshotInt(r.readUint64(path + ".imageBounds.maxX"))
			maxY, maxYOK := decodeSnapshotInt(r.readUint64(path + ".imageBounds.maxY"))
			if !minXOK || !minYOK || !maxXOK || !maxYOK {
				return nil, snapshotErrorf(path+".imageBounds", "an image bounds coordinate is out of the platform int range")
			}
			g.ImageBounds = image.Rect(minX, minY, maxX, maxY)
			if g.ImageBounds != (image.Rectangle{
				Min: image.Pt(minX, minY),
				Max: image.Pt(maxX, maxY),
			}) {
				return nil, snapshotErrorf(path+".imageBounds", "the rectangle (%d, %d, %d, %d) overflows", minX, minY, maxX, maxY)
			}
			flags := r.readUint64(path + ".flags")
			g.HasImage = flags&1 != 0
			g.Colored = flags&2 != 0
			if flags&^3 != 0 {
				return nil, snapshotErrorf(path+".flags", "unknown flag bits %#x", flags)
			}
			g.OriginX = r.readFloat64(path + ".originX")
			g.OriginY = r.readFloat64(path + ".originY")
			g.OriginOffsetX = r.readFloat64(path + ".originOffsetX")
			g.OriginOffsetY = r.readFloat64(path + ".originOffsetY")
			g.AdvanceX = r.readFloat64(path + ".advanceX")
			g.AdvanceY = r.readFloat64(path + ".advanceY")
		}
	}
	if r.err != nil {
		return nil, r.err
	}
	if r.remaining() != 0 {
		return nil, snapshotErrorf("data", "%d trailing bytes after the snapshot fields", r.remaining())
	}
	if err := s.validateDecoded(); err != nil {
		return nil, err
	}
	return s, nil
}

// validate checks the snapshot's invariants before encoding.
func (s *TextLayoutSnapshot) validate() error {
	if s == nil {
		return snapshotErrorf("snapshot", "the snapshot is nil")
	}
	if len(s.text) > maxTextLayoutSnapshotTextBytes {
		return snapshotErrorf("text", "text is %d bytes, which exceeds the limit %d", len(s.text), maxTextLayoutSnapshotTextBytes)
	}
	if !utf8.ValidString(s.text) {
		return snapshotErrorf("text", "text is not valid UTF-8")
	}
	if !s.direction.isValid() {
		return snapshotErrorf("direction", "invalid direction %d", s.direction)
	}
	if err := validateLayoutOptions("options", &s.layout); err != nil {
		return err
	}
	if err := validateGeoM("geoM", &s.geoM); err != nil {
		return err
	}
	if err := validateMetrics("metrics", &s.metrics); err != nil {
		return err
	}
	if len(s.lines) > maxTextLayoutSnapshotLines {
		return snapshotErrorf("lines", "%d lines exceed the limit %d", len(s.lines), maxTextLayoutSnapshotLines)
	}
	if len(s.glyphs) > maxTextLayoutSnapshotGlyphs {
		return snapshotErrorf("glyphs", "%d glyphs exceed the limit %d", len(s.glyphs), maxTextLayoutSnapshotGlyphs)
	}
	for i := range s.lines {
		if err := validateLine(s.text, i, &s.lines[i]); err != nil {
			return err
		}
	}
	for i := range s.glyphs {
		if err := validateGlyph(s.text, i, &s.glyphs[i]); err != nil {
			return err
		}
	}
	return nil
}

// validateDecoded checks invariants that an attacker or a corrupt writer could
// violate. It reuses the build-time field validation.
func (s *TextLayoutSnapshot) validateDecoded() error {
	if !s.direction.isValid() {
		return snapshotErrorf("direction", "invalid direction %d", s.direction)
	}
	if err := validateLayoutOptions("options", &s.layout); err != nil {
		return err
	}
	if err := validateGeoM("geoM", &s.geoM); err != nil {
		return err
	}
	if err := validateMetrics("metrics", &s.metrics); err != nil {
		return err
	}
	for _, v := range [...]struct {
		name  string
		value float64
	}{
		{"width", s.width},
		{"height", s.height},
	} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return snapshotErrorf(v.name, "value must be finite, but was %v", v.value)
		}
	}
	if !utf8.ValidString(s.fingerprint.Description) {
		return snapshotErrorf("fingerprint.description", "description is not valid UTF-8")
	}
	if len(s.lines) == 0 && len(s.glyphs) != 0 {
		return snapshotErrorf("glyphs", "a snapshot without lines cannot have glyphs")
	}
	if len(s.text) > 0 && len(s.lines) == 0 {
		return snapshotErrorf("lines", "non-empty text has no lines")
	}
	for i := range s.lines {
		if err := validateLine(s.text, i, &s.lines[i]); err != nil {
			return err
		}
	}
	for i := range s.glyphs {
		if err := validateGlyph(s.text, i, &s.glyphs[i]); err != nil {
			return err
		}
	}
	return nil
}

func validateLine(text string, index int, line *TextLayoutSnapshotLine) error {
	path := fmt.Sprintf("lines[%d]", index)
	if line.IndexInBytes < 0 || line.IndexInBytes > len(text) {
		return snapshotErrorf(path+".indexInBytes", "%d is out of range [0, %d]", line.IndexInBytes, len(text))
	}
	if line.EndIndexInBytes < line.IndexInBytes || line.EndIndexInBytes > len(text) {
		return snapshotErrorf(path+".endIndexInBytes", "%d is out of range [%d, %d]", line.EndIndexInBytes, line.IndexInBytes, len(text))
	}
	if line.LineBreakLen < 0 || line.EndIndexInBytes+line.LineBreakLen > len(text) {
		return snapshotErrorf(path+".lineBreakLen", "%d is out of range", line.LineBreakLen)
	}
	if line.IndexInBytes != line.EndIndexInBytes && line.IndexInBytes < len(text) && !utf8.RuneStart(text[line.IndexInBytes]) {
		return snapshotErrorf(path+".indexInBytes", "%d is not a rune boundary", line.IndexInBytes)
	}
	if line.EndIndexInBytes < len(text) && !utf8.RuneStart(text[line.EndIndexInBytes]) {
		return snapshotErrorf(path+".endIndexInBytes", "%d is not a rune boundary", line.EndIndexInBytes)
	}
	for _, v := range [...]struct {
		name  string
		value float64
	}{
		{"advance", line.Advance},
		{"baselineX", line.BaselineX},
		{"baselineY", line.BaselineY},
	} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return snapshotErrorf(path+"."+v.name, "value must be finite, but was %v", v.value)
		}
	}
	return nil
}

func validateGlyph(text string, index int, g *TextLayoutSnapshotGlyph) error {
	path := fmt.Sprintf("glyphs[%d]", index)
	if g.StartIndexInBytes < 0 || g.StartIndexInBytes > len(text) {
		return snapshotErrorf(path+".startIndexInBytes", "%d is out of range [0, %d]", g.StartIndexInBytes, len(text))
	}
	if g.EndIndexInBytes < g.StartIndexInBytes || g.EndIndexInBytes > len(text) {
		return snapshotErrorf(path+".endIndexInBytes", "%d is out of range [%d, %d]", g.EndIndexInBytes, g.StartIndexInBytes, len(text))
	}
	if g.ImageBounds.Dx() < 0 || g.ImageBounds.Dy() < 0 {
		return snapshotErrorf(path+".imageBounds", "the rectangle %v is inverted", g.ImageBounds)
	}
	if !g.HasImage && g.Colored {
		return snapshotErrorf(path+".colored", "a glyph without an image cannot be colored")
	}
	for _, v := range [...]struct {
		name  string
		value float64
	}{
		{"originX", g.OriginX},
		{"originY", g.OriginY},
		{"originOffsetX", g.OriginOffsetX},
		{"originOffsetY", g.OriginOffsetY},
		{"advanceX", g.AdvanceX},
		{"advanceY", g.AdvanceY},
	} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return snapshotErrorf(path+"."+v.name, "value must be finite, but was %v", v.value)
		}
	}
	return nil
}

// snapshotWriter writes a length-delimited byte stream. The first error stops
// all subsequent writes.
type snapshotWriter struct {
	w   *bytes.Buffer
	err error
}

func (w *snapshotWriter) writeUint64(path string, v uint64) {
	if w.err != nil {
		return
	}
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	if _, err := w.w.Write(b[:]); err != nil {
		w.err = &TextLayoutSnapshotError{Path: path, Message: "write failed", Err: err}
	}
}

func (w *snapshotWriter) writeFloat64(path string, v float64) {
	if w.err != nil {
		return
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		w.err = snapshotErrorf(path, "value must be finite, but was %v", v)
		return
	}
	w.writeUint64(path, math.Float64bits(v))
}

func (w *snapshotWriter) writeString(path, s string) {
	if w.err != nil {
		return
	}
	w.writeUint64(path+".len", uint64(len(s)))
	if w.err != nil {
		return
	}
	if _, err := w.w.WriteString(s); err != nil {
		w.err = &TextLayoutSnapshotError{Path: path, Message: "write failed", Err: err}
	}
}

func (w *snapshotWriter) writeHash(path string, h [32]byte) {
	if w.err != nil {
		return
	}
	if _, err := w.w.Write(h[:]); err != nil {
		w.err = &TextLayoutSnapshotError{Path: path, Message: "write failed", Err: err}
	}
}

func (w *snapshotWriter) writeMetrics(path string, m *Metrics) {
	w.writeFloat64(path+".hLineGap", m.HLineGap)
	w.writeFloat64(path+".hAscent", m.HAscent)
	w.writeFloat64(path+".hDescent", m.HDescent)
	w.writeFloat64(path+".vLineGap", m.VLineGap)
	w.writeFloat64(path+".vAscent", m.VAscent)
	w.writeFloat64(path+".vDescent", m.VDescent)
	w.writeFloat64(path+".xHeight", m.XHeight)
	w.writeFloat64(path+".capHeight", m.CapHeight)
}

// snapshotReader reads a length-delimited byte stream. The first error stops
// all subsequent reads and is reported by err.
type snapshotReader struct {
	data []byte
	off  int
	err  error
}

var errSnapshotTruncated = errors.New("unexpected end of data")

func (r *snapshotReader) remaining() int {
	return len(r.data) - r.off
}

func (r *snapshotReader) fail(path string, err error) {
	if r.err == nil {
		r.err = &TextLayoutSnapshotError{Path: path, Message: "cannot read the snapshot", Err: err}
	}
}

func (r *snapshotReader) readUint64(path string) uint64 {
	if r.err != nil {
		return 0
	}
	if r.remaining() < 8 {
		r.fail(path, errSnapshotTruncated)
		return 0
	}
	v := binary.LittleEndian.Uint64(r.data[r.off : r.off+8])
	r.off += 8
	return v
}

func (r *snapshotReader) readInt64(path string) int64 {
	return int64(r.readUint64(path))
}

func (r *snapshotReader) readFloat64(path string) float64 {
	v := math.Float64frombits(r.readUint64(path))
	if r.err != nil {
		return 0
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		r.err = snapshotErrorf(path, "value must be finite, but was %v", v)
		return 0
	}
	return v
}

func (r *snapshotReader) readString(path string, limit int) string {
	n := r.readUint64(path + ".len")
	if r.err != nil {
		return ""
	}
	if n > uint64(limit) {
		r.err = snapshotErrorf(path, "length %d exceeds the limit %d", n, limit)
		return ""
	}
	if n > uint64(r.remaining()) {
		r.fail(path, errSnapshotTruncated)
		return ""
	}
	s := string(r.data[r.off : r.off+int(n)])
	r.off += int(n)
	return s
}

func (r *snapshotReader) readHash(path string) [32]byte {
	var h [32]byte
	if r.err != nil {
		return h
	}
	if r.remaining() < 32 {
		r.fail(path, errSnapshotTruncated)
		return h
	}
	copy(h[:], r.data[r.off:r.off+32])
	r.off += 32
	return h
}

// readCount reads an element count. entryBytes is the encoded size of one
// element; the count must fit in the remaining data, so a corrupt count cannot
// force an allocation larger than the payload.
func (r *snapshotReader) readCount(path string, limit, entryBytes int) int {
	n := r.readUint64(path)
	if r.err != nil {
		return 0
	}
	if n > uint64(limit) {
		r.err = snapshotErrorf(path, "count %d exceeds the limit %d", n, limit)
		return 0
	}
	if n > uint64(r.remaining()/entryBytes) {
		r.fail(path, errSnapshotTruncated)
		return 0
	}
	return int(n)
}

func (r *snapshotReader) readMetrics(path string, m *Metrics) {
	m.HLineGap = r.readFloat64(path + ".hLineGap")
	m.HAscent = r.readFloat64(path + ".hAscent")
	m.HDescent = r.readFloat64(path + ".hDescent")
	m.VLineGap = r.readFloat64(path + ".vLineGap")
	m.VAscent = r.readFloat64(path + ".vAscent")
	m.VDescent = r.readFloat64(path + ".vDescent")
	m.XHeight = r.readFloat64(path + ".xHeight")
	m.CapHeight = r.readFloat64(path + ".capHeight")
}

// decodeSnapshotInt decodes v, which was written from int64, back into an int.
// It reports false when v does not fit in the platform int range.
func decodeSnapshotInt(v uint64) (int, bool) {
	n := int64(v)
	if int64(int(n)) != n {
		return 0, false
	}
	return int(n), true
}
