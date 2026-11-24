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
	"errors"
	"fmt"
	"slices"

	glanguage "github.com/go-text/typesetting/language"
	"golang.org/x/text/language"

	"github.com/hajimehoshi/ebiten/v2/text/v2/internal/textutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

var _ Face = (*TextFontFallbackResolver)(nil)

// maxFallbackCandidates bounds the length of a resolver's fallback chain.
// A longer chain is rejected at construction: font collections with hundreds
// of entries are almost always a configuration mistake, and walking them for
// every unsupported cluster would make layout pathologically slow.
const maxFallbackCandidates = 255

// Common fallback errors. They are wrapped with context (the face and the
// cluster) when reported, so errors.Is remains usable.
var (
	// ErrEmptyFallbackChain is returned when a resolver is created without a
	// primary face or without any fallback candidate.
	ErrEmptyFallbackChain = errors.New("text: empty font fallback chain")

	// ErrFaceUnavailable reports that a fallback face could not be used
	// (for example, its underlying font resource was closed). The resolver
	// skips the face and continues with the next candidate instead of
	// failing the whole layout.
	ErrFaceUnavailable = errors.New("text: fallback face is unavailable")
)

// FallbackCandidate is one entry of a [TextFontFallbackResolver]'s fallback
// chain. Candidates are tried in chain order after the primary face.
//
// A candidate matches clusters of the given script and language. Either field
// may be left at its zero value to act as a wildcard for that dimension:
// an empty script matches every script (including emoji and Common/Inherited
// clusters), and the Und language tag matches every language.
//
// Use [NewScriptFallbackCandidate] or [NewEmojiFallbackCandidate] instead of
// constructing a FallbackCandidate directly.
type FallbackCandidate struct {
	face     Face
	script   glanguage.Script
	language language.Tag
	emoji    bool
}

// NewScriptFallbackCandidate creates a fallback candidate for clusters of the
// given ISO 15924 script code (e.g. "Arab", "Hani", "Latn"). The language tag
// may be [language.Und] to match every language.
//
// The candidate's face must have the same direction as the resolver's
// primary face; otherwise [NewTextFontFallbackResolver] rejects it.
func NewScriptFallbackCandidate(scriptCode string, lang language.Tag, face Face) (FallbackCandidate, error) {
	if face == nil {
		return FallbackCandidate{}, errors.New("text: a fallback candidate's face must not be nil")
	}
	s, err := glanguage.ParseScript(scriptCode)
	if err != nil {
		return FallbackCandidate{}, fmt.Errorf("text: invalid script code %q: %w", scriptCode, err)
	}
	return FallbackCandidate{face: face, script: s, language: lang}, nil
}

// NewEmojiFallbackCandidate creates a fallback candidate for emoji clusters:
// clusters containing an Extended_Pictographic codepoint, such as a single
// emoji, an emoji with a variation selector, or a ZWJ emoji sequence.
//
// The whole grapheme cluster is sent to the face, so sequences like
// "🫱🏼‍🫲🏽" are never split across faces.
func NewEmojiFallbackCandidate(face Face) (FallbackCandidate, error) {
	if face == nil {
		return FallbackCandidate{}, errors.New("text: a fallback candidate's face must not be nil")
	}
	return FallbackCandidate{face: face, emoji: true}, nil
}

// NewGeneralFallbackCandidate creates a catch-all candidate that matches every
// cluster not handled by an earlier candidate. It is typically placed last in
// the chain.
func NewGeneralFallbackCandidate(face Face) (FallbackCandidate, error) {
	if face == nil {
		return FallbackCandidate{}, errors.New("text: a fallback candidate's face must not be nil")
	}
	return FallbackCandidate{face: face}, nil
}

// FaceHealthChecker is an optional interface implemented by faces that can
// become unusable after construction, for example because their underlying
// font resource was closed. When a face in a fallback chain implements it,
// [TextFontFallbackResolver] calls CheckFace before selecting the face and
// skips it on error, recording the error through the resolver's error
// handler instead of crashing.
type FaceHealthChecker interface {
	CheckFace() error
}

// languageFace is implemented by faces carrying an explicit BCP 47 language
// tag, such as [GoTextFace]. It lets a language-specific candidate match only
// when the resolver's primary face requests that language.
type languageFace interface {
	languageTag() language.Tag
}

// ResolvedCluster describes one grapheme cluster and the face selected to
// render it, as returned by [TextFontFallbackResolver.ResolveClusters].
type ResolvedCluster struct {
	// Text is the cluster's sub-string of the input line.
	Text string

	// StartIndexInBytes and EndIndexInBytes are the cluster's byte range
	// within the input line.
	StartIndexInBytes int
	EndIndexInBytes   int

	// Face is the face selected for this cluster: the primary face or one
	// of the fallback candidates.
	Face Face

	// FaceIndex is the selected face's position in the chain: 0 for the
	// primary face, otherwise 1 + the candidate index passed to
	// NewTextFontFallbackResolver.
	FaceIndex int

	// Metrics are the selected face's metrics, captured at resolution time.
	Metrics Metrics

	// Advance is the cluster's advance in pixels as shaped by the selected
	// face, along the face's primary direction.
	Advance float64
}

// fallbackSegment is a run of consecutive clusters assigned to one face.
type fallbackSegment struct {
	face      Face
	faceIndex int
	clusters  []graphemeCluster
}

// clusterPlan is the resolved per-cluster face assignment for one line.
type clusterPlan struct {
	clusters []graphemeCluster
	// faceIndex[i] is the selected chain index for clusters[i]: 0 is the
	// primary face, >=1 a fallback candidate.
	faceIndex []uint8
	// segments groups consecutive clusters sharing a face, in logical order.
	segments []fallbackSegment
}

// TextFontFallbackResolver is a [Face] that shapes each grapheme cluster with
// the first face in a priority chain that can render it.
//
// The primary face always has the highest priority: a cluster whose runes the
// primary face covers is never sent to a fallback face, even when a fallback
// candidate nominally matches the cluster's script. Fallback candidates are
// tried in chain order for the clusters the primary face misses.
//
// Unlike [MultiFace], which picks a face per rune, the resolver works on
// extended grapheme clusters (UAX #29). Combining-mark sequences, variation
// selectors and ZWJ emoji sequences stay on one face, so the selected font
// shapes them as a unit instead of rendering broken fragments.
//
// All the faces, including the primary face and every candidate face, must
// share the same [Direction]; the resolver's direction is the primary face's
// direction.
//
// A TextFontFallbackResolver must not be copied by value.
type TextFontFallbackResolver struct {
	primary Face
	chain   []Face
	// candidates[i-1] is the matching rule for chain[i] (chain[0] is the
	// primary face, which has no rule).
	candidates []FallbackCandidate

	segmenter graphemeSegmenter

	// planCache memoizes the per-line cluster plan. Its key is the line
	// text; a face's glyph availability is itself cached inside the face,
	// and callers must configure faces before building a resolver, as with
	// MultiFace.
	planCache *cache[string, *clusterPlan]

	// errHandler receives non-fatal fallback errors: unavailable faces,
	// candidates that fail their health check, and clusters no face covers.
	// When nil, such errors are ignored and the primary face is used as
	// the last resort.
	errHandler func(error)

	addr *TextFontFallbackResolver
}

// NewTextFontFallbackResolver creates a [TextFontFallbackResolver] with the
// given primary face and fallback candidate chain.
//
// It returns [ErrEmptyFallbackChain] when primary is nil, and an error when a
// candidate is invalid, the chain exceeds [maxFallbackCandidates], or the
// faces' directions disagree.
func NewTextFontFallbackResolver(primary Face, candidates ...FallbackCandidate) (*TextFontFallbackResolver, error) {
	if primary == nil {
		return nil, fmt.Errorf("%w: a primary face is required", ErrEmptyFallbackChain)
	}
	if len(candidates) > maxFallbackCandidates {
		return nil, fmt.Errorf("text: too many fallback candidates: %d > %d", len(candidates), maxFallbackCandidates)
	}

	d := primary.direction()
	chain := make([]Face, 0, len(candidates)+1)
	chain = append(chain, primary)
	seen := map[Face]struct{}{primary: {}}
	for i, c := range candidates {
		if c.face == nil {
			return nil, fmt.Errorf("%w: candidate %d has no face", ErrEmptyFallbackChain, i)
		}
		if c.face.direction() != d {
			return nil, errors.New("text: all the faces' directions must agree at NewTextFontFallbackResolver")
		}
		if _, dup := seen[c.face]; dup {
			return nil, fmt.Errorf("text: fallback face at index %d is already in the chain; duplicate faces are not allowed", i)
		}
		seen[c.face] = struct{}{}
		chain = append(chain, c.face)
	}
	// Reject nested resolvers that reach back into this chain: such a cycle
	// would make cluster selection recurse forever.
	for i := 1; i < len(chain); i++ {
		if nested, ok := chain[i].(*TextFontFallbackResolver); ok {
			for _, f := range chain {
				if nested.containsFace(f) {
					return nil, fmt.Errorf("text: fallback candidate %d contains a cyclic TextFontFallbackResolver", i-1)
				}
			}
		}
	}

	r := &TextFontFallbackResolver{
		primary:    primary,
		chain:      chain,
		candidates: slices.Clone(candidates),
		planCache:  newCache[string, *clusterPlan](256),
	}
	r.addr = r
	return r, nil
}

// SetErrorHandler installs a callback invoked for non-fatal fallback errors:
// a candidate failing its [FaceHealthChecker], a face that cannot render a
// cluster it was expected to cover, a cluster no face in the chain covers, or
// a face whose shaping call panics (for example because its resource was
// closed after the plan was built).
//
// Layout never panics on these conditions and never blocks: the failing face
// is skipped or contributes no glyphs, and the primary face remains the
// last-resort face. The callback itself must not block for long; pass nil to
// restore the default, which silently ignores these conditions.
func (r *TextFontFallbackResolver) SetErrorHandler(handler func(error)) {
	r.errHandler = handler
}

func (r *TextFontFallbackResolver) report(err error) {
	if r.errHandler != nil && err != nil {
		r.errHandler(err)
	}
}

// ResolveClusters splits the first line of text into extended grapheme
// clusters and resolves, for each one, the face that will actually render it
// together with that face's metrics and the cluster's shaped advance.
//
// The returned slice is in logical (source) order and is freshly allocated, so
// the caller may retain and mutate it. Only the first line is considered,
// matching the package's line-oriented layout APIs.
//
// ResolveClusters is concurrent-safe.
func (r *TextFontFallbackResolver) ResolveClusters(text string) []ResolvedCluster {
	r.copyCheck()
	firstLineLen := textutil.FirstLineLen(text)
	if firstLineLen == 0 {
		return nil
	}
	firstLine := text[:firstLineLen]
	plan := r.plan(firstLine)

	out := make([]ResolvedCluster, 0, len(plan.clusters))
	for i, c := range plan.clusters {
		faceIndex := int(plan.faceIndex[i])
		f := r.chain[faceIndex]
		clusterText := firstLine[c.start:c.end]
		out = append(out, ResolvedCluster{
			Text:              clusterText,
			StartIndexInBytes: c.start,
			EndIndexInBytes:   c.end,
			Face:              f,
			FaceIndex:         faceIndex,
			Metrics:           f.Metrics(),
			Advance:           f.advanceAt(clusterText, len(clusterText)),
		})
	}
	return out
}

// copyCheck panics if r was copied by value after NewTextFontFallbackResolver.
func (r *TextFontFallbackResolver) copyCheck() {
	if r.addr != r {
		panic("text: illegal use of non-zero TextFontFallbackResolver copied by value")
	}
}

// containsFace reports whether f is the resolver's primary face or reachable
// through a nested resolver in its chain. It is used at construction to reject
// cyclic fallback configurations.
func (r *TextFontFallbackResolver) containsFace(f Face) bool {
	for _, c := range r.chain {
		if c == f {
			return true
		}
		if nested, ok := c.(*TextFontFallbackResolver); ok && nested.containsFace(f) {
			return true
		}
	}
	return false
}

// Metrics implements Face. The metrics are the maximum over every face in the
// chain, so fallback glyphs never overflow the line box.
func (r *TextFontFallbackResolver) Metrics() Metrics {
	r.copyCheck()
	var m Metrics
	for _, f := range r.chain {
		fm := r.safeMetrics(f)
		m.HLineGap = max(m.HLineGap, fm.HLineGap)
		m.HAscent = max(m.HAscent, fm.HAscent)
		m.HDescent = max(m.HDescent, fm.HDescent)
		m.VLineGap = max(m.VLineGap, fm.VLineGap)
		m.VAscent = max(m.VAscent, fm.VAscent)
		m.VDescent = max(m.VDescent, fm.VDescent)
		m.XHeight = max(m.XHeight, fm.XHeight)
		m.CapHeight = max(m.CapHeight, fm.CapHeight)
	}
	return m
}

// safeMetrics calls f.Metrics, treating a panic from a closed or broken face
// as an unavailable-face error and falling back to zero metrics. A panic in a
// fallback face must not take down the whole layout.
func (r *TextFontFallbackResolver) safeMetrics(f Face) (m Metrics) {
	defer func() {
		if rec := recover(); rec != nil {
			r.report(fmt.Errorf("%w: %v", ErrFaceUnavailable, rec))
			m = Metrics{}
		}
	}()
	return f.Metrics()
}

// hasGlyph implements Face. It reports whether any face in the chain covers r.
func (r *TextFontFallbackResolver) hasGlyph(rn rune) bool {
	for _, f := range r.chain {
		if r.faceAvailable(f) && safeHasGlyph(f, rn) {
			return true
		}
	}
	return false
}

// faceAvailable runs the optional [FaceHealthChecker] check, treating a panic
// as unavailability.
func (r *TextFontFallbackResolver) faceAvailable(f Face) (ok bool) {
	defer func() {
		if rec := recover(); rec != nil {
			r.report(fmt.Errorf("%w: %v", ErrFaceUnavailable, rec))
			ok = false
		}
	}()
	if hc, ok := f.(FaceHealthChecker); ok {
		if err := hc.CheckFace(); err != nil {
			r.report(fmt.Errorf("%w: %w", ErrFaceUnavailable, err))
			return false
		}
	}
	return true
}

// safeHasGlyph calls f.hasGlyph, converting a panic into a false result.
func safeHasGlyph(f Face, r rune) (has bool) {
	defer func() {
		if rec := recover(); rec != nil {
			has = false
		}
	}()
	return f.hasGlyph(r)
}

// direction implements Face.
func (r *TextFontFallbackResolver) direction() Direction {
	return r.primary.direction()
}

// private implements Face.
func (r *TextFontFallbackResolver) private() {}

// plan resolves the face assignment for one line.
func (r *TextFontFallbackResolver) plan(line string) *clusterPlan {
	r.copyCheck()
	return r.planCache.getOrCreate(line, func() (*clusterPlan, bool) {
		return r.buildPlan(line), true
	})
}

// buildPlan splits the line into grapheme clusters and assigns each cluster a
// chain index. The result is deterministic: repeated calls with the same line
// and the same faces produce identical assignments and cache keys.
func (r *TextFontFallbackResolver) buildPlan(line string) *clusterPlan {
	var clusters []graphemeCluster
	clusters = r.segmenter.split(clusters, line)

	plan := &clusterPlan{
		clusters: clusters,
	}
	if len(clusters) == 0 {
		return plan
	}
	plan.faceIndex = make([]uint8, len(clusters))

	var segments []fallbackSegment
	segStart := 0
	prevIndex := -1
	for i, c := range clusters {
		idx := r.selectFace(line[c.start:c.end])
		plan.faceIndex[i] = uint8(idx)
		if prevIndex < 0 {
			prevIndex = idx
			segStart = i
			continue
		}
		if idx != prevIndex {
			segments = append(segments, r.makeSegment(plan, segStart, i, prevIndex))
			segStart = i
			prevIndex = idx
		}
	}
	segments = append(segments, r.makeSegment(plan, segStart, len(clusters), prevIndex))
	plan.segments = segments
	return plan
}

// makeSegment builds a fallbackSegment covering plan.clusters[start:end].
func (r *TextFontFallbackResolver) makeSegment(plan *clusterPlan, start, end, faceIndex int) fallbackSegment {
	return fallbackSegment{
		face:      r.chain[faceIndex],
		faceIndex: faceIndex,
		clusters:  plan.clusters[start:end],
	}
}

// selectFace returns the chain index that should render cluster.
//
// The primary face (index 0) wins whenever it covers the cluster's runes.
// Otherwise candidates are tried in order, filtered by their matching rule
// (emoji flag, script, language). A candidate whose face is unavailable or
// does not cover the cluster is skipped. If nothing matches, the primary face
// is selected as the last resort (it renders the font's missing-glyph form),
// and the error handler is notified.
func (r *TextFontFallbackResolver) selectFace(cluster string) int {
	if r.clusterCoveredBy(r.primary, cluster) {
		return 0
	}

	emojiCluster := clusterIsEmoji(cluster)
	clusterScript, _ := clusterScriptAndLanguage(cluster)
	primaryLang := r.primaryLanguage()

	for i, cand := range r.candidates {
		if !candidateMatches(cand, emojiCluster, clusterScript, primaryLang) {
			continue
		}
		f := r.chain[i+1]
		if !r.faceAvailable(f) {
			continue
		}
		if r.clusterCoveredBy(f, cluster) {
			return i + 1
		}
	}

	// No face covers the cluster. The primary face's missing-glyph rendering
	// is the stable last resort; never fail or loop.
	r.report(fmt.Errorf("text: no face in the fallback chain covers cluster %q", cluster))
	return 0
}

// primaryLanguage returns the primary face's language tag, or Und when the
// face does not carry one.
func (r *TextFontFallbackResolver) primaryLanguage() language.Tag {
	if lf, ok := r.primary.(languageFace); ok {
		return lf.languageTag()
	}
	return language.Und
}

// candidateMatches reports whether cand applies to a cluster with the given
// emoji flag, strong script and the primary face's language.
//
// An emoji candidate matches either a recognized emoji cluster (one containing
// an Extended_Pictographic codepoint, a keycap or a regional indicator) or a
// cluster whose script is Unknown (Private-Use symbols, pictographic glyphs in
// PUA fonts such as color-emoji test fonts): such clusters cannot be assigned
// a real script, so the emoji rule is their natural home.
//
// When the candidate specifies a language, it matches only if the primary
// face's language shares that base language. A candidate with the Und tag
// matches every language.
func candidateMatches(cand FallbackCandidate, emojiCluster bool, script glanguage.Script, primaryLang language.Tag) bool {
	if cand.emoji {
		return emojiCluster || script == glanguage.Unknown
	}
	if cand.script != 0 && cand.script != script {
		// Common/Inherited/Unknown clusters (spaces, punctuation, combining
		// marks, PUA symbols) follow the surrounding strong script: treat
		// them as matching any scripted candidate rather than only a Common
		// candidate, unless the cluster was claimed by an earlier emoji rule.
		if script.Strong() && script != glanguage.Unknown {
			return false
		}
	}
	if cand.language != (language.Tag{}) && !cand.language.IsRoot() {
		// A language-specific candidate matches only when the primary face's
		// language shares the same base language (e.g. zh-Hant and zh-Hans
		// share base "zh"). An Und primary language never matches a specific
		// candidate, so generic chains keep their script-only ordering.
		if primaryLang.IsRoot() {
			return false
		}
		candBase, candConf := cand.language.Base()
		primBase, primConf := primaryLang.Base()
		if candConf == language.No || primConf == language.No || candBase != primBase {
			return false
		}
	}
	return true
}

// clusterCoveredBy reports whether every non-default, non-line-break rune in
// cluster has a glyph in f. Control characters and ZWJ/join controls that
// legitimately produce no glyph are ignored, so an emoji ZWJ sequence still
// "covered" when the font maps the join control to nothing.
func (r *TextFontFallbackResolver) clusterCoveredBy(f Face, cluster string) bool {
	for _, ru := range cluster {
		if isIgnoredForCoverage(ru) {
			continue
		}
		if !safeHasGlyph(f, ru) {
			return false
		}
	}
	return true
}

// advanceAt implements Face.
//
// Each cluster is shaped by its selected face in isolation, so the line
// advance is the sum of cluster advances. A caret that falls strictly inside
// a cluster (between bytes of a multi-rune emoji sequence or combining mark)
// snaps to its leading edge, matching [AdvanceAt]'s cluster convention.
func (r *TextFontFallbackResolver) advanceAt(text string, indexInBytes int) float64 {
	firstLineLen := textutil.FirstLineLen(text)
	indexInBytes = min(indexInBytes, firstLineLen)
	if indexInBytes <= 0 || firstLineLen == 0 {
		return 0
	}
	firstLine := text[:firstLineLen]
	plan := r.plan(firstLine)

	rtl := r.direction() == DirectionRightToLeft

	// Each cluster's own advance; the total is the sum. A broken face's
	// advance is recovered as zero by safeAdvance.
	widths := make([]float64, len(plan.clusters))
	var total float64
	for i, c := range plan.clusters {
		f := r.chain[plan.faceIndex[i]]
		clusterText := firstLine[c.start:c.end]
		w := r.safeAdvance(f, clusterText)
		widths[i] = w
		total += w
	}

	// Find the cluster containing the caret. A caret at the end of the line
	// has no containing cluster.
	target := -1
	for i, c := range plan.clusters {
		if indexInBytes < c.end {
			target = i
			break
		}
	}
	if target < 0 {
		return total
	}

	// Snap a caret inside a cluster back to the cluster's leading byte. The
	// leading edge is the cluster's start in logical order for both
	// directions; the visual side differs and is handled below.
	if indexInBytes > plan.clusters[target].start {
		indexInBytes = plan.clusters[target].start
	}

	var before float64
	for i := 0; i < target; i++ {
		before += widths[i]
	}
	if rtl {
		// RTL lines grow toward the left: the leading edge of the target
		// cluster is reached after the clusters that follow it logically.
		return total - before
	}
	return before
}

// appendLazyGlyphsForLine implements Face.
func (r *TextFontFallbackResolver) appendLazyGlyphsForLine(glyphs []LazyGlyph, line string, indexOffset int, originX, originY float64, keepGlyph func(originX, originY float64) bool) []LazyGlyph {
	plan := r.plan(line)
	horizontal := r.direction().isHorizontal()

	// Segments are laid out in visual order, which reverses the logical
	// segments for an RTL line. Each segment is a contiguous sub-string of
	// the line, and the child face receives the absolute index offset.
	for _, segIdx := range r.segmentVisualOrder(plan) {
		seg := plan.segments[segIdx]
		f := seg.face
		start := seg.clusters[0].start
		end := seg.clusters[len(seg.clusters)-1].end
		sub := line[start:end]

		glyphs = r.safeAppendLazyGlyphs(glyphs, f, sub, indexOffset+start, originX, originY, keepGlyph)

		originX, originY = r.advanceOrigin(f, sub, originX, originY, horizontal)
	}
	return glyphs
}

// safeAppendLazyGlyphs calls f.appendLazyGlyphsForLine, recovering from a panic
// (for example, a face whose resource was closed after the plan was built). On
// panic the segment contributes no glyphs, the error is reported, and layout
// continues with the next segment instead of crashing the whole line.
func (r *TextFontFallbackResolver) safeAppendLazyGlyphs(glyphs []LazyGlyph, f Face, sub string, indexOffset int, originX, originY float64, keepGlyph func(originX, originY float64) bool) (result []LazyGlyph) {
	defer func() {
		if rec := recover(); rec != nil {
			r.report(fmt.Errorf("%w: shaping %q: %v", ErrFaceUnavailable, sub, rec))
			result = glyphs
		}
	}()
	return f.appendLazyGlyphsForLine(glyphs, sub, indexOffset, originX, originY, keepGlyph)
}

// safeAdvance calls f.advanceAt, recovering from a panic and returning zero so
// a broken face cannot halt layout.
func (r *TextFontFallbackResolver) safeAdvance(f Face, sub string) (a float64) {
	defer func() {
		if rec := recover(); rec != nil {
			r.report(fmt.Errorf("%w: measuring %q: %v", ErrFaceUnavailable, sub, rec))
			a = 0
		}
	}()
	return f.advanceAt(sub, len(sub))
}

// advanceOrigin moves the line origin by sub's advance along the primary axis.
func (r *TextFontFallbackResolver) advanceOrigin(f Face, sub string, originX, originY float64, horizontal bool) (float64, float64) {
	a := r.safeAdvance(f, sub)
	if horizontal {
		return originX + a, originY
	}
	return originX, originY + a
}

// segmentVisualOrder returns segment indices in layout order.
func (r *TextFontFallbackResolver) segmentVisualOrder(plan *clusterPlan) []int {
	n := len(plan.segments)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if r.direction() == DirectionRightToLeft {
		slices.Reverse(order)
	}
	return order
}

// appendVectorPathForLine implements Face.
func (r *TextFontFallbackResolver) appendVectorPathForLine(path *vector.Path, line string, originX, originY float64) {
	plan := r.plan(line)
	horizontal := r.direction().isHorizontal()
	for _, segIdx := range r.segmentVisualOrder(plan) {
		seg := plan.segments[segIdx]
		f := seg.face
		start := seg.clusters[0].start
		end := seg.clusters[len(seg.clusters)-1].end
		sub := line[start:end]
		r.safeAppendVectorPath(path, f, sub, originX, originY)
		originX, originY = r.advanceOrigin(f, sub, originX, originY, horizontal)
	}
}

// safeAppendVectorPath calls f.appendVectorPathForLine, recovering from a
// panic so one broken face cannot stop vector-path construction.
func (r *TextFontFallbackResolver) safeAppendVectorPath(path *vector.Path, f Face, sub string, originX, originY float64) {
	defer func() {
		if rec := recover(); rec != nil {
			r.report(fmt.Errorf("%w: vector path for %q: %v", ErrFaceUnavailable, sub, rec))
		}
	}()
	f.appendVectorPathForLine(path, sub, originX, originY)
}

// clusterIsEmoji reports whether cluster is an emoji cluster: it contains an
// Extended_Pictographic codepoint, a keycap combining enclosing mark, or a
// regional indicator (flag sequences). Emoji presentation (with U+FE0F) and
// ZWJ sequences all contain an Extended_Pictographic codepoint, so the flag
// covers the whole cluster.
func clusterIsEmoji(cluster string) bool {
	for _, r := range cluster {
		if isExtendedPictographic(r) || isEmojiKeycap(r) || isRegionalIndicator(r) {
			return true
		}
	}
	return false
}

// isEmojiKeycap reports whether r is U+20E3 COMBINING ENCLOSING KEYCAP, which
// turns a base digit, '#', or '*' into a keycap emoji sequence.
func isEmojiKeycap(r rune) bool {
	return r == 0x20E3
}

// isRegionalIndicator reports whether r is a Regional Indicator Symbol
// (U+1F1E6..U+1F1FF), used in flag emoji sequences.
func isRegionalIndicator(r rune) bool {
	return r >= 0x1F1E6 && r <= 0x1F1FF
}

// clusterScriptAndLanguage returns the UAX #24 script of the cluster's first
// strongly-scripted rune. When no rune has a strong script (punctuation-only
// clusters), Common is returned. A Private-Use cluster resolves to Unknown.
//
// Runes carry no language identity on their own, so the language component is
// always Und here; a candidate's language tag is an explicit matcher and is
// compared against the resolver primary face's language.
func clusterScriptAndLanguage(cluster string) (glanguage.Script, language.Tag) {
	for _, r := range cluster {
		s := glanguage.LookupScript(r)
		if s.Strong() {
			return s, language.Und
		}
	}
	return glanguage.Common, language.Und
}

// isIgnoredForCoverage reports whether r is expected to produce no glyph on
// its own and therefore must not make a cluster count as uncovered.
func isIgnoredForCoverage(r rune) bool {
	if r < 0x20 || (r >= 0x7f && r < 0xa0) {
		// C0 and C1 control characters.
		return true
	}
	switch r {
	case 0x200D, // ZERO WIDTH JOINER
		0x200C, // ZERO WIDTH NON-JOINER
		0xFE00, 0xFE01, 0xFE02, 0xFE03, 0xFE04, 0xFE05, 0xFE06, 0xFE07,
		0xFE08, 0xFE09, 0xFE0A, 0xFE0B, 0xFE0C, 0xFE0D, 0xFE0E, 0xFE0F: // variation selectors
		return true
	}
	return false
}
