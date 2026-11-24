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
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"math"

	"github.com/go-text/typesetting/font/opentype"
)

// FontFingerprint identifies the font resource and the face configuration a
// layout was computed with.
//
// Two faces that report the same fingerprint render and lay out text the same
// way at the same size, direction, language, and options. A snapshot records
// the fingerprint of its face so that replaying it with a different font is
// rejected instead of silently reinterpreting glyph IDs.
type FontFingerprint struct {
	// Hash is a SHA-256 digest over the font resource and the face
	// configuration.
	Hash [32]byte

	// Description summarizes the face in a human-readable form. It is meant
	// for diagnostics and is not used for comparison.
	Description string
}

// String returns a human-readable representation of f.
func (f FontFingerprint) String() string {
	return fmt.Sprintf("%s (%x)", f.Description, f.Hash[:8])
}

// Equal reports whether f and other identify the same face configuration.
func (f FontFingerprint) Equal(other FontFingerprint) bool {
	return f.Hash == other.Hash
}

// FaceFingerprint returns the fingerprint of face.
//
// The fingerprint covers the font resource's binary content when the face
// exposes it (GoTextFace and its sources) and the face configuration that
// affects layout: direction, size, language, script, variations, features,
// and the structure of composite faces.
func FaceFingerprint(face Face) (FontFingerprint, error) {
	if face == nil {
		return FontFingerprint{}, snapshotErrorf("face", "no face is given")
	}
	return face.fingerprint()
}

// fingerprintHasher accumulates a canonical, unambiguous byte stream.
type fingerprintHasher struct {
	hash.Hash
}

func newFingerprintHasher(kind string) *fingerprintHasher {
	h := &fingerprintHasher{Hash: sha256.New()}
	h.Hash.Write([]byte("ebiten:text:fingerprint:v1"))
	h.writeString(kind)
	return h
}

func (h *fingerprintHasher) writeString(s string) {
	var b [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(b[:], uint64(len(s)))
	h.Hash.Write(b[:n])
	h.Hash.Write([]byte(s))
}

func (h *fingerprintHasher) writeUint64(v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	h.Hash.Write(b[:])
}

func (h *fingerprintHasher) writeFloat64(v float64) {
	h.writeUint64(math.Float64bits(v))
}

func (h *fingerprintHasher) fingerprint(description string) FontFingerprint {
	var fp FontFingerprint
	copy(fp.Hash[:], h.Sum(nil))
	fp.Description = description
	return fp
}

// fontResourceHash returns a digest of every table in loader. The digest
// covers the whole binary as parsed, so any change to glyph outlines, layout
// tables, color tables, or embedded SVG documents changes the fingerprint.
//
// loader may be nil; the zero hash is returned then.
func fontResourceHash(loader *opentype.Loader) [32]byte {
	var out [32]byte
	if loader == nil {
		return out
	}
	h := newFingerprintHasher("fontresource")
	tags := loader.Tables()
	h.writeUint64(uint64(len(tags)))
	for _, tag := range tags {
		var tagBytes [4]byte
		binary.BigEndian.PutUint32(tagBytes[:], uint32(tag))
		h.Hash.Write(tagBytes[:])
		raw, err := loader.RawTable(tag)
		if err != nil {
			// Record the failure instead of silently producing a partial
			// digest, so two unreadable-but-different resources differ.
			h.writeString("error:" + err.Error())
			continue
		}
		var lenBytes [binary.MaxVarintLen64]byte
		n := binary.PutUvarint(lenBytes[:], uint64(len(raw)))
		h.Hash.Write(lenBytes[:n])
		h.Hash.Write(raw)
	}
	copy(out[:], h.Sum(nil))
	return out
}

// fingerprint implements Face.
func (g *GoTextFace) fingerprint() (FontFingerprint, error) {
	h := newFingerprintHasher("gotextface")
	description := "GoTextFace"
	if g.Source == nil {
		return FontFingerprint{}, snapshotErrorf("face.source", "GoTextFace has no source")
	}
	if !g.Direction.isValid() {
		return FontFingerprint{}, snapshotErrorf("face.direction", "invalid direction %d", g.Direction)
	}
	if math.IsNaN(g.Size) || math.IsInf(g.Size, 0) {
		return FontFingerprint{}, snapshotErrorf("face.size", "size must be finite, but was %v", g.Size)
	}

	h.Hash.Write(g.Source.resourceHash[:])
	h.writeString(g.Source.metadata.Family)
	h.writeFloat64(float64(g.Source.metadata.Weight))
	h.writeFloat64(float64(g.Source.metadata.Stretch))
	h.writeUint64(uint64(g.Source.metadata.Style))
	if g.Source.metadata.Family != "" {
		description += fmt.Sprintf(" %q", g.Source.metadata.Family)
	}
	h.writeUint64(uint64(g.Direction))
	h.writeFloat64(g.Size)
	h.writeString(g.Language.String())
	h.writeString(g.Script.String())
	h.writeString(g.variations.key)
	h.writeString(g.features.key)
	description += fmt.Sprintf(" size=%g direction=%d", g.Size, g.Direction)
	return h.fingerprint(description), nil
}

// goXFingerprintProbes is the rune set whose advances and bounds are hashed
// for GoXFace. A font.Face does not expose its binary content, so the
// fingerprint is derived from observable metrics over a broad script sample.
var goXFingerprintProbes = []rune{
	' ', '0', 'A', 'a', 'i', 'W', '~',
	'ä', 'ß', 'é', 'ñ', 'Ω', 'Ж', 'א', 'ش', 'अ', 'ก', 'あ', '漢', 'ア', '한', '😀',
	'́', '‍', '�',
}

// fingerprint implements Face.
func (g *GoXFace) fingerprint() (FontFingerprint, error) {
	g.copyCheck()

	h := newFingerprintHasher("goxface")
	m := g.Metrics()
	for _, v := range [...]float64{
		m.HLineGap, m.HAscent, m.HDescent,
		m.VLineGap, m.VAscent, m.VDescent,
		m.XHeight, m.CapHeight,
	} {
		h.writeFloat64(v)
	}
	for _, r := range goXFingerprintProbes {
		adv, ok := g.f.GlyphAdvance(r)
		if ok {
			h.writeUint64(1)
			h.writeUint64(uint64(int64(adv)))
		} else {
			h.writeUint64(0)
		}
		bounds, _, boundsOK := g.f.GlyphBounds(r)
		if boundsOK {
			h.writeUint64(1)
			h.writeUint64(uint64(int64(bounds.Min.X)))
			h.writeUint64(uint64(int64(bounds.Min.Y)))
			h.writeUint64(uint64(int64(bounds.Max.X)))
			h.writeUint64(uint64(int64(bounds.Max.Y)))
		} else {
			h.writeUint64(0)
		}
	}
	return h.fingerprint("GoXFace"), nil
}

// fingerprint implements Face.
func (m *MultiFace) fingerprint() (FontFingerprint, error) {
	if !m.direction().isValid() {
		return FontFingerprint{}, snapshotErrorf("face.direction", "invalid direction %d", m.direction())
	}
	h := newFingerprintHasher("multiface")
	h.writeUint64(uint64(m.direction()))
	for i, f := range m.faces {
		fp, err := f.fingerprint()
		if err != nil {
			return FontFingerprint{}, prefixSnapshotErrorPath(fmt.Sprintf("face.faces[%d]", i), err)
		}
		h.Hash.Write(fp.Hash[:])
		h.writeString(fp.Description)
	}
	return h.fingerprint(fmt.Sprintf("MultiFace(%d faces) direction=%d", len(m.faces), m.direction())), nil
}

// fingerprint implements Face.
func (l *LimitedFace) fingerprint() (FontFingerprint, error) {
	h := newFingerprintHasher("limitedface")
	fp, err := l.face.fingerprint()
	if err != nil {
		return FontFingerprint{}, prefixSnapshotErrorPath("face.face", err)
	}
	h.Hash.Write(fp.Hash[:])
	h.writeString(fp.Description)
	ranges := l.unicodeRanges.Ranges()
	h.writeUint64(uint64(len(ranges)))
	for _, r := range ranges {
		h.writeUint64(uint64(int64(r[0])))
		h.writeUint64(uint64(int64(r[1])))
	}
	return h.fingerprint(fmt.Sprintf("LimitedFace(%d ranges) over %s", len(ranges), fp.Description)), nil
}
