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

package shadermanifest

import (
	"bytes"
	"encoding/json"
	"sort"
)

// CurrentVersion is the manifest format version written by this package and the highest version it can
// read. A manifest with any other version is rejected with [CodeUnknownVersion].
const CurrentVersion = 1

// Manifest is the versioned, auditable inventory of precompiled shaders.
//
// The JSON representation is canonical: entries, artifacts, uniforms and macros are sorted, paths use
// forward slashes, and the document is tab-indented and ends with a newline. Re-generating an unchanged
// manifest therefore produces byte-identical output, which makes manifests safe to check in and to diff.
type Manifest struct {
	// Version is the format version. Currently always [CurrentVersion].
	Version int

	// Entries are the per-shader records, sorted by SourceID and then by the canonical macro key.
	Entries []*Spec
}

// Spec describes one shader source, possibly specialized by a set of macros, and everything precompiled
// for it.
type Spec struct {
	// SourceID is the Ebitengine shader source ID, the base32 string reported by shadercollector.
	SourceID string

	// SourcePath is the path to the Kage source, relative to the manifest root, using forward slashes.
	SourcePath string

	// SourceHash is the lowercase hex SHA-256 of the exact source bytes.
	SourceHash string

	// SourceSize is the size of the source in bytes.
	SourceSize int

	// Macros are the compile-time macros this entry is specialized for. A nil and an empty map both mean
	// the default (macro-free) variant. Keys are unique and the JSON form is a sorted object.
	Macros map[string]string

	// Uniforms is the shader's uniform layout, sorted by index.
	Uniforms []Uniform

	// TextureCount is the number of source textures the shader samples.
	TextureCount int

	// Artifacts are the generated artifacts, sorted by backend and then by role.
	Artifacts []*Artifact
}

// Uniform is one entry of a shader's uniform layout.
type Uniform struct {
	// Index is the uniform variable index assigned by Ebitengine.
	Index int

	// Name is the uniform's name.
	Name string

	// Type is the shaderir type rendered as a stable string (e.g. "vec4" or "[8]float").
	Type string

	// Size is the uniform's size in 32-bit dwords, as reported by the shader IR.
	Size int
}

// Artifact is one generated file for a backend and role.
type Artifact struct {
	// Backend is the backend the artifact targets.
	Backend Backend

	// Role is the artifact's role within its backend, e.g. "vertex".
	Role string

	// Path is the artifact file, relative to the manifest root, using forward slashes.
	Path string

	// Hash is the lowercase hex SHA-256 of the uncompressed artifact bytes.
	Hash string

	// Size is the size of the uncompressed artifact in bytes.
	Size int

	// Gzip reports whether the file at Path is gzip-compressed. The hash always covers the
	// uncompressed content, so verification is independent of the on-disk compression.
	Gzip bool
}

// The structs below mirror Manifest for JSON with the field ordering Ebitengine uses elsewhere
// (PascalCase names, tab indentation). Custom MarshalJSON enforces canonical ordering of every slice and
// map, which is what makes the serialized form stable across platforms and runs.

type jsonManifest struct {
	Version int         `json:"Version"`
	Entries []*jsonSpec `json:"Entries"`
}

type jsonSpec struct {
	SourceID     string            `json:"SourceID"`
	SourcePath   string            `json:"SourcePath"`
	SourceHash   string            `json:"SourceHash"`
	SourceSize   int               `json:"SourceSize"`
	Macros       map[string]string `json:"Macros"`
	Uniforms     []jsonUniform     `json:"Uniforms"`
	TextureCount int               `json:"TextureCount"`
	Artifacts    []*jsonArtifact   `json:"Artifacts"`
}

// ensureObject rewrites a nil slice/map as an empty, allocated one so empty collections render as JSON
// arrays/objects rather than null, keeping the document shape uniform across entries.
func ensureObject(m map[string]string) map[string]string {
	if m == nil {
		return map[string]string{}
	}
	return m
}

type jsonUniform struct {
	Index int    `json:"Index"`
	Name  string `json:"Name"`
	Type  string `json:"Type"`
	Size  int    `json:"Size"`
}

type jsonArtifact struct {
	Backend Backend `json:"Backend"`
	Role    string  `json:"Role"`
	Path    string  `json:"Path"`
	Hash    string  `json:"Hash"`
	Size    int     `json:"Size"`
	Gzip    bool    `json:"Gzip"`
}

// canonicalize sorts every collection in the manifest into its canonical order and clones macro maps so
// later mutations of the caller's maps cannot change the manifest's identity.
func (m *Manifest) canonicalize() {
	for _, s := range m.Entries {
		if len(s.Macros) > 0 {
			cloned := make(map[string]string, len(s.Macros))
			for k, v := range s.Macros {
				cloned[k] = v
			}
			s.Macros = cloned
		} else {
			s.Macros = nil
		}

		sort.Slice(s.Uniforms, func(i, j int) bool {
			return s.Uniforms[i].Index < s.Uniforms[j].Index
		})

		sort.Slice(s.Artifacts, func(i, j int) bool {
			if s.Artifacts[i].Backend != s.Artifacts[j].Backend {
				return s.Artifacts[i].Backend < s.Artifacts[j].Backend
			}
			return s.Artifacts[i].Role < s.Artifacts[j].Role
		})
	}

	sort.SliceStable(m.Entries, func(i, j int) bool {
		if m.Entries[i].SourceID != m.Entries[j].SourceID {
			return m.Entries[i].SourceID < m.Entries[j].SourceID
		}
		return macroKey(m.Entries[i].Macros) < macroKey(m.Entries[j].Macros)
	})
}

func (m *Manifest) toJSON() *jsonManifest {
	m.canonicalize()
	jm := &jsonManifest{Version: m.Version, Entries: []*jsonSpec{}}
	for _, s := range m.Entries {
		js := &jsonSpec{
			SourceID:     s.SourceID,
			SourcePath:   s.SourcePath,
			SourceHash:   s.SourceHash,
			SourceSize:   s.SourceSize,
			Macros:       ensureObject(s.Macros),
			Uniforms:     []jsonUniform{},
			TextureCount: s.TextureCount,
			Artifacts:    []*jsonArtifact{},
		}
		for _, u := range s.Uniforms {
			js.Uniforms = append(js.Uniforms, jsonUniform{
				Index: u.Index,
				Name:  u.Name,
				Type:  u.Type,
				Size:  u.Size,
			})
		}
		for _, a := range s.Artifacts {
			js.Artifacts = append(js.Artifacts, &jsonArtifact{
				Backend: a.Backend,
				Role:    a.Role,
				Path:    a.Path,
				Hash:    a.Hash,
				Size:    a.Size,
				Gzip:    a.Gzip,
			})
		}
		jm.Entries = append(jm.Entries, js)
	}
	return jm
}

func fromJSON(jm *jsonManifest) *Manifest {
	m := &Manifest{Version: jm.Version}
	for _, js := range jm.Entries {
		s := &Spec{
			SourceID:     js.SourceID,
			SourcePath:   js.SourcePath,
			SourceHash:   js.SourceHash,
			SourceSize:   js.SourceSize,
			Macros:       js.Macros,
			TextureCount: js.TextureCount,
		}
		for _, u := range js.Uniforms {
			s.Uniforms = append(s.Uniforms, Uniform{
				Index: u.Index,
				Name:  u.Name,
				Type:  u.Type,
				Size:  u.Size,
			})
		}
		for _, ja := range js.Artifacts {
			s.Artifacts = append(s.Artifacts, &Artifact{
				Backend: ja.Backend,
				Role:    ja.Role,
				Path:    ja.Path,
				Hash:    ja.Hash,
				Size:    ja.Size,
				Gzip:    ja.Gzip,
			})
		}
		m.Entries = append(m.Entries, s)
	}
	return m
}

// MarshalJSON emits the canonical JSON form.
func (m *Manifest) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.toJSON())
}

// UnmarshalJSON parses JSON without assuming the arrays are sorted; the manifest is re-canonicalized on
// write and comparisons always operate on canonical forms.
func (m *Manifest) UnmarshalJSON(data []byte) error {
	var jm jsonManifest
	if err := json.Unmarshal(data, &jm); err != nil {
		return err
	}
	*m = *fromJSON(&jm)
	return nil
}

// MarshalText renders the manifest as tab-indented JSON terminated by a single LF, the form written to disk.
func (m *Manifest) MarshalText() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "\t")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(m.toJSON()); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
