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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultMaxSourceSize is the default upper bound on a shader source file: 1 MiB. A Kage shader larger
// than this is almost certainly an input error (for example a generated blob committed by mistake).
const DefaultMaxSourceSize = 1 << 20

// ShaderInput describes one shader source to precompile, specialized by an optional macro set.
type ShaderInput struct {
	// SourcePath is the Kage source file, relative to the root passed to [Generate], using forward slashes.
	SourcePath string

	// Macros optionally specializes this entry. The default variant passes nil. Two inputs with the same
	// source and an identical macro set are a [CodeDuplicateSpec] error.
	Macros map[string]string
}

// GenerateOptions controls [Generate].
type GenerateOptions struct {
	// Backends selects the text backends to generate. Defaults to all text backends when empty. Binary
	// backends are added afterwards with [Manifest.AddArtifact].
	Backends []Backend

	// ArtifactDir is the directory, relative to root, that artifacts are written under.
	// Defaults to [DefaultArtifactDir].
	ArtifactDir string

	// Gzip compresses every generated artifact when true.
	Gzip bool

	// MaxSourceSize bounds a source file's size in bytes. Defaults to [DefaultMaxSourceSize].
	MaxSourceSize int
}

// Generate compiles every input shader for the selected text backends, writes the artifacts under root,
// and returns the canonical manifest describing them.
//
// All per-shader failures are collected into an [ErrorList] and the run continues with the remaining
// inputs, so one broken shader does not hide the others. A non-nil error means the manifest must not be
// treated as complete.
func Generate(root string, inputs []ShaderInput, opts *GenerateOptions) (*Manifest, ErrorList) {
	o := resolveOptions(opts)
	var errs ErrorList

	dir, ok := cleanRelPath(o.ArtifactDir)
	if !ok {
		errs.add(CodeInvalidPath, "", "", "", o.ArtifactDir, "invalid artifact directory", nil)
		return nil, errs
	}

	// Detect duplicate (source, macros) inputs up front, while keeping first-seen order deterministic.
	seen := map[string]struct{}{}

	m := &Manifest{Version: CurrentVersion}
	for _, in := range inputs {
		sourcePath, ok := cleanRelPath(in.SourcePath)
		if !ok {
			errs.add(CodeInvalidPath, "", "", "", in.SourcePath, "invalid source path", nil)
			continue
		}
		if !validateMacros(in.Macros) {
			errs.add(CodeInvalidPath, "", "", "", sourcePath, "invalid macro name or value", nil)
			continue
		}

		absSource := filepath.Join(root, filepath.FromSlash(sourcePath))
		data, err := os.ReadFile(absSource)
		if err != nil {
			errs.add(CodeMissingSource, "", "", "", sourcePath, "cannot read shader source", err)
			continue
		}
		if len(data) > o.MaxSourceSize {
			errs.add(CodeSourceTooLarge, "", "", "", sourcePath,
				fmt.Sprintf("source is %d bytes, limit is %d", len(data), o.MaxSourceSize), nil)
			continue
		}

		sourceID, err := calcSourceID(data)
		if err != nil {
			errs.add(CodeCompileFailed, "", "", "", sourcePath, "cannot compute source ID", err)
			continue
		}

		dedupKey := sourceID + "\x00" + macroKey(in.Macros)
		if _, dup := seen[dedupKey]; dup {
			errs.add(CodeDuplicateSpec, sourceID, "", "", sourcePath, "duplicate source and macro set", nil)
			continue
		}
		seen[dedupKey] = struct{}{}

		compiled, err := compileSource(data)
		if err != nil {
			errs.add(CodeCompileFailed, sourceID, "", "", sourcePath, "compiling shader failed", err)
			continue
		}

		spec := &Spec{
			SourceID:     sourceID,
			SourcePath:   sourcePath,
			SourceHash:   hashBytes(data),
			SourceSize:   len(data),
			Macros:       cloneMacros(in.Macros),
			Uniforms:     compiled.uniforms,
			TextureCount: compiled.textureCount,
		}

		for _, backend := range o.Backends {
			if !IsTextBackend(backend) {
				errs.add(CodeCompileFailed, sourceID, backend, "", sourcePath,
					"binary backends cannot be generated; add them with AddArtifact", nil)
				continue
			}
			roleBytes, err := compiled.textArtifacts(backend)
			if err != nil {
				errs.add(CodeCompileFailed, sourceID, backend, "", sourcePath, err.Error(), err)
				continue
			}
			roles := backendRoles[backend]
			for _, role := range roles {
				content := roleBytes[role]
				artPath := artifactPath(dir, sourceID, spec.Macros, backend, role)
				record, err := writeArtifact(root, artPath, backend, role, content, o.Gzip)
				if err != nil {
					errs.add(CodeCompressionFailed, sourceID, backend, role, artPath, "writing artifact failed", err)
					continue
				}
				spec.Artifacts = append(spec.Artifacts, record)
			}
		}

		m.Entries = append(m.Entries, spec)
	}

	m.canonicalize()
	if len(errs) > 0 {
		return m, errs
	}
	return m, nil
}

// AddArtifact records an externally produced binary (or text) artifact for an entry already in the
// manifest. It is the entry point for binary backends such as DXBC or Metal libraries whose bytes come
// from a platform toolchain.
//
// The artifact bytes are written under root in compressed form when gzip is true; the recorded hash and
// size always describe the uncompressed content. Adding a (backend, role) already present is an error so
// the manifest never silently overwrites coverage.
func (m *Manifest) AddArtifact(root, sourceID string, macros map[string]string, backend Backend, role string, content []byte, gzip bool) (*Artifact, ErrorList) {
	var errs ErrorList

	if !IsKnownBackend(backend) {
		errs.add(CodeCompileFailed, sourceID, backend, role, "", "unknown backend", nil)
		return nil, errs
	}
	if !validRole(backend, role) {
		errs.add(CodePartialArtifact, sourceID, backend, role, "", "role is not part of this backend", nil)
		return nil, errs
	}
	spec := m.findSpec(sourceID, macros)
	if spec == nil {
		errs.add(CodeMissingSource, sourceID, backend, role, "", "no manifest entry for source and macros", nil)
		return nil, errs
	}
	for _, a := range spec.Artifacts {
		if a.Backend == backend && a.Role == role {
			errs.add(CodeDuplicateSpec, sourceID, backend, role, a.Path, "artifact already recorded", nil)
			return nil, errs
		}
	}

	// Place the new artifact in the same directory tree as the entry's existing artifacts, falling back
	// to the default directory for an entry that has none yet. This keeps a custom ArtifactDir consistent
	// without an extra parameter.
	dir := DefaultArtifactDir
	if len(spec.Artifacts) > 0 {
		if d, ok := artifactDirOf(spec.SourceID, spec.Artifacts[0].Path); ok {
			dir = d
		}
	}
	cleanDir, ok := cleanRelPath(dir)
	if !ok {
		errs.add(CodeInvalidPath, sourceID, backend, role, dir, "invalid artifact directory", nil)
		return nil, errs
	}

	artPath := artifactPath(cleanDir, sourceID, macros, backend, role)
	record, err := writeArtifact(root, artPath, backend, role, content, gzip)
	if err != nil {
		errs.add(CodeCompressionFailed, sourceID, backend, role, artPath, "writing artifact failed", err)
		return nil, errs
	}
	spec.Artifacts = append(spec.Artifacts, record)
	m.canonicalize()
	return record, nil
}

// Load reads and parses a manifest from disk. A parse failure is reported as [CodeMalformedManifest].
func Load(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, &SampleError{Code: CodeMalformedManifest, Path: path, Message: "parsing manifest failed", Err: err}
	}
	return &m, nil
}

// Save writes the manifest in its canonical text form. It first rejects an unsupported version.
func (m *Manifest) Save(path string) error {
	if m.Version != CurrentVersion {
		return &SampleError{Code: CodeUnknownVersion, Message: fmt.Sprintf("version %d is not supported", m.Version)}
	}
	m.canonicalize()
	data, err := m.MarshalText()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// writeArtifact stores content (optionally gzipped) at root/relPath and returns the recorded Artifact.
// The hash and size are computed over the uncompressed bytes, independent of gzip.
func writeArtifact(root, relPath string, backend Backend, role string, content []byte, gzip bool) (*Artifact, error) {
	stored := content
	if gzip {
		compressed, err := gzipBytes(content)
		if err != nil {
			return nil, err
		}
		stored = compressed
	}
	abs := filepath.Join(root, filepath.FromSlash(relPath))
	if err := os.MkdirAll(filepath.Dir(abs), 0755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(abs, stored, 0644); err != nil {
		return nil, err
	}
	return &Artifact{
		Backend: backend,
		Role:    role,
		Path:    relPath,
		Hash:    hashBytes(content),
		Size:    len(content),
		Gzip:    gzip,
	}, nil
}

func resolveOptions(opts *GenerateOptions) GenerateOptions {
	var o GenerateOptions
	if opts != nil {
		o = *opts
	}
	if o.ArtifactDir == "" {
		o.ArtifactDir = DefaultArtifactDir
	}
	if o.MaxSourceSize == 0 {
		o.MaxSourceSize = DefaultMaxSourceSize
	}
	if len(o.Backends) == 0 {
		o.Backends = []Backend{
			BackendGLSL, BackendGLSLES, BackendHLSL, BackendMSL,
		}
	}
	// De-duplicate backends while preserving first-seen order for readable output.
	o.Backends = dedupBackends(o.Backends)
	return o
}

func dedupBackends(in []Backend) []Backend {
	seen := map[Backend]struct{}{}
	out := in[:0:0]
	for _, b := range in {
		if _, ok := seen[b]; ok {
			continue
		}
		seen[b] = struct{}{}
		out = append(out, b)
	}
	return out
}

func cloneMacros(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (m *Manifest) findSpec(sourceID string, macros map[string]string) *Spec {
	key := macroKey(macros)
	for _, s := range m.Entries {
		if s.SourceID == sourceID && macroKey(s.Macros) == key {
			return s
		}
	}
	return nil
}

func validRole(backend Backend, role string) bool {
	for _, r := range backendRoles[backend] {
		if r == role {
			return true
		}
	}
	return false
}

// artifactDirOf returns the artifact root directory of an artifact path of the form
// "<dir>/<sourceID>[/<macroTag>]/<backend>/<role><ext>", by locating the SourceID segment and returning
// everything that precedes it. The optional macro tag makes a fixed-offset split unsafe.
func artifactDirOf(sourceID, p string) (string, bool) {
	marker := "/" + sourceID + "/"
	idx := strings.Index(p, marker)
	if idx <= 0 {
		return "", false
	}
	dir := p[:idx]
	if dir == "" {
		return "", false
	}
	return dir, true
}
