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
	"fmt"
	"os"
	"path/filepath"

	"github.com/hajimehoshi/ebiten/v2/internal/shaderir"
)

// ValidateOptions controls [Validate].
type ValidateOptions struct {
	// MaxSourceSize bounds a source file's size in bytes. Defaults to [DefaultMaxSourceSize].
	MaxSourceSize int

	// RequireBackends lists the backends every entry must cover completely. A backend present with only
	// some of its roles is reported as [CodePartialArtifact]; a backend that is absent entirely is also a
	// partial-coverage error so CI never mistakes an incomplete manifest for full coverage.
	RequireBackends []Backend
}

// Coverage summarizes the result of a successful [Validate] run. It never reports a backend as covered
// unless every one of its required artifacts is present and verified.
type Coverage struct {
	// Entries is the number of validated shader entries.
	Entries int

	// Covered maps each backend that is fully present across every entry to the number of entries
	// covering it. A backend appears only when all its roles verified for all entries.
	Covered map[Backend]int

	// Missing maps each required backend to the number of entries that do not fully cover it.
	Missing map[Backend]int
}

// Validate verifies the manifest against the files under root.
//
// It checks the format version, every entry's source path safety and size, the source ID and hash, the
// uniform layout, and each artifact's existence, required roles, and content hash (decompressing gzip
// files first). All problems are accumulated into the returned [ErrorList]; validation does not stop at
// the first failure. The returned [Coverage] is non-nil even when errors are found, but must not be
// treated as a pass while the error list is non-empty.
func Validate(root string, m *Manifest, opts *ValidateOptions) (*Coverage, ErrorList) {
	o := ValidateOptions{MaxSourceSize: DefaultMaxSourceSize}
	if opts != nil {
		if opts.MaxSourceSize > 0 {
			o.MaxSourceSize = opts.MaxSourceSize
		}
		o.RequireBackends = opts.RequireBackends
	}

	var errs ErrorList
	cov := &Coverage{
		Covered: map[Backend]int{},
		Missing: map[Backend]int{},
	}

	if m.Version != CurrentVersion {
		errs.add(CodeUnknownVersion, "", "", "", "",
			fmt.Sprintf("manifest version %d is not supported (want %d)", m.Version, CurrentVersion), nil)
		return cov, errs
	}

	// Track duplicate (source, macro) entries across the whole manifest.
	seenSpecs := map[string]struct{}{}

	for _, spec := range m.Entries {
		cov.Entries++

		if _, err := shaderir.ParseSourceID(spec.SourceID); err != nil {
			errs.add(CodeBadSourceID, spec.SourceID, "", "", spec.SourcePath, "source ID is not a valid base32 ID", err)
		}

		sourcePath, ok := cleanRelPath(spec.SourcePath)
		if !ok {
			errs.add(CodeInvalidPath, spec.SourceID, "", "", spec.SourcePath, "invalid source path", nil)
		}

		macroIdentifier := spec.SourceID + "\x00" + macroKey(spec.Macros)
		if _, dup := seenSpecs[macroIdentifier]; dup {
			errs.add(CodeDuplicateSpec, spec.SourceID, "", "", sourcePath, "duplicate source and macro set", nil)
		}
		seenSpecs[macroIdentifier] = struct{}{}

		if !validateMacros(spec.Macros) {
			errs.add(CodeInvalidPath, spec.SourceID, "", "", sourcePath, "invalid macro name or value", nil)
		}

		// Source-level checks. Read once and reuse for ID/hash/layout.
		if ok {
			abs := filepath.Join(root, filepath.FromSlash(sourcePath))
			data, err := os.ReadFile(abs)
			if err != nil {
				errs.add(CodeMissingSource, spec.SourceID, "", "", sourcePath, "cannot read shader source", err)
			} else {
				if len(data) > o.MaxSourceSize {
					errs.add(CodeSourceTooLarge, spec.SourceID, "", "", sourcePath,
						fmt.Sprintf("source is %d bytes, limit is %d", len(data), o.MaxSourceSize), nil)
				}
				wantID, err := calcSourceID(data)
				if err != nil {
					errs.add(CodeCompileFailed, spec.SourceID, "", "", sourcePath, "cannot compute source ID", err)
				} else if wantID != spec.SourceID {
					errs.add(CodeBadSourceID, spec.SourceID, "", "", sourcePath,
						fmt.Sprintf("recorded source ID %s does not match source %s", spec.SourceID, wantID), nil)
				}
				if got := hashBytes(data); got != spec.SourceHash {
					errs.add(CodeBadSourceHash, spec.SourceID, "", "", sourcePath,
						fmt.Sprintf("recorded source hash %s does not match %s", spec.SourceHash, got), nil)
				}
				if spec.SourceSize != len(data) {
					errs.add(CodeBadSourceHash, spec.SourceID, "", "", sourcePath,
						fmt.Sprintf("recorded source size %d does not match %d", spec.SourceSize, len(data)), nil)
				}

				// Uniform layout and texture count must match what the current source compiles to.
				if compiled, err := compileSource(data); err != nil {
					errs.add(CodeCompileFailed, spec.SourceID, "", "", sourcePath, "compiling shader failed", err)
				} else {
					if !uniformsEqual(compiled.uniforms, spec.Uniforms) {
						errs.add(CodeBadUniformLayout, spec.SourceID, "", "", sourcePath,
							"recorded uniform layout does not match the compiled shader", nil)
					}
					if compiled.textureCount != spec.TextureCount {
						errs.add(CodeBadUniformLayout, spec.SourceID, "", "", sourcePath,
							fmt.Sprintf("recorded texture count %d does not match %d", spec.TextureCount, compiled.textureCount), nil)
					}
				}
			}
		}

		// Artifact checks: group roles by backend and verify each. integrityOK tracks whether every role
		// of a backend exists on disk and matches its recorded hash and size; a backend is only ever
		// reported as covered when it is both complete and intact.
		artifactRoles := map[Backend]map[string]*Artifact{}
		integrityOK := map[Backend]bool{}
		markBad := func(backend Backend) {
			integrityOK[backend] = false
		}
		for _, art := range spec.Artifacts {
			if !IsKnownBackend(art.Backend) {
				errs.add(CodeCompileFailed, spec.SourceID, art.Backend, art.Role, art.Path, "unknown backend", nil)
				continue
			}
			if !validRole(art.Backend, art.Role) {
				errs.add(CodePartialArtifact, spec.SourceID, art.Backend, art.Role, art.Path, "role is not part of this backend", nil)
				markBad(art.Backend)
			}
			relPath, ok := cleanRelPath(art.Path)
			if !ok {
				errs.add(CodeInvalidPath, spec.SourceID, art.Backend, art.Role, art.Path, "invalid artifact path", nil)
				markBad(art.Backend)
				continue
			}
			if artifactRoles[art.Backend] == nil {
				artifactRoles[art.Backend] = map[string]*Artifact{}
				integrityOK[art.Backend] = true
			}
			if _, dup := artifactRoles[art.Backend][art.Role]; dup {
				errs.add(CodeDuplicateSpec, spec.SourceID, art.Backend, art.Role, relPath, "duplicate artifact role", nil)
				markBad(art.Backend)
			}
			artifactRoles[art.Backend][art.Role] = art

			abs := filepath.Join(root, filepath.FromSlash(relPath))
			stored, err := os.ReadFile(abs)
			if err != nil {
				errs.add(CodeMissingArtifact, spec.SourceID, art.Backend, art.Role, relPath, "artifact file is missing", err)
				markBad(art.Backend)
				continue
			}
			content := stored
			if art.Gzip {
				decompressed, err := gunzipBytes(stored)
				if err != nil {
					errs.add(CodeCompressionFailed, spec.SourceID, art.Backend, art.Role, relPath, "cannot decompress gzip artifact", err)
					markBad(art.Backend)
					continue
				}
				content = decompressed
			}
			if got := hashBytes(content); got != art.Hash {
				errs.add(CodeBadArtifactHash, spec.SourceID, art.Backend, art.Role, relPath,
					fmt.Sprintf("recorded artifact hash %s does not match %s", art.Hash, got), nil)
				markBad(art.Backend)
			}
			if art.Size != len(content) {
				errs.add(CodeBadArtifactHash, spec.SourceID, art.Backend, art.Role, relPath,
					fmt.Sprintf("recorded artifact size %d does not match %d", art.Size, len(content)), nil)
				markBad(art.Backend)
			}
		}

		// Determine which present backends are complete and intact for this entry.
		complete := map[Backend]bool{}
		for backend, roles := range artifactRoles {
			if backendComplete(backend, roles) && integrityOK[backend] {
				complete[backend] = true
				cov.Covered[backend]++
				continue
			}
			if !backendComplete(backend, roles) {
				errs.add(CodePartialArtifact, spec.SourceID, backend, "", "",
					fmt.Sprintf("backend is missing roles: %v", missingRoles(backend, roles)), nil)
			}
		}

		// Required backends must be complete on every entry; anything short of that is counted once as
		// missing, so an incomplete manifest can never be summarized as fully covered.
		for _, required := range o.RequireBackends {
			if complete[required] {
				continue
			}
			cov.Missing[required]++
			if _, present := artifactRoles[required]; !present {
				errs.add(CodePartialArtifact, spec.SourceID, required, "", "",
					"required backend has no artifacts", nil)
			}
		}
	}

	return cov, errs
}

func backendComplete(backend Backend, roles map[string]*Artifact) bool {
	for _, want := range backendRoles[backend] {
		if _, ok := roles[want]; !ok {
			return false
		}
	}
	return true
}

func missingRoles(backend Backend, roles map[string]*Artifact) []string {
	var missing []string
	for _, want := range backendRoles[backend] {
		if _, ok := roles[want]; !ok {
			missing = append(missing, want)
		}
	}
	return missing
}

func uniformsEqual(got, want []Uniform) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
