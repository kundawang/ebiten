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

// Package shadermanifest builds, verifies and compares versioned manifests of precompiled shaders.
//
// A [Manifest] is an auditable inventory that records, for every shader source, its [shaderir.SourceID],
// a hash of the source bytes, the macros and graphics backends it is compiled for, its uniform layout,
// and the generated artifacts together with their hashes. Unlike a single-shot shader precompilation run,
// the manifest can be checked into source control and re-verified in CI: a missing file, a corrupted
// artifact, a changed source or an only-partially-covered backend is reported with the offending entry
// instead of being silently treated as covered.
//
// The package is hermetic: it only generates the text backends (GLSL, GLSL ES, HLSL, MSL and PSSL) that
// Ebitengine can convert to in pure Go. Binary backends such as DXBC or Metal libraries are produced by
// platform toolchains, but are described and verified by the manifest in exactly the same way.
package shadermanifest

// Code identifies the kind of a manifest verification or generation error.
//
// The set of codes is part of the package's contract: CI rules and other tooling may match on them, so
// existing codes are not reused with a different meaning.
type Code string

const (
	// CodeMalformedManifest means the manifest bytes cannot be parsed as the expected JSON shape.
	CodeMalformedManifest Code = "malformed_manifest"

	// CodeUnknownVersion means the manifest's Version is not supported by this package.
	CodeUnknownVersion Code = "unknown_version"

	// CodeMissingSource means a shader source file referenced by an entry cannot be read.
	CodeMissingSource Code = "missing_source"

	// CodeSourceTooLarge means a shader source file is larger than the configured limit.
	CodeSourceTooLarge Code = "source_too_large"

	// CodeBadSourceID means the SourceID recorded in the manifest does not match the source bytes.
	CodeBadSourceID Code = "bad_source_id"

	// CodeBadSourceHash means the source hash recorded in the manifest does not match the source bytes.
	CodeBadSourceHash Code = "bad_source_hash"

	// CodeDuplicateSpec means two entries describe the same source and macro set.
	CodeDuplicateSpec Code = "duplicate_spec"

	// CodeInvalidPath means a source or artifact path is empty, absolute, escapes its root or is not clean.
	CodeInvalidPath Code = "invalid_path"

	// CodeMissingArtifact means an artifact declared by the manifest is absent on disk.
	CodeMissingArtifact Code = "missing_artifact"

	// CodeBadArtifactHash means an artifact's bytes do not match its recorded hash.
	CodeBadArtifactHash Code = "bad_artifact_hash"

	// CodePartialArtifact means a backend is missing some of its required artifacts, so it must not be
	// reported as covered.
	CodePartialArtifact Code = "partial_artifact"

	// CodeBadUniformLayout means the recorded uniform layout does not match the compiled shader.
	CodeBadUniformLayout Code = "bad_uniform_layout"

	// CodeCompressionFailed means an artifact cannot be compressed or decompressed.
	CodeCompressionFailed Code = "compression_failed"

	// CodeCompileFailed means a shader source cannot be compiled for a requested backend.
	CodeCompileFailed Code = "compile_failed"
)

// SampleError is an error located at a specific manifest entry, called a sample. Every error carries
// enough context (the source ID, backend, role and path involved) to identify the offending item without
// having to reproduce the run.
type SampleError struct {
	// Code classifies the error.
	Code Code

	// SourceID is the SourceID of the entry the error belongs to. It may be empty for manifest-wide errors.
	SourceID string

	// Backend is the backend the error relates to, or empty for source- or manifest-level errors.
	Backend Backend

	// Role is the artifact role (e.g. "vertex") the error relates to, or empty when not applicable.
	Role string

	// Path is the source or artifact path the error relates to, or empty when not applicable.
	Path string

	// Message is a human-readable description.
	Message string

	// Err is the wrapped cause, if any.
	Err error
}

func (e *SampleError) Error() string {
	var b []byte
	b = append(b, string(e.Code)...)
	if e.SourceID != "" {
		b = append(b, " ("...)
		b = append(b, e.SourceID...)
		b = append(b, ')')
	}
	if e.Backend != "" {
		b = append(b, ": backend="...)
		b = append(b, string(e.Backend)...)
	}
	if e.Role != "" {
		b = append(b, " role="...)
		b = append(b, e.Role...)
	}
	if e.Path != "" {
		b = append(b, " path="...)
		b = append(b, e.Path...)
	}
	if e.Message != "" {
		b = append(b, ": "...)
		b = append(b, e.Message...)
	}
	if e.Err != nil {
		b = append(b, ": "...)
		b = append(b, e.Err.Error()...)
	}
	return string(b)
}

// Unwrap returns the wrapped cause.
func (e *SampleError) Unwrap() error {
	return e.Err
}

// ErrorList is a collection of [SampleError]s. Generation and validation collect every problem they can
// rather than stopping at the first, so a single CI run reports the full set of broken or missing samples.
type ErrorList []*SampleError

func (l ErrorList) Error() string {
	switch len(l) {
	case 0:
		return "shadermanifest: no errors"
	case 1:
		return l[0].Error()
	default:
		return "shadermanifest: " + itoa(len(l)) + " errors: first: " + l[0].Error()
	}
}

// Has reports whether the list contains an error with the given code.
func (l ErrorList) Has(code Code) bool {
	for _, e := range l {
		if e.Code == code {
			return true
		}
	}
	return false
}

// Codes returns the distinct error codes present, in a stable order.
func (l ErrorList) Codes() []Code {
	seen := map[Code]struct{}{}
	var codes []Code
	for _, e := range l {
		if _, ok := seen[e.Code]; ok {
			continue
		}
		seen[e.Code] = struct{}{}
		codes = append(codes, e.Code)
	}
	return codes
}

func (l *ErrorList) add(code Code, sourceID string, backend Backend, role, path, message string, cause error) {
	*l = append(*l, &SampleError{
		Code:     code,
		SourceID: sourceID,
		Backend:  backend,
		Role:     role,
		Path:     path,
		Message:  message,
		Err:      cause,
	})
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
