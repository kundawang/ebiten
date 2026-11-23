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

package shaderprecomp

import (
	"github.com/hajimehoshi/ebiten/v2/internal/shadermanifest"
)

// ManifestVersion is the manifest format version produced by this package.
const ManifestVersion = shadermanifest.CurrentVersion

// Backend identifies a graphics backend a shader is precompiled for.
type Backend = shadermanifest.Backend

// Text and binary backends. See the internal package for the full semantics.
const (
	BackendGLSL         = shadermanifest.BackendGLSL
	BackendGLSLES       = shadermanifest.BackendGLSLES
	BackendHLSL         = shadermanifest.BackendHLSL
	BackendMSL          = shadermanifest.BackendMSL
	BackendPSSL         = shadermanifest.BackendPSSL
	BackendDXBCWindows  = shadermanifest.BackendDXBCWindows
	BackendDXBCXbox     = shadermanifest.BackendDXBCXbox
	BackendMetalMacOS   = shadermanifest.BackendMetalMacOS
	BackendMetalIOS     = shadermanifest.BackendMetalIOS
	BackendPlayStation5 = shadermanifest.BackendPlayStation5
)

// Manifest is a versioned, auditable inventory of precompiled shaders.
type Manifest = shadermanifest.Manifest

// Spec describes one shader source and everything precompiled for it.
type Spec = shadermanifest.Spec

// Artifact is one generated file for a backend and role.
type Artifact = shadermanifest.Artifact

// Uniform is one entry of a shader's uniform layout.
type Uniform = shadermanifest.Uniform

// ShaderInput describes one shader source to precompile.
type ShaderInput = shadermanifest.ShaderInput

// GenerateOptions controls [Generate].
type GenerateOptions = shadermanifest.GenerateOptions

// ValidateOptions controls [Validate].
type ValidateOptions = shadermanifest.ValidateOptions

// Coverage summarizes a successful validation.
type Coverage = shadermanifest.Coverage

// ErrorCode classifies a manifest generation or validation error.
type ErrorCode = shadermanifest.Code

// Manifest error codes.
const (
	CodeMalformedManifest = shadermanifest.CodeMalformedManifest
	CodeUnknownVersion    = shadermanifest.CodeUnknownVersion
	CodeMissingSource     = shadermanifest.CodeMissingSource
	CodeSourceTooLarge    = shadermanifest.CodeSourceTooLarge
	CodeBadSourceID       = shadermanifest.CodeBadSourceID
	CodeBadSourceHash     = shadermanifest.CodeBadSourceHash
	CodeDuplicateSpec     = shadermanifest.CodeDuplicateSpec
	CodeInvalidPath       = shadermanifest.CodeInvalidPath
	CodeMissingArtifact   = shadermanifest.CodeMissingArtifact
	CodeBadArtifactHash   = shadermanifest.CodeBadArtifactHash
	CodePartialArtifact   = shadermanifest.CodePartialArtifact
	CodeBadUniformLayout  = shadermanifest.CodeBadUniformLayout
	CodeCompressionFailed = shadermanifest.CodeCompressionFailed
	CodeCompileFailed     = shadermanifest.CodeCompileFailed
)

// SampleError is an error located at a specific manifest entry.
type SampleError = shadermanifest.SampleError

// ErrorList is a collection of [SampleError]s.
type ErrorList = shadermanifest.ErrorList

// DiffReport is the result of comparing two manifests.
type DiffReport = shadermanifest.DiffReport

// Change is one difference between two manifests.
type Change = shadermanifest.Change

// Generate compiles the input shaders for the selected text backends, writes their artifacts under root,
// and returns the canonical manifest. See the internal package's Generate for the full behavior.
func Generate(root string, inputs []ShaderInput, opts *GenerateOptions) (*Manifest, ErrorList) {
	return shadermanifest.Generate(root, inputs, opts)
}

// LoadManifest reads and parses a manifest from path.
func LoadManifest(path string) (*Manifest, error) {
	return shadermanifest.Load(path)
}

// Validate verifies a manifest against the files under root.
func Validate(root string, m *Manifest, opts *ValidateOptions) (*Coverage, ErrorList) {
	return shadermanifest.Validate(root, m, opts)
}

// Diff compares two manifests and returns every difference.
func Diff(old, new *Manifest) *DiffReport {
	return shadermanifest.Diff(old, new)
}
