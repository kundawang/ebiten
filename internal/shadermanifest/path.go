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

import "strings"

// DefaultArtifactDir is the directory, relative to the manifest root, under which generated artifacts
// are placed unless GenerateOptions.ArtifactDir overrides it.
const DefaultArtifactDir = "shaders"

// artifactExtension returns the conventional file extension for a backend's role.
func artifactExtension(backend Backend, role string) string {
	switch backend {
	case BackendGLSL, BackendGLSLES:
		return ".glsl"
	case BackendHLSL:
		return ".hlsl"
	case BackendMSL:
		return ".metal"
	case BackendPSSL:
		return ".pssl"
	case BackendDXBCWindows, BackendDXBCXbox:
		return ".dxbc"
	case BackendMetalMacOS, BackendMetalIOS:
		return ".metallib"
	case BackendPlayStation5:
		if role == roleHeader {
			return ".h"
		}
		return ".pssl"
	default:
		return ".bin"
	}
}

// artifactPath builds the stable, slash-separated artifact path for a given entry, backend and role.
//
// The layout is "<dir>/<sourceID>[/<macroTag>]/<backend>/<role><ext>". Every component is derived from
// manifest data rather than filesystem order, so the path is identical on every platform and in every
// run. The macro tag is omitted for the default (macro-free) variant.
func artifactPath(dir, sourceID string, macros map[string]string, backend Backend, role string) string {
	var b strings.Builder
	b.WriteString(dir)
	b.WriteByte('/')
	b.WriteString(sourceID)
	if tag := macroTag(macros); tag != "" {
		b.WriteByte('/')
		b.WriteString(tag)
	}
	b.WriteByte('/')
	b.WriteString(string(backend))
	b.WriteByte('/')
	b.WriteString(role)
	b.WriteString(artifactExtension(backend, role))
	return b.String()
}

// macroTag renders macros as a filesystem-safe, order-independent tag. Macro keys and values are
// restricted by [validateMacros] so the tag cannot contain path separators, "." or "..".
func macroTag(macros map[string]string) string {
	if len(macros) == 0 {
		return ""
	}
	keys := make([]string, 0, len(macros))
	for k := range macros {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j-1] > keys[j]; j-- {
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte('_')
		}
		b.WriteString(k)
		b.WriteByte('-')
		b.WriteString(macros[k])
	}
	return b.String()
}

// validateMacros rejects macro names or values that are empty or contain characters that would make an
// artifact path ambiguous or unsafe.
func validateMacros(macros map[string]string) bool {
	for k, v := range macros {
		if k == "" || v == "" {
			return false
		}
		if !safeMacroToken(k) || !safeMacroToken(v) {
			return false
		}
	}
	return true
}

func safeMacroToken(s string) bool {
	if s == "." || s == ".." {
		return false
	}
	for _, r := range s {
		switch r {
		case '/', '\\', 0, ':':
			return false
		}
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}
