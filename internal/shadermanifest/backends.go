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

import "slices"

// Backend identifies a graphics backend a shader is precompiled for.
//
// The textual backends ([BackendGLSL], [BackendGLSLES], [BackendHLSL], [BackendMSL] and [BackendPSSL])
// are produced in pure Go by Ebitengine and can be generated directly. The binary backends
// ([BackendDXBCWindows], [BackendDXBCXbox], [BackendMetalMacOS], [BackendMetalIOS] and
// [BackendPlayStation5]) require a platform toolchain and are described and verified by the manifest,
// but their bytes are produced externally.
type Backend string

const (
	// BackendGLSL is desktop OpenGL GLSL (vertex and fragment).
	BackendGLSL Backend = "glsl"

	// BackendGLSLES is OpenGL ES GLSL ES (vertex and fragment).
	BackendGLSLES Backend = "glsl-es"

	// BackendHLSL is HLSL source (vertex and pixel) for DirectX.
	BackendHLSL Backend = "hlsl"

	// BackendMSL is Metal Shading Language source.
	BackendMSL Backend = "msl"

	// BackendPSSL is PlayStation Shader Language source (vertex and pixel).
	BackendPSSL Backend = "pssl"

	// BackendDXBCWindows is a DXBC binary compiled for Windows with the Windows SDK fxc tool.
	BackendDXBCWindows Backend = "dxbc-windows"

	// BackendDXBCXbox is a DXBC binary compiled for Xbox with the GDK shader compiler.
	BackendDXBCXbox Backend = "dxbc-xbox"

	// BackendMetalMacOS is a Metal .metallib built with the macOS SDK.
	BackendMetalMacOS Backend = "metal-macos"

	// BackendMetalIOS is a Metal .metallib built with the iOS device SDK.
	BackendMetalIOS Backend = "metal-ios"

	// BackendPlayStation5 is a precompiled PlayStation 5 shader.
	BackendPlayStation5 Backend = "playstation5"
)

// Artifact roles. Each backend is made up of a fixed set of roles; coverage means every role is present.
const (
	roleVertex   = "vertex"
	rolePixel    = "pixel"
	roleFragment = "fragment"
	roleShader   = "shader"
	roleLibrary  = "library"
	roleHeader   = "header"
	roleText     = "text"
)

// backendRoles maps each backend to the roles a complete set of artifacts must provide, in stable order.
var backendRoles = map[Backend][]string{
	BackendGLSL:         {roleVertex, roleFragment},
	BackendGLSLES:       {roleVertex, roleFragment},
	BackendHLSL:         {roleVertex, rolePixel},
	BackendMSL:          {roleShader},
	BackendPSSL:         {roleVertex, rolePixel},
	BackendDXBCWindows:  {roleVertex, rolePixel},
	BackendDXBCXbox:     {roleVertex, rolePixel},
	BackendMetalMacOS:   {roleLibrary},
	BackendMetalIOS:     {roleLibrary},
	BackendPlayStation5: {roleVertex, rolePixel, roleHeader, roleText},
}

// textBackends are the backends Ebitengine can generate without an external toolchain. PSSL is a known
// backend but its in-process compiler is not implemented in this Ebitengine version, so it is excluded:
// its artifacts are produced externally and still fully described and verified by the manifest.
var textBackends = map[Backend]struct{}{
	BackendGLSL:   {},
	BackendGLSLES: {},
	BackendHLSL:   {},
	BackendMSL:    {},
}

// IsTextBackend reports whether b is a textual backend that [Generate] can produce in pure Go.
func IsTextBackend(b Backend) bool {
	_, ok := textBackends[b]
	return ok
}

// IsKnownBackend reports whether b is one of the defined backends.
func IsKnownBackend(b Backend) bool {
	_, ok := backendRoles[b]
	return ok
}

// Roles returns the roles a complete set of artifacts for b must provide.
// It returns nil for an unknown backend.
func Roles(b Backend) []string {
	roles := backendRoles[b]
	return slices.Clone(roles)
}
