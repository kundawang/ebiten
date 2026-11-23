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

package shaderprecomp_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/exp/shaderprecomp"
)

// kageSource is a minimal valid pixel-unit Kage shader used by the manifest wrapper tests.
const kageSource = `//kage:unit pixels

package main

var T float

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return vec4(T, 0, 0, 1)
}
`

func writeKage(t *testing.T, root, name string) string {
	t.Helper()
	rel := filepath.Join("shaders", name)
	if err := os.MkdirAll(filepath.Join(root, "shaders"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, rel), []byte(kageSource), 0644); err != nil {
		t.Fatal(err)
	}
	return filepath.ToSlash(rel)
}

func TestManifestGenerateValidateDiff(t *testing.T) {
	if shaderprecomp.ManifestVersion < 1 {
		t.Fatalf("unexpected manifest version %d", shaderprecomp.ManifestVersion)
	}

	root := t.TempDir()
	src := writeKage(t, root, "effect.kage")

	backends := []shaderprecomp.Backend{
		shaderprecomp.BackendGLSL,
		shaderprecomp.BackendGLSLES,
		shaderprecomp.BackendHLSL,
		shaderprecomp.BackendMSL,
	}
	m, errs := shaderprecomp.Generate(root,
		[]shaderprecomp.ShaderInput{{SourcePath: src}},
		&shaderprecomp.GenerateOptions{Backends: backends})
	if errs != nil {
		t.Fatalf("Generate: %v", errs)
	}
	if len(m.Entries) != 1 {
		t.Fatalf("got %d entries, want 1", len(m.Entries))
	}
	if len(m.Entries[0].Uniforms) == 0 {
		t.Error("uniform layout was not recorded")
	}

	// Save and reload through the public API; validation must still pass.
	manifestPath := filepath.Join(root, "manifest.json")
	if err := m.Save(manifestPath); err != nil {
		t.Fatal(err)
	}
	reloaded, err := shaderprecomp.LoadManifest(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	cov, errs := shaderprecomp.Validate(root, reloaded, &shaderprecomp.ValidateOptions{RequireBackends: backends})
	if errs != nil {
		t.Fatalf("Validate: %v", errs)
	}
	if cov.Entries != 1 {
		t.Errorf("coverage entries: got %d, want 1", cov.Entries)
	}

	// A manifest compared against itself has no differences.
	if report := shaderprecomp.Diff(reloaded, reloaded); !report.Empty() {
		t.Errorf("self-diff must be empty:\n%s", report.Text())
	}
}

func TestManifestErrorCodesAreExposed(t *testing.T) {
	root := t.TempDir()

	// A missing source yields the exposed missing-source error code.
	_, errs := shaderprecomp.Generate(root,
		[]shaderprecomp.ShaderInput{{SourcePath: "shaders/absent.kage"}},
		&shaderprecomp.GenerateOptions{Backends: []shaderprecomp.Backend{shaderprecomp.BackendGLSL}})
	if errs == nil {
		t.Fatal("a missing source must produce an error")
	}
	if !errs.Has(shaderprecomp.CodeMissingSource) {
		t.Errorf("expected %q, got %v", shaderprecomp.CodeMissingSource, errs.Codes())
	}

	// Each returned error is a *SampleError carrying sample-level context.
	for _, e := range errs {
		if e.Path != "shaders/absent.kage" {
			t.Errorf("error path: got %q, want the missing source path", e.Path)
		}
	}
}
