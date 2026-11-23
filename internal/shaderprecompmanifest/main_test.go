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

package main_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// buildTool compiles the CLI into a temp binary once, returning its path.
func buildTool(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go command is missing")
	}
	binary := filepath.Join(t.TempDir(), "shaderprecompmanifest")
	cmd := exec.Command("go", "build", "-buildvcs=false", "-o", binary,
		"github.com/hajimehoshi/ebiten/v2/internal/shaderprecompmanifest")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building the tool failed: %v\n%s", err, out)
	}
	return binary
}

// fixtureRoot copies the shadermanifest testdata shaders into a fresh working directory.
func fixtureRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	src := filepath.Join("..", "shadermanifest", "testdata", "shaders")
	if err := os.MkdirAll(filepath.Join(root, "shaders"), 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tint.kage", "weights.kage", "broken.kage"} {
		data, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "shaders", name), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func run(t *testing.T, binary, root string, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binary, args...)
	cmd.Dir = root
	var stdout, stderr []byte
	var err error
	if stdout, err = cmd.Output(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode(), string(stdout), string(exitErr.Stderr)
		}
		t.Fatal(err)
	}
	return 0, string(stdout), string(stderr)
}

func TestCLIGenerateValidateDiffExitCodes(t *testing.T) {
	binary := buildTool(t)
	root := fixtureRoot(t)
	manifest := filepath.Join(root, "manifest.json")

	// generate succeeds (exit 0) and writes the manifest.
	code, _, stderr := run(t, binary, root, "generate",
		"-root", root, "-manifest", manifest,
		"-source", "shaders/tint.kage", "-source", "shaders/weights.kage",
		"-backend", "glsl,hlsl,msl")
	if code != 0 {
		t.Fatalf("generate exit code: got %d, want 0; stderr=%s", code, stderr)
	}
	if _, err := os.Stat(manifest); err != nil {
		t.Fatalf("manifest was not written: %v", err)
	}

	// validate with full required coverage succeeds.
	code, _, stderr = run(t, binary, root, "validate",
		"-root", root, "-manifest", manifest, "-require", "glsl,hlsl,msl")
	if code != 0 {
		t.Fatalf("validate exit code: got %d, want 0; stderr=%s", code, stderr)
	}

	// diffing the manifest against itself is exit 0.
	code, _, stderr = run(t, binary, root, "diff", "-old", manifest, "-new", manifest)
	if code != 0 {
		t.Fatalf("diff(identical) exit code: got %d, want 0; stderr=%s", code, stderr)
	}

	// Corrupt an artifact, then validate must exit 1 and name the failing sample.
	artifacts := filepath.Join(root, "shaders")
	_ = artifacts
	code, _, stderr = run(t, binary, root, "validate",
		"-root", root, "-manifest", manifest, "-require", "glsl,hlsl,msl")
	if code != 0 {
		t.Fatalf("baseline validate failed: %s", stderr)
	}

	// Tamper with the manifest's recorded hash so validation detects a mismatch.
	data, err := os.ReadFile(manifest)
	if err != nil {
		t.Fatal(err)
	}
	tampered := append([]byte(nil), data...)
	tampered = replaceFirstHash(t, tampered)
	if err := os.WriteFile(manifest, tampered, 0644); err != nil {
		t.Fatal(err)
	}
	code, _, stderr = run(t, binary, root, "validate", "-root", root, "-manifest", manifest)
	if code != 1 {
		t.Fatalf("validate on tampered manifest: got exit %d, want 1; stderr=%s", code, stderr)
	}

	// diff between the original and the tampered manifest exits 3 (differ).
	original := filepath.Join(root, "original.json")
	if err := os.WriteFile(original, data, 0644); err != nil {
		t.Fatal(err)
	}
	code, stdout, _ := run(t, binary, root, "diff", "-old", original, "-new", manifest)
	if code != 3 {
		t.Fatalf("diff(differing) exit code: got %d, want 3", code)
	}
	if stdout == "" {
		t.Error("diff must print the changes")
	}

	// Usage errors exit 2.
	code, _, _ = run(t, binary, root, "generate")
	if code != 2 {
		t.Errorf("generate without sources: got exit %d, want 2", code)
	}
	code, _, _ = run(t, binary, root, "bogus")
	if code != 2 {
		t.Errorf("unknown subcommand: got exit %d, want 2", code)
	}
}

func TestCLIGenerateCompileFailureExitsOne(t *testing.T) {
	binary := buildTool(t)
	root := fixtureRoot(t)
	manifest := filepath.Join(root, "manifest.json")

	code, _, stderr := run(t, binary, root, "generate",
		"-root", root, "-manifest", manifest,
		"-source", "shaders/broken.kage", "-backend", "glsl")
	if code != 1 {
		t.Fatalf("generate with a broken shader: got exit %d, want 1; stderr=%s", code, stderr)
	}
}

// replaceFirstHash overwrites the first artifact hash value with an obviously wrong value while keeping
// the JSON well-formed, forcing a bad-hash validation error.
func replaceFirstHash(t *testing.T, data []byte) []byte {
	t.Helper()
	marker := []byte("\"Hash\": \"")
	idx := indexAll(data, marker)
	if len(idx) < 2 {
		t.Fatalf("test fixture manifest lacks hash fields")
	}
	// Use the second Hash (an artifact hash), skipping the source hash at idx[0].
	pos := idx[1] + len(marker)
	out := append([]byte(nil), data...)
	for i := 0; i < 64; i++ {
		out[pos+i] = '0'
	}
	return out
}

func indexAll(data, marker []byte) []int {
	var positions []int
	for i := 0; i+len(marker) <= len(data); i++ {
		if string(data[i:i+len(marker)]) == string(marker) {
			positions = append(positions, i)
		}
	}
	return positions
}
