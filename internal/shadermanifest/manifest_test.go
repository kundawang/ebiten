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

package shadermanifest_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/legacyshader"
	"github.com/hajimehoshi/ebiten/v2/internal/shadermanifest"
)

// allTextBackends are the backends Generate can produce without an external toolchain.
var allTextBackends = []shadermanifest.Backend{
	shadermanifest.BackendGLSL,
	shadermanifest.BackendGLSLES,
	shadermanifest.BackendHLSL,
	shadermanifest.BackendMSL,
}

func generateFixture(t *testing.T, root string, gzip bool) *shadermanifest.Manifest {
	t.Helper()
	m, errs := shadermanifest.Generate(root, []shadermanifest.ShaderInput{
		{SourcePath: "shaders/tint.kage"},
		{SourcePath: "shaders/weights.kage"},
	}, &shadermanifest.GenerateOptions{Backends: allTextBackends, Gzip: gzip})
	if errs != nil {
		t.Fatalf("Generate returned errors: %v", errs)
	}
	return m
}

func TestGenerateAndValidateRoundTrip(t *testing.T) {
	root := copyTestdata(t)

	m := generateFixture(t, root, false)

	if got, want := m.Version, shadermanifest.CurrentVersion; got != want {
		t.Errorf("Version: got %d, want %d", got, want)
	}
	if got, want := len(m.Entries), 2; got != want {
		t.Fatalf("entries: got %d, want %d", got, want)
	}

	// Each entry must cover every requested backend with all required roles.
	for _, spec := range m.Entries {
		roleCount := map[shadermanifest.Backend]int{}
		for _, a := range spec.Artifacts {
			roleCount[a.Backend]++
		}
		for _, b := range allTextBackends {
			if got, want := roleCount[b], len(shadermanifest.Roles(b)); got != want {
				t.Errorf("entry %s backend %s: got %d artifacts, want %d", spec.SourceID, b, got, want)
			}
		}
		// The recorded source ID must match what Ebitengine computes at runtime.
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(spec.SourcePath)))
		if err != nil {
			t.Fatal(err)
		}
		wantID, err := legacyshader.CalcSourceID(data)
		if err != nil {
			t.Fatal(err)
		}
		if spec.SourceID != wantID.String() {
			t.Errorf("SourceID: got %q, want %q", spec.SourceID, wantID.String())
		}
		if len(spec.Uniforms) == 0 {
			t.Errorf("entry %s recorded no uniforms", spec.SourceID)
		}
	}

	// Every artifact file must exist on disk.
	for _, spec := range m.Entries {
		for _, a := range spec.Artifacts {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(a.Path))); err != nil {
				t.Errorf("artifact missing on disk: %s: %v", a.Path, err)
			}
		}
	}

	// Validation must pass on the freshly generated tree, with full coverage.
	cov, errs := shadermanifest.Validate(root, m, &shadermanifest.ValidateOptions{RequireBackends: allTextBackends})
	if errs != nil {
		t.Fatalf("Validate returned errors: %v", errs)
	}
	for _, b := range allTextBackends {
		if cov.Covered[b] != 2 {
			t.Errorf("coverage for %s: got %d, want 2", b, cov.Covered[b])
		}
	}
	if len(cov.Missing) != 0 {
		t.Errorf("unexpected missing coverage: %v", cov.Missing)
	}
}

func TestUniformLayoutRecorded(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	var weights *shadermanifest.Spec
	for _, s := range m.Entries {
		if strings.HasSuffix(s.SourcePath, "weights.kage") {
			weights = s
		}
	}
	if weights == nil {
		t.Fatal("weights.kage entry not found")
	}
	found := false
	for _, u := range weights.Uniforms {
		if u.Name == "Weights" {
			found = true
			// [4]float occupies four dwords.
			if u.Size != 4 {
				t.Errorf("Weights size: got %d, want 4", u.Size)
			}
			if u.Type != "[4]float" {
				t.Errorf("Weights type: got %q, want [4]float", u.Type)
			}
		}
	}
	if !found {
		t.Errorf("Weights uniform not recorded: %+v", weights.Uniforms)
	}
}

func TestJSONStability(t *testing.T) {
	root := copyTestdata(t)

	first := generateFixture(t, root, false)
	data1, err := first.MarshalText()
	if err != nil {
		t.Fatal(err)
	}

	// Re-generate into a second, independently populated tree.
	root2 := copyTestdata(t)
	second := generateFixture(t, root2, false)
	data2, err := second.MarshalText()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(data1, data2) {
		t.Errorf("manifest serialization is not stable:\n--- first ---\n%s\n--- second ---\n%s", data1, data2)
	}

	// Canonical formatting: tab indentation, trailing newline, forward-slash paths.
	if !bytes.HasSuffix(data1, []byte("\n")) {
		t.Error("manifest must end with a newline")
	}
	if bytes.Contains(data1, []byte("\r")) {
		t.Error("manifest must use LF line endings")
	}
	if bytes.Contains(data1, []byte("\\\\")) || bytes.Contains(data1, []byte("shaders\\")) {
		t.Error("manifest paths must use forward slashes")
	}
	if !bytes.Contains(data1, []byte("\n\t\"")) {
		t.Error("manifest must be tab-indented")
	}

	// Re-reading the JSON and re-marshaling must reproduce the identical bytes.
	var reloaded shadermanifest.Manifest
	if err := json.Unmarshal(data1, &reloaded); err != nil {
		t.Fatal(err)
	}
	reloaded.Version = shadermanifest.CurrentVersion
	data3, err := reloaded.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data1, data3) {
		t.Error("round-tripping through JSON changed the canonical form")
	}

	// Entries must be sorted by SourceID and artifacts by backend/role.
	var ids []string
	for _, s := range second.Entries {
		ids = append(ids, s.SourceID)
	}
	if !sortedStrings(ids) {
		t.Errorf("entries are not sorted by SourceID: %v", ids)
	}
	for _, s := range second.Entries {
		var keys []string
		for _, a := range s.Artifacts {
			keys = append(keys, string(a.Backend)+"\x00"+a.Role)
		}
		// Canonical order is lexical over backend then role, independent of role semantics.
		if !sortedStrings(keys) {
			t.Errorf("artifacts are not sorted lexically: %v", keys)
		}
	}
}

func TestGzipArtifacts(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, true)

	for _, spec := range m.Entries {
		for _, a := range spec.Artifacts {
			if !a.Gzip {
				t.Fatalf("artifact %s is not marked gzip", a.Path)
			}
			stored, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(a.Path)))
			if err != nil {
				t.Fatal(err)
			}
			// gzip magic number.
			if len(stored) < 2 || stored[0] != 0x1f || stored[1] != 0x8b {
				t.Fatalf("artifact %s is not gzip-compressed", a.Path)
			}
		}
	}

	// Validation must transparently decompress and verify.
	if _, errs := shadermanifest.Validate(root, m, nil); errs != nil {
		t.Fatalf("Validate failed for gzip artifacts: %v", errs)
	}

	// Deterministic compression: regenerate and compare raw bytes of one artifact.
	path := m.Entries[0].Artifacts[0].Path
	first, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
	root2 := copyTestdata(t)
	m2 := generateFixture(t, root2, true)
	var secondPath string
	for _, a := range m2.Entries[0].Artifacts {
		if a.Backend == m.Entries[0].Artifacts[0].Backend && a.Role == m.Entries[0].Artifacts[0].Role {
			secondPath = a.Path
		}
	}
	second, _ := os.ReadFile(filepath.Join(root2, filepath.FromSlash(secondPath)))
	if !bytes.Equal(first, second) {
		t.Error("gzip output is not deterministic")
	}
}

func TestCorruptedArtifactDetected(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	// Corrupt the bytes of one artifact and confirm the hash mismatch is flagged at that sample.
	target := m.Entries[0].Artifacts[0]
	abs := filepath.Join(root, filepath.FromSlash(target.Path))
	if err := os.WriteFile(abs, []byte("corrupted contents"), 0644); err != nil {
		t.Fatal(err)
	}

	_, errs := shadermanifest.Validate(root, m, nil)
	if errs == nil {
		t.Fatal("corruption was not detected")
	}
	if !errs.Has(shadermanifest.CodeBadArtifactHash) {
		t.Fatalf("expected %s, got %v", shadermanifest.CodeBadArtifactHash, errs.Codes())
	}
	found := false
	for _, e := range errs {
		if e.Code == shadermanifest.CodeBadArtifactHash && e.Path == target.Path {
			found = true
			if e.Backend != target.Backend || e.Role != target.Role {
				t.Errorf("error context mismatch: backend=%q role=%q", e.Backend, e.Role)
			}
		}
	}
	if !found {
		t.Errorf("corrupted sample not located by path %q in errors", target.Path)
	}

	// The corrupted entry's backend must not count as covered. With two entries, only the untouched
	// entry still covers it, so coverage is 1 rather than 2.
	cov, _ := shadermanifest.Validate(root, m, nil)
	if got, want := cov.Covered[target.Backend], 1; got != want {
		t.Errorf("corrupted backend coverage: got %d, want %d (only the intact entry)", got, want)
	}
}

func TestMissingArtifactDetected(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	target := m.Entries[0].Artifacts[0]
	if err := os.Remove(filepath.Join(root, filepath.FromSlash(target.Path))); err != nil {
		t.Fatal(err)
	}

	_, errs := shadermanifest.Validate(root, m, nil)
	if errs == nil || !errs.Has(shadermanifest.CodeMissingArtifact) {
		t.Fatalf("expected %s, got %v", shadermanifest.CodeMissingArtifact, errs)
	}
}

func TestPartialCoverageNotReportedAsCovered(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	// Drop the fragment role of GLSL from one entry's manifest while leaving the file on disk. This is a
	// genuinely partial backend (the vertex role remains), distinct from a missing file.
	var kept []*shadermanifest.Artifact
	var dropped bool
	for _, a := range m.Entries[0].Artifacts {
		if a.Backend == shadermanifest.BackendGLSL && a.Role == "fragment" {
			dropped = true
			continue
		}
		kept = append(kept, a)
	}
	if !dropped {
		t.Fatal("GLSL fragment artifact not found")
	}
	m.Entries[0].Artifacts = kept

	cov, errs := shadermanifest.Validate(root, m, &shadermanifest.ValidateOptions{
		RequireBackends: []shadermanifest.Backend{shadermanifest.BackendGLSL},
	})
	if errs == nil || !errs.Has(shadermanifest.CodePartialArtifact) {
		t.Fatalf("expected partial-artifact error, got %v", errs)
	}
	if cov.Covered[shadermanifest.BackendGLSL] != 1 {
		t.Errorf("only the intact entry should count as covered, got %d", cov.Covered[shadermanifest.BackendGLSL])
	}
	if cov.Missing[shadermanifest.BackendGLSL] != 1 {
		t.Errorf("the partial entry must be reported missing, got %d", cov.Missing[shadermanifest.BackendGLSL])
	}
}

func TestMissingRequiredBackendNotReportedAsCovered(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	// Require a backend that no entry provides at all. That must fail rather than pass vacuously.
	cov, errs := shadermanifest.Validate(root, m, &shadermanifest.ValidateOptions{
		RequireBackends: []shadermanifest.Backend{shadermanifest.BackendMetalMacOS},
	})
	if errs == nil || !errs.Has(shadermanifest.CodePartialArtifact) {
		t.Fatalf("expected partial-artifact error for absent backend, got %v", errs)
	}
	if cov.Missing[shadermanifest.BackendMetalMacOS] != len(m.Entries) {
		t.Errorf("every entry must be reported missing for an absent backend")
	}
}

func TestMissingSourceDetected(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	if err := os.Remove(filepath.Join(root, "shaders", "tint.kage")); err != nil {
		t.Fatal(err)
	}
	_, errs := shadermanifest.Validate(root, m, nil)
	if errs == nil || !errs.Has(shadermanifest.CodeMissingSource) {
		t.Fatalf("expected %s, got %v", shadermanifest.CodeMissingSource, errs)
	}
}

func TestBadSourceHashDetected(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)

	// Tamper with the source without regenerating.
	src := filepath.Join(root, filepath.FromSlash(m.Entries[0].SourcePath))
	original, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	tampered := append(bytes.Clone(original), []byte("\n// tampered")...)
	if err := os.WriteFile(src, tampered, 0644); err != nil {
		t.Fatal(err)
	}

	_, errs := shadermanifest.Validate(root, m, nil)
	if errs == nil {
		t.Fatal("tampered source was not detected")
	}
	if !errs.Has(shadermanifest.CodeBadSourceHash) {
		t.Errorf("expected bad source hash, got %v", errs.Codes())
	}
	if !errs.Has(shadermanifest.CodeBadSourceID) {
		t.Errorf("expected bad source ID, got %v", errs.Codes())
	}
}

func TestUnknownVersionRejected(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)
	m.Version = 999

	_, errs := shadermanifest.Validate(root, m, nil)
	if errs == nil || !errs.Has(shadermanifest.CodeUnknownVersion) {
		t.Fatalf("expected unknown version error, got %v", errs)
	}
	if err := m.Save(filepath.Join(root, "x.json")); err == nil {
		t.Error("Save must reject an unsupported version")
	}
}

func TestOversizedSourceRejected(t *testing.T) {
	root := copyTestdata(t)

	big := strings.Repeat("// padding line to exceed the tiny limit\n", 5)
	bigPath := filepath.Join(root, "shaders", "big.kage")
	if err := os.WriteFile(bigPath, []byte(big), 0644); err != nil {
		t.Fatal(err)
	}
	_, errs := shadermanifest.Generate(root,
		[]shadermanifest.ShaderInput{{SourcePath: "shaders/big.kage"}},
		&shadermanifest.GenerateOptions{MaxSourceSize: 32})
	if errs == nil || !errs.Has(shadermanifest.CodeSourceTooLarge) {
		t.Fatalf("expected source-too-large error, got %v", errs)
	}
}

func TestCompileFailureInjection(t *testing.T) {
	root := copyTestdata(t)

	// One good and one broken shader: generation must keep going and report the broken sample.
	m, errs := shadermanifest.Generate(root, []shadermanifest.ShaderInput{
		{SourcePath: "shaders/tint.kage"},
		{SourcePath: "shaders/broken.kage"},
	}, &shadermanifest.GenerateOptions{Backends: []shadermanifest.Backend{shadermanifest.BackendGLSL}})
	if errs == nil || !errs.Has(shadermanifest.CodeCompileFailed) {
		t.Fatalf("expected compile failure, got %v", errs)
	}
	// The good shader must still have been produced.
	if len(m.Entries) != 1 {
		t.Fatalf("got %d entries, want 1 (the broken one is skipped)", len(m.Entries))
	}
}

func TestDuplicateAndMacroVariants(t *testing.T) {
	root := copyTestdata(t)

	// The same source with two distinct macro sets is allowed; an exact duplicate is rejected.
	_, errs := shadermanifest.Generate(root, []shadermanifest.ShaderInput{
		{SourcePath: "shaders/tint.kage", Macros: map[string]string{"QUALITY": "low"}},
		{SourcePath: "shaders/tint.kage", Macros: map[string]string{"QUALITY": "high"}},
		{SourcePath: "shaders/tint.kage", Macros: map[string]string{"QUALITY": "low"}},
	}, &shadermanifest.GenerateOptions{Backends: []shadermanifest.Backend{shadermanifest.BackendGLSL}})
	if errs == nil || !errs.Has(shadermanifest.CodeDuplicateSpec) {
		t.Fatalf("expected duplicate-spec error, got %v", errs)
	}
	// Count the duplicate error exactly once while the two distinct variants succeed.
	dupCount := 0
	for _, e := range errs {
		if e.Code == shadermanifest.CodeDuplicateSpec {
			dupCount++
		}
	}
	if dupCount != 1 {
		t.Errorf("duplicate count: got %d, want 1", dupCount)
	}

	// Macro variants get distinct, stable artifact directories.
	m, errs := shadermanifest.Generate(root, []shadermanifest.ShaderInput{
		{SourcePath: "shaders/tint.kage", Macros: map[string]string{"QUALITY": "low"}},
		{SourcePath: "shaders/tint.kage", Macros: map[string]string{"QUALITY": "high"}},
	}, &shadermanifest.GenerateOptions{Backends: []shadermanifest.Backend{shadermanifest.BackendGLSL}})
	if errs != nil {
		t.Fatal(errs)
	}
	paths := map[string]bool{}
	for _, s := range m.Entries {
		for _, a := range s.Artifacts {
			paths[a.Path] = true
			if !strings.Contains(a.Path, "QUALITY-low") && !strings.Contains(a.Path, "QUALITY-high") {
				t.Errorf("macro artifact path missing macro tag: %s", a.Path)
			}
		}
	}
	if len(paths) != 4 { // 2 variants x 2 GLSL roles
		t.Errorf("got %d distinct artifact paths, want 4", len(paths))
	}
}

func TestAddBinaryArtifact(t *testing.T) {
	root := copyTestdata(t)
	m := generateFixture(t, root, false)
	spec := m.Entries[0]

	// Simulate an externally compiled DXBC binary for Windows (vertex + pixel).
	vx := []byte("DXBC-vertex-bytes")
	px := []byte("DXBC-pixel-bytes")
	if _, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.BackendDXBCWindows, "vertex", vx, false); errs != nil {
		t.Fatal(errs)
	}
	if _, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.BackendDXBCWindows, "pixel", px, false); errs != nil {
		t.Fatal(errs)
	}

	// Adding the same role again must fail rather than silently overwrite.
	if _, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.BackendDXBCWindows, "vertex", vx, false); errs == nil {
		t.Fatal("duplicate AddArtifact must fail")
	}

	// An unknown backend and an invalid role must be rejected.
	if _, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.Backend("nope"), "vertex", vx, false); errs == nil {
		t.Error("unknown backend must be rejected")
	}
	if _, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.BackendDXBCWindows, "frobnicate", vx, false); errs == nil {
		t.Error("invalid role must be rejected")
	}

	// Without requiring it across all entries, the binary backend validates as complete for its entry.
	cov, errs := shadermanifest.Validate(root, m, nil)
	if errs != nil {
		t.Fatalf("validating binary artifact failed: %v", errs)
	}
	if cov.Covered[shadermanifest.BackendDXBCWindows] != 1 {
		t.Errorf("DXBC coverage: got %d, want 1", cov.Covered[shadermanifest.BackendDXBCWindows])
	}

	// Requiring the binary backend across all entries fails until every entry provides it.
	if _, errs := shadermanifest.Validate(root, m, &shadermanifest.ValidateOptions{
		RequireBackends: []shadermanifest.Backend{shadermanifest.BackendDXBCWindows},
	}); errs == nil {
		t.Error("required binary backend must fail while only one entry covers it")
	}
}

func TestDuplicateContentAtDifferentPathsDetected(t *testing.T) {
	root := copyTestdata(t)

	// Copy tint.kage to a second path. Both files have identical bytes, hence the same SourceID and an
	// empty macro set, so the second must be reported as a duplicate rather than a second entry.
	data, err := os.ReadFile(filepath.Join(root, "shaders", "tint.kage"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "shaders", "tint-copy.kage"), data, 0644); err != nil {
		t.Fatal(err)
	}
	m, errs := shadermanifest.Generate(root, []shadermanifest.ShaderInput{
		{SourcePath: "shaders/tint.kage"},
		{SourcePath: "shaders/tint-copy.kage"},
	}, &shadermanifest.GenerateOptions{Backends: []shadermanifest.Backend{shadermanifest.BackendGLSL}})
	if errs == nil || !errs.Has(shadermanifest.CodeDuplicateSpec) {
		t.Fatalf("expected duplicate-spec error, got %v", errs)
	}
	if len(m.Entries) != 1 {
		t.Errorf("got %d entries, want exactly one for identical content", len(m.Entries))
	}
}

func TestAddBinaryArtifactCustomDir(t *testing.T) {
	root := copyTestdata(t)
	m, errs := shadermanifest.Generate(root,
		[]shadermanifest.ShaderInput{{SourcePath: "shaders/tint.kage"}},
		&shadermanifest.GenerateOptions{
			Backends:    []shadermanifest.Backend{shadermanifest.BackendGLSL},
			ArtifactDir: "build/gpu",
		})
	if errs != nil {
		t.Fatal(errs)
	}
	spec := m.Entries[0]

	// An externally added binary artifact must land in the same custom directory tree.
	art, errs := m.AddArtifact(root, spec.SourceID, nil, shadermanifest.BackendMetalMacOS, "library", []byte("metallib"), false)
	if errs != nil {
		t.Fatal(errs)
	}
	if !strings.HasPrefix(art.Path, "build/gpu/") {
		t.Fatalf("artifact path %q is not under the custom directory build/gpu", art.Path)
	}
	if _, errs := shadermanifest.Validate(root, m, nil); errs != nil {
		t.Fatalf("validating custom-dir artifact failed: %v", errs)
	}
}

func TestUnsafePathsRejected(t *testing.T) {
	root := copyTestdata(t)
	bad := []string{"", "/abs/path.kage", "../escape.kage", "dir/../../escape.kage"}
	for _, p := range bad {
		_, errs := shadermanifest.Generate(root,
			[]shadermanifest.ShaderInput{{SourcePath: p}},
			&shadermanifest.GenerateOptions{Backends: []shadermanifest.Backend{shadermanifest.BackendGLSL}})
		if errs == nil || !errs.Has(shadermanifest.CodeInvalidPath) {
			t.Errorf("path %q must be rejected, got %v", p, errs)
		}
	}
}

func TestDiffDetectsChanges(t *testing.T) {
	root := copyTestdata(t)
	oldM := generateFixture(t, root, false)

	root2 := copyTestdata(t)
	newM := generateFixture(t, root2, false)

	// Identical manifests -> empty report.
	if r := shadermanifest.Diff(oldM, newM); !r.Empty() {
		t.Fatalf("identical manifests produced a diff:\n%s", r.Text())
	}

	// Remove an artifact and change a source hash.
	newM.Entries[0].Artifacts = newM.Entries[0].Artifacts[1:]
	newM.Entries[1].SourceHash = "deadbeef"

	report := shadermanifest.Diff(oldM, newM)
	if report.Empty() {
		t.Fatal("expected differences, got an empty report")
	}
	kinds := map[shadermanifest.ChangeKind]bool{}
	for _, c := range report.Changes {
		kinds[c.Kind] = true
	}
	if !kinds[shadermanifest.ChangeArtifactRemoved] {
		t.Errorf("missing artifact-removed change: %v", kinds)
	}
	if !kinds[shadermanifest.ChangeSourceChanged] {
		t.Errorf("missing source-changed change: %v", kinds)
	}
	if !strings.Contains(report.Text(), string(shadermanifest.ChangeArtifactRemoved)) {
		t.Error("text report must enumerate change kinds")
	}
}

func TestLoadMalformedAndMissingManifest(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{not json"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := shadermanifest.Load(bad); err == nil {
		t.Fatal("malformed manifest must fail to load")
	} else if se, ok := err.(*shadermanifest.SampleError); !ok || se.Code != shadermanifest.CodeMalformedManifest {
		t.Fatalf("expected malformed-manifest error, got %v", err)
	}
	if _, err := shadermanifest.Load(filepath.Join(dir, "absent.json")); err == nil {
		t.Fatal("missing manifest file must fail to load")
	}
}

// copyTestdata copies the package's testdata tree into a fresh temp directory so tests can freely
// mutate sources and artifacts without touching the committed fixtures.
func copyTestdata(t *testing.T) string {
	t.Helper()
	dst := t.TempDir()
	src := "testdata"
	err := filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return dst
}

func sortedStrings(s []string) bool {
	for i := 1; i < len(s); i++ {
		if s[i-1] > s[i] {
			return false
		}
	}
	return true
}
