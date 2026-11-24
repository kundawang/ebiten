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

package shaderdiag_test

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/shader"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderdiag"
)

var update = flag.Bool("update", false, "update the expected diagnostic JSON files")

// compileFixture compiles a fixture and returns the indented JSON of its diagnostics.
func compileFixture(t *testing.T, path string) (string, *shaderdiag.Diagnostics) {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	_, d := shaderdiag.Compile(src, nil)
	if d == nil {
		t.Fatalf("Compile(%s) must report diagnostics but does not", path)
	}
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return string(b) + "\n", d
}

// TestBrokenShaders compares the diagnostics of intentionally broken shaders against the expected
// JSON. Run with -update to regenerate the expectations.
func TestBrokenShaders(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.kage")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no fixtures found")
	}
	for _, path := range matches {
		name := strings.TrimSuffix(filepath.Base(path), ".kage")
		t.Run(name, func(t *testing.T) {
			got, _ := compileFixture(t, path)
			wantPath := strings.TrimSuffix(path, ".kage") + ".expected.json"
			if *update {
				if err := os.WriteFile(wantPath, []byte(got), 0644); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(wantPath)
			if err != nil {
				t.Fatalf("read the expected diagnostics (run with -update to create it): %v", err)
			}
			if got != string(want) {
				t.Errorf("diagnostics mismatch for %s\ngot:\n%s\nwant:\n%s", path, got, want)
			}
		})
	}
}

// TestDiagnosticsAreSorted checks that multiple errors are ordered by source position.
func TestDiagnosticsAreSorted(t *testing.T) {
	_, d := compileFixture(t, "testdata/recursive.kage")
	all := d.All()
	if len(all) < 2 {
		t.Fatalf("got %d diagnostics, want at least 2", len(all))
	}
	for i := 1; i < len(all); i++ {
		prev, cur := all[i-1], all[i]
		if prev.File > cur.File ||
			(prev.File == cur.File && prev.Line > cur.Line) ||
			(prev.File == cur.File && prev.Line == cur.Line && prev.Column > cur.Column) {
			t.Errorf("diagnostics are not sorted by position: %+v before %+v", prev, cur)
		}
	}
}

// TestLineColumnMapping checks that //line directives map positions to the named files, including
// a nested include that returns to the including file.
func TestLineColumnMapping(t *testing.T) {
	_, d := compileFixture(t, "testdata/include.kage")
	var found bool
	for _, diag := range d.All() {
		if diag.File == "common/inner.kage" {
			found = true
			if diag.Line != 20 {
				t.Errorf("line for the nested include: got %d, want 20", diag.Line)
			}
			if len(diag.CallStack) == 0 || diag.CallStack[0].Function != "helper" {
				t.Errorf("call stack for the nested include: got %+v, want helper", diag.CallStack)
			}
			if diag.CallStack[0].File != "common/util.kage" || diag.CallStack[0].Line != 10 {
				t.Errorf("call stack position: got %+v, want common/util.kage:10", diag.CallStack[0])
			}
		}
	}
	if !found {
		t.Errorf("no diagnostic mapped to common/inner.kage: %v", d)
	}
}

// TestUnicodeColumns checks that positions on lines containing multi-byte runes are reported.
func TestUnicodeColumns(t *testing.T) {
	_, d := compileFixture(t, "testdata/typecheck.kage")
	for _, diag := range d.All() {
		if diag.Line == 0 || diag.Column == 0 {
			t.Errorf("diagnostic without a position: %+v", diag)
		}
	}
}

// TestStages checks that parse, IR, and backend failures are attributed to their stages.
func TestStages(t *testing.T) {
	_, d := compileFixture(t, "testdata/syntax.kage")
	if got := d.All()[0].Stage; got != shaderdiag.StageParse {
		t.Errorf("stage: got %s, want %s", got, shaderdiag.StageParse)
	}
	if got := d.All()[0].Code; got != shaderdiag.CodeSyntax {
		t.Errorf("code: got %s, want %s", got, shaderdiag.CodeSyntax)
	}

	_, d = compileFixture(t, "testdata/no_fragment.kage")
	if got := d.All()[0].Stage; got != shaderdiag.StageIR {
		t.Errorf("stage: got %s, want %s", got, shaderdiag.StageIR)
	}
	if got := d.All()[0].Code; got != shaderdiag.CodeIR {
		t.Errorf("code: got %s, want %s", got, shaderdiag.CodeIR)
	}
}

// TestBackendDiagnostics checks backend failures: an unknown backend name and a nil program that
// makes a backend panic must result in diagnostics, not a panic.
func TestBackendDiagnostics(t *testing.T) {
	d := shaderdiag.CompileBackends(nil, []string{"spirv"}, nil)
	if d == nil || d.Len() != 1 {
		t.Fatalf("got %v, want 1 diagnostic", d)
	}
	if got := d.All()[0].Code; got != shaderdiag.CodeUnknownBackend {
		t.Errorf("code: got %s, want %s", got, shaderdiag.CodeUnknownBackend)
	}

	d = shaderdiag.CompileBackends(nil, []string{"glsl", "hlsl", "msl", "pssl"}, nil)
	if d == nil || d.Len() != 4 {
		t.Fatalf("got %v, want 4 diagnostics", d)
	}
	for _, diag := range d.All() {
		if diag.Code != shaderdiag.CodeBackendFailure {
			t.Errorf("code: got %s, want %s", diag.Code, shaderdiag.CodeBackendFailure)
		}
		if diag.Stage != shaderdiag.StageBackend {
			t.Errorf("stage: got %s, want %s", diag.Stage, shaderdiag.StageBackend)
		}
		if diag.BackendLog == "" {
			t.Errorf("backend log must hold the raw backend output: %+v", diag)
		}
	}
}

// TestValidShader checks that a valid shader compiles for every backend without diagnostics.
func TestValidShader(t *testing.T) {
	const src = `//kage:unit pixels

package main

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return color
}
`
	ir, d := shaderdiag.Compile([]byte(src), nil)
	if d != nil {
		t.Fatalf("Compile must not report diagnostics but reported %v", d)
	}
	if ir == nil {
		t.Fatal("Compile must return a program")
	}
	if d := shaderdiag.CompileBackends(ir, nil, nil); d != nil {
		t.Fatalf("CompileBackends must not report diagnostics but reported %v", d)
	}
}

// TestErrorCompatibility checks that Diagnostics keeps working as a plain error and still unwraps
// to the original compiler error.
func TestErrorCompatibility(t *testing.T) {
	src, err := os.ReadFile("testdata/typecheck.kage")
	if err != nil {
		t.Fatal(err)
	}
	_, err2 := compileErr(src)
	if err2 == nil {
		t.Fatal("compile must fail")
	}
	var d *shaderdiag.Diagnostics
	if !errors.As(err2, &d) {
		t.Fatalf("the error must be a *Diagnostics: %T", err2)
	}
	var perr *shader.ParseError
	if !errors.As(err2, &perr) {
		t.Errorf("the error must unwrap to a *shader.ParseError: %v", err2)
	}
	// The error text must keep the traditional "file:line:column: message" form.
	if !strings.Contains(d.Error(), ": ") {
		t.Errorf("unexpected error text: %q", d.Error())
	}
}

func compileErr(src []byte) (*shaderdiag.Diagnostics, error) {
	_, d := shaderdiag.Compile(src, nil)
	if d == nil {
		return nil, nil
	}
	return d, d
}

// TestDuplicateDiagnostics checks that identical diagnostics are reported only once.
func TestDuplicateDiagnostics(t *testing.T) {
	d := shaderdiag.New(nil, nil)
	diag := shaderdiag.Diagnostic{
		Code:    shaderdiag.CodeTypeCheck,
		Stage:   shaderdiag.StageTypeCheck,
		Message: "duplicated",
		File:    "a.kage",
		Line:    1,
		Column:  1,
	}
	d.Add(diag)
	d.Add(diag)
	if got := d.Len(); got != 1 {
		t.Errorf("got %d diagnostics, want 1", got)
	}
}

// TestOverlongDiagnostics checks that long messages and logs are truncated and that diagnostics
// beyond the limit are dropped and counted.
func TestOverlongDiagnostics(t *testing.T) {
	d := shaderdiag.New(nil, &shaderdiag.Options{
		MaxDiagnostics:     2,
		MaxMessageBytes:    16,
		MaxBackendLogBytes: 8,
	})
	d.Add(shaderdiag.Diagnostic{
		Code:       shaderdiag.CodeBackendFailure,
		Stage:      shaderdiag.StageBackend,
		Message:    strings.Repeat("m", 100),
		BackendLog: strings.Repeat("l", 100),
	})
	// A multi-byte rune must not be split at the cut point.
	d.Add(shaderdiag.Diagnostic{
		Code:    shaderdiag.CodeSyntax,
		Stage:   shaderdiag.StageParse,
		Message: strings.Repeat("あ", 10),
	})
	d.Add(shaderdiag.Diagnostic{Code: shaderdiag.CodeIR, Stage: shaderdiag.StageIR, Message: "dropped"})

	if got := d.Len(); got != 2 {
		t.Fatalf("got %d diagnostics, want 2", got)
	}
	if got := d.Truncated(); got != 1 {
		t.Errorf("got %d truncated, want 1", got)
	}
	all := d.All()
	var backend, unicode *shaderdiag.Diagnostic
	for i := range all {
		switch all[i].Code {
		case shaderdiag.CodeBackendFailure:
			backend = &all[i]
		case shaderdiag.CodeSyntax:
			unicode = &all[i]
		}
	}
	if backend == nil || unicode == nil {
		t.Fatalf("expected diagnostics not found: %+v", all)
	}
	if !strings.HasSuffix(backend.Message, "(truncated)") {
		t.Errorf("message must be truncated: %q", backend.Message)
	}
	if !strings.HasSuffix(backend.BackendLog, "(truncated)") {
		t.Errorf("backend log must be truncated: %q", backend.BackendLog)
	}
	for _, r := range unicode.Message {
		if r == '�' {
			t.Errorf("message must not contain a replacement rune: %q", unicode.Message)
		}
	}
}

// TestRedaction checks that sensitive paths are redacted from file names and messages.
func TestRedaction(t *testing.T) {
	const src = `//kage:unit pixels

package main

//line /home/alice/secret/project/shader.kage:10
func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
	return unknownVar
}
`
	_, d := shaderdiag.Compile([]byte(src), &shaderdiag.Options{
		Redactor: shaderdiag.PrefixRedactor(map[string]string{
			"/home/alice/secret": "<redacted>",
		}),
	})
	if d == nil {
		t.Fatal("Compile must report diagnostics")
	}
	for _, diag := range d.All() {
		if strings.Contains(diag.File, "alice") || strings.Contains(diag.Message, "alice") {
			t.Errorf("diagnostic must be redacted: %+v", diag)
		}
		for _, f := range diag.CallStack {
			if strings.Contains(f.File, "alice") {
				t.Errorf("call stack must be redacted: %+v", f)
			}
		}
	}
	if !strings.HasPrefix(d.All()[0].File, "<redacted>/") {
		t.Errorf("unexpected redacted file: %q", d.All()[0].File)
	}
}

// TestNoSharedMutableState checks that returned diagnostics cannot mutate the Diagnostics.
func TestNoSharedMutableState(t *testing.T) {
	d := shaderdiag.New(nil, nil)
	d.Add(shaderdiag.Diagnostic{
		Code:      shaderdiag.CodeTypeCheck,
		Stage:     shaderdiag.StageTypeCheck,
		Message:   "original",
		CallStack: []shaderdiag.Frame{{Function: "f"}},
	})
	all := d.All()
	all[0].Message = "mutated"
	all[0].CallStack[0].Function = "mutated"
	if got := d.All()[0].Message; got != "original" {
		t.Errorf("message was mutated through the returned slice: %q", got)
	}
	if got := d.All()[0].CallStack[0].Function; got != "f" {
		t.Errorf("call stack was mutated through the returned slice: %q", got)
	}
}

// TestConcurrentCompile checks that concurrent compilations neither race nor share state, and
// report identical diagnostics.
func TestConcurrentCompile(t *testing.T) {
	src, err := os.ReadFile("testdata/typecheck.kage")
	if err != nil {
		t.Fatal(err)
	}
	_, want := shaderdiag.Compile(src, nil)
	if want == nil {
		t.Fatal("Compile must report diagnostics")
	}
	wantJSON, err := want.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	const n = 16
	var wg sync.WaitGroup
	results := make([][]byte, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, d := shaderdiag.Compile(src, nil)
			if d == nil {
				t.Error("Compile must report diagnostics")
				return
			}
			b, err := d.MarshalJSON()
			if err != nil {
				t.Error(err)
				return
			}
			results[i] = b
		}()
	}
	wg.Wait()
	for i, b := range results {
		if string(b) != string(wantJSON) {
			t.Errorf("result %d differs from the sequential result", i)
		}
	}
}
