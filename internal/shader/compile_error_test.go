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

package shader_test

import (
	"errors"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/shader"
)

func TestParseErrorCompileErrors(t *testing.T) {
	const src = `package main

func helper() vec4 {
	return vec4(unknownVar)
}
`
	_, err := shader.Compile([]byte(src), "Vertex", "Fragment", 0)
	if err == nil {
		t.Fatal("Compile must return an error")
	}
	var perr *shader.ParseError
	if !errors.As(err, &perr) {
		t.Fatalf("the error must be a *shader.ParseError: %T", err)
	}

	ces := perr.CompileErrors()
	if len(ces) != 1 {
		t.Fatalf("got %d errors, want 1", len(ces))
	}
	ce := ces[0]
	if ce.Function != "helper" {
		t.Errorf("function: got %q, want %q", ce.Function, "helper")
	}
	if ce.FunctionPosition.Line != 3 {
		t.Errorf("function position line: got %d, want 3", ce.FunctionPosition.Line)
	}
	if ce.Position.Line != 4 {
		t.Errorf("position line: got %d, want 4", ce.Position.Line)
	}
	if len(perr.Positions()) != len(ces) {
		t.Errorf("Positions and CompileErrors must have the same length")
	}

	// The result must be a copy.
	ces[0].Message = "mutated"
	if perr.CompileErrors()[0].Message == "mutated" {
		t.Errorf("CompileErrors must return a copy")
	}
}
