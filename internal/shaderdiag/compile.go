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

package shaderdiag

import (
	"bytes"
	"errors"
	"fmt"
	"go/scanner"
	"go/token"
	"runtime/debug"
	"slices"
	"strings"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/legacyshader"
	"github.com/hajimehoshi/ebiten/v2/internal/shader"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/glsl"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/hlsl"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/msl"
	"github.com/hajimehoshi/ebiten/v2/internal/shaderir/pssl"
)

// Options configures [Compile] and [CompileBackends]. The zero value is valid and compiles for all
// backends without redaction.
type Options struct {
	// Backends lists the shader language backends to generate code for. The supported names are
	// "glsl", "glsl-es", "hlsl", "msl", and "pssl". An unknown name is reported as a diagnostic
	// with the code KAGE4001 instead of failing the compilation. When nil, all supported backends
	// are used.
	Backends []string

	// Redactor, when non-nil, rewrites sensitive text such as local paths in file names, messages,
	// macro contexts, and backend logs. See [PrefixRedactor].
	Redactor func(string) string

	// MaxDiagnostics is the maximum number of diagnostics kept. Zero or negative selects
	// [DefaultMaxDiagnostics].
	MaxDiagnostics int

	// MaxMessageBytes is the maximum length of a diagnostic message in bytes. Zero or negative
	// selects [DefaultMaxMessageBytes].
	MaxMessageBytes int

	// MaxBackendLogBytes is the maximum length of a raw backend log in bytes. Zero or negative
	// selects [DefaultMaxBackendLogBytes].
	MaxBackendLogBytes int
}

// DefaultBackends lists the backends compiled by default, in order. The "pssl" backend is
// supported but excluded, as its implementation is a stub that always fails.
var DefaultBackends = []string{"glsl", "glsl-es", "hlsl", "msl"}

// resolvedOptions is Options with defaults applied.
type resolvedOptions struct {
	Options
	backends []string
}

func resolveOptions(opts *Options) resolvedOptions {
	var o Options
	if opts != nil {
		o = *opts
		o.Backends = slices.Clone(opts.Backends)
	}
	if o.MaxDiagnostics <= 0 {
		o.MaxDiagnostics = DefaultMaxDiagnostics
	}
	if o.MaxMessageBytes <= 0 {
		o.MaxMessageBytes = DefaultMaxMessageBytes
	}
	if o.MaxBackendLogBytes <= 0 {
		o.MaxBackendLogBytes = DefaultMaxBackendLogBytes
	}
	backends := o.Backends
	if backends == nil {
		backends = slices.Clone(DefaultBackends)
	}
	return resolvedOptions{Options: o, backends: backends}
}

// PrefixRedactor returns a redactor that replaces each given path prefix with its replacement
// string. Longer prefixes are replaced first, so nested paths redact to the most specific
// replacement.
func PrefixRedactor(replacements map[string]string) func(string) string {
	type entry struct {
		prefix string
		repl   string
	}
	entries := make([]entry, 0, len(replacements))
	for p, r := range replacements {
		entries = append(entries, entry{p, r})
	}
	slices.SortStableFunc(entries, func(a, b entry) int {
		if c := len(b.prefix) - len(a.prefix); c != 0 {
			return c
		}
		return strings.Compare(a.prefix, b.prefix)
	})
	return func(s string) string {
		for _, e := range entries {
			s = strings.ReplaceAll(s, e.prefix, e.repl)
		}
		return s
	}
}

// Compile compiles a Kage shader source and reports every error found as structured diagnostics.
//
// The returned program is nil unless the front-end stages (parse, type check, and IR) succeeded.
// Backend failures are reported without discarding the program. The returned Diagnostics is nil
// when no error was found.
//
// Compile is safe for concurrent use; no state is shared between calls.
func Compile(src []byte, opts *Options) (*shaderir.Program, *Diagnostics) {
	o := resolveOptions(opts)

	if len(bytes.TrimSpace(src)) == 0 {
		d := New(nil, &o.Options)
		d.Add(Diagnostic{
			Code:    CodeEmptySource,
			Stage:   StageParse,
			Message: "the shader source is empty",
		})
		return nil, d
	}

	unit, err := legacyshader.ParseCompilerDirectives(src)
	macro := macroContext(src, unit)
	if err != nil {
		d := New(err, &o.Options)
		d.Add(Diagnostic{
			Code:         CodeInvalidDirective,
			Stage:        StageParse,
			Message:      err.Error(),
			MacroContext: macro,
		})
		return nil, d
	}

	ir, _, err := legacyshader.CompileShader(src)
	if err != nil {
		d := New(err, &o.Options)
		d.addFrontEndError(err, macro)
		return nil, d
	}

	d := New(nil, &o.Options)
	d.addBackendDiagnostics(ir, o.backends)
	if d.Len() == 0 {
		return ir, nil
	}
	return ir, d
}

// CompileBackends generates code for the given backends and reports the failures as diagnostics.
// A nil program, an unknown backend name, or a panicking backend results in a diagnostic, never a
// panic. The result is nil when every backend succeeded.
//
// CompileBackends is safe for concurrent use.
func CompileBackends(p *shaderir.Program, backends []string, opts *Options) *Diagnostics {
	o := resolveOptions(opts)
	if backends == nil {
		backends = o.backends
	}
	d := New(nil, &o.Options)
	d.addBackendDiagnostics(p, backends)
	if d.Len() == 0 {
		return nil
	}
	return d
}

// macroContext describes the source transformations the compiler applies before parsing.
func macroContext(src []byte, unit legacyshader.Unit) string {
	var b strings.Builder
	switch unit {
	case legacyshader.Pixels:
		b.WriteString("//kage:unit pixels")
	default:
		b.WriteString("//kage:unit texels")
	}
	if unit == legacyshader.Texels {
		b.WriteString("; the source is converted from texels to pixels and the deprecated builtin definitions are appended, so reported positions refer to the converted source")
	}
	if bytes.Contains(src, []byte(shader.InternalRegionBegin)) {
		b.WriteString("; an internal region (//kage:internalregion) is present")
	}
	return b.String()
}

// addFrontEndError converts an error returned by the front-end into diagnostics.
func (d *Diagnostics) addFrontEndError(err error, macro string) {
	var perr *shader.ParseError
	var elist scanner.ErrorList
	switch {
	case errors.As(err, &perr):
		for _, ce := range perr.CompileErrors() {
			diag := Diagnostic{
				Code:         CodeTypeCheck,
				Stage:        StageTypeCheck,
				Message:      ce.Message,
				MacroContext: macro,
			}
			setPosition(&diag, ce.Position)
			if ce.Function != "" {
				frame := Frame{Function: ce.Function}
				if ce.FunctionPosition.IsValid() {
					frame.File = ce.FunctionPosition.Filename
					frame.Line = ce.FunctionPosition.Line
					frame.Column = ce.FunctionPosition.Column
				}
				diag.CallStack = []Frame{frame}
			}
			d.Add(diag)
		}
	case errors.As(err, &elist):
		for _, e := range elist {
			diag := Diagnostic{
				Code:         CodeSyntax,
				Stage:        StageParse,
				Message:      e.Msg,
				MacroContext: macro,
			}
			setPosition(&diag, e.Pos)
			d.Add(diag)
		}
	default:
		// An error that is neither a syntax error nor a type error is a structural problem of the
		// resulting program, such as a missing entry point.
		d.Add(Diagnostic{
			Code:         CodeIR,
			Stage:        StageIR,
			Message:      err.Error(),
			MacroContext: macro,
		})
	}
}

// setPosition sets the position fields of diag from pos.
func setPosition(diag *Diagnostic, pos token.Position) {
	diag.File = pos.Filename
	diag.Line = pos.Line
	diag.Column = pos.Column
}

// addBackendDiagnostics compiles p for each backend and adds a diagnostic for each failure.
func (d *Diagnostics) addBackendDiagnostics(p *shaderir.Program, backends []string) {
	for _, b := range backends {
		name := strings.ToLower(strings.TrimSpace(b))
		switch name {
		case "glsl", "glsl-es", "hlsl", "msl", "pssl":
		default:
			d.Add(Diagnostic{
				Code:    CodeUnknownBackend,
				Stage:   StageBackend,
				Backend: name,
				Message: fmt.Sprintf("unknown backend: %q", b),
			})
			continue
		}
		if log := compileOneBackend(p, name); log != "" {
			d.Add(Diagnostic{
				Code:       CodeBackendFailure,
				Stage:      StageBackend,
				Backend:    name,
				Message:    fmt.Sprintf("the %s backend failed to generate code", name),
				BackendLog: log,
			})
		}
	}
}

// compileOneBackend runs a single backend, recovering a panic into a raw log.
func compileOneBackend(p *shaderir.Program, backend string) (log string) {
	defer func() {
		if r := recover(); r != nil {
			log = fmt.Sprintf("panic: %v\n%s", r, debug.Stack())
		}
	}()
	switch backend {
	case "glsl":
		glsl.Compile(p, glsl.GLSLVersionDefault, graphics.ProjectionMatrixUniformVariableIndex)
	case "glsl-es":
		glsl.Compile(p, glsl.GLSLVersionES300, graphics.ProjectionMatrixUniformVariableIndex)
	case "hlsl":
		hlsl.Compile(p)
	case "msl":
		msl.Compile(p)
	case "pssl":
		pssl.Compile(p)
	}
	return ""
}
