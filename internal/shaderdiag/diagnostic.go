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

// Package shaderdiag provides structured diagnostics for Kage shader compilation.
//
// Where the regular compiler entry points report a single opaque error string, this package
// aggregates the errors of every compilation stage (parse, type check, IR, and the shader language
// backends) into a sorted, deduplicated list of [Diagnostic] values with stable error codes, source
// positions, call stacks, macro-expansion context, and the raw backend output. A [Diagnostics]
// value still implements the error interface and unwraps to the original error, so existing callers
// keep working.
package shaderdiag

import (
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

// Stage identifies the compilation stage that reported a diagnostic.
type Stage int

const (
	// StageParse is the syntactic analysis stage, including compiler directives.
	StageParse Stage = iota

	// StageTypeCheck is the semantic and type checking stage.
	StageTypeCheck

	// StageIR is the intermediate representation stage.
	StageIR

	// StageBackend is the backend code generation stage (GLSL, HLSL, MSL, PSSL).
	StageBackend
)

func (s Stage) String() string {
	switch s {
	case StageParse:
		return "parse"
	case StageTypeCheck:
		return "typecheck"
	case StageIR:
		return "ir"
	case StageBackend:
		return "backend"
	}
	return "unknown"
}

// MarshalJSON implements json.Marshaler.
func (s Stage) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// Stable error codes. These codes are part of the diagnostics contract and must not be reused or
// renumbered.
const (
	// CodeEmptySource indicates that the shader source is empty.
	CodeEmptySource = "KAGE0001"

	// CodeInvalidDirective indicates a malformed or unsupported compiler directive such as
	// //kage:unit.
	CodeInvalidDirective = "KAGE0002"

	// CodeSyntax indicates a syntax error reported by the parser.
	CodeSyntax = "KAGE1001"

	// CodeTypeCheck indicates a semantic or type error.
	CodeTypeCheck = "KAGE2001"

	// CodeIR indicates an invalid intermediate representation, such as a missing entry point.
	CodeIR = "KAGE3001"

	// CodeUnknownBackend indicates a backend name that is not supported.
	CodeUnknownBackend = "KAGE4001"

	// CodeBackendFailure indicates that a backend failed to generate code, for example by panicking.
	CodeBackendFailure = "KAGE4002"
)

// Limits on the collected diagnostics. Zero or negative values in [Options] select these defaults.
const (
	// DefaultMaxDiagnostics is the default maximum number of diagnostics kept per compilation.
	DefaultMaxDiagnostics = 64

	// DefaultMaxMessageBytes is the default maximum length of a diagnostic message in bytes.
	DefaultMaxMessageBytes = 4096

	// DefaultMaxBackendLogBytes is the default maximum length of a raw backend log in bytes.
	DefaultMaxBackendLogBytes = 8192
)

// truncationMarker is appended to messages and logs that exceed their length limit.
const truncationMarker = "... (truncated)"

// Frame is a single stack frame associated with a diagnostic.
type Frame struct {
	// Function is the name of the function.
	Function string `json:"function"`

	// File is the source file of the function declaration, if known.
	File string `json:"file,omitempty"`

	// Line is the 1-based line of the function declaration, if known.
	Line int `json:"line,omitempty"`

	// Column is the 1-based byte column of the function declaration, if known.
	Column int `json:"column,omitempty"`
}

// Diagnostic is a single structured compilation error.
type Diagnostic struct {
	// Code is the stable error code, e.g. "KAGE2001".
	Code string `json:"code"`

	// Stage is the compilation stage that reported the error.
	Stage Stage `json:"stage"`

	// Message is the human-readable error message.
	Message string `json:"message"`

	// File is the source file of the error, if known. A //line directive in the source sets this
	// to the file named by the directive.
	File string `json:"file,omitempty"`

	// Line is the 1-based line of the error, if known.
	Line int `json:"line,omitempty"`

	// Column is the 1-based byte column of the error, if known.
	Column int `json:"column,omitempty"`

	// CallStack is the chain of functions enclosing the error, innermost first.
	CallStack []Frame `json:"callStack,omitempty"`

	// MacroContext describes source transformations applied before compilation, such as the
	// //kage:unit directive and the texel-to-pixel conversion.
	MacroContext string `json:"macroContext,omitempty"`

	// Backend is the name of the backend that reported the error, if any.
	Backend string `json:"backend,omitempty"`

	// BackendLog is the raw output of the backend, if any.
	BackendLog string `json:"backendLog,omitempty"`
}

// Diagnostics is an aggregated list of compilation diagnostics.
//
// Diagnostics implements the error interface and unwraps to the original error returned by the
// compiler, so it can be used anywhere the previous error value was used.
type Diagnostics struct {
	diags     []Diagnostic
	cause     error
	truncated int

	redactor           func(string) string
	maxDiagnostics     int
	maxMessageBytes    int
	maxBackendLogBytes int
}

// New returns an empty Diagnostics that unwraps to cause. A nil cause is allowed.
func New(cause error, opts *Options) *Diagnostics {
	o := resolveOptions(opts)
	return &Diagnostics{
		cause:              cause,
		redactor:           o.Redactor,
		maxDiagnostics:     o.MaxDiagnostics,
		maxMessageBytes:    o.MaxMessageBytes,
		maxBackendLogBytes: o.MaxBackendLogBytes,
	}
}

// Add appends a diagnostic. The diagnostic's fields are redacted and truncated according to the
// options, and exact duplicates are dropped. Once the maximum number of diagnostics is reached,
// further diagnostics are dropped and counted by [Diagnostics.Truncated].
func (d *Diagnostics) Add(diag Diagnostic) {
	if d.redactor != nil {
		diag.File = d.redactor(diag.File)
		diag.Message = d.redactor(diag.Message)
		diag.MacroContext = d.redactor(diag.MacroContext)
		diag.BackendLog = d.redactor(diag.BackendLog)
		for i := range diag.CallStack {
			diag.CallStack[i].File = d.redactor(diag.CallStack[i].File)
		}
	}
	diag.Message = truncate(diag.Message, d.maxMessageBytes)
	diag.BackendLog = truncate(diag.BackendLog, d.maxBackendLogBytes)

	for _, e := range d.diags {
		if equalDiagnostic(e, diag) {
			return
		}
	}
	if len(d.diags) >= d.maxDiagnostics {
		d.truncated++
		return
	}
	d.diags = append(d.diags, diag)
}

// Len returns the number of collected diagnostics.
func (d *Diagnostics) Len() int {
	if d == nil {
		return 0
	}
	return len(d.diags)
}

// Truncated returns the number of diagnostics dropped because the maximum was reached.
func (d *Diagnostics) Truncated() int {
	if d == nil {
		return 0
	}
	return d.truncated
}

// All returns the collected diagnostics, sorted stably by source position. The result is a deep
// copy; modifying it does not affect the Diagnostics.
func (d *Diagnostics) All() []Diagnostic {
	if d == nil {
		return nil
	}
	d.sort()
	ds := make([]Diagnostic, len(d.diags))
	for i, diag := range d.diags {
		ds[i] = diag
		ds[i].CallStack = slices.Clone(diag.CallStack)
	}
	return ds
}

// Error implements the error interface. The format matches the one of the underlying compiler
// errors: one "file:line:column: message" entry per line.
func (d *Diagnostics) Error() string {
	if d == nil {
		return ""
	}
	d.sort()
	msgs := make([]string, 0, len(d.diags))
	for _, e := range d.diags {
		if e.File != "" || e.Line != 0 {
			msgs = append(msgs, fmt.Sprintf("%s:%d:%d: %s", e.File, e.Line, e.Column, e.Message))
		} else {
			msgs = append(msgs, e.Message)
		}
	}
	if d.truncated > 0 {
		msgs = append(msgs, fmt.Sprintf("... and %d more error(s)", d.truncated))
	}
	return strings.Join(msgs, "\n")
}

// Unwrap returns the original error returned by the compiler, if any.
func (d *Diagnostics) Unwrap() error {
	if d == nil {
		return nil
	}
	return d.cause
}

// MarshalJSON implements json.Marshaler.
func (d *Diagnostics) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Errors    []Diagnostic `json:"errors"`
		Truncated int          `json:"truncated"`
	}{
		Errors:    d.All(),
		Truncated: d.truncated,
	})
}

// sort stably sorts the diagnostics by source position, then by stage, code, and message.
func (d *Diagnostics) sort() {
	slices.SortStableFunc(d.diags, func(a, b Diagnostic) int {
		if c := strings.Compare(a.File, b.File); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Line, b.Line); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Column, b.Column); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Stage, b.Stage); c != 0 {
			return c
		}
		if c := strings.Compare(a.Code, b.Code); c != 0 {
			return c
		}
		return strings.Compare(a.Message, b.Message)
	})
}

// equalDiagnostic reports whether a and b are identical.
func equalDiagnostic(a, b Diagnostic) bool {
	return a.Code == b.Code &&
		a.Stage == b.Stage &&
		a.Message == b.Message &&
		a.File == b.File &&
		a.Line == b.Line &&
		a.Column == b.Column &&
		a.MacroContext == b.MacroContext &&
		a.Backend == b.Backend &&
		a.BackendLog == b.BackendLog &&
		slices.Equal(a.CallStack, b.CallStack)
}

// truncate shortens s to at most max bytes without splitting a multi-byte rune, appending a
// truncation marker when s was shortened. A non-positive max returns s unchanged.
func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	s = s[:max]
	// Back off at most utf8.UTFMax-1 bytes to avoid splitting a rune at the cut point.
	for range utf8.UTFMax - 1 {
		if utf8.ValidString(s) {
			break
		}
		s = s[:len(s)-1]
	}
	return s + truncationMarker
}
