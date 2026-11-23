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

// Command shaderprecompmanifest builds, verifies and diffs versioned shader precompilation manifests.
//
// A manifest is an auditable inventory of the shaders in a project: each entry records its Ebitengine
// source ID, a hash of the source, its macros and uniform layout, and every generated backend artifact
// with its own hash. The command is intended for CI, where its exit code gates a build:
//
//	shaderprecompmanifest generate -root . -manifest shadermanifest.json -source shaders/a.kage -source shaders/b.kage
//	shaderprecompmanifest validate -root . -manifest shadermanifest.json -require glsl,hlsl,msl
//	shaderprecompmanifest diff    -old old.json -new shadermanifest.json
//
// Exit codes:
//
//   - 0: success. For diff, the two manifests are identical; for generate/validate, no errors.
//   - 1: a manifest error (missing file, bad hash, unknown version, partial coverage, ...).
//   - 2: a usage error or an I/O failure unrelated to manifest content.
//   - 3: diff only: the manifests differ but both are well-formed.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/hajimehoshi/ebiten/v2/internal/shadermanifest"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

// Exit codes. They are part of the CI contract and must stay stable.
const (
	exitOK      = 0
	exitError   = 1
	exitUsage   = 2
	exitDiffers = 3
)

func run(args []string) int {
	if len(args) == 0 {
		usage()
		return exitUsage
	}
	switch args[0] {
	case "generate":
		return runGenerate(args[1:])
	case "validate":
		return runValidate(args[1:])
	case "diff":
		return runDiff(args[1:])
	case "-h", "--help", "help":
		usage()
		return exitUsage
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n\n", args[0])
		usage()
		return exitUsage
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `shaderprecompmanifest <subcommand> [flags]

Subcommands:
  generate   compile shader sources and write a manifest plus artifacts
  validate   verify a manifest against the files under the root
  diff       report the differences between two manifests

Run a subcommand with -h for its flags.`)
}

type stringList []string

func (l *stringList) String() string { return strings.Join(*l, ",") }
func (l *stringList) Set(v string) error {
	for _, part := range strings.Split(v, ",") {
		if part = strings.TrimSpace(part); part != "" {
			*l = append(*l, part)
		}
	}
	return nil
}

func runGenerate(args []string) int {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	root := fs.String("root", ".", "directory the source and artifact paths are relative to")
	manifestPath := fs.String("manifest", "shadermanifest.json", "path to write the manifest")
	var sources stringList
	fs.Var(&sources, "source", "shader source path relative to root, repeatable or comma-separated")
	artifactDir := fs.String("artifact-dir", shadermanifest.DefaultArtifactDir, "directory for generated artifacts, relative to root")
	var backends stringList
	fs.Var(&backends, "backend", "text backends to generate (default: all text backends)")
	gzip := fs.Bool("gzip", false, "gzip-compress generated artifacts")
	maxSize := fs.Int("max-source-size", shadermanifest.DefaultMaxSourceSize, "maximum shader source size in bytes")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	if len(sources) == 0 {
		fmt.Fprintln(os.Stderr, "generate requires at least one -source")
		return exitUsage
	}

	inputs := make([]shadermanifest.ShaderInput, 0, len(sources))
	for _, s := range sources {
		inputs = append(inputs, shadermanifest.ShaderInput{SourcePath: filepath.ToSlash(s)})
	}
	opts := &shadermanifest.GenerateOptions{
		ArtifactDir:   *artifactDir,
		Gzip:          *gzip,
		MaxSourceSize: *maxSize,
	}
	for _, b := range backends {
		opts.Backends = append(opts.Backends, shadermanifest.Backend(b))
	}

	m, errs := shadermanifest.Generate(*root, inputs, opts)
	printErrors(errs)
	if errs != nil {
		return exitError
	}
	if err := m.Save(*manifestPath); err != nil {
		fmt.Fprintf(os.Stderr, "writing manifest failed: %v\n", err)
		return exitUsage
	}
	fmt.Printf("wrote %s with %d shader entries\n", *manifestPath, len(m.Entries))
	return exitOK
}

func runValidate(args []string) int {
	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	root := fs.String("root", ".", "directory the source and artifact paths are relative to")
	manifestPath := fs.String("manifest", "shadermanifest.json", "path to the manifest to verify")
	var required stringList
	fs.Var(&required, "require", "backends every entry must cover, comma-separated")
	maxSize := fs.Int("max-source-size", shadermanifest.DefaultMaxSourceSize, "maximum shader source size in bytes")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}

	m, err := shadermanifest.Load(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "loading manifest failed: %v\n", err)
		return exitError
	}
	opts := &shadermanifest.ValidateOptions{MaxSourceSize: *maxSize}
	for _, b := range required {
		opts.RequireBackends = append(opts.RequireBackends, shadermanifest.Backend(b))
	}

	cov, errs := shadermanifest.Validate(*root, m, opts)
	printErrors(errs)
	printCoverage(cov)
	if errs != nil {
		return exitError
	}
	fmt.Println("manifest is valid")
	return exitOK
}

func runDiff(args []string) int {
	fs := flag.NewFlagSet("diff", flag.ContinueOnError)
	oldPath := fs.String("old", "", "path to the old manifest")
	newPath := fs.String("new", "", "path to the new manifest")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	if *oldPath == "" || *newPath == "" {
		fmt.Fprintln(os.Stderr, "diff requires both -old and -new")
		return exitUsage
	}

	oldM, err := shadermanifest.Load(*oldPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "loading old manifest failed: %v\n", err)
		return exitError
	}
	newM, err := shadermanifest.Load(*newPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "loading new manifest failed: %v\n", err)
		return exitError
	}

	report := shadermanifest.Diff(oldM, newM)
	fmt.Print(report.Text())
	if report.Empty() {
		return exitOK
	}
	return exitDiffers
}

func printErrors(errs shadermanifest.ErrorList) {
	for _, e := range errs {
		fmt.Fprintf(os.Stderr, "%s\n", e.Error())
	}
	if len(errs) > 0 {
		fmt.Fprintf(os.Stderr, "%d error(s)\n", len(errs))
	}
}

func printCoverage(cov *shadermanifest.Coverage) {
	if cov == nil {
		return
	}
	fmt.Printf("validated %d shader entries\n", cov.Entries)
	for _, b := range sortedBackends(cov.Covered) {
		fmt.Printf("  covered: %-16s %d/%d entries\n", b, cov.Covered[b], cov.Entries)
	}
	for _, b := range sortedBackends(cov.Missing) {
		fmt.Printf("  missing: %-16s %d entries incomplete\n", b, cov.Missing[b])
	}
}

func sortedBackends(m map[shadermanifest.Backend]int) []shadermanifest.Backend {
	out := make([]shadermanifest.Backend, 0, len(m))
	for b := range m {
		out = append(out, b)
	}
	// Simple insertion sort keeps output deterministic without importing sort just for this.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}
