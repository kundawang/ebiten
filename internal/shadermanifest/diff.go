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

import (
	"fmt"
	"sort"
	"strings"
)

// ChangeKind classifies a single difference between two manifests.
type ChangeKind string

const (
	// ChangeEntryAdded is an entry present in the new manifest but not in the old one.
	ChangeEntryAdded ChangeKind = "entry-added"

	// ChangeEntryRemoved is an entry present in the old manifest but not in the new one.
	ChangeEntryRemoved ChangeKind = "entry-removed"

	// ChangeSourceChanged is an entry whose source hash, source ID or source path changed.
	ChangeSourceChanged ChangeKind = "source-changed"

	// ChangeUniformsChanged is an entry whose uniform layout or texture count changed.
	ChangeUniformsChanged ChangeKind = "uniforms-changed"

	// ChangeArtifactAdded is an artifact present in the new entry but not in the old one.
	ChangeArtifactAdded ChangeKind = "artifact-added"

	// ChangeArtifactRemoved is an artifact present in the old entry but not in the new one.
	ChangeArtifactRemoved ChangeKind = "artifact-removed"

	// ChangeArtifactChanged is an artifact whose path, hash, size or compression changed.
	ChangeArtifactChanged ChangeKind = "artifact-changed"
)

// Change is one difference, located at a shader entry and optionally a backend/role.
type Change struct {
	Kind     ChangeKind
	SourceID string
	Backend  Backend
	Role     string
	Detail   string
}

// DiffReport is the result of comparing two canonical manifests.
type DiffReport struct {
	Changes []Change
}

// Empty reports whether the two manifests were identical.
func (r *DiffReport) Empty() bool {
	return len(r.Changes) == 0
}

// Diff compares an old manifest against a new one and returns every difference in a stable order.
// Both inputs are canonicalized first, so manifest formatting or array ordering never produces a
// spurious change.
func Diff(old, new *Manifest) *DiffReport {
	old.canonicalize()
	new.canonicalize()

	report := &DiffReport{}
	oldByKey := map[string]*Spec{}
	newByKey := map[string]*Spec{}
	for _, s := range old.Entries {
		oldByKey[specKey(s)] = s
	}
	for _, s := range new.Entries {
		newByKey[specKey(s)] = s
	}

	allKeys := make(map[string]struct{}, len(oldByKey)+len(newByKey))
	for k := range oldByKey {
		allKeys[k] = struct{}{}
	}
	for k := range newByKey {
		allKeys[k] = struct{}{}
	}
	keys := make([]string, 0, len(allKeys))
	for k := range allKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		o := oldByKey[k]
		n := newByKey[k]
		switch {
		case o == nil:
			report.Changes = append(report.Changes, Change{
				Kind:     ChangeEntryAdded,
				SourceID: n.SourceID,
				Detail:   fmt.Sprintf("entry for macros {%s} added", macroKey(n.Macros)),
			})
			continue
		case n == nil:
			report.Changes = append(report.Changes, Change{
				Kind:     ChangeEntryRemoved,
				SourceID: o.SourceID,
				Detail:   fmt.Sprintf("entry for macros {%s} removed", macroKey(o.Macros)),
			})
			continue
		}

		if o.SourceHash != n.SourceHash || o.SourceID != n.SourceID || o.SourcePath != n.SourcePath || o.SourceSize != n.SourceSize {
			report.Changes = append(report.Changes, Change{
				Kind:     ChangeSourceChanged,
				SourceID: n.SourceID,
				Detail:   fmt.Sprintf("source changed (path %q -> %q, hash %s -> %s)", o.SourcePath, n.SourcePath, shortHash(o.SourceHash), shortHash(n.SourceHash)),
			})
		}
		if !uniformsEqual(o.Uniforms, n.Uniforms) || o.TextureCount != n.TextureCount {
			report.Changes = append(report.Changes, Change{
				Kind:     ChangeUniformsChanged,
				SourceID: n.SourceID,
				Detail:   "uniform layout or texture count changed",
			})
		}

		diffArtifacts(report, o, n)
	}

	return report
}

func diffArtifacts(report *DiffReport, o, n *Spec) {
	oldArt := map[string]*Artifact{}
	newArt := map[string]*Artifact{}
	for _, a := range o.Artifacts {
		oldArt[artifactKey(a)] = a
	}
	for _, a := range n.Artifacts {
		newArt[artifactKey(a)] = a
	}

	keys := map[string]struct{}{}
	for k := range oldArt {
		keys[k] = struct{}{}
	}
	for k := range newArt {
		keys[k] = struct{}{}
	}
	ordered := make([]string, 0, len(keys))
	for k := range keys {
		ordered = append(ordered, k)
	}
	sort.Strings(ordered)

	for _, k := range ordered {
		oa := oldArt[k]
		na := newArt[k]
		switch {
		case oa == nil:
			report.Changes = append(report.Changes, Change{
				Kind: ChangeArtifactAdded, SourceID: n.SourceID,
				Backend: na.Backend, Role: na.Role, Detail: "artifact added at " + na.Path,
			})
		case na == nil:
			report.Changes = append(report.Changes, Change{
				Kind: ChangeArtifactRemoved, SourceID: n.SourceID,
				Backend: oa.Backend, Role: oa.Role, Detail: "artifact removed from " + oa.Path,
			})
		default:
			var details []string
			if oa.Path != na.Path {
				details = append(details, fmt.Sprintf("path %q -> %q", oa.Path, na.Path))
			}
			if oa.Hash != na.Hash {
				details = append(details, fmt.Sprintf("hash %s -> %s", shortHash(oa.Hash), shortHash(na.Hash)))
			}
			if oa.Size != na.Size {
				details = append(details, fmt.Sprintf("size %d -> %d", oa.Size, na.Size))
			}
			if oa.Gzip != na.Gzip {
				details = append(details, fmt.Sprintf("gzip %t -> %t", oa.Gzip, na.Gzip))
			}
			if len(details) > 0 {
				report.Changes = append(report.Changes, Change{
					Kind: ChangeArtifactChanged, SourceID: n.SourceID,
					Backend: na.Backend, Role: na.Role, Detail: strings.Join(details, "; "),
				})
			}
		}
	}
}

// Text renders the report as indented, human-readable lines for CI logs.
func (r *DiffReport) Text() string {
	if r.Empty() {
		return "manifests are identical\n"
	}
	var b strings.Builder
	for _, c := range r.Changes {
		b.WriteString(string(c.Kind))
		b.WriteByte(' ')
		b.WriteString(c.SourceID)
		if c.Backend != "" {
			b.WriteByte(' ')
			b.WriteString(string(c.Backend))
		}
		if c.Role != "" {
			b.WriteByte('/')
			b.WriteString(c.Role)
		}
		if c.Detail != "" {
			b.WriteString(": ")
			b.WriteString(c.Detail)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func specKey(s *Spec) string {
	return s.SourceID + "\x00" + macroKey(s.Macros)
}

func artifactKey(a *Artifact) string {
	return string(a.Backend) + "\x00" + a.Role
}

func shortHash(h string) string {
	if len(h) <= 12 {
		return h
	}
	return h[:12]
}
