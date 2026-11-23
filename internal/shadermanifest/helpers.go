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
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"path"
	"strings"
)

// hashBytes returns the lowercase hex SHA-256 of data.
func hashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// calcSourceID reproduces Ebitengine's shader source ID for the given Kage source. It is the same
// computation shadercollector performs, so a manifest ID matches the ID Ebitengine computes at runtime.
func calcSourceID(source []byte) (string, error) {
	id, err := legacyshaderSourceID(source)
	if err != nil {
		return "", err
	}
	return id.String(), nil
}

// macroKey renders a macro map as a stable, comparable string: keys are sorted and joined as key=value
// pairs. The default variant always has the empty key.
func macroKey(macros map[string]string) string {
	if len(macros) == 0 {
		return ""
	}
	keys := make([]string, 0, len(macros))
	for k := range macros {
		keys = append(keys, k)
	}
	// Insertion order-independent: sort the keys.
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j-1] > keys[j]; j-- {
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(';')
		}
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(macros[k])
	}
	return b.String()
}

// cleanRelPath normalizes a slash-separated relative path and reports whether it is safe: non-empty,
// not absolute, lexical-clean, and never escaping the root (no leading ".."). All paths stored in a
// manifest pass this check so the manifest can never point outside its root directory.
func cleanRelPath(p string) (string, bool) {
	if p == "" {
		return "", false
	}
	// Normalize OS separators to forward slashes.
	p = strings.ReplaceAll(p, "\\", "/")
	if path.IsAbs(p) {
		return "", false
	}
	cleaned := path.Clean(p)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return "", false
	}
	// path.Clean keeps a leading "./" only as nothing (it strips it); reject any residual traversal token.
	for _, part := range strings.Split(cleaned, "/") {
		if part == ".." {
			return "", false
		}
	}
	return cleaned, true
}

// gzipBytes compresses data deterministically. The gzip header is zero-valued (no name, no modification
// time, fixed OS byte), so identical input always yields identical output and compressed artifacts stay
// reproducible.
func gzipBytes(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	// Leave the gzip Header at its zero value: ModTime is the zero time and Name/Comment/Extra are empty,
	// so identical input always compresses to identical output.
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(data); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// gunzipBytes decompresses a gzip stream and returns the original bytes.
func gunzipBytes(data []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	out, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return out, nil
}
