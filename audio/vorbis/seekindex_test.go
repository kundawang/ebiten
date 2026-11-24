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

package vorbis_test

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jfreymuth/oggvorbis"

	"github.com/hajimehoshi/ebiten/v2/audio/vorbis"
)

//go:embed testdata/long_stereo.ogg
var longStereoOGG []byte

//go:embed testdata/long_mono.ogg
var longMonoOGG []byte

//go:embed testdata/sparse.ogg
var sparseOGG []byte

//go:embed testdata/chained.ogg
var chainedOGG []byte

// decodeKinds enumerates the stream constructors exercised by the seek-index
// tests. Every kind is checked against the others to guarantee that channels,
// sample rate, length and decoded samples stay consistent.
type decodeKind struct {
	name           string
	decode         func(io.Reader) (*vorbis.Stream, error)
	bytesPerSample int64
}

func seekDecodeKinds() []decodeKind {
	return []decodeKind{
		{
			name:           "I16",
			decode:         vorbis.DecodeWithoutResampling,
			bytesPerSample: 4,
		},
		{
			name:           "F32",
			decode:         vorbis.DecodeF32,
			bytesPerSample: 8,
		},
		{
			name: "Resampling",
			decode: func(r io.Reader) (*vorbis.Stream, error) {
				// All fixtures are 44100Hz or 22050Hz, so this always resamples.
				return vorbis.DecodeWithSampleRate(48000, r)
			},
			bytesPerSample: 4,
		},
	}
}

func buildIndex(t *testing.T, data []byte, opts ...vorbis.VorbisSeekIndexOption) *vorbis.VorbisSeekIndex {
	t.Helper()
	idx := vorbis.NewVorbisSeekIndex(opts...)
	if err := idx.Build(bytes.NewReader(data)); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if idx.Len() == 0 {
		t.Fatal("Build produced an empty index")
	}
	return idx
}

// referenceSamples decodes the whole stream with the upstream oggvorbis reader
// and returns its interleaved samples as little-endian float32 bytes.
func referenceSamples(t *testing.T, data []byte) []byte {
	t.Helper()
	r, err := oggvorbis.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("oggvorbis.NewReader: %v", err)
	}
	all, err := io.ReadAll(newFloat32Collector(r))
	if err != nil {
		t.Fatalf("reference decode: %v", err)
	}
	return all
}

type float32Collector struct {
	r        *oggvorbis.Reader
	leftover []byte
}

func newFloat32Collector(r *oggvorbis.Reader) *float32Collector {
	return &float32Collector{r: r}
}

func (c *float32Collector) Read(p []byte) (int, error) {
	if len(c.leftover) > 0 {
		n := copy(p, c.leftover)
		c.leftover = c.leftover[n:]
		return n, nil
	}
	frames := make([]float32, 4096*c.r.Channels())
	n, err := c.r.Read(frames)
	c.leftover = c.leftover[:0]
	for _, v := range frames[:n] {
		u := math.Float32bits(v)
		c.leftover = append(c.leftover, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	nn := copy(p, c.leftover)
	c.leftover = c.leftover[nn:]
	if nn > 0 {
		return nn, nil
	}
	return 0, err
}

func TestVorbisSeekIndexBuild(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"Stereo", longStereoOGG},
		{"Mono", longMonoOGG},
		{"Sparse", sparseOGG},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dense := buildIndex(t, tc.data, vorbis.WithVorbisSeekIndexInterval(1))
			sparse := buildIndex(t, tc.data, vorbis.WithVorbisSeekIndexInterval(16))

			if sparse.Len() >= dense.Len() {
				t.Errorf("an index with a larger interval must be sparser: dense=%d sparse=%d", dense.Len(), sparse.Len())
			}
			if sparse.Len() == 0 {
				t.Errorf("the sparse index unexpectedly has no entries")
			}

			// Building again replaces entries instead of appending to them.
			before := dense.Len()
			if err := dense.Build(bytes.NewReader(tc.data)); err != nil {
				t.Fatalf("rebuild: %v", err)
			}
			if dense.Len() != before {
				t.Errorf("rebuild changed the entry count: got %d, want %d", dense.Len(), before)
			}

			dense.Invalidate()
			if dense.Len() != 0 {
				t.Errorf("Invalidate: got %d entries, want 0", dense.Len())
			}
		})
	}
}

func TestVorbisSeekIndexBuildRestoresPosition(t *testing.T) {
	r := bytes.NewReader(longStereoOGG)
	const mark = 1234
	if _, err := r.Seek(mark, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.Build(r); err != nil {
		t.Fatalf("Build: %v", err)
	}
	got, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	if got != mark {
		t.Errorf("source position after Build: got %d, want %d", got, mark)
	}
}

func TestVorbisSeekIndexRandomTargets(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"Stereo", longStereoOGG},
		{"Mono", longMonoOGG},
		{"Sparse", sparseOGG},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The full, continuous decode fixes channels, sample rate and length.
			continuous, err := vorbis.DecodeWithoutResampling(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal(err)
			}
			fullBytes, err := io.ReadAll(continuous)
			if err != nil {
				t.Fatalf("continuous decode: %v", err)
			}

			for _, interval := range []int{1, 3, 8, 32} {
				t.Run(fmt.Sprintf("interval=%d", interval), func(t *testing.T) {
					for _, kind := range seekDecodeKinds() {
						t.Run(kind.name, func(t *testing.T) {
							idx := buildIndex(t, tc.data, vorbis.WithVorbisSeekIndexInterval(interval))

							// A plain, unindexed stream of the same kind establishes the
							// linear-seek behavior every indexed seek must reproduce.
							plain, err := kind.decode(bytes.NewReader(tc.data))
							if err != nil {
								t.Fatal(err)
							}
							indexed, err := kind.decode(bytes.NewReader(tc.data))
							if err != nil {
								t.Fatal(err)
							}
							if err := indexed.SetVorbisSeekIndex(idx); err != nil {
								t.Fatalf("SetVorbisSeekIndex: %v", err)
							}
							if got, want := indexed.Length(), plain.Length(); got != want {
								t.Errorf("Length: got %d, want %d", got, want)
							}
							if got, want := indexed.SampleRate(), plain.SampleRate(); got != want {
								t.Errorf("SampleRate: got %d, want %d", got, want)
							}

							rng := rand.New(rand.NewPCG(1, 2))
							sample := kind.bytesPerSample
							totalFrames := int64(len(fullBytes)) / sample
							nTargets := min(int(totalFrames), 300)
							for range nTargets {
								targetFrame := rng.Int64N(totalFrames + 1)
								target := targetFrame * sample

								// Independent linear seek on a fresh plain stream.
								refStream, err := kind.decode(bytes.NewReader(tc.data))
								if err != nil {
									t.Fatal(err)
								}
								refPos, err := refStream.Seek(target, io.SeekStart)
								if err != nil {
									t.Fatalf("plain Seek(%d): %v", target, err)
								}

								pos, err := indexed.Seek(target, io.SeekStart)
								if err != nil {
									t.Fatalf("indexed Seek(%d): %v", target, err)
								}
								if pos != refPos {
									t.Fatalf("Seek(%d) returned %d, plain returned %d", target, pos, refPos)
								}

								n := int64(32) * sample
								if target+n > int64(len(fullBytes)) {
									n = int64(len(fullBytes)) - target
								}
								if n == 0 {
									continue
								}
								want := make([]byte, n)
								if _, err := io.ReadFull(refStream, want); err != nil {
									t.Fatalf("plain ReadFull after Seek(%d): %v", target, err)
								}
								got := make([]byte, n)
								if _, err := io.ReadFull(indexed, got); err != nil {
									t.Fatalf("indexed ReadFull after Seek(%d): %v", target, err)
								}
								if !bytes.Equal(got, want) {
									t.Fatalf("decoded bytes after Seek(%d) differ from an independent linear seek", target)
								}
							}

							stats := idx.Stats()
							// Fallbacks must never happen on an intact source that the
							// index was built from, and the random targets include seeks
							// well past the first page.
							if stats.Fallbacks != 0 {
								t.Errorf("unexpected fallbacks: %+v", stats)
							}
							if stats.Hits == 0 {
								t.Errorf("no indexed seek was recorded: %+v", stats)
							}

							// A fresh indexed stream must still reproduce the continuous
							// decode from the start, proving the index never alters
							// linearity.
							head := make([]byte, min(len(fullBytes), 8*1024))
							start, err := kind.decode(bytes.NewReader(tc.data))
							if err != nil {
								t.Fatal(err)
							}
							if err := start.SetVorbisSeekIndex(idx); err != nil {
								t.Fatal(err)
							}
							if _, err := io.ReadFull(start, head); err != nil {
								t.Fatal(err)
							}
							plainHead := make([]byte, len(head))
							plainStart, err := kind.decode(bytes.NewReader(tc.data))
							if err != nil {
								t.Fatal(err)
							}
							if _, err := io.ReadFull(plainStart, plainHead); err != nil {
								t.Fatal(err)
							}
							if !bytes.Equal(head, plainHead) {
								t.Error("the indexed stream's head differs from the unindexed stream")
							}
						})
					}
				})
			}
		})
	}
}

// TestVorbisSeekIndexReadToEOF confirms that after an indexed seek the rest of
// the stream, read all the way to EOF across many Ogg pages, matches an
// independent linear seek. This guards the hand-off where the seek window
// closes and ordinary decoding resumes.
func TestVorbisSeekIndexReadToEOF(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"Stereo", longStereoOGG},
		{"Mono", longMonoOGG},
		{"Sparse", sparseOGG},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, kind := range seekDecodeKinds() {
				t.Run(kind.name, func(t *testing.T) {
					idx := buildIndex(t, tc.data, vorbis.WithVorbisSeekIndexInterval(5))
					// Seek roughly one third in, then read every remaining byte.
					plain, err := kind.decode(bytes.NewReader(tc.data))
					if err != nil {
						t.Fatal(err)
					}
					target := plain.Length() / 3
					target -= target % kind.bytesPerSample
					if _, err := plain.Seek(target, io.SeekStart); err != nil {
						t.Fatal(err)
					}
					want, err := io.ReadAll(plain)
					if err != nil {
						t.Fatalf("plain ReadAll: %v", err)
					}

					indexed, err := kind.decode(bytes.NewReader(tc.data))
					if err != nil {
						t.Fatal(err)
					}
					if err := indexed.SetVorbisSeekIndex(idx); err != nil {
						t.Fatal(err)
					}
					if _, err := indexed.Seek(target, io.SeekStart); err != nil {
						t.Fatal(err)
					}
					got, err := io.ReadAll(indexed)
					if err != nil {
						t.Fatalf("indexed ReadAll: %v", err)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("the %d bytes read after an indexed seek to %d differ from the linear seek", len(got), target)
					}
					if stats := idx.Stats(); stats.Hits == 0 {
						t.Errorf("the seek was not served by the index: %+v", stats)
					}
				})
			}
		})
	}
}

// TestVorbisSeekIndexFrameAccuracy verifies that an indexed seek reaches exactly
// the same samples an independent, unindexed linear seek reaches. That is the
// frame-error guarantee the index must provide: it changes how the starting
// page is found, never the decoded samples.
//
// For stereo sources the linear seek itself matches a continuous decode, which
// is checked as an additional, stronger invariant.
func TestVorbisSeekIndexFrameAccuracy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		data   []byte
		stereo bool
	}{
		{"Stereo", longStereoOGG, true},
		{"Mono", longMonoOGG, false},
		{"Sparse", sparseOGG, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The F32 stream is always stereo float32 LE: 8 bytes per frame.
			const frameSize = 8

			continuous, err := vorbis.DecodeF32(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal(err)
			}
			ref, err := io.ReadAll(continuous)
			if err != nil {
				t.Fatal(err)
			}
			// The upstream oggvorbis reference must agree on the decoded length.
			upstream := referenceSamples(t, tc.data)
			oggReader, err := oggvorbis.NewReader(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal(err)
			}
			if oggReader.Channels() == 1 {
				if len(ref) != len(upstream)*2 {
					t.Errorf("mono upmix length: got %d, want %d", len(ref), len(upstream)*2)
				}
			} else if len(ref) != len(upstream) {
				t.Errorf("stereo length: got %d, want %d", len(ref), len(upstream))
			}

			idx := buildIndex(t, tc.data, vorbis.WithVorbisSeekIndexInterval(6))
			totalFrames := int64(len(ref)) / frameSize
			rng := rand.New(rand.NewPCG(11, 23))
			for range 200 {
				frame := rng.Int64N(totalFrames)
				target := frame * frameSize

				// The independent linear seek is the reference behavior.
				plain, err := vorbis.DecodeF32(bytes.NewReader(tc.data))
				if err != nil {
					t.Fatal(err)
				}
				if _, err := plain.Seek(target, io.SeekStart); err != nil {
					t.Fatalf("plain Seek(%d): %v", target, err)
				}

				indexed, err := vorbis.DecodeF32(bytes.NewReader(tc.data))
				if err != nil {
					t.Fatal(err)
				}
				if err := indexed.SetVorbisSeekIndex(idx); err != nil {
					t.Fatal(err)
				}
				if _, err := indexed.Seek(target, io.SeekStart); err != nil {
					t.Fatalf("indexed Seek(%d): %v", target, err)
				}

				// Compare a window spanning several decoder blocks.
				nFrames := int64(128)
				if frame+nFrames > totalFrames {
					nFrames = totalFrames - frame
				}
				n := nFrames * frameSize
				want := make([]byte, n)
				got := make([]byte, n)
				if _, err := io.ReadFull(plain, want); err != nil {
					t.Fatalf("plain ReadFull: %v", err)
				}
				if _, err := io.ReadFull(indexed, got); err != nil {
					t.Fatalf("indexed ReadFull: %v", err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("frame error after Seek to frame %d: indexed samples differ from an independent linear seek", frame)
				}
				// On stereo sources the linear seek itself equals the continuous
				// decode, so assert zero frame error end to end.
				if tc.stereo && !bytes.Equal(got, ref[target:target+n]) {
					t.Fatalf("stereo seek to frame %d differs from a continuous decode", frame)
				}
			}
			if stats := idx.Stats(); stats.Hits == 0 {
				t.Errorf("no indexed seeks were used: %+v", stats)
			}
		})
	}
}

func TestVorbisSeekIndexHitAndMissStats(t *testing.T) {
	idx := buildIndex(t, longStereoOGG, vorbis.WithVorbisSeekIndexInterval(16))

	s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}

	// A seek near the end has an indexed page before it and must be a hit.
	end := s.Length() - s.Length()%4
	before := idx.Stats()
	if _, err := s.Seek(end-4, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	after := idx.Stats()
	if after.Hits != before.Hits+1 {
		t.Errorf("the late seek was not counted as a hit: before=%+v after=%+v", before, after)
	}
	if after.Fallbacks != 0 {
		t.Errorf("an intact source must not trigger fallbacks: %+v", after)
	}

	// A target before the first indexed entry is served by the linear path.
	// Seek to zero repeatedly until a miss is observed; a granule-zero first
	// entry makes Seek(0) a legitimate hit, so target the earliest frame that is
	// still before every entry by forcing a fresh, empty index comparison.
	empty := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(16))
	emptyStream, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := emptyStream.SetVorbisSeekIndex(empty); err != nil {
		t.Fatal(err)
	}
	if _, err := emptyStream.Seek(end/2, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if st := empty.Stats(); st.Misses != 1 || st.Hits != 0 {
		t.Errorf("an empty index must produce one miss: got %+v", st)
	}

	// Repeated seeks keep counting and never deadlock.
	hitsBefore := after.Hits
	for range 50 {
		if _, err := s.Seek(end/2, io.SeekStart); err != nil {
			t.Fatal(err)
		}
	}
	stats := idx.Stats()
	if stats.Hits != hitsBefore+50 {
		t.Errorf("after repeated seeks: got %d hits, want %d", stats.Hits, hitsBefore+50)
	}
	if stats.Fallbacks != 0 {
		t.Errorf("repeated seeks triggered fallbacks: %+v", stats)
	}
}

func TestVorbisSeekIndexSharedAcrossStreams(t *testing.T) {
	idx := buildIndex(t, longStereoOGG)

	var streams []*vorbis.Stream
	for range 3 {
		s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SetVorbisSeekIndex(idx); err != nil {
			t.Fatal(err)
		}
		streams = append(streams, s)
	}

	plain, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	want, err := io.ReadAll(plain)
	if err != nil {
		t.Fatal(err)
	}

	for i, s := range streams {
		target := int64((i + 1) * len(want) / 4)
		target -= target % 4
		if _, err := s.Seek(target, io.SeekStart); err != nil {
			t.Fatalf("stream %d Seek: %v", i, err)
		}
		got := make([]byte, 4096)
		if _, err := io.ReadFull(s, got); err != nil {
			t.Fatalf("stream %d ReadFull: %v", i, err)
		}
		if !bytes.Equal(got, want[target:target+4096]) {
			t.Errorf("stream %d decoded different bytes at %d", i, target)
		}
	}
	if stats := idx.Stats(); stats.Hits < 3 {
		t.Errorf("shared index recorded %+v, want at least 3 hits", stats)
	}
}

func TestVorbisSeekIndexMarshalRoundTrip(t *testing.T) {
	idx := buildIndex(t, longStereoOGG, vorbis.WithVorbisSeekIndexInterval(5))
	data, err := idx.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	loaded := vorbis.NewVorbisSeekIndex()
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}
	if err := loaded.Load(bytes.NewReader(data)); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != idx.Len() {
		t.Errorf("loaded index has %d entries, want %d", loaded.Len(), idx.Len())
	}

	// The loaded index must produce byte-identical seeks to the built one.
	for _, kind := range seekDecodeKinds()[:2] {
		t.Run(kind.name, func(t *testing.T) {
			plain, err := kind.decode(bytes.NewReader(longStereoOGG))
			if err != nil {
				t.Fatal(err)
			}
			want, err := io.ReadAll(plain)
			if err != nil {
				t.Fatal(err)
			}
			s, err := kind.decode(bytes.NewReader(longStereoOGG))
			if err != nil {
				t.Fatal(err)
			}
			if err := s.SetVorbisSeekIndex(loaded); err != nil {
				t.Fatal(err)
			}
			sample := kind.bytesPerSample
			for _, target := range []int64{sample * 100, sample * 5000, sample * 12345} {
				if target >= int64(len(want)) {
					continue
				}
				if _, err := s.Seek(target, io.SeekStart); err != nil {
					t.Fatalf("Seek(%d): %v", target, err)
				}
				got := make([]byte, min(2048, int64(len(want))-target))
				if _, err := io.ReadFull(s, got); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want[target:target+int64(len(got))]) {
					t.Errorf("loaded-index seek to %d produced different bytes", target)
				}
			}
			if stats := loaded.Stats(); stats.Fallbacks != 0 {
				t.Errorf("a valid loaded index must not fall back: %+v", stats)
			}
		})
	}
}

func TestVorbisSeekIndexUnmarshalRejectsBadData(t *testing.T) {
	good, err := buildIndex(t, longStereoOGG).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]byte{
		"nil":          nil,
		"short":        good[:10],
		"truncated":    good[:len(good)-3],
		"garbage":      bytes.Repeat([]byte{0xAB}, 64),
		"bad interval": buildModifiedIndex(good, 8, []byte{0, 0, 0, 0}),
	}
	// Bad magic variant.
	badMagic := append([]byte{}, good...)
	copy(badMagic[:7], []byte("NOTVBSE"))
	cases["bad magic"] = badMagic
	// Bad version variant.
	badVersion := append([]byte{}, good...)
	badVersion[7] = 99
	cases["bad version"] = badVersion
	// Negative granule variant.
	if len(good) >= 16+8 {
		badGranule := append([]byte{}, good...)
		copy(badGranule[16:24], []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF})
		cases["negative granule"] = badGranule
	}

	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			idx := vorbis.NewVorbisSeekIndex()
			if err := idx.UnmarshalBinary(data); err == nil {
				t.Errorf("UnmarshalBinary(%s): got no error, want one", name)
			}
		})
	}
}

// buildModifiedIndex returns a copy of data with the interval field replaced.
func buildModifiedIndex(data []byte, at int, replacement []byte) []byte {
	out := append([]byte{}, data...)
	copy(out[at:at+len(replacement)], replacement)
	return out
}

func TestVorbisSeekIndexNonSeekableSource(t *testing.T) {
	idx := buildIndex(t, longStereoOGG)
	s, err := vorbis.DecodeWithoutResampling(&reader{r: bytes.NewReader(longStereoOGG)})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(idx); !errors.Is(err, errors.ErrUnsupported) {
		t.Errorf("SetVorbisSeekIndex on a non-seekable stream: got %v, want errors.ErrUnsupported", err)
	}
	if _, err := s.Seek(0, io.SeekStart); !errors.Is(err, errors.ErrUnsupported) {
		t.Errorf("Seek on a non-seekable stream: got %v, want errors.ErrUnsupported", err)
	}
}
