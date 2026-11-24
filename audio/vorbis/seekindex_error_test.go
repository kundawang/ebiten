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
	"encoding/binary"
	"errors"
	"io"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio/vorbis"
)

// --- synthetic Ogg framing helpers ---------------------------------------

var testCRCTable [256]uint32

func init() {
	for i := range testCRCTable {
		r := uint32(i) << 24
		for j := 0; j < 8; j++ {
			if r&0x80000000 != 0 {
				r = (r << 1) ^ 0x04c11db7
			} else {
				r <<= 1
			}
		}
		testCRCTable[i] = r
	}
}

func testCRC(crc uint32, data []byte) uint32 {
	for _, b := range data {
		crc = (crc << 8) ^ testCRCTable[byte(crc>>24)^b]
	}
	return crc
}

// buildOggPage assembles a single Ogg page with a valid checksum.
// When continued is true, the final packet is treated as spilling onto the
// next page: its content is emitted with only 0xFF segment entries and no
// terminating entry, matching a real lacing of a packet larger than the page.
func buildOggPage(flag byte, granule int64, serial, sequence uint32, packets [][]byte, continued bool) []byte {
	var segTable []byte
	var content []byte
	for pi, pkt := range packets {
		content = append(content, pkt...)
		last := pi == len(packets)-1
		if last && continued {
			n := len(pkt) / 255
			for range n {
				segTable = append(segTable, 255)
			}
			// A remainder carried by the continuation page is left without a
			// terminating segment here.
			continue
		}
		remaining := len(pkt)
		for remaining >= 255 {
			segTable = append(segTable, 255)
			remaining -= 255
		}
		segTable = append(segTable, byte(remaining))
	}
	header := make([]byte, 27)
	copy(header[0:4], []byte("OggS"))
	header[4] = 0
	header[5] = flag
	binary.LittleEndian.PutUint64(header[6:14], uint64(granule))
	binary.LittleEndian.PutUint32(header[14:18], serial)
	binary.LittleEndian.PutUint32(header[18:22], sequence)
	// checksum zero while computing
	header[26] = byte(len(segTable))
	page := append(append(header, segTable...), content...)
	crc := testCRC(0, page)
	binary.LittleEndian.PutUint32(page[22:26], crc)
	return page
}

// findPageOffsets returns the byte offsets of every OggS capture pattern.
func findPageOffsets(data []byte) []int64 {
	var offs []int64
	for i := 0; i+4 <= len(data); i++ {
		if data[i] == 'O' && data[i+1] == 'g' && data[i+2] == 'g' && data[i+3] == 'S' {
			offs = append(offs, int64(i))
		}
	}
	return offs
}

// --- index build: corruption and truncation ------------------------------

func TestVorbisSeekIndexBuildTruncatedPage(t *testing.T) {
	data := longStereoOGG
	offs := findPageOffsets(data)
	if len(offs) < 6 {
		t.Fatalf("fixture has only %d pages", len(offs))
	}
	// Cut the stream halfway through a middle page's content.
	cut := (offs[4] + offs[5]) / 2
	truncated := data[:cut]

	idx := vorbis.NewVorbisSeekIndex()
	err := idx.Build(bytes.NewReader(truncated))
	if err == nil {
		t.Fatal("Build on a truncated stream must report an error")
	}
	// The intact prefix must still be indexed and remain usable.
	if idx.Len() == 0 {
		t.Fatal("the intact prefix produced no index entries")
	}
}

func TestVorbisSeekIndexBuildWrongChecksum(t *testing.T) {
	data := append([]byte(nil), longStereoOGG...)
	offs := findPageOffsets(data)
	// Corrupt a content byte of a middle page (well past its 27-byte header).
	pos := offs[4] + 40
	data[pos] ^= 0xFF

	idx := vorbis.NewVorbisSeekIndex()
	err := idx.Build(bytes.NewReader(data))
	if err == nil {
		t.Fatal("Build on a checksum-corrupt stream must report an error")
	}
	// The error must identify the page by its sequence number.
	if got := err.Error(); !bytes.Contains([]byte(got), []byte("sequence")) {
		t.Errorf("error %q does not carry page/sequence context", got)
	}
}

func TestVorbisSeekIndexBuildGranuleRegression(t *testing.T) {
	data := append([]byte(nil), longStereoOGG...)
	offs := findPageOffsets(data)
	// Rewrite a late page's granule position so it regresses relative to the
	// preceding audio page while staying non-negative, then fix its CRC so
	// framing itself is otherwise intact.
	target := offs[len(offs)-2]
	prev := offs[len(offs)-3]
	prevGranule := int64(binary.LittleEndian.Uint64(data[prev+6:]))
	regressed := prevGranule / 2
	binary.LittleEndian.PutUint64(data[target+6:], uint64(regressed))
	nseg := int(data[target+26])
	contentStart := target + 27 + int64(nseg)
	contentSize := int64(0)
	for _, v := range data[target+27 : contentStart] {
		contentSize += int64(v)
	}
	for i := 22; i < 26; i++ {
		data[target+int64(i)] = 0
	}
	crc := testCRC(0, data[target:contentStart])
	crc = testCRC(crc, data[contentStart:contentStart+contentSize])
	binary.LittleEndian.PutUint32(data[target+22:target+26], crc)

	idx := vorbis.NewVorbisSeekIndex()
	err := idx.Build(bytes.NewReader(data))
	if err == nil {
		t.Fatal("Build on a granule-regressing stream must report an error")
	}
	if got := err.Error(); !bytes.Contains([]byte(got), []byte("granule")) {
		t.Errorf("error %q does not mention the granule regression", got)
	}
}

func TestVorbisSeekIndexBuildChainedStream(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex()
	err := idx.Build(bytes.NewReader(chainedOGG))
	if err == nil {
		t.Fatal("Build on a chained Ogg stream must report an error")
	}
	if got := err.Error(); !bytes.Contains([]byte(got), []byte("serial")) {
		t.Errorf("error %q does not mention the stream serial number", got)
	}
}

func TestVorbisSeekIndexBuildShortRead(t *testing.T) {
	data := longStereoOGG
	// A reader that fails in the middle of the scan with a short read.
	src := &failingReadSeeker{
		inner:  bytes.NewReader(data),
		failAt: int64(len(data)) / 2,
	}
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.Build(src); err == nil {
		t.Fatal("Build must surface a short-read error instead of looping forever")
	}
}

// failingReadSeeker returns errFail once a read crosses failAt.
type failingReadSeeker struct {
	inner  io.ReadSeeker
	failAt int64
	failed bool
}

var errFail = errors.New("seekindex_test: forced short read")

func (f *failingReadSeeker) Read(p []byte) (int, error) {
	pos, _ := f.inner.Seek(0, io.SeekCurrent)
	if !f.failed && f.failAt > pos && f.failAt <= pos+int64(len(p)) {
		n, _ := f.inner.Read(p[:f.failAt-pos])
		f.failed = true
		return int(n), io.ErrUnexpectedEOF
	}
	return f.inner.Read(p)
}

func (f *failingReadSeeker) Seek(offset int64, whence int) (int64, error) {
	return f.inner.Seek(offset, whence)
}

// --- synthetic framing: continued packets are never indexed ---------------

func TestVorbisSeekIndexSkipsContinuedPackets(t *testing.T) {
	const serial uint32 = 7
	var stream []byte
	// BOS + a header-ish page.
	page0 := buildOggPage(0x02, 0, serial, 0, [][]byte{{1, 2, 3}}, false)
	page1 := buildOggPage(0, 0, serial, 1, [][]byte{{4, 5}, {6, 7}}, false)
	// A page whose last packet continues on the next page: not an anchor. The
	// 510-byte packet fills two 255-byte segments and leaves a remainder carried
	// by the continuation page.
	big := make([]byte, 510)
	page2 := buildOggPage(0, 1024, serial, 2, [][]byte{big}, true)
	// The continuation page: carries the continued-packet flag, not an anchor.
	page3 := buildOggPage(0x01, 2048, serial, 3, [][]byte{{8, 9}}, false)
	// A normal, self-contained page: a valid anchor.
	page4 := buildOggPage(0, 3072, serial, 4, [][]byte{{10, 11, 12}}, false)
	for _, p := range [][]byte{page0, page1, page2, page3, page4} {
		stream = append(stream, p...)
	}
	page2Off := int64(len(page0) + len(page1))
	page3Off := page2Off + int64(len(page2))
	page4Off := page3Off + int64(len(page3))

	idx := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(1))
	if err := idx.Build(bytes.NewReader(stream)); err != nil {
		t.Fatalf("Build: %v", err)
	}
	raw, err := idx.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	// Entry layout: 28-byte fixed header, then 20-byte entries whose offset is
	// the second uint64.
	const entrySize = 20
	n := int(binary.LittleEndian.Uint32(raw[24:28]))
	var offsets []int64
	for k := range n {
		off := int64(binary.LittleEndian.Uint64(raw[28+k*entrySize+8:]))
		offsets = append(offsets, off)
	}
	for _, off := range offsets {
		if off == page2Off || off == page3Off {
			t.Errorf("a page involved in a cross-page packet was indexed at offset %d: %v", off, offsets)
		}
	}
	foundPage4 := false
	for _, off := range offsets {
		if off == page4Off {
			foundPage4 = true
		}
	}
	if !foundPage4 {
		t.Errorf("the self-contained page at offset %d was not indexed; got %v", page4Off, offsets)
	}
}

// --- runtime fallback: mismatched and damaged index ----------------------

func TestVorbisSeekIndexFallsBackOnMismatch(t *testing.T) {
	// Build a valid index, then shift every offset so the recorded pages no
	// longer line up with the actual source. The on-demand verification must
	// reject the first entry used and trigger the linear fallback.
	good := vorbis.NewVorbisSeekIndex()
	if err := good.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}
	raw, err := good.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	// Corrupt the first entry's offset field (header is 28 bytes; each entry is
	// 20 bytes and the offset is its second uint64).
	if len(raw) < 28+20 {
		t.Fatal("serialized index is unexpectedly short")
	}
	binary.LittleEndian.PutUint64(raw[28+8:28+16], binary.LittleEndian.Uint64(raw[28+8:28+16])+7)
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.UnmarshalBinary(raw); err != nil {
		t.Fatalf("injecting a shifted offset: %v", err)
	}

	plain, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	indexed, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := indexed.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}

	target := int64(4096)
	refPos, err := plain.Seek(target, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}
	pos, err := indexed.Seek(target, io.SeekStart)
	if err != nil {
		t.Fatalf("a mismatched index must fall back, not fail: %v", err)
	}
	if pos != refPos {
		t.Errorf("seek position: got %d, want %d", pos, refPos)
	}
	want := make([]byte, 1024)
	got := make([]byte, 1024)
	if _, err := io.ReadFull(plain, want); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(indexed, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Error("the fallback seek produced different bytes than the linear seek")
	}
	if stats := idx.Stats(); stats.Fallbacks == 0 {
		t.Errorf("a mismatched index must record a fallback: %+v", stats)
	}
}

func TestVorbisSeekIndexFallsBackOnDamagedSource(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}

	// Remove a whole middle page: every recorded offset past it now points at
	// the wrong bytes, so the indexed path must fail verification and fall back.
	offs := findPageOffsets(longStereoOGG)
	removed := offs[len(offs)/2]
	next := offs[len(offs)/2+1]
	damaged := append(append([]byte(nil), longStereoOGG[:removed]...), longStereoOGG[next:]...)

	indexed, err := vorbis.DecodeWithoutResampling(bytes.NewReader(damaged))
	if err != nil {
		t.Fatal(err)
	}
	if err := indexed.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}
	// The seek must terminate (no infinite loop). A damaged region either
	// fails verification and falls back, or errors when the linear scan cannot
	// resync; both outcomes are acceptable.
	done := make(chan error, 1)
	go func() {
		_, seekErr := indexed.Seek(int64(len(damaged))*2, io.SeekStart)
		done <- seekErr
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the seek did not terminate on a damaged source")
	}
}

// --- invalid targets with an index attached ------------------------------

func TestVorbisSeekIndexInvalidTargets(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"Mono", longMonoOGG},
		{"Stereo", longStereoOGG},
	} {
		t.Run(tc.name, func(t *testing.T) {
			idx := vorbis.NewVorbisSeekIndex()
			if err := idx.Build(bytes.NewReader(tc.data)); err != nil {
				t.Fatal(err)
			}
			s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal(err)
			}
			if err := s.SetVorbisSeekIndex(idx); err != nil {
				t.Fatal(err)
			}

			// Negative and overflowing targets are rejected without breaking the
			// stream.
			if _, err := s.Seek(-8, io.SeekStart); err == nil {
				t.Error("Seek(-8, SeekStart): expected an error")
			}
			if _, err := s.Seek(-s.Length()-1, io.SeekEnd); err == nil {
				t.Error("Seek beyond the start from SeekEnd: expected an error")
			}
			// An over-long target is accepted and frame-aligned, matching the
			// unindexed contract; the following read must immediately report EOF.
			const sampleSize = 4
			over := (s.Length() + 4096) / sampleSize * sampleSize
			pos, err := s.Seek(s.Length()+4096, io.SeekStart)
			if err != nil {
				t.Fatalf("seeking past the end: %v", err)
			}
			if pos != over {
				t.Errorf("over-long seek returned %d, want frame-aligned %d", pos, over)
			}
			if n, rerr := s.Read(make([]byte, 64)); n != 0 || !errors.Is(rerr, io.EOF) {
				t.Errorf("read past the end = (%d, %v), want (0, EOF)", n, rerr)
			}
			// The stream still decodes after the rejected seeks.
			if _, err := s.Seek(0, io.SeekStart); err != nil {
				t.Fatal(err)
			}
			buf := make([]byte, 64)
			if _, err := io.ReadFull(s, buf); err != nil {
				t.Fatalf("Read after the rejected seeks: %v", err)
			}
		})
	}
}

// --- repeated and looping seeks stay deterministic -----------------------

func TestVorbisSeekIndexRepeatedAndLoopingSeeks(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(3))
	if err := idx.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}
	s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}

	const sampleSize = 4
	first := int64(1024 * sampleSize)
	second := s.Length() / 2 / sampleSize * sampleSize
	want := make([]byte, 256)
	if _, err := s.Seek(first, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(s, want); err != nil {
		t.Fatal(err)
	}

	// Alternate between the two positions many times; every result must match.
	for range 100 {
		if _, err := s.Seek(second, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Seek(first, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		got := make([]byte, len(want))
		if _, err := io.ReadFull(s, got); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatal("repeated seeks returned different samples")
		}
	}
}

// --- concurrency: one index, many independent streams --------------------

func TestVorbisSeekIndexConcurrentStreams(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(4))
	if err := idx.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}

	const goroutines = 8
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for g := range goroutines {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			// Every goroutine owns its own source and stream; only the index is
			// shared, which is the documented concurrency boundary.
			s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
			if err != nil {
				errCh <- err
				return
			}
			if err := s.SetVorbisSeekIndex(idx); err != nil {
				errCh <- err
				return
			}
			rng := rand.New(rand.NewPCG(seed, seed+1))
			for range 200 {
				target := rng.Int64N(s.Length()/4) * 4
				if _, err := s.Seek(target, io.SeekStart); err != nil {
					errCh <- err
					return
				}
				buf := make([]byte, 128)
				if _, err := io.ReadFull(s, buf); err != nil {
					errCh <- err
					return
				}
			}
		}(uint64(g + 1))
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	stats := idx.Stats()
	if stats.Fallbacks != 0 {
		t.Errorf("concurrent seeks on an intact source must not fall back: %+v", stats)
	}
	if stats.Hits == 0 {
		t.Errorf("no hits recorded: %+v", stats)
	}
}

// --- index stats and invalidation behave under churn ---------------------

func TestVorbisSeekIndexInvalidationFallsBack(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}
	s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Seek(4096, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if stats := idx.Stats(); stats.Hits == 0 {
		t.Fatalf("expected a hit before invalidation: %+v", stats)
	}

	idx.Invalidate()
	if idx.Len() != 0 {
		t.Fatalf("Invalidate left %d entries", idx.Len())
	}
	missesBefore := idx.Stats().Misses
	if _, err := s.Seek(8192, io.SeekStart); err != nil {
		t.Fatalf("seeking with an invalidated index must use the linear fallback: %v", err)
	}
	if stats := idx.Stats(); stats.Misses != missesBefore+1 {
		t.Errorf("a miss was not recorded after invalidation: %+v", stats)
	}
}

// --- the original, unindexed behavior is unchanged end to end -------------

func TestVorbisSeekIndexDetach(t *testing.T) {
	idx := vorbis.NewVorbisSeekIndex()
	if err := idx.Build(bytes.NewReader(longStereoOGG)); err != nil {
		t.Fatal(err)
	}
	s, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}
	if err := s.SetVorbisSeekIndex(nil); err != nil {
		t.Fatalf("detaching the index: %v", err)
	}

	plain, err := vorbis.DecodeWithoutResampling(bytes.NewReader(longStereoOGG))
	if err != nil {
		t.Fatal(err)
	}
	target := int64(4096)
	if _, err := s.Seek(target, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if _, err := plain.Seek(target, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	got, want := make([]byte, 1024), make([]byte, 1024)
	if _, err := io.ReadFull(s, got); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(plain, want); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Error("after detaching the index, seeks differ from an unindexed stream")
	}
}

// TestVorbisSeekIndexOnTruncatedStream covers a stream whose last page is cut,
// so the decoder cannot determine its length. The intact prefix must still be
// indexed and an indexed seek inside that prefix must match the linear seek.
func TestVorbisSeekIndexOnTruncatedStream(t *testing.T) {
	offs := findPageOffsets(longStereoOGG)
	if len(offs) < 8 {
		t.Fatalf("fixture has only %d pages", len(offs))
	}
	// Keep a generous intact prefix and drop the final two pages entirely.
	truncated := longStereoOGG[:offs[len(offs)-2]]

	idx := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(2))
	if err := idx.Build(bytes.NewReader(truncated)); err != nil {
		t.Fatalf("Build over the intact prefix: %v", err)
	}
	if idx.Len() == 0 {
		t.Fatal("no entries were built from the intact prefix")
	}

	plain, err := vorbis.DecodeWithoutResampling(bytes.NewReader(truncated))
	if err != nil {
		t.Fatal(err)
	}
	indexed, err := vorbis.DecodeWithoutResampling(bytes.NewReader(truncated))
	if err != nil {
		t.Fatal(err)
	}
	if err := indexed.SetVorbisSeekIndex(idx); err != nil {
		t.Fatal(err)
	}

	// Seek well inside the indexed, intact region and compare against a fresh
	// linear seek to the same position.
	target := int64(2048)
	if _, err := plain.Seek(target, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if _, err := indexed.Seek(target, io.SeekStart); err != nil {
		t.Fatalf("indexed seek on a truncated stream: %v", err)
	}
	got, want := make([]byte, 1024), make([]byte, 1024)
	if _, err := io.ReadFull(plain, want); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(indexed, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Error("an indexed seek on a truncated stream differs from the linear seek")
	}
}

// TestVorbisSeekIndexDifferentBlockSizes runs the indexed-seek equivalence
// check across all generated fixtures, which use distinct page durations and
// therefore exercise different Vorbis block sizes and packet/page ratios.
func TestVorbisSeekIndexDifferentBlockSizes(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"LongStereo", longStereoOGG},
		{"LongMono", longMonoOGG},
		{"Sparse", sparseOGG},
	} {
		t.Run(tc.name, func(t *testing.T) {
			idx := vorbis.NewVorbisSeekIndex(vorbis.WithVorbisSeekIndexInterval(3))
			if err := idx.Build(bytes.NewReader(tc.data)); err != nil {
				t.Fatal(err)
			}
			indexed, err := vorbis.DecodeWithoutResampling(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatal(err)
			}
			if err := indexed.SetVorbisSeekIndex(idx); err != nil {
				t.Fatal(err)
			}
			for _, target := range []int64{0, 4096, 65536, indexed.Length() / 2} {
				target -= target % 4
				plain, err := vorbis.DecodeWithoutResampling(bytes.NewReader(tc.data))
				if err != nil {
					t.Fatal(err)
				}
				if _, err := plain.Seek(target, io.SeekStart); err != nil {
					t.Fatal(err)
				}
				if _, err := indexed.Seek(target, io.SeekStart); err != nil {
					t.Fatal(err)
				}
				got, want := make([]byte, 512), make([]byte, 512)
				if _, err := io.ReadFull(plain, want); err != nil {
					t.Fatal(err)
				}
				if _, err := io.ReadFull(indexed, got); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Errorf("target %d: indexed and linear seek differ", target)
				}
			}
		})
	}
}
