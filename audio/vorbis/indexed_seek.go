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

package vorbis

import (
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/jfreymuth/oggvorbis"
)

// indexedSource wraps the seekable source of an oggvorbis.Reader.
//
// Outside an indexed seek it is an identity wrapper. During an indexed seek a
// virtual window is opened. The virtual Ogg stream presented to the decoder is
// the source's own header region (the pages carrying the three Vorbis header
// packets) immediately followed by the source's tail beginning at the chosen
// anchor page. The bytes in both regions are served verbatim, so every page
// keeps its original, absolute granule position and its valid CRC: the decoder
// sees a perfectly ordinary Ogg/Vorbis stream, only shorter by the skipped
// prefix. Its page scan therefore covers just the few pages between the anchor
// and the target instead of the whole file.
//
// While a window is open, virtual offsets map to physical source offsets by
// headerEnd + tailBase - virtualOffset for tail reads. When the window closes
// the physical source is already positioned on the right page, because every
// read and seek was translated to a real physical offset; no re-anchoring is
// needed.
type indexedSource struct {
	source io.ReadSeeker

	// When active is false the wrapper is an identity view and the other window
	// fields are ignored.
	active    bool
	headerEnd int64
	tailBase  int64
	virtual   int64
}

// openWindow starts presenting a virtual stream made of the header region
// [0, headerEnd) followed by the tail beginning at tailBase.
func (s *indexedSource) openWindow(headerEnd, tailBase int64) {
	s.active = true
	s.headerEnd = headerEnd
	s.tailBase = tailBase
	s.virtual = 0
}

// closeWindow restores the identity view. The underlying source's physical
// position is left untouched.
func (s *indexedSource) closeWindow() {
	s.active = false
	s.headerEnd = 0
	s.tailBase = 0
	s.virtual = 0
}

// physical maps a virtual offset to a physical source offset inside a window.
func (s *indexedSource) physical(virtual int64) int64 {
	if virtual < s.headerEnd {
		return virtual
	}
	return s.tailBase + (virtual - s.headerEnd)
}

func (s *indexedSource) Read(p []byte) (int, error) {
	if !s.active {
		return s.source.Read(p)
	}

	// Translate the current virtual position to a physical one and keep the two
	// positions synchronized as bytes are delivered.
	phys := s.physical(s.virtual)
	if _, err := s.source.Seek(phys, io.SeekStart); err != nil {
		return 0, err
	}
	n, err := s.source.Read(p)
	s.virtual += int64(n)
	return n, err
}

func (s *indexedSource) Seek(offset int64, whence int) (int64, error) {
	if !s.active {
		return s.source.Seek(offset, whence)
	}

	switch whence {
	case io.SeekStart:
		s.virtual = offset
	case io.SeekCurrent:
		s.virtual += offset
	case io.SeekEnd:
		size, err := s.source.Seek(0, io.SeekEnd)
		if err != nil {
			return 0, err
		}
		// The virtual size excludes the skipped prefix between headerEnd and
		// tailBase.
		virtualSize := s.headerEnd + (size - s.tailBase)
		s.virtual = virtualSize + offset
	default:
		return 0, fmt.Errorf("vorbis: invalid whence %d", whence)
	}
	if s.virtual < 0 {
		return 0, errors.New("vorbis: negative seek position")
	}
	return s.virtual, nil
}

// seekIndexer gives the oggvorbis-backed streams an optional seek index.
//
// All of its methods are safe for concurrent use. The mutex serializes seeks
// against one another; it does not cover Read, which each Stream documents as
// not being safe for concurrent use anyway.
type seekIndexer struct {
	source *indexedSource
	index  *VorbisSeekIndex

	mu sync.Mutex
}

// attach stores the index. A nil index restores the unindexed behavior.
func (x *seekIndexer) attach(index *VorbisSeekIndex) {
	x.mu.Lock()
	x.index = index
	x.source.closeWindow()
	x.mu.Unlock()
}

// verifyEntry reads the indexed page's header and checks that the source still
// contains the page recorded by the index.
func (x *seekIndexer) verifyEntry(entry VorbisSeekIndexEntry) error {
	if entry.Offset < 0 {
		return fmt.Errorf("vorbis: seek index entry has a negative offset %d", entry.Offset)
	}
	if _, err := x.source.source.Seek(entry.Offset, io.SeekStart); err != nil {
		return fmt.Errorf("vorbis: failed to seek to the indexed page at offset %d: %w", entry.Offset, err)
	}
	p, err := readOggPageHeader(x.source.source, int(entry.PageSequence))
	if err != nil {
		return fmt.Errorf("vorbis: the indexed page (sequence %d, offset %d) could not be read: %w", entry.PageSequence, entry.Offset, err)
	}
	if p.granule != entry.GranulePosition {
		return fmt.Errorf("vorbis: the indexed page (sequence %d) granule position changed: got %d, want %d", entry.PageSequence, p.granule, entry.GranulePosition)
	}
	if p.sequence != entry.PageSequence {
		return fmt.Errorf("vorbis: the page at the indexed offset %d has sequence number %d, want %d", entry.Offset, p.sequence, entry.PageSequence)
	}
	if p.continues || p.headerType&oggHeaderFlagContinuedPacket != 0 || p.packetCount == 0 {
		return fmt.Errorf("vorbis: the indexed page (sequence %d) cannot anchor a seek", entry.PageSequence)
	}
	return nil
}

// seekToSample positions the oggvorbis reader on the given PCM sample position,
// using the index when possible and otherwise falling back to the reader's
// ordinary linear page scan.
//
// The reader itself performs the final, sample-accurate step: once a safe page
// is selected, it decodes from the page and discards samples until the target
// frame is reached.
func (x *seekIndexer) seekToSample(r *oggvorbis.Reader, pos int64) error {
	x.mu.Lock()
	defer x.mu.Unlock()

	// Every seek starts from an unwindowed view; a previous window must never
	// leak into either the linear fallback or a fresh window.
	x.source.closeWindow()

	index := x.index
	if index == nil || index.Len() == 0 {
		if index != nil {
			index.recordMiss()
		}
		return r.SetPosition(pos)
	}

	entry, ok := index.lookup(pos)
	if !ok {
		// The target is before the first indexed page: a full linear scan is the
		// only safe option.
		index.recordMiss()
		return r.SetPosition(pos)
	}

	headerEnd := index.headerEndOffset()
	if headerEnd < 0 || entry.Offset < headerEnd {
		// A structurally unusable index: fall back rather than risk a bad window.
		index.recordFallback()
		return r.SetPosition(pos)
	}

	if err := x.verifyEntry(entry); err != nil {
		// The index does not match this source. Drop the bad entry, count the
		// fallback and retry exactly once with the reader's own linear scan,
		// which can never loop forever as it advances through bounded pages.
		index.invalidateEntry(entry.Offset)
		index.recordFallback()
		if fallbackErr := r.SetPosition(pos); fallbackErr != nil {
			return errors.Join(fmt.Errorf("%w; linear fallback also failed", err), fallbackErr)
		}
		return nil
	}

	// Open a virtual stream: original header pages followed by the tail starting
	// at the anchor page, each byte served verbatim. The target sample position
	// is absolute and unchanged because the anchor page keeps its real granule.
	x.source.openWindow(headerEnd, entry.Offset)
	if err := r.SetPosition(pos); err != nil {
		// Close the window and retry the ordinary, coordinate-correct linear seek.
		x.source.closeWindow()
		index.recordFallback()
		if fallbackErr := r.SetPosition(pos); fallbackErr != nil {
			return errors.Join(fmt.Errorf("vorbis: indexed seek to sample %d failed: %w; linear fallback also failed", pos, err), fallbackErr)
		}
		return nil
	}

	// The decoder finished its (short) page scan and holds the relevant packets
	// in memory. The physical source already points at the page following the
	// ones it scanned, so closing the window needs no re-anchoring.
	x.source.closeWindow()
	index.recordHit()
	return nil
}
