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
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"
)

// DefaultVorbisSeekIndexInterval is the default spacing, in Ogg pages, between
// the entries of a [VorbisSeekIndex].
const DefaultVorbisSeekIndexInterval = 8

var (
	oggCapturePattern = [4]byte{'O', 'g', 'g', 'S'}

	oggHeaderFlagContinuedPacket   = byte(0x01)
	oggHeaderFlagBeginningOfStream = byte(0x02)
)

// VorbisSeekIndexEntry is one entry of a [VorbisSeekIndex].
//
// An entry records an Ogg page that can safely be used as the starting point of
// a seek: the page begins a packet and completes at least one packet, so a
// Vorbis decoder can be reset on it without reconstructing a packet that spans
// multiple pages.
type VorbisSeekIndexEntry struct {
	// GranulePosition is the page's absolute granule position in PCM samples.
	GranulePosition int64

	// Offset is the page's byte offset in the seekable source.
	Offset int64

	// PageSequence is the page's Ogg page sequence number.
	PageSequence uint32
}

// VorbisSeekIndexStats reports how often an attached index was able to serve a
// seek and how often a seek had to fall back to a linear scan.
type VorbisSeekIndexStats struct {
	// Hits is the number of seeks that started from an indexed page.
	Hits uint64

	// Misses is the number of seeks for which the index had no usable entry,
	// for example because the target was before the first indexed page.
	Misses uint64

	// Fallbacks is the number of seeks that started with an indexed entry but
	// could not use it, because the source did not match the index or the
	// indexed page turned out to be corrupt, and were retried linearly.
	Fallbacks uint64

	// Entries is the number of entries currently held by the index.
	Entries int
}

// VorbisSeekIndex is an optional, sparse index over the Ogg pages of an
// Ogg/Vorbis stream.
//
// Without an index, [Stream.Seek] re-reads the stream from its beginning and
// scans every Ogg page header until it finds the page before the target. With
// an index, Seek jumps directly to the nearest indexed page that is known to be
// safe and lets the decoder discard the remaining samples, so the amount of
// work per seek no longer depends on the target position.
//
// An index is built for one particular byte stream. The same index can be
// attached to several [Stream]s decoding identical copies of that stream, for
// example the streams of a chained audio player. A VorbisSeekIndex is safe for
// concurrent use; the streams it is attached to are not, as with any
// [io.Reader].
type VorbisSeekIndex struct {
	interval int

	mu      sync.RWMutex
	entries []VorbisSeekIndexEntry

	// headerEnd is the byte offset immediately after the pages holding the
	// three Vorbis header packets, or -1 while the index is empty.
	headerEnd int64

	hits      uint64
	misses    uint64
	fallbacks uint64
}

// VorbisSeekIndexOption configures a [VorbisSeekIndex].
type VorbisSeekIndexOption func(*VorbisSeekIndex)

// WithVorbisSeekIndexInterval makes the index keep at most one entry for every
// interval Ogg pages. A value of 1 indexes every usable page.
//
// A larger interval produces a smaller index at the cost of more decoding work
// per seek.
func WithVorbisSeekIndexInterval(interval int) VorbisSeekIndexOption {
	return func(i *VorbisSeekIndex) {
		if interval > 0 {
			i.interval = interval
		}
	}
}

// NewVorbisSeekIndex creates an empty [VorbisSeekIndex].
//
// The index must be populated with [VorbisSeekIndex.Build],
// [VorbisSeekIndex.UnmarshalBinary] or [VorbisSeekIndex.Load] before it can
// speed up seeks; an empty index makes every seek use the ordinary linear
// fallback.
func NewVorbisSeekIndex(opts ...VorbisSeekIndexOption) *VorbisSeekIndex {
	i := &VorbisSeekIndex{
		interval:  DefaultVorbisSeekIndexInterval,
		headerEnd: -1,
	}
	for _, opt := range opts {
		opt(i)
	}
	return i
}

// Build scans src from its beginning and records usable Ogg pages.
//
// src must be the same seekable source that will be decoded, or a source
// containing the identical bytes. Build restores src's original position on
// success.
//
// A truncated or corrupt page terminates the scan: the entries collected up to
// that page are kept and the error is returned, so an index over the intact
// prefix of a damaged stream remains usable. Building an already built index
// replaces its entries.
func (i *VorbisSeekIndex) Build(src io.ReadSeeker) error {
	start, err := src.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("vorbis: failed to determine the source position: %w", err)
	}
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("vorbis: failed to rewind the source for index building: %w", err)
	}

	entries, headerEnd, err := scanOggPages(src, i.interval)

	// Always restore the source, even when the scan failed, and keep whatever
	// entries were collected from the intact prefix.
	if _, seekErr := src.Seek(start, io.SeekStart); seekErr != nil && err == nil {
		err = seekErr
	}

	i.mu.Lock()
	i.entries = entries
	i.headerEnd = headerEnd
	i.mu.Unlock()
	if err != nil {
		return fmt.Errorf("vorbis: building the seek index failed: %w", err)
	}
	return nil
}

// Load builds the index from a reader containing the serialized format produced
// by [VorbisSeekIndex.MarshalBinary].
func (i *VorbisSeekIndex) Load(r io.Reader) error {
	lr := io.LimitedReader{R: r, N: int64(1<<48) + 1}
	data, err := io.ReadAll(&lr)
	if err != nil {
		return fmt.Errorf("vorbis: failed to read the seek index: %w", err)
	}
	if err := i.UnmarshalBinary(data); err != nil {
		return err
	}
	return nil
}

// Invalidate drops every entry. Subsequent seeks use the linear fallback until
// the index is built or loaded again.
func (i *VorbisSeekIndex) Invalidate() {
	i.mu.Lock()
	i.entries = nil
	i.headerEnd = -1
	i.mu.Unlock()
}

// Len returns the number of entries currently held by the index.
func (i *VorbisSeekIndex) Len() int {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.entries)
}

// headerEndOffset returns the byte offset immediately after the pages holding
// the three Vorbis header packets, or -1 when the index is empty.
func (i *VorbisSeekIndex) headerEndOffset() int64 {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.headerEnd
}

// Stats returns a snapshot of the index's seek statistics and size.
func (i *VorbisSeekIndex) Stats() VorbisSeekIndexStats {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return VorbisSeekIndexStats{
		Hits:      i.hits,
		Misses:    i.misses,
		Fallbacks: i.fallbacks,
		Entries:   len(i.entries),
	}
}

// lookup returns the entry of the page nearest to but not after the sample
// position pos, and whether such an entry exists.
func (i *VorbisSeekIndex) lookup(pos int64) (VorbisSeekIndexEntry, bool) {
	i.mu.RLock()
	defer i.mu.RUnlock()
	// Find the first entry whose granule position is greater than pos; the
	// entry before it is the last one not after pos.
	idx := sort.Search(len(i.entries), func(n int) bool {
		return i.entries[n].GranulePosition > pos
	})
	if idx == 0 {
		return VorbisSeekIndexEntry{}, false
	}
	return i.entries[idx-1], true
}

func (i *VorbisSeekIndex) recordHit() {
	i.mu.Lock()
	i.hits++
	i.mu.Unlock()
}

func (i *VorbisSeekIndex) recordMiss() {
	i.mu.Lock()
	i.misses++
	i.mu.Unlock()
}

func (i *VorbisSeekIndex) recordFallback() {
	i.mu.Lock()
	i.fallbacks++
	i.mu.Unlock()
}

// invalidateEntry drops the entry at the given offset after it proved unusable.
func (i *VorbisSeekIndex) invalidateEntry(offset int64) {
	i.mu.Lock()
	defer i.mu.Unlock()
	idx := sort.Search(len(i.entries), func(n int) bool {
		return i.entries[n].Offset >= offset
	})
	if idx < len(i.entries) && i.entries[idx].Offset == offset {
		i.entries = append(i.entries[:idx], i.entries[idx+1:]...)
	}
}

// seekIndexBinaryVersion is the first byte of a serialized seek index.
const seekIndexBinaryVersion byte = 3

// seekIndexBinaryMagic identifies a serialized seek index.
var seekIndexBinaryMagic = [7]byte{'V', 'B', 'S', 'E', 'E', 'K', 'I'}

// MarshalBinary serializes the index.
//
// The format is self-describing and versioned: a magic marker, a version byte,
// the page interval and header-end offset as uint64s, the entry count as a
// uint32 and a sequence of fixed-size entries. It is stable across
// little-endian machines, which all supported Ebitengine platforms are.
func (i *VorbisSeekIndex) MarshalBinary() ([]byte, error) {
	i.mu.RLock()
	defer i.mu.RUnlock()

	var buf bytes.Buffer
	buf.Write(seekIndexBinaryMagic[:])
	buf.WriteByte(seekIndexBinaryVersion)
	var hdr [8]byte
	binary.LittleEndian.PutUint64(hdr[:], uint64(i.interval))
	buf.Write(hdr[:])
	binary.LittleEndian.PutUint64(hdr[:], uint64(i.headerEnd))
	buf.Write(hdr[:])
	var cnt [4]byte
	binary.LittleEndian.PutUint32(cnt[:], uint32(len(i.entries)))
	buf.Write(cnt[:])

	var entry [20]byte
	for _, e := range i.entries {
		binary.LittleEndian.PutUint64(entry[0:8], uint64(e.GranulePosition))
		binary.LittleEndian.PutUint64(entry[8:16], uint64(e.Offset))
		binary.LittleEndian.PutUint32(entry[16:20], e.PageSequence)
		buf.Write(entry[:])
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary replaces the index's entries with data produced by
// [VorbisSeekIndex.MarshalBinary].
//
// Malformed data is rejected without modifying the existing index. Entries are
// validated structurally; their correspondence with the actual source is
// checked lazily on the first seek that uses them, and a mismatching entry is
// discarded in favor of the linear fallback.
func (i *VorbisSeekIndex) UnmarshalBinary(data []byte) error {
	const fixedHeader = len(seekIndexBinaryMagic) + 1 + 8 + 8 + 4
	const entrySize = 20
	if len(data) < fixedHeader {
		return fmt.Errorf("vorbis: seek index data is too short: %d bytes", len(data))
	}
	if !bytes.Equal(data[:len(seekIndexBinaryMagic)], seekIndexBinaryMagic[:]) {
		return errors.New("vorbis: invalid seek index data: bad magic marker")
	}
	if data[len(seekIndexBinaryMagic)] != seekIndexBinaryVersion {
		return fmt.Errorf("vorbis: unsupported seek index version: %d", data[len(seekIndexBinaryMagic)])
	}
	interval := int64(binary.LittleEndian.Uint64(data[8:16]))
	headerEnd := int64(binary.LittleEndian.Uint64(data[16:24]))
	n := int(binary.LittleEndian.Uint32(data[24:28]))
	if interval <= 0 {
		return fmt.Errorf("vorbis: invalid seek index interval: %d", interval)
	}
	if headerEnd < 0 {
		return fmt.Errorf("vorbis: invalid seek index header-end offset: %d", headerEnd)
	}
	if n < 0 || fixedHeader+n*entrySize != len(data) {
		return fmt.Errorf("vorbis: seek index data has %d bytes, want %d for %d entries", len(data), fixedHeader+n*entrySize, n)
	}

	entries := make([]VorbisSeekIndexEntry, n)
	var lastGranule int64 = -1
	var lastOffset int64 = -1
	for k := range n {
		b := data[fixedHeader+k*entrySize : fixedHeader+(k+1)*entrySize]
		e := VorbisSeekIndexEntry{
			GranulePosition: int64(binary.LittleEndian.Uint64(b[0:8])),
			Offset:          int64(binary.LittleEndian.Uint64(b[8:16])),
			PageSequence:    binary.LittleEndian.Uint32(b[16:20]),
		}
		if e.GranulePosition < 0 {
			return fmt.Errorf("vorbis: seek index entry %d has a negative granule position: %d", k, e.GranulePosition)
		}
		if e.Offset < headerEnd {
			return fmt.Errorf("vorbis: seek index entry %d points inside the header region (offset %d < header end %d)", k, e.Offset, headerEnd)
		}
		if e.Offset < 0 {
			return fmt.Errorf("vorbis: seek index entry %d has a negative offset: %d", k, e.Offset)
		}
		if e.GranulePosition < lastGranule || e.Offset <= lastOffset {
			return fmt.Errorf("vorbis: seek index entry %d is not strictly ordered", k)
		}
		lastGranule = e.GranulePosition
		lastOffset = e.Offset
		entries[k] = e
	}

	i.mu.Lock()
	i.interval = int(interval)
	i.headerEnd = headerEnd
	i.entries = entries
	i.mu.Unlock()
	return nil
}

// oggPageInfo is the result of parsing one Ogg page header.
type oggPageInfo struct {
	offset       int64
	headerType   byte
	granule      int64
	serial       uint32
	sequence     uint32
	checksum     uint32
	segmentCount int
	contentSize  int
	packetCount  int
	continues    bool
	header       [27]byte
	segmentTable []byte
}

// totalSize returns the page's full size in bytes.
func (p *oggPageInfo) totalSize() int {
	return 27 + p.segmentCount + p.contentSize
}

// readOggPageHeader parses an Ogg page header at the current position of r.
//
// Short reads are reported with their context rather than retried forever.
func readOggPageHeader(r io.Reader, pageNo int) (*oggPageInfo, error) {
	var hdr [27]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("vorbis: Ogg page %d: truncated header: %w", pageNo, err)
	}
	if hdr[0] != oggCapturePattern[0] || hdr[1] != oggCapturePattern[1] || hdr[2] != oggCapturePattern[2] || hdr[3] != oggCapturePattern[3] {
		return nil, fmt.Errorf("vorbis: Ogg page %d: missing capture pattern", pageNo)
	}
	if hdr[4] != 0 {
		return nil, fmt.Errorf("vorbis: Ogg page %d: unsupported stream structure version %d", pageNo, hdr[4])
	}
	p := &oggPageInfo{
		header:       hdr,
		headerType:   hdr[5],
		granule:      int64(binary.LittleEndian.Uint64(hdr[6:14])),
		serial:       binary.LittleEndian.Uint32(hdr[14:18]),
		sequence:     binary.LittleEndian.Uint32(hdr[18:22]),
		checksum:     binary.LittleEndian.Uint32(hdr[22:26]),
		segmentCount: int(hdr[26]),
	}
	p.segmentTable = make([]byte, p.segmentCount)
	if _, err := io.ReadFull(r, p.segmentTable); err != nil {
		return nil, fmt.Errorf("vorbis: Ogg page %d (sequence %d): truncated segment table: %w", pageNo, p.sequence, err)
	}
	size := 0
	for _, s := range p.segmentTable {
		p.contentSize += int(s)
		size += int(s)
		if s < 0xFF {
			p.packetCount++
			size = 0
		}
	}
	p.continues = p.segmentTable[p.segmentCount-1] == 0xFF
	return p, nil
}

// scanOggPages reads every Ogg page of src and records the pages that can
// safely anchor a seek.
//
// Pages are fully checksummed. A page whose packet continues into the next
// page, a page that continues another page's packet, and the pages carrying the
// three Vorbis header packets are not usable anchors and are skipped.
//
// headerEnd is the byte offset of the first page beyond the header pages, i.e.
// the byte immediately after the pages holding the three Vorbis header packets.
func scanOggPages(src io.ReadSeeker, interval int) (entries []VorbisSeekIndexEntry, headerEnd int64, err error) {
	var offset int64
	var lastSequence int64 = -1
	var serial uint32
	serialKnown := false
	usablePages := 0
	pageNo := 0
	completedPackets := 0
	headerEnd = -1
	lastGranule := int64(-1)

	// Bound the loop defensively: every parsed page consumes at least 27 bytes,
	// so a stream can never contain more than size/27 pages. A corrupt length
	// field cannot make the scan spin, since every read is bounded.
	for {
		p, err := readOggPageHeader(src, pageNo)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return entries, headerEnd, nil
			}
			return entries, headerEnd, err
		}

		// Read and checksum the whole page, so an index entry never points at a
		// page whose content is corrupt.
		content := make([]byte, p.contentSize)
		if _, err := io.ReadFull(src, content); err != nil {
			return entries, headerEnd, fmt.Errorf("vorbis: Ogg page %d (sequence %d): truncated content: %w", pageNo, p.sequence, err)
		}
		headerForCRC := p.header
		headerForCRC[22], headerForCRC[23], headerForCRC[24], headerForCRC[25] = 0, 0, 0, 0
		crc := oggCRCUpdate(0, headerForCRC[:])
		crc = oggCRCUpdate(crc, p.segmentTable)
		crc = oggCRCUpdate(crc, content)
		if crc != p.checksum {
			return entries, headerEnd, fmt.Errorf("vorbis: Ogg page %d (sequence %d): wrong checksum", pageNo, p.sequence)
		}

		if !serialKnown {
			serial = p.serial
			serialKnown = true
		} else if p.serial != serial {
			// A chained stream starts a new logical bitstream; its granule
			// positions restart at zero, so a single index cannot cover both.
			return entries, headerEnd, fmt.Errorf("vorbis: Ogg page %d (sequence %d): unexpected stream serial number %d (chained streams are not supported)", pageNo, p.sequence, p.serial)
		}
		if int64(p.sequence) != lastSequence+1 {
			return entries, headerEnd, fmt.Errorf("vorbis: Ogg page %d: sequence number went from %d to %d", pageNo, lastSequence, p.sequence)
		}
		lastSequence = int64(p.sequence)

		// The three Vorbis header packets occupy the beginning of the stream.
		// Record the start of the first page beyond the header pages.
		if headerEnd < 0 {
			completedPackets += p.packetCount
			if completedPackets >= 3 {
				headerEnd = offset
			}
		}

		usable := p.granule >= 0 &&
			p.packetCount > 0 &&
			!p.continues &&
			p.headerType&oggHeaderFlagContinuedPacket == 0 &&
			p.headerType&oggHeaderFlagBeginningOfStream == 0
		if usable {
			if lastGranule >= 0 && p.granule < lastGranule {
				return entries, headerEnd, fmt.Errorf("vorbis: Ogg page %d (sequence %d): granule position went backwards from %d to %d", pageNo, p.sequence, lastGranule, p.granule)
			}
			lastGranule = p.granule
			if usablePages%interval == 0 {
				entries = append(entries, VorbisSeekIndexEntry{
					GranulePosition: p.granule,
					Offset:          offset,
					PageSequence:    p.sequence,
				})
			}
			usablePages++
		}

		offset += int64(p.totalSize())
		pageNo++
	}
}

// oggCRCTable is the lookup table for the Ogg framing CRC.
var oggCRCTable [256]uint32

func init() {
	for i := range oggCRCTable {
		r := uint32(i) << 24
		for j := 0; j < 8; j++ {
			if r&0x80000000 != 0 {
				r = (r << 1) ^ 0x04c11db7
			} else {
				r <<= 1
			}
		}
		oggCRCTable[i] = r
	}
}

// oggCRCUpdate extends crc with the Ogg framing checksum of data.
func oggCRCUpdate(crc uint32, data []byte) uint32 {
	for _, b := range data {
		crc = (crc << 8) ^ oggCRCTable[byte(crc>>24)^b]
	}
	return crc
}
