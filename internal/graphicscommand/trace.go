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

package graphicscommand

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
)

// TraceVersion is the version of the trace format produced and consumed by this package.
//
// The version is incremented whenever the event or frame layout changes in an incompatible way.
const TraceVersion = 1

// DefaultMaxTraceEvents is the default upper bound of events a GraphicsCommandTrace retains.
// Recording stops at the bound and the affected frames are marked as truncated.
const DefaultMaxTraceEvents = 1 << 18

// ErrTraceVersionMismatch is returned when a trace of an unsupported version is decoded, replayed, or verified.
var ErrTraceVersionMismatch = errors.New("graphicscommand: trace version mismatch")

// TraceEventKind classifies a recorded command event.
type TraceEventKind string

const (
	// TraceEventBarrier marks a structural boundary: the begin or end of a command batch, or a device reset.
	TraceEventBarrier TraceEventKind = "barrier"

	TraceEventNewImage      TraceEventKind = "new-image"
	TraceEventWritePixels   TraceEventKind = "write-pixels"
	TraceEventDrawTriangles TraceEventKind = "draw-triangles"
	TraceEventReadPixels    TraceEventKind = "read-pixels"
	TraceEventDisposeImage  TraceEventKind = "dispose-image"
	TraceEventNewShader     TraceEventKind = "new-shader"
	TraceEventDisposeShader TraceEventKind = "dispose-shader"

	// TraceEventError records a failed command or graphics driver call with its context.
	TraceEventError TraceEventKind = "error"

	// TraceEventCommand is a command without a dedicated event kind.
	TraceEventCommand TraceEventKind = "command"
)

// TraceParam is a single key-value parameter summary of a TraceEvent.
//
// Values are redacted summaries: pixel contents and uniform values are never stored, only their
// sizes and digests are.
type TraceParam struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// TraceEvent is a single recorded command.
//
// A TraceEvent holds only value data: sequence number, resource identifiers, and redacted
// parameter summaries. It never references images, shaders, pixel buffers, or uniform buffers,
// so a recorded trace cannot retain resources that the command queue recycles after a flush.
type TraceEvent struct {
	// Seq is the stable sequence number of the event. It starts at 0 after a reset and
	// increments by 1 for every recorded event.
	Seq  uint64         `json:"seq"`
	Kind TraceEventKind `json:"kind"`

	// Image is the identifier of the primary image the command operates on, if any.
	Image int `json:"image,omitempty"`
	// Shader is the identifier of the shader the command uses, if any.
	Shader int `json:"shader,omitempty"`
	// Srcs lists the source image identifiers of a draw command. A nil source is -1.
	Srcs []int `json:"srcs,omitempty"`

	// Params is an ordered summary of the command's key parameters.
	Params []TraceParam `json:"params,omitempty"`

	// Err is the error message of a failed command or driver call.
	Err string `json:"err,omitempty"`
}

// Param returns the value of the first parameter with the given key.
func (e TraceEvent) Param(key string) (string, bool) {
	for _, p := range e.Params {
		if p.Key == key {
			return p.Value, true
		}
	}
	return "", false
}

// TraceFrame is the sequence of events submitted between the begin and the end of a frame.
//
// A frame spans all the command batches flushed for it, including intermediate flushes.
type TraceFrame struct {
	Index  uint64       `json:"index"`
	Events []TraceEvent `json:"events"`

	// Truncated reports that events of this frame were dropped because the trace reached
	// its event limit.
	Truncated bool `json:"truncated,omitempty"`
}

// Trace is an exported snapshot of recorded frames.
type Trace struct {
	Version int `json:"version"`

	// Backend identifies the graphics backend the trace was recorded on, e.g. "opengl".
	// Traces recorded on different backends can legitimately differ in their command streams.
	Backend string `json:"backend,omitempty"`

	Frames []TraceFrame `json:"frames"`
}

// checkVersion reports whether the trace can be consumed by this package.
func (t *Trace) checkVersion() error {
	if t.Version != TraceVersion {
		return fmt.Errorf("graphicscommand: trace version %d is not supported, want %d: %w", t.Version, TraceVersion, ErrTraceVersionMismatch)
	}
	return nil
}

// Encode writes the trace as JSON. The output is deterministic: encoding the same trace twice
// produces identical bytes.
func (t *Trace) Encode(w io.Writer) error {
	return json.NewEncoder(w).Encode(t)
}

// TraceError describes a problem with a specific event of a trace.
type TraceError struct {
	Frame uint64
	Seq   uint64
	Err   error
}

func (e *TraceError) Error() string {
	return fmt.Sprintf("graphicscommand: trace: frame %d, event seq %d: %v", e.Frame, e.Seq, e.Err)
}

func (e *TraceError) Unwrap() error {
	return e.Err
}

// Replay visits every recorded event in its original order.
//
// Replay does not modify the trace, so replaying the same trace multiple times yields the same
// event sequence. If visit returns an error, Replay stops and returns it wrapped in a TraceError
// identifying the frame and the event.
func (t *Trace) Replay(visit func(ev TraceEvent) error) error {
	if err := t.checkVersion(); err != nil {
		return err
	}
	for _, f := range t.Frames {
		for _, ev := range f.Events {
			if err := visit(ev); err != nil {
				return &TraceError{Frame: f.Index, Seq: ev.Seq, Err: err}
			}
		}
	}
	return nil
}

// resourceState is the lifecycle state of a traced resource for Verify.
type resourceState int

const (
	resourceLive resourceState = iota
	resourceDisposed
	resourceInvalidated
)

// Verify checks the resource lifecycle of the trace: resources must not be used after they were
// disposed, must not be disposed twice, must not be created twice, and must not be used after a
// device reset invalidated them.
//
// Resources first referenced by a use or a dispose are assumed to have been created before the
// trace started, so a trace recorded from the middle of a session is still verifiable.
func (t *Trace) Verify() error {
	if err := t.checkVersion(); err != nil {
		return err
	}

	images := map[int]resourceState{}
	shaders := map[int]resourceState{}

	use := func(table map[int]resourceState, id int, what string, f *TraceFrame, ev TraceEvent) error {
		switch table[id] {
		case resourceDisposed:
			return &TraceError{Frame: f.Index, Seq: ev.Seq, Err: fmt.Errorf("%s %d is used after being disposed", what, id)}
		case resourceInvalidated:
			return &TraceError{Frame: f.Index, Seq: ev.Seq, Err: fmt.Errorf("%s %d is used after a device reset", what, id)}
		}
		if _, ok := table[id]; !ok {
			// The resource might have been created before the trace started.
			table[id] = resourceLive
		}
		return nil
	}
	create := func(table map[int]resourceState, id int, what string, f *TraceFrame, ev TraceEvent) error {
		if _, ok := table[id]; ok {
			return &TraceError{Frame: f.Index, Seq: ev.Seq, Err: fmt.Errorf("%s %d is created more than once", what, id)}
		}
		table[id] = resourceLive
		return nil
	}
	dispose := func(table map[int]resourceState, id int, what string, f *TraceFrame, ev TraceEvent) error {
		if err := use(table, id, what, f, ev); err != nil {
			return err
		}
		table[id] = resourceDisposed
		return nil
	}

	for i := range t.Frames {
		f := &t.Frames[i]
		for _, ev := range f.Events {
			var err error
			switch ev.Kind {
			case TraceEventNewImage:
				err = create(images, ev.Image, "image", f, ev)
			case TraceEventDisposeImage:
				err = dispose(images, ev.Image, "image", f, ev)
			case TraceEventWritePixels, TraceEventReadPixels:
				err = use(images, ev.Image, "image", f, ev)
			case TraceEventDrawTriangles:
				err = use(images, ev.Image, "image", f, ev)
				if err == nil {
					err = use(shaders, ev.Shader, "shader", f, ev)
				}
				for _, src := range ev.Srcs {
					if err != nil {
						break
					}
					if src < 0 {
						continue
					}
					err = use(images, src, "image", f, ev)
				}
			case TraceEventNewShader:
				err = create(shaders, ev.Shader, "shader", f, ev)
			case TraceEventDisposeShader:
				err = dispose(shaders, ev.Shader, "shader", f, ev)
			case TraceEventBarrier:
				if phase, ok := ev.Param("phase"); ok && phase == "reset" {
					// A device reset invalidates every live resource.
					for id, s := range images {
						if s == resourceLive {
							images[id] = resourceInvalidated
						}
					}
					for id, s := range shaders {
						if s == resourceLive {
							shaders[id] = resourceInvalidated
						}
					}
				}
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// DecodeTrace decodes a trace exported by GraphicsCommandTrace.Export or Trace.Encode.
func DecodeTrace(r io.Reader) (*Trace, error) {
	var t Trace
	if err := json.NewDecoder(r).Decode(&t); err != nil {
		return nil, fmt.Errorf("graphicscommand: decoding a trace: %w", err)
	}
	if err := t.checkVersion(); err != nil {
		return nil, err
	}
	return &t, nil
}

// GraphicsCommandTrace records the graphics commands submitted per frame.
//
// Recording is disabled by default, and a disabled trace has no effect on the command queue
// beyond a single atomic load per command. Enable recording with SetEnabled, then read the
// recorded frames with Snapshot or Export.
//
// A GraphicsCommandTrace is safe for concurrent use: commands are recorded on the render thread
// while snapshots and exports can be taken from any goroutine.
type GraphicsCommandTrace struct {
	enabled atomic.Bool

	mu          sync.Mutex
	maxEvents   int
	backend     string
	frames      []TraceFrame
	open        int // index of the frame currently being recorded, -1 if none
	depth       int // nesting depth of open frames
	seq         uint64
	frameIndex  uint64
	totalEvents int
}

// theTrace is the process-wide trace that the command queue records into.
var theTrace = newGraphicsCommandTrace()

func newGraphicsCommandTrace() *GraphicsCommandTrace {
	return &GraphicsCommandTrace{
		maxEvents: DefaultMaxTraceEvents,
		open:      -1,
	}
}

// DefaultTrace returns the process-wide trace that records the commands of the global command
// queue manager.
func DefaultTrace() *GraphicsCommandTrace {
	return theTrace
}

// SetEnabled enables or disables recording.
//
// Disabling closes the frame currently being recorded, if any. A disabled trace records nothing
// and adds no behavior to the command queue.
func (t *GraphicsCommandTrace) SetEnabled(enabled bool) {
	t.enabled.Store(enabled)
	if !enabled {
		t.mu.Lock()
		t.closeFrameLocked()
		t.mu.Unlock()
	}
}

// Enabled reports whether recording is enabled.
func (t *GraphicsCommandTrace) Enabled() bool {
	return t.enabled.Load()
}

// SetMaxEvents sets the maximum number of events the trace retains. Recording stops at the
// limit and the affected frames are marked as truncated. A non-positive limit means no limit.
func (t *GraphicsCommandTrace) SetMaxEvents(n int) {
	t.mu.Lock()
	t.maxEvents = n
	t.mu.Unlock()
}

// SetBackend sets the backend name recorded in exported traces.
func (t *GraphicsCommandTrace) SetBackend(backend string) {
	t.mu.Lock()
	t.backend = backend
	t.mu.Unlock()
}

// Reset discards every recorded frame and restarts the sequence and frame numbering.
// The enabled state, the event limit, and the backend name are kept.
func (t *GraphicsCommandTrace) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.frames = nil
	t.open = -1
	t.depth = 0
	t.seq = 0
	t.frameIndex = 0
	t.totalEvents = 0
}

// BeginFrame opens a frame explicitly. Frames can be nested: the frame is completed only when
// every BeginFrame is matched by an EndFrame.
//
// The command queue opens and closes frames implicitly at flushes; explicit BeginFrame and
// EndFrame calls are needed only to group commands outside regular flushes.
func (t *GraphicsCommandTrace) BeginFrame() {
	if !t.enabled.Load() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.depth++
	if t.open < 0 {
		t.beginFrameLocked()
	}
}

// EndFrame closes the innermost open frame. It returns an error when no frame is open, e.g. on
// a duplicate EndFrame.
func (t *GraphicsCommandTrace) EndFrame() error {
	if !t.enabled.Load() {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.open < 0 || t.depth == 0 {
		return errors.New("graphicscommand: trace: EndFrame without a matching BeginFrame")
	}
	t.depth--
	if t.depth == 0 {
		t.closeFrameLocked()
	}
	return nil
}

// Snapshot returns a deep copy of the recorded frames. The result is independent of the trace
// and can be exported, replayed, and verified while recording continues.
func (t *GraphicsCommandTrace) Snapshot() *Trace {
	t.mu.Lock()
	defer t.mu.Unlock()
	tr := &Trace{
		Version: TraceVersion,
		Backend: t.backend,
		Frames:  make([]TraceFrame, len(t.frames)),
	}
	for i, f := range t.frames {
		nf := TraceFrame{
			Index:     f.Index,
			Truncated: f.Truncated,
			Events:    make([]TraceEvent, len(f.Events)),
		}
		for j, ev := range f.Events {
			nev := ev
			nev.Srcs = slices.Clone(ev.Srcs)
			nev.Params = slices.Clone(ev.Params)
			nf.Events[j] = nev
		}
		tr.Frames[i] = nf
	}
	return tr
}

// Export writes a snapshot of the recorded frames as JSON. The output is deterministic for the
// same recorded commands.
func (t *GraphicsCommandTrace) Export(w io.Writer) error {
	return t.Snapshot().Encode(w)
}

func (t *GraphicsCommandTrace) beginFrameLocked() {
	t.frames = append(t.frames, TraceFrame{Index: t.frameIndex})
	t.frameIndex++
	t.open = len(t.frames) - 1
}

func (t *GraphicsCommandTrace) closeFrameLocked() {
	t.open = -1
	t.depth = 0
}

// appendEventLocked assigns the sequence number and appends the event to the open frame,
// opening one when necessary. Events beyond the event limit are dropped and the frame is
// marked as truncated.
func (t *GraphicsCommandTrace) appendEventLocked(ev TraceEvent) {
	if t.open < 0 {
		t.beginFrameLocked()
	}
	f := &t.frames[t.open]
	if t.maxEvents > 0 && t.totalEvents >= t.maxEvents {
		f.Truncated = true
		return
	}
	ev.Seq = t.seq
	t.seq++
	f.Events = append(f.Events, ev)
	t.totalEvents++
}

// beginBatch records the begin of a command batch. A batch belongs to the open frame; a new
// frame is opened when none is open.
func (t *GraphicsCommandTrace) beginBatch(mode graphicsdriver.FlushMode) {
	if !t.enabled.Load() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.depth++
	if t.open < 0 {
		t.beginFrameLocked()
	}
	t.appendEventLocked(TraceEvent{
		Kind: TraceEventBarrier,
		Params: []TraceParam{
			{Key: "phase", Value: "begin"},
			{Key: "mode", Value: flushModeString(mode)},
		},
	})
}

// endBatch records the end of a command batch. The frame is completed when the batch completes
// the frame and no enclosing frame is open.
func (t *GraphicsCommandTrace) endBatch(mode graphicsdriver.FlushMode) {
	if !t.enabled.Load() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.open < 0 {
		return
	}
	t.appendEventLocked(TraceEvent{
		Kind: TraceEventBarrier,
		Params: []TraceParam{
			{Key: "phase", Value: "end"},
			{Key: "mode", Value: flushModeString(mode)},
		},
	})
	t.depth--
	if t.depth < 0 {
		t.depth = 0
	}
	if mode != graphicsdriver.FlushModeIntermediate && t.depth == 0 {
		t.closeFrameLocked()
	}
}

// recordCommand records a command that is about to be executed and returns its event kind.
// The returned kind is TraceEventCommand for an unknown command, and empty when recording is
// disabled.
func (t *GraphicsCommandTrace) recordCommand(c command) TraceEventKind {
	if !t.enabled.Load() {
		return ""
	}
	ev := summarizeCommand(c)
	t.mu.Lock()
	t.appendEventLocked(ev)
	t.mu.Unlock()
	return ev.Kind
}

// recordCommandError records the failure of a command that recordCommand has just recorded.
func (t *GraphicsCommandTrace) recordCommandError(kind TraceEventKind, err error) {
	if kind == "" {
		return
	}
	t.recordError("command "+string(kind), err)
}

// recordError records a graphics driver or command failure with its context.
func (t *GraphicsCommandTrace) recordError(where string, err error) {
	if !t.enabled.Load() {
		return
	}
	t.mu.Lock()
	t.appendEventLocked(TraceEvent{
		Kind:   TraceEventError,
		Params: []TraceParam{{Key: "at", Value: where}},
		Err:    err.Error(),
	})
	t.mu.Unlock()
}

// recordReset records a graphics device reset. A reset between frames forms its own frame.
func (t *GraphicsCommandTrace) recordReset(err error) {
	if !t.enabled.Load() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	own := t.open < 0
	if own {
		t.beginFrameLocked()
	}
	ev := TraceEvent{
		Kind:   TraceEventBarrier,
		Params: []TraceParam{{Key: "phase", Value: "reset"}},
	}
	if err != nil {
		ev.Err = err.Error()
	}
	t.appendEventLocked(ev)
	if own {
		t.closeFrameLocked()
	}
}

// flushModeString returns a stable name for a flush mode.
func flushModeString(mode graphicsdriver.FlushMode) string {
	switch mode {
	case graphicsdriver.FlushModeIntermediate:
		return "intermediate"
	case graphicsdriver.FlushModeEndFrame:
		return "end-frame"
	case graphicsdriver.FlushModePresent:
		return "present"
	}
	return "unknown"
}

// summarizeCommand extracts the event kind, the resource identifiers, and a redacted parameter
// summary from a command. The result holds only value data, so recording never retains the
// resources the command queue recycles after the flush.
func summarizeCommand(c command) TraceEvent {
	switch c := c.(type) {
	case *drawTrianglesCommand:
		srcs := make([]int, 0, graphics.ShaderSrcImageCount)
		for _, src := range c.srcs {
			if src == nil {
				srcs = append(srcs, -1)
				continue
			}
			srcs = append(srcs, src.id)
		}
		return TraceEvent{
			Kind:   TraceEventDrawTriangles,
			Image:  c.dst.id,
			Shader: c.shader.id,
			Srcs:   srcs,
			Params: []TraceParam{
				{Key: "indices", Value: strconv.Itoa(c.numIndices())},
				{Key: "dst-regions", Value: strconv.Itoa(len(c.dstRegions))},
				{Key: "blend", Value: blendSummary(c.blend)},
				{Key: "uniforms", Value: uniformSummary(c.uniforms)},
			},
		}
	case *writePixelsCommand:
		var regions []string
		var bytes int
		for _, a := range c.args {
			regions = append(regions, a.region.String())
			// Record only the size. The pixel contents are sensitive and are never traced.
			bytes += a.pixels.Len()
		}
		return TraceEvent{
			Kind:  TraceEventWritePixels,
			Image: c.dst.id,
			Params: []TraceParam{
				{Key: "regions", Value: strings.Join(regions, ";")},
				{Key: "bytes", Value: strconv.Itoa(bytes)},
			},
		}
	case *disposeImageCommand:
		return TraceEvent{
			Kind:  TraceEventDisposeImage,
			Image: c.target.id,
		}
	case *newImageCommand:
		params := []TraceParam{
			{Key: "width", Value: strconv.Itoa(c.width)},
			{Key: "height", Value: strconv.Itoa(c.height)},
			{Key: "screen", Value: strconv.FormatBool(c.screen)},
		}
		if c.attribute != "" {
			params = append(params, TraceParam{Key: "attribute", Value: c.attribute})
		}
		return TraceEvent{
			Kind:   TraceEventNewImage,
			Image:  c.result.id,
			Params: params,
		}
	case *newShaderCommand:
		params := []TraceParam{
			{Key: "uniforms", Value: strconv.Itoa(len(c.ir.UniformNames))},
		}
		if c.result.name != "" {
			params = append(params, TraceParam{Key: "name", Value: c.result.name})
		}
		return TraceEvent{
			Kind:   TraceEventNewShader,
			Shader: c.result.id,
			Params: params,
		}
	case *disposeShaderCommand:
		return TraceEvent{
			Kind:   TraceEventDisposeShader,
			Shader: c.target.id,
		}
	case *readPixelsAsyncCommand:
		var regions []string
		var bytes int
		for _, a := range c.req.args {
			regions = append(regions, a.Region.String())
			// Record only the size. The pixel contents are sensitive and are never traced.
			bytes += len(a.Pixels)
		}
		return TraceEvent{
			Kind:  TraceEventReadPixels,
			Image: c.img.id,
			Params: []TraceParam{
				{Key: "regions", Value: strings.Join(regions, ";")},
				{Key: "bytes", Value: strconv.Itoa(bytes)},
			},
		}
	default:
		return TraceEvent{
			Kind:   TraceEventCommand,
			Params: []TraceParam{{Key: "description", Value: c.String()}},
		}
	}
}

// blendSummary returns a stable short name for a blend.
func blendSummary(blend graphicsdriver.Blend) string {
	switch blend {
	case graphicsdriver.BlendSourceOver:
		return "source-over"
	case graphicsdriver.BlendClear:
		return "clear"
	case graphicsdriver.BlendCopy:
		return "copy"
	case graphicsdriver.BlendDestination:
		return "destination"
	case graphicsdriver.BlendSourceIn:
		return "source-in"
	case graphicsdriver.BlendDestinationIn:
		return "destination-in"
	case graphicsdriver.BlendSourceOut:
		return "source-out"
	case graphicsdriver.BlendDestinationOut:
		return "destination-out"
	case graphicsdriver.BlendSourceAtop:
		return "source-atop"
	case graphicsdriver.BlendDestinationAtop:
		return "destination-atop"
	case graphicsdriver.BlendXor:
		return "xor"
	case graphicsdriver.BlendLighter:
		return "lighter"
	}
	return "custom"
}

// uniformSummary returns a redacted summary of uniform values: their count and an FNV-1a
// digest. The values themselves are sensitive and are never traced.
func uniformSummary(uniforms []uint32) string {
	h := fnv.New32a()
	var b [4]byte
	for _, u := range uniforms {
		binary.LittleEndian.PutUint32(b[:], u)
		_, _ = h.Write(b[:])
	}
	return fmt.Sprintf("count=%d,digest=%08x", len(uniforms), h.Sum32())
}
