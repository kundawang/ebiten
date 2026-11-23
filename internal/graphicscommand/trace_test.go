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

package graphicscommand_test

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

// enableTrace resets and enables the process-wide trace for the duration of a test.
// A non-positive maxEvents means no event limit.
func enableTrace(t *testing.T, maxEvents int) *graphicscommand.GraphicsCommandTrace {
	t.Helper()
	tr := graphicscommand.DefaultTrace()
	tr.Reset()
	tr.SetMaxEvents(maxEvents)
	tr.SetBackend("")
	tr.SetEnabled(true)
	t.Cleanup(func() {
		tr.SetEnabled(false)
		tr.Reset()
		tr.SetMaxEvents(graphicscommand.DefaultMaxTraceEvents)
		tr.SetBackend("")
	})
	return tr
}

func allEvents(tr *graphicscommand.Trace) []graphicscommand.TraceEvent {
	var events []graphicscommand.TraceEvent
	for _, f := range tr.Frames {
		events = append(events, f.Events...)
	}
	return events
}

func eventKinds(events []graphicscommand.TraceEvent) []graphicscommand.TraceEventKind {
	kinds := make([]graphicscommand.TraceEventKind, len(events))
	for i, ev := range events {
		kinds[i] = ev.Kind
	}
	return kinds
}

// containsSubsequence reports whether kinds contains want as a subsequence.
func containsSubsequence(kinds, want []graphicscommand.TraceEventKind) bool {
	var n int
	for _, k := range kinds {
		if k == want[n] {
			n++
			if n == len(want) {
				return true
			}
		}
	}
	return false
}

func equalTraceEvent(a, b graphicscommand.TraceEvent) bool {
	return a.Seq == b.Seq &&
		a.Kind == b.Kind &&
		a.Image == b.Image &&
		a.Shader == b.Shader &&
		a.Err == b.Err &&
		slices.Equal(a.Srcs, b.Srcs) &&
		slices.Equal(a.Params, b.Params)
}

type noopCommand struct{}

func (*noopCommand) Exec(_ *graphicscommand.CommandQueueForTesting, _ graphicsdriver.Graphics, _ int) error {
	return nil
}

func (*noopCommand) NeedsSync() bool { return false }
func (*noopCommand) String() string  { return "noop" }

func TestTraceDisabledRecordsNothing(t *testing.T) {
	tr := graphicscommand.DefaultTrace()
	wasEnabled := tr.Enabled()
	tr.SetEnabled(false)
	defer tr.SetEnabled(wasEnabled)
	tr.Reset()

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	manager.EnqueueCommandForTesting(&noopCommand{})
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}
	if snap := tr.Snapshot(); len(snap.Frames) != 0 {
		t.Errorf("a disabled trace recorded %d frames, want 0", len(snap.Frames))
	}
}

// TestTraceCommandOrderAndPixels records a fixed command sequence and checks both the pixel
// output and the recorded command order.
func TestTraceCommandOrderAndPixels(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	const w, h = 16, 16
	driver := ui.Get().GraphicsDriverForTesting()
	dst := graphicscommand.NewImage(w, h, false, "")
	src := graphicscommand.NewImage(w, h, false, "")
	defer dst.Dispose()
	defer src.Dispose()

	// Fill src with opaque red and copy it to dst.
	src.WritePixels(graphics.NewManagedBytes(4*w*h, func(bs []byte) {
		for i := 0; i < len(bs); i += 4 {
			bs[i] = 0xff
			bs[i+1] = 0
			bs[i+2] = 0
			bs[i+3] = 0xff
		}
	}), image.Rect(0, 0, w, h))
	vs := quadVertices(w, h)
	is := graphics.QuadIndices()
	dr := image.Rect(0, 0, w, h)
	sr := image.Rect(0, 0, w, h)
	dst.DrawTriangles([graphics.ShaderSrcImageCount]*graphicscommand.Image{src}, vs, is, graphicsdriver.BlendCopy, dr, [graphics.ShaderSrcImageCount]image.Rectangle{sr}, nearestFilterShader, nil)

	pix := make([]byte, 4*w*h)
	if err := dst.ReadPixels(driver, []graphicsdriver.PixelsArgs{{Pixels: pix, Region: dr}}); err != nil {
		t.Fatal(err)
	}
	if err := graphicscommand.FlushCommands(driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}

	// The traced commands must have produced the expected pixels.
	for j := range h {
		for i := range w {
			idx := 4 * (i + w*j)
			got := color.RGBA{R: pix[idx], G: pix[idx+1], B: pix[idx+2], A: pix[idx+3]}
			if want := (color.RGBA{R: 0xff, A: 0xff}); got != want {
				t.Fatalf("dst.At(%d, %d): got %v, want %v", i, j, got, want)
			}
		}
	}

	snap := tr.Snapshot()
	if len(snap.Frames) == 0 {
		t.Fatal("no frames recorded")
	}
	events := allEvents(snap)

	// The sequence numbers must be stable and contiguous.
	for i, ev := range events {
		if ev.Seq != uint64(i) {
			t.Errorf("events[%d].Seq = %d, want %d", i, ev.Seq, i)
		}
	}

	// The fixed command sequence must appear in order.
	want := []graphicscommand.TraceEventKind{
		graphicscommand.TraceEventNewImage,
		graphicscommand.TraceEventNewImage,
		graphicscommand.TraceEventWritePixels,
		graphicscommand.TraceEventDrawTriangles,
		graphicscommand.TraceEventReadPixels,
	}
	if !containsSubsequence(eventKinds(events), want) {
		t.Errorf("command order %v does not contain %v", eventKinds(events), want)
	}

	// Correlate the resource identifiers: the draw must name the created images.
	var newImages []graphicscommand.TraceEvent
	var draw *graphicscommand.TraceEvent
	for i := range events {
		switch events[i].Kind {
		case graphicscommand.TraceEventNewImage:
			newImages = append(newImages, events[i])
		case graphicscommand.TraceEventDrawTriangles:
			draw = &events[i]
		}
	}
	if len(newImages) < 2 {
		t.Fatalf("new-image events = %d, want at least 2", len(newImages))
	}
	if draw == nil {
		t.Fatal("no draw-triangles event recorded")
	}
	dstID, srcID := newImages[0].Image, newImages[1].Image
	if draw.Image != dstID {
		t.Errorf("draw destination = %d, want %d", draw.Image, dstID)
	}
	if len(draw.Srcs) == 0 || draw.Srcs[0] != srcID {
		t.Errorf("draw sources = %v, want the first source %d", draw.Srcs, srcID)
	}
	if draw.Shader == 0 {
		t.Error("draw does not name its shader")
	}

	// The draw must summarize its uniforms without exposing their values.
	u, ok := draw.Param("uniforms")
	if !ok || !strings.HasPrefix(u, "count=") || !strings.Contains(u, ",digest=") {
		t.Errorf("uniform summary = %q, want a redacted count and digest", u)
	}
	if b, ok := draw.Param("blend"); !ok || b != "copy" {
		t.Errorf("blend = %q, want %q", b, "copy")
	}

	// The write-pixels event must summarize sizes only, never pixel contents.
	for i := range events {
		if events[i].Kind != graphicscommand.TraceEventWritePixels {
			continue
		}
		if b, ok := events[i].Param("bytes"); !ok || b != fmt.Sprint(4*w*h) {
			t.Errorf("write-pixels bytes = %q, want %d", b, 4*w*h)
		}
		for _, p := range events[i].Params {
			if len(p.Value) > 64 {
				t.Errorf("write-pixels param %q is suspiciously long (%d bytes); pixel contents must not be traced", p.Key, len(p.Value))
			}
		}
	}

	// The recorded resource lifecycle must be consistent.
	if err := snap.Verify(); err != nil {
		t.Errorf("Verify: %v", err)
	}
}

func TestTraceEmptyFrame(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	manager.EnqueueCommandForTesting(&noopCommand{})
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}
	// The queue is now empty: the next flush completes an empty frame.
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}

	snap := tr.Snapshot()
	var empty *graphicscommand.TraceFrame
	for i := range snap.Frames {
		f := &snap.Frames[i]
		allBarrier := len(f.Events) > 0
		for _, ev := range f.Events {
			if ev.Kind != graphicscommand.TraceEventBarrier {
				allBarrier = false
				break
			}
		}
		if allBarrier {
			empty = f
			break
		}
	}
	if empty == nil {
		t.Fatal("no empty frame recorded")
	}
	if len(empty.Events) != 2 {
		t.Errorf("empty frame events = %d, want 2 (begin and end barriers)", len(empty.Events))
	}
	if err := snap.Verify(); err != nil {
		t.Errorf("Verify: %v", err)
	}
}

func TestTraceTruncation(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	const maxEvents = 8
	tr := enableTrace(t, maxEvents)

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	for range 4 * maxEvents {
		manager.EnqueueCommandForTesting(&noopCommand{})
	}
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}

	snap := tr.Snapshot()
	var total int
	var truncated bool
	for _, f := range snap.Frames {
		total += len(f.Events)
		if f.Truncated {
			truncated = true
		}
	}
	if total > maxEvents {
		t.Errorf("recorded events = %d, want at most %d", total, maxEvents)
	}
	if !truncated {
		t.Error("no frame is marked as truncated")
	}
	// A truncated trace must still be exportable, replayable, and verifiable.
	var buf bytes.Buffer
	if err := snap.Encode(&buf); err != nil {
		t.Fatal(err)
	}
	if err := snap.Replay(func(graphicscommand.TraceEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := snap.Verify(); err != nil {
		t.Fatal(err)
	}
}

func TestTraceNestedAndDuplicateEndFrame(t *testing.T) {
	tr := enableTrace(t, 0)

	if err := tr.EndFrame(); err == nil {
		t.Error("EndFrame without BeginFrame must fail")
	}
	tr.BeginFrame()
	tr.BeginFrame() // nested
	if err := tr.EndFrame(); err != nil {
		t.Errorf("inner EndFrame: %v", err)
	}
	// The outer frame is still open: the snapshot contains it, but it has no events yet.
	if snap := tr.Snapshot(); len(snap.Frames) != 1 || len(snap.Frames[0].Events) != 0 {
		t.Errorf("after the inner EndFrame: frames = %d, events = %d; want 1 open frame without events",
			len(snap.Frames), len(snap.Frames[0].Events))
	}
	if err := tr.EndFrame(); err != nil {
		t.Errorf("outer EndFrame: %v", err)
	}
	if err := tr.EndFrame(); err == nil {
		t.Error("a duplicate EndFrame must fail")
	}
	if snap := tr.Snapshot(); len(snap.Frames) != 1 {
		t.Errorf("frames = %d, want 1", len(snap.Frames))
	}
}

func TestTraceExportJSONStability(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)
	tr.SetBackend("test-backend")

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	manager.EnqueueCommandForTesting(&noopCommand{})
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}

	var b1, b2 bytes.Buffer
	if err := tr.Export(&b1); err != nil {
		t.Fatal(err)
	}
	if err := tr.Export(&b2); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1.Bytes(), b2.Bytes()) {
		t.Error("two exports of the same trace differ")
	}
	if !strings.Contains(b1.String(), `"backend":"test-backend"`) {
		t.Error("the export does not record the backend")
	}

	// A decode/encode round-trip must be stable as well. Decode from a copy: decoding
	// consumes the reader.
	decoded, err := graphicscommand.DecodeTrace(bytes.NewReader(b1.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	var b3 bytes.Buffer
	if err := decoded.Encode(&b3); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1.Bytes(), b3.Bytes()) {
		t.Error("a decode/encode round-trip changed the trace")
	}
	if got, want := allEvents(decoded), allEvents(tr.Snapshot()); !slices.EqualFunc(got, want, equalTraceEvent) {
		t.Errorf("decoded events = %v, want %v", got, want)
	}
}

func TestTraceVersionMismatch(t *testing.T) {
	for _, version := range []int{0, -1, graphicscommand.TraceVersion + 1} {
		raw := fmt.Sprintf(`{"version":%d,"frames":[]}`, version)
		if _, err := graphicscommand.DecodeTrace(strings.NewReader(raw)); !errors.Is(err, graphicscommand.ErrTraceVersionMismatch) {
			t.Errorf("DecodeTrace(version %d): got %v, want ErrTraceVersionMismatch", version, err)
		}
		tr := &graphicscommand.Trace{Version: version}
		if err := tr.Replay(func(graphicscommand.TraceEvent) error { return nil }); !errors.Is(err, graphicscommand.ErrTraceVersionMismatch) {
			t.Errorf("Replay(version %d): got %v, want ErrTraceVersionMismatch", version, err)
		}
		if err := tr.Verify(); !errors.Is(err, graphicscommand.ErrTraceVersionMismatch) {
			t.Errorf("Verify(version %d): got %v, want ErrTraceVersionMismatch", version, err)
		}
	}
}

func TestTraceReplayIdempotent(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	for range 5 {
		manager.EnqueueCommandForTesting(&noopCommand{})
	}
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}
	snap := tr.Snapshot()

	collect := func() []graphicscommand.TraceEvent {
		var events []graphicscommand.TraceEvent
		if err := snap.Replay(func(ev graphicscommand.TraceEvent) error {
			events = append(events, ev)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return events
	}
	first, second := collect(), collect()
	if !slices.EqualFunc(first, second, equalTraceEvent) {
		t.Error("two replays of the same trace differ")
	}
	if len(first) != len(allEvents(snap)) {
		t.Errorf("replayed %d events, want %d", len(first), len(allEvents(snap)))
	}

	// A replay error must identify the frame and the event.
	errSentinel := errors.New("replay error for testing")
	want := first[3]
	err := snap.Replay(func(ev graphicscommand.TraceEvent) error {
		if ev.Seq == want.Seq {
			return errSentinel
		}
		return nil
	})
	var traceErr *graphicscommand.TraceError
	if !errors.As(err, &traceErr) {
		t.Fatalf("replay error = %v, want a TraceError", err)
	}
	if !errors.Is(err, errSentinel) {
		t.Errorf("replay error does not wrap the visit error: %v", err)
	}
	if traceErr.Seq != want.Seq {
		t.Errorf("replay error seq = %d, want %d", traceErr.Seq, want.Seq)
	}

	// Replaying must not modify the trace.
	var before, after bytes.Buffer
	if err := snap.Encode(&before); err != nil {
		t.Fatal(err)
	}
	collect()
	if err := snap.Encode(&after); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Error("replaying modified the trace")
	}
}

func traceEvent(seq uint64, kind graphicscommand.TraceEventKind, image, shader int, params ...graphicscommand.TraceParam) graphicscommand.TraceEvent {
	return graphicscommand.TraceEvent{
		Seq:    seq,
		Kind:   kind,
		Image:  image,
		Shader: shader,
		Params: params,
	}
}

func singleFrameTrace(events ...graphicscommand.TraceEvent) *graphicscommand.Trace {
	return &graphicscommand.Trace{
		Version: graphicscommand.TraceVersion,
		Frames: []graphicscommand.TraceFrame{
			{Index: 0, Events: events},
		},
	}
}

func TestTraceVerifyResourceLifecycle(t *testing.T) {
	tests := []struct {
		name    string
		events  []graphicscommand.TraceEvent
		wantErr string
	}{
		{
			name: "valid",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventNewShader, 0, 7),
				traceEvent(2, graphicscommand.TraceEventWritePixels, 1, 0),
				{
					Seq: 3, Kind: graphicscommand.TraceEventDrawTriangles,
					Image: 1, Shader: 7, Srcs: []int{1, -1, -1, -1},
				},
				traceEvent(4, graphicscommand.TraceEventDisposeImage, 1, 0),
				traceEvent(5, graphicscommand.TraceEventDisposeShader, 0, 7),
			},
		},
		{
			name: "use after dispose",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventDisposeImage, 1, 0),
				traceEvent(2, graphicscommand.TraceEventWritePixels, 1, 0),
			},
			wantErr: "image 1 is used after being disposed",
		},
		{
			name: "double dispose",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventDisposeImage, 1, 0),
				traceEvent(2, graphicscommand.TraceEventDisposeImage, 1, 0),
			},
			wantErr: "image 1 is used after being disposed",
		},
		{
			name: "missing source",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventNewImage, 2, 0),
				traceEvent(2, graphicscommand.TraceEventDisposeImage, 2, 0),
				{
					Seq: 3, Kind: graphicscommand.TraceEventDrawTriangles,
					Image: 1, Srcs: []int{2, -1, -1, -1},
				},
			},
			wantErr: "image 2 is used after being disposed",
		},
		{
			name: "device reset",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventBarrier, 0, 0, graphicscommand.TraceParam{Key: "phase", Value: "reset"}),
				traceEvent(2, graphicscommand.TraceEventWritePixels, 1, 0),
			},
			wantErr: "image 1 is used after a device reset",
		},
		{
			name: "duplicate creation",
			events: []graphicscommand.TraceEvent{
				traceEvent(0, graphicscommand.TraceEventNewImage, 1, 0),
				traceEvent(1, graphicscommand.TraceEventNewImage, 1, 0),
			},
			wantErr: "image 1 is created more than once",
		},
		{
			name: "resource predating the trace",
			events: []graphicscommand.TraceEvent{
				// Image 9 was created before the trace started.
				traceEvent(0, graphicscommand.TraceEventWritePixels, 9, 0),
				traceEvent(1, graphicscommand.TraceEventDisposeImage, 9, 0),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := singleFrameTrace(tc.events...).Verify()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Verify: %v", err)
				}
				return
			}
			var traceErr *graphicscommand.TraceError
			if !errors.As(err, &traceErr) {
				t.Fatalf("Verify error = %v, want a TraceError", err)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Verify error = %q, want it to mention %q", err, tc.wantErr)
			}
		})
	}
}

func TestTraceCommandAndDriverErrors(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	manager.EnqueueCommandForTesting(&failingCommand{})
	if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); !errors.Is(err, errFlushForTesting) {
		t.Fatalf("flush error = %v, want %v", err, errFlushForTesting)
	}

	events := allEvents(tr.Snapshot())
	var errEvent *graphicscommand.TraceEvent
	for i := range events {
		if events[i].Kind == graphicscommand.TraceEventError {
			errEvent = &events[i]
			break
		}
	}
	if errEvent == nil {
		t.Fatal("no error event recorded")
	}
	if !strings.Contains(errEvent.Err, errFlushForTesting.Error()) {
		t.Errorf("error event = %q, want it to contain %q", errEvent.Err, errFlushForTesting)
	}
	if at, ok := errEvent.Param("at"); !ok || !strings.HasPrefix(at, "command ") {
		t.Errorf("error event context = %q, want a command kind", at)
	}
}

type failingBeginDriver struct {
	frameDriver
}

func (*failingBeginDriver) Begin() error {
	return errFlushForTesting
}

func TestTraceDriverBeginError(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var q graphicscommand.CommandQueueForTesting
	q.Enqueue(&noopCommand{})
	var manager graphicscommand.CommandQueueManagerForTesting
	if err := q.Flush(&manager, &failingBeginDriver{}, graphicsdriver.FlushModePresent); !errors.Is(err, errFlushForTesting) {
		t.Fatalf("flush error = %v, want %v", err, errFlushForTesting)
	}

	events := allEvents(tr.Snapshot())
	var found bool
	for _, ev := range events {
		if ev.Kind == graphicscommand.TraceEventError && strings.Contains(ev.Err, errFlushForTesting.Error()) {
			if at, ok := ev.Param("at"); ok && at == "begin" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("no error event with context %q recorded in %v", "begin", eventKinds(events))
	}
}

type resetDriver struct {
	frameDriver
	resets   int
	resetErr error
}

func (g *resetDriver) Reset() error {
	g.resets++
	return g.resetErr
}

func TestTraceDeviceReset(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var manager graphicscommand.CommandQueueManagerForTesting
	driver := &resetDriver{resetErr: errFlushForTesting}
	manager.EnqueueCommandForTesting(&noopCommand{})
	if err := manager.FlushForTesting(driver, graphicsdriver.FlushModePresent); err != nil {
		t.Fatal(err)
	}
	if err := graphicscommand.ResetGraphicsDriverState(driver); !errors.Is(err, errFlushForTesting) {
		t.Fatalf("reset error = %v, want %v", err, errFlushForTesting)
	}
	if driver.resets != 1 {
		t.Fatalf("resets = %d, want 1", driver.resets)
	}

	events := allEvents(tr.Snapshot())
	var reset *graphicscommand.TraceEvent
	for i := range events {
		if events[i].Kind != graphicscommand.TraceEventBarrier {
			continue
		}
		if phase, ok := events[i].Param("phase"); ok && phase == "reset" {
			reset = &events[i]
			break
		}
	}
	if reset == nil {
		t.Fatal("no reset barrier recorded")
	}
	if !strings.Contains(reset.Err, errFlushForTesting.Error()) {
		t.Errorf("reset error = %q, want it to contain %q", reset.Err, errFlushForTesting)
	}
}

func TestTraceConcurrentAccess(t *testing.T) {
	graphicscommand.SetVsyncEnabled(true)
	tr := enableTrace(t, 0)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var buf bytes.Buffer
			for {
				select {
				case <-stop:
					return
				default:
				}
				_ = tr.Snapshot()
				buf.Reset()
				_ = tr.Export(&buf)
				tr.BeginFrame()
				_ = tr.EndFrame()
			}
		}()
	}

	var manager graphicscommand.CommandQueueManagerForTesting
	var driver frameDriver
	for range 64 {
		manager.EnqueueCommandForTesting(&noopCommand{})
		if err := manager.FlushForTesting(&driver, graphicsdriver.FlushModePresent); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()

	// Every recorded event must have a unique, contiguous sequence number.
	snap := tr.Snapshot()
	var seqs []uint64
	for _, ev := range allEvents(snap) {
		seqs = append(seqs, ev.Seq)
	}
	slices.Sort(seqs)
	for i, s := range seqs {
		if s != uint64(i) {
			t.Fatalf("sequence numbers are not contiguous: seqs[%d] = %d", i, s)
		}
	}
}
