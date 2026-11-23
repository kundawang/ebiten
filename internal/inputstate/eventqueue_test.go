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

package inputstate

import (
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

func mustQueue(t *testing.T, capacity int) *InputEventQueue {
	t.Helper()
	q, err := NewInputEventQueue(capacity)
	if err != nil {
		t.Fatalf("NewInputEventQueue(%d): %v", capacity, err)
	}
	return q
}

func drain(q *InputEventQueue) []InputEvent {
	var out []InputEvent
	for {
		ev, ok := q.Consume()
		if !ok {
			return out
		}
		out = append(out, ev)
	}
}

func eventKinds(events []InputEvent) []EventKind {
	kinds := make([]EventKind, len(events))
	for i, ev := range events {
		kinds[i] = ev.Kind
	}
	return kinds
}

func TestInputEventQueueFIFO(t *testing.T) {
	q := mustQueue(t, 8)
	for i := 0; i < 4; i++ {
		if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.Key(i)}); err != nil {
			t.Fatal(err)
		}
	}

	// Peek must not consume.
	ev, ok := q.Peek()
	if !ok {
		t.Fatal("Peek on a non-empty queue returned false")
	}
	if ev.Key != ui.Key(0) {
		t.Fatalf("Peek: got key %v, want %v", ev.Key, ui.Key(0))
	}
	if got := q.Stats().Len; got != 4 {
		t.Fatalf("Len after Peek: got %d, want 4", got)
	}

	events := drain(q)
	if len(events) != 4 {
		t.Fatalf("drained %d events, want 4", len(events))
	}
	for i, ev := range events {
		if ev.Key != ui.Key(i) {
			t.Errorf("event %d: got key %v, want %v", i, ev.Key, ui.Key(i))
		}
		if ev.Seq != uint64(i) {
			t.Errorf("event %d: got seq %d, want %d", i, ev.Seq, i)
		}
	}
	if _, ok := q.Consume(); ok {
		t.Error("Consume on a drained queue returned true")
	}
}

func TestInputEventQueueEmpty(t *testing.T) {
	q := mustQueue(t, 4)
	if _, ok := q.Peek(); ok {
		t.Error("Peek on an empty queue returned true")
	}
	if _, ok := q.Consume(); ok {
		t.Error("Consume on an empty queue returned true")
	}
	if got := q.ConsumeFrame(1); got != nil {
		t.Errorf("ConsumeFrame on an empty queue: got %v, want nil", got)
	}
	if got := q.PeekFrame(1); got != nil {
		t.Errorf("PeekFrame on an empty queue: got %v, want nil", got)
	}
	if s := q.Stats(); s.Len != 0 || s.Enqueued != 0 || s.Dropped != 0 {
		t.Errorf("Stats on an empty queue: got %+v", s)
	}
}

func TestInputEventQueueZeroCapacity(t *testing.T) {
	q := mustQueue(t, 0)
	for i := 0; i < 3; i++ {
		if err := q.Push(InputEvent{Kind: EventKindKeyPress}); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := q.Peek(); ok {
		t.Error("Peek on a zero-capacity queue returned true")
	}
	s := q.Stats()
	if s.Len != 0 || s.Enqueued != 0 || s.Dropped != 3 {
		t.Errorf("Stats: got %+v, want Len=0 Enqueued=0 Dropped=3", s)
	}
}

func TestInputEventQueueNegativeCapacity(t *testing.T) {
	if _, err := NewInputEventQueue(-1); err == nil {
		t.Error("NewInputEventQueue(-1) did not return an error")
	}
}

func TestInputEventQueueOverflow(t *testing.T) {
	q := mustQueue(t, 3)
	for i := 0; i < 5; i++ {
		if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.Key(i)}); err != nil {
			t.Fatal(err)
		}
	}
	s := q.Stats()
	if s.Len != 3 || s.Enqueued != 5 || s.Dropped != 2 {
		t.Fatalf("Stats: got %+v, want Len=3 Enqueued=5 Dropped=2", s)
	}
	// The oldest events must have been evicted.
	events := drain(q)
	for i, ev := range events {
		if want := ui.Key(i + 2); ev.Key != want {
			t.Errorf("event %d: got key %v, want %v", i, ev.Key, want)
		}
	}
	// Overflow statistics survive a reset.
	q.Reset()
	if got := q.Stats().Dropped; got != 2 {
		t.Errorf("Dropped after Reset: got %d, want 2", got)
	}
	if got := q.Stats().Len; got != 0 {
		t.Errorf("Len after Reset: got %d, want 0", got)
	}
}

func TestInputEventQueueFrames(t *testing.T) {
	q := mustQueue(t, 8)
	frames := []int64{1, 2, 2, 3, 2}
	for i, f := range frames {
		if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.Key(i), Frame: f}); err != nil {
			t.Fatal(err)
		}
	}

	peeked := q.PeekFrame(2)
	if len(peeked) != 3 {
		t.Fatalf("PeekFrame(2): got %d events, want 3", len(peeked))
	}
	for _, ev := range peeked {
		if ev.Frame != 2 {
			t.Errorf("PeekFrame(2) returned an event of frame %d", ev.Frame)
		}
	}
	// PeekFrame must not consume.
	if got := q.Stats().Len; got != 5 {
		t.Fatalf("Len after PeekFrame: got %d, want 5", got)
	}

	consumed := q.ConsumeFrame(2)
	if !reflect.DeepEqual(eventKinds(peeked), eventKinds(consumed)) {
		t.Errorf("ConsumeFrame(2) = %v, want the peeked %v", eventKinds(consumed), eventKinds(peeked))
	}
	// The consumed events must keep their relative order.
	for i, ev := range consumed {
		if want := ui.Key([]int{1, 2, 4}[i]); ev.Key != want {
			t.Errorf("consumed event %d: got key %v, want %v", i, ev.Key, want)
		}
	}
	if got := q.ConsumeFrame(2); got != nil {
		t.Errorf("second ConsumeFrame(2): got %v, want nil", got)
	}

	rest := drain(q)
	if len(rest) != 2 || rest[0].Frame != 1 || rest[1].Frame != 3 {
		t.Errorf("remaining events: got %v", rest)
	}
}

func TestInputEventQueueClose(t *testing.T) {
	q := mustQueue(t, 4)
	if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.KeyA}); err != nil {
		t.Fatal(err)
	}
	q.Close()

	if err := q.Push(InputEvent{Kind: EventKindKeyPress}); !errors.Is(err, ErrQueueClosed) {
		t.Errorf("Push after Close: got %v, want %v", err, ErrQueueClosed)
	}
	if err := q.Inject(InputEvent{Kind: EventKindKeyPress}); !errors.Is(err, ErrQueueClosed) {
		t.Errorf("Inject after Close: got %v, want %v", err, ErrQueueClosed)
	}
	// A closed queue still drains.
	ev, ok := q.Consume()
	if !ok || ev.Key != ui.KeyA {
		t.Errorf("Consume after Close: got (%v, %v)", ev, ok)
	}
	if _, ok := q.Consume(); ok {
		t.Error("Consume on a drained closed queue returned true")
	}
}

func TestInputEventQueueInject(t *testing.T) {
	q := mustQueue(t, 8)
	if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.KeyA, Frame: 1}); err != nil {
		t.Fatal(err)
	}
	// A replay can inject events whose frames go backwards; the sequence
	// numbers must still order them after the live events.
	replay := []InputEvent{
		{Kind: EventKindKeyPress, Key: ui.KeyB, Frame: 0},
		{Kind: EventKindKeyRelease, Key: ui.KeyB, Frame: 0},
	}
	if err := q.Inject(replay...); err != nil {
		t.Fatal(err)
	}
	events := drain(q)
	wantKeys := []ui.Key{ui.KeyA, ui.KeyB, ui.KeyB}
	wantKinds := []EventKind{EventKindKeyPress, EventKindKeyPress, EventKindKeyRelease}
	for i, ev := range events {
		if ev.Key != wantKeys[i] || ev.Kind != wantKinds[i] {
			t.Errorf("event %d: got (%v, %v), want (%v, %v)", i, ev.Kind, ev.Key, wantKinds[i], wantKeys[i])
		}
		if ev.Seq != uint64(i) {
			t.Errorf("event %d: got seq %d, want %d", i, ev.Seq, i)
		}
	}
}

func TestInputEventQueueConcurrent(t *testing.T) {
	const (
		producers       = 4
		eventsPerWriter = 250
	)
	q := mustQueue(t, producers*eventsPerWriter)

	var wg sync.WaitGroup
	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < eventsPerWriter; i++ {
				if err := q.Push(InputEvent{Kind: EventKindKeyPress, Key: ui.Key(p)}); err != nil {
					t.Error(err)
					return
				}
			}
		}(p)
	}

	consumed := make(chan int, 1)
	go func() {
		n := 0
		for n < producers*eventsPerWriter {
			if _, ok := q.Consume(); ok {
				n++
			}
		}
		consumed <- n
	}()

	wg.Wait()
	if got := <-consumed; got != producers*eventsPerWriter {
		t.Errorf("consumed %d events, want %d", got, producers*eventsPerWriter)
	}
	if s := q.Stats(); s.Dropped != 0 || s.Len != 0 {
		t.Errorf("Stats: got %+v, want Len=0 Dropped=0", s)
	}
}

// pressedState returns a snapshot with the given keys and buttons pressed at the
// given tick, mirroring how the ui package stamps events. The press and release
// times of a ui.InputState are cumulative: they keep their values while a key is
// held and after it is released, so fixtures evolve one state instead of
// rebuilding it per frame.
func pressedState(tick int64, keys []ui.Key, buttons []ui.MouseButton) *ui.InputState {
	s := &ui.InputState{}
	tm := ui.NewInputTimeFromTick(tick)
	for _, k := range keys {
		s.KeyPressedTimes[k] = tm
	}
	for _, b := range buttons {
		s.MouseButtonPressedTimes[b] = tm
	}
	return s
}

// releaseAll mirrors what the ui package does to the accumulated state when the
// window loses focus: every pressed key and button is released and all touches
// end.
func releaseAll(s *ui.InputState, tick int64) {
	tm := ui.NewInputTimeFromTick(tick)
	for k := ui.Key(0); k <= ui.KeyMax; k++ {
		if s.KeyPressedTimes[k] > s.KeyReleasedTimes[k] {
			s.KeyReleasedTimes[k] = tm
		}
	}
	for b := ui.MouseButton(0); b <= ui.MouseButtonMax; b++ {
		if s.MouseButtonPressedTimes[b] > s.MouseButtonReleasedTimes[b] {
			s.MouseButtonReleasedTimes[b] = tm
		}
	}
	s.Touches = s.Touches[:0]
}

func newDeterministicRecorder(t *testing.T, capacity int) (*EventRecorder, *InputEventQueue, *time.Duration) {
	t.Helper()
	q := mustQueue(t, capacity)
	r := NewEventRecorder(q)
	clock := new(time.Duration)
	r.SetClock(func() time.Duration { return *clock })
	return r, q, clock
}

func TestEventRecorderDeterministicSequence(t *testing.T) {
	r, q, clock := newDeterministicRecorder(t, 64)

	// Frame 1: key A goes down, the cursor moves, and a character is typed.
	*clock = 10 * time.Millisecond
	s := pressedState(1, []ui.Key{ui.KeyA}, nil)
	s.CursorX, s.CursorY = 10, 20
	s.Runes = []rune{'a'}
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	s1 := *s

	// Frame 2: key A goes up, mouse button 0 goes down, the wheel turns, and a
	// touch starts.
	*clock = 20 * time.Millisecond
	s.KeyReleasedTimes[ui.KeyA] = ui.NewInputTimeFromTick(2)
	s.MouseButtonPressedTimes[ui.MouseButton0] = ui.NewInputTimeFromTick(2)
	s.WheelY = 1
	s.Touches = []ui.Touch{{ID: 0, X: 5, Y: 6}}
	s.Runes = s.Runes[:0]
	if err := r.Observe(s, 2); err != nil {
		t.Fatal(err)
	}
	s2 := *s

	// Frame 3: the touch moves. The wheel delta is consumed per frame.
	*clock = 30 * time.Millisecond
	s.WheelY = 0
	s.Touches = []ui.Touch{{ID: 0, X: 7, Y: 8}}
	if err := r.Observe(s, 3); err != nil {
		t.Fatal(err)
	}

	// Frame 4: the touch ends.
	s.Touches = s.Touches[:0]
	if err := r.Observe(s, 4); err != nil {
		t.Fatal(err)
	}

	type wantEvent struct {
		frame int64
		kind  EventKind
		time  time.Duration
	}
	want := []wantEvent{
		{1, EventKindKeyPress, 10 * time.Millisecond},
		{1, EventKindCursorMove, 10 * time.Millisecond},
		{1, EventKindRune, 10 * time.Millisecond},
		{2, EventKindKeyRelease, 20 * time.Millisecond},
		{2, EventKindMouseButtonPress, 20 * time.Millisecond},
		{2, EventKindWheel, 20 * time.Millisecond},
		{2, EventKindTouchPress, 20 * time.Millisecond},
		{3, EventKindTouchMove, 30 * time.Millisecond},
		{4, EventKindTouchRelease, 30 * time.Millisecond},
	}
	events := drain(q)
	if len(events) != len(want) {
		t.Fatalf("got %d events (%v), want %d", len(events), eventKinds(events), len(want))
	}
	for i, ev := range events {
		if ev.Kind != want[i].kind || ev.Frame != want[i].frame {
			t.Errorf("event %d: got (%v, frame %d), want (%v, frame %d)", i, ev.Kind, ev.Frame, want[i].kind, want[i].frame)
		}
		if ev.Seq != uint64(i) {
			t.Errorf("event %d: got seq %d, want %d", i, ev.Seq, i)
		}
		if ev.Time != want[i].time {
			t.Errorf("event %d: got time %v, want %v", i, ev.Time, want[i].time)
		}
	}

	// Cross-check the recorded events against the live state queries: the
	// snapshots must report pressed exactly what the event stream recorded.
	if !s1.IsKeyPressed(ui.KeyA, 1) {
		t.Error("s1: KeyA should be pressed at tick 1")
	}
	if s2.IsKeyPressed(ui.KeyA, 2) {
		t.Error("s2: KeyA should not be pressed at tick 2")
	}
	if !s2.IsMouseButtonPressed(ui.MouseButton0, 2) {
		t.Error("s2: MouseButton0 should be pressed at tick 2")
	}
	// The events must carry the data of the state they were derived from.
	if events[0].Key != ui.KeyA || events[0].Device != InputDeviceKeyboard {
		t.Errorf("event 0: got key %v device %v", events[0].Key, events[0].Device)
	}
	if events[1].X != 10 || events[1].Y != 20 {
		t.Errorf("event 1: got cursor (%v, %v), want (10, 20)", events[1].X, events[1].Y)
	}
	if events[2].Rune != 'a' || events[2].Device != InputDeviceText {
		t.Errorf("event 2: got rune %q device %v", events[2].Rune, events[2].Device)
	}
	if events[5].WheelY != 1 {
		t.Errorf("event 5: got wheel Y %v, want 1", events[5].WheelY)
	}
	if events[6].TouchID != 0 || events[6].X != 5 || events[6].Y != 6 {
		t.Errorf("event 6: got touch %v at (%v, %v)", events[6].TouchID, events[6].X, events[6].Y)
	}
	if events[8].X != 7 || events[8].Y != 8 {
		t.Errorf("event 8: got touch release at (%v, %v), want (7, 8)", events[8].X, events[8].Y)
	}
}

func TestEventRecorderRepeatedSampling(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 16)

	s := pressedState(1, []ui.Key{ui.KeyA}, nil)
	s.Runes = []rune{'x'}
	s.WheelY = 1
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	n := q.Stats().Len
	if n == 0 {
		t.Fatal("no events recorded")
	}
	// Observing the same frame again must not duplicate events, even though
	// the snapshot still holds the frame's runes and wheel deltas.
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	if got := q.Stats().Len; got != n {
		t.Errorf("Len after re-observing frame 1: got %d, want %d", got, n)
	}

	// A new frame with unchanged key state produces no key events.
	s.Runes = s.Runes[:0]
	s.WheelY = 0
	if err := r.Observe(s, 2); err != nil {
		t.Fatal(err)
	}
	for _, ev := range drain(q) {
		if ev.Frame == 2 {
			t.Errorf("unexpected event in frame 2: %v", ev.Kind)
		}
	}
}

func TestEventRecorderBlurRecovery(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 32)

	// Frame 1: two keys, a button, and a touch are down.
	s := pressedState(1, []ui.Key{ui.KeyA, ui.KeyB}, []ui.MouseButton{ui.MouseButton0})
	s.Touches = []ui.Touch{{ID: 0, X: 1, Y: 2}}
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	drain(q)

	// Frame 2: the window loses focus; everything is released at once.
	releaseAll(s, 2)
	if err := r.Observe(s, 2); err != nil {
		t.Fatal(err)
	}

	events := drain(q)
	want := []EventKind{
		EventKindKeyRelease, EventKindKeyRelease,
		EventKindMouseButtonRelease,
		EventKindTouchRelease,
	}
	if !reflect.DeepEqual(eventKinds(events), want) {
		t.Fatalf("blur events: got %v, want %v", eventKinds(events), want)
	}
	for _, ev := range events {
		if ev.Frame != 2 {
			t.Errorf("blur event %v: got frame %d, want 2", ev.Kind, ev.Frame)
		}
	}
	// The blur release events must match the live state: nothing is pressed.
	if s.IsKeyPressed(ui.KeyA, 2) || s.IsKeyPressed(ui.KeyB, 2) || s.IsMouseButtonPressed(ui.MouseButton0, 2) {
		t.Error("the state after the blur should report nothing pressed")
	}

	// Frame 3: the window is focused again and a key goes down; the recorder
	// must report a fresh press.
	s.KeyPressedTimes[ui.KeyA] = ui.NewInputTimeFromTick(3)
	if err := r.Observe(s, 3); err != nil {
		t.Fatal(err)
	}
	events = drain(q)
	if len(events) != 1 || events[0].Kind != EventKindKeyPress || events[0].Key != ui.KeyA {
		t.Fatalf("recovery events: got %v", events)
	}
}

func TestEventRecorderReset(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 16)

	s := pressedState(1, []ui.Key{ui.KeyA}, nil)
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	drain(q)

	r.Reset()
	// After a reset the same state is reported as fresh: a new press event.
	if err := r.Observe(s, 2); err != nil {
		t.Fatal(err)
	}
	events := drain(q)
	if len(events) != 1 || events[0].Kind != EventKindKeyPress || events[0].Key != ui.KeyA {
		t.Fatalf("events after reset: got %v", events)
	}
}

func TestEventRecorderBackwardsTime(t *testing.T) {
	r, q, clock := newDeterministicRecorder(t, 16)

	*clock = 100 * time.Millisecond
	s := pressedState(1, []ui.Key{ui.KeyA}, nil)
	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	// The clock and the frame number go backwards, as when a replay restarts.
	// The events must still be recorded, in order, with monotonic timestamps.
	*clock = 50 * time.Millisecond
	s.KeyPressedTimes[ui.KeyB] = ui.NewInputTimeFromTick(2)
	if err := r.Observe(s, 0); err != nil {
		t.Fatal(err)
	}

	events := drain(q)
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
	if events[0].Key != ui.KeyA || events[1].Key != ui.KeyB {
		t.Errorf("got keys %v, %v", events[0].Key, events[1].Key)
	}
	if events[1].Time < events[0].Time {
		t.Errorf("event times went backwards: %v then %v", events[0].Time, events[1].Time)
	}
	if events[1].Seq <= events[0].Seq {
		t.Errorf("sequence numbers went backwards: %d then %d", events[0].Seq, events[1].Seq)
	}
}

func TestEventRecorderInvalidRune(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 16)

	s := &ui.InputState{}
	s.Runes = []rune{'a', 0x110000, 0xD800, 'b'}
	err := r.Observe(s, 1)
	if err == nil {
		t.Fatal("Observe with invalid runes did not return an error")
	}
	// The valid runes must still be recorded, the invalid ones skipped.
	events := drain(q)
	if len(events) != 2 || events[0].Rune != 'a' || events[1].Rune != 'b' {
		t.Fatalf("got events %v, want runes 'a' and 'b'", events)
	}
}

func TestEventRecorderDoesNotModifySnapshot(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 16)

	s := pressedState(1, []ui.Key{ui.KeyA}, []ui.MouseButton{ui.MouseButton0})
	s.CursorX, s.CursorY = 3, 4
	s.Touches = []ui.Touch{{ID: 0, X: 1, Y: 2}}
	s.Runes = []rune{'a'}
	orig := *s
	orig.Touches = append([]ui.Touch(nil), s.Touches...)
	orig.Runes = append([]rune(nil), s.Runes...)

	if err := r.Observe(s, 1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*s, orig) {
		t.Error("Observe modified the snapshot")
	}
	drain(q)

	// Mutating the observed snapshot afterwards must not corrupt the
	// recorder's next diff: observing the original state at a new frame
	// produces no state events, only the per-frame accumulators.
	s.Touches[0].X = 99
	s.Runes[0] = 'z'
	s.WheelY = 1
	if err := r.Observe(&orig, 2); err != nil {
		t.Fatal(err)
	}
	for _, ev := range drain(q) {
		switch ev.Kind {
		case EventKindKeyPress, EventKindKeyRelease,
			EventKindMouseButtonPress, EventKindMouseButtonRelease,
			EventKindCursorMove,
			EventKindTouchPress, EventKindTouchMove, EventKindTouchRelease:
			t.Errorf("unexpected state event from a corrupted diff: %v", ev.Kind)
		}
	}
}

func TestEventRecorderClosedQueue(t *testing.T) {
	r, q, _ := newDeterministicRecorder(t, 16)
	q.Close()
	if err := r.Observe(pressedState(1, []ui.Key{ui.KeyA}, nil), 1); !errors.Is(err, ErrQueueClosed) {
		t.Errorf("Observe on a closed queue: got %v, want %v", err, ErrQueueClosed)
	}
}

func TestSnapshotIsolation(t *testing.T) {
	// Reset the global state to not interfere with other tests.
	state := Get()
	state.Update(func(s *ui.InputState) {
		*s = ui.InputState{}
	})

	state.Update(func(s *ui.InputState) {
		s.KeyPressedTimes[ui.KeyA] = ui.NewInputTimeFromTick(1)
		s.Runes = []rune{'a'}
		s.Touches = []ui.Touch{{ID: 0, X: 1, Y: 2}}
		s.WheelY = 1
	})

	s1 := state.Snapshot()
	s2 := state.Snapshot()
	if !reflect.DeepEqual(s1, s2) {
		t.Error("two Snapshots differ")
	}

	// Mutating the snapshot must not affect the internal state.
	s1.Runes[0] = 'z'
	s1.Touches[0].X = 99
	s1.WheelY = 5
	s3 := state.Snapshot()
	if s3.Runes[0] != 'a' || s3.Touches[0].X != 1 || s3.WheelY != 1 {
		t.Error("mutating a Snapshot leaked into the internal state")
	}

	// Reading a snapshot must not reset delta members or affect the Is*
	// queries: the state is still there for the regular consumers.
	if !state.IsKeyPressed(ui.KeyA) {
		t.Error("IsKeyPressed(KeyA) after Snapshot: got false, want true")
	}
	if got := state.AppendInputChars(nil); !reflect.DeepEqual(got, []rune{'a'}) {
		t.Errorf("AppendInputChars after Snapshot: got %q, want %q", got, []rune{'a'})
	}

	// A recorder can consume the snapshot without disturbing the live state.
	q := mustQueue(t, 8)
	r := NewEventRecorder(q)
	if err := r.Observe(&s3, 1); err != nil {
		t.Fatal(err)
	}
	if !state.IsKeyPressed(ui.KeyA) {
		t.Error("IsKeyPressed(KeyA) after Observe: got false, want true")
	}
	// Key press, wheel, touch press, and rune: the cursor did not move.
	if got := q.Stats().Len; got != 4 {
		t.Errorf("recorded %d events, want 4", got)
	}
}
