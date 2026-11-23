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
	"fmt"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

// EventKind identifies the type of a recorded input event.
type EventKind uint8

const (
	EventKindKeyPress EventKind = iota
	EventKindKeyRelease
	EventKindMouseButtonPress
	EventKindMouseButtonRelease
	EventKindCursorMove
	EventKindWheel
	EventKindTouchPress
	EventKindTouchMove
	EventKindTouchRelease
	EventKindRune
)

// String returns a short description of the event kind.
func (k EventKind) String() string {
	switch k {
	case EventKindKeyPress:
		return "KeyPress"
	case EventKindKeyRelease:
		return "KeyRelease"
	case EventKindMouseButtonPress:
		return "MouseButtonPress"
	case EventKindMouseButtonRelease:
		return "MouseButtonRelease"
	case EventKindCursorMove:
		return "CursorMove"
	case EventKindWheel:
		return "Wheel"
	case EventKindTouchPress:
		return "TouchPress"
	case EventKindTouchMove:
		return "TouchMove"
	case EventKindTouchRelease:
		return "TouchRelease"
	case EventKindRune:
		return "Rune"
	}
	return fmt.Sprintf("EventKind(%d)", uint8(k))
}

// InputDevice identifies the category of device an input event came from.
type InputDevice uint8

const (
	InputDeviceKeyboard InputDevice = iota
	InputDeviceMouse
	InputDeviceTouch
	// InputDeviceText is the device for character input events, which do not
	// necessarily correspond to physical key presses (e.g. IME compositions).
	InputDeviceText
)

// InputEvent is a single input event recorded for frame-accurate consumption by
// editors and replay systems.
//
// Seq is a monotonically increasing sequence number assigned by the queue when the
// event is pushed. It defines the total order of events even when Frame or Time go
// backwards, which can happen with replay injection or clock adjustments. Frame and
// Time are advisory timestamps supplied by the recorder.
type InputEvent struct {
	Seq    uint64
	Frame  int64
	Time   time.Duration
	Kind   EventKind
	Device InputDevice

	Key     ui.Key
	Button  ui.MouseButton
	TouchID ui.TouchID
	X, Y    float64
	WheelX  float64
	WheelY  float64
	Rune    rune
}

// ErrQueueClosed is returned by push operations on a closed InputEventQueue.
var ErrQueueClosed = errors.New("inputstate: input event queue is closed")

// QueueStats reports cumulative statistics of an InputEventQueue.
type QueueStats struct {
	// Len is the number of events currently pending in the queue.
	Len int
	// Capacity is the maximum number of events the queue can hold.
	Capacity int
	// Enqueued is the total number of events accepted into the queue,
	// including events that were later evicted or consumed.
	Enqueued uint64
	// Dropped is the total number of events lost to overflow: events evicted
	// because the queue was full, or rejected because the capacity is zero.
	Dropped uint64
}

// InputEventQueue is a bounded, goroutine-safe FIFO queue of input events.
//
// The queue never blocks: when it is full, pushing evicts the oldest event and
// counts the eviction in the queue's overflow statistics. A queue with zero
// capacity accepts no events and counts every push as dropped. After Close, push
// operations return ErrQueueClosed while Peek, Consume, and ConsumeFrame keep
// draining the events that remain.
type InputEventQueue struct {
	m      sync.Mutex
	events []InputEvent // ring buffer
	head   int          // index of the oldest event
	length int

	nextSeq  uint64
	enqueued uint64
	dropped  uint64
	closed   bool
}

// NewInputEventQueue returns a queue that holds at most capacity events.
// A zero capacity is valid and drops every pushed event. A negative capacity is
// an error.
func NewInputEventQueue(capacity int) (*InputEventQueue, error) {
	if capacity < 0 {
		return nil, fmt.Errorf("inputstate: input event queue capacity must not be negative: %d", capacity)
	}
	return &InputEventQueue{
		events: make([]InputEvent, capacity),
	}, nil
}

// Push appends ev to the queue, assigning it the next sequence number. The Seq
// field of ev is overwritten.
//
// Push never blocks. If the queue is full, the oldest event is evicted and counted
// as dropped. If the queue has zero capacity, ev is dropped. Push returns
// ErrQueueClosed after the queue was closed.
func (q *InputEventQueue) Push(ev InputEvent) error {
	q.m.Lock()
	defer q.m.Unlock()
	return q.pushLocked(ev)
}

func (q *InputEventQueue) pushLocked(ev InputEvent) error {
	if q.closed {
		return ErrQueueClosed
	}
	ev.Seq = q.nextSeq
	q.nextSeq++
	if len(q.events) == 0 {
		q.dropped++
		return nil
	}
	if q.length == len(q.events) {
		// Evict the oldest event.
		q.head = (q.head + 1) % len(q.events)
		q.length--
		q.dropped++
	}
	q.events[(q.head+q.length)%len(q.events)] = ev
	q.length++
	q.enqueued++
	return nil
}

// Inject appends a batch of events, e.g. from a recorded replay, preserving their
// order. Each event is assigned a fresh sequence number, so injected events are
// ordered after every event pushed before them. Inject follows the same overflow
// policy as Push and never blocks. If the queue is closed, no event is injected
// and ErrQueueClosed is returned.
func (q *InputEventQueue) Inject(events ...InputEvent) error {
	q.m.Lock()
	defer q.m.Unlock()
	if q.closed {
		return ErrQueueClosed
	}
	for _, ev := range events {
		if err := q.pushLocked(ev); err != nil {
			return err
		}
	}
	return nil
}

// Peek returns the oldest pending event without removing it. The second return
// value reports whether an event was pending.
func (q *InputEventQueue) Peek() (InputEvent, bool) {
	q.m.Lock()
	defer q.m.Unlock()
	if q.length == 0 {
		return InputEvent{}, false
	}
	return q.events[q.head], true
}

// Consume removes and returns the oldest pending event. The second return value
// reports whether an event was pending.
func (q *InputEventQueue) Consume() (InputEvent, bool) {
	q.m.Lock()
	defer q.m.Unlock()
	if q.length == 0 {
		return InputEvent{}, false
	}
	ev := q.events[q.head]
	q.events[q.head] = InputEvent{}
	q.head = (q.head + 1) % len(q.events)
	q.length--
	return ev, true
}

// PeekFrame returns copies of the pending events recorded for frame, in order,
// without removing them.
func (q *InputEventQueue) PeekFrame(frame int64) []InputEvent {
	q.m.Lock()
	defer q.m.Unlock()
	var out []InputEvent
	for i := 0; i < q.length; i++ {
		if ev := q.events[(q.head+i)%len(q.events)]; ev.Frame == frame {
			out = append(out, ev)
		}
	}
	return out
}

// ConsumeFrame removes and returns the pending events recorded for frame, in
// order. Events of other frames stay pending. It returns nil when no event
// matches.
func (q *InputEventQueue) ConsumeFrame(frame int64) []InputEvent {
	q.m.Lock()
	defer q.m.Unlock()
	if q.length == 0 {
		return nil
	}
	var out []InputEvent
	kept := make([]InputEvent, 0, q.length)
	for i := 0; i < q.length; i++ {
		ev := q.events[(q.head+i)%len(q.events)]
		if ev.Frame == frame {
			out = append(out, ev)
		} else {
			kept = append(kept, ev)
		}
	}
	copy(q.events, kept)
	q.head = 0
	q.length = len(kept)
	return out
}

// Stats returns the queue's current and cumulative statistics.
func (q *InputEventQueue) Stats() QueueStats {
	q.m.Lock()
	defer q.m.Unlock()
	return QueueStats{
		Len:      q.length,
		Capacity: len(q.events),
		Enqueued: q.enqueued,
		Dropped:  q.dropped,
	}
}

// Reset drops all pending events. Cumulative statistics are kept, so overflow
// accounting survives a reset.
func (q *InputEventQueue) Reset() {
	q.m.Lock()
	defer q.m.Unlock()
	for i := 0; i < q.length; i++ {
		q.events[(q.head+i)%len(q.events)] = InputEvent{}
	}
	q.head = 0
	q.length = 0
}

// Close shuts the queue down, e.g. because the input device went away. Pushing
// after Close returns ErrQueueClosed; draining operations keep working.
func (q *InputEventQueue) Close() {
	q.m.Lock()
	defer q.m.Unlock()
	q.closed = true
}

// EventRecorder derives a stream of input events from successive input state
// snapshots and records them into an InputEventQueue.
//
// Observe never modifies the snapshot it is given, so recording stays isolated
// from the live state that the Is* queries read. Observing the same frame twice
// is a no-op, which keeps repeated sampling from duplicating events. Frame
// numbers are expected to advance with each new snapshot, but a frame that goes
// backwards (e.g. when a replay restarts) is still recorded; the queue's
// sequence numbers keep the event order unambiguous.
type EventRecorder struct {
	queue *InputEventQueue

	// now returns the time of an event. It defaults to a monotonic clock
	// started when the recorder was created and can be replaced with SetClock,
	// e.g. by tests.
	now   func() time.Duration
	start time.Time

	prev      ui.InputState
	hasPrev   bool
	lastFrame int64
	lastTime  time.Duration
}

// NewEventRecorder returns a recorder that records events derived from observed
// snapshots into queue. The queue must not be nil.
func NewEventRecorder(queue *InputEventQueue) *EventRecorder {
	r := &EventRecorder{
		queue: queue,
		start: time.Now(),
	}
	r.now = func() time.Duration {
		return time.Since(r.start)
	}
	return r
}

// SetClock replaces the clock used to stamp events. Passing nil restores the
// default monotonic clock.
func (r *EventRecorder) SetClock(now func() time.Duration) {
	if now == nil {
		now = func() time.Duration {
			return time.Since(r.start)
		}
	}
	r.now = now
}

// Reset makes the recorder forget the previously observed snapshot. The next
// Observe reports the state it sees as fresh: keys and buttons that are down
// produce press events, and touches produce touch press events.
func (r *EventRecorder) Reset() {
	r.prev = ui.InputState{}
	r.hasPrev = false
	r.lastFrame = 0
	r.lastTime = 0
}

// Observe compares state with the previously observed snapshot and records the
// difference as events stamped with frame. Observing the same frame twice is a
// no-op and returns nil.
//
// state is only read, never modified. An invalid rune in state.Runes (a
// surrogate half or a value outside the Unicode range) is skipped and reported
// in the returned error; the remaining events are still recorded. If the queue
// was closed, Observe stops recording and returns ErrQueueClosed.
func (r *EventRecorder) Observe(state *ui.InputState, frame int64) error {
	if state == nil {
		return errors.New("inputstate: cannot observe a nil input state")
	}
	if r.queue == nil {
		return errors.New("inputstate: event recorder has no queue")
	}
	if r.hasPrev && frame == r.lastFrame {
		// Repeated sampling of the same frame must not duplicate events.
		return nil
	}

	t := r.now()
	if t < r.lastTime {
		// A clock going backwards must not disorder events: clamp the
		// timestamp and let the sequence numbers define the order.
		t = r.lastTime
	}

	var errs []error
	push := func(ev InputEvent) bool {
		ev.Frame = frame
		ev.Time = t
		if err := r.queue.Push(ev); err != nil {
			errs = append(errs, err)
			return false
		}
		return true
	}

	prev := &r.prev

	for k := ui.Key(0); k <= ui.KeyMax; k++ {
		if state.KeyPressedTimes[k] != prev.KeyPressedTimes[k] {
			if !push(InputEvent{Kind: EventKindKeyPress, Device: InputDeviceKeyboard, Key: k}) {
				return errors.Join(errs...)
			}
		}
		if state.KeyReleasedTimes[k] != prev.KeyReleasedTimes[k] {
			if !push(InputEvent{Kind: EventKindKeyRelease, Device: InputDeviceKeyboard, Key: k}) {
				return errors.Join(errs...)
			}
		}
	}

	for b := ui.MouseButton(0); b <= ui.MouseButtonMax; b++ {
		if state.MouseButtonPressedTimes[b] != prev.MouseButtonPressedTimes[b] {
			if !push(InputEvent{Kind: EventKindMouseButtonPress, Device: InputDeviceMouse, Button: b}) {
				return errors.Join(errs...)
			}
		}
		if state.MouseButtonReleasedTimes[b] != prev.MouseButtonReleasedTimes[b] {
			if !push(InputEvent{Kind: EventKindMouseButtonRelease, Device: InputDeviceMouse, Button: b}) {
				return errors.Join(errs...)
			}
		}
	}

	if state.CursorX != prev.CursorX || state.CursorY != prev.CursorY {
		if !push(InputEvent{Kind: EventKindCursorMove, Device: InputDeviceMouse, X: state.CursorX, Y: state.CursorY}) {
			return errors.Join(errs...)
		}
	}

	// WheelX and WheelY are deltas accumulated since the previous snapshot, so
	// they are recorded as they are rather than diffed.
	if state.WheelX != 0 || state.WheelY != 0 {
		if !push(InputEvent{Kind: EventKindWheel, Device: InputDeviceMouse, WheelX: state.WheelX, WheelY: state.WheelY}) {
			return errors.Join(errs...)
		}
	}

	prevTouches := map[ui.TouchID]ui.Touch{}
	for _, touch := range prev.Touches {
		prevTouches[touch.ID] = touch
	}
	curTouches := map[ui.TouchID]ui.Touch{}
	for _, touch := range state.Touches {
		curTouches[touch.ID] = touch
		p, ok := prevTouches[touch.ID]
		switch {
		case !ok:
			if !push(InputEvent{Kind: EventKindTouchPress, Device: InputDeviceTouch, TouchID: touch.ID, X: touch.X, Y: touch.Y}) {
				return errors.Join(errs...)
			}
		case p.X != touch.X || p.Y != touch.Y:
			if !push(InputEvent{Kind: EventKindTouchMove, Device: InputDeviceTouch, TouchID: touch.ID, X: touch.X, Y: touch.Y}) {
				return errors.Join(errs...)
			}
		}
	}
	for _, touch := range prev.Touches {
		if _, ok := curTouches[touch.ID]; !ok {
			if !push(InputEvent{Kind: EventKindTouchRelease, Device: InputDeviceTouch, TouchID: touch.ID, X: touch.X, Y: touch.Y}) {
				return errors.Join(errs...)
			}
		}
	}

	// Runes are accumulated per snapshot like the wheel deltas.
	for _, ch := range state.Runes {
		if !utf8.ValidRune(ch) {
			errs = append(errs, fmt.Errorf("inputstate: invalid rune U+%08X in input state", ch))
			continue
		}
		if !push(InputEvent{Kind: EventKindRune, Device: InputDeviceText, Rune: ch}) {
			return errors.Join(errs...)
		}
	}

	// Keep a private copy of the snapshot so later mutations of state, or of
	// the slices it shares with the caller, cannot corrupt the next diff.
	r.prev = *state
	r.prev.Touches = append([]ui.Touch(nil), state.Touches...)
	r.prev.Runes = append([]rune(nil), state.Runes...)
	r.hasPrev = true
	r.lastFrame = frame
	r.lastTime = t

	return errors.Join(errs...)
}
