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

package mipmap_test

import (
	"bytes"
	"context"
	"errors"
	"image"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/internal/atlas"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
	"github.com/hajimehoshi/ebiten/v2/internal/mipmap"
	t "github.com/hajimehoshi/ebiten/v2/internal/testing"
	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

func TestMain(m *testing.M) {
	t.MainWithRunLoop(m)
}

// hugeViewport is a viewport comfortably covering any destination used in these tests.
var hugeViewport = image.Rect(-1<<20, -1<<20, 1<<20, 1<<20)

func TestPrefetchPlanLevels(t *testing.T) {
	rect64 := image.Rect(0, 0, 64, 64)
	huge := hugeViewport

	testCases := []struct {
		name      string
		width     int
		height    int
		bounds    image.Rectangle
		matrix    mipmap.PrefetchMatrix
		filter    mipmap.PrefetchFilter
		viewports []image.Rectangle
		want      []mipmap.PrefetchJob
	}{
		{
			name:      "scale 0.1 hits levels 1-3",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(0.1, 0.1),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(0, 0, 32, 32)},
				{Level: 2, Region: image.Rect(0, 0, 16, 16)},
				{Level: 3, Region: image.Rect(0, 0, 8, 8)},
			},
		},
		{
			name:      "scale 0.5 needs no mipmap",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(0.5, 0.5),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want:      nil,
		},
		{
			name:      "magnification needs no mipmap",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(1.5, 1.5),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want:      nil,
		},
		{
			name:      "nearest filter needs no mipmap",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(0.1, 0.1),
			filter:    mipmap.PrefetchFilterNearest,
			viewports: []image.Rectangle{huge},
			want:      nil,
		},
		{
			name:      "rotation by 90 degrees keeps the level",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchMatrix{A: 0, B: -0.1, C: 0.1, D: 0, Tx: 6.4},
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(0, 0, 32, 32)},
				{Level: 2, Region: image.Rect(0, 0, 16, 16)},
				{Level: 3, Region: image.Rect(0, 0, 8, 8)},
			},
		},
		{
			name:   "rotation by 45 degrees",
			width:  64,
			height: 64,
			bounds: rect64,
			matrix: mipmap.PrefetchMatrix{
				A: 0.2 * math.Cos(math.Pi/4), B: -0.2 * math.Sin(math.Pi/4),
				C: 0.2 * math.Sin(math.Pi/4), D: 0.2 * math.Cos(math.Pi/4),
			},
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(0, 0, 32, 32)},
				{Level: 2, Region: image.Rect(0, 0, 16, 16)},
			},
		},
		{
			name:      "negative translation clips the source region",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchMatrix{A: 0.1, D: 0.1, Tx: -3.2, Ty: -3.2},
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(16, 16, 32, 32)},
				{Level: 2, Region: image.Rect(8, 8, 16, 16)},
				{Level: 3, Region: image.Rect(4, 4, 8, 8)},
			},
		},
		{
			name:      "viewport outside the drawn quad",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(0.1, 0.1),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{image.Rect(100, 100, 200, 200)},
			want:      nil,
		},
		{
			name:      "empty viewports are skipped",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchScale(0.1, 0.1),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{{}, image.Rect(5, 5, 5, 5), huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(0, 0, 32, 32)},
				{Level: 2, Region: image.Rect(0, 0, 16, 16)},
				{Level: 3, Region: image.Rect(0, 0, 8, 8)},
			},
		},
		{
			name:      "sub-image bounds",
			width:     64,
			height:    64,
			bounds:    image.Rect(16, 16, 48, 48),
			matrix:    mipmap.PrefetchScale(0.25, 0.25),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(8, 8, 24, 24)},
			},
		},
		{
			name:      "non-power-of-two size",
			width:     63,
			height:    65,
			bounds:    image.Rect(0, 0, 63, 65),
			matrix:    mipmap.PrefetchScale(0.1, 0.1),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want: []mipmap.PrefetchJob{
				{Level: 1, Region: image.Rect(0, 0, 31, 32)},
				{Level: 2, Region: image.Rect(0, 0, 15, 16)},
				{Level: 3, Region: image.Rect(0, 0, 7, 8)},
			},
		},
		{
			name:      "degenerate matrix",
			width:     64,
			height:    64,
			bounds:    rect64,
			matrix:    mipmap.PrefetchMatrix{A: 0.1, B: 0.1, C: 0.2, D: 0.2},
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want:      nil,
		},
		{
			name:      "level images exceeding the limit are skipped",
			width:     16384,
			height:    16384,
			bounds:    image.Rect(0, 0, 16384, 16384),
			matrix:    mipmap.PrefetchScale(0.001, 0.001),
			filter:    mipmap.PrefetchFilterLinear,
			viewports: []image.Rectangle{huge},
			want:      nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := mipmap.New(tc.width, tc.height, atlas.ImageTypeRegular)
			defer m.Deallocate()
			req := &mipmap.PrefetchRequest{
				Bounds:    tc.bounds,
				Matrix:    tc.matrix,
				Filter:    tc.filter,
				Viewports: tc.viewports,
			}
			plan, err := m.NewPrefetchPlan(req)
			if err != nil {
				t.Fatal(err)
			}
			got := plan.Jobs()
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			for i := range got {
				got[i].Priority = 0
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("jobs: got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPrefetchPlanErrors(t *testing.T) {
	rect64 := image.Rect(0, 0, 64, 64)
	huge := hugeViewport

	t.Run("empty viewports", func(t *testing.T) {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		defer m.Deallocate()
		_, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds: rect64,
			Matrix: mipmap.PrefetchScale(0.1, 0.1),
			Filter: mipmap.PrefetchFilterLinear,
		})
		if !errors.Is(err, mipmap.ErrPrefetchEmptyViewports) {
			t.Errorf("got %v, want %v", err, mipmap.ErrPrefetchEmptyViewports)
		}
	})

	t.Run("empty bounds", func(t *testing.T) {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		defer m.Deallocate()
		_, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:    image.Rect(4, 4, 4, 8),
			Matrix:    mipmap.PrefetchScale(0.1, 0.1),
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{huge},
		})
		if !errors.Is(err, mipmap.ErrPrefetchInvalidBounds) {
			t.Errorf("got %v, want %v", err, mipmap.ErrPrefetchInvalidBounds)
		}
	})

	t.Run("zero and negative image size", func(t *testing.T) {
		for _, size := range [][2]int{{0, 0}, {0, 64}, {-1, -1}} {
			m := mipmap.New(size[0], size[1], atlas.ImageTypeRegular)
			_, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
				Bounds:    image.Rect(0, 0, 1, 1),
				Matrix:    mipmap.PrefetchScale(0.1, 0.1),
				Filter:    mipmap.PrefetchFilterLinear,
				Viewports: []image.Rectangle{huge},
			})
			if !errors.Is(err, mipmap.ErrPrefetchInvalidSize) {
				t.Errorf("size %v: got %v, want %v", size, err, mipmap.ErrPrefetchInvalidSize)
			}
		}
	})

	t.Run("memory limit", func(t *testing.T) {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		defer m.Deallocate()
		// Levels 1-6 of a 64x64 image estimate (32^2+16^2+8^2+4^2+2^2+1^2)*4 = 5460 bytes.
		_, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:        rect64,
			Matrix:        mipmap.PrefetchScale(0.01, 0.01),
			Filter:        mipmap.PrefetchFilterLinear,
			Viewports:     []image.Rectangle{huge},
			MaxTotalBytes: 4096,
		})
		if !errors.Is(err, mipmap.ErrPrefetchTooLarge) {
			t.Errorf("got %v, want %v", err, mipmap.ErrPrefetchTooLarge)
		}
	})

	t.Run("nil request", func(t *testing.T) {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		defer m.Deallocate()
		if _, err := m.NewPrefetchPlan(nil); err == nil {
			t.Error("expected an error, got nil")
		}
	})
}

func TestPrefetchPlanDeterministic(t *testing.T) {
	m := mipmap.New(63, 65, atlas.ImageTypeRegular)
	defer m.Deallocate()
	newReq := func() *mipmap.PrefetchRequest {
		return &mipmap.PrefetchRequest{
			Bounds: image.Rect(1, 2, 60, 63),
			Matrix: mipmap.PrefetchMatrix{
				A: 0.1 * math.Cos(math.Pi/6), B: -0.1 * math.Sin(math.Pi/6),
				C: 0.1 * math.Sin(math.Pi/6), D: 0.1 * math.Cos(math.Pi/6),
				Tx: -3, Ty: 5,
			},
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{image.Rect(-4, -4, 3, 2), image.Rect(0, 0, 8, 8)},
		}
	}
	p1, err := m.NewPrefetchPlan(newReq())
	if err != nil {
		t.Fatal(err)
	}
	p2, err := m.NewPrefetchPlan(newReq())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p1.Jobs(), p2.Jobs()) {
		t.Errorf("duplicate requests differ: %v vs %v", p1.Jobs(), p2.Jobs())
	}
	if len(p1.Jobs()) == 0 {
		t.Error("expected a non-empty plan")
	}
}

func TestPrefetchExecutorGenerates(t *testing.T) {
	m := mipmap.New(64, 64, atlas.ImageTypeRegular)
	defer m.Deallocate()
	plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
		Bounds:    image.Rect(0, 0, 64, 64),
		Matrix:    mipmap.PrefetchScale(0.1, 0.1),
		Filter:    mipmap.PrefetchFilterLinear,
		Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
	})
	if err != nil {
		t.Fatal(err)
	}
	e := &mipmap.PrefetchExecutor{}
	if err := e.Execute(context.Background(), plan); err != nil {
		t.Fatal(err)
	}
	for l := 1; l <= 3; l++ {
		if !m.HasCleanLevel(l) {
			t.Errorf("level %d was not generated", l)
		}
	}
	if got := plan.State(); got != mipmap.PrefetchPlanCompleted {
		t.Errorf("state: got %v, want %v", got, mipmap.PrefetchPlanCompleted)
	}

	// Executing a completed plan is a safe no-op: no job is generated again.
	calls := 0
	e2 := &mipmap.PrefetchExecutor{
		Generate: func(plan *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			calls++
			return nil
		},
	}
	if err := e2.Execute(context.Background(), plan); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Errorf("Generate was called %d times for a completed plan, want 0", calls)
	}
}

func TestPrefetchExecutorPartialLevelsAndDuplicates(t *testing.T) {
	m := mipmap.New(64, 64, atlas.ImageTypeRegular)
	defer m.Deallocate()

	// Generate level 1 in advance, simulating a partially existing chain.
	if err := m.PrefetchLevel(1); err != nil {
		t.Fatal(err)
	}

	newPlan := func() *mipmap.PrefetchPlan {
		plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:    image.Rect(0, 0, 64, 64),
			Matrix:    mipmap.PrefetchScale(0.1, 0.1),
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
		})
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}

	var mu sync.Mutex
	var generated []int
	e := &mipmap.PrefetchExecutor{
		Generate: func(plan *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			mu.Lock()
			generated = append(generated, job.Level)
			mu.Unlock()
			return plan.Mipmap().PrefetchLevel(job.Level)
		},
	}

	// Two identical plans in one call: the second plan's jobs must observe the levels
	// generated for the first plan and skip them.
	if err := e.Execute(context.Background(), newPlan(), newPlan()); err != nil {
		t.Fatal(err)
	}
	want := []int{2, 3}
	if !reflect.DeepEqual(generated, want) {
		t.Errorf("generated levels: got %v, want %v", generated, want)
	}
	for l := 1; l <= 3; l++ {
		if !m.HasCleanLevel(l) {
			t.Errorf("level %d is not clean after execution", l)
		}
	}
}

func TestPrefetchExecutorPriority(t *testing.T) {
	newPlan := func(priority int) *mipmap.PrefetchPlan {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		t.Cleanup(m.Deallocate)
		plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:    image.Rect(0, 0, 64, 64),
			Matrix:    mipmap.PrefetchScale(0.1, 0.1),
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
			Priority:  priority,
		})
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}

	low := newPlan(1)
	high := newPlan(10)

	var mu sync.Mutex
	var order []int
	e := &mipmap.PrefetchExecutor{
		MaxConcurrent: 1,
		Generate: func(plan *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			mu.Lock()
			order = append(order, job.Priority)
			mu.Unlock()
			return nil
		},
	}
	if err := e.Execute(context.Background(), low, high); err != nil {
		t.Fatal(err)
	}
	want := []int{10, 10, 10, 1, 1, 1}
	if !reflect.DeepEqual(order, want) {
		t.Errorf("execution order: got %v, want %v", order, want)
	}
}

func TestPrefetchExecutorBoundedConcurrency(t *testing.T) {
	const planCount = 3
	var plans []*mipmap.PrefetchPlan
	for i := 0; i < planCount; i++ {
		m := mipmap.New(64, 64, atlas.ImageTypeRegular)
		t.Cleanup(m.Deallocate)
		plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:    image.Rect(0, 0, 64, 64),
			Matrix:    mipmap.PrefetchScale(0.1, 0.1),
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
		})
		if err != nil {
			t.Fatal(err)
		}
		plans = append(plans, plan)
	}

	var cur, max atomic.Int32
	bothInside := make(chan struct{})
	var once sync.Once
	e := &mipmap.PrefetchExecutor{
		MaxConcurrent: 2,
		Generate: func(plan *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			c := cur.Add(1)
			defer cur.Add(-1)
			for {
				if m := max.Load(); c <= m || max.CompareAndSwap(m, c) {
					break
				}
			}
			if c == 2 {
				once.Do(func() { close(bothInside) })
			}
			select {
			case <-bothInside:
			case <-time.After(10 * time.Second):
				return errors.New("timed out waiting for a concurrent job")
			}
			return nil
		},
	}
	if err := e.Execute(context.Background(), plans...); err != nil {
		t.Fatal(err)
	}
	if got := max.Load(); got != 2 {
		t.Errorf("max concurrency: got %d, want 2", got)
	}
}

func TestPrefetchExecutorCancel(t *testing.T) {
	m := mipmap.New(64, 64, atlas.ImageTypeRegular)
	defer m.Deallocate()
	plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
		Bounds:          image.Rect(0, 0, 64, 64),
		Matrix:          mipmap.PrefetchScale(0.1, 0.1),
		Filter:          mipmap.PrefetchFilterLinear,
		Viewports:       []image.Rectangle{image.Rect(0, 0, 100, 100)},
		ReclaimOnCancel: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	e := &mipmap.PrefetchExecutor{
		Generate: func(p *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			if job.Level == 1 {
				// Complete one level so that cancellation has something to reclaim.
				return p.Mipmap().PrefetchLevel(1)
			}
			once.Do(func() { close(started) })
			<-release
			return nil
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- e.Execute(ctx, plan)
	}()

	<-started
	cancel()
	close(release)

	if err := <-errCh; !errors.Is(err, mipmap.ErrPrefetchCanceled) {
		t.Fatalf("got %v, want %v", err, mipmap.ErrPrefetchCanceled)
	}
	if got := plan.State(); got != mipmap.PrefetchPlanCanceled {
		t.Errorf("state: got %v, want %v", got, mipmap.PrefetchPlanCanceled)
	}
	// The level generated before the cancellation must have been reclaimed.
	if m.HasCleanLevel(1) {
		t.Error("level 1 was not reclaimed after cancellation")
	}
}

func TestPrefetchExecutorStalePlan(t *testing.T) {
	m := mipmap.New(64, 64, atlas.ImageTypeRegular)
	defer m.Deallocate()
	newPlan := func() *mipmap.PrefetchPlan {
		plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
			Bounds:    image.Rect(0, 0, 64, 64),
			Matrix:    mipmap.PrefetchScale(0.1, 0.1),
			Filter:    mipmap.PrefetchFilterLinear,
			Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
		})
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}

	plan := newPlan()
	epoch := m.EpochForTesting()

	// Simulate a device rebuild: the images are deallocated and re-created on demand.
	m.Deallocate()
	if m.EpochForTesting() == epoch {
		t.Fatal("epoch did not change after Deallocate")
	}

	e := &mipmap.PrefetchExecutor{}
	if err := e.Execute(context.Background(), plan); !errors.Is(err, mipmap.ErrPrefetchStalePlan) {
		t.Fatalf("got %v, want %v", err, mipmap.ErrPrefetchStalePlan)
	}
	if got := plan.State(); got != mipmap.PrefetchPlanFailed {
		t.Errorf("state: got %v, want %v", got, mipmap.PrefetchPlanFailed)
	}

	// A plan created after the rebuild works deterministically.
	plan2 := newPlan()
	if err := e.Execute(context.Background(), plan2); err != nil {
		t.Fatal(err)
	}
	for l := 1; l <= 3; l++ {
		if !m.HasCleanLevel(l) {
			t.Errorf("level %d was not generated after the rebuild", l)
		}
	}
}

func TestPrefetchExecutorGenerationFailure(t *testing.T) {
	m := mipmap.New(64, 64, atlas.ImageTypeRegular)
	defer m.Deallocate()
	plan, err := m.NewPrefetchPlan(&mipmap.PrefetchRequest{
		Bounds:    image.Rect(0, 0, 64, 64),
		Matrix:    mipmap.PrefetchScale(0.1, 0.1),
		Filter:    mipmap.PrefetchFilterLinear,
		Viewports: []image.Rectangle{image.Rect(0, 0, 100, 100)},
	})
	if err != nil {
		t.Fatal(err)
	}

	errGen := errors.New("generation failed")
	e := &mipmap.PrefetchExecutor{
		Generate: func(p *mipmap.PrefetchPlan, job mipmap.PrefetchJob) error {
			if job.Level == 2 {
				return errGen
			}
			return nil
		},
	}
	if err := e.Execute(context.Background(), plan); !errors.Is(err, errGen) {
		t.Fatalf("got %v, want %v", err, errGen)
	}
	if got := plan.State(); got != mipmap.PrefetchPlanFailed {
		t.Errorf("state: got %v, want %v", got, mipmap.PrefetchPlanFailed)
	}
	if !errors.Is(plan.Err(), errGen) {
		t.Errorf("plan.Err(): got %v, want %v", plan.Err(), errGen)
	}
}

// patternPixels returns 64x64 premultiplied RGBA pixels for a test pattern.
func patternPixels(pattern string, w, h int) []byte {
	pix := make([]byte, 4*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := 4 * (y*w + x)
			switch pattern {
			case "checkerboard":
				if (x+y)%2 == 0 {
					pix[i] = 0xff
					pix[i+1] = 0xff
					pix[i+2] = 0xff
				}
				pix[i+3] = 0xff
			case "thinlines":
				if x%4 == 0 {
					pix[i] = 0xff
					pix[i+1] = 0xff
					pix[i+2] = 0xff
				}
				pix[i+3] = 0xff
			case "alphaedge":
				var a byte
				switch {
				case x >= 24 && x < 40:
					a = byte((x - 24) * 16)
				case x >= 40:
					a = 0xff
				}
				pix[i] = a
				pix[i+1] = a
				pix[i+2] = a
				pix[i+3] = a
			}
		}
	}
	return pix
}

func drawScaled(dst, src *mipmap.Mipmap, width, height int, scale float32) {
	vs := make([]float32, 4*graphics.VertexFloatCount)
	graphics.QuadVerticesFromSrcAndMatrix(vs, 0, 0, float32(width), float32(height), scale, 0, 0, scale, 0, 0, 1, 1, 1, 1)
	is := graphics.QuadIndices()
	dstRegion := image.Rect(0, 0, width, height)
	srcRegions := [graphics.ShaderSrcImageCount]image.Rectangle{image.Rect(0, 0, width, height)}
	dst.DrawTriangles([graphics.ShaderSrcImageCount]*mipmap.Mipmap{src}, vs, is, graphicsdriver.BlendCopy, dstRegion, srcRegions, atlas.LinearFilterShader, nil, false)
}

func readPixels(m *mipmap.Mipmap, pixels []byte, region image.Rectangle) error {
	result := m.ReadPixelsAsync(pixels, region)
	if err := atlas.FlushAndWaitPixelReadbacks(ui.Get().GraphicsDriverForTesting(), []*graphicscommand.PixelReadback{result}); err != nil {
		return err
	}
	_, err := result.Poll()
	return err
}

func TestPrefetchPixelsMatchSynchronous(t *testing.T) {
	const size = 64
	bounds := image.Rect(0, 0, size, size)
	// A sequence of rapidly increasing minification, as when a camera zooms out quickly.
	scales := []float32{0.4, 0.2, 0.1, 0.05, 0.02}

	for _, pattern := range []string{"checkerboard", "thinlines", "alphaedge"} {
		t.Run(pattern, func(t *testing.T) {
			pixels := patternPixels(pattern, size, size)

			// prefetched is reused across the whole scale sequence so that later plans
			// observe partially existing level chains.
			prefetched := mipmap.New(size, size, atlas.ImageTypeRegular)
			defer prefetched.Deallocate()
			prefetched.WritePixels(pixels, bounds)

			for _, scale := range scales {
				func() {
					// Synchronous reference: levels are generated on demand during the draw.
					syncSrc := mipmap.New(size, size, atlas.ImageTypeRegular)
					defer syncSrc.Deallocate()
					syncSrc.WritePixels(pixels, bounds)

					plan, err := prefetched.NewPrefetchPlan(&mipmap.PrefetchRequest{
						Bounds:    bounds,
						Matrix:    mipmap.PrefetchScale(float64(scale), float64(scale)),
						Filter:    mipmap.PrefetchFilterLinear,
						Viewports: []image.Rectangle{bounds},
					})
					if err != nil {
						t.Fatal(err)
					}
					e := &mipmap.PrefetchExecutor{}
					if err := e.Execute(context.Background(), plan); err != nil {
						t.Fatal(err)
					}

					// The plan's deepest job must match the level selected for this scale.
					wantLevel := mipmap.MipmapLevelFromDistance(0, 0, float32(size)*scale, 0, 0, 0, float32(size), 0)
					jobs := plan.Jobs()
					if got := jobs[len(jobs)-1].Level; got != wantLevel {
						t.Errorf("scale %v: deepest prefetched level: got %d, want %d", scale, got, wantLevel)
					}
					if !prefetched.HasCleanLevel(wantLevel) {
						t.Errorf("scale %v: level %d is not clean after prefetching", scale, wantLevel)
					}

					syncDst := mipmap.New(size, size, atlas.ImageTypeRegular)
					defer syncDst.Deallocate()
					drawScaled(syncDst, syncSrc, size, size, scale)

					prefetchDst := mipmap.New(size, size, atlas.ImageTypeRegular)
					defer prefetchDst.Deallocate()
					drawScaled(prefetchDst, prefetched, size, size, scale)

					got := make([]byte, 4*size*size)
					if err := readPixels(prefetchDst, got, bounds); err != nil {
						t.Fatal(err)
					}
					want := make([]byte, 4*size*size)
					if err := readPixels(syncDst, want, bounds); err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(got, want) {
						maxDiff := 0
						for i := range got {
							d := int(got[i]) - int(want[i])
							if d < 0 {
								d = -d
							}
							if d > maxDiff {
								maxDiff = d
							}
						}
						t.Errorf("scale %v: prefetched pixels differ from synchronous ones, max channel diff %d", scale, maxDiff)
					}
				}()
			}
		})
	}
}
