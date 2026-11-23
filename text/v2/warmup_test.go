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

package text_test

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/bitmapfont/v4"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
)

// warmupFixture returns a long multilingual text. It intentionally contains
// runes missing from MPLUS1p (Hangul, Arabic, emoji) to exercise the
// missing-glyph path, and repeats segments so deduplication has work to do.
func warmupFixture() string {
	parts := []string{
		"The quick brown fox jumps over the lazy dog. ",
		"こんにちは世界。今日はいい天気ですね。",
		"日本語の長い文章をここに書きます。漢字も交じります。",
		"안녕하세요 세계. ",
		"مرحبا بالعالم. ",
		"🙂🎉✨ ",
	}
	var sb strings.Builder
	for range 3 {
		for _, p := range parts {
			sb.WriteString(p)
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func mustGoTextFaceSource(t *testing.T, fontFile string) *text.GoTextFaceSource {
	t.Helper()
	fontdata, err := os.ReadFile(filepath.Join("testdata", fontFile))
	if err != nil {
		t.Fatal(err)
	}
	src, err := text.NewGoTextFaceSource(bytes.NewReader(fontdata))
	if err != nil {
		t.Fatal(err)
	}
	return src
}

func renderFixture(dst *ebiten.Image, fixture string, face text.Face) {
	op := &text.DrawOptions{}
	op.LineSpacing = 32
	text.Draw(dst, fixture, face, op)
}

func assertSamePixels(t *testing.T, want, got *ebiten.Image) {
	t.Helper()
	b := want.Bounds()
	if got.Bounds() != b {
		t.Fatalf("bounds differ: got %v, want %v", got.Bounds(), b)
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			w := want.At(x, y)
			g := got.At(x, y)
			if w != g {
				t.Fatalf("pixel (%d, %d): got %v, want %v", x, y, g, w)
			}
		}
	}
}

func TestTextGlyphWarmupPlanColdWarmPixels(t *testing.T) {
	fixture := warmupFixture()

	newGoTextFace := func() (*text.GoTextFaceSource, *text.GoTextFace) {
		src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
		return src, &text.GoTextFace{Source: src, Size: 24}
	}

	t.Run("GoTextFace", func(t *testing.T) {
		_, coldFace := newGoTextFace()
		cold := ebiten.NewImage(2048, 256)
		renderFixture(cold, fixture, coldFace)
		coldGlyphs := len(text.AppendLazyGlyphs(nil, fixture, coldFace, &text.LayoutOptions{LineSpacing: 32}))

		_, warmFace := newGoTextFace()
		plan := text.NewTextGlyphWarmupPlan(fixture, warmFace)
		if plan.Len() == 0 {
			t.Fatal("plan must not be empty")
		}
		if n := plan.Run(plan.Len()); n != plan.Len() {
			t.Fatalf("Run executed %d tasks, want %d", n, plan.Len())
		}
		if !plan.Wait() {
			t.Fatalf("plan did not complete: err=%v", plan.Err())
		}
		warm := ebiten.NewImage(2048, 256)
		renderFixture(warm, fixture, warmFace)
		warmGlyphs := len(text.AppendLazyGlyphs(nil, fixture, warmFace, &text.LayoutOptions{LineSpacing: 32}))

		if coldGlyphs != warmGlyphs {
			t.Errorf("glyph counts differ: cold %d, warm %d", coldGlyphs, warmGlyphs)
		}
		assertSamePixels(t, cold, warm)
	})

	t.Run("MultiFace", func(t *testing.T) {
		newMultiFace := func() text.Face {
			src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
			primary := &text.GoTextFace{Source: src, Size: 24}
			fallbackSrc := mustGoTextFaceSource(t, "Roboto-Regular.ttf")
			fallback := &text.GoTextFace{Source: fallbackSrc, Size: 24}
			mf, err := text.NewMultiFace(primary, fallback)
			if err != nil {
				t.Fatal(err)
			}
			return mf
		}

		cold := ebiten.NewImage(2048, 256)
		renderFixture(cold, fixture, newMultiFace())

		warmFace := newMultiFace()
		plan := text.NewTextGlyphWarmupPlan(fixture, warmFace)
		plan.Start()
		if !plan.Wait() {
			t.Fatalf("plan did not complete: err=%v", plan.Err())
		}
		warm := ebiten.NewImage(2048, 256)
		renderFixture(warm, fixture, warmFace)

		assertSamePixels(t, cold, warm)
	})

	t.Run("GoXFace", func(t *testing.T) {
		cold := ebiten.NewImage(2048, 256)
		renderFixture(cold, fixture, text.NewGoXFace(bitmapfont.Face))

		warmFace := text.NewGoXFace(bitmapfont.Face)
		plan := text.NewTextGlyphWarmupPlan(fixture, warmFace)
		plan.Start()
		if !plan.Wait() {
			t.Fatalf("plan did not complete: err=%v", plan.Err())
		}
		warm := ebiten.NewImage(2048, 256)
		renderFixture(warm, fixture, warmFace)

		assertSamePixels(t, cold, warm)
	})
}

func TestTextGlyphWarmupPlanCacheHitRate(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}
	fixture := warmupFixture()

	plan := text.NewTextGlyphWarmupPlan(fixture, face)
	plan.Start()
	if !plan.Wait() {
		t.Fatalf("plan did not complete: err=%v", plan.Err())
	}

	before := text.GlyphImageCacheEntryCount(src, 24)
	if before == 0 {
		t.Fatal("warmup populated no glyph images")
	}

	dst := ebiten.NewImage(2048, 256)
	renderFixture(dst, fixture, face)

	after := text.GlyphImageCacheEntryCount(src, 24)
	if after != before {
		t.Errorf("Draw after warmup created %d new glyph image entries (hit rate %.1f%%), want 0 (100%%)",
			after-before, 100*float64(before)/float64(after))
	}
}

func TestTextGlyphWarmupPlanCancel(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}

	// Many unique runes: CJK unified ideographs, most missing from the font.
	var sb strings.Builder
	for r := rune(0x4E00); r < 0x4E00+2000; r++ {
		sb.WriteRune(r)
	}
	plan := text.NewTextGlyphWarmupPlan(sb.String(), face)
	if plan.Len() != 2000 {
		t.Fatalf("Len = %d, want 2000", plan.Len())
	}

	if n := plan.Run(10); n != 10 {
		t.Fatalf("Run(10) executed %d tasks, want 10", n)
	}
	plan.Cancel()
	plan.Cancel() // Cancel is idempotent.
	if !plan.Canceled() {
		t.Error("Canceled must report true")
	}
	done, total := plan.Progress()
	if done != 10 || total != 2000 {
		t.Errorf("Progress = (%d, %d), want (10, 2000)", done, total)
	}
	if plan.Wait() {
		t.Error("a canceled plan must not report completion")
	}
	if err := plan.Err(); err != nil {
		t.Errorf("Err = %v, want nil", err)
	}

	// Execution after cancellation is a stable no-op.
	if n := plan.Run(5); n != 0 {
		t.Errorf("Run after Cancel executed %d tasks, want 0", n)
	}
	plan.Start() // must not spawn a worker
	if done, _ := plan.Progress(); done != 10 {
		t.Errorf("Start after Cancel advanced progress to %d, want 10", done)
	}
}

func TestTextGlyphWarmupPlanCancelAsync(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}

	var sb strings.Builder
	for r := rune(0x4E00); r < 0x4E00+2000; r++ {
		sb.WriteRune(r)
	}
	plan := text.NewTextGlyphWarmupPlan(sb.String(), face)
	plan.Start()
	plan.Cancel()
	if plan.Wait() {
		t.Error("a canceled plan must not report completion")
	}
	done, total := plan.Progress()
	if done >= total {
		t.Errorf("canceled plan completed all tasks: done=%d total=%d", done, total)
	}
}

func TestTextGlyphWarmupPlanEdgeCases(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}

	t.Run("EmptyText", func(t *testing.T) {
		plan := text.NewTextGlyphWarmupPlan("", face)
		if plan.Len() != 0 {
			t.Errorf("Len = %d, want 0", plan.Len())
		}
		if n := plan.Run(1); n != 0 {
			t.Errorf("Run executed %d tasks, want 0", n)
		}
		plan.Start() // no-op, must not block Wait
		if !plan.Wait() {
			t.Error("an empty plan must report completion")
		}
	})

	t.Run("NilFace", func(t *testing.T) {
		plan := text.NewTextGlyphWarmupPlan("hello", nil)
		if plan.Len() != 0 {
			t.Errorf("Len = %d, want 0", plan.Len())
		}
		if n := plan.Run(1); n != 0 {
			t.Errorf("Run executed %d tasks, want 0", n)
		}
		if !plan.Wait() {
			t.Error("a nil-face plan must terminate")
		}
	})

	t.Run("InvalidFace", func(t *testing.T) {
		// A GoTextFace without a Source cannot even report its metrics.
		plan := text.NewTextGlyphWarmupPlan("hello", &text.GoTextFace{})
		if plan.Err() == nil {
			t.Error("Err must be non-nil for an invalid face")
		}
		if plan.Len() != 0 {
			t.Errorf("Len = %d, want 0", plan.Len())
		}
		if n := plan.Run(1); n != 0 {
			t.Errorf("Run executed %d tasks, want 0", n)
		}
		plan.Start() // no-op
		if plan.Wait() {
			t.Error("a failed plan must not report completion")
		}
	})

	t.Run("ZeroBudget", func(t *testing.T) {
		plan := text.NewTextGlyphWarmupPlan(warmupFixture(), face)
		defer plan.Cancel()
		if n := plan.Run(0); n != 0 {
			t.Errorf("Run(0) executed %d tasks, want 0", n)
		}
		if n := plan.Run(-1); n != 0 {
			t.Errorf("Run(-1) executed %d tasks, want 0", n)
		}
		if done, _ := plan.Progress(); done != 0 {
			t.Errorf("done = %d, want 0", done)
		}
	})

	t.Run("DuplicateText", func(t *testing.T) {
		plan := text.NewTextGlyphWarmupPlan("aaaaaaaaaa", face)
		defer plan.Cancel()
		if plan.Len() != 1 {
			t.Errorf("Len = %d, want 1", plan.Len())
		}
	})

	t.Run("InvalidUTF8", func(t *testing.T) {
		plan := text.NewTextGlyphWarmupPlan("\xff\xfe\xff", face)
		if plan.Len() != 1 {
			t.Errorf("Len = %d, want 1 (U+FFFD deduped)", plan.Len())
		}
		if n := plan.Run(plan.Len()); n != 1 {
			t.Errorf("Run executed %d tasks, want 1", n)
		}
		if !plan.Wait() {
			t.Errorf("plan did not complete: err=%v", plan.Err())
		}
	})

	t.Run("DuplicateSubmission", func(t *testing.T) {
		fixture := warmupFixture()
		p1 := text.NewTextGlyphWarmupPlan(fixture, face)
		p2 := text.NewTextGlyphWarmupPlan(fixture, face)
		p1.Start()
		p1.Start() // duplicate Start is a no-op
		p2.Start()
		if !p1.Wait() {
			t.Errorf("p1 did not complete: err=%v", p1.Err())
		}
		if !p2.Wait() {
			t.Errorf("p2 did not complete: err=%v", p2.Err())
		}
		if done, total := p1.Progress(); done != total {
			t.Errorf("p1 Progress = (%d, %d), want done == total", done, total)
		}
	})
}

func TestTextGlyphWarmupPlanConcurrentDraw(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}
	fixture := warmupFixture()

	// Cold reference from a separate, untouched source.
	refSrc := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	refFace := &text.GoTextFace{Source: refSrc, Size: 24}
	ref := ebiten.NewImage(2048, 256)
	renderFixture(ref, fixture, refFace)

	plan := text.NewTextGlyphWarmupPlan(fixture, face)
	plan.Start()

	// Draw concurrently with the running warmup. Cache consistency must hold.
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := ebiten.NewImage(512, 256)
			for range 8 {
				renderFixture(dst, fixture, face)
			}
		}()
	}
	wg.Wait()

	if !plan.Wait() {
		t.Fatalf("plan did not complete: err=%v", plan.Err())
	}
	done, total := plan.Progress()
	if done != total {
		t.Fatalf("Progress = (%d, %d), want done == total", done, total)
	}

	got := ebiten.NewImage(2048, 256)
	renderFixture(got, fixture, face)
	assertSamePixels(t, ref, got)
}

func TestTextGlyphWarmupPlanNoGoroutineLeak(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}
	fixture := warmupFixture()

	before := runtime.NumGoroutine()
	for range 4 {
		plan := text.NewTextGlyphWarmupPlan(fixture, face)
		plan.Start()
		plan.Wait()
	}
	for range 4 {
		plan := text.NewTextGlyphWarmupPlan(fixture, face)
		plan.Start()
		plan.Cancel()
		plan.Wait()
	}
	// Workers close doneCh just before exiting; give stragglers a moment.
	time.Sleep(100 * time.Millisecond)
	if after := runtime.NumGoroutine(); after > before {
		t.Errorf("goroutines leaked: before=%d, after=%d", before, after)
	}
}

func TestTextGlyphWarmupPlanMemoryPeak(t *testing.T) {
	src := mustGoTextFaceSource(t, "MPLUS1p-Regular.ttf")
	face := &text.GoTextFace{Source: src, Size: 24}

	// A large unique rune set, like a full CJK dialog dictionary.
	var sb strings.Builder
	for r := rune(0x4E00); r < 0x4E00+1024; r++ {
		sb.WriteRune(r)
	}

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	plan := text.NewTextGlyphWarmupPlan(sb.String(), face)
	if n := plan.Run(plan.Len()); n != plan.Len() {
		t.Fatalf("Run executed %d tasks, want %d", n, plan.Len())
	}
	if !plan.Wait() {
		t.Fatalf("plan did not complete: err=%v", plan.Err())
	}

	runtime.ReadMemStats(&after)
	alloc := after.TotalAlloc - before.TotalAlloc
	t.Logf("warmup of %d runes allocated %d bytes total", plan.Len(), alloc)
	// Generous bound to catch runaway allocation or unbounded queueing.
	if alloc > 1<<30 {
		t.Errorf("warmup allocated %d bytes, want < 1GiB", alloc)
	}
}
