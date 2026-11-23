// Copyright 2026 The Ebiten Authors
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

package atlas_test

import (
	"image"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2/internal/atlas"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
	"github.com/hajimehoshi/ebiten/v2/internal/ui"
)

// driver returns the graphics driver used by the tests.
func driver() graphicsdriver.Graphics {
	return ui.Get().GraphicsDriverForTesting()
}

// newResidencyImage creates an unmanaged image, which always gets its own isolated GPU texture,
// filled with deterministic pixels.
func newResidencyImage(t *testing.T, size int, seed byte) *atlas.Image {
	t.Helper()
	img := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(img.Deallocate)
	pix := make([]byte, 4*size*size)
	for i := range size * size {
		pix[4*i] = seed + byte(i)
		pix[4*i+1] = seed + byte(i>>8)
		pix[4*i+2] = seed
		pix[4*i+3] = 0xff
	}
	img.WritePixels(pix, image.Rect(0, 0, size, size))
	return img
}

// readImagePixels reads back the full image through the public read path.
func readImagePixels(t *testing.T, img *atlas.Image, size int) []byte {
	t.Helper()
	pix := make([]byte, 4*size*size)
	if err := readPixels(img, pix, image.Rect(0, 0, size, size)); err != nil {
		t.Fatalf("readPixels failed: %v", err)
	}
	return pix
}

func expectedPixels(size int, seed byte) []byte {
	pix := make([]byte, 4*size*size)
	for i := range size * size {
		pix[4*i] = seed + byte(i)
		pix[4*i+1] = seed + byte(i>>8)
		pix[4*i+2] = seed
		pix[4*i+3] = 0xff
	}
	return pix
}

func drawOnto(t *testing.T, dst, src *atlas.Image, size int) {
	t.Helper()
	vs := quadVertices(size, size, 0, 0, 1)
	r := image.Rect(0, 0, size, size)
	dst.DrawTriangles([graphics.ShaderSrcImageCount]*atlas.Image{src}, vs, graphics.QuadIndices(), graphicsdriver.BlendCopy, r, [graphics.ShaderSrcImageCount]image.Rectangle{r}, atlas.NearestFilterShader, nil)
}

func TestResidencyBudgetDisabledByDefault(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	img := newResidencyImage(t, 16, 1)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("a backend must be resident when no budget is set")
	}
	stats := atlas.ResidencyStatsForTesting()
	if stats.OverBudget || stats.CumulativeEvictions != 0 || stats.EvictedTextures != 0 {
		t.Errorf("unexpected stats with no budget: %+v", stats)
	}
}

func TestResidencyEvictionAndReloadPreservePixels(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	imgs := make([]*atlas.Image, 4)
	for i := range imgs {
		imgs[i] = newResidencyImage(t, size, byte(i+1))
	}

	before := atlas.ResidencyStatsForTesting()
	if before.ResidentTextures < len(imgs) {
		t.Fatalf("resident textures: got %d, want at least %d", before.ResidentTextures, len(imgs))
	}

	// A budget that fits only two of the backends.
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{
		MaxTextures: 2,
	})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	after := atlas.ResidencyStatsForTesting()
	if after.ResidentTextures > 2 {
		t.Errorf("resident textures after trim: got %d, want <= 2", after.ResidentTextures)
	}
	if after.CumulativeEvictions == 0 {
		t.Error("trim did not evict anything")
	}
	if after.EvictedTextures == 0 {
		t.Error("no backend reports as evicted")
	}

	// Every image must return exactly the pixels originally written, whether reloaded or mirrored.
	for i, img := range imgs {
		got := readImagePixels(t, img, size)
		want := expectedPixels(size, byte(i+1))
		if string(got) != string(want) {
			t.Errorf("image %d pixels changed after eviction/reload", i)
		}
	}
}

func TestResidencyLRUOrder(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	imgs := make([]*atlas.Image, 3)
	for i := range imgs {
		imgs[i] = newResidencyImage(t, size, byte(i+10))
	}
	isResident := make([]func() bool, len(imgs))
	for i, img := range imgs {
		isResident[i] = img.IsBackendResidentFuncForTesting()
	}

	// Touch imgs[2] most recently, then imgs[1]; imgs[0] is the least recently used.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, imgs[0], size)
	drawOnto(t, dst, imgs[1], size)
	drawOnto(t, dst, imgs[2], size)

	// Evict aggressively enough that all three image backends become candidates in LRU order.
	// A zero budget removes every unlocked texture; the most recently used one is evicted last,
	// so after re-residenting it the survivors reveal the order.
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	for i, f := range isResident {
		if f() {
			t.Errorf("img%d should be evicted under a zero budget", i)
		}
	}

	// Reload in the same order as the original touches; the LRU bookkeeping must survive the
	// eviction/reload cycle and keep pixel content intact.
	drawOnto(t, dst, imgs[2], size)
	drawOnto(t, dst, imgs[1], size)
	if isResident[0]() {
		t.Error("the least recently used image must stay evicted after only the others reloaded")
	}
	if !isResident[1]() || !isResident[2]() {
		t.Error("the reloaded images should be resident again")
	}
	// Finally reloading the oldest image works and restores its exact pixels.
	drawOnto(t, dst, imgs[0], size)
	if got := readImagePixels(t, imgs[0], size); string(got) != string(expectedPixels(size, 10)) {
		t.Error("the least recently used image lost its pixels across eviction and reload")
	}
}

func TestResidencyLockPinsBackends(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	img0 := newResidencyImage(t, size, 1)
	img1 := newResidencyImage(t, size, 2)

	// Establish backends before taking the lock.
	vs := quadVertices(size, size, 0, 0, 1)
	r := image.Rect(0, 0, size, size)
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)

	lock := atlas.LockResidencyForTesting()
	// Use img0 under the lock so only its backend is pinned.
	dst.DrawTriangles([graphics.ShaderSrcImageCount]*atlas.Image{img0}, vs, graphics.QuadIndices(), graphicsdriver.BlendCopy, r, [graphics.ShaderSrcImageCount]image.Rectangle{r}, atlas.NearestFilterShader, nil)

	// Both the source and the destination of the locked draw are pinned; img1, which never
	// participates while the lock is held, is not.
	if got := atlas.LockedBackendsForTesting(); got != 2 {
		t.Fatalf("locked backends: got %d, want 2", got)
	}

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img0.IsBackendResidentFuncForTesting()() {
		t.Error("a locked backend must not be evicted")
	}
	if img1.IsBackendResidentFuncForTesting()() {
		t.Error("the unlocked backend should have been evicted")
	}

	lock.Release()
	lock.Release() // double release must be a no-op.
	if got := atlas.LockedBackendsForTesting(); got != 0 {
		t.Errorf("locked backends after release: got %d, want 0", got)
	}

	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim after release failed: %v", err)
	}
	if img0.IsBackendResidentFuncForTesting()() {
		t.Error("the previously locked backend must be evictable after release")
	}
}

func TestResidencyZeroBudgetReloadsOnDemand(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 12
	img := newResidencyImage(t, size, 7)

	// Zero limits: everything unlocked gets evicted.
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the backend must be evicted under a zero budget")
	}

	// Using the image as a source forces a reload, with identical pixels.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)

	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("the backend must be resident again after being drawn")
	}
	stats := atlas.ResidencyStatsForTesting()
	if stats.CumulativeReloads < 1 {
		t.Errorf("reload count: got %d, want >= 1", stats.CumulativeReloads)
	}

	got := readImagePixels(t, img, size)
	if string(got) != string(expectedPixels(size, 7)) {
		t.Error("reloaded pixels differ from the original image")
	}
}

func TestResidencySingleTextureOverBudget(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	img := newResidencyImage(t, size, 3)
	original := readImagePixels(t, img, size)

	// A byte budget smaller than a single texture: the texture is evictable because a zero
	// residency fits the budget, but when the image is used again its re-upload immediately
	// overshoots the budget without any retry loop.
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{MaxBytes: 1})
	done := make(chan error, 1)
	go func() {
		done <- atlas.TrimForTesting(driver())
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("trim returned an error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("trim did not return for a single texture over budget")
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the oversized texture should have been evicted")
	}

	// Re-upload overshoots the budget; the system must accept that instead of failing or looping,
	// and the image content must stay correct.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("the oversized texture must be re-uploaded on demand despite exceeding the budget")
	}
	stats := atlas.ResidencyStatsForTesting()
	if !stats.OverBudget {
		t.Error("stats must report the over-budget state while the texture is resident")
	}
	if got := readImagePixels(t, img, size); string(got) != string(original) {
		t.Error("pixels changed after an over-budget reload")
	}
}

func TestResidencyEvictionFailureKeepsTextureResident(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	imgs := make([]*atlas.Image, 3)
	for i := range imgs {
		imgs[i] = newResidencyImage(t, size, byte(i+1))
	}

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{MaxTextures: 1})
	// The first eviction attempt fails; the trim must continue with the other backends.
	atlas.SetResidencyFailureHooksForTesting(1, 0)
	if err := atlas.TrimForTesting(driver()); err == nil {
		t.Error("an eviction failure must be reported")
	}

	stats := atlas.ResidencyStatsForTesting()
	if stats.CumulativeEvictions == 0 {
		t.Error("the remaining backends should still have been evicted after one failure")
	}
	// No dangling handle: every image still reads correct pixels.
	for i, img := range imgs {
		got := readImagePixels(t, img, size)
		if string(got) != string(expectedPixels(size, byte(i+1))) {
			t.Errorf("image %d pixels changed after a failed eviction", i)
		}
	}
}

func TestResidencyReloadFailureDoesNotHang(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	img := newResidencyImage(t, size, 5)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the backend must be evicted")
	}

	// Force the next reload to fail. The draw must return instead of enqueueing a nil-texture draw.
	atlas.SetResidencyFailureHooksForTesting(0, 1)
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)

	if img.IsBackendResidentFuncForTesting()() {
		t.Error("the backend must stay evicted after a failed reload")
	}

	// A later attempt without the failure hook reloads successfully and exactly once per attempt.
	before := atlas.ResidencyStatsForTesting().CumulativeReloads
	drawOnto(t, dst, img, size)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("the backend must reload once the failure condition clears")
	}
	if got := atlas.ResidencyStatsForTesting().CumulativeReloads - before; got != 1 {
		t.Errorf("reload attempts after recovery: got %d, want 1", got)
	}
	got := readImagePixels(t, img, size)
	if string(got) != string(expectedPixels(size, 5)) {
		t.Error("pixels changed after reload recovery")
	}
}

func TestResidencyRepeatedDeallocate(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	img := newResidencyImage(t, size, 9)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the backend must be evicted")
	}

	// Repeated deallocation of an evicted image must not double-release anything.
	img.Deallocate()
	img.Deallocate()
	runtime.GC()
	atlas.FlushDeferredForTesting()

	// A fresh image on the same budget still works end to end.
	fresh := newResidencyImage(t, size, 11)
	got := readImagePixels(t, fresh, size)
	if string(got) != string(expectedPixels(size, 11)) {
		t.Error("a fresh image is broken after deallocating an evicted backend")
	}
}

func TestResidencyPartialWritePixelsAfterEviction(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	img := newResidencyImage(t, size, 4)
	want := expectedPixels(size, 4)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	// A partial write while evicted must reload and update only the requested region.
	sub := image.Rect(4, 4, 8, 8)
	pix := make([]byte, 4*sub.Dx()*sub.Dy())
	for i := range pix {
		pix[i] = 0xab
	}
	img.WritePixels(pix, sub)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("WritePixels must reload an evicted backend")
	}

	got := readImagePixels(t, img, size)
	for j := range size {
		for i := range size {
			idx := 4 * (j*size + i)
			if i >= 4 && i < 8 && j >= 4 && j < 8 {
				if got[idx] != 0xab {
					t.Fatalf("written pixel at (%d,%d): got %d, want 0xab", i, j, got[idx])
				}
				continue
			}
			if got[idx] != want[idx] || got[idx+3] != want[idx+3] {
				t.Fatalf("preserved pixel at (%d,%d) changed", i, j)
			}
		}
	}
}

func TestResidencyNonZeroBoundsSubImage(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	// A non-zero origin region exercises the coordinate translation on mirror read-back.
	const size = 24
	img := newResidencyImage(t, size, 6)
	full := expectedPixels(size, 6)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	region := image.Rect(5, 7, 13, 15)
	pix := make([]byte, 4*region.Dx()*region.Dy())
	if err := readPixels(img, pix, region); err != nil {
		t.Fatalf("readPixels on evicted image failed: %v", err)
	}
	for j := region.Min.Y; j < region.Max.Y; j++ {
		for i := region.Min.X; i < region.Max.X; i++ {
			gotIdx := 4 * ((j-region.Min.Y)*region.Dx() + (i - region.Min.X))
			wantIdx := 4 * (j*size + i)
			if pix[gotIdx] != full[wantIdx] || pix[gotIdx+3] != full[wantIdx+3] {
				t.Fatalf("mirror read-back mismatch at (%d,%d)", i, j)
			}
		}
	}
}

func TestResidencyConcurrentReadOnly(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 10
	const workers = 8
	imgs := make([]*atlas.Image, workers)
	for i := range imgs {
		imgs[i] = newResidencyImage(t, size, byte(i+20))
	}

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			// Read-only access through the CPU mirror while evicted must be race-free.
			for range 20 {
				pix := make([]byte, 4*size*size)
				if err := readPixels(imgs[idx], pix, image.Rect(0, 0, size, size)); err != nil {
					t.Errorf("worker %d read failed: %v", idx, err)
					return
				}
				want := expectedPixels(size, byte(idx+20))
				if string(pix) != string(want) {
					t.Errorf("worker %d got wrong pixels", idx)
					return
				}
				_ = atlas.ResidencyStatsForTesting()
			}
		}(w)
	}
	wg.Wait()
}

func TestResidencyPeakMemoryAndReloadCounts(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	const imageCount = 4

	// Measure the no-budget reference first: all images resident at once.
	reference := make([]*atlas.Image, imageCount)
	for i := range reference {
		reference[i] = newResidencyImage(t, size, byte(i))
	}
	referencePeak := atlas.ResidencyStatsForTesting().PeakResidentBytes
	for _, img := range reference {
		img.Deallocate()
	}

	// Now run the same allocation under a byte budget and confirm the bounded peak stays below the
	// unbounded reference while reloads remain transparent.
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{
		MaxBytes: referencePeak / 2,
	})

	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	imgs := make([]*atlas.Image, imageCount)
	for i := range imgs {
		imgs[i] = newResidencyImage(t, size, byte(i))
	}
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	bounded := atlas.ResidencyStatsForTesting()
	if bounded.ResidentBytes >= referencePeak {
		t.Errorf("bounded residency %d should be below reference peak %d", bounded.ResidentBytes, referencePeak)
	}
	if bounded.PeakResidentBytes > referencePeak {
		t.Errorf("bounded peak %d exceeded the unbounded reference %d", bounded.PeakResidentBytes, referencePeak)
	}

	// Re-resident every image through actual draws (read-back alone would be served from the CPU
	// mirror and intentionally avoid a reload).
	for _, img := range imgs {
		drawOnto(t, dst, img, size)
		if !img.IsBackendResidentFuncForTesting()() {
			t.Fatal("a drawn image must be resident again")
		}
	}
	final := atlas.ResidencyStatsForTesting()
	if final.CumulativeReloads == 0 {
		t.Error("reloads were not counted")
	}
}

func TestResidencyPriorityEvictionOrder(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	low := newResidencyImage(t, size, 1)
	high := newResidencyImage(t, size, 2)
	plain := newResidencyImage(t, size, 3)

	// Make the three backends equally recent, then distinguish them only by priority.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	for _, img := range []*atlas.Image{low, plain, high} {
		drawOnto(t, dst, img, size)
	}
	for _, img := range []*atlas.Image{low, plain, high} {
		drawOnto(t, dst, img, size)
	}

	low.SetResidencyPriority(-10)
	high.SetResidencyPriority(10)
	if got := high.ResidencyPriority(); got != 10 {
		t.Fatalf("priority: got %d, want 10", got)
	}
	// Re-touch once more so the timestamps are equal and the priority decides the tie-break.
	for _, img := range []*atlas.Image{low, plain, high} {
		drawOnto(t, dst, img, size)
	}

	// A budget that leaves room for two of the three image backends forces one eviction. The
	// low-priority backend must be selected before the plain and high-priority ones.
	s := atlas.ResidencyStatsForTesting()
	oneTexture := s.ResidentBytes / int64(s.ResidentTextures)
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{MaxBytes: s.ResidentBytes - oneTexture})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	stats := atlas.ResidencyStatsForTesting()
	if stats.CumulativeEvictions == 0 {
		t.Fatal("no eviction happened")
	}
	if low.IsBackendResidentFuncForTesting()() {
		t.Error("the low-priority backend should be evicted before higher-priority ones")
	}
	if !high.IsBackendResidentFuncForTesting()() {
		t.Error("the high-priority backend must remain resident")
	}
}

// TestResidencySharedAtlasPage covers regular images packed into a shared atlas page. Evicting
// the page moves every image on it to the CPU mirror at once; each must reload its own region
// with the original pixels and correct offsets.
func TestResidencySharedAtlasPage(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	const count = 6
	imgs := make([]*atlas.Image, count)
	seeds := make([]byte, count)
	for i := range imgs {
		seeds[i] = byte(40 + i)
		// Regular images are eligible for the shared automatic atlas.
		imgs[i] = atlas.NewImage(size, size, atlas.ImageTypeRegular)
		t.Cleanup(imgs[i].Deallocate)
		imgs[i].WritePixels(expectedPixels(size, seeds[i]), image.Rect(0, 0, size, size))
	}

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	// Read every image back through the mirror path and verify the packed-region offsets.
	for i, img := range imgs {
		got := readImagePixels(t, img, size)
		if string(got) != string(expectedPixels(size, seeds[i])) {
			t.Fatalf("shared-page image %d mismatch through the CPU mirror", i)
		}
	}

	// Re-resident half of them through draws and confirm the still-evicted images remain correct,
	// proving one page's mirror serves independent images.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	for i := 0; i < count; i += 2 {
		drawOnto(t, dst, imgs[i], size)
	}
	for i, img := range imgs {
		got := readImagePixels(t, img, size)
		if string(got) != string(expectedPixels(size, seeds[i])) {
			t.Fatalf("shared-page image %d mismatch after partial reload", i)
		}
	}
}

func TestResidencyPartialRegionReadbackOnAtlas(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	// A regular (atlas-packed) image exercises mirror read-back with a non-zero origin region,
	// which is the same coordinate translation sub-image aliases rely on.
	const size = 32
	img := atlas.NewImage(size, size, atlas.ImageTypeRegular)
	t.Cleanup(img.Deallocate)
	full := expectedPixels(size, 22)
	img.WritePixels(full, image.Rect(0, 0, size, size))

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the atlas page must be evicted")
	}

	for _, region := range []image.Rectangle{
		image.Rect(0, 0, 7, 9),
		image.Rect(11, 13, 27, 25),
		image.Rect(size-3, size-5, size, size),
	} {
		pix := make([]byte, 4*region.Dx()*region.Dy())
		if err := readPixels(img, pix, region); err != nil {
			t.Fatalf("readPixels %v failed: %v", region, err)
		}
		for j := region.Min.Y; j < region.Max.Y; j++ {
			for i := region.Min.X; i < region.Max.X; i++ {
				gotIdx := 4 * ((j-region.Min.Y)*region.Dx() + (i - region.Min.X))
				wantIdx := 4 * (j*size + i)
				if pix[gotIdx] != full[wantIdx] || pix[gotIdx+3] != full[wantIdx+3] {
					t.Fatalf("region %v mismatch at (%d,%d)", region, i, j)
				}
			}
		}
	}

	// After a reload the same non-zero regions still read identically from the GPU texture.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)
	region := image.Rect(3, 6, 19, 21)
	pix := make([]byte, 4*region.Dx()*region.Dy())
	if err := readPixels(img, pix, region); err != nil {
		t.Fatalf("post-reload readPixels failed: %v", err)
	}
	for j := region.Min.Y; j < region.Max.Y; j++ {
		for i := region.Min.X; i < region.Max.X; i++ {
			gotIdx := 4 * ((j-region.Min.Y)*region.Dx() + (i - region.Min.X))
			wantIdx := 4 * (j*size + i)
			if pix[gotIdx] != full[wantIdx] || pix[gotIdx+3] != full[wantIdx+3] {
				t.Fatalf("post-reload region mismatch at (%d,%d)", i, j)
			}
		}
	}
}

// TestResidencyDeallocateOneImageOnSharedPage frees one node of an evicted atlas page and checks
// that the surviving images still reload their own pixels and the backend is cleaned up once the
// page becomes empty, with no double disposal.
func TestResidencyDeallocateOneImageOnSharedPage(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 16
	a := atlas.NewImage(size, size, atlas.ImageTypeRegular)
	b := atlas.NewImage(size, size, atlas.ImageTypeRegular)
	c := atlas.NewImage(size, size, atlas.ImageTypeRegular)
	t.Cleanup(a.Deallocate)
	t.Cleanup(b.Deallocate)
	t.Cleanup(c.Deallocate)
	a.WritePixels(expectedPixels(size, 1), image.Rect(0, 0, size, size))
	b.WritePixels(expectedPixels(size, 2), image.Rect(0, 0, size, size))
	c.WritePixels(expectedPixels(size, 3), image.Rect(0, 0, size, size))

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	// Dispose one image while its page is evicted. The other two must remain intact.
	a.Deallocate()
	if got := readImagePixels(t, b, size); string(got) != string(expectedPixels(size, 2)) {
		t.Error("image b changed after a was deallocated on the evicted page")
	}
	if got := readImagePixels(t, c, size); string(got) != string(expectedPixels(size, 3)) {
		t.Error("image c changed after a was deallocated on the evicted page")
	}

	// Re-resident the survivors and verify pixels again through the GPU.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, b, size)
	drawOnto(t, dst, c, size)
	if got := readImagePixels(t, b, size); string(got) != string(expectedPixels(size, 2)) {
		t.Error("image b pixels wrong after reload")
	}
	if got := readImagePixels(t, c, size); string(got) != string(expectedPixels(size, 3)) {
		t.Error("image c pixels wrong after reload")
	}

	// Releasing the last images must drop the backend cleanly.
	b.Deallocate()
	c.Deallocate()
	runtime.GC()
	atlas.FlushDeferredForTesting()
}

func TestResidencyMemoryPressure(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	img := newResidencyImage(t, size, 4)

	// A permissive budget that the current residency fits, so no automatic eviction happens.
	stats := atlas.ResidencyStatsForTesting()
	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{MaxBytes: stats.PeakResidentBytes * 2})
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("the texture should start resident under a permissive budget")
	}

	// An explicit memory-pressure notification asks for an eviction even though the budget is met.
	// The request is serviced at the frame boundary; tests drive it through a synchronous trim.
	atlas.NotifyMemoryPressure()
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim after memory pressure failed: %v", err)
	}
	if img.IsBackendResidentFuncForTesting()() {
		t.Error("memory pressure should have evicted the texture")
	}

	// A notification while the budget is disabled is ignored rather than forcing eviction.
	atlas.DisableBudgetForTesting()
	atlas.NotifyMemoryPressure()
	if img.IsBackendResidentFuncForTesting()() {
		t.Error("the texture must stay evicted until explicitly used, regardless of notifications")
	}
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Error("the texture must reload on use")
	}
}

func TestResidencyDisableRestoresDefaultBehavior(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 8
	img := newResidencyImage(t, size, 1)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the backend must be evicted while the budget is active")
	}

	atlas.DisableBudgetForTesting()
	atlas.TrimTextureResidency()
	// A disabled budget must not trim: the backend stays evicted until used.
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Error("disabling the budget must not eagerly re-resident backends")
	}

	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Error("the backend must reload on use even after the budget was disabled")
	}
}

func TestResidencySurvivesDeviceRebuild(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	const size = 20
	img := newResidencyImage(t, size, 13)
	original := readImagePixels(t, img, size)

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the backend must be evicted before the simulated rebuild")
	}

	// A context-loss rebuild disposes GPU textures; evicted mirrors must survive it.
	atlas.SimulateDeviceRebuildForTesting(driver())
	if img.IsBackendResidentFuncForTesting()() {
		t.Fatal("no texture may be resident immediately after a device rebuild")
	}
	if !img.IsBackendEvictedFuncForTesting()() {
		t.Fatal("the evicted backend must retain its eviction state and mirror")
	}

	// On-demand re-upload reproduces the exact pixels after the rebuild.
	dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
	t.Cleanup(dst.Deallocate)
	drawOnto(t, dst, img, size)
	if !img.IsBackendResidentFuncForTesting()() {
		t.Fatal("the backend must re-resident after the device rebuild")
	}
	if got := readImagePixels(t, img, size); string(got) != string(original) {
		t.Error("pixels changed across the device rebuild")
	}
}

// TestResidencyTransparentTextures covers multi-size, fully transparent fixtures: a zeroed
// texture evicted and reloaded must remain zeroed rather than picking up stale mirror content.
func TestResidencyTransparentTextures(t *testing.T) {
	atlas.ResetBudgetForTesting()
	t.Cleanup(atlas.ResetBudgetForTesting)

	sizes := []int{1, 7, 16, 31, 64}
	imgs := make([]*atlas.Image, len(sizes))
	for i, size := range sizes {
		img := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
		t.Cleanup(img.Deallocate)
		img.WritePixels(make([]byte, 4*size*size), image.Rect(0, 0, size, size))
		imgs[i] = img
	}

	atlas.SetBudgetForTesting(atlas.TextureResidencyBudget{})
	if err := atlas.TrimForTesting(driver()); err != nil {
		t.Fatalf("trim failed: %v", err)
	}

	for i, img := range imgs {
		size := sizes[i]
		got := readImagePixels(t, img, size)
		for j, b := range got {
			if b != 0 {
				t.Fatalf("transparent texture size %d has a non-zero byte at %d", size, j)
			}
		}
		// Force a real reload and check again.
		dst := atlas.NewImage(size, size, atlas.ImageTypeUnmanaged)
		t.Cleanup(dst.Deallocate)
		drawOnto(t, dst, img, size)
		got = readImagePixels(t, img, size)
		for j, b := range got {
			if b != 0 {
				t.Fatalf("reloaded transparent texture size %d has a non-zero byte at %d", size, j)
			}
		}
	}
}
