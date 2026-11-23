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

package atlas

import (
	"errors"
	"fmt"
	"image"
	"slices"
	"sync/atomic"

	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
)

var (
	// errSimulatedResidencyFailure is returned by the test-only failure injection hooks.
	errSimulatedResidencyFailure = errors.New("atlas: simulated residency failure")

	// errBackendNotResident reports that a backend could not be made resident for a drawing command.
	errBackendNotResident = errors.New("atlas: a backend texture could not be made resident")

	// failNextEvictions is decremented by evictBackend to force eviction failures in tests.
	failNextEvictions int

	// failNextReloads is decremented by ensureResident to force reload failures in tests.
	failNextReloads int
)

// residencyCounters holds the monotonically growing counters used to order backends by recency.
type residencyCounters struct {
	// clock is increased every time a resident backend is used.
	clock atomic.Uint64

	// evictionSeq is increased every time a backend is evicted.
	evictionSeq atomic.Uint64

	// reloadSeq is increased every time an evicted backend is made resident again.
	reloadSeq atomic.Uint64
}

func (c *residencyCounters) tick() uint64 {
	return c.clock.Add(1)
}

// ResidencyStats is a point-in-time snapshot of the texture residency budget state.
type ResidencyStats struct {
	// ResidentBytes is the number of bytes currently resident on the GPU.
	ResidentBytes int64

	// PeakResidentBytes is the maximum ResidentBytes observed since the budget was set.
	PeakResidentBytes int64

	// ResidentTextures is the number of backend textures currently resident on the GPU.
	ResidentTextures int

	// PeakResidentTextures is the maximum ResidentTextures observed since the budget was set.
	ResidentTexturesPeak int

	// EvictedTextures is the number of managed backends whose pixels are kept only in CPU memory.
	EvictedTextures int

	// CumulativeEvictions is the total number of evictions performed since the budget was set.
	CumulativeEvictions uint64

	// CumulativeReloads is the total number of evicted backends that were made resident again.
	CumulativeReloads uint64

	// OverBudget reports whether the current residency still exceeds one of the budgets.
	OverBudget bool
}

// TextureResidencyBudget limits the GPU memory used by atlas backend textures.
//
// When a budget is active, unlocked backend textures are evicted to CPU memory in
// least-recently-used order until the residency fits the budget. Evicted textures are
// transparently re-uploaded when an image on them is used again.
//
// A zero field means the corresponding dimension is not limited. A budget whose fields are both
// zero still enforces residency: every unlocked texture is evicted at the next frame boundary and
// re-uploaded only while an image needs it. Call [DisableTextureResidencyBudget] to turn the
// feature off entirely.
type TextureResidencyBudget struct {
	// MaxBytes limits the total number of bytes of resident backend textures.
	// A texture's size is its internal width times its internal height times four bytes.
	MaxBytes int64

	// MaxTextures limits the number of resident backend textures.
	MaxTextures int
}

// residencyBudget is the mutable budget state. All its methods must be called with backendsM held.
type residencyBudget struct {
	maxBytes    int64
	maxTextures int
	enabled     bool

	counters residencyCounters

	// activeLocks are the residency locks currently held. Backends touched while a lock is active
	// cannot be evicted until that lock is released.
	activeLocks []*ResidencyLock

	// evictions counts completed evictions.
	evictions uint64

	// reloads counts completed re-residencies.
	reloads uint64

	// peakBytes and peakTextures record the highest observed residency.
	peakBytes    int64
	peakTextures int

	// frameErr records the first residency failure in the current frame so it can surface at
	// EndFrame instead of being swallowed by drawing calls whose signatures cannot return errors.
	frameErr error

	// trimRequested records an explicit trim requested between frames.
	trimRequested bool

	// trimInProgress prevents nested trims from re-entering eviction.
	trimInProgress bool
}

var theResidencyBudget residencyBudget

// SetTextureResidencyBudget enables residency management with the given budget.
//
// The budget takes effect at the next frame boundary; it never disposes an [Image]. Setting a new
// budget (including the same values) resets the statistics and high-water marks.
func SetTextureResidencyBudget(budget TextureResidencyBudget) {
	backendsM.Lock()
	defer backendsM.Unlock()

	if budget.MaxBytes < 0 || budget.MaxTextures < 0 {
		panic(fmt.Sprintf("atlas: budget values must not be negative: bytes: %d, textures: %d", budget.MaxBytes, budget.MaxTextures))
	}

	theResidencyBudget = residencyBudget{
		maxBytes:    budget.MaxBytes,
		maxTextures: budget.MaxTextures,
		enabled:     true,
	}
}

// DisableTextureResidencyBudget turns residency management off. Evicted backends stay evicted
// until their images are used again, at which point they are re-uploaded as usual.
func DisableTextureResidencyBudget() {
	backendsM.Lock()
	defer backendsM.Unlock()

	theResidencyBudget.enabled = false
	theResidencyBudget.maxBytes = 0
	theResidencyBudget.maxTextures = 0
	theResidencyBudget.activeLocks = nil
	theResidencyBudget.trimRequested = false
}

// ResetTextureResidencyBudgetForTesting disables the budget and clears all its state.
func ResetTextureResidencyBudgetForTesting() {
	backendsM.Lock()
	defer backendsM.Unlock()

	theResidencyBudget = residencyBudget{}
	failNextEvictions = 0
	failNextReloads = 0
}

// TrimTextureResidency requests an explicit eviction pass, executed at the next frame boundary.
//
// This is useful after a scene change: textures no longer referenced by the working set can be
// moved to CPU memory immediately instead of waiting for the budget to be exceeded naturally.
func TrimTextureResidency() {
	backendsM.Lock()
	defer backendsM.Unlock()

	if theResidencyBudget.enabled {
		theResidencyBudget.trimRequested = true
	}
}

// NotifyMemoryPressure asks the residency manager to release as much GPU memory as possible at
// the next frame boundary. It is equivalent to an explicit trim while a budget is active.
func NotifyMemoryPressure() {
	TrimTextureResidency()
}

// backendTextureBytes returns the GPU byte size of a backend texture.
//
// The internal size, rather than the logical atlas size, is used so that the reported peak matches
// the memory actually allocated by the graphics driver.
func backendTextureBytes(b *backend) int64 {
	if b == nil || b.backendImage == nil {
		return 0
	}
	w, h := b.backendImage.InternalSize()
	return 4 * int64(w) * int64(h)
}

// residentUsage reports the current residency totals over all backends.
func (r *residencyBudget) residentUsage() (bytes int64, textures int) {
	for _, b := range theBackends {
		if b.backendImage != nil {
			bytes += backendTextureBytes(b)
			textures++
		}
	}
	return bytes, textures
}

// recordPeak updates the high-water marks from the current residency.
func (r *residencyBudget) recordPeak() {
	bytes, textures := r.residentUsage()
	r.peakBytes = max(r.peakBytes, bytes)
	r.peakTextures = max(r.peakTextures, textures)
}

// overBudget reports whether the residency exceeds a configured limit. An empty residency is
// never over budget, even when a limit is zero.
func (r *residencyBudget) overBudget(bytes int64, textures int) bool {
	if !r.enabled || textures == 0 {
		return false
	}
	if r.maxBytes > 0 && bytes > r.maxBytes {
		return true
	}
	if r.maxTextures > 0 && textures > r.maxTextures {
		return true
	}
	return false
}

// evictable reports whether a backend can be evicted under the budget.
//
// Screen backends and backends without a GPU texture are never evictable. A backend that is
// currently protected by a residency lock is not evictable either.
func (r *residencyBudget) evictable(b *backend) bool {
	if !r.enabled {
		return false
	}
	if b.screen || b.backendImage == nil {
		return false
	}
	if b.residencyLocks != 0 {
		return false
	}
	return true
}

// touch records that a backend was used. Resident backends get a fresh access time, and while a
// residency lock is active the backend is pinned until that lock is released.
func (r *residencyBudget) touch(b *backend) {
	if b == nil {
		return
	}
	if b.backendImage != nil {
		b.lastUsed = r.counters.tick()
	}
	if len(r.activeLocks) == 0 {
		return
	}
	for _, l := range r.activeLocks {
		if slices.Contains(l.backends, b) {
			continue
		}
		l.backends = append(l.backends, b)
		b.residencyLocks++
	}
}

// evictionCandidates returns evictable backends in eviction order: the least recently used
// backend has the lowest keep-priority and is evicted first. When two backends have the same
// access time, the lower residency priority is evicted first, then the lower backend id for
// determinism.
func (r *residencyBudget) evictionCandidates(failed map[*backend]bool) []*backend {
	candidates := make([]*backend, 0, len(theBackends))
	for _, b := range theBackends {
		if r.evictable(b) && !failed[b] {
			candidates = append(candidates, b)
		}
	}
	slices.SortFunc(candidates, func(a, b *backend) int {
		// Wrap-safe LRU ordering: the difference is computed in wrapping arithmetic, and a smaller
		// wrapped difference means a was touched longer ago than b.
		if a.lastUsed != b.lastUsed {
			if d := int64(a.lastUsed - b.lastUsed); d != 0 {
				if d < 0 {
					return -1
				}
				return 1
			}
		}
		if a.residencyPriority != b.residencyPriority {
			return a.residencyPriority - b.residencyPriority
		}
		return int(a.id - b.id)
	})
	return candidates
}

// applyPriorityToBackend raises the backend's effective keep-priority to the image's requested
// value when the image joins the backend.
func (r *residencyBudget) applyPriorityToBackend(i *imageImpl) {
	if i.backend != nil && i.residencyPriority > i.backend.residencyPriority {
		i.backend.residencyPriority = i.residencyPriority
	}
}

// SetResidencyPriority assigns a keep-priority to the image's backend.
//
// A higher priority makes the backend less likely to be evicted when several backends are equally
// recently used. On a shared atlas page the effective priority is the maximum requested by the
// images on it. The default priority is zero. Setting a priority never evicts or re-uploads by
// itself; the next trim observes it.
func (i *Image) SetResidencyPriority(priority int) {
	backendsM.Lock()
	defer backendsM.Unlock()

	i.residencyPriority = priority
	if i.backend != nil && priority > i.backend.residencyPriority {
		i.backend.residencyPriority = priority
	}
}

// ResidencyPriority returns the image's keep-priority.
func (i *Image) ResidencyPriority() int {
	backendsM.Lock()
	defer backendsM.Unlock()
	return i.residencyPriority
}

// mirrorClear zeros a rectangle in an evicted backend's CPU mirror, keeping it consistent with
// the explicit clears performed on resident textures.
func mirrorClear(pixels *graphics.ManagedBytes, mirrorRegion, clearRegion image.Rectangle) {
	if pixels == nil || !clearRegion.Overlaps(mirrorRegion) {
		return
	}
	c := clearRegion.Intersect(mirrorRegion).Sub(mirrorRegion.Min)
	bs := pixels.Bytes()
	for y := c.Min.Y; y < c.Max.Y; y++ {
		start := 4 * (y*mirrorRegion.Dx() + c.Min.X)
		clear(bs[start : start+4*c.Dx()])
	}
}

// evictBackend saves the backend's pixels into CPU memory and disposes its GPU texture.
//
// On success the backend stays registered and its images keep their packing nodes; a later
// ensureResident restores the exact pixels. On failure the backend is left fully resident and its
// error is returned, so callers never observe a dangling texture handle.
func (r *residencyBudget) evictBackend(b *backend, driver graphicsdriver.Graphics) error {
	if b.backendImage == nil {
		return nil
	}
	if failNextEvictions > 0 {
		failNextEvictions--
		return errSimulatedResidencyFailure
	}

	// Read back the whole allocated region. For an atlas page this is the union of the allocated
	// nodes; for an isolated image it is its full rectangle.
	var region image.Rectangle
	if b.page != nil {
		region = b.page.AllocatedRegion()
	} else {
		region = image.Rect(0, 0, b.width, b.height)
	}
	if region.Empty() {
		// Nothing meaningful to preserve: just drop the texture. A re-resident backend starts clear.
		b.disposeGPUTexture()
		b.evicted = true
		b.evictionSequence = r.counters.evictionSeq.Add(1)
		r.evictions++
		return nil
	}

	pixels := graphics.NewManagedBytes(4*region.Dx()*region.Dy(), func(bs []byte) {})
	args := []graphicsdriver.PixelsArgs{
		{
			Pixels: pixels.Bytes(),
			Region: region,
		},
	}
	if err := b.backendImage.ReadPixels(driver, args); err != nil {
		// The read-back failed. Release the scratch buffer and keep the texture resident so the
		// backend remains usable. There is no retry loop: the next trim tries again.
		pixels.Release()
		return err
	}

	// Dispose the GPU texture only after the pixels have been captured successfully.
	b.disposeGPUTexture()

	// Replace any stale mirror; the old one cannot be valid after a successful read-back.
	if b.residentPixels != nil {
		b.residentPixels.Release()
	}
	b.residentPixels = pixels
	b.residentRegion = region
	b.evicted = true
	b.evictionSequence = r.counters.evictionSeq.Add(1)
	r.evictions++
	return nil
}

// ensureResident makes an evicted backend resident again by re-uploading its saved pixels.
//
// If the backend is already resident this is a no-op. On a simulated or real upload failure the
// backend stays evicted with its mirror intact, and the error is returned without any retry.
func (r *residencyBudget) ensureResident(b *backend) error {
	if b == nil || b.backendImage != nil || !b.evicted {
		return nil
	}
	if failNextReloads > 0 {
		failNextReloads--
		return errSimulatedResidencyFailure
	}

	img := graphicscommand.NewImage(b.width, b.height, b.screen, "")
	if b.residentPixels != nil {
		// Clear first: the saved region might be smaller than the texture for a partially allocated
		// page, and the rest must stay transparent just like a fresh atlas.
		if b.residentRegion != image.Rect(0, 0, b.width, b.height) {
			clearImage(img, image.Rect(0, 0, b.width, b.height))
		}
		img.WritePixels(b.residentPixels, b.residentRegion)
	} else {
		clearImage(img, image.Rect(0, 0, b.width, b.height))
	}

	b.backendImage = img
	b.evicted = false
	if b.residentPixels != nil {
		// The queued WritePixels command still references the buffer, so it cannot be released here.
		// Drop the mirror reference; the ManagedBytes cleanup returns the underlying bytes to the
		// pool once the command queue no longer needs them.
		b.residentPixels = nil
		b.residentRegion = image.Rectangle{}
	}
	r.reloads++
	b.reloadSequence = r.counters.reloadSeq.Add(1)
	r.recordPeak()
	r.touch(b)
	return nil
}

// trim evicts least-recently-used backends until the residency fits the budget or no evictable
// backend remains.
//
// When force is true (an explicit trim or memory-pressure notification), eviction continues even
// if the budget is already met, down to zero unlocked textures. Locked backends are always
// skipped. The returned error is the first eviction failure, if any; trimming continues after a
// failure so a single problematic texture cannot block the others. A backend that fails eviction
// is not retried within the same trim pass, which bounds the work even when every read-back fails.
func (r *residencyBudget) trim(driver graphicsdriver.Graphics, force bool) error {
	if !r.enabled || r.trimInProgress {
		return nil
	}
	r.trimInProgress = true
	defer func() {
		r.trimInProgress = false
	}()

	failed := map[*backend]bool{}
	var firstErr error
	for {
		bytes, textures := r.residentUsage()

		if !force {
			// Automatic trimming stops as soon as the residency fits every configured limit.
			// With a zero budget every unlocked backend is evictable.
			if r.maxBytes > 0 || r.maxTextures > 0 {
				if !r.overBudget(bytes, textures) {
					break
				}
			} else if textures == 0 {
				break
			}
		} else if textures == 0 {
			break
		}

		candidates := r.evictionCandidates(failed)
		if len(candidates) == 0 {
			// The overshoot cannot be reduced (a single texture bigger than the budget, everything
			// locked, or every candidate failed). Accept the current residency instead of looping.
			break
		}

		victim := candidates[0]
		if err := r.evictBackend(victim, driver); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			failed[victim] = true
			continue
		}
	}
	r.recordPeak()
	return firstErr
}

// recordFrameError keeps the first failure of a frame.
func (r *residencyBudget) recordFrameError(err error) {
	if err != nil && r.frameErr == nil {
		r.frameErr = err
	}
}

// consumeFrameError returns and clears the frame's first residency failure.
func (r *residencyBudget) consumeFrameError() error {
	err := r.frameErr
	r.frameErr = nil
	return err
}

// processFrameBoundaryTasks runs the trim requests that must be executed with a graphics driver.
// An explicit trim always runs; otherwise an automatic trim restores the budget when needed.
func (r *residencyBudget) processFrameBoundaryTasks(driver graphicsdriver.Graphics) error {
	if !r.enabled {
		r.trimRequested = false
		return nil
	}
	requested := r.trimRequested
	r.trimRequested = false
	if !requested {
		bytes, textures := r.residentUsage()
		if r.maxBytes == 0 && r.maxTextures == 0 {
			// A zero budget forces every unlocked texture out, so the trim is forced.
			if textures == 0 {
				return nil
			}
		} else if !r.overBudget(bytes, textures) {
			// A non-zero budget only trims when exceeded, so it is not forced.
			return nil
		}
	}
	return r.trim(driver, requested || (r.maxBytes == 0 && r.maxTextures == 0))
}

// TrimToBudgetForTesting runs a trim synchronously, honoring any pending explicit trim request,
// and returns its first error.
func TrimToBudgetForTesting(driver graphicsdriver.Graphics) error {
	backendsM.Lock()
	defer backendsM.Unlock()
	return theResidencyBudget.processFrameBoundaryTasks(driver)
}

// ResidencyStatsSnapshot returns a snapshot of the budget state.
func ResidencyStatsSnapshot() ResidencyStats {
	backendsM.Lock()
	defer backendsM.Unlock()

	theResidencyBudget.recordPeak()
	bytes, textures := theResidencyBudget.residentUsage()
	evicted := 0
	for _, b := range theBackends {
		if b.evicted {
			evicted++
		}
	}
	return ResidencyStats{
		ResidentBytes:        bytes,
		PeakResidentBytes:    theResidencyBudget.peakBytes,
		ResidentTextures:     textures,
		ResidentTexturesPeak: theResidencyBudget.peakTextures,
		EvictedTextures:      evicted,
		CumulativeEvictions:  theResidencyBudget.evictions,
		CumulativeReloads:    theResidencyBudget.reloads,
		OverBudget:           theResidencyBudget.overBudget(bytes, textures),
	}
}

// ResidencyLock keeps the backends touched while it is active resident on the GPU until it is
// released. It is safe to release concurrently or more than once.
type ResidencyLock struct {
	released atomic.Bool

	// backends are the backends pinned by this lock. Only mutated under backendsM.
	backends []*backend
}

// LockResidency starts a residency lock. Images drawn while the lock is held cannot be evicted
// until [ResidencyLock.Release] is called.
func LockResidency() *ResidencyLock {
	backendsM.Lock()
	defer backendsM.Unlock()

	l := &ResidencyLock{}
	theResidencyBudget.activeLocks = append(theResidencyBudget.activeLocks, l)
	return l
}

// Release ends the residency lock and unpins the backends it protected. Releasing an already
// released lock is a no-op so deferred releases stay safe on error paths.
func (l *ResidencyLock) Release() {
	if l == nil {
		return
	}
	if !l.released.CompareAndSwap(false, true) {
		return
	}

	backendsM.Lock()
	defer backendsM.Unlock()

	for idx, active := range theResidencyBudget.activeLocks {
		if active != l {
			continue
		}
		theResidencyBudget.activeLocks = append(theResidencyBudget.activeLocks[:idx], theResidencyBudget.activeLocks[idx+1:]...)
		break
	}
	for _, b := range l.backends {
		if b.residencyLocks > 0 {
			b.residencyLocks--
		}
	}
	l.backends = nil
}

// LockedBackendsForTesting reports how many backends currently hold residency locks.
func LockedBackendsForTesting() int {
	backendsM.Lock()
	defer backendsM.Unlock()

	n := 0
	for _, b := range theBackends {
		if b.residencyLocks != 0 {
			n++
		}
	}
	return n
}
