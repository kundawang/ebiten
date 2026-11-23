// Copyright 2018 The Ebiten Authors
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
	"math"
	"math/bits"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/hajimehoshi/ebiten/v2/internal/debug"
	"github.com/hajimehoshi/ebiten/v2/internal/graphics"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicscommand"
	"github.com/hajimehoshi/ebiten/v2/internal/graphicsdriver"
	"github.com/hajimehoshi/ebiten/v2/internal/packing"
)

var (
	minSourceSize      = 0
	minDestinationSize = 0
	maxSize            = 0
)

// maxPageSizeOn32Bit is the maximum width and height of an atlas page on 32-bit architectures.
//
// Textures can be in the process's address space, e.g. with a software rasterizer. Extending a page allocates
// a larger texture while the current one is alive: extending an 8192x4096 page to 8192x8192 needs a contiguous
// 256 MiB block in addition to the current 128 MiB. A 32-bit address space fragmented by other allocations often
// has no free block that large, while a 4096x4096 page needs only 64 MiB (#3882).
const maxPageSizeOn32Bit = 4096

// maxPageSize returns the maximum width and height of an atlas page.
func maxPageSize() int {
	if bits.UintSize == 32 {
		return min(maxSize, maxPageSizeOn32Bit)
	}
	return maxSize
}

func appendDeferred(f func()) {
	deferredM.Lock()
	defer deferredM.Unlock()
	deferred = append(deferred, f)
}

func flushDeferred() {
	deferredM.Lock()
	fs := deferred
	deferred = nil
	deferredM.Unlock()

	for _, f := range fs {
		f()
	}
}

// baseCountToPutOnSourceBackend represents the base time duration when the image can be put onto an atlas.
// Actual time duration is increased in an exponential way for each usage as a rendering target.
const baseCountToPutOnSourceBackend = 10

func putImagesOnSourceBackend() {
	// The counter usedAsDestinationCount is updated at most once per frame (#2676).
	imagesUsedAsDestination.forEach(func(i *Image) {
		// i.backend can be nil after deallocate is called.
		if i.backend == nil {
			i.usedAsDestinationCount = 0
			return
		}
		// This counter is not updated when the backend is created in this frame.
		if !i.backendCreatedInThisFrame && i.usedAsDestinationCount < math.MaxInt {
			i.usedAsDestinationCount++
		}
		i.backendCreatedInThisFrame = false
	})
	imagesUsedAsDestination.clear()

	imagesToPutOnSourceBackend.forEach(func(i *Image) {
		// i.backend can be nil after deallocate is called.
		if i.backend == nil {
			i.usedAsSourceCount = 0
			return
		}
		if i.usedAsSourceCount < math.MaxInt {
			i.usedAsSourceCount++
		}
		if i.usedAsSourceCount >= baseCountToPutOnSourceBackend*(1<<uint(min(i.usedAsDestinationCount, 31))) {
			i.putOnSourceBackend()
			i.usedAsSourceCount = 0
		}
	})
	imagesToPutOnSourceBackend.clear()
}

// backend is a big texture atlas that can have multiple images.
// backend is a texture in GPU.
type backend struct {
	width  int
	height int

	// id uniquely identifies a backend for deterministic residency ordering.
	id uint64

	// backendImage is an implementation backendImage stored in the GPU.
	// backendImage is nil while the backend is evicted by the texture residency budget.
	backendImage *graphicscommand.Image

	// page is an atlas map. Each part is called a node.
	// If page is nil, the backend's image is isolated and not on an atlas.
	page *packing.Page

	// screen indicates whether this backend is used as a screen image or not.
	screen bool

	// source reports whether this backend is mainly used a rendering source, but this is not 100%.
	//
	// If a non-source (destination) image is used as a source many times,
	// the image's backend might be turned into a source backend to optimize draw calls.
	//
	// source is always false when the image type is not ImageTypeRegular.
	source bool

	// sourceInThisFrame reports whether this backend is used as a source in this frame.
	// sourceInThisFrame is reset every frame.
	sourceInThisFrame bool

	// restoreInfo is information for restoration from a context loss.
	restoreInfo restoreInfo

	// The following fields manage residency when a TextureResidencyBudget is active.
	// All are accessed only while backendsM is held.

	// evicted reports whether the GPU texture was disposed and the pixels live in residentPixels.
	evicted bool

	// residentPixels and residentRegion preserve the backend's pixels while it is evicted.
	residentPixels *graphics.ManagedBytes
	residentRegion image.Rectangle

	// lastUsed is the access clock at the most recent use while resident.
	lastUsed uint64

	// residencyLocks counts active ResidencyLocks pinning this backend.
	residencyLocks int

	// evictionSequence and reloadSequence record the budget counters at the last state change.
	evictionSequence uint64
	reloadSequence   uint64

	// residencyPriority is the keep-priority used by the budget eviction order. A higher value
	// keeps the backend resident longer when recency is otherwise equal. It is the maximum priority
	// requested by the images sharing this backend.
	residencyPriority int
}

// nextBackendID hands out backend identifiers.
var nextBackendID atomic.Uint64

// disposeGPUTexture enqueues disposal of the backend's GPU texture and clears the context-loss
// restore info. It is safe on both resident and evicted backends and never double-disposes.
func (b *backend) disposeGPUTexture() {
	if b.backendImage == nil {
		return
	}
	b.backendImage.Dispose()
	b.backendImage = nil
	b.restoreInfo = restoreInfo{}
}

type restoreInfo struct {
	valid  bool
	screen bool
	pixels *graphics.ManagedBytes
	region image.Rectangle
}

func (b *backend) tryAlloc(width, height int) (*packing.Node, bool) {
	if b.page == nil {
		return nil, false
	}
	// An evicted backend has no GPU texture and must not receive new allocations until its images
	// make it resident again, which happens through a different backend.
	if b.evicted || b.backendImage == nil {
		return nil, false
	}
	n := b.page.Alloc(width, height)
	if n == nil {
		// The page can't be extended anymore. Return as failure.
		return nil, false
	}

	pageW, pageH := b.page.Size()
	if w, h := b.backendImage.InternalSize(); pageW <= w && pageH <= h {
		return n, true
	}

	// Extend the image.
	newImg := graphicscommand.NewImage(pageW, pageH, b.screen, "")
	clearImage(newImg, image.Rect(0, 0, pageW, pageH))
	src := b.backendImage
	srcs := [graphics.ShaderSrcImageCount]*graphicscommand.Image{src}
	sw, sh := src.InternalSize()
	vs := make([]float32, 4*graphics.VertexFloatCount)
	graphics.QuadVerticesFromDstAndSrc(vs, 0, 0, float32(sw), float32(sh), 0, 0, float32(sw), float32(sh), 1, 1, 1, 1)
	is := graphics.QuadIndices()
	dr := image.Rect(0, 0, sw, sh)
	newImg.DrawTriangles(srcs, vs, is, graphicsdriver.BlendCopy, dr, [graphics.ShaderSrcImageCount]image.Rectangle{dr}, NearestFilterShader.ensureShader(), nil)
	src.Dispose()
	b.backendImage = newImg
	b.width = pageW
	b.height = pageH

	return n, true
}

func (b *backend) clear(region image.Rectangle) {
	if b.backendImage == nil {
		// While evicted, keep the CPU mirror consistent with the explicit clears performed on
		// resident textures so a later re-upload reproduces the same pixels.
		mirrorClear(b.residentPixels, b.residentRegion, region)
		return
	}
	clearImage(b.backendImage, region)
}

func clearImage(dst *graphicscommand.Image, region image.Rectangle) {
	vs := make([]float32, 4*graphics.VertexFloatCount)
	graphics.QuadVerticesFromDstAndSrc(vs, float32(region.Min.X), float32(region.Min.Y), float32(region.Max.X), float32(region.Max.Y), 0, 0, 0, 0, 0, 0, 0, 0)
	is := graphics.QuadIndices()
	dst.DrawTriangles([graphics.ShaderSrcImageCount]*graphicscommand.Image{}, vs, is, graphicsdriver.BlendClear, region, [graphics.ShaderSrcImageCount]image.Rectangle{}, clearShader.ensureShader(), nil)
}

var (
	// backendsM is a mutex for critical sections of the backend and packing.Node objects.
	backendsM sync.Mutex

	// inFrame indicates whether the current state is in between BeginFrame and EndFrame or not.
	// If inFrame is false, function calls on an image should be deferred until the next BeginFrame.
	inFrame bool

	// theDeferredReadPixels is the pixel read-backs requested between frames and not queued yet.
	theDeferredReadPixels []*deferredReadPixels

	initOnce  sync.Once
	inited    atomic.Bool
	suspended atomic.Bool

	// theBackends is a set of atlases.
	theBackends []*backend

	imagesToPutOnSourceBackend imageSmallSet

	imagesUsedAsDestination imageSmallSet

	deferred []func()

	// deferredM is a mutex for the slice operations. This must not be used for other usages.
	deferredM sync.Mutex
)

// ImageType represents the type of an image.
type ImageType int

const (
	// ImageTypeRegular is a regular image, that can be on a big texture atlas (backend).
	ImageTypeRegular ImageType = iota

	// ImageTypeScreen is a screen image that is not on an atlas.
	// A screen image is also unmanaged.
	ImageTypeScreen

	// ImageTypeUnmanaged is an unmanaged image that is not on an atlas.
	ImageTypeUnmanaged
)

// Image is a rectangle pixel set that might be on an atlas.
type Image struct {
	*imageImpl

	cleanup runtime.Cleanup
}

type imageImpl struct {
	width     int
	height    int
	imageType ImageType

	backend                   *backend
	backendCreatedInThisFrame bool

	node *packing.Node

	// usedAsSourceCount represents how long the image is used as a rendering source and kept not modified with
	// DrawTriangles.
	// In the current implementation, if an image is being modified by DrawTriangles, the image is separated from
	// a backend image for an atlas by ensureIsolatedFromSource.
	//
	// The type is int64 instead of int to avoid overflow when comparing the limitation.
	//
	// usedAsSourceCount is increased if the image is used as a rendering source, or set to 0 if the image is
	// modified. WritePixels also resets this value.
	usedAsSourceCount int64

	// usedAsDestinationCount represents how many times an image is used as a rendering destination at DrawTriangles.
	// usedAsDestinationCount affects the calculation when to put the image onto a texture atlas again.
	//
	// usedAsDestinationCount is not reset while the image is alive.
	usedAsDestinationCount int

	// residencyPriority is the image's requested keep-priority for budget eviction. The effective
	// backend priority is the maximum over the images sharing it.
	residencyPriority int
}

// moveTo moves its content to the given image dst.
// After moveTo is called, the image i is no longer available.
//
// moveTo is similar to C++'s move semantics.
func (i *Image) moveTo(dst *Image) {
	// Backend migration must preserve the destination image's usage history.
	usedAsDestinationCount := dst.usedAsDestinationCount

	dst.deallocateImpl()
	dst.cleanup.Stop()

	impl := *i.imageImpl
	impl.usedAsDestinationCount = usedAsDestinationCount
	dst.imageImpl = &impl
	if dst.backend != nil {
		dst.cleanup = runtime.AddCleanup(dst, (*imageImpl).cleanup, dst.imageImpl)
	}

	// i is no longer available but the finalizer must not be called
	// since i and dst share the same backend and the same node.
	i.cleanup.Stop()
}

func (i *imageImpl) isOnAtlas() bool {
	return i.node != nil
}

func (i *Image) isOnSourceBackend() bool {
	if i.backend == nil {
		return false
	}
	return i.backend.source
}

func (i *Image) resetUsedAsSourceCount() {
	i.usedAsSourceCount = 0
	imagesToPutOnSourceBackend.remove(i)
}

func (i *imageImpl) paddingSize() int {
	if i.imageType == ImageTypeRegular {
		return 1
	}
	return 0
}

func (i *Image) ensureIsolatedFromSource(backends []*backend) {
	i.resetUsedAsSourceCount()

	// imagesUsedAsDestination affects the counter usedAsDestination.
	// The larger this counter is, the harder it is for the image to be transferred to the source backend.
	imagesUsedAsDestination.add(i)

	if i.backend == nil {
		// `sourceInThisFrame` of `backends` should be true, so `backends` should be in `bs`.
		var bs []*backend
		for _, b := range theBackends {
			if b.sourceInThisFrame {
				bs = append(bs, b)
			}
		}
		i.allocate(bs, false)
		i.backendCreatedInThisFrame = true
		return
	}

	if !i.isOnAtlas() {
		return
	}

	// Check if i has the same backend as the given backends.
	if !slices.ContainsFunc(backends, func(b *backend) bool {
		return i.backend == b
	}) {
		return
	}

	newI := NewImage(i.width, i.height, i.imageType)

	// Call allocate explicitly in order to have an isolated backend from the specified backends.
	// `sourceInThisFrame` of `backends` should be true, so `backends` should be in `bs`.
	bs := []*backend{i.backend}
	for _, b := range theBackends {
		if b.sourceInThisFrame {
			bs = append(bs, b)
		}
	}
	newI.allocate(bs, false)

	w, h := float32(i.width), float32(i.height)
	vs := make([]float32, 4*graphics.VertexFloatCount)
	graphics.QuadVerticesFromDstAndSrc(vs, 0, 0, w, h, 0, 0, w, h, 1, 1, 1, 1)
	is := graphics.QuadIndices()
	dr := image.Rect(0, 0, i.width, i.height)
	sr := image.Rect(0, 0, i.width, i.height)

	newI.drawTriangles([graphics.ShaderSrcImageCount]*Image{i}, vs, is, graphicsdriver.BlendCopy, dr, [graphics.ShaderSrcImageCount]image.Rectangle{sr}, NearestFilterShader, nil)
	newI.moveTo(i)
}

func (i *Image) putOnSourceBackend() {
	if i.backend == nil {
		i.allocate(nil, true)
		return
	}

	if i.isOnSourceBackend() {
		return
	}

	if !i.canBePutOnAtlas() {
		panic("atlas: putOnSourceBackend cannot be called on an image that cannot be on an atlas")
	}

	if i.imageType != ImageTypeRegular {
		panic(fmt.Sprintf("atlas: the image type must be ImageTypeRegular but %d", i.imageType))
	}

	newI := NewImage(i.width, i.height, ImageTypeRegular)
	newI.allocate(nil, true)

	w, h := float32(i.width), float32(i.height)
	vs := make([]float32, 4*graphics.VertexFloatCount)
	graphics.QuadVerticesFromDstAndSrc(vs, 0, 0, w, h, 0, 0, w, h, 1, 1, 1, 1)
	is := graphics.QuadIndices()
	dr := image.Rect(0, 0, i.width, i.height)
	sr := image.Rect(0, 0, i.width, i.height)
	newI.drawTriangles([graphics.ShaderSrcImageCount]*Image{i}, vs, is, graphicsdriver.BlendCopy, dr, [graphics.ShaderSrcImageCount]image.Rectangle{sr}, NearestFilterShader, nil)

	newI.moveTo(i)
	i.usedAsSourceCount = 0

	if !i.isOnSourceBackend() {
		panic("atlas: image must be on a source backend")
	}
}

func (i *imageImpl) regionWithPadding() image.Rectangle {
	if i.backend == nil {
		panic("atlas: backend must not be nil: not allocated yet")
	}
	if !i.isOnAtlas() {
		return image.Rect(0, 0, i.width+i.paddingSize(), i.height+i.paddingSize())
	}
	// A node at the right or the bottom edge of the page sticks out of the backend image by its padding.
	return i.node.Region().Intersect(image.Rect(0, 0, i.backend.width, i.backend.height))
}

// ensureResident makes the image's backend resident if it was evicted by the residency budget.
// Failures are recorded on the budget and reported at EndFrame; the caller must check
// i.backend.backendImage before using the texture.
func (i *Image) ensureResident() {
	if i.backend == nil || i.backend.backendImage != nil {
		return
	}
	if err := theResidencyBudget.ensureResident(i.backend); err != nil {
		theResidencyBudget.recordFrameError(err)
	}
}

// DrawTriangles draws triangles with the given image.
//
// The vertex floats are:
//
//	0: Destination X in pixels
//	1: Destination Y in pixels
//	2: Source X in pixels (the upper-left is (0, 0))
//	3: Source Y in pixels
//	4: Color R [0.0-1.0]
//	5: Color G
//	6: Color B
//	7: Color A
//	8: Custom0
//	9: Custom1
//	10: Custom2
//	11: Custom3
func (i *Image) DrawTriangles(srcs [graphics.ShaderSrcImageCount]*Image, vertices []float32, indices []uint32, blend graphicsdriver.Blend, dstRegion image.Rectangle, srcRegions [graphics.ShaderSrcImageCount]image.Rectangle, shader *Shader, uniforms []uint32) {
	// Check this before deferring the draw, so that the panic is at the caller.
	for _, src := range srcs {
		if src == i {
			panic("atlas: Image.DrawTriangles: source must be different from the receiver")
		}
	}

	backendsM.Lock()
	defer backendsM.Unlock()

	if !inFrame {
		vs := make([]float32, len(vertices))
		copy(vs, vertices)
		is := make([]uint32, len(indices))
		copy(is, indices)
		us := make([]uint32, len(uniforms))
		copy(us, uniforms)

		appendDeferred(func() {
			i.drawTriangles(srcs, vs, is, blend, dstRegion, srcRegions, shader, us)
		})
		return
	}

	i.drawTriangles(srcs, vertices, indices, blend, dstRegion, srcRegions, shader, uniforms)
}

func (i *Image) drawTriangles(srcs [graphics.ShaderSrcImageCount]*Image, vertices []float32, indices []uint32, blend graphicsdriver.Blend, dstRegion image.Rectangle, srcRegions [graphics.ShaderSrcImageCount]image.Rectangle, shader *Shader, uniforms []uint32) {
	backends := make([]*backend, 0, len(srcs))
	for _, src := range srcs {
		if src == nil {
			continue
		}
		if src.backend == nil {
			// It is possible to specify i.backend as a forbidden backend, but this might prevent a good allocation for a source image.
			// If the backend becomes the same as i's, i's backend will be changed at ensureIsolatedFromSource.
			src.allocate(nil, src.imageType == ImageTypeRegular)
		}
		src.ensureResident()
		if src.backend.backendImage == nil {
			// Re-upload failed. Record the failure and skip this draw rather than enqueue a command
			// referencing a disposed texture.
			theResidencyBudget.recordFrameError(errBackendNotResident)
			return
		}
		backends = append(backends, src.backend)
		src.backend.sourceInThisFrame = true
		theResidencyBudget.touch(src.backend)
	}

	i.ensureIsolatedFromSource(backends)
	i.ensureResident()
	if i.backend == nil || i.backend.backendImage == nil {
		theResidencyBudget.recordFrameError(errBackendNotResident)
		return
	}
	theResidencyBudget.touch(i.backend)

	for _, src := range srcs {
		// Compare i and source images after ensuring i is not on an atlas, or
		// i and a source image might share the same atlas even though i != src.
		if src != nil && i.backend.backendImage == src.backend.backendImage {
			panic("atlas: Image.DrawTriangles: source must be different from the receiver")
		}
	}

	r := i.regionWithPadding()
	// TODO: Check if dstRegion does not to violate the region.
	dstRegion = dstRegion.Add(r.Min)

	dx, dy := float32(r.Min.X), float32(r.Min.Y)

	var oxf, oyf float32
	if srcs[0] != nil {
		r := srcs[0].regionWithPadding()
		oxf, oyf = float32(r.Min.X), float32(r.Min.Y)
		if dx != 0 || dy != 0 || oxf != 0 || oyf != 0 {
			n := len(vertices)
			for i := 0; i < n; i += graphics.VertexFloatCount {
				vertices[i] += dx
				vertices[i+1] += dy
				vertices[i+2] += oxf
				vertices[i+3] += oyf
			}
		}
	} else if dx != 0 || dy != 0 {
		n := len(vertices)
		for i := 0; i < n; i += graphics.VertexFloatCount {
			vertices[i] += dx
			vertices[i+1] += dy
		}
	}

	var imgs [graphics.ShaderSrcImageCount]*graphicscommand.Image
	for i, src := range srcs {
		if src == nil {
			continue
		}

		// A source region can be deliberately empty when this is not needed in order to avoid unexpected
		// performance issue (#1293).
		// TODO: This should no longer be needed but is kept just in case. Remove this later.
		if !srcRegions[i].Empty() {
			r := src.regionWithPadding()
			srcRegions[i] = srcRegions[i].Add(r.Min)
		}
		imgs[i] = src.backend.backendImage
		if !src.isOnSourceBackend() && src.canBePutOnAtlas() {
			// src might already registered, but assigning it again is not harmful.
			imagesToPutOnSourceBackend.add(src)
		}
	}

	i.backend.backendImage.DrawTriangles(imgs, vertices, indices, blend, dstRegion, srcRegions, shader.ensureShader(), uniforms)
}

// WritePixels replaces the pixels on the image.
func (i *Image) WritePixels(pix []byte, region image.Rectangle) {
	backendsM.Lock()
	defer backendsM.Unlock()

	if !inFrame {
		copied := make([]byte, len(pix))
		copy(copied, pix)

		appendDeferred(func() {
			i.writePixels(copied, region)
		})
		return
	}

	i.writePixels(pix, region)
}

func (i *Image) writePixels(pix []byte, region image.Rectangle) {
	if l := 4 * region.Dx() * region.Dy(); len(pix) != l {
		panic(fmt.Sprintf("atlas: len(pix) must be %d but %d", l, len(pix)))
	}

	i.resetUsedAsSourceCount()

	if i.backend == nil {
		if pix == nil {
			return
		}
		// Allocate as a source as this image will likely be used as a source.
		i.allocate(nil, i.imageType == ImageTypeRegular)
	}

	// A budget eviction may have moved the backend's texture to CPU memory. Re-upload it before
	// applying the new pixels so partial writes keep the preserved content around them.
	i.ensureResident()
	if i.backend.backendImage == nil {
		// Re-upload failed: record the error and drop the write instead of touching a nil texture.
		theResidencyBudget.recordFrameError(errBackendNotResident)
		return
	}
	theResidencyBudget.touch(i.backend)

	r := i.regionWithPadding()

	if !region.Eq(image.Rect(0, 0, i.width, i.height)) || i.paddingSize() == 0 {
		region = region.Add(r.Min)

		if pix == nil {
			i.backend.clear(region)
			return
		}

		// Copy pixels in the case when pix is modified before the graphics command is executed.
		pix2 := graphics.NewManagedBytes(len(pix), func(bs []byte) {
			copy(bs, pix)
		})
		i.backend.backendImage.WritePixels(pix2, region)
		return
	}

	// TODO: These loops assume that paddingSize is 1.
	// TODO: Is clearing edges explicitly really needed?
	const paddingSize = 1
	if paddingSize != i.paddingSize() {
		panic(fmt.Sprintf("atlas: writePixels assumes the padding is always 1 but the actual padding was %d", i.paddingSize()))
	}

	pixb := graphics.NewManagedBytes(4*r.Dx()*r.Dy(), func(bs []byte) {
		// Clear the edges. bs might not be zero-cleared.
		rowPixels := 4 * r.Dx()
		if r.Dy() > region.Dy() {
			for i := range rowPixels {
				bs[rowPixels*(r.Dy()-1)+i] = 0
			}
		}
		if r.Dx() > region.Dx() {
			for j := 1; j <= region.Dy(); j++ {
				bs[rowPixels*j-4] = 0
				bs[rowPixels*j-3] = 0
				bs[rowPixels*j-2] = 0
				bs[rowPixels*j-1] = 0
			}
		}

		// Copy the content.
		for j := 0; j < region.Dy(); j++ {
			copy(bs[4*j*r.Dx():], pix[4*j*region.Dx():4*(j+1)*region.Dx()])
		}
	})
	i.backend.backendImage.WritePixels(pixb, r)
}

// deferredReadPixels is a pixel read-back requested between frames.
type deferredReadPixels struct {
	result *graphicscommand.PixelReadback

	// queued is closed when the read-back is queued or aborted.
	queued chan struct{}
}

// ReadPixelsAsync starts a pixel read-back and returns its result without waiting for a frame.
func (i *Image) ReadPixelsAsync(pixels []byte, region image.Rectangle) *graphicscommand.PixelReadback {
	backendsM.Lock()
	defer backendsM.Unlock()

	result := graphicscommand.NewPixelReadback()

	if !inFrame {
		// Queue the read-back at the next BeginFrame, in order with the other deferred operations, so
		// that it includes the operations before it and excludes the operations after it.
		d := &deferredReadPixels{
			result: result,
			queued: make(chan struct{}),
		}
		theDeferredReadPixels = append(theDeferredReadPixels, d)
		appendDeferred(func() {
			idx := slices.Index(theDeferredReadPixels, d)
			if idx < 0 {
				// The read-back has been aborted.
				return
			}
			theDeferredReadPixels = slices.Delete(theDeferredReadPixels, idx, idx+1)
			i.readPixelsAsync(pixels, region, result)
			close(d.queued)
		})
		return result
	}

	// In the tests, BeginFrame might not be called often and then images might not be disposed (#2292).
	// To prevent memory leaks, flush the deferred functions here.
	flushDeferred()

	i.readPixelsAsync(pixels, region, result)
	return result
}

// readPixelsAsync queues a pixel read-back that completes result.
// readPixelsAsync must be called in a frame with backendsM locked.
func (i *Image) readPixelsAsync(pixels []byte, region image.Rectangle, result *graphicscommand.PixelReadback) {
	if i.backend == nil || i.backend.backendImage == nil {
		// While evicted, serve the read-back from the CPU mirror without forcing a re-upload.
		if i.backend != nil && i.backend.evicted {
			i.readPixelsFromResidencyMirror(pixels, region, result)
			return
		}
		clear(pixels)
		result.Complete(nil)
		return
	}

	args := []graphicsdriver.PixelsArgs{
		{
			Pixels: pixels,
			Region: region.Add(i.regionWithPadding().Min),
		},
	}
	i.backend.backendImage.ReadPixelsAsync(args, result)
}

// readPixelsFromResidencyMirror copies requested pixels out of an evicted backend's CPU mirror.
// The mirror covers the backend's allocated region; anything outside it is transparent.
func (i *Image) readPixelsFromResidencyMirror(pixels []byte, region image.Rectangle, result *graphicscommand.PixelReadback) {
	b := i.backend
	clear(pixels)
	if b.residentPixels == nil {
		result.Complete(nil)
		return
	}

	// region is given in the image's coordinates; translate it into backend coordinates.
	full := region.Add(i.regionWithPadding().Min)
	inter := full.Intersect(b.residentRegion)
	if inter.Empty() {
		result.Complete(nil)
		return
	}

	mirror := b.residentPixels.Bytes()
	mirrorW := b.residentRegion.Dx()
	for y := inter.Min.Y; y < inter.Max.Y; y++ {
		for x := inter.Min.X; x < inter.Max.X; x++ {
			mIdx := 4 * ((y-b.residentRegion.Min.Y)*mirrorW + (x - b.residentRegion.Min.X))
			dIdx := 4 * ((y-full.Min.Y)*full.Dx() + (x - full.Min.X))
			copy(pixels[dIdx:dIdx+4], mirror[mIdx:mIdx+4])
		}
	}
	result.Complete(nil)
}

// Deallocate deallocates the internal state.
// Even after this call, the image is still available as a new cleared image.
func (i *Image) Deallocate() {
	backendsM.Lock()
	defer backendsM.Unlock()

	// i.cleanup is rewritten under backendsM e.g. at moveTo.
	i.cleanup.Stop()

	if !inFrame {
		appendDeferred(func() {
			i.deallocateImpl()
		})
		return
	}

	i.deallocateImpl()
}

func (i *imageImpl) deallocateImpl() {
	defer func() {
		i.backend = nil
		i.node = nil
	}()

	i.usedAsSourceCount = 0
	i.usedAsDestinationCount = 0

	if i.backend == nil {
		// Not allocated yet.
		return
	}

	b := i.backend
	if i.isOnAtlas() {
		b.page.Free(i.node)
		if !b.page.IsEmpty() {
			// As this part can be reused, this should be cleared explicitly.
			r := i.regionWithPadding()
			b.clear(r)
			return
		}
	}

	// The page is empty or the backend is isolated: release the whole backend.
	b.disposeGPUTexture()
	if b.residentPixels != nil {
		b.residentPixels.Release()
		b.residentPixels = nil
		b.residentRegion = image.Rectangle{}
	}
	b.evicted = false

	for idx, sh := range theBackends {
		if sh == b {
			copy(theBackends[idx:], theBackends[idx+1:])
			theBackends[len(theBackends)-1] = nil
			theBackends = theBackends[:len(theBackends)-1]
			return
		}
	}

	panic("atlas: backend not found at an image being deallocated")
}

// MaxImageSize returns the maximum width and height of an image in pixels.
// MaxImageSize returns 0 before the graphics driver is ready.
func MaxImageSize() int {
	backendsM.Lock()
	defer backendsM.Unlock()
	return maxSize
}

func NewImage(width, height int, imageType ImageType) *Image {
	// Actual allocation is done lazily, and the lock is not needed.
	return &Image{
		imageImpl: &imageImpl{
			width:     width,
			height:    height,
			imageType: imageType,
		},
	}
}

func (i *Image) canBePutOnAtlas() bool {
	if minSourceSize == 0 || minDestinationSize == 0 || maxSize == 0 {
		panic("atlas: min*Size or maxSize must be initialized")
	}
	if i.imageType != ImageTypeRegular {
		return false
	}
	s := maxPageSize()
	return i.width <= s && i.height <= s
}

func (i *imageImpl) cleanup() {
	// A function from finalizer must not be blocked, but disposing operation can be blocked.
	// Defer this operation until it becomes safe. (#913)
	appendDeferred(func() {
		i.deallocateImpl()
	})
}

func (i *Image) allocate(forbiddenBackends []*backend, asSource bool) {
	if i.backend != nil {
		panic("atlas: the image is already allocated")
	}

	i.cleanup = runtime.AddCleanup(i, (*imageImpl).cleanup, i.imageImpl)

	if i.imageType == ImageTypeScreen {
		if asSource {
			panic("atlas: a screen image cannot be created as a source")
		}
		// A screen image doesn't have a padding.
		i.backend = &backend{
			id:           nextBackendID.Add(1),
			width:        i.width,
			height:       i.height,
			backendImage: graphicscommand.NewImage(i.width, i.height, true, ""),
			screen:       true,
		}
		i.backend.clear(image.Rect(0, 0, i.width, i.height))
		theBackends = append(theBackends, i.backend)
		theResidencyBudget.applyPriorityToBackend(i.imageImpl)
		theResidencyBudget.recordPeak()
		return
	}

	wp := i.width + i.paddingSize()
	hp := i.height + i.paddingSize()

	if !i.canBePutOnAtlas() {
		if wp > maxSize || hp > maxSize {
			panic(fmt.Sprintf("atlas: the image being put on an atlas is too big: width: %d, height: %d", i.width, i.height))
		}

		i.backend = &backend{
			id:           nextBackendID.Add(1),
			width:        wp,
			height:       hp,
			backendImage: graphicscommand.NewImage(wp, hp, false, ""),
			source:       asSource,
		}
		i.backend.clear(image.Rect(0, 0, wp, hp))
		theBackends = append(theBackends, i.backend)
		theResidencyBudget.applyPriorityToBackend(i.imageImpl)
		theResidencyBudget.recordPeak()
		return
	}

	// Check if an existing backend is available.
loop:
	for _, b := range theBackends {
		if b.source != asSource {
			continue
		}
		for _, bb := range forbiddenBackends {
			if b == bb {
				continue loop
			}
		}

		if n, ok := b.tryAlloc(wp, hp); ok {
			i.backend = b
			i.node = n
			theResidencyBudget.applyPriorityToBackend(i.imageImpl)
			return
		}
	}

	var width, height int
	if asSource {
		width, height = minSourceSize, minSourceSize
	} else {
		width, height = minDestinationSize, minDestinationSize
	}
	maxPage := maxPageSize()
	for i.width > width {
		if width == maxPage {
			panic(fmt.Sprintf("atlas: the image being put on an atlas is too big: width: %d, height: %d", i.width, i.height))
		}
		width *= 2
	}
	for i.height > height {
		if height == maxPage {
			panic(fmt.Sprintf("atlas: the image being put on an atlas is too big: width: %d, height: %d", i.width, i.height))
		}
		height *= 2
	}

	// Only a regular image can be put on an atlas, so every image on this page has the same padding.
	page := packing.NewPage(width, height, maxPage, i.paddingSize())
	b := &backend{
		id:           nextBackendID.Add(1),
		width:        width,
		height:       height,
		backendImage: graphicscommand.NewImage(width, height, false, ""),
		page:         page,
		source:       asSource,
	}
	b.clear(image.Rect(0, 0, width, height))
	theBackends = append(theBackends, b)

	n := b.page.Alloc(wp, hp)
	if n == nil {
		panic("atlas: Alloc result must not be nil at allocate")
	}
	i.backend = b
	i.node = n
	theResidencyBudget.applyPriorityToBackend(i.imageImpl)
	theResidencyBudget.recordPeak()
}

func (i *Image) DumpScreenshot(graphicsDriver graphicsdriver.Graphics, path string, blackbg bool) (string, error) {
	backendsM.Lock()
	defer backendsM.Unlock()

	if !inFrame {
		panic("atlas: DumpScreenshot must be called in between BeginFrame and EndFrame")
	}

	i.ensureResident()
	if i.backend == nil || i.backend.backendImage == nil {
		return "", errBackendNotResident
	}

	return i.backend.backendImage.Dump(graphicsDriver, path, blackbg, image.Rect(0, 0, i.width, i.height))
}

// FlushAndPollPixelReadbacks submits pending commands and checks read-back completion.
func FlushAndPollPixelReadbacks(graphicsDriver graphicsdriver.Graphics) error {
	backendsM.Lock()
	defer backendsM.Unlock()

	return graphicscommand.FlushAndPollPixelReadbacks(graphicsDriver)
}

// FlushAndWaitPixelReadbacks submits pending commands and completes the given read-backs.
// A read-back requested between frames is waited for until the next frame queues it.
func FlushAndWaitPixelReadbacks(graphicsDriver graphicsdriver.Graphics, results []*graphicscommand.PixelReadback) error {
	// backendsM must not be held while waiting, as BeginFrame needs it to queue the read-backs.
	for {
		queued := deferredReadPixelsQueued(results)
		if queued == nil {
			break
		}
		<-queued
	}

	backendsM.Lock()
	defer backendsM.Unlock()

	return graphicscommand.FlushAndWaitPixelReadbacks(graphicsDriver, results)
}

// deferredReadPixelsQueued returns the channel closed when the first of results that is not queued yet
// is queued, or nil when all of results are queued.
func deferredReadPixelsQueued(results []*graphicscommand.PixelReadback) <-chan struct{} {
	backendsM.Lock()
	defer backendsM.Unlock()

	for _, d := range theDeferredReadPixels {
		if slices.Contains(results, d.result) {
			return d.queued
		}
	}
	return nil
}

// AbortReadPixels completes pending, queued, and deferred pixel read-backs with an error.
func AbortReadPixels() {
	backendsM.Lock()
	defer backendsM.Unlock()

	for _, d := range theDeferredReadPixels {
		d.result.Abort()
		close(d.queued)
	}
	clear(theDeferredReadPixels)
	theDeferredReadPixels = theDeferredReadPixels[:0]

	graphicscommand.AbortReadPixels()
}

func EndFrame(graphicsDriver graphicsdriver.Graphics) error {
	backendsM.Lock()
	defer backendsM.Unlock()
	defer func() {
		inFrame = false
	}()

	if !inFrame {
		panic("atlas: inFrame must be true in EndFrame")
	}

	for _, b := range theBackends {
		b.sourceInThisFrame = false
	}

	// Apply residency trims while the graphics driver is available. A residency failure observed
	// during the frame is reported before the trim so callers see both kinds of errors.
	if err := theResidencyBudget.consumeFrameError(); err != nil {
		// Still attempt the trim, but prefer the earlier error.
		if trimErr := theResidencyBudget.processFrameBoundaryTasks(graphicsDriver); trimErr != nil {
			return errors.Join(err, trimErr)
		}
		return err
	}
	if err := theResidencyBudget.processFrameBoundaryTasks(graphicsDriver); err != nil {
		return err
	}

	if theGPUResourcesState.isSavingGPUResourcesRequested() {
		flushDeferred()
		for _, b := range theBackends {
			if b.backendImage == nil {
				// Evicted backends have no GPU texture to save. Their CPU mirror is enough to
				// re-upload after the device is rebuilt.
				if b.evicted {
					b.restoreInfo = restoreInfo{
						valid:  true,
						screen: b.screen,
					}
				}
				continue
			}
			var pixels *graphics.ManagedBytes
			var region image.Rectangle
			if !b.screen {
				if b.page != nil {
					region = b.page.AllocatedRegion()
				} else {
					// If a page doesn't exist, the image is not on an atlas.
					region = image.Rect(0, 0, b.width, b.height)
				}
				if !region.Empty() {
					var err error
					pixels = graphics.NewManagedBytes(4*region.Dx()*region.Dy(), func(bytes []byte) {
						args := []graphicsdriver.PixelsArgs{
							{
								Pixels: bytes,
								Region: region,
							},
						}
						err = b.backendImage.ReadPixels(graphicsDriver, args)
					})
					if err != nil {
						// The saved pixels are incomplete, so the resources cannot be restored from them.
						theGPUResourcesState.finishSavingGPUResources(false)
						return err
					}
				}
			}
			b.restoreInfo = restoreInfo{
				valid:  true,
				screen: b.screen,
				pixels: pixels,
				region: region,
			}
		}
		theGPUResourcesState.finishSavingGPUResources(true)
	}

	return nil
}

func FlushCommands(graphicsDriver graphicsdriver.Graphics, present bool) error {
	backendsM.Lock()
	defer backendsM.Unlock()

	if inFrame {
		panic("atlas: inFrame must be false in FlushCommands")
	}

	if debug.IsDebug {
		debug.FrameLogf("Internal image sizes:\n")
		imgs := make([]*graphicscommand.Image, 0, len(theBackends))
		for _, img := range theBackends {
			if img.backendImage == nil {
				continue
			}
			imgs = append(imgs, img.backendImage)
		}
		graphicscommand.LogImagesInfo(imgs)
	}

	mode := graphicsdriver.FlushModeEndFrame
	if present {
		mode = graphicsdriver.FlushModePresent
	}
	if err := graphicscommand.FlushCommands(graphicsDriver, mode); err != nil {
		return err
	}

	return nil
}

func floorPowerOf2(x int) int {
	if x <= 0 {
		return 0
	}
	return 1 << (bits.Len(uint(x)) - 1)
}

func BeginFrame(graphicsDriver graphicsdriver.Graphics) error {
	backendsM.Lock()
	defer backendsM.Unlock()

	if inFrame {
		panic("atlas: inFrame must be false in BeginFrame")
	}

	inFrame = true

	var err error
	initOnce.Do(func() {
		err = graphicscommand.InitializeGraphicsDriverState(graphicsDriver)
		if err != nil {
			return
		}
		if len(theBackends) != 0 {
			panic("atlas: no images must be on an atlas before the game starts")
		}

		// min*Size and maxSize can already be set for testings.
		if minSourceSize == 0 {
			minSourceSize = 1024
		}
		if minDestinationSize == 0 {
			minDestinationSize = 16
		}
		if maxSize == 0 {
			maxSize = floorPowerOf2(graphicscommand.MaxImageSize(graphicsDriver))
		}
		inited.Store(true)
	})
	if err != nil {
		return err
	}

	if theGPUResourcesState.startRestoringGPUResourcesIfNeeded() {
		if err := graphicscommand.ResetGraphicsDriverState(graphicsDriver); err != nil {
			return err
		}

		// Dispose all the internal shaders.
		// The current internal shader objects might have references to old GPU resources.
		theShadersWithInternalShader.deallocateInternalShaders()

		// Dispose all the images.
		// The current internal image objects might have references to old GPU resources.
		for _, b := range theBackends {
			if b.backendImage == nil {
				continue
			}
			b.backendImage.Dispose()
			b.backendImage = nil
		}

		// Restore all the images.
		for _, b := range theBackends {
			// Evicted backends keep their pixels in CPU memory and must not be re-created here:
			// a later ensureResident re-uploads them on demand.
			if b.evicted {
				b.restoreInfo = restoreInfo{}
				continue
			}
			if !b.restoreInfo.valid {
				continue
			}
			b.backendImage = graphicscommand.NewImage(b.width, b.height, b.restoreInfo.screen, "")
			if b.restoreInfo.region != image.Rect(0, 0, b.width, b.height) {
				b.clear(image.Rect(0, 0, b.width, b.height))
			}
			if b.restoreInfo.pixels != nil {
				b.backendImage.WritePixels(b.restoreInfo.pixels, b.restoreInfo.region)
			}
		}
	}

	// Discard restore info whichever restoration is done or not.
	for _, b := range theBackends {
		// Do not release pixels here, as the command buffer might not be flushed yet.
		b.restoreInfo = restoreInfo{}
	}

	flushDeferred()
	putImagesOnSourceBackend()

	return nil
}

func DumpImages(graphicsDriver graphicsdriver.Graphics, dir string) (string, error) {
	backendsM.Lock()
	defer backendsM.Unlock()

	if !inFrame {
		panic("atlas: DumpImages must be called in between BeginFrame and EndFrame")
	}

	images := make([]*graphicscommand.Image, 0, len(theBackends))
	for _, backend := range theBackends {
		if backend.backendImage == nil {
			continue
		}
		images = append(images, backend.backendImage)
	}
	return graphicscommand.DumpImages(images, graphicsDriver, dir)
}

func TotalGPUImageMemoryUsageInBytes() int64 {
	backendsM.Lock()
	defer backendsM.Unlock()

	var sum int64
	for _, b := range theBackends {
		if b.backendImage == nil {
			continue
		}
		w, h := b.backendImage.InternalSize()
		sum += 4 * int64(w) * int64(h)
	}
	return sum
}

// SaveGPUResources saves GPU resources (textures and shaders) in CPU memory.
//
// Until GPU resources are saved, the app becomes suspended
// until [RestoreGPUResources] or [ResumeApp] is called.
//
// If [RestoreGPUResources] is called, the saved resources are restored in the next frame.
// If [ResumeApp] is called but not [RestoreGPUResources], the saved resources are discarded.
func SaveGPUResources() {
	if !inited.Load() {
		return
	}
	// Suspend the app. [IsSuspended] will return true after the GPU resources are actually saved.
	suspended.Store(true)
	theGPUResourcesState.requestToSaveGPUResources()
}

// IsSuspended reports whether the app is currently suspended.
func IsSuspended() bool {
	return suspended.Load() && theGPUResourcesState.areGPUResourcesSaved()
}

// AreGPUResourcesSaved reports whether GPU resources are currently saved.
func AreGPUResourcesSaved() bool {
	return theGPUResourcesState.areGPUResourcesSaved()
}

// ResumeApp unfreezes the app previously frozen by [SaveGPUResources].
func ResumeApp() {
	suspended.Store(false)
}

// RestoreGPUResources restores GPU resources in the next frame if needed.
// RestoreGPUResources reports whether there are GPU resources to be restored or not.
func RestoreGPUResources() bool {
	suspended.Store(false)
	return theGPUResourcesState.requestToRestoreGPUResources()
}
