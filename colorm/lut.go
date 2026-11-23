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

package colorm

import (
	"errors"
	"fmt"
	"image/color"
	"math"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/internal/colormshader"
)

// DefaultLUTSize is the number of quantization levels per channel used when
// LUTOptions.Size is 0. With 256 levels, an 8-bit input channel maps to its
// own level, so the quantization error is invisible on 8-bit output.
const DefaultLUTSize = 256

// MaxLUTSize is the maximum number of quantization levels per channel.
// Larger tables cost memory (16*size float32 values plus a size*8 GPU image)
// without improving precision on 8-bit render targets.
const MaxLUTSize = 1024

// lutImageHeight is the height of the GPU image backing a ColorTransformLUT:
// 2 rows per input channel.
const lutImageHeight = 8

// srgbGammaApprox is the gamma used to approximate the sRGB electro-optical
// transfer function for LUTColorSpaceSRGB. A single 2.2 gamma curve is a
// well-known approximation of the piecewise sRGB curve.
const srgbGammaApprox = 2.2

// Errors returned by ColorTransformLUT constructors and mutating methods.
var (
	// ErrNonFiniteCoefficients is returned when a color matrix or color scale
	// contains NaN or infinite coefficients. A LUT built from such values
	// could produce NaN pixels, so the construction is rejected instead.
	ErrNonFiniteCoefficients = errors.New("colorm: the color transform contains non-finite coefficients")

	// ErrInvalidLUTSize is returned when LUTOptions.Size is negative or 1.
	ErrInvalidLUTSize = errors.New("colorm: the LUT size must be 0 (the default size) or between 2 and MaxLUTSize")

	// ErrLUTTooLarge is returned when LUTOptions.Size exceeds MaxLUTSize.
	ErrLUTTooLarge = errors.New("colorm: the LUT size exceeds MaxLUTSize")

	// ErrInvalidLUTColorSpace is returned when LUTOptions.ColorSpace is not a
	// valid LUTColorSpace value.
	ErrInvalidLUTColorSpace = errors.New("colorm: invalid LUT color space")

	// ErrLUTNotInvertible is returned by Invert when the source matrix is not
	// invertible. The LUT is left unchanged.
	ErrLUTNotInvertible = errors.New("colorm: the color transform is not invertible")
)

// LUTColorSpace determines how the quantization levels of a
// ColorTransformLUT are distributed over the input range.
type LUTColorSpace int

const (
	// LUTColorSpaceLinear distributes the levels uniformly over the linear
	// (straight-alpha) input values. This is the default.
	LUTColorSpaceLinear LUTColorSpace = iota

	// LUTColorSpaceSRGB distributes the levels uniformly over an
	// sRGB-encoded (gamma 2.2 approximation) scale. This gives darker tones
	// finer resolution at the cost of coarser highlights, which is usually
	// perceptually more even for color grading.
	LUTColorSpaceSRGB
)

// LUTOptions represents options for building a ColorTransformLUT.
type LUTOptions struct {
	// Size is the number of quantization levels per channel.
	// 0 means DefaultLUTSize. The value must be between 2 and MaxLUTSize.
	Size int

	// ColorSpace is the color space in which the levels are distributed.
	// The default (zero) value is LUTColorSpaceLinear.
	ColorSpace LUTColorSpace
}

func resolveLUTOptions(opts *LUTOptions) (size int, gamma float32, err error) {
	size = DefaultLUTSize
	gamma = 1
	if opts == nil {
		return size, gamma, nil
	}
	switch {
	case opts.Size == 0:
	case opts.Size < 2:
		return 0, 0, fmt.Errorf("%w, but got %d", ErrInvalidLUTSize, opts.Size)
	case opts.Size > MaxLUTSize:
		return 0, 0, fmt.Errorf("%w (%d), but got %d", ErrLUTTooLarge, MaxLUTSize, opts.Size)
	default:
		size = opts.Size
	}
	switch opts.ColorSpace {
	case LUTColorSpaceLinear:
	case LUTColorSpaceSRGB:
		gamma = srgbGammaApprox
	default:
		return 0, 0, fmt.Errorf("%w: %d", ErrInvalidLUTColorSpace, opts.ColorSpace)
	}
	return size, gamma, nil
}

// lutSnapshot is an immutable view of a ColorTransformLUT's state.
// Mutating methods replace the snapshot atomically under a lock, so readers
// never observe a half-updated LUT.
type lutSnapshot struct {
	body        [16]float32 // column-major, same layout as ColorM.ReadElements
	translation [4]float32
	size        int
	gamma       float32
	scale       float32
	table       []float32 // 16*size entries: table[(i*4+j)*size+k] = body[j*4+i] * level(k)
}

// newSnapshot builds a snapshot from the given matrix elements.
// The elements must be finite; the caller is responsible for validating them.
func newSnapshot(body [16]float32, translation [4]float32, size int, gamma float32) *lutSnapshot {
	s := &lutSnapshot{
		body:        body,
		translation: translation,
		size:        size,
		gamma:       gamma,
		table:       make([]float32, 16*size),
	}
	for k := 0; k < size; k++ {
		v := float64(k) / float64(size-1)
		if gamma != 1 {
			v = math.Pow(v, float64(gamma))
		}
		f := float32(v)
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				s.table[(i*4+j)*size+k] = body[j*4+i] * f
			}
		}
	}
	for _, v := range s.table {
		if a := float32(math.Abs(float64(v))); a > s.scale {
			s.scale = a
		}
	}
	if s.scale == 0 {
		// A zero transform would make the encoded range degenerate.
		// Any scale works since every decoded value is 0; use 1 to avoid
		// division by zero in the shader and in encodePixels.
		s.scale = 1
	}
	return s
}

// encodePixels encodes the table into the RGBA8 pixel layout sampled by the
// LUT shader: for input channel j, row 2*j holds output channels 0 (R,G) and
// 1 (B,A), and row 2*j+1 holds output channels 2 (R,G) and 3 (B,A). Each
// value is stored as a 16-bit fixed-point number across two channels,
// covering [-scale, scale].
func (s *lutSnapshot) encodePixels() []byte {
	n := s.size
	pix := make([]byte, n*lutImageHeight*4)
	enc := func(v float32) (hi, lo byte) {
		q := math.Round((float64(v)/float64(s.scale) + 1) / 2 * 65535)
		if q < 0 {
			q = 0
		}
		if q > 65535 {
			q = 65535
		}
		return byte(uint16(q) >> 8), byte(uint16(q))
	}
	for j := 0; j < 4; j++ {
		for k := 0; k < n; k++ {
			for half := 0; half < 2; half++ {
				i0 := 2 * half
				i1 := 2*half + 1
				off := ((2*j+half)*n + k) * 4
				hi0, lo0 := enc(s.table[(i0*4+j)*n+k])
				hi1, lo1 := enc(s.table[(i1*4+j)*n+k])
				pix[off+0] = hi0
				pix[off+1] = lo0
				pix[off+2] = hi1
				pix[off+3] = lo1
			}
		}
	}
	return pix
}

// ColorTransformLUT is a reusable lookup table for a color transform.
//
// A ColorTransformLUT precomputes a ColorM (or a ColorScale) into a table so
// that applying the transform to each pixel is a table lookup instead of a
// 4x4 matrix multiplication. This is useful when the same transform is
// applied repeatedly, for example for full-screen color grading.
//
// The transform is decomposed exactly as out_i = sum_j(M[j][i]*in_j) + t_i,
// so the only approximation is the quantization of each input channel to a
// fixed number of levels (and the fixed-point encoding of the table itself).
// Like ColorM, the transform is applied to straight-alpha colors: pixels are
// un-premultiplied before the lookup and re-premultiplied after it.
//
// The zero value is not usable; create a ColorTransformLUT with
// NewColorTransformLUTFromColorM or NewColorTransformLUTFromColorScale.
//
// A ColorTransformLUT is safe for concurrent read-only use (Apply, and draws
// through DrawImage/DrawTriangles with the LUT set). Concat and Invert must
// not be called concurrently with other methods.
type ColorTransformLUT struct {
	// mu guards snap and image. Readers hold it briefly to copy the current
	// snapshot pointer; the snapshot itself is immutable.
	mu    sync.RWMutex
	snap  *lutSnapshot
	image *ebiten.Image // lazily built GPU cache of snap's table

	_ [0]func() // Marks as non-comparable.
}

// NewColorTransformLUTFromColorM builds a reusable lookup table from c.
//
// The matrix is copied: later changes to c do not affect the LUT, and
// building the LUT never modifies c.
//
// NewColorTransformLUTFromColorM returns an error if c contains non-finite
// (NaN or infinite) coefficients, or if opts is invalid.
func NewColorTransformLUTFromColorM(c ColorM, opts *LUTOptions) (*ColorTransformLUT, error) {
	size, gamma, err := resolveLUTOptions(opts)
	if err != nil {
		return nil, err
	}
	var body [16]float32
	var translation [4]float32
	c.affineColorM().Elements(body[:], translation[:])
	if !finiteElements(body[:]) || !finiteElements(translation[:]) {
		return nil, ErrNonFiniteCoefficients
	}
	return &ColorTransformLUT{
		snap: newSnapshot(body, translation, size, gamma),
	}, nil
}

// NewColorTransformLUTFromColorScale builds a reusable lookup table from a
// color scale.
//
// A ColorScale is applied to premultiplied-alpha colors while a ColorM is
// applied to straight-alpha colors. The LUT follows the ColorM convention, so
// the scale (r, g, b, a) is converted to the equivalent straight-alpha
// transform ColorM.Scale(r/a, g/a, b/a, a). If a is 0, every output pixel is
// transparent black after clamping, so the LUT is built from the zero matrix.
//
// NewColorTransformLUTFromColorScale returns an error if cs contains
// non-finite components or the conversion overflows, or if opts is invalid.
func NewColorTransformLUTFromColorScale(cs ebiten.ColorScale, opts *LUTOptions) (*ColorTransformLUT, error) {
	r, g, b, a := cs.R(), cs.G(), cs.B(), cs.A()
	if !finiteElements([]float32{r, g, b, a}) {
		return nil, ErrNonFiniteCoefficients
	}
	var cm ColorM
	if a == 0 {
		cm.Scale(0, 0, 0, 0)
	} else {
		cm.Scale(float64(r)/float64(a), float64(g)/float64(a), float64(b)/float64(a), float64(a))
	}
	return NewColorTransformLUTFromColorM(cm, opts)
}

func finiteElements(vs []float32) bool {
	for _, v := range vs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func (l *ColorTransformLUT) snapshot() *lutSnapshot {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.snap
}

// Size returns the number of quantization levels per channel.
func (l *ColorTransformLUT) Size() int {
	return l.snapshot().size
}

// MemorySize returns the approximate memory footprint of the LUT in bytes:
// the CPU-side table plus the GPU-side image.
func (l *ColorTransformLUT) MemorySize() int {
	s := l.snapshot()
	return len(s.table)*4 + s.size*lutImageHeight*4
}

// ColorM returns a copy of the source color matrix of l.
// Modifying the returned ColorM does not affect l.
func (l *ColorTransformLUT) ColorM() ColorM {
	s := l.snapshot()
	var cm ColorM
	for i := 0; i < Dim-1; i++ {
		for j := 0; j < Dim-1; j++ {
			cm.SetElement(i, j, float64(s.body[j*4+i]))
		}
		cm.SetElement(i, Dim-1, float64(s.translation[i]))
	}
	return cm
}

// Apply applies the lookup table to clr and returns the result.
// clr is interpreted as a straight-alpha color, and the result is clamped to
// the range [0, 1], in the same way as ColorM.Apply. The only difference
// from ColorM.Apply is the quantization of the input channels.
//
// Apply panics if clr is nil.
func (l *ColorTransformLUT) Apply(clr color.Color) color.Color {
	if clr == nil {
		panic("colorm: the given color to Apply must not be nil")
	}
	s := l.snapshot()
	r, g, b, a := clr.RGBA()
	var in [4]float32
	if a > 0 {
		in[0] = float32(r) / float32(a)
		in[1] = float32(g) / float32(a)
		in[2] = float32(b) / float32(a)
		in[3] = float32(a) / 0xffff
	}
	out := s.translation
	n := s.size
	for j := 0; j < 4; j++ {
		v := clamp01(in[j])
		if s.gamma != 1 {
			v = float32(math.Pow(float64(v), float64(1.0/s.gamma)))
		}
		k := int(v*float32(n-1) + 0.5)
		for i := 0; i < 4; i++ {
			out[i] += s.table[(i*4+j)*n+k]
		}
	}
	return color.NRGBA64{
		R: uint16(clamp01(out[0]) * 0xffff),
		G: uint16(clamp01(out[1]) * 0xffff),
		B: uint16(clamp01(out[2]) * 0xffff),
		A: uint16(clamp01(out[3]) * 0xffff),
	}
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Concat composes l's transform with c: the resulting LUT applies l's current
// transform first and then c. This mirrors ColorM.Concat.
//
// The composition is computed on the exact source matrix, not on the
// quantized table, so repeated Concat calls do not accumulate quantization
// error. c is not modified.
//
// Concat returns an error if c contains non-finite coefficients; in that
// case l is left unchanged.
func (l *ColorTransformLUT) Concat(c ColorM) error {
	var cb [16]float32
	var ct [4]float32
	c.affineColorM().Elements(cb[:], ct[:])
	if !finiteElements(cb[:]) || !finiteElements(ct[:]) {
		return ErrNonFiniteCoefficients
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	old := l.snap
	nb := mulMat4(cb, old.body)
	nt := mulMat4Vec(cb, old.translation)
	for i := range nt {
		nt[i] += ct[i]
	}
	l.snap = newSnapshot(nb, nt, old.size, old.gamma)
	l.image = nil
	return nil
}

// Invert inverts l's transform. This mirrors ColorM.Invert, but instead of
// panicking it returns ErrLUTNotInvertible and leaves l unchanged when the
// transform is not invertible.
func (l *ColorTransformLUT) Invert() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	old := l.snap
	inv, ok := invertMat4(old.body)
	if !ok {
		return ErrLUTNotInvertible
	}
	nt := mulMat4Vec(inv, old.translation)
	for i := range nt {
		nt[i] = -nt[i]
	}
	l.snap = newSnapshot(inv, nt, old.size, old.gamma)
	l.image = nil
	return nil
}

// Dispose releases the GPU image backing the LUT, if any. The CPU-side table
// is kept, so the LUT remains usable: the image is rebuilt lazily on the next
// draw. Use Dispose to bound GPU memory when many LUTs are cached.
func (l *ColorTransformLUT) Dispose() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.image != nil {
		l.image.Deallocate()
		l.image = nil
	}
}

// drawState returns the current snapshot and the GPU image encoding it,
// building the image lazily on first use.
func (l *ColorTransformLUT) drawState() (*lutSnapshot, *ebiten.Image) {
	for {
		l.mu.RLock()
		snap, img := l.snap, l.image
		l.mu.RUnlock()
		if img != nil {
			return snap, img
		}
		newImg := ebiten.NewImage(snap.size, lutImageHeight)
		newImg.WritePixels(snap.encodePixels())
		l.mu.Lock()
		if l.snap == snap && l.image == nil {
			l.image = newImg
			l.mu.Unlock()
			return snap, newImg
		}
		// The snapshot was rebuilt or another goroutine won the race while
		// the image was being created. Use the current state, or retry.
		snap, img = l.snap, l.image
		l.mu.Unlock()
		if img != nil {
			return snap, img
		}
	}
}

// lutUniformsPoolValue reuses the uniform maps for LUT draws.
type lutUniformsPoolValue struct {
	uniforms    map[string]any
	translation [4]float32
}

var theLUTUniformsPool = sync.Pool{
	New: func() any {
		return &lutUniformsPoolValue{
			uniforms: map[string]any{},
		}
	},
}

func acquireLUTUniforms(s *lutSnapshot) *lutUniformsPoolValue {
	u := theLUTUniformsPool.Get().(*lutUniformsPoolValue)
	u.translation = s.translation
	u.uniforms[colormshader.UniformLUTScale] = s.scale
	u.uniforms[colormshader.UniformLUTGamma] = s.gamma
	u.uniforms[colormshader.UniformLUTTranslation] = u.translation[:]
	return u
}

func releaseLUTUniforms(u *lutUniformsPoolValue) {
	clear(u.uniforms)
	theLUTUniformsPool.Put(u)
}

// mulMat4 multiplies two column-major 4x4 matrices: the result is a*b, i.e.
// applying b first and then a.
func mulMat4(a, b [16]float32) [16]float32 {
	var r [16]float32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 4; k++ {
				r[j*4+i] += a[k*4+i] * b[j*4+k]
			}
		}
	}
	return r
}

// mulMat4Vec multiplies a column-major 4x4 matrix by a vector.
func mulMat4Vec(m [16]float32, v [4]float32) [4]float32 {
	var r [4]float32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			r[i] += m[j*4+i] * v[j]
		}
	}
	return r
}

// invertMat4 inverts a column-major 4x4 matrix with Gauss-Jordan elimination
// and partial pivoting, in float64 for numerical stability. It returns false
// if the matrix is singular.
func invertMat4(m [16]float32) ([16]float32, bool) {
	// a is row-major [row][col] with the identity appended.
	var a [4][8]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			a[i][j] = float64(m[j*4+i])
		}
		a[i][4+i] = 1
	}
	for col := 0; col < 4; col++ {
		pivot := col
		for row := col + 1; row < 4; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if a[pivot][col] == 0 {
			return [16]float32{}, false
		}
		a[col], a[pivot] = a[pivot], a[col]
		d := a[col][col]
		for j := 0; j < 8; j++ {
			a[col][j] /= d
		}
		for row := 0; row < 4; row++ {
			if row == col {
				continue
			}
			f := a[row][col]
			for j := 0; j < 8; j++ {
				a[row][j] -= f * a[col][j]
			}
		}
	}
	var inv [16]float32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[j*4+i] = float32(a[i][4+j])
		}
	}
	return inv, true
}
