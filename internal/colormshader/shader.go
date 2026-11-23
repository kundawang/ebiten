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

//go:generate go run gen.go
//go:generate gofmt -s -w .

package colormshader

import (
	"bytes"
	"fmt"
	"sync"
	"text/template"
)

// Filter must have the same values as builtinshader.Filter.
// In v3, colormshader can be moved into the colorm package, making syncing easier.
type Filter int

const (
	FilterNearest Filter = iota
	FilterLinear
	FilterPixelated
)

const FilterCount = 3

// Address must have the same values as builtinshader.Address.
// In v3, colormshader can be moved into the colorm package, making syncing easier.
type Address int

const (
	AddressUnsafe Address = iota
	AddressClampToZero
	AddressRepeat
)

const AddressCount = 3

const (
	UniformColorMBody        = "ColorMBody"
	UniformColorMTranslation = "ColorMTranslation"

	UniformLUTScale       = "LUTScale"
	UniformLUTTranslation = "LUTTranslation"
	UniformLUTGamma       = "LUTGamma"
)

var (
	shaders  [FilterCount][AddressCount][2][]byte
	shadersM sync.Mutex
)

var tmpl = template.Must(template.New("tmpl").Parse(`//kage:unit pixels

package main

{{if .LUT}}
var LUTScale float
var LUTTranslation vec4
var LUTGamma float
{{else}}
var ColorMBody mat4
var ColorMTranslation vec4
{{end}}

{{if eq .Address .AddressRepeat}}
func adjustSrcPosForAddressRepeat(p vec2) vec2 {
	origin := imageSrc0Origin()
	size := imageSrc0Size()
	return mod(p - origin, size) + origin
}
{{end}}

{{if .LUT}}
// lutDecode decodes a 16-bit signed fixed-point value stored in two 8-bit
// channels. The encoded range is [-LUTScale, LUTScale].
func lutDecode(p vec2) float {
	return ((p.x*255.0*256.0+p.y*255.0)/65535.0*2.0 - 1.0) * LUTScale
}

// lutAccumulate applies the color transform by summing per-channel lookup
// results. The LUT image is n texels wide and 8 texels tall. For input
// channel j, row 2*j holds output channels 0 (RG) and 1 (BA), and row 2*j+1
// holds output channels 2 (RG) and 3 (BA). pos is in texel units.
//
// imageSrc1UnsafeAt takes a position in source 0's texture space, so the
// LUT texel positions are offset by imageSrc0Origin().
func lutAccumulate(pos vec4) vec4 {
	acc := LUTTranslation
	o := imageSrc0Origin()
	t := imageSrc1UnsafeAt(o + vec2(pos.x+0.5, 0.5))
	acc.x += lutDecode(t.rg)
	acc.y += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.x+0.5, 1.5))
	acc.z += lutDecode(t.rg)
	acc.w += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.y+0.5, 2.5))
	acc.x += lutDecode(t.rg)
	acc.y += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.y+0.5, 3.5))
	acc.z += lutDecode(t.rg)
	acc.w += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.z+0.5, 4.5))
	acc.x += lutDecode(t.rg)
	acc.y += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.z+0.5, 5.5))
	acc.z += lutDecode(t.rg)
	acc.w += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.w+0.5, 6.5))
	acc.x += lutDecode(t.rg)
	acc.y += lutDecode(t.ba)
	t = imageSrc1UnsafeAt(o + vec2(pos.w+0.5, 7.5))
	acc.z += lutDecode(t.rg)
	acc.w += lutDecode(t.ba)
	return acc
}
{{end}}

func Fragment(dstPos vec4, src0Pos vec2, color vec4) vec4 {
{{if eq .Filter .FilterNearest}}
{{if eq .Address .AddressUnsafe}}
	clr := imageSrc0UnsafeAt(src0Pos)
{{else if eq .Address .AddressClampToZero}}
	clr := imageSrc0At(src0Pos)
{{else if eq .Address .AddressRepeat}}
	clr := imageSrc0At(adjustSrcPosForAddressRepeat(src0Pos))
{{end}}
{{else}}
{{if eq .Filter .FilterLinear}}
	p0 := src0Pos - 1/2.0
	p1 := src0Pos + 1/2.0
{{else if eq .Filter .FilterPixelated}}
	// inversedScale is the size of the region on the source image.
	// Include both screen axes so that rotating the image does not change its scale.
	dx, dy := dfdx(src0Pos), dfdy(src0Pos)
	inversedScale := vec2(length(vec2(dx.x, dy.x)), length(vec2(dx.y, dy.y)))
	// Cap the inversedScale to 1 as dfdx/dfdy is not accurate on some machines (#3182).
	inversedScale = min(inversedScale, vec2(1))
	p0 := src0Pos - inversedScale/2.0
	p1 := src0Pos + inversedScale/2.0
{{end}}

{{if eq .Address .AddressRepeat}}
	p0 = adjustSrcPosForAddressRepeat(p0)
	p1 = adjustSrcPosForAddressRepeat(p1)
{{end}}

{{if eq .Address .AddressUnsafe}}
	c0 := imageSrc0UnsafeAt(p0)
	c1 := imageSrc0UnsafeAt(vec2(p1.x, p0.y))
	c2 := imageSrc0UnsafeAt(vec2(p0.x, p1.y))
	c3 := imageSrc0UnsafeAt(p1)
{{else}}
	c0 := imageSrc0At(p0)
	c1 := imageSrc0At(vec2(p1.x, p0.y))
	c2 := imageSrc0At(vec2(p0.x, p1.y))
	c3 := imageSrc0At(p1)
{{end}}

{{if eq .Filter .FilterLinear}}
	rate := fract(p1)
{{else if eq .Filter .FilterPixelated}}
	rate := clamp(fract(p1)/inversedScale, 0, 1)
{{end}}
	clr := mix(mix(c0, c1, rate.x), mix(c2, c3, rate.x), rate.y)
{{end}}

	// Un-premultiply alpha.
	// When the alpha is 0, 1-sign(alpha) is 1.0, which means division does nothing.
	clr.rgb /= clr.a + (1-sign(clr.a))
{{if .LUT}}
	// Apply the color transform via the LUT.
	q := clamp(clr, 0, 1)
	if LUTGamma != 1.0 {
		q = pow(q, vec4(1.0/LUTGamma))
	}
	n := imageSrc1Size().x
	clr = lutAccumulate(q * (n - 1.0))
{{else}}
	// Apply the ColorM matrix.
	clr = (ColorMBody * clr) + ColorMTranslation
{{end}}
	// Premultiply alpha
	clr.rgb *= clr.a
	// Apply the color scale.
	clr *= color
	// Clamp the output.
	clr.rgb = min(clr.rgb, clr.a)

	return clr
}

`))

// ShaderSource returns the ColorM shader source based on the given parameters.
// If lut is true, the shader applies the color transform through a lookup
// table texture (source image 1) instead of a matrix.
func ShaderSource(filter Filter, address Address, lut bool) []byte {
	shadersM.Lock()
	defer shadersM.Unlock()

	lutIdx := 0
	if lut {
		lutIdx = 1
	}
	if s := shaders[filter][address][lutIdx]; s != nil {
		return s
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, struct {
		Filter             Filter
		FilterNearest      Filter
		FilterLinear       Filter
		FilterPixelated    Filter
		Address            Address
		AddressUnsafe      Address
		AddressClampToZero Address
		AddressRepeat      Address
		LUT                bool
	}{
		Filter:             filter,
		FilterNearest:      FilterNearest,
		FilterLinear:       FilterLinear,
		FilterPixelated:    FilterPixelated,
		Address:            address,
		AddressUnsafe:      AddressUnsafe,
		AddressClampToZero: AddressClampToZero,
		AddressRepeat:      AddressRepeat,
		LUT:                lut,
	}); err != nil {
		panic(fmt.Sprintf("colormshader: tmpl.Execute failed: %v", err))
	}

	b := buf.Bytes()
	shaders[filter][address][lutIdx] = b
	return b
}
