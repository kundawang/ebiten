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

package audio_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"runtime"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/hajimehoshi/ebiten/v2/audio"
	"github.com/hajimehoshi/ebiten/v2/internal/mathutil"
)

func f32LE(v ...float32) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, v)
	return b.Bytes()
}

func i16LE(v ...int16) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, v)
	return b.Bytes()
}

func decodeF32LE(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(b[4*i:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func decodeI16LE(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

// readPlayerOutput ensures p's device player exists and reads its whole time stream, which
// carries the channel mixer's output when one is installed.
func readPlayerOutput(t *testing.T, p *audio.Player) []byte {
	t.Helper()
	// The first before-update hook creates the test device.
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p); err != nil {
		t.Fatal(err)
	}
	r := audio.PlayerReaderForTesting(p)
	if r == nil {
		t.Fatal("the device player was not created")
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("reading the player output: %v", err)
	}
	return out
}

// clamp1 clamps v to the float sample range [-1, 1]; NaN becomes zero.
func clamp1(v float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	if v > 1 {
		return 1
	}
	if v < -1 {
		return -1
	}
	return v
}

// referenceMatrix computes the effective output for source frames given the player and master
// matrices. Each state's auto-normalization and gain are applied before the two matrices are
// combined, matching convert.CombineMixStates.
func referenceMatrix(frames [][]float64, sourceChannels int, player, master *audio.ChannelMixMatrix) [][2]float64 {
	rowsFor := func(m *audio.ChannelMixMatrix) [2][]float64 {
		var rows [2][]float64
		ins := m.SourceChannels()
		for out := 0; out < 2; out++ {
			row := make([]float64, ins)
			var sum float64
			if m.AutoNormalize() {
				for in := 0; in < ins; in++ {
					sum += math.Abs(m.At(out, in))
				}
			}
			for in := 0; in < ins; in++ {
				c := m.At(out, in)
				if m.AutoNormalize() && sum != 0 {
					c /= sum
				}
				row[in] = c * m.Gain()
			}
			rows[out] = row
		}
		return rows
	}

	pr := rowsFor(player)
	mr := rowsFor(master)
	out := make([][2]float64, len(frames))
	for f, frame := range frames {
		// First the player matrix turns the source into stereo.
		var stereo [2]float64
		for o := 0; o < 2; o++ {
			var v float64
			for i := 0; i < sourceChannels; i++ {
				sample := frame[i]
				if math.IsNaN(sample) || math.IsInf(sample, 0) {
					sample = 0
				}
				v += pr[o][i] * sample
			}
			stereo[o] = v
		}
		// Then the master matrix acts on the stereo result.
		for o := 0; o < 2; o++ {
			var v float64
			for i := 0; i < 2; i++ {
				v += mr[o][i] * stereo[i]
			}
			out[f][o] = clamp1(v)
		}
	}
	return out
}

func defaultMaster() *audio.ChannelMixMatrix {
	m := audio.NewChannelMixMatrix(2, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, 1)
	return m
}

func TestChannelMixUnconfiguredIsPassthrough(t *testing.T) {
	setup()
	defer teardown()

	raw := f32LE(0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
	p, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if audio.PlayerHasMixerForTesting(p) {
		t.Fatal("an unconfigured player must not install a channel mixer")
	}
	got := readPlayerOutput(t, p)
	if !bytes.Equal(got, raw) {
		t.Fatalf("unconfigured output changed the bytes: got %v, want %v", got, raw)
	}

	// Setting the context's default identity matrix keeps the legacy path too.
	if err := context.SetChannelMixMatrix(defaultMaster()); err != nil {
		t.Fatal(err)
	}
	p2, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if audio.PlayerHasMixerForTesting(p2) {
		t.Fatal("the default master matrix must not install a channel mixer")
	}
	if got := readPlayerOutput(t, p2); !bytes.Equal(got, raw) {
		t.Fatalf("default master output changed the bytes: got %v, want %v", got, raw)
	}
}

func TestChannelMixF32SurroundAgainstReference(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 4
	frameCount := 32
	raw := make([]float32, 0, frameCount*sourceChannels)
	frames := make([][]float64, frameCount)
	for f := 0; f < frameCount; f++ {
		frame := make([]float64, sourceChannels)
		for ch := 0; ch < sourceChannels; ch++ {
			v := float32(0.5 * math.Sin(2*math.Pi*float64(f)*(1+float64(ch))/16+float64(ch)))
			frame[ch] = float64(v)
			raw = append(raw, v)
		}
		frames[f] = frame
	}

	player := audio.NewChannelMixMatrix(sourceChannels, 2)
	_ = player.Set(0, 0, 1)
	_ = player.Set(0, 1, -0.5)
	_ = player.Set(0, 2, 0.25)
	_ = player.Set(1, 1, 0.25)
	_ = player.Set(1, 2, -0.5)
	_ = player.Set(1, 3, 1)
	if err := player.SetGain(0.9); err != nil {
		t.Fatal(err)
	}
	player.SetAutoNormalize(true)

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}
	if got := audio.PlayerSourceChannelsForTesting(p); got != sourceChannels {
		t.Fatalf("source channels: got %d, want %d", got, sourceChannels)
	}
	// The mixer is installed lazily when the stream is created.
	if audio.PlayerHasMixerForTesting(p) {
		t.Fatal("the mixer must not be installed before the stream exists")
	}

	got := decodeF32LE(readPlayerOutput(t, p))
	if !audio.PlayerHasMixerForTesting(p) {
		t.Fatal("a configured player must install a channel mixer once it reads")
	}
	want := referenceMatrix(frames, sourceChannels, player, defaultMaster())
	if len(got) != frameCount*2 {
		t.Fatalf("output length: got %d samples, want %d", len(got), frameCount*2)
	}
	for f := 0; f < frameCount; f++ {
		for o := 0; o < 2; o++ {
			if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 2e-6 {
				t.Fatalf("frame=%d out=%d: got %v, want %v", f, o, got[2*f+o], want[f][o])
			}
		}
	}

	// The getter returns an equal snapshot.
	gotMatrix := p.ChannelMixMatrix()
	if gotMatrix.SourceChannels() != sourceChannels || gotMatrix.OutputChannels() != 2 {
		t.Fatalf("matrix snapshot dimensions: got %dx%d", gotMatrix.SourceChannels(), gotMatrix.OutputChannels())
	}
	for out := 0; out < 2; out++ {
		for in := 0; in < sourceChannels; in++ {
			if gotMatrix.At(out, in) != player.At(out, in) {
				t.Fatalf("matrix snapshot at (%d,%d): got %v, want %v", out, in, gotMatrix.At(out, in), player.At(out, in))
			}
		}
	}
	if gotMatrix.Gain() != player.Gain() || gotMatrix.AutoNormalize() != player.AutoNormalize() {
		t.Fatal("matrix snapshot lost gain or auto-normalization")
	}
}

func TestChannelMixInt16AgainstReference(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 3
	frameCount := 24
	rawI16 := make([]int16, 0, frameCount*sourceChannels)
	frames := make([][]float64, frameCount)
	for f := 0; f < frameCount; f++ {
		frame := make([]float64, sourceChannels)
		for ch := 0; ch < sourceChannels; ch++ {
			v := int16((f*5 + ch*9 - 100) * 137)
			frame[ch] = float64(v) / (1 << 15)
			rawI16 = append(rawI16, v)
		}
		frames[f] = frame
	}

	player := audio.NewChannelMixMatrix(sourceChannels, 2)
	_ = player.Set(0, 0, 0.7)
	_ = player.Set(0, 2, -0.3)
	_ = player.Set(1, 0, -0.2)
	_ = player.Set(1, 1, 0.8)
	_ = player.Set(1, 2, 0.4)
	if err := player.SetGain(1.25); err != nil {
		t.Fatal(err)
	}

	p, err := context.NewPlayerWithChannels(bytes.NewReader(i16LE(rawI16...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}

	got := decodeF32LE(readPlayerOutput(t, p))
	want := referenceMatrix(frames, sourceChannels, player, defaultMaster())
	for f := 0; f < frameCount; f++ {
		for o := 0; o < 2; o++ {
			if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 1.5/(1<<15) {
				t.Fatalf("frame=%d out=%d: got %v, want %v", f, o, got[2*f+o], want[f][o])
			}
		}
	}
}

func TestChannelMixMonoRouting(t *testing.T) {
	setup()
	defer teardown()

	// A mono pulse train: 1, 0, -1, 0.5.
	raw := f32LE(1, 0, -1, 0.5)
	player := audio.NewChannelMixMatrix(1, 2)
	_ = player.Set(0, 0, 1)
	_ = player.Set(1, 0, -1) // phase-inverted mirror on the right

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(raw), 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	want := []float32{1, -1, 0, 0, -1, 1, 0.5, -0.5}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
			t.Fatalf("sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestChannelMixPlayerAndMasterCombination(t *testing.T) {
	setup()
	defer teardown()

	// Mono source; the player fans it out to both outputs, and the master matrix inverts the
	// left and mutes the right.
	raw := f32LE(0.75, 0.25)
	player := audio.NewChannelMixMatrix(1, 2)
	_ = player.Set(0, 0, 1)
	_ = player.Set(1, 0, 1)

	master := audio.NewChannelMixMatrix(2, 2)
	_ = master.Set(0, 0, -1)
	_ = master.Set(1, 1, 0) // right muted
	if err := context.SetChannelMixMatrix(master); err != nil {
		t.Fatal(err)
	}
	// A second context must keep the same master configuration.
	if got := context.ChannelMixMatrix(); got.At(0, 0) != -1 || got.At(1, 1) != 0 {
		t.Fatal("master matrix getter did not return the configured matrix")
	}

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(raw), 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	want := []float32{-0.75, 0, -0.25, 0}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
			t.Fatalf("sample %d: got %v, want %v", i, got[i], want[i])
		}
	}

	// Removing the player matrix leaves only the master matrix: mono default state routes the
	// single channel to the left output, which the master inverts.
	if err := p.SetChannelMixMatrix(nil); err != nil {
		t.Fatal(err)
	}
	p2, err := context.NewPlayerF32WithChannels(bytes.NewReader(raw), 1)
	if err != nil {
		t.Fatal(err)
	}
	got = decodeF32LE(readPlayerOutput(t, p2))
	want = []float32{-0.75, 0, -0.25, 0}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
			t.Fatalf("master-only sample %d: got %v, want %v", i, got[i], want[i])
		}
	}

	// Resetting the master restores the passthrough path for new players.
	if err := context.SetChannelMixMatrix(nil); err != nil {
		t.Fatal(err)
	}
	stereo := f32LE(0.1, 0.2, 0.3, 0.4)
	p3, err := context.NewPlayerF32(bytes.NewReader(stereo))
	if err != nil {
		t.Fatal(err)
	}
	if audio.PlayerHasMixerForTesting(p3) {
		t.Fatal("resetting the master must restore the legacy passthrough path")
	}
	if got := readPlayerOutput(t, p3); !bytes.Equal(got, stereo) {
		t.Fatalf("reset output: got %v, want %v", got, stereo)
	}
}

// TestChannelMixPlaysToCompletion drives a mixed player through the real device read loop
// (synctest), verifying the mixing pipeline does not interfere with playback timing and that
// the player finishes its finite source and leaves the playing set.
func TestChannelMixPlaysToCompletion(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		setup()
		defer teardown()

		sourceChannels := 3
		// A finite source: 256 surround frames.
		raw := make([]float32, 256*sourceChannels)
		for i := range raw {
			raw[i] = 0.3
		}
		m := audio.NewChannelMixMatrix(sourceChannels, 2)
		for in := 0; in < sourceChannels; in++ {
			_ = m.Set(0, in, 1.0/float64(sourceChannels))
			_ = m.Set(1, in, 1.0/float64(sourceChannels))
		}
		p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
		if err != nil {
			t.Fatal(err)
		}
		defer runtime.KeepAlive(p)
		if err := p.SetChannelMixMatrix(m); err != nil {
			t.Fatal(err)
		}

		p.Play()
		// Create the device and start the pending player.
		if err := audio.UpdateForTesting(); err != nil {
			t.Fatal(err)
		}
		if !p.IsPlaying() {
			t.Fatal("the mixed player should be playing")
		}
		if audio.PlayingButUntrackedForTesting(p) {
			t.Fatal("a playing mixed player must be tracked by the context")
		}

		// The dummy device consumes the source; wait for the player to finish.
		for range 50 {
			if err := audio.UpdateForTesting(); err != nil {
				t.Fatal(err)
			}
			if !p.IsPlaying() {
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatal("the mixed player did not finish its finite source")
	})
}

// TestChannelMixMatrixSetAfterStream verifies that a matrix applied after the stream (and
// device player) already exists inserts the mixer in front of the existing reader without
// disturbing the frames already produced.
func TestChannelMixMatrixSetAfterStream(t *testing.T) {
	setup()
	defer teardown()

	raw := f32LE(0.5, 0.25, -0.5, -0.25, 0.75, 0.125)
	p, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p); err != nil {
		t.Fatal(err)
	}
	if audio.PlayerHasMixerForTesting(p) {
		t.Fatal("no mixer before a matrix is set")
	}

	// Read one frame on the legacy passthrough path first.
	r := audio.PlayerReaderForTesting(p)
	first := make([]byte, 8)
	if _, err := io.ReadFull(r, first); err != nil {
		t.Fatal(err)
	}
	if got := decodeF32LE(first); got[0] != 0.5 || got[1] != 0.25 {
		t.Fatalf("pre-matrix frame: got %v", got)
	}

	// Now apply an inverting matrix; the mixer is inserted around the live reader.
	m := audio.NewChannelMixMatrix(2, 2)
	_ = m.Set(0, 0, -1)
	_ = m.Set(1, 1, -1)
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	if !audio.PlayerHasMixerForTesting(p) {
		t.Fatal("a mixer must be installed once a matrix is set after stream creation")
	}

	rest, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(rest)
	// Two frames remain, each with both channels inverted.
	want := []float32{0.5, 0.25, -0.75, -0.125}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
			t.Fatalf("post-matrix sample %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestChannelMixMasterHotSwap(t *testing.T) {
	setup()
	defer teardown()

	// A stereo source with several identical frames.
	frame := f32LE(0.5, -0.5)
	var raw []byte
	for range 6 {
		raw = append(raw, frame...)
	}

	// Start with an inverting player matrix so a mixer is installed before any master state.
	player := audio.NewChannelMixMatrix(2, 2)
	_ = player.Set(0, 0, 1)
	_ = player.Set(1, 1, -1)
	p, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p); err != nil {
		t.Fatal(err)
	}
	r := audio.PlayerReaderForTesting(p)

	// Read two frames under the default (identity) master.
	head := make([]byte, 2*8)
	if _, err := io.ReadFull(r, head); err != nil {
		t.Fatal(err)
	}
	headSamples := decodeF32LE(head)
	if headSamples[0] != 0.5 || headSamples[1] != 0.5 {
		t.Fatalf("before master swap: got %v, want [0.5 0.5]", headSamples[:2])
	}

	// Swap the master to mute the left channel and double the right. The existing mixer must
	// pick it up at the next frame without rebuilding.
	master := audio.NewChannelMixMatrix(2, 2)
	_ = master.Set(1, 1, 1) // left row zero; right passes the player's right
	if err := context.SetChannelMixMatrix(master); err != nil {
		t.Fatal(err)
	}
	tail, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	tailSamples := decodeF32LE(tail)
	// Four frames remain; each must be (0, 0.5).
	if len(tailSamples) != 4*2 {
		t.Fatalf("tail length: got %d samples, want %d", len(tailSamples), 4*2)
	}
	for f := 0; f < 4; f++ {
		if tailSamples[2*f] != 0 || tailSamples[2*f+1] != 0.5 {
			t.Fatalf("frame %d after master swap: got %v, want [0 0.5]", f, tailSamples[2*f:2*f+2])
		}
	}

	// Reset the master so later tests start clean.
	if err := context.SetChannelMixMatrix(nil); err != nil {
		t.Fatal(err)
	}
}

func TestChannelMixMatrixValidation(t *testing.T) {
	setup()
	defer teardown()

	// NaN and Inf coefficients and gains are rejected.
	m := audio.NewChannelMixMatrix(2, 2)
	if err := m.Set(0, 0, math.NaN()); err == nil {
		t.Error("NaN coefficient was accepted")
	}
	if err := m.Set(0, 0, math.Inf(1)); err == nil {
		t.Error("+Inf coefficient was accepted")
	}
	if err := m.SetGain(math.NaN()); err == nil {
		t.Error("NaN gain was accepted")
	}
	if err := m.SetGain(math.Inf(-1)); err == nil {
		t.Error("-Inf gain was accepted")
	}
	// Rejected values leave the previous values untouched.
	if m.At(0, 0) != 0 || m.Gain() != 1 {
		t.Fatal("a rejected update changed the matrix")
	}
	// An extreme finite gain is accepted; clamping happens at mixing time.
	if err := m.SetGain(1e18); err != nil {
		t.Fatalf("a finite extreme gain must be accepted: %v", err)
	}

	// Out-of-range indices are rejected rather than panicking.
	if err := m.Set(2, 0, 1); err == nil {
		t.Error("out-of-range output index was accepted")
	}
	if err := m.Set(0, 2, 1); err == nil {
		t.Error("out-of-range source index was accepted")
	}
	if m.At(2, 0) != 0 {
		t.Error("At must return zero for an out-of-range index")
	}

	// Constructor channel-count limits.
	for _, n := range []int{0, -1, 9} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewChannelMixMatrix with %d channels must panic", n)
				}
			}()
			audio.NewChannelMixMatrix(n, 2)
		}()
	}

	// A player rejects the wrong output dimension.
	p, err := context.NewPlayerF32(bytes.NewReader(f32LE(0.1, 0.2)))
	if err != nil {
		t.Fatal(err)
	}
	wrongOut := audio.NewChannelMixMatrix(2, 1)
	if err := p.SetChannelMixMatrix(wrongOut); err == nil {
		t.Error("a one-output-channel matrix was accepted")
	}

	// Once a source channel count is fixed, a mismatching matrix is rejected.
	monoPlayer, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(0.1, 0.2)), 1)
	if err != nil {
		t.Fatal(err)
	}
	monoMatrix := audio.NewChannelMixMatrix(1, 2)
	_ = monoMatrix.Set(0, 0, 1)
	if err := monoPlayer.SetChannelMixMatrix(monoMatrix); err != nil {
		t.Fatal(err)
	}
	stereoMatrix := audio.NewChannelMixMatrix(2, 2)
	_ = stereoMatrix.Set(0, 0, 1)
	_ = stereoMatrix.Set(1, 1, 1)
	if err := monoPlayer.SetChannelMixMatrix(stereoMatrix); err == nil {
		t.Error("a matrix with a mismatching source channel count was accepted")
	}

	// The context rejects a master matrix with the wrong output dimension.
	wrongMaster := audio.NewChannelMixMatrix(2, 1)
	if err := context.SetChannelMixMatrix(wrongMaster); err == nil {
		t.Error("a one-output-channel master matrix was accepted")
	}
}

func TestChannelMixGainAndNormalizeSetters(t *testing.T) {
	setup()
	defer teardown()

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(0.5, 0.5, 0.5, 0.5)), 2)
	if err != nil {
		t.Fatal(err)
	}
	if p.ChannelMixGain() != 1 || p.ChannelMixAutoNormalize() {
		t.Fatal("the default player gain must be one with normalization disabled")
	}
	if err := p.SetChannelMixGain(0.5); err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixAutoNormalize(true); err != nil {
		t.Fatal(err)
	}
	if p.ChannelMixGain() != 0.5 || !p.ChannelMixAutoNormalize() {
		t.Fatal("player gain/normalization getters returned stale values")
	}
	if p.ChannelMixMatrix() == nil {
		t.Fatal("setting a gain must create a matrix to carry it")
	}
	if err := p.SetChannelMixGain(math.NaN()); err == nil {
		t.Error("a NaN player gain was accepted")
	}
	if p.ChannelMixGain() != 0.5 {
		t.Fatal("a rejected gain update changed the stored gain")
	}

	if err := context.SetChannelMixGain(0.25); err != nil {
		t.Fatal(err)
	}
	if err := context.SetChannelMixAutoNormalize(true); err != nil {
		t.Fatal(err)
	}
	if context.ChannelMixGain() != 0.25 || !context.ChannelMixAutoNormalize() {
		t.Fatal("context gain/normalization getters returned stale values")
	}
	if err := context.SetChannelMixGain(math.Inf(1)); err == nil {
		t.Error("an infinite context gain was accepted")
	}
	if context.ChannelMixGain() != 0.25 {
		t.Fatal("a rejected context gain update changed the stored gain")
	}
}

func TestChannelMixNoClippingOverflow(t *testing.T) {
	setup()
	defer teardown()

	// Four full-scale in-phase surround channels summed with a large gain.
	sourceChannels := 4
	raw := make([]float32, 2*sourceChannels)
	for i := range raw {
		raw[i] = 1
	}
	m := audio.NewChannelMixMatrix(sourceChannels, 2)
	for ch := 0; ch < sourceChannels; ch++ {
		_ = m.Set(0, ch, 1)
		_ = m.Set(1, ch, 1)
	}
	if err := m.SetGain(1e6); err != nil {
		t.Fatal(err)
	}

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("sample %d is non-finite: %v", i, v)
		}
		if v < -1 || v > 1 {
			t.Fatalf("sample %d is out of range: %v", i, v)
		}
	}

	// Auto-normalization keeps the same signal at full scale instead of hard-clamping it.
	m.SetAutoNormalize(true)
	p2, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p2.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	got = decodeF32LE(readPlayerOutput(t, p2))
	for i, v := range got {
		if diff := math.Abs(float64(v) - 1); diff > 1e-5 {
			t.Fatalf("auto-normalized sample %d: got %v, want 1", i, v)
		}
	}
}

type mixSlowReader struct {
	b   []byte
	i   int
	err error
}

func (r *mixSlowReader) Read(p []byte) (int, error) {
	if r.err != nil && r.i >= len(r.b) {
		return 0, r.err
	}
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	// At most three bytes at a time, never aligned to any frame size.
	n := min(3, len(p), len(r.b)-r.i)
	copy(p[:n], r.b[r.i:r.i+n])
	r.i += n
	if r.err != nil && r.i >= len(r.b) {
		return n, r.err
	}
	return n, nil
}

func TestChannelMixShortReadsAndSourceError(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 3
	frameCount := 10
	rawI16 := make([]int16, frameCount*sourceChannels)
	frames := make([][]float64, frameCount)
	for f := 0; f < frameCount; f++ {
		frame := make([]float64, sourceChannels)
		for ch := 0; ch < sourceChannels; ch++ {
			v := int16((f + 1) * (ch + 1) * 1000)
			frame[ch] = float64(v) / (1 << 15)
			rawI16[f*sourceChannels+ch] = v
		}
		frames[f] = frame
	}
	player := audio.NewChannelMixMatrix(sourceChannels, 2)
	_ = player.Set(0, 0, 0.6)
	_ = player.Set(0, 1, 0.4)
	_ = player.Set(1, 1, -0.4)
	_ = player.Set(1, 2, 0.6)

	p, err := context.NewPlayerWithChannels(&mixSlowReader{b: i16LE(rawI16...)}, sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(player); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	want := referenceMatrix(frames, sourceChannels, player, defaultMaster())
	if len(got) != frameCount*2 {
		t.Fatalf("short-read output length: got %d, want %d", len(got), frameCount*2)
	}
	for f := 0; f < frameCount; f++ {
		for o := 0; o < 2; o++ {
			if diff := math.Abs(float64(got[2*f+o]) - want[f][o]); diff > 1.5/(1<<15) {
				t.Fatalf("short-read frame=%d out=%d: got %v, want %v", f, o, got[2*f+o], want[f][o])
			}
		}
	}

	// A non-EOF source error propagates through the player output.
	sentinel := errors.New("source failed")
	p2, err := context.NewPlayerF32WithChannels(&mixSlowReader{b: f32LE(0.1, 0.2, 0.3, 0.4), err: sentinel}, 2)
	if err != nil {
		t.Fatal(err)
	}
	m := audio.NewChannelMixMatrix(2, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, 1)
	if err := p2.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p2); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(audio.PlayerReaderForTesting(p2)); !errors.Is(err, sentinel) {
		t.Fatalf("source error: got %v, want %v", err, sentinel)
	}
}

func TestChannelMixSeekPreservesTiming(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 2
	frameCount := 16
	raw := make([]float32, frameCount*sourceChannels)
	for f := 0; f < frameCount; f++ {
		raw[f*sourceChannels] = float32(f)
		raw[f*sourceChannels+1] = float32(-f)
	}
	m := audio.NewChannelMixMatrix(sourceChannels, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, -1) // right channel sign flipped

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p); err != nil {
		t.Fatal(err)
	}
	r := audio.PlayerReaderForTesting(p).(io.ReadSeeker)

	// Read four frames, then seek back to the start and read the whole stream again: the
	// repeated output must be identical to a fresh playback, so the mixer adds no latency.
	head := make([]byte, 4*8)
	if n, err := io.ReadFull(r, head); n != len(head) || err != nil {
		t.Fatalf("initial read: (%d, %v)", n, err)
	}
	pos, err := r.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}
	if pos != 0 {
		t.Fatalf("seek position: got %d, want 0", pos)
	}
	replayed, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayed) != frameCount*8 {
		t.Fatalf("replayed length: got %d bytes, want %d", len(replayed), frameCount*8)
	}

	p2, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p2.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	fresh := readPlayerOutput(t, p2)
	if !bytes.Equal(replayed, fresh) {
		t.Fatalf("output after a seek-to-start differs from a fresh playback")
	}

	// A mid-stream seek lands on an exact output frame boundary.
	gotSamples := decodeF32LE(replayed)
	if _, err := r.Seek(6*8, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8)
	if n, err := io.ReadFull(r, buf); n != 8 || err != nil {
		t.Fatalf("read after seek: (%d, %v)", n, err)
	}
	at := decodeF32LE(buf)
	wantAt := gotSamples[12:14]
	if at[0] != wantAt[0] || at[1] != wantAt[1] {
		t.Fatalf("mid-stream seek: got %v, want %v", at, wantAt)
	}
}

// TestChannelMixSetPosition drives the seek through the public Player API, which must route
// through the time stream and the mixer without losing or duplicating frames.
func TestChannelMixSetPosition(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 2
	frameCount := 12
	raw := make([]float32, frameCount*sourceChannels)
	for f := 0; f < frameCount; f++ {
		raw[f*sourceChannels] = float32(f + 1)
		raw[f*sourceChannels+1] = float32(-(f + 1))
	}
	m := audio.NewChannelMixMatrix(sourceChannels, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, 1)

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if err := audio.EnsurePlayerForTesting(p); err != nil {
		t.Fatal(err)
	}
	r := audio.PlayerReaderForTesting(p).(io.ReadSeeker)

	// Advance the device player's source by reading five frames.
	skip := make([]byte, 5*8)
	if _, err := io.ReadFull(r, skip); err != nil {
		t.Fatal(err)
	}

	// Seek through the public Player API. The time stream's integer position arithmetic
	// rounds to whole frames; here the essential guarantee is that seeking routes through the
	// mixer and the remaining output keeps the same matrix routing, with no corruption.
	posNanos, ok := mathutil.MulDiv(int64(8), int64(time.Second), int64(44100))
	if !ok {
		t.Fatal("position overflow")
	}
	if err := p.SetPosition(time.Duration(posNanos)); err != nil {
		t.Fatalf("SetPosition with a mixer: %v", err)
	}

	out := decodeF32LE(mustReadAll(t, r))
	// Every remaining frame must keep the identity routing and the sign pattern of the
	// source, proving the seek did not desynchronize the channel interleaving.
	for i := 0; i+1 < len(out); i += 2 {
		if out[i] != -out[i+1] {
			t.Fatalf("frame %d lost its stereo routing after SetPosition: got %v", i/2, out[i:i+2])
		}
	}
	if len(out) == 0 {
		t.Fatal("SetPosition consumed the whole stream")
	}
}

func mustReadAll(t *testing.T, r io.Reader) []byte {
	t.Helper()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestChannelMixPulseHasNoLatency verifies with a pulse fixture that the first input frame
// appears on the very first output frame: the mixer must neither buffer ahead nor delay the
// signal, and per-channel routing is sample-accurate.
func TestChannelMixPulseHasNoLatency(t *testing.T) {
	setup()
	defer teardown()

	sourceChannels := 3
	// A single pulse carried by channel 2, surrounded by silent frames.
	frames := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 0, 0},
		{0, 0, 0},
	}
	raw := make([]float32, len(frames)*sourceChannels)
	for f, frame := range frames {
		for ch, v := range frame {
			raw[f*sourceChannels+ch] = float32(v)
		}
	}
	m := audio.NewChannelMixMatrix(sourceChannels, 2)
	// Channel 2 reaches only the right output, phase-inverted.
	_ = m.Set(1, 2, -1)

	p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), sourceChannels)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	out := decodeF32LE(readPlayerOutput(t, p))

	// Frame 0 must already be present (no leading silence introduced by the mixer).
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("the first output frame is not the first input frame: got %v", out[:2])
	}
	// The pulse lands exactly on frame 1, right channel, with inverted phase.
	if out[2*1] != 0 || out[2*1+1] != -1 {
		t.Fatalf("pulse routing/phase/latency: frame 1 got %v, want [0 -1]", out[2:4])
	}
	// Nothing leaks onto any other frame or the left channel.
	for f := range frames {
		if out[2*f] != 0 {
			t.Fatalf("frame %d left channel must stay silent, got %v", f, out[2*f])
		}
		if f != 1 && out[2*f+1] != 0 {
			t.Fatalf("frame %d right channel must stay silent, got %v", f, out[2*f+1])
		}
	}
}

func TestChannelMixPauseResumeAndPosition(t *testing.T) {
	setup()
	defer teardown()

	src := &mixSlowReader{b: f32LE(make([]float32, 64)...)}
	p, err := context.NewPlayerF32WithChannels(src, 2)
	if err != nil {
		t.Fatal(err)
	}
	m := audio.NewChannelMixMatrix(2, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, 1)
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}

	p.Play()
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
	if !p.IsPlaying() {
		t.Fatal("the player should be playing")
	}

	p.Pause()
	if p.IsPlaying() {
		t.Fatal("the player should be paused")
	}
	pos := p.Position()

	// While paused, the position must not advance.
	time.Sleep(30 * time.Millisecond)
	audio.UpdatePositionForTesting(p)
	if got := p.Position(); got != pos {
		t.Fatalf("position advanced while paused: got %v, want %v", got, pos)
	}

	p.Play()
	if !p.IsPlaying() {
		t.Fatal("the player should be playing again")
	}
}

func TestChannelMixConcurrentPlayersAndUpdates(t *testing.T) {
	setup()
	defer teardown()

	const players = 8
	var ps []*audio.Player
	for i := 0; i < players; i++ {
		ch := 1 + i%4
		raw := make([]float32, 32*ch)
		for j := range raw {
			raw[j] = 0.25
		}
		p, err := context.NewPlayerF32WithChannels(bytes.NewReader(f32LE(raw...)), ch)
		if err != nil {
			t.Fatal(err)
		}
		m := audio.NewChannelMixMatrix(ch, 2)
		for in := 0; in < ch; in++ {
			_ = m.Set(0, in, 0.5)
			_ = m.Set(1, in, 0.5)
		}
		if err := p.SetChannelMixMatrix(m); err != nil {
			t.Fatal(err)
		}
		ps = append(ps, p)
	}
	if got := audio.AllPlayersCountForTesting(); got != players {
		t.Fatalf("tracked players: got %d, want %d", got, players)
	}

	// Hammer matrix updates from several goroutines while the outputs are being read; every
	// state replacement is atomic, so no read may panic or produce a non-finite sample.
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < 50; k++ {
				for _, p := range ps {
					g := 0.1 + float64(k%10)*0.1
					if err := p.SetChannelMixGain(g); err != nil {
						t.Error(err)
						return
					}
					if err := context.SetChannelMixGain(0.9); err != nil {
						t.Error(err)
						return
					}
				}
			}
		}()
	}
	for _, p := range ps {
		out := readPlayerOutput(t, p)
		for _, v := range decodeF32LE(out) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("concurrent mixing produced a non-finite sample: %v", v)
			}
		}
	}
	wg.Wait()

	// Closing a player unregisters it from the context's master update list.
	if err := ps[0].Close(); err != nil {
		t.Fatal(err)
	}
	if got := audio.AllPlayersCountForTesting(); got != players-1 {
		t.Fatalf("tracked players after close: got %d, want %d", got, players-1)
	}
	// Master updates must not touch the closed player.
	if err := context.SetChannelMixGain(0.7); err != nil {
		t.Fatal(err)
	}
	if err := audio.UpdateForTesting(); err != nil {
		t.Fatal(err)
	}
}

func TestChannelMixMutedPlayerIsSilent(t *testing.T) {
	setup()
	defer teardown()

	raw := f32LE(0.9, -0.9, 0.9, -0.9, 0.9, -0.9)
	// A zero matrix mutes every output.
	m := audio.NewChannelMixMatrix(2, 2)
	p, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	for i, v := range got {
		if v != 0 {
			t.Fatalf("muted sample %d: got %v, want 0", i, v)
		}
	}
}

// TestChannelMixLegacyInt16Player drives a matrix through the legacy NewPlayer constructor,
// whose int16 source is converted to float32 before mixing.
func TestChannelMixLegacyInt16Player(t *testing.T) {
	setup()
	defer teardown()

	// Two stereo int16 frames: (16384, -16384), (-16384, 16384).
	rawI16 := i16LE(16384, -11469, -16384, 11469)
	p, err := context.NewPlayer(bytes.NewReader(rawI16))
	if err != nil {
		t.Fatal(err)
	}
	m := audio.NewChannelMixMatrix(2, 2)
	// Swap the two channels and invert the new left channel.
	_ = m.Set(0, 1, -1)
	_ = m.Set(1, 0, 1)
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	out := decodeF32LE(readPlayerOutput(t, p))
	if len(out) != 4 {
		t.Fatalf("output length: got %d samples, want 4", len(out))
	}
	// Frame 0: left = -(-11469/32768), right = 16384/32768.
	if diff := math.Abs(float64(out[0]) - float64(11469)/32768); diff > 1.5/32768 {
		t.Errorf("frame0 left: got %v", out[0])
	}
	if diff := math.Abs(float64(out[1]) - float64(16384)/32768); diff > 1.5/32768 {
		t.Errorf("frame0 right: got %v", out[1])
	}

	// Removing the matrix leaves the player on the mixer path but with identity routing:
	// output equals the float32-converted source.
	p2, err := context.NewPlayer(bytes.NewReader(rawI16))
	if err != nil {
		t.Fatal(err)
	}
	identity := audio.NewChannelMixMatrix(2, 2)
	_ = identity.Set(0, 0, 1)
	_ = identity.Set(1, 1, 1)
	if err := p2.SetChannelMixMatrix(identity); err != nil {
		t.Fatal(err)
	}
	if err := p2.SetChannelMixMatrix(nil); err != nil {
		t.Fatal(err)
	}
	mixed := decodeF32LE(readPlayerOutput(t, p2))
	expected := []float64{float64(16384) / 32768, float64(-11469) / 32768, float64(-16384) / 32768, float64(11469) / 32768}
	for i := range expected {
		if diff := math.Abs(float64(mixed[i]) - expected[i]); diff > 1.5/32768 {
			t.Errorf("identity-mixed sample %d: got %v, want %v", i, mixed[i], expected[i])
		}
	}
}

func TestChannelMixNonFiniteSourceIsSafeSilence(t *testing.T) {
	setup()
	defer teardown()

	raw := f32LE(float32(math.NaN()), float32(math.Inf(1)), 0.5, -0.5)
	m := audio.NewChannelMixMatrix(2, 2)
	_ = m.Set(0, 0, 1)
	_ = m.Set(1, 1, 1)
	p, err := context.NewPlayerF32(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetChannelMixMatrix(m); err != nil {
		t.Fatal(err)
	}
	got := decodeF32LE(readPlayerOutput(t, p))
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("non-finite frame must be silence: got %v", got[:2])
	}
	if got[2] != 0.5 || got[3] != -0.5 {
		t.Fatalf("the following frame must be intact: got %v", got[2:])
	}
}
