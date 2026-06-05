// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// ChaCha12 PRNG compatible with Rust's StdRng (rand 0.8+, ChaCha12Rng).
//
// Implements:
//   1. seed_from_u64 expansion via splitmix64 → 32-byte key
//   2. ChaCha12 block function (6 double rounds)
//   3. Float generation matching rand::Standard for f32:
//      f32::from_bits((u32 >> 9) | 0x3f800000) - 1.0  → [0, 1)
//      then × 2 − 1  → [−1, 1)  (as used in rebuild_proj)
//
// State layout (Bernstein / rand_chacha 0.3):
//   [0..3]   = constants ("expand 32-byte k")
//   [4..11]  = key (32 bytes as 8 × u32 LE)
//   [12..13] = counter (64-bit; we only ever need the lower word for dim ≤ 4096)
//   [14..15] = nonce (zero for from_seed)
package ailake

import (
	"encoding/binary"
	"math"
)

// splitmix64Expand expands a u64 seed to a 32-byte ChaCha12 key, matching
// Rust rand 0.8 SeedableRng::seed_from_u64 (4 × splitmix64 rounds → 32 bytes LE).
func splitmix64Expand(seed uint64) [32]byte {
	var out [32]byte
	for i := 0; i < 4; i++ {
		seed += 0x9e3779b97f4a7c15
		z := seed
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z ^= z >> 31
		binary.LittleEndian.PutUint64(out[i*8:], z)
	}
	return out
}

// chacha12Rng is a ChaCha12 PRNG. Matches Rust ChaCha12Rng::from_seed output.
type chacha12Rng struct {
	state  [16]uint32
	output [16]uint32
	idx    int // next word to consume from output; ≥ 16 triggers block generation
}

// newChaCha12FromSeed creates a ChaCha12Rng seeded as Rust's StdRng::seed_from_u64(seed).
func newChaCha12FromSeed(seed uint64) *chacha12Rng {
	key := splitmix64Expand(seed)
	c := &chacha12Rng{idx: 16}
	c.state[0] = 0x61707865
	c.state[1] = 0x3320646e
	c.state[2] = 0x79622d32
	c.state[3] = 0x6b206574
	for i := 0; i < 8; i++ {
		c.state[4+i] = binary.LittleEndian.Uint32(key[i*4:])
	}
	// state[12..15] = counter=0, nonce=0
	return c
}

func rotl32cc(v uint32, n uint) uint32 { return (v << n) | (v >> (32 - n)) }

func chacha12QR(s *[16]uint32, a, b, c, d int) {
	s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32cc(s[d], 16)
	s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32cc(s[b], 12)
	s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl32cc(s[d], 8)
	s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl32cc(s[b], 7)
}

func (c *chacha12Rng) generateBlock() {
	w := c.state // copy
	for i := 0; i < 6; i++ { // 6 double rounds = 12 rounds
		chacha12QR(&w, 0, 4, 8, 12)
		chacha12QR(&w, 1, 5, 9, 13)
		chacha12QR(&w, 2, 6, 10, 14)
		chacha12QR(&w, 3, 7, 11, 15)
		chacha12QR(&w, 0, 5, 10, 15)
		chacha12QR(&w, 1, 6, 11, 12)
		chacha12QR(&w, 2, 7, 8, 13)
		chacha12QR(&w, 3, 4, 9, 14)
	}
	for i := range c.output {
		c.output[i] = w[i] + c.state[i]
	}
	c.state[12]++ // increment lower counter word
	c.idx = 0
}

func (c *chacha12Rng) nextU32() uint32 {
	if c.idx >= 16 {
		c.generateBlock()
	}
	v := c.output[c.idx]
	c.idx++
	return v
}

// nextF32Uniform returns a float32 in [0, 1) matching Rust rand::Standard for f32:
// f32::from_bits((u32 >> 9) | 0x3f800000) - 1.0
func (c *chacha12Rng) nextF32Uniform() float32 {
	v := c.nextU32()
	bits := (v >> 9) | 0x3f800000
	return math.Float32frombits(bits) - 1.0
}

// nextF32 returns a float32 in [-1, 1) — equivalent to gen::<f32>() * 2.0 - 1.0 in Rust.
func (c *chacha12Rng) nextF32() float32 {
	return c.nextF32Uniform()*2.0 - 1.0
}
