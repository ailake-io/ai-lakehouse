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
//   [12..13] = counter (64-bit; lower word only increments for dim ≤ 4096)
//   [14..15] = nonce (zero for from_seed)
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace ailake {

// ---------------------------------------------------------------------------
// splitmix64_expand: u64 seed → 32-byte ChaCha12 key.
// Matches Rust rand 0.8 SeedableRng::seed_from_u64 (4 × splitmix64 rounds).
// ---------------------------------------------------------------------------

inline std::array<uint8_t, 32> splitmix64_expand(uint64_t seed) noexcept {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        // Store as little-endian
        for (int b = 0; b < 8; ++b)
            out[static_cast<size_t>(i * 8 + b)] = static_cast<uint8_t>(z >> (8 * b));
    }
    return out;
}

// ---------------------------------------------------------------------------
// ChaCha12Rng: ChaCha12 PRNG. Matches Rust ChaCha12Rng output sequence.
// ---------------------------------------------------------------------------

class ChaCha12Rng {
public:
    // Seed exactly as Rust StdRng::seed_from_u64(seed): splitmix64 → ChaCha12.
    explicit ChaCha12Rng(uint64_t seed) noexcept : idx_(16) {
        auto key = splitmix64_expand(seed);
        state_[0] = 0x61707865u;
        state_[1] = 0x3320646eu;
        state_[2] = 0x79622d32u;
        state_[3] = 0x6b206574u;
        for (int i = 0; i < 8; ++i) {
            state_[static_cast<size_t>(4 + i)] =
                  static_cast<uint32_t>(key[static_cast<size_t>(i * 4)])
                | (static_cast<uint32_t>(key[static_cast<size_t>(i * 4 + 1)]) << 8)
                | (static_cast<uint32_t>(key[static_cast<size_t>(i * 4 + 2)]) << 16)
                | (static_cast<uint32_t>(key[static_cast<size_t>(i * 4 + 3)]) << 24);
        }
        // state_[12..15] = counter=0, nonce=0
        state_[12] = state_[13] = state_[14] = state_[15] = 0u;
    }

    uint32_t next_u32() noexcept {
        if (idx_ >= 16) generate_block();
        return output_[idx_++];
    }

    // Returns float in [0, 1) matching Rust rand::Standard for f32:
    // f32::from_bits((u32 >> 9) | 0x3f800000) - 1.0
    float next_f32_uniform() noexcept {
        uint32_t v    = next_u32();
        uint32_t bits = (v >> 9) | 0x3f800000u; // IEEE 754 mantissa → [1.0, 2.0)
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f - 1.0f; // → [0, 1)
    }

    // Returns float in [-1, 1) matching Rust gen::<f32>() * 2.0 - 1.0.
    float next_f32() noexcept {
        return next_f32_uniform() * 2.0f - 1.0f;
    }

private:
    uint32_t state_[16];
    uint32_t output_[16];
    int      idx_; // next word index in output_; ≥ 16 → regenerate

    static uint32_t rotl(uint32_t v, int n) noexcept {
        return (v << n) | (v >> (32 - n));
    }

    static void qr(uint32_t* s, int a, int b, int c, int d) noexcept {
        s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl(s[d], 16);
        s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl(s[b], 12);
        s[a] += s[b]; s[d] ^= s[a]; s[d] = rotl(s[d], 8);
        s[c] += s[d]; s[b] ^= s[c]; s[b] = rotl(s[b], 7);
    }

    void generate_block() noexcept {
        uint32_t w[16];
        std::memcpy(w, state_, sizeof w);
        for (int i = 0; i < 6; ++i) { // 6 double rounds = 12 rounds
            qr(w, 0, 4, 8, 12);
            qr(w, 1, 5, 9, 13);
            qr(w, 2, 6, 10, 14);
            qr(w, 3, 7, 11, 15);
            qr(w, 0, 5, 10, 15);
            qr(w, 1, 6, 11, 12);
            qr(w, 2, 7, 8, 13);
            qr(w, 3, 4, 9, 14);
        }
        for (int i = 0; i < 16; ++i)
            output_[i] = w[i] + state_[i];
        ++state_[12]; // increment lower counter word
        idx_ = 0;
    }
};

} // namespace ailake
