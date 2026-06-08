// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Binary Hamming flat index: deserialization and brute-force search.
//
// Wire format: bincode v1 of ailake_index::BinaryIndex:
//   codes:         Vec<u8>          (u64 length + flat packed-bit bytes)
//   bytes_per_vec: usize            (u64 LE — ceil(dim/8))
//   row_ids:       Vec<u64>
//   metric:        u32
//   dim:           u32
//   raw_f16:       Option<Vec<u16>> (tag 0=None, 1=Some + u64 length + u16 values)
//
// Binarization: sign(x) >= 0 → bit 1, MSB-first within each byte
//   (dimension 0 = bit 7 of byte 0; same as ailake-vec/src/binary_quant.rs)
//
// SIMD Hamming: AVX2+SSSE3 nibble-LUT → NEON vcntq_u8 → scalar __builtin_popcountll
#pragma once

#include "bincode.hpp"
#include "distance.hpp"
#include "footer.hpp"
#include "hnsw.hpp" // SearchResult

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__SSSE3__)
#  include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace ailake {

// ---------------------------------------------------------------------------
// BinaryIndex — deserialized AI-Lake Binary Hamming flat index
// ---------------------------------------------------------------------------

struct BinaryIndex {
    // Packed bit codes, flat: entry i at codes[i*bytes_per_vec : (i+1)*bytes_per_vec].
    std::vector<uint8_t> codes;
    size_t               bytes_per_vec; // ceil(dim/8)
    std::vector<uint64_t> row_ids;
    Metric               metric;
    uint32_t             dim;
    // Raw F16 vectors for reranking (row-major: entry i at raw_f16[i*dim..(i+1)*dim]).
    // Empty when keep_raw=false.
    std::vector<uint16_t> raw_f16;
};

// ---------------------------------------------------------------------------
// f32_to_bits — binarize float vector: sign(x) >= 0 → bit 1, MSB-first
// ---------------------------------------------------------------------------

inline std::vector<uint8_t> f32_to_bits(const float* v, size_t dim) {
    size_t n = (dim + 7) / 8;
    std::vector<uint8_t> out(n, 0u);
    for (size_t i = 0; i < dim; ++i) {
        if (v[i] >= 0.f)
            out[i / 8] |= static_cast<uint8_t>(0x80u >> (i & 7u));
    }
    return out;
}

// ---------------------------------------------------------------------------
// SIMD Hamming distance kernels
// ---------------------------------------------------------------------------

#if defined(__AVX2__) && defined(__SSSE3__)
inline int hamming_avx2(const uint8_t* __restrict__ a,
                        const uint8_t* __restrict__ b, size_t n) noexcept
{
    const __m256i low4 = _mm256_set1_epi8(static_cast<char>(0x0F));
    // Lookup: popcount of nibble 0..15
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    __m256i acc = _mm256_setzero_si256();
    const size_t chunks = n / 32;
    for (size_t i = 0; i < chunks; ++i) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i * 32));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i * 32));
        __m256i xv = _mm256_xor_si256(va, vb);
        __m256i lo = _mm256_and_si256(xv, low4);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xv, 4), low4);
        __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                       _mm256_shuffle_epi8(lookup, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
    }
    uint64_t sum = static_cast<uint64_t>(_mm256_extract_epi64(acc, 0))
                 + static_cast<uint64_t>(_mm256_extract_epi64(acc, 1))
                 + static_cast<uint64_t>(_mm256_extract_epi64(acc, 2))
                 + static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));
    int total = static_cast<int>(sum);
    for (size_t i = chunks * 32; i < n; ++i)
        total += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return total;
}
#endif // __AVX2__ && __SSSE3__

#if defined(__ARM_NEON)
inline int hamming_neon(const uint8_t* __restrict__ a,
                        const uint8_t* __restrict__ b, size_t n) noexcept
{
    uint32x4_t acc = vdupq_n_u32(0u);
    const size_t chunks = n / 16;
    for (size_t i = 0; i < chunks; ++i) {
        uint8x16_t va  = vld1q_u8(a + i * 16);
        uint8x16_t vb  = vld1q_u8(b + i * 16);
        uint8x16_t xv  = veorq_u8(va, vb);
        uint8x16_t cnt = vcntq_u8(xv);
        acc = vpadalq_u16(acc, vpaddlq_u8(cnt));
    }
    int total = static_cast<int>(vaddvq_u32(acc));
    for (size_t i = chunks * 16; i < n; ++i)
        total += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return total;
}
#endif // __ARM_NEON

inline int hamming_scalar(const uint8_t* __restrict__ a,
                          const uint8_t* __restrict__ b, size_t n) noexcept
{
    int total = 0;
    const size_t chunks = n / 8;
    for (size_t i = 0; i < chunks; ++i) {
        uint64_t va, vb;
        std::memcpy(&va, a + i * 8, 8);
        std::memcpy(&vb, b + i * 8, 8);
        total += __builtin_popcountll(va ^ vb);
    }
    for (size_t i = chunks * 8; i < n; ++i)
        total += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return total;
}

inline int hamming_distance(const uint8_t* a, const uint8_t* b, size_t n) noexcept {
#if defined(__AVX2__) && defined(__SSSE3__)
    return hamming_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return hamming_neon(a, b, n);
#else
    return hamming_scalar(a, b, n);
#endif
}

// ---------------------------------------------------------------------------
// deserialize_binary — parse bincode v1 blob into BinaryIndex
// ---------------------------------------------------------------------------

inline BinaryIndex deserialize_binary(const uint8_t* buf, size_t len) {
    BincodeReader r(buf, len);

    BinaryIndex idx{};

    // codes: Vec<u8>
    idx.codes = r.read_u8_vec_flat();

    // bytes_per_vec: usize
    idx.bytes_per_vec = static_cast<size_t>(r.read_usize());

    // row_ids: Vec<u64>
    idx.row_ids = r.read_u64_vec();

    // metric: u32
    idx.metric = static_cast<Metric>(r.read_u32());

    // dim: u32
    idx.dim = r.read_u32();

    // raw_f16: Option<Vec<u16>> — tag 0=None, 1=Some
    if (r.read_u8() == 1)
        idx.raw_f16 = r.read_u16_vec();

    return idx;
}

// ---------------------------------------------------------------------------
// binary_search — O(N) Hamming scan + partial select + optional F16 reranking.
// rerank_factor: candidates = rerank_factor × top_k; reranked with exact F16.
// ---------------------------------------------------------------------------

inline std::vector<SearchResult>
binary_search(const BinaryIndex& idx, const float* query, int top_k,
              int rerank_factor = 3) noexcept(false)
{
    if (idx.row_ids.empty() || top_k <= 0) return {};

    const size_t n   = idx.row_ids.size();
    const size_t bpv = idx.bytes_per_vec;
    const size_t dim = idx.dim;

    auto q_bits = f32_to_bits(query, dim);

    // Phase 1: Hamming scan
    std::vector<std::pair<float, size_t>> scored(n);
    for (size_t i = 0; i < n; ++i) {
        int h = hamming_distance(q_bits.data(), idx.codes.data() + i * bpv, bpv);
        scored[i] = {static_cast<float>(h), i};
    }

    // Phase 2: O(N) partial select
    auto cmp = [](const std::pair<float,size_t>& a,
                  const std::pair<float,size_t>& b){ return a.first < b.first; };
    const size_t candidates = std::min(n,
        static_cast<size_t>(std::max(1, rerank_factor)) *
        static_cast<size_t>(top_k));
    if (candidates < n) {
        std::nth_element(scored.begin(),
                         scored.begin() + static_cast<std::ptrdiff_t>(candidates) - 1,
                         scored.end(), cmp);
        scored.resize(candidates);
    }
    std::sort(scored.begin(), scored.end(), cmp);

    // Phase 3: Optional F16 reranking
    if (!idx.raw_f16.empty() && rerank_factor > 1) {
        std::vector<std::pair<float,size_t>> reranked;
        reranked.reserve(candidates);
        std::vector<float> db_f32(dim);
        for (auto& [_, i] : scored) {
            const uint16_t* fp = idx.raw_f16.data() + i * dim;
            for (size_t d = 0; d < dim; ++d) db_f32[d] = f16_to_f32(fp[d]);
            reranked.push_back({compute_distance(idx.metric, query, db_f32.data(), dim), i});
        }
        std::sort(reranked.begin(), reranked.end(), cmp);
        if (static_cast<int>(reranked.size()) > top_k)
            reranked.resize(static_cast<size_t>(top_k));
        std::vector<SearchResult> out;
        out.reserve(reranked.size());
        for (auto& [d, i] : reranked)
            out.push_back({idx.row_ids[i], d});
        return out;
    }

    // No reranking
    const size_t take = std::min(static_cast<size_t>(top_k), scored.size());
    std::vector<SearchResult> out;
    out.reserve(take);
    for (size_t t = 0; t < take; ++t)
        out.push_back({idx.row_ids[scored[t].second], scored[t].first});
    return out;
}

} // namespace ailake
