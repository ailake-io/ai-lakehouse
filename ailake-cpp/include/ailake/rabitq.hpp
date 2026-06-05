// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// RaBitQ flat index: deserialization and brute-force search.
// Wire format: bincode v1 of ailake_index::RaBitQIndex (§6.3 FILE_FORMAT.md)
//
// Rotation matrix: modified Gram-Schmidt orthonormal (P^T·P = I), matching
// ailake-vec/src/rabitq.rs rebuild_proj().
//
// WARNING: std::mt19937_64 produces different sequences than Rust StdRng
// (ChaCha12). For the same seed, this implementation generates a different
// projection matrix than the Rust/Go SDKs — cross-language RaBitQ search is
// currently BROKEN for files written by the Rust SDK.
// TODO: replace std::mt19937_64 with a ChaCha12-compatible PRNG that matches
// Rust's StdRng::seed_from_u64 seed expansion.
#pragma once

#include "bincode.hpp"
#include "distance.hpp"
#include "hnsw.hpp" // SearchResult
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ailake {

// ---------------------------------------------------------------------------
// Portable popcount for a single byte
// ---------------------------------------------------------------------------

inline int popcount8(uint8_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    x = static_cast<uint8_t>(x - ((x >> 1) & 0x55u));
    x = static_cast<uint8_t>((x & 0x33u) + ((x >> 2) & 0x33u));
    return static_cast<int>((x + (x >> 4)) & 0x0Fu);
#endif
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct RaBitQVec {
    std::vector<uint8_t> code;  // ceil(dim/8) packed sign bits
    float norm;
    float scale;
};

struct RaBitQIndex {
    uint64_t seed;
    uint32_t dim;
    Metric   metric;
    std::vector<RaBitQVec> entries;
    std::vector<uint64_t>  row_ids;
    // raw F16 for reranking (row-major: entry i at raw_f16[i*dim .. (i+1)*dim])
    // empty when keep_raw=false
    std::vector<uint16_t>  raw_f16;
    // Runtime projection matrix (dim×dim, row-major). Not serialized.
    std::vector<float>     proj;
};

// ---------------------------------------------------------------------------
// build_proj — regenerate the dim×dim rotation matrix from seed.
// ---------------------------------------------------------------------------

inline void build_proj(RaBitQIndex& idx) {
    const size_t dim = idx.dim;
    idx.proj.assign(dim * dim, 0.f);

    // Fill row-major with uniform [-1, 1] values.
    std::mt19937_64 rng(idx.seed);
    std::uniform_real_distribution<float> udist(-1.f, 1.f);
    for (auto& v : idx.proj) v = udist(rng);

    // Modified Gram-Schmidt: orthogonalize columns in place.
    for (size_t col = 0; col < dim; ++col) {
        // Subtract projection onto all previous orthonormal columns.
        for (size_t prev = 0; prev < col; ++prev) {
            double dot = 0.0;
            for (size_t row = 0; row < dim; ++row)
                dot += static_cast<double>(idx.proj[row * dim + col])
                     * static_cast<double>(idx.proj[row * dim + prev]);
            auto fdot = static_cast<float>(dot);
            for (size_t row = 0; row < dim; ++row)
                idx.proj[row * dim + col] -= fdot * idx.proj[row * dim + prev];
        }
        // Normalize to unit length.
        double norm_sq = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            double v = idx.proj[row * dim + col];
            norm_sq += v * v;
        }
        float inv = 1.f / static_cast<float>(std::sqrt(norm_sq + 1e-24));
        for (size_t row = 0; row < dim; ++row)
            idx.proj[row * dim + col] *= inv;
    }
}

// ---------------------------------------------------------------------------
// deserialize_rabitq — parse bincode v1 blob into RaBitQIndex.
// ---------------------------------------------------------------------------

inline RaBitQIndex deserialize_rabitq(const uint8_t* buf, size_t len) {
    BincodeReader r(buf, len);

    RaBitQIndex idx{};
    // codebook.dim (usize → u64), codebook.seed (u64)
    (void)r.read_usize(); // codebook_dim — same as idx.dim below
    idx.seed = r.read_u64();

    // entries: Vec<RaBitQVec>
    uint64_t n_entries = r.read_usize();
    idx.entries.resize(static_cast<size_t>(n_entries));
    for (auto& e : idx.entries) {
        e.code  = r.read_u8_vec_flat();
        e.norm  = r.read_f32();
        e.scale = r.read_f32();
    }

    idx.row_ids = r.read_u64_vec();
    idx.metric  = static_cast<Metric>(r.read_u32());
    idx.dim     = r.read_u32();

    // raw_f16: Option<Vec<u16>> — tag 0=None, 1=Some
    if (r.read_u8() == 1)
        idx.raw_f16 = r.read_u16_vec();

    build_proj(idx);
    return idx;
}

// ---------------------------------------------------------------------------
// rabitq_search — O(N) scan + partial select + optional F16 reranking.
// ---------------------------------------------------------------------------

inline std::vector<SearchResult>
rabitq_search(RaBitQIndex& idx, const float* query, int top_k,
              int rerank_factor = 3)
{
    if (idx.entries.empty() || top_k <= 0) return {};

    const size_t dim      = idx.dim;
    const size_t n        = idx.entries.size();
    const size_t code_len = (dim + 7) / 8;

    // 1. Normalize query to unit L2
    float q_norm_sq = 0.f;
    for (size_t i = 0; i < dim; ++i) q_norm_sq += query[i] * query[i];
    float q_norm = std::sqrt(q_norm_sq + 1e-24f);
    std::vector<float> q_hat(dim);
    for (size_t i = 0; i < dim; ++i) q_hat[i] = query[i] / q_norm;

    // 2. Project: q_proj = P · q_hat  (P row-major: row i at proj[i*dim..])
    std::vector<float> q_proj(dim, 0.f);
    for (size_t i = 0; i < dim; ++i) {
        const float* row = idx.proj.data() + i * dim;
        for (size_t j = 0; j < dim; ++j)
            q_proj[i] += row[j] * q_hat[j];
    }

    // 3. Binarize query once + compute q_scale
    std::vector<uint8_t> b_q(code_len, 0u);
    float q_scale = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        if (q_proj[i] > 0.f) b_q[i / 8] |= static_cast<uint8_t>(1u << (i & 7u));
        q_scale += std::abs(q_proj[i]);
    }
    q_scale /= std::sqrt(static_cast<float>(dim));

    // 4. Sequential scan: XOR + popcount → IP estimate → distance
    std::vector<std::pair<float, size_t>> scored(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& e = idx.entries[i];
        int hamming = 0;
        const size_t cl = std::min(code_len, e.code.size());
        for (size_t b = 0; b < cl; ++b)
            hamming += popcount8(static_cast<uint8_t>(b_q[b] ^ e.code[b]));

        float ip = (1.f - 2.f * static_cast<float>(hamming) / static_cast<float>(dim))
                   * q_scale * e.scale;

        float dist;
        switch (idx.metric) {
            case Metric::Cosine:
            case Metric::NormalizedCosine:
                dist = 1.f - ip;
                break;
            case Metric::DotProduct:
                dist = -ip * q_norm * e.norm;
                break;
            case Metric::Euclidean: {
                float nx = e.norm;
                float d2 = q_norm_sq + nx * nx - 2.f * ip * q_norm * nx;
                dist = std::sqrt(d2 > 0.f ? d2 : 0.f);
                break;
            }
            default:
                dist = 1.f - ip;
        }
        scored[i] = {dist, i};
    }

    // 5. O(N) partial select: bring top candidates to front
    size_t candidates = std::min(n,
        static_cast<size_t>(std::max(1, rerank_factor))
        * static_cast<size_t>(top_k));

    auto cmp = [](const std::pair<float, size_t>& a,
                  const std::pair<float, size_t>& b) {
        return a.first < b.first;
    };
    if (candidates < n) {
        std::nth_element(scored.begin(),
                         scored.begin() + static_cast<std::ptrdiff_t>(candidates) - 1,
                         scored.end(), cmp);
        scored.resize(candidates);
    }
    std::sort(scored.begin(), scored.end(), cmp);

    // 6. Reranking with exact F16 distances (when raw_f16 present + rerank_factor > 1)
    if (!idx.raw_f16.empty() && rerank_factor > 1) {
        std::vector<std::pair<float, size_t>> reranked;
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

    // No reranking: return top-k from coarse scored list
    const size_t take = std::min(static_cast<size_t>(top_k), scored.size());
    std::vector<SearchResult> out;
    out.reserve(take);
    for (size_t t = 0; t < take; ++t)
        out.push_back({idx.row_ids[scored[t].second], scored[t].first});
    return out;
}

} // namespace ailake
