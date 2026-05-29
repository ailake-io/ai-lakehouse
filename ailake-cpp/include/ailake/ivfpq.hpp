// SPDX-License-Identifier: MIT OR Apache-2.0
// IVF-PQ index deserialization and CPU search.
// For GPU search see cuda/distance.cuh.
#pragma once

#include "bincode.hpp"
#include "distance.hpp"
#include "hnsw.hpp" // SearchResult
#include <algorithm>
#include <cmath>
#include <vector>

namespace ailake {

struct IvfPqConfig {
    uint64_t nlist;
    uint64_t nprobe;
    uint64_t pq_m;
    uint64_t pq_k;
    uint64_t max_iter;
};

struct PQCodebook {
    uint64_t m;
    uint64_t k;
    std::vector<std::vector<float>> centroids; // [m * k][sub_dim]
};

struct IvfPqIndex {
    IvfPqConfig config;
    Metric metric;
    uint64_t dim;

    std::vector<std::vector<float>>   coarse_centroids; // [nlist][dim]
    PQCodebook pq;
    std::vector<std::vector<uint64_t>> inv_row_ids;     // [nlist][n_per_cell]
    std::vector<std::vector<uint8_t>>  inv_codes;       // [nlist][n*pq_m]
};

inline IvfPqIndex deserialize_ivfpq(const uint8_t* buf, size_t len) {
    BincodeReader r(buf, len);

    IvfPqIndex idx{};
    idx.config.nlist    = r.read_usize();
    idx.config.nprobe   = r.read_usize();
    idx.config.pq_m     = r.read_usize();
    idx.config.pq_k     = r.read_usize();
    idx.config.max_iter = r.read_usize();
    idx.metric          = static_cast<Metric>(r.read_u8());
    idx.dim             = r.read_usize();
    idx.coarse_centroids= r.read_f32_vec2d();
    idx.pq.m            = r.read_usize();
    idx.pq.k            = r.read_usize();
    idx.pq.centroids    = r.read_f32_vec2d();
    idx.inv_row_ids     = r.read_u64_vec2d();
    idx.inv_codes       = r.read_u8_vec2d();
    return idx;
}

// CPU IVF-PQ search using Asymmetric Distance Computation (ADC).
inline std::vector<SearchResult>
ivfpq_search(const IvfPqIndex& idx, const float* query, int top_k, int nprobe = 0) {
    if (nprobe <= 0) nprobe = (int)idx.config.nprobe;
    size_t m       = idx.pq.m;
    size_t k       = idx.pq.k;
    size_t sub_dim = (size_t)idx.dim / m;

    // 1. Find nearest coarse centroids
    std::vector<std::pair<float,size_t>> cell_dists(idx.coarse_centroids.size());
    for (size_t i = 0; i < idx.coarse_centroids.size(); ++i) {
        cell_dists[i] = {
            compute_distance(idx.metric, query, idx.coarse_centroids[i].data(), (size_t)idx.dim),
            i
        };
    }
    std::partial_sort(cell_dists.begin(),
                      cell_dists.begin() + std::min((size_t)nprobe, cell_dists.size()),
                      cell_dists.end());

    // 2. Precompute ADC LUT: dist(query_sub_j, codebook[j][c]) for all j, c
    //    lut[j * k + c] = squared euclidean distance
    std::vector<float> lut(m * k);
    for (size_t j = 0; j < m; ++j) {
        const float* q_sub = query + j * sub_dim;
        for (size_t c = 0; c < k; ++c) {
            size_t cb_idx = j * k + c;
            float d = 0;
            if (cb_idx < idx.pq.centroids.size()) {
                const float* cb = idx.pq.centroids[cb_idx].data();
                for (size_t s = 0; s < sub_dim; ++s) {
                    float diff = q_sub[s] - cb[s];
                    d += diff * diff;
                }
            }
            lut[j * k + c] = d;
        }
    }

    // 3. Scan probed cells
    std::vector<std::pair<float,uint64_t>> hits;
    hits.reserve((size_t)top_k + 1);

    for (int p = 0; p < nprobe && p < (int)cell_dists.size(); ++p) {
        size_t cell = cell_dists[p].second;
        const auto& row_ids = idx.inv_row_ids[cell];
        const auto& codes   = idx.inv_codes[cell];
        for (size_t r = 0; r < row_ids.size(); ++r) {
            float d = 0;
            size_t base = r * m;
            for (size_t j = 0; j < m; ++j) {
                uint8_t c = codes[base + j];
                d += lut[j * k + c];
            }
            if ((int)hits.size() < top_k || d < hits.back().first) {
                hits.push_back({d, row_ids[r]});
                std::sort(hits.begin(), hits.end());
                if ((int)hits.size() > top_k) hits.resize(top_k);
            }
        }
    }

    std::vector<SearchResult> out;
    out.reserve(hits.size());
    for (auto [d, rid] : hits)
        out.push_back({rid, std::sqrt(d)});
    return out;
}

} // namespace ailake
