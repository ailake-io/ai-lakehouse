// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// IVF-PQ index deserialization and CPU search.
// For GPU search see cuda/distance.cuh.
//
// Wire format: bincode v1 IvfPqSnapshotCore, then optional trailing byte:
//   [residual: u8]  — 0x01 = residual PQ encoding active; absent in legacy files.
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
    bool     residual = false; // codes are per-cluster residuals; use per-cluster ADC
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
    // Optional trailing byte: residual flag. Legacy files have no trailing byte.
    if (r.remaining() > 0)
        idx.config.residual = (r.read_u8() != 0);
    return idx;
}

// build_adc_lut: dist(q_sub_j, codebook[j][c]) for all j, c.
// lut[j * k + c] = squared euclidean distance.
inline std::vector<float>
build_adc_lut(const float* query, size_t m, size_t k, size_t sub_dim,
              const std::vector<std::vector<float>>& centroids) {
    std::vector<float> lut(m * k);
    for (size_t j = 0; j < m; ++j) {
        const float* q_sub = query + j * sub_dim;
        for (size_t c = 0; c < k; ++c) {
            size_t cb_idx = j * k + c;
            float d = 0;
            if (cb_idx < centroids.size()) {
                const float* cb = centroids[cb_idx].data();
                for (size_t s = 0; s < sub_dim; ++s) {
                    float diff = q_sub[s] - cb[s];
                    d += diff * diff;
                }
            }
            lut[j * k + c] = d;
        }
    }
    return lut;
}

// CPU IVF-PQ search using Asymmetric Distance Computation (ADC).
// For residual indexes, builds a per-cluster LUT over q - coarse_centroid[cell].
inline std::vector<SearchResult>
ivfpq_search(const IvfPqIndex& idx, const float* query, int top_k, int nprobe = 0) {
    if (nprobe <= 0) nprobe = (int)idx.config.nprobe;
    size_t m       = idx.pq.m;
    size_t k       = idx.pq.k;
    size_t dim     = (size_t)idx.dim;
    size_t sub_dim = dim / m;

    // 1. Find nearest coarse centroids
    std::vector<std::pair<float,size_t>> cell_dists(idx.coarse_centroids.size());
    for (size_t i = 0; i < idx.coarse_centroids.size(); ++i) {
        cell_dists[i] = {
            compute_distance(idx.metric, query, idx.coarse_centroids[i].data(), dim),
            i
        };
    }
    std::partial_sort(cell_dists.begin(),
                      cell_dists.begin() + std::min((size_t)nprobe, cell_dists.size()),
                      cell_dists.end());

    // 2. For non-residual: one global ADC LUT.
    //    For residual: computed per cluster in step 3.
    std::vector<float> global_lut;
    if (!idx.config.residual)
        global_lut = build_adc_lut(query, m, k, sub_dim, idx.pq.centroids);

    // 3. Scan probed cells
    std::vector<std::pair<float,uint64_t>> hits;
    hits.reserve((size_t)top_k + 1);

    std::vector<float> q_res(dim); // reused scratch for residual mode

    for (int p = 0; p < nprobe && p < (int)cell_dists.size(); ++p) {
        size_t cell = cell_dists[p].second;

        const std::vector<float>* lut_ptr = &global_lut;
        std::vector<float> cluster_lut;
        if (idx.config.residual) {
            // q_res = query - coarse_centroid[cell]
            const float* cent = idx.coarse_centroids[cell].data();
            for (size_t d = 0; d < dim; ++d) q_res[d] = query[d] - cent[d];
            cluster_lut = build_adc_lut(q_res.data(), m, k, sub_dim, idx.pq.centroids);
            lut_ptr = &cluster_lut;
        }

        const auto& row_ids = idx.inv_row_ids[cell];
        const auto& codes   = idx.inv_codes[cell];
        for (size_t r = 0; r < row_ids.size(); ++r) {
            float d = 0;
            size_t base = r * m;
            for (size_t j = 0; j < m; ++j) {
                uint8_t c = codes[base + j];
                d += (*lut_ptr)[j * k + c];
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
