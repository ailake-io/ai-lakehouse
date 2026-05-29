// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// HNSW index deserialization and search.
// Decodes ailake_index::HnswSnapshot (bincode v1).
#pragma once

#include "bincode.hpp"
#include "distance.hpp"
#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_set>
#include <vector>

namespace ailake {

struct HnswIndex {
    uint64_t m;
    uint64_t ef_construction;
    Metric   metric;
    uint32_t dim;

    std::vector<uint64_t>  row_ids;
    std::vector<float>     flat_vecs; // stride = dim

    // neighbors[node][layer] = {neighbor_node_indices}
    std::vector<std::vector<std::vector<uint64_t>>> neighbors;

    std::vector<uint64_t>     node_levels;
    std::optional<uint64_t>   entry_point;
    uint64_t                  max_layer;

    const float* vec(uint64_t i) const {
        return flat_vecs.data() + i * dim;
    }

    float dist(const float* q, uint64_t i) const {
        return compute_distance(metric, q, vec(i), dim);
    }
};

struct SearchResult {
    uint64_t row_id;
    float    distance;
};

// Deserialize a bincode-encoded HnswSnapshot.
inline HnswIndex deserialize_hnsw(const uint8_t* buf, size_t len) {
    BincodeReader r(buf, len);

    HnswIndex idx{};
    idx.m              = r.read_usize();
    idx.ef_construction= r.read_usize();
    /* max_elements */  r.read_usize();
    uint8_t m          = r.read_u8();
    idx.metric         = static_cast<Metric>(m);
    idx.dim            = r.read_u32();
    idx.row_ids        = r.read_u64_vec();
    idx.flat_vecs      = r.read_f32_vec();
    idx.neighbors      = r.read_neighbors();
    idx.node_levels    = r.read_u64_vec();
    idx.entry_point    = r.read_option_usize();
    idx.max_layer      = r.read_usize();
    return idx;
}

// ---------------------------------------------------------------------------
// HNSW greedy search
// ---------------------------------------------------------------------------

namespace detail {

inline uint64_t greedy_nearest(const HnswIndex& idx, const float* q,
                                uint64_t entry, int layer) {
    uint64_t best = entry;
    float best_d  = idx.dist(q, entry);
    for (;;) {
        bool improved = false;
        if (best >= idx.neighbors.size() || layer >= (int)idx.neighbors[best].size())
            break;
        for (uint64_t nb : idx.neighbors[best][layer]) {
            float d = idx.dist(q, nb);
            if (d < best_d) { best_d = d; best = nb; improved = true; }
        }
        if (!improved) break;
    }
    return best;
}

// max-heap candidate: farthest on top
using MaxPQ = std::priority_queue<
    std::pair<float,uint64_t>,
    std::vector<std::pair<float,uint64_t>>,
    std::less<std::pair<float,uint64_t>>>;
// min-heap candidate: closest on top
using MinPQ = std::priority_queue<
    std::pair<float,uint64_t>,
    std::vector<std::pair<float,uint64_t>>,
    std::greater<std::pair<float,uint64_t>>>;

inline std::vector<std::pair<float,uint64_t>>
beam_search(const HnswIndex& idx, const float* q,
            uint64_t entry, int layer, int ef) {
    std::unordered_set<uint64_t> visited;
    float d0 = idx.dist(q, entry);
    visited.insert(entry);

    MinPQ candidates; candidates.push({d0, entry});
    MaxPQ results;    results.push({d0, entry});

    while (!candidates.empty()) {
        auto [cd, ci] = candidates.top(); candidates.pop();
        if (results.size() >= (size_t)ef && cd > results.top().first) break;
        if (ci >= idx.neighbors.size() || layer >= (int)idx.neighbors[ci].size()) continue;
        for (uint64_t nb : idx.neighbors[ci][layer]) {
            if (visited.count(nb)) continue;
            visited.insert(nb);
            float d = idx.dist(q, nb);
            if ((int)results.size() < ef || d < results.top().first) {
                candidates.push({d, nb});
                results.push({d, nb});
                if ((int)results.size() > ef) results.pop();
            }
        }
    }
    std::vector<std::pair<float,uint64_t>> out;
    out.reserve(results.size());
    while (!results.empty()) { out.push_back(results.top()); results.pop(); }
    std::sort(out.begin(), out.end()); // ascending distance
    return out;
}

} // namespace detail

// Run HNSW search; ef_search defaults to top_k * 5.
inline std::vector<SearchResult>
hnsw_search(const HnswIndex& idx, const float* query, int top_k, int ef_search = 0) {
    if (!idx.entry_point || idx.row_ids.empty()) return {};
    if (ef_search < top_k) ef_search = top_k * 5;

    uint64_t ep = *idx.entry_point;
    for (int layer = (int)idx.max_layer; layer > 0; --layer)
        ep = detail::greedy_nearest(idx, query, ep, layer);

    auto cands = detail::beam_search(idx, query, ep, 0, ef_search);
    if ((int)cands.size() > top_k) cands.resize(top_k);

    std::vector<SearchResult> out;
    out.reserve(cands.size());
    for (auto [d, i] : cands)
        out.push_back({idx.row_ids[i], d});
    return out;
}

// Brute-force flat scan (fallback for empty graph / old-format files).
inline std::vector<SearchResult>
flat_search(const HnswIndex& idx, const float* query, int top_k) {
    size_t n = idx.row_ids.size();
    std::vector<std::pair<float,uint64_t>> hits;
    hits.reserve(std::min((size_t)top_k + 1, n));
    for (size_t i = 0; i < n; ++i) {
        float d = idx.dist(query, (uint64_t)i);
        if ((int)hits.size() < top_k || d < hits.back().first) {
            hits.push_back({d, i});
            std::sort(hits.begin(), hits.end());
            if ((int)hits.size() > top_k) hits.resize(top_k);
        }
    }
    std::vector<SearchResult> out;
    out.reserve(hits.size());
    for (auto [d, i] : hits)
        out.push_back({idx.row_ids[i], d});
    return out;
}

} // namespace ailake
