// SPDX-License-Identifier: MIT OR Apache-2.0
// ailake.hpp — main include. Pulls in all AI-Lake C++ headers.
//
// CPU-only:
//   #include <ailake/ailake.hpp>
//
// With CUDA GPU acceleration:
//   #define AILAKE_CUDA_ENABLED
//   #include <ailake/ailake.hpp>
//   #include <ailake/cuda/distance.cuh>
//
// High-level search API:
//   ailake::HadoopCatalog cat("/data/warehouse");
//   auto results = ailake::search(cat, "default", "docs", query, {.top_k=10});
#pragma once

#include "footer.hpp"
#include "bincode.hpp"
#include "distance.hpp"
#include "hnsw.hpp"
#include "ivfpq.hpp"
#include "catalog.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ailake {

// ---------------------------------------------------------------------------
// Search options
// ---------------------------------------------------------------------------

struct SearchOptions {
    int   top_k             = 10;
    int   ef_search         = 0;   // 0 → top_k * 5
    float pruning_threshold = 0.8f;
    bool  use_flat_fallback = true; // flat scan when HNSW graph is empty
};

// ---------------------------------------------------------------------------
// Per-file search using the AILK index embedded in the file.
// ---------------------------------------------------------------------------

inline std::vector<SearchResult>
search_file(const std::string& abs_path,
            const DataFileEntry& entry,
            const float* query,
            const SearchOptions& opts)
{
    if (!entry.hnsw_offset || !entry.hnsw_len) return {};

    // Memory-map or read the file
    std::ifstream f(abs_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ailake: cannot open " + abs_path);
    std::streamsize file_size = f.tellg();

    // Read AILK header at hnsw_offset
    uint8_t header_buf[kHeaderSize];
    f.seekg((std::streamoff)*entry.hnsw_offset);
    f.read(reinterpret_cast<char*>(header_buf), kHeaderSize);
    if (!f) throw std::runtime_error("ailake: cannot read AILK header in " + abs_path);
    auto hdr = parse_header(header_buf);

    // Read index blob
    size_t index_start = *entry.hnsw_offset + hdr.hnsw_offset;
    std::vector<uint8_t> index_buf(hdr.hnsw_len);
    f.seekg((std::streamoff)index_start);
    f.read(reinterpret_cast<char*>(index_buf.data()), (std::streamsize)hdr.hnsw_len);
    if (!f) throw std::runtime_error("ailake: cannot read index blob in " + abs_path);

    if (hdr.is_ivf_pq()) {
        auto idx = deserialize_ivfpq(index_buf.data(), index_buf.size());
        return ivfpq_search(idx, query, opts.top_k);
    } else {
        auto idx = deserialize_hnsw(index_buf.data(), index_buf.size());
        if (idx.entry_point && !idx.neighbors.empty()) {
            int ef = opts.ef_search > 0 ? opts.ef_search : opts.top_k * 5;
            return hnsw_search(idx, query, opts.top_k, ef);
        } else if (opts.use_flat_fallback) {
            return flat_search(idx, query, opts.top_k);
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// High-level Search() — geometric pruning + per-file search + top-K merge.
// ---------------------------------------------------------------------------

inline std::vector<FileSearchResult>
search(HadoopCatalog& catalog,
       const std::string& ns, const std::string& tbl,
       const float* query, size_t dim,
       const SearchOptions& opts = {})
{
    auto info    = catalog.load_table(ns, tbl);
    auto entries = catalog.list_files(ns, tbl);
    auto metric  = metric_from_str(info.vector_metric);

    // Geometric pruning
    std::vector<DataFileEntry> survivors;
    for (auto& e : entries) {
        if (e.centroid.empty()) { survivors.push_back(e); continue; }
        float d = compute_distance(metric, query, e.centroid.data(), e.centroid.size());
        if (d - e.radius <= opts.pruning_threshold)
            survivors.push_back(e);
    }

    // Per-file HNSW/IVF-PQ search
    std::vector<FileSearchResult> all;
    for (auto& e : survivors) {
        std::string abs = catalog.resolve_path(ns, tbl, e.path);
        try {
            auto hits = search_file(abs, e, query, opts);
            for (auto& h : hits)
                all.push_back({h.row_id, h.distance, e.path});
        } catch (const std::exception& ex) {
            // Skip files that can't be read (e.g. still indexing)
        }
    }

    // Global top-K merge
    std::sort(all.begin(), all.end(),
        [](const FileSearchResult& a, const FileSearchResult& b){
            return a.distance < b.distance; });
    if ((int)all.size() > opts.top_k)
        all.resize((size_t)opts.top_k);
    return all;
}

} // namespace ailake
