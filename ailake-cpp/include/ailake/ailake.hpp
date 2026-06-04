// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
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
#include "hardware.hpp"
#include "hnsw.hpp"
#include "ivfpq.hpp"
#include "catalog.hpp"
#include "rocm/blas.hpp"

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

    // Hardware profile override.
    // When nullptr (default), detect_hardware() is called automatically.
    // Pass an explicit profile to force CPU-only or a specific backend.
    const HardwareProfile* hw = nullptr;

    const HardwareProfile& hardware() const {
        return hw ? *hw : detect_hardware();
    }
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

    std::ifstream f(abs_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ailake: cannot open " + abs_path);

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

    const auto& hw = opts.hardware();

    if (hdr.is_ivf_pq()) {
        auto idx = deserialize_ivfpq(index_buf.data(), index_buf.size());

#ifdef AILAKE_CUDA_ENABLED
        // NVIDIA CUDA: GPU IVF-PQ via ADC kernels in cuda/distance.cuh
        if (hw.has_cuda) {
            size_t m = idx.config.pq_m, k = idx.config.pq_k;
            size_t sub_dim = (size_t)idx.dim / m;
            std::vector<float> lut(m * k);
            for (size_t j = 0; j < m; ++j) {
                const float* q_sub = query + j * sub_dim;
                for (size_t c = 0; c < k; ++c) {
                    size_t cb = j * k + c;
                    float d = 0;
                    if (cb < idx.pq.centroids.size()) {
                        const float* cv = idx.pq.centroids[cb].data();
                        for (size_t s = 0; s < sub_dim; ++s) { float dd = q_sub[s]-cv[s]; d += dd*dd; }
                    }
                    lut[j * k + c] = d;
                }
            }
            cuda::GpuIvfPqContext gpu_ctx;
            gpu_ctx.upload(idx.inv_codes, idx.inv_row_ids, (int)m, (int)k);
            auto gpu_hits = gpu_ctx.search(lut.data(), opts.top_k, (int)idx.config.nprobe);
            std::vector<SearchResult> out;
            out.reserve(gpu_hits.size());
            for (auto& h : gpu_hits) out.push_back({h.row_id, h.distance});
            return out;
        }
#endif
        // IVF-PQ search: ROCm does not accelerate ADC (PQ codes, not raw vectors).
        // CPU ADC is used regardless of backend — same as Rust's IVF-PQ search path.
        return ivfpq_search(idx, query, opts.top_k);

    } else {
        // HNSW graph traversal — always CPU (graph is inherently sequential)
        auto idx = deserialize_hnsw(index_buf.data(), index_buf.size());
        if (idx.entry_point && !idx.neighbors.empty()) {
            int ef = opts.ef_search > 0 ? opts.ef_search : opts.top_k * 5;
            return hnsw_search(idx, query, opts.top_k, ef);
        } else if (opts.use_flat_fallback) {
#ifdef AILAKE_CUDA_ENABLED
            // NVIDIA CUDA: GPU batch flat scan
            if (hw.has_cuda && !idx.flat_vecs.empty()) {
                cuda::GpuSearchContext gpu_ctx((int)idx.dim, (int)idx.row_ids.size(),
                                               (int)idx.metric);
                gpu_ctx.upload(idx.flat_vecs.data(), idx.row_ids.data());
                auto gpu_hits = gpu_ctx.search(query, opts.top_k);
                std::vector<SearchResult> out;
                out.reserve(gpu_hits.size());
                for (auto& h : gpu_hits) out.push_back({h.row_id, h.distance});
                return out;
            }
#endif
            // AMD ROCm: flat scan via hipBLAS SGEMM (runtime dlopen — no SDK needed)
            if (hw.has_rocm && !idx.flat_vecs.empty()) {
                std::vector<const float*> q_ptrs = {query};
                auto rocm_hits = rocm::try_rocm_search_batch(
                    q_ptrs, idx.row_ids, idx.flat_vecs.data(),
                    idx.dim, idx.metric, opts.top_k);
                if (!rocm_hits.empty() && !rocm_hits[0].empty()) {
                    std::vector<SearchResult> out;
                    out.reserve(rocm_hits[0].size());
                    for (auto& h : rocm_hits[0]) {
                        SearchResult r; r.row_id = h.row_id; r.distance = h.distance;
                        out.push_back(r);
                    }
                    return out;
                }
                // hipBLAS unavailable or error — fall through to CPU
            }
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

    // NormalizedCosine requires unit-length query — normalize here so callers
    // don't need to pre-normalize manually.
    std::vector<float> norm_query;
    const float* q = query;
    if (metric == Metric::NormalizedCosine) {
        float sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) sq += query[i] * query[i];
        if (sq > 1e-12f) {
            float inv = 1.0f / std::sqrt(sq);
            norm_query.resize(dim);
            for (size_t i = 0; i < dim; ++i) norm_query[i] = query[i] * inv;
            q = norm_query.data();
        }
    }

    // Geometric pruning
    std::vector<DataFileEntry> survivors;
    for (auto& e : entries) {
        if (e.centroid.empty()) { survivors.push_back(e); continue; }
        float d = compute_distance(metric, q, e.centroid.data(), e.centroid.size());
        if (d - e.radius <= opts.pruning_threshold)
            survivors.push_back(e);
    }

    // Per-file HNSW/IVF-PQ search
    std::vector<FileSearchResult> all;
    for (auto& e : survivors) {
        std::string abs = catalog.resolve_path(ns, tbl, e.path);
        try {
            auto hits = search_file(abs, e, q, opts);
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
