// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// AMD ROCm/HIP batch vector search via hipBLAS SGEMM.
//
// Mirrors Rust ailake_index::gpu::rocm_impl exactly:
//   - dlopen libamdhip64.so + libhipblas.so at runtime (no compile-time ROCm SDK)
//   - SGEMM formulation: C[N×Q] = α · db[N×dim]ᵀ · queries[Q×dim], then top-K on CPU
//   - Falls back to CPU (returns false) when libraries absent or GPU error
//
// No #include <hip/*> — everything via function pointers loaded at runtime.
// This file is CPU C++17 only; no HIP compilation required.
#pragma once

#include "../distance.hpp"
#include "../footer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#if defined(__linux__)
#  include <dlfcn.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace ailake {
namespace rocm {

// ---------------------------------------------------------------------------
// Library names
// ---------------------------------------------------------------------------

#if defined(__linux__)
static constexpr const char* kHipLib  = "libamdhip64.so";
static constexpr const char* kBlasLib = "libhipblas.so";
#elif defined(_WIN32)
static constexpr const char* kHipLib  = "amdhip64.dll";
static constexpr const char* kBlasLib = "hipblas.dll";
#else
static constexpr const char* kHipLib  = "";
static constexpr const char* kBlasLib = "";
#endif

// hipblasOperation_t constants (same int values as cuBLAS)
static constexpr int kOpN = 111; // HIPBLAS_OP_N
static constexpr int kOpT = 112; // HIPBLAS_OP_T

// hipMemcpyKind
static constexpr int kH2D = 1;
static constexpr int kD2H = 2;

// ---------------------------------------------------------------------------
// Function pointer typedefs (HIP runtime + hipBLAS)
// ---------------------------------------------------------------------------

using HipResult   = int;
using fn_hipMalloc           = HipResult(*)(void**, size_t);
using fn_hipFree             = HipResult(*)(void*);
using fn_hipMemcpy           = HipResult(*)(void*, const void*, size_t, int);
using fn_hipDeviceSynchronize= HipResult(*)();
using fn_hipblasCreate       = HipResult(*)(void**);
using fn_hipblasDestroy      = HipResult(*)(void*);
using fn_hipblasSgemm        = HipResult(*)(
    void*,          // handle
    int,            // transa
    int,            // transb
    int,            // m
    int,            // n
    int,            // k
    const float*,   // alpha
    const void*, int, // A, lda
    const void*, int, // B, ldb
    const float*,   // beta
    void*, int      // C, ldc
);

// ---------------------------------------------------------------------------
// RAII helpers
// ---------------------------------------------------------------------------

namespace detail {

inline void* open_lib(const char* name) {
    if (!name || name[0] == '\0') return nullptr;
#if defined(__linux__)
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#elif defined(_WIN32)
    return (void*)LoadLibraryA(name);
#else
    return nullptr;
#endif
}
inline void close_lib(void* h) {
    if (!h) return;
#if defined(__linux__)
    dlclose(h);
#elif defined(_WIN32)
    FreeLibrary((HMODULE)h);
#endif
}
inline void* get_sym(void* h, const char* sym) {
#if defined(__linux__)
    return dlsym(h, sym);
#elif defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)h, sym);
#else
    return nullptr;
#endif
}

struct LibGuard {
    void* h = nullptr;
    explicit LibGuard(const char* name) : h(open_lib(name)) {}
    ~LibGuard() { close_lib(h); }
    bool ok() const { return h != nullptr; }
    void* sym(const char* s) const { return get_sym(h, s); }
    template<typename F> F fn(const char* s) const { return reinterpret_cast<F>(sym(s)); }
};

struct DevBuf {
    void*       ptr     = nullptr;
    fn_hipFree  free_fn = nullptr;
    DevBuf() = default;
    DevBuf(void* p, fn_hipFree f) : ptr(p), free_fn(f) {}
    ~DevBuf() { if (ptr && free_fn) free_fn(ptr); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), free_fn(o.free_fn) { o.ptr = nullptr; }
};

inline DevBuf alloc(size_t bytes, fn_hipMalloc malloc_fn, fn_hipFree free_fn) {
    void* p = nullptr;
    if (malloc_fn(&p, bytes) != 0) return {};
    return {p, free_fn};
}

inline DevBuf upload(const float* data, size_t n, fn_hipMalloc malloc_fn,
                     fn_hipFree free_fn, fn_hipMemcpy memcpy_fn) {
    auto buf = alloc(n * sizeof(float), malloc_fn, free_fn);
    if (!buf.ptr) return {};
    if (memcpy_fn(buf.ptr, data, n * sizeof(float), kH2D) != 0) return {};
    return buf;
}

inline std::vector<float> normalize_rows(std::vector<float> data, size_t dim) {
    for (size_t r = 0; r < data.size() / dim; ++r) {
        float* row = data.data() + r * dim;
        float norm = 0;
        for (size_t j = 0; j < dim; ++j) norm += row[j] * row[j];
        norm = std::sqrt(norm);
        if (norm > 1e-8f) for (size_t j = 0; j < dim; ++j) row[j] /= norm;
    }
    return data;
}

} // namespace detail

// ---------------------------------------------------------------------------
// SearchResult (row_id, distance)
// ---------------------------------------------------------------------------

struct RocmSearchResult {
    uint64_t row_id;
    float    distance;
};

// ---------------------------------------------------------------------------
// try_rocm_search_batch
//
// Batch top-K search via hipBLAS SGEMM.
// Returns empty vector on failure (caller falls back to CPU).
// ---------------------------------------------------------------------------

inline std::vector<std::vector<RocmSearchResult>>
try_rocm_search_batch(
    const std::vector<const float*>& queries, // [Q] pointers, each [dim]
    const std::vector<uint64_t>&     row_ids, // [N]
    const float*                     flat_vecs,// [N × dim]
    size_t dim, Metric metric, int top_k)
{
    if (kHipLib[0] == '\0' || kBlasLib[0] == '\0') return {};
    size_t n = row_ids.size(), q = queries.size();
    if (n == 0 || q == 0) return std::vector<std::vector<RocmSearchResult>>(q);

    detail::LibGuard hip(kHipLib), blas(kBlasLib);
    if (!hip.ok() || !blas.ok()) return {};

    auto hip_malloc  = hip.fn<fn_hipMalloc>("hipMalloc");
    auto hip_free    = hip.fn<fn_hipFree>("hipFree");
    auto hip_memcpy  = hip.fn<fn_hipMemcpy>("hipMemcpy");
    auto hip_sync    = hip.fn<fn_hipDeviceSynchronize>("hipDeviceSynchronize");
    if (!hip_malloc || !hip_free || !hip_memcpy || !hip_sync) return {};

    void* raw_handle = nullptr;
    auto blas_create  = blas.fn<fn_hipblasCreate>("hipblasCreate");
    auto blas_destroy = blas.fn<fn_hipblasDestroy>("hipblasDestroy");
    auto sgemm        = blas.fn<fn_hipblasSgemm>("hipblasSgemm");
    if (!blas_create || !blas_destroy || !sgemm) return {};
    if (blas_create(&raw_handle) != 0) return {};

    // Flatten queries
    std::vector<float> q_flat(q * dim);
    for (size_t qi = 0; qi < q; ++qi)
        std::memcpy(q_flat.data() + qi * dim, queries[qi], dim * sizeof(float));

    // Normalise for cosine
    std::vector<float> db_norm;
    const float* db_ptr = flat_vecs;
    const float* q_ptr  = q_flat.data();
    std::vector<float> db_owned, q_owned;
    if (metric == Metric::Cosine) {
        db_owned = detail::normalize_rows(std::vector<float>(flat_vecs, flat_vecs + n * dim), dim);
        q_owned  = detail::normalize_rows(std::move(q_flat), dim);
        db_ptr   = db_owned.data();
        q_ptr    = q_owned.data();
    }

    auto db_dev = detail::upload(db_ptr, n * dim, hip_malloc, hip_free, hip_memcpy);
    auto q_dev  = detail::upload(q_ptr,  q * dim, hip_malloc, hip_free, hip_memcpy);
    auto c_dev  = detail::alloc(n * q * sizeof(float), hip_malloc, hip_free);
    if (!db_dev.ptr || !q_dev.ptr || !c_dev.ptr) { blas_destroy(raw_handle); return {}; }

    float alpha = (metric == Metric::Euclidean) ? -2.f : -1.f;
    float beta  = 0.f;
    // SGEMM: C[N×Q col-major] = alpha * db[N×dim]^T * queries[Q×dim]
    int rc = sgemm(raw_handle, kOpT, kOpN,
                   (int)n, (int)q, (int)dim,
                   &alpha,
                   db_dev.ptr, (int)dim,
                   q_dev.ptr,  (int)dim,
                   &beta,
                   c_dev.ptr, (int)n);
    if (rc != 0 || hip_sync() != 0) { blas_destroy(raw_handle); return {}; }

    std::vector<float> c_host(n * q);
    if (hip_memcpy(c_host.data(), c_dev.ptr, n * q * sizeof(float), kD2H) != 0)
        { blas_destroy(raw_handle); return {}; }
    blas_destroy(raw_handle);

    // Per-vector norms for Euclidean
    std::vector<float> db_sq;
    if (metric == Metric::Euclidean) {
        db_sq.resize(n);
        for (size_t ni = 0; ni < n; ++ni) {
            float s = 0;
            for (size_t d = 0; d < dim; ++d) { float v = flat_vecs[ni*dim+d]; s += v*v; }
            db_sq[ni] = s;
        }
    }

    std::vector<std::vector<RocmSearchResult>> results(q);
    for (size_t qi = 0; qi < q; ++qi) {
        float q_sq = 0;
        if (metric == Metric::Euclidean)
            for (size_t d = 0; d < dim; ++d) { float v = queries[qi][d]; q_sq += v*v; }

        std::vector<std::pair<float,size_t>> dists(n);
        for (size_t ni = 0; ni < n; ++ni) {
            float raw = c_host[ni + qi * n];
            float dist;
            switch (metric) {
                case Metric::DotProduct: dist = raw; break;
                case Metric::Cosine:     dist = 1.f + raw; break;
                default:                 dist = std::sqrt(std::max(0.f, q_sq + db_sq[ni] + raw));
            }
            dists[ni] = {dist, ni};
        }
        size_t k = std::min((size_t)top_k, n);
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        results[qi].resize(k);
        for (size_t i = 0; i < k; ++i)
            results[qi][i] = {row_ids[dists[i].second], dists[i].first};
    }
    return results;
}

} // namespace rocm
} // namespace ailake
