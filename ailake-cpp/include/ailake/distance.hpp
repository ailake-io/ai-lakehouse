// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// CPU distance functions with optional AVX2/AVX-512 SIMD.
#pragma once

#include "footer.hpp"
#include <cmath>
#include <cstddef>
#include <numeric>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace ailake {

// ---------------------------------------------------------------------------
// Scalar implementations (always available)
// ---------------------------------------------------------------------------

inline float cosine_distance_scalar(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 1.f;
    double sim = dot / (std::sqrt(na) * std::sqrt(nb));
    if (sim >  1.0) sim =  1.0;
    if (sim < -1.0) sim = -1.0;
    return static_cast<float>(1.0 - sim);
}

inline float euclidean_distance_scalar(const float* a, const float* b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

inline float dot_product_scalar(const float* a, const float* b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += (double)a[i] * b[i];
    return static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
// AVX2 implementations (x86_64 with -march=native or /arch:AVX2)
// ---------------------------------------------------------------------------

#if defined(__AVX2__)

inline float dot_product_avx2(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

inline float norm_sq_avx2(const float* a, size_t n) {
    return dot_product_avx2(a, a, n);
}

inline float cosine_distance_avx2(const float* a, const float* b, size_t n) {
    float dot = dot_product_avx2(a, b, n);
    float na  = norm_sq_avx2(a, n);
    float nb  = norm_sq_avx2(b, n);
    if (na == 0 || nb == 0) return 1.f;
    float sim = dot / std::sqrt(na * nb);
    if (sim >  1.f) sim =  1.f;
    if (sim < -1.f) sim = -1.f;
    return 1.f - sim;
}

inline float euclidean_distance_avx2(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sq = _mm_cvtss_f32(lo);
    for (; i < n; ++i) { float d = a[i] - b[i]; sq += d * d; }
    return std::sqrt(sq);
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// Dispatch: pick best implementation at compile time
// ---------------------------------------------------------------------------

inline float compute_distance(Metric metric, const float* a, const float* b, size_t n) {
#if defined(__AVX2__)
    switch (metric) {
        case Metric::Euclidean:         return euclidean_distance_avx2(a, b, n);
        case Metric::DotProduct:        return -dot_product_avx2(a, b, n);
        case Metric::NormalizedCosine:  return 1.0f - dot_product_avx2(a, b, n);
        default:                        return cosine_distance_avx2(a, b, n);
    }
#else
    switch (metric) {
        case Metric::Euclidean:         return euclidean_distance_scalar(a, b, n);
        case Metric::DotProduct:        return -dot_product_scalar(a, b, n);
        case Metric::NormalizedCosine:  return 1.0f - dot_product_scalar(a, b, n);
        default:                        return cosine_distance_scalar(a, b, n);
    }
#endif
}

} // namespace ailake
