// SPDX-License-Identifier: MIT OR Apache-2.0
use ailake_core::{Centroid, VectorMetric};
use half::f16;

// ── Public API ────────────────────────────────────────────────────────────────

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_product: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::dot(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::dot(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon_impl::dot(a, b) };
    }
    dot_scalar(a, b)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "euclidean_distance: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::euclidean(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::euclidean(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon_impl::euclidean(a, b) };
    }
    euclidean_scalar(a, b)
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_distance: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::cosine(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::cosine(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon_impl::cosine(a, b) };
    }
    cosine_scalar(a, b)
}

pub fn exact_distance(metric: VectorMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        VectorMetric::Cosine => cosine_distance(a, b),
        VectorMetric::Euclidean => euclidean_distance(a, b),
        VectorMetric::DotProduct => -dot_product(a, b),
        VectorMetric::NormalizedCosine => normalized_cosine_distance(a, b),
    }
}

// ── F16 distance functions ────────────────────────────────────────────────────
//
// Query `a` stays F32 (one vector, lives in registers).
// Database vector `b` is F16 (dequantized inline — no allocation).
//
// Fast path: F16C converts 8 F16 values to F32 in one instruction via
// _mm256_cvtph_ps, then FMA accumulates. Eliminates scalar half::to_f32()
// loop that dominates HNSW graph traversal on dim=128 vectors.

pub fn cosine_distance_f16(a: &[f32], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_distance_f16: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { avx512::cosine_f16(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { avx2_f16c::cosine(a, b) };
        }
    }
    cosine_f16_scalar(a, b)
}

pub fn euclidean_distance_f16(a: &[f32], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "euclidean_distance_f16: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { avx512::euclidean_f16(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { avx2_f16c::euclidean(a, b) };
        }
    }
    euclidean_f16_scalar(a, b)
}

pub fn dot_product_f16(a: &[f32], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_product_f16: dimension mismatch {} vs {}", a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { avx512::dot_f16(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { avx2_f16c::dot(a, b) };
        }
    }
    dot_f16_scalar(a, b)
}

/// Normalize a vector to unit L2 length. Returns a zero vector unchanged.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq < 1e-12 {
        return v.to_vec();
    }
    let inv = 1.0 / norm_sq.sqrt();
    v.iter().map(|x| x * inv).collect()
}

/// 1 - dot(a, b) for pre-normalized unit vectors — no sqrt, no norm computation.
/// Equivalent to cosine distance but ~2× faster in the HNSW hot loop.
pub fn normalized_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

pub fn normalized_cosine_distance_f16(a: &[f32], b: &[f16]) -> f32 {
    1.0 - dot_product_f16(a, b)
}

pub fn compute_centroid_and_radius(vectors: &[Vec<f32>], metric: VectorMetric) -> Centroid {
    if vectors.is_empty() {
        return Centroid {
            values: vec![],
            radius: 0.0,
            metric,
        };
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let centroid: Vec<f32> = (0..dim)
        .map(|i| vectors.iter().map(|v| v[i]).sum::<f32>() / n)
        .collect();
    let radius = vectors
        .iter()
        .map(|v| exact_distance(metric, &centroid, v))
        .fold(0.0_f32, f32::max);
    Centroid {
        values: centroid,
        radius,
        metric,
    }
}

// ── Scalar fallbacks ──────────────────────────────────────────────────────────

#[inline(always)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline(always)]
fn euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[inline(always)]
fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

#[inline(always)]
fn cosine_f16_scalar(a: &[f32], b: &[f16]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..n {
        let ai = a[i];
        let bi = b[i].to_f32();
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-8 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

#[inline(always)]
fn euclidean_f16_scalar(a: &[f32], b: &[f16]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let diff = a[i] - b[i].to_f32();
        sum += diff * diff;
    }
    sum.sqrt()
}

#[inline(always)]
fn dot_f16_scalar(a: &[f32], b: &[f16]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += a[i] * b[i].to_f32();
    }
    acc
}

// ── x86_64 AVX2 + FMA ────────────────────────────────────────────────────────
//
// Compiled with target_feature = "avx2,fma". The compiler emits vfmadd231ps
// instead of separate vmulps + vaddps, cutting inner-loop instruction count
// by ~33% and reducing latency via fused operations.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn hsum256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf))
    }

    /// dot(a, b) — AVX2+FMA, 2× unrolled (16 f32/iter).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let a0 = _mm256_loadu_ps(ap.add(base));
            let b0 = _mm256_loadu_ps(bp.add(base));
            let a1 = _mm256_loadu_ps(ap.add(base + 8));
            let b1 = _mm256_loadu_ps(bp.add(base + 8));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        }

        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let a0 = _mm256_loadu_ps(ap.add(base));
            let b0 = _mm256_loadu_ps(bp.add(base));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            sum += *ap.add(i) * *bp.add(i);
        }
        sum
    }

    /// ||a - b||² — AVX2+FMA, 2× unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base)), _mm256_loadu_ps(bp.add(base)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(ap.add(base + 8)),
                _mm256_loadu_ps(bp.add(base + 8)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        }

        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base)), _mm256_loadu_ps(bp.add(base)));
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            let d = *ap.add(i) - *bp.add(i);
            sum += d * d;
        }
        sum.sqrt()
    }

    /// 1 - cos(a, b) — AVX2+FMA, single pass for dot + norms².
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();

        let chunks8 = n / 8;
        for i in 0..chunks8 {
            let base = i * 8;
            let av = _mm256_loadu_ps(ap.add(base));
            let bv = _mm256_loadu_ps(bp.add(base));
            dot_acc = _mm256_fmadd_ps(av, bv, dot_acc);
            na_acc = _mm256_fmadd_ps(av, av, na_acc);
            nb_acc = _mm256_fmadd_ps(bv, bv, nb_acc);
        }

        let mut dot = hsum256(dot_acc);
        let mut na2 = hsum256(na_acc);
        let mut nb2 = hsum256(nb_acc);

        for i in (chunks8 * 8)..n {
            let ai = *ap.add(i);
            let bi = *bp.add(i);
            dot += ai * bi;
            na2 += ai * ai;
            nb2 += bi * bi;
        }

        let na = na2.sqrt();
        let nb = nb2.sqrt();
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - dot / (na * nb)
    }
}

// ── x86_64 AVX2 + F16C — F16 hot path ────────────────────────────────────────
//
// _mm256_cvtph_ps converts 8 packed F16 (as __m128i) to 8 F32 in one cycle.
// Combined with FMA, this replaces 8 scalar half::to_f32() calls per iteration.
// Critical hot path: every HNSW edge traversal calls one of these functions.

#[cfg(target_arch = "x86_64")]
mod avx2_f16c {
    use half::f16;
    use std::arch::x86_64::*;

    use super::avx2::hsum256;

    /// dot(a_f32, b_f16) — AVX2+F16C+FMA, 16 F16/iter.
    #[target_feature(enable = "avx2,f16c,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let b1 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base + 8) as *const __m128i));
            let a0 = _mm256_loadu_ps(ap.add(base));
            let a1 = _mm256_loadu_ps(ap.add(base + 8));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        }

        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let a0 = _mm256_loadu_ps(ap.add(base));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            sum += *ap.add(i) * f16::from_bits(*bp.add(i)).to_f32();
        }
        sum
    }

    /// ||a_f32 - b_f16||² — AVX2+F16C+FMA, 16 F16/iter.
    #[target_feature(enable = "avx2,f16c,fma")]
    pub unsafe fn euclidean(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let b1 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base + 8) as *const __m128i));
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base)), b0);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base + 8)), b1);
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        }

        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base)), b0);
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            let diff = *ap.add(i) - f16::from_bits(*bp.add(i)).to_f32();
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// 1 - cos(a_f32, b_f16) — AVX2+F16C+FMA, single pass.
    #[target_feature(enable = "avx2,f16c,fma")]
    pub unsafe fn cosine(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;

        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();

        let chunks8 = n / 8;
        for i in 0..chunks8 {
            let base = i * 8;
            let av = _mm256_loadu_ps(ap.add(base));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            dot_acc = _mm256_fmadd_ps(av, bv, dot_acc);
            na_acc = _mm256_fmadd_ps(av, av, na_acc);
            nb_acc = _mm256_fmadd_ps(bv, bv, nb_acc);
        }

        let mut dot = hsum256(dot_acc);
        let mut na2 = hsum256(na_acc);
        let mut nb2 = hsum256(nb_acc);

        for i in (chunks8 * 8)..n {
            let ai = *ap.add(i);
            let bi = f16::from_bits(*bp.add(i)).to_f32();
            dot += ai * bi;
            na2 += ai * ai;
            nb2 += bi * bi;
        }

        let denom = (na2 * nb2).sqrt();
        if denom < 1e-8 {
            1.0
        } else {
            1.0 - dot / denom
        }
    }
}

// ── x86_64 AVX-512F — forward compatibility ───────────────────────────────────
//
// 16 f32/iter (vs 8 for AVX2). Runtime-detected — skipped on this machine
// (no avx512f), active on Xeon Scalable, Zen 4+, and Intel Core 12th gen+.
// Requires Rust ≥ 1.89 (AVX-512 intrinsics stabilised there). Gated behind
// the `avx512` feature so the default/manylinux build always succeeds.

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512 {
    use half::f16;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum512(v: __m512) -> f32 {
        // _mm512_reduce_add_ps stabilized Rust 1.89; _mm512_extractf32x8_ps needs avx512dq.
        // Store all 16 lanes to stack (avx512f), reload as two __m256 (avx), then reduce.
        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), v);
        let lo = _mm256_loadu_ps(buf.as_ptr());
        let hi = _mm256_loadu_ps(buf.as_ptr().add(8));
        let sum256 = _mm256_add_ps(lo, hi);
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    #[target_feature(enable = "avx512f,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(ap.add(base)),
                _mm512_loadu_ps(bp.add(base)),
                acc,
            );
        }
        let mut sum = hsum512(acc);
        for i in (chunks16 * 16)..n {
            sum += *ap.add(i) * *bp.add(i);
        }
        sum
    }

    #[target_feature(enable = "avx512f,fma")]
    pub unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let d = _mm512_sub_ps(_mm512_loadu_ps(ap.add(base)), _mm512_loadu_ps(bp.add(base)));
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        let mut sum = hsum512(acc);
        for i in (chunks16 * 16)..n {
            let d = *ap.add(i) - *bp.add(i);
            sum += d * d;
        }
        sum.sqrt()
    }

    #[target_feature(enable = "avx512f,fma")]
    pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut dot_acc = _mm512_setzero_ps();
        let mut na_acc = _mm512_setzero_ps();
        let mut nb_acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let av = _mm512_loadu_ps(ap.add(base));
            let bv = _mm512_loadu_ps(bp.add(base));
            dot_acc = _mm512_fmadd_ps(av, bv, dot_acc);
            na_acc = _mm512_fmadd_ps(av, av, na_acc);
            nb_acc = _mm512_fmadd_ps(bv, bv, nb_acc);
        }
        let mut dot = hsum512(dot_acc);
        let mut na2 = hsum512(na_acc);
        let mut nb2 = hsum512(nb_acc);
        for i in (chunks16 * 16)..n {
            let ai = *ap.add(i);
            let bi = *bp.add(i);
            dot += ai * bi;
            na2 += ai * ai;
            nb2 += bi * bi;
        }
        let (na, nb) = (na2.sqrt(), nb2.sqrt());
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - dot / (na * nb)
    }

    /// dot(a_f32, b_f16) — AVX-512F+F16C+FMA, 16 F16/iter.
    #[target_feature(enable = "avx512f,f16c,fma")]
    pub unsafe fn dot_f16(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;
        let mut acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let b_lo = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let b_hi = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base + 8) as *const __m128i));
            let bv = _mm512_insertf32x8(_mm512_castps256_ps512(b_lo), b_hi, 1);
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(base)), bv, acc);
        }
        let mut sum = hsum512(acc);
        for i in (chunks16 * 16)..n {
            sum += *ap.add(i) * f16::from_bits(*bp.add(i)).to_f32();
        }
        sum
    }

    #[target_feature(enable = "avx512f,f16c,fma")]
    pub unsafe fn euclidean_f16(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;
        let mut acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let b_lo = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let b_hi = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base + 8) as *const __m128i));
            let bv = _mm512_insertf32x8(_mm512_castps256_ps512(b_lo), b_hi, 1);
            let d = _mm512_sub_ps(_mm512_loadu_ps(ap.add(base)), bv);
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        let mut sum = hsum512(acc);
        for i in (chunks16 * 16)..n {
            let d = *ap.add(i) - f16::from_bits(*bp.add(i)).to_f32();
            sum += d * d;
        }
        sum.sqrt()
    }

    #[target_feature(enable = "avx512f,f16c,fma")]
    pub unsafe fn cosine_f16(a: &[f32], b: &[f16]) -> f32 {
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr() as *const u16;
        let mut dot_acc = _mm512_setzero_ps();
        let mut na_acc = _mm512_setzero_ps();
        let mut nb_acc = _mm512_setzero_ps();
        let chunks16 = n / 16;
        for i in 0..chunks16 {
            let base = i * 16;
            let b_lo = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base) as *const __m128i));
            let b_hi = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(base + 8) as *const __m128i));
            let bv = _mm512_insertf32x8(_mm512_castps256_ps512(b_lo), b_hi, 1);
            let av = _mm512_loadu_ps(ap.add(base));
            dot_acc = _mm512_fmadd_ps(av, bv, dot_acc);
            na_acc = _mm512_fmadd_ps(av, av, na_acc);
            nb_acc = _mm512_fmadd_ps(bv, bv, nb_acc);
        }
        let mut dot = hsum512(dot_acc);
        let mut na2 = hsum512(na_acc);
        let mut nb2 = hsum512(nb_acc);
        for i in (chunks16 * 16)..n {
            let ai = *ap.add(i);
            let bi = f16::from_bits(*bp.add(i)).to_f32();
            dot += ai * bi;
            na2 += ai * ai;
            nb2 += bi * bi;
        }
        let denom = (na2 * nb2).sqrt();
        if denom < 1e-8 {
            1.0
        } else {
            1.0 - dot / denom
        }
    }
}

// ── aarch64 NEON ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let av = vld1q_f32(a.as_ptr().add(base));
            let bv = vld1q_f32(b.as_ptr().add(base));
            acc = vmlaq_f32(acc, av, bv);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let d = vsubq_f32(
                vld1q_f32(a.as_ptr().add(base)),
                vld1q_f32(b.as_ptr().add(base)),
            );
            acc = vmlaq_f32(acc, d, d);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt()
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut na_acc = vdupq_n_f32(0.0);
        let mut nb_acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let av = vld1q_f32(a.as_ptr().add(base));
            let bv = vld1q_f32(b.as_ptr().add(base));
            dot_acc = vmlaq_f32(dot_acc, av, bv);
            na_acc = vmlaq_f32(na_acc, av, av);
            nb_acc = vmlaq_f32(nb_acc, bv, bv);
        }
        let mut dot = vaddvq_f32(dot_acc);
        let mut na2 = vaddvq_f32(na_acc);
        let mut nb2 = vaddvq_f32(nb_acc);
        for i in (chunks * 4)..n {
            dot += a[i] * b[i];
            na2 += a[i] * a[i];
            nb2 += b[i] * b[i];
        }
        let (na, nb) = (na2.sqrt(), nb2.sqrt());
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - dot / (na * nb)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!(cosine_distance(&v, &v).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal() {
        assert!((cosine_distance(&[1.0f32, 0.0], &[0.0f32, 1.0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn euclidean_basic() {
        assert!((euclidean_distance(&[0.0f32, 0.0], &[3.0f32, 4.0]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn dot_basic() {
        assert!((dot_product(&[1.0f32, 2.0, 3.0], &[4.0f32, 5.0, 6.0]) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn simd_matches_scalar_dim128() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);
        let a: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let dot_s = dot_scalar(&a, &b);
        let euclid_s = euclidean_scalar(&a, &b);
        let cos_s = cosine_scalar(&a, &b);

        let dot_f = dot_product(&a, &b);
        let euclid_f = euclidean_distance(&a, &b);
        let cos_f = cosine_distance(&a, &b);

        assert!(
            (dot_f - dot_s).abs() < 1e-4,
            "dot mismatch: {dot_f} vs {dot_s}"
        );
        assert!(
            (euclid_f - euclid_s).abs() < 1e-4,
            "euclidean mismatch: {euclid_f} vs {euclid_s}"
        );
        assert!(
            (cos_f - cos_s).abs() < 1e-4,
            "cosine mismatch: {cos_f} vs {cos_s}"
        );
    }

    #[test]
    fn f16_simd_matches_scalar() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let b_f32: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let dot_s = dot_f16_scalar(&a, &b);
        let euclid_s = euclidean_f16_scalar(&a, &b);
        let cos_s = cosine_f16_scalar(&a, &b);

        let dot_f = dot_product_f16(&a, &b);
        let euclid_f = euclidean_distance_f16(&a, &b);
        let cos_f = cosine_distance_f16(&a, &b);

        // F16 rounding introduces small error — tolerate 1e-3
        assert!(
            (dot_f - dot_s).abs() < 1e-3,
            "f16 dot mismatch: {dot_f} vs {dot_s}"
        );
        assert!(
            (euclid_f - euclid_s).abs() < 1e-3,
            "f16 euclidean mismatch: {euclid_f} vs {euclid_s}"
        );
        assert!(
            (cos_f - cos_s).abs() < 1e-3,
            "f16 cosine mismatch: {cos_f} vs {cos_s}"
        );
    }

    #[test]
    fn normalize_l2_unit() {
        let v = vec![3.0f32, 4.0];
        let n = normalize_l2(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalized_cosine_matches_cosine_on_unit_vecs() {
        let a = normalize_l2(&[1.0f32, 1.0, 0.0]);
        let b = normalize_l2(&[1.0f32, 0.0, 1.0]);
        let cos = cosine_distance(&a, &b);
        let ncos = normalized_cosine_distance(&a, &b);
        assert!((cos - ncos).abs() < 1e-5, "cos={cos} ncos={ncos}");
    }

    #[test]
    fn centroid_single() {
        let v = vec![vec![1.0f32, 2.0, 3.0]];
        let c = compute_centroid_and_radius(&v, VectorMetric::Cosine);
        assert_eq!(c.values, vec![1.0, 2.0, 3.0]);
        assert!(c.radius < 1e-6, "radius={}", c.radius);
    }

    #[test]
    fn centroid_two_points() {
        let vs = vec![vec![0.0f32, 0.0], vec![2.0f32, 2.0]];
        let c = compute_centroid_and_radius(&vs, VectorMetric::Euclidean);
        assert!((c.values[0] - 1.0).abs() < 1e-6);
        assert!(c.radius > 0.0);
    }
}
