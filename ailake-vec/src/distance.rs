use ailake_core::{Centroid, VectorMetric};

// ── Public API ────────────────────────────────────────────────────────────────

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { dot_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { dot_neon(a, b) };
    }
    dot_scalar(a, b)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { euclidean_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { euclidean_neon(a, b) };
    }
    euclidean_scalar(a, b)
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { cosine_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { cosine_neon(a, b) };
    }
    cosine_scalar(a, b)
}

pub fn exact_distance(metric: VectorMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        VectorMetric::Cosine => cosine_distance(a, b),
        VectorMetric::Euclidean => euclidean_distance(a, b),
        VectorMetric::DotProduct => -dot_product(a, b),
    }
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

// ── x86_64 AVX2 ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of all 8 f32 lanes in a __m256 register.
    #[inline(always)]
    pub unsafe fn hsum256(v: __m256) -> f32 {
        // Fold upper 4 lanes into lower 4
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        // Horizontal sum of 4 lanes in __m128
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf))
    }

    /// dot(a, b) — AVX2, 2× unrolled (16 f32 per iteration).
    #[target_feature(enable = "avx2")]
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
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
        }

        // Handle remaining 8-wide chunk
        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let a0 = _mm256_loadu_ps(ap.add(base));
            let b0 = _mm256_loadu_ps(bp.add(base));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            sum += *ap.add(i) * *bp.add(i);
        }
        sum
    }

    /// ||a - b||  — AVX2, 2× unrolled.
    #[target_feature(enable = "avx2")]
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
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
        }

        let chunks8 = n / 8;
        if chunks8 > chunks16 * 2 {
            let base = chunks16 * 16;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(ap.add(base)), _mm256_loadu_ps(bp.add(base)));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
        }

        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        for i in (chunks8 * 8)..n {
            let d = *ap.add(i) - *bp.add(i);
            sum += d * d;
        }
        sum.sqrt()
    }

    /// 1 - cos(a, b) — AVX2, single pass for dot + norms².
    #[target_feature(enable = "avx2")]
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
            dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(av, bv));
            na_acc = _mm256_add_ps(na_acc, _mm256_mul_ps(av, av));
            nb_acc = _mm256_add_ps(nb_acc, _mm256_mul_ps(bv, bv));
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    avx2::dot(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    avx2::euclidean(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    avx2::cosine(a, b)
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    neon_impl::dot(a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
    neon_impl::euclidean(a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    neon_impl::cosine(a, b)
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

    /// Verify SIMD results match scalar within tolerance for dim=128 vectors.
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
