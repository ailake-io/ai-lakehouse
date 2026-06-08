// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hamming binary quantization.
//!
//! Binarization: sign of each F32 dimension → 1 bit (positive = 1, negative/zero = 0),
//! packed MSB-first within each byte. Result: `ceil(dim / 8)` bytes per vector.
//!
//! This is **not** RaBitQ — no random rotation applied. Designed for embedding
//! models trained to produce binary-compatible vectors (e.g. Cohere embed-v3
//! `input_type="search_document"` with binary output, Jina ColBERT, etc.).
//! For general-purpose float embeddings use RaBitQ instead.
//!
//! Distance = Hamming(a, b) = popcount(a XOR b) — lower is more similar.
//!
//! SIMD paths:
//!   - x86_64 AVX2/SSSE3: 32 bytes/iter via nibble LUT + PSADBW horizontal sum.
//!   - aarch64 NEON: 16 bytes/iter via vcntq_u8 (per-byte popcount).
//!   - scalar fallback: 8 bytes/iter via u64::count_ones() → maps to `popcnt`.

// ── Encoding ─────────────────────────────────────────────────────────────────

/// Pack an F32 vector into a bit-packed byte vector (1 bit per dimension).
/// Bit at position `i`: 1 if v[i] >= 0.0, else 0.
/// Bits are packed MSB-first: dimension 0 maps to bit 7 of byte 0.
///
/// Output length: `ceil(dim / 8)` bytes.
pub fn f32_to_bits(v: &[f32]) -> Vec<u8> {
    let n_bytes = (v.len() + 7) / 8;
    let mut out = vec![0u8; n_bytes];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            out[i / 8] |= 0x80u8 >> (i % 8);
        }
    }
    out
}

/// Binarize a batch of F32 vectors. Each result has length `ceil(dim / 8)`.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
    vectors.iter().map(|v| f32_to_bits(v)).collect()
}

// ── Hamming distance ─────────────────────────────────────────────────────────

/// Hamming distance between two equal-length bit-packed byte slices.
/// Returns the count of differing bits (lower = more similar).
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("ssse3") {
        return unsafe { hamming_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { hamming_neon(a, b) };
    }
    hamming_scalar(a, b)
}

/// Binarize a query F32 vector and compute Hamming distance to a packed code.
#[inline(always)]
pub fn query_hamming(query: &[f32], code: &[u8]) -> u32 {
    let q_bits = f32_to_bits(query);
    hamming_distance(&q_bits, code)
}

// ── Scalar fallback (u64 chunks → popcnt intrinsic on x86/aarch64) ───────────

#[inline]
fn hamming_scalar(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut dist = 0u32;
    for i in 0..chunks {
        let base = i * 8;
        let a64 = u64::from_le_bytes(a[base..base + 8].try_into().unwrap());
        let b64 = u64::from_le_bytes(b[base..base + 8].try_into().unwrap());
        dist += (a64 ^ b64).count_ones();
    }
    for i in (chunks * 8)..n {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// ── x86_64 AVX2 + SSSE3 (Mula method: nibble LUT + PSADBW) ──────────────────
//
// Reference: Wojciech Mula, "SIMD-friendly algorithms for substring searching"
// 32 bytes/iteration → 4× throughput vs scalar u64 path.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    // 4-bit nibble popcount LUT broadcast to 256-bit
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // lo lane
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // hi lane
    );
    let low4 = _mm256_set1_epi8(0x0F_u8 as i8);
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_si256(); // 4 × u64 accumulators

    let chunks32 = n / 32;
    for i in 0..chunks32 {
        let base = i * 32;
        let av = _mm256_loadu_si256(ap.add(base) as *const __m256i);
        let bv = _mm256_loadu_si256(bp.add(base) as *const __m256i);
        let xored = _mm256_xor_si256(av, bv);

        // Count bits per nibble via LUT
        let lo_nibble = _mm256_and_si256(xored, low4);
        let hi_nibble = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low4);
        let cnt_lo = _mm256_shuffle_epi8(lut, lo_nibble);
        let cnt_hi = _mm256_shuffle_epi8(lut, hi_nibble);
        let cnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);

        // Horizontal byte sum per 64-bit lane via PSADBW (sum of absolute differences vs 0)
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt_bytes, zero));
    }

    // Sum 4 × u64 accumulators
    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    let mut dist = buf[0]
        .wrapping_add(buf[1])
        .wrapping_add(buf[2])
        .wrapping_add(buf[3]) as u32;

    // Tail bytes (< 32)
    for i in (chunks32 * 32)..n {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// ── aarch64 NEON (vcntq_u8 — one instruction per byte) ───────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;

    let n = a.len().min(b.len());
    let mut acc = vdupq_n_u32(0);
    let chunks16 = n / 16;

    for i in 0..chunks16 {
        let base = i * 16;
        let av = vld1q_u8(a.as_ptr().add(base));
        let bv = vld1q_u8(b.as_ptr().add(base));
        let xored = veorq_u8(av, bv);
        let cnt8 = vcntq_u8(xored); // popcount per byte
        let cnt16 = vpaddlq_u8(cnt8); // pairwise widen to u16
        let cnt32 = vpaddlq_u16(cnt16); // pairwise widen to u32
        acc = vaddq_u32(acc, cnt32);
    }

    let mut dist = vaddvq_u32(acc);
    for i in (chunks16 * 16)..n {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_sign_bit_packing() {
        // 8 dims → 1 byte. [+,+,+,+,+,+,+,+] → 0xFF
        let v: Vec<f32> = vec![1.0; 8];
        assert_eq!(f32_to_bits(&v), vec![0xFF]);

        // [+,-,+,-,+,-,+,-] → 0b10101010 = 0xAA
        let v: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert_eq!(f32_to_bits(&v), vec![0xAA]);

        // 9 dims → 2 bytes, last dim in bit 7 of second byte
        let mut v: Vec<f32> = vec![-1.0; 9];
        v[0] = 1.0; // bit 7 of byte 0 → 0x80 in byte 0
        v[8] = 1.0; // bit 7 of byte 1 → 0x80 in byte 1
        assert_eq!(f32_to_bits(&v), vec![0x80, 0x80]);
    }

    #[test]
    fn hamming_identical_zero() {
        let a = f32_to_bits(&vec![1.0, -1.0, 1.0, -1.0]);
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_all_different() {
        let a = f32_to_bits(&vec![1.0; 8]);
        let b = f32_to_bits(&vec![-1.0; 8]);
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn hamming_one_bit() {
        let a = f32_to_bits(&vec![1.0, -1.0]);
        let b = f32_to_bits(&vec![1.0, 1.0]);
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn scalar_simd_agree_dim128() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(77);
        let a: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let a_bits = f32_to_bits(&a);
        let b_bits = f32_to_bits(&b);
        let scalar = hamming_scalar(&a_bits, &b_bits);
        let dispatch = hamming_distance(&a_bits, &b_bits);
        assert_eq!(scalar, dispatch, "scalar={scalar} dispatch={dispatch}");
    }

    #[test]
    fn scalar_simd_agree_dim1536() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..1536).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..1536).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let a_bits = f32_to_bits(&a);
        let b_bits = f32_to_bits(&b);
        let scalar = hamming_scalar(&a_bits, &b_bits);
        let dispatch = hamming_distance(&a_bits, &b_bits);
        assert_eq!(scalar, dispatch, "scalar={scalar} dispatch={dispatch}");
    }
}
