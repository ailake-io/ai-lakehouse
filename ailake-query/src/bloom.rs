// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compact Bloom filter for per-file BM25 term pruning (Phase F).
//!
//! Uses k=4 independent hash probes derived from two FNV-64 seeds (double-hashing
//! trick: h_i(x) = h1(x) + i*h2(x) mod m). No external hash dep — FNV-64 is
//! trivially inlined.
//!
//! Serialization: 8-byte little-endian num_bits header + bit words (u64, LE).

const K: usize = 4;

fn fnv64a(data: &[u8], seed: u64) -> u64 {
    const PRIME: u64 = 0x00000100000001B3;
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Probabilistic set membership test for string terms.
///
/// Optimized for the BM25 file-pruning use case: insert all tokenized terms from
/// a data file at write time; at search time, check whether any query term
/// *may* be present. False positives keep the file (safe); false negatives
/// are impossible.
#[derive(Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
}

impl BloomFilter {
    /// Construct a Bloom filter sized for `capacity` items at `fpr` false-positive rate.
    ///
    /// Formula: `m = -n * ln(p) / ln(2)²`, rounded up to the next 64-bit boundary.
    /// A `capacity=10_000, fpr=0.01` filter uses ~12 KB of bit storage.
    pub fn with_capacity(capacity: usize, fpr: f64) -> Self {
        let n = capacity.max(1) as f64;
        let bits_f = -n * fpr.ln() / std::f64::consts::LN_2.powi(2);
        let num_bits = (bits_f.ceil() as usize).max(64);
        // Round up to next multiple of 64 so word-aligned ops are always valid.
        let num_bits = (num_bits + 63) & !63;
        Self {
            bits: vec![0u64; num_bits / 64],
            num_bits,
        }
    }

    fn probes(&self, term: &[u8]) -> [usize; K] {
        let h1 = fnv64a(term, 0);
        let h2 = fnv64a(term, 0xcbf29ce484222325u64);
        let m = self.num_bits as u64;
        let mut out = [0usize; K];
        for i in 0..K {
            out[i] = (h1.wrapping_add((i as u64).wrapping_mul(h2)) % m) as usize;
        }
        out
    }

    pub fn insert(&mut self, term: &str) {
        for pos in self.probes(term.as_bytes()) {
            self.bits[pos / 64] |= 1u64 << (pos % 64);
        }
    }

    /// Returns `true` if `term` *might* be in the set (false positives possible).
    /// Returns `false` only when the term is *definitely absent*.
    pub fn may_contain(&self, term: &str) -> bool {
        for pos in self.probes(term.as_bytes()) {
            if self.bits[pos / 64] & (1u64 << (pos % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Serialize to bytes: `u64_le(num_bits) || u64_le(word_0) || ... || u64_le(word_N)`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.bits.len() * 8);
        out.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        for &w in &self.bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Deserialize from bytes produced by `to_bytes`. Returns `None` on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let num_bits = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        if num_bits == 0 || num_bits % 64 != 0 {
            return None;
        }
        let word_count = num_bits / 64;
        if bytes.len() < 8 + word_count * 8 {
            return None;
        }
        let bits: Vec<u64> = (0..word_count)
            .map(|i| u64::from_le_bytes(bytes[8 + i * 8..16 + i * 8].try_into().unwrap()))
            .collect();
        Some(Self { bits, num_bits })
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserted_terms_always_found() {
        let mut bf = BloomFilter::with_capacity(100, 0.01);
        let terms = ["rust", "iceberg", "puffin", "vector", "bloom"];
        for t in &terms {
            bf.insert(t);
        }
        for t in &terms {
            assert!(bf.may_contain(t), "term '{t}' should be found after insertion");
        }
    }

    #[test]
    fn absent_terms_mostly_absent() {
        let mut bf = BloomFilter::with_capacity(1000, 0.01);
        for i in 0..500u32 {
            bf.insert(&format!("term_{i}"));
        }
        // Check 100 absent terms — expect ~1% false positives max with fpr=0.01
        let fp: usize = (500u32..600)
            .filter(|i| bf.may_contain(&format!("absent_{i}")))
            .count();
        // Allow up to 5% in practice (probabilistic)
        assert!(fp <= 5, "too many false positives: {fp}/100");
    }

    #[test]
    fn roundtrip_serialization() {
        let mut bf = BloomFilter::with_capacity(50, 0.01);
        bf.insert("hello");
        bf.insert("world");
        let bytes = bf.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).expect("deserialization should succeed");
        assert!(restored.may_contain("hello"));
        assert!(restored.may_contain("world"));
        // num_bits preserved
        assert_eq!(restored.num_bits(), bf.num_bits());
    }

    #[test]
    fn from_bytes_returns_none_on_short_input() {
        assert!(BloomFilter::from_bytes(&[]).is_none());
        assert!(BloomFilter::from_bytes(&[0u8; 7]).is_none());
    }

    #[test]
    fn definitely_absent_term_returns_false() {
        let bf = BloomFilter::with_capacity(64, 0.001);
        // Never inserted anything — every term must be absent.
        assert!(!bf.may_contain("anything"), "empty filter must return false for all terms");
    }
}
