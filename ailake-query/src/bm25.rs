// SPDX-License-Identifier: MIT OR Apache-2.0
//! BM25 scoring and corpus IDF statistics for hybrid vector+lexical search.
//!
//! # Storage
//!
//! `IdfStats` is persisted as a compressed binary blob at
//! `<table_root>/metadata/ailake_bm25_stats.bin` (zstd-compressed bincode).
//! The path is recorded in Iceberg table properties under `ailake.bm25.stats-path`
//! so readers know where to find it.
//!
//! # Accuracy
//!
//! IDF is computed from ALL documents written through `TableWriter::write_batch`
//! when a `bm25_text_column` is configured. Concurrent writers may lose DF deltas
//! due to a read-modify-write race on the stats file (same caveat as Iceberg without
//! OCC). Compaction rebuilds stats accurately from all surviving data files.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use ailake_core::{AilakeError, AilakeResult};

/// BM25 hyperparameters.
const K1: f32 = 1.2;
const B: f32 = 0.75;
/// Maximum vocabulary size. Prunes lowest-DF terms when exceeded.
const MAX_VOCAB: usize = 50_000;
/// Minimum term length to index.
const MIN_TERM_LEN: usize = 2;

/// Tokenize text into lowercase alphanumeric terms, dropping single-char tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= MIN_TERM_LEN)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Corpus-level IDF statistics accumulated from all ingested documents.
///
/// Serialized via bincode + zstd and stored as a file alongside the Iceberg metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IdfStats {
    /// Number of documents (rows) seen.
    pub doc_count: u64,
    /// Sum of all document token lengths (used for avg_doc_len).
    pub total_tokens: u64,
    /// Document frequency: number of documents containing each term.
    pub term_df: HashMap<String, u64>,
}

impl IdfStats {
    pub fn avg_doc_len(&self) -> f32 {
        if self.doc_count == 0 {
            1.0
        } else {
            self.total_tokens as f32 / self.doc_count as f32
        }
    }

    /// BM25+ IDF: always positive, avoids negative values for terms appearing in >50% of docs.
    pub fn idf(&self, term: &str) -> f32 {
        let df = self.term_df.get(term).copied().unwrap_or(0) as f32;
        let n = self.doc_count as f32;
        // Terms absent from stats get max IDF (treated as appearing in 1 doc).
        // ln((N - 0 + 0.5) / (0 + 0.5) + 1) ≈ ln(2N + 1) — correctly very high for rare terms.
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Merge DF counts from a new batch of text documents into this stats object.
    ///
    /// Each `&str` is one document. Prunes to `MAX_VOCAB` by dropping lowest-DF
    /// terms after merge (keeps highest-DF terms which are most useful for BM25 normalization).
    pub fn merge_batch(&mut self, texts: &[&str]) {
        for &text in texts {
            let terms = tokenize(text);
            self.doc_count += 1;
            self.total_tokens += terms.len() as u64;

            // Count each unique term at most once per document (for DF, not TF).
            let mut seen = HashMap::<&str, ()>::new();
            for term in &terms {
                if seen.insert(term.as_str(), ()).is_none() {
                    *self.term_df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }

        if self.term_df.len() > MAX_VOCAB {
            // Keep highest-DF terms — common terms anchor BM25 normalization (avgdl)
            // and appear most in queries. Rare unseen terms get max-IDF approximation.
            let mut pairs: Vec<(String, u64)> = self.term_df.drain().collect();
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            pairs.truncate(MAX_VOCAB);
            self.term_df = pairs.into_iter().collect();
        }
    }

    /// Serialize to zstd-compressed bincode bytes.
    pub fn to_bytes(&self) -> AilakeResult<Vec<u8>> {
        let raw = bincode::serialize(self)
            .map_err(|e| AilakeError::Bincode(e.to_string()))?;
        zstd::encode_all(&raw[..], 3)
            .map_err(|e| AilakeError::Io(e))
    }

    /// Deserialize from zstd-compressed bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<Self> {
        let raw = zstd::decode_all(bytes)
            .map_err(|e| AilakeError::Io(e))?;
        bincode::deserialize(&raw)
            .map_err(|e| AilakeError::Bincode(e.to_string()))
    }
}

/// BM25 scorer backed by global [`IdfStats`].
pub struct BM25Scorer<'a> {
    stats: &'a IdfStats,
}

impl<'a> BM25Scorer<'a> {
    pub fn new(stats: &'a IdfStats) -> Self {
        Self { stats }
    }

    /// Score `doc_text` against `query_text`. Returns BM25 score (higher = more relevant).
    pub fn score(&self, query_text: &str, doc_text: &str) -> f32 {
        let query_terms = tokenize(query_text);
        if query_terms.is_empty() {
            return 0.0;
        }

        let doc_terms = tokenize(doc_text);
        let doc_len = doc_terms.len() as f32;
        let avgdl = self.stats.avg_doc_len();

        let mut tf_map: HashMap<&str, u32> = HashMap::new();
        for term in &doc_terms {
            *tf_map.entry(term.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0f32;
        for term in &query_terms {
            let tf = tf_map.get(term.as_str()).copied().unwrap_or(0) as f32;
            if tf == 0.0 {
                continue;
            }
            let idf = self.stats.idf(term);
            // BM25 TF normalization with length penalty
            let tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len / avgdl));
            score += idf * tf_norm;
        }
        score
    }

    /// Compute BM25 scores for a slice of document texts. Returns parallel scores.
    pub fn score_batch(&self, query_text: &str, docs: &[&str]) -> Vec<f32> {
        docs.iter().map(|doc| self.score(query_text, doc)).collect()
    }
}

/// Fusion method for combining vector and BM25 ranked lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridFusion {
    /// Reciprocal Rank Fusion: `score = w_vec/(k+rank_vec) + w_bm25/(k+rank_bm25)`.
    /// Default k=60 (standard RRF). Returned score is negated so sort-ascending = best first.
    Rrf,
    /// Linear combination after min-max normalization:
    /// `score = (1-bm25_weight) * norm_dist + bm25_weight * (1 - norm_bm25)`.
    Linear,
}

impl Default for HybridFusion {
    fn default() -> Self {
        Self::Rrf
    }
}

/// Configuration for hybrid vector+BM25 search.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Raw text query. Tokenized internally for BM25 scoring.
    pub query_text: String,
    /// Parquet column(s) containing the text to score. Typically `["chunk_text"]`.
    /// When multiple columns are given, texts are concatenated with a space separator.
    pub text_columns: Vec<String>,
    /// BM25 weight in RRF fusion (0.0–1.0). Vector weight = 1.0 - bm25_weight in Linear mode.
    /// In Rrf mode, both weights scale their respective RRF rank term.
    pub bm25_weight: f32,
    /// Fusion strategy. Default: `Rrf`.
    pub fusion: HybridFusion,
    /// Minimum number of HNSW candidates to fetch before BM25 re-ranking.
    /// Ensures the BM25 pool is large enough to find lexically relevant results.
    /// Defaults to `max(rerank_factor * top_k, 10 * top_k)` if not set.
    pub candidate_pool: Option<usize>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            text_columns: vec!["chunk_text".to_string()],
            bm25_weight: 0.5,
            fusion: HybridFusion::Rrf,
            candidate_pool: None,
        }
    }
}

impl HybridConfig {
    pub fn new(query_text: impl Into<String>) -> Self {
        Self {
            query_text: query_text.into(),
            ..Default::default()
        }
    }

    pub fn with_text_column(mut self, col: impl Into<String>) -> Self {
        self.text_columns = vec![col.into()];
        self
    }

    pub fn with_text_columns(mut self, cols: Vec<String>) -> Self {
        self.text_columns = cols;
        self
    }

    pub fn with_bm25_weight(mut self, w: f32) -> Self {
        self.bm25_weight = w.clamp(0.0, 1.0);
        self
    }

    pub fn with_fusion(mut self, fusion: HybridFusion) -> Self {
        self.fusion = fusion;
        self
    }

    pub fn with_candidate_pool(mut self, n: usize) -> Self {
        self.candidate_pool = Some(n);
        self
    }
}

/// Apply RRF fusion over (vec_rank, bm25_rank) pairs.
///
/// Returns `-rrf_score` so that sort-ascending-by-distance gives best results first.
pub fn rrf_score(vec_rank: usize, bm25_rank: usize, bm25_weight: f32) -> f32 {
    const RRF_K: f32 = 60.0;
    let vec_weight = 1.0 - bm25_weight;
    let rrf = vec_weight / (RRF_K + vec_rank as f32)
        + bm25_weight / (RRF_K + bm25_rank as f32);
    -rrf
}

/// Apply linear fusion over normalized (vec_dist, bm25_score) pairs.
///
/// Both inputs are normalized to [0,1] across the candidate set before fusion.
/// Returns value in [0,1] where 0 = best (lower = better convention).
pub fn linear_score(
    vec_dist: f32,
    min_vec: f32,
    max_vec: f32,
    bm25: f32,
    min_bm25: f32,
    max_bm25: f32,
    bm25_weight: f32,
) -> f32 {
    let norm_vec = if (max_vec - min_vec).abs() < f32::EPSILON {
        0.0
    } else {
        (vec_dist - min_vec) / (max_vec - min_vec)
    };
    let norm_bm25 = if (max_bm25 - min_bm25).abs() < f32::EPSILON {
        0.5
    } else {
        (bm25 - min_bm25) / (max_bm25 - min_bm25)
    };
    let vec_weight = 1.0 - bm25_weight;
    // Higher BM25 = better = lower final distance: use (1 - norm_bm25)
    vec_weight * norm_vec + bm25_weight * (1.0 - norm_bm25)
}

/// Constant used in Iceberg properties to point at the BM25 stats file.
pub const BM25_STATS_PATH_PROP: &str = "ailake.bm25.stats-path";
/// Default relative path for the BM25 stats file within the table root.
pub const BM25_STATS_FILE: &str = "metadata/ailake_bm25_stats.bin";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // "a" and "is" are filtered (len < 2 and len == 2)
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn idf_empty_corpus_returns_positive() {
        let stats = IdfStats::default();
        let idf = stats.idf("unknown_term");
        assert!(idf > 0.0, "IDF should be positive for unseen term");
    }

    #[test]
    fn merge_batch_accumulates_df() {
        let mut stats = IdfStats::default();
        stats.merge_batch(&["the quick brown fox", "the lazy dog"]);
        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.term_df["the"], 2, "the appears in both docs");
        assert_eq!(stats.term_df["fox"], 1);
        assert_eq!(stats.term_df["dog"], 1);
    }

    #[test]
    fn bm25_scorer_ranks_relevant_doc_higher() {
        let mut stats = IdfStats::default();
        let docs = [
            "rust programming language systems",
            "python machine learning data science",
            "rust memory safety zero cost abstractions",
        ];
        stats.merge_batch(&docs);

        let scorer = BM25Scorer::new(&stats);
        let query = "rust systems programming";
        let s0 = scorer.score(query, docs[0]);
        let s1 = scorer.score(query, docs[1]);
        let s2 = scorer.score(query, docs[2]);

        // docs[0] and docs[2] are about Rust — should score higher than docs[1]
        assert!(s0 > s1, "rust doc scores higher than python doc: s0={s0}, s1={s1}");
        assert!(s2 > s1, "rust doc scores higher than python doc: s2={s2}, s1={s1}");
    }

    #[test]
    fn idf_stats_roundtrip() {
        let mut stats = IdfStats::default();
        stats.merge_batch(&["hello world foo bar", "foo baz qux"]);
        let bytes = stats.to_bytes().unwrap();
        let restored = IdfStats::from_bytes(&bytes).unwrap();
        assert_eq!(restored.doc_count, stats.doc_count);
        assert_eq!(restored.term_df["foo"], 2);
        assert_eq!(restored.term_df["hello"], 1);
    }

    #[test]
    fn vocab_cap_prunes_to_max() {
        let mut stats = IdfStats::default();
        // Generate more than MAX_VOCAB unique terms
        let doc: String = (0..=MAX_VOCAB + 100)
            .map(|i| format!("term{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        stats.merge_batch(&[doc.as_str()]);
        assert!(
            stats.term_df.len() <= MAX_VOCAB,
            "vocab should be capped at {MAX_VOCAB}"
        );
    }

    #[test]
    fn rrf_score_is_negative() {
        let s = rrf_score(0, 0, 0.5);
        assert!(s < 0.0, "RRF score should be negated for sort-ascending convention");
    }

    #[test]
    fn linear_score_in_range() {
        let s = linear_score(0.5, 0.0, 1.0, 0.8, 0.0, 1.0, 0.5);
        assert!((0.0..=1.0).contains(&s), "linear score should be in [0,1]: {s}");
    }
}
