// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ailake-fts` — Tantivy per-file full-text search index for AI-Lake files.
//!
//! Embedded in a separate AILK_FTS section appended after the vector AILK section(s).
//! Opt-in: zero overhead when not configured.

pub mod blob;
pub mod builder;
pub mod searcher;

pub use builder::{build_fts_blob_from_batch, merge_fts_blobs, FtsConfig};
pub use searcher::{FtsHit, FtsSearcher};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(texts: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::Utf8, true)]));
        let arr = Arc::new(StringArray::from(texts.to_vec()));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn build_and_search_round_trip() {
        let batch = make_batch(&["the quick brown fox", "hello world", "rust is fast"]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        assert!(!blob.is_empty());

        let searcher = FtsSearcher::from_blob(&blob).unwrap();
        let hits = searcher.search("fox", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row_id, 0);
    }

    #[test]
    fn search_no_match_returns_empty() {
        let batch = make_batch(&["the quick brown fox", "hello world"]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();
        let hits = searcher.search("zzznonexistent", 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn blob_is_compressed() {
        let texts: Vec<&str> = (0..100).map(|_| "rust programming language fast safe").collect();
        let batch = make_batch(&texts);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        // blob must start with our MAGIC
        assert_eq!(&blob[0..4], b"AFTS");
    }
}
