// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ailake-fts` — Tantivy per-file full-text search index for AI-Lake files.
//!
//! Embedded in a separate AILK_FTS section appended after the vector AILK section(s).
//! Opt-in: zero overhead when not configured.

pub mod blob;
pub mod builder;
pub mod searcher;
pub mod tokenizers;

pub use builder::{build_fts_blob_from_batch, merge_fts_blobs, FtsConfig};
pub use searcher::{FtsHit, FtsSearcher};
pub use tokenizers::register_cjk_ngram;
#[cfg(feature = "fts-stemmer-langs")]
pub use tokenizers::register_stemmer_langs;

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
        let texts: Vec<&str> = (0..100)
            .map(|_| "rust programming language fast safe")
            .collect();
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

    /// merge_fts_blobs (compaction path): re-index a merged batch,
    /// result must be searchable for content from both original batches.
    #[test]
    fn merge_fts_blobs_reindexes_combined_batch() {
        let merged_texts = &[
            "rust programming language systems",
            "python machine learning data",
        ];
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let merged_batch = make_batch(merged_texts);
        let blob = merge_fts_blobs(&cfg, &merged_batch).unwrap();
        assert_eq!(&blob[0..4], b"AFTS");

        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        let hits_rust = searcher.search("rust", 5).unwrap();
        assert!(!hits_rust.is_empty(), "expected hit for 'rust' from doc 0");
        assert_eq!(hits_rust[0].row_id, 0);

        let hits_python = searcher.search("python", 5).unwrap();
        assert!(
            !hits_python.is_empty(),
            "expected hit for 'python' from doc 1"
        );
        assert_eq!(hits_python[0].row_id, 1);
    }

    /// Multi-column indexing: term found only in second column must rank.
    #[test]
    fn multi_column_indexing_finds_term_in_second_column() {
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("body", DataType::Utf8, true),
            Field::new("title", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    "generic content only",
                    "other text",
                ])),
                Arc::new(StringArray::from(vec![
                    "introduction to rust",
                    "python basics",
                ])),
            ],
        )
        .unwrap();

        let cfg = FtsConfig {
            text_columns: vec!["body".to_string(), "title".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        // "rust" only in title[0]
        let hits = searcher.search("rust", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "expected hit for 'rust' from title column"
        );
        assert_eq!(hits[0].row_id, 0);
    }

    /// `cjk_ngram` tokenizer: CJK text must produce hits via bigram overlap.
    #[test]
    fn cjk_ngram_finds_japanese_substring() {
        // "人工知能" = artificial intelligence; "機械学習" = machine learning
        let batch = make_batch(&[
            "人工知能システム",
            "機械学習アルゴリズム",
            "rust programming",
        ]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "cjk_ngram".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        // Query "知能" must hit doc 0 (contains bigram 知能)
        let hits = searcher.search("知能", 5).unwrap();
        assert!(!hits.is_empty(), "cjk_ngram: expected hit for 知能");
        assert_eq!(hits[0].row_id, 0, "知能 must rank doc 0 highest");

        // "機械" must hit doc 1
        let hits2 = searcher.search("機械", 5).unwrap();
        assert!(!hits2.is_empty(), "cjk_ngram: expected hit for 機械");
        assert_eq!(hits2[0].row_id, 1, "機械 must rank doc 1 highest");

        // Latin "rust" must still work (falls back to character n-grams)
        let hits3 = searcher.search("rust", 5).unwrap();
        assert!(!hits3.is_empty(), "cjk_ngram: latin term 'rust' must match");
    }

    /// `fr_stem` tokenizer: French stemming must normalize plural → stem.
    #[cfg(feature = "fts-stemmer-langs")]
    #[test]
    fn fr_stem_normalizes_french_words() {
        let batch = make_batch(&["les ordinateurs sont rapides", "le chien aboie dans la rue"]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "fr_stem".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();
        let hits = searcher.search("ordinateur", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "fr_stem: singular must match plural via stem"
        );
        assert_eq!(hits[0].row_id, 0);
    }

    /// `pt_br`: stemming + stop words — plurals must match, stop words must not rank.
    #[cfg(feature = "fts-stemmer-langs")]
    #[test]
    fn pt_br_stems_and_filters_stopwords() {
        let batch = make_batch(&[
            "os computadores modernos são rápidos e eficientes",
            "o cachorro late na rua de manhã",
        ]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "pt_br".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        // "computador" (singular) must match "computadores" (plural) via PT Snowball stem
        let hits = searcher.search("computador", 5).unwrap();
        assert!(
            !hits.is_empty(),
            "pt_br: singular 'computador' must match plural 'computadores'"
        );
        assert_eq!(hits[0].row_id, 0);

        // "cachorro" must rank doc 1
        let hits2 = searcher.search("cachorro", 5).unwrap();
        assert!(!hits2.is_empty(), "pt_br: 'cachorro' must match doc 1");
        assert_eq!(hits2[0].row_id, 1);
    }

    /// `en_stop`: English stop words removed — query on stop word alone must produce no hit.
    #[cfg(feature = "fts-stemmer-langs")]
    #[test]
    fn en_stop_filters_english_stopwords() {
        let batch = make_batch(&[
            "the quick brown fox jumps over the lazy dog",
            "rust programming language memory safety",
        ]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "en_stop".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        // "the" is a stop word — must not be indexed → no hits
        let stop_hits = searcher.search("the", 5).unwrap();
        assert!(
            stop_hits.is_empty(),
            "en_stop: 'the' is stop word, must return no hits"
        );

        // "fox" must still match doc 0
        let hits = searcher.search("fox", 5).unwrap();
        assert!(!hits.is_empty(), "en_stop: 'fox' must still match");
        assert_eq!(hits[0].row_id, 0);

        // Stemming: "programming" → "program" → must match "rust programming"
        let hits2 = searcher.search("program", 5).unwrap();
        assert!(
            !hits2.is_empty(),
            "en_stop: 'program' stem must match 'programming'"
        );
        assert_eq!(hits2[0].row_id, 1);
    }

    /// Query with Tantivy special chars must not panic — escape fallback.
    #[test]
    fn query_with_special_chars_does_not_panic() {
        let batch = make_batch(&["rust OR python", "go AND java", "c++ templates"]);
        let cfg = FtsConfig {
            text_columns: vec!["body".to_string()],
            tokenizer: "default".to_string(),
            writer_heap_bytes: 16 * 1024 * 1024,
        };
        let blob = build_fts_blob_from_batch(&cfg, &batch).unwrap();
        let searcher = FtsSearcher::from_blob(&blob).unwrap();

        // These would be Tantivy parse errors if not escaped; must not panic.
        let r1 = searcher.search("rust (OR) python", 5);
        assert!(r1.is_ok(), "query with parens should not error: {:?}", r1);

        let r2 = searcher.search("go AND OR", 5);
        assert!(
            r2.is_ok(),
            "query with only operators should not error: {:?}",
            r2
        );

        let r3 = searcher.search("c++ templates", 5);
        assert!(r3.is_ok(), "query with '+' should not error: {:?}", r3);
    }
}
