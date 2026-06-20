// SPDX-License-Identifier: MIT OR Apache-2.0
// Tests for the AILK_FTS section: write, read, Tantivy round-trip, legacy compat.

use ailake_core::{VectorMetric, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};
use ailake_fts::{FtsConfig, FtsSearcher};
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use std::sync::Arc;

// ── helpers ──────────────────────────────────────────────────────────────────

fn policy() -> VectorStoragePolicy {
    VectorStoragePolicy::default_f16("embedding", 4, VectorMetric::Euclidean)
}

fn make_batch(texts: &[&str]) -> (RecordBatch, Vec<Vec<f32>>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("chunk_text", DataType::Utf8, false),
    ]));
    let ids: Vec<i64> = (0..texts.len() as i64).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();
    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| vec![i as f32, 0.0, 0.0, 0.0])
        .collect();
    (batch, embeddings)
}

fn fts_cfg(cols: &[&str]) -> FtsConfig {
    FtsConfig {
        text_columns: cols.iter().map(|s| s.to_string()).collect(),
        tokenizer: "default".into(),
        writer_heap_bytes: 16 * 1024 * 1024,
    }
}

fn write_file(texts: &[&str], cfg: Option<FtsConfig>) -> Bytes {
    let (batch, embeddings) = make_batch(texts);
    let mut writer = AilakeFileWriter::new(policy());
    if let Some(c) = cfg {
        writer = writer.with_fts(c);
    }
    writer.write(&batch, &embeddings).unwrap()
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// File written with FTS config must have a readable AILK_FTS section.
#[test]
fn write_with_fts_creates_ailk_fts_section() {
    let bytes = write_file(
        &["the quick brown fox", "rust programming language"],
        Some(fts_cfg(&["chunk_text"])),
    );
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let blob = reader.load_fts_blob().expect("load_fts_blob failed");
    assert!(blob.is_some(), "expected AILK_FTS section, got None");
}

/// File written WITHOUT FTS config must return None for load_fts_blob.
#[test]
fn load_fts_blob_returns_none_for_legacy_file() {
    let bytes = write_file(
        &["the quick brown fox", "rust programming language"],
        None, // no FTS
    );
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let blob = reader
        .load_fts_blob()
        .expect("load_fts_blob must not error");
    assert!(blob.is_none(), "legacy file should have no FTS section");
}

/// write → load_fts_blob → FtsSearcher::search → correct row_id returned.
#[test]
fn fts_section_tantivy_search_roundtrip() {
    let texts = &[
        "the quick brown fox jumps",
        "rust programming language systems",
        "machine learning deep neural networks",
    ];
    let bytes = write_file(texts, Some(fts_cfg(&["chunk_text"])));
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let blob = reader
        .load_fts_blob()
        .expect("load_fts_blob failed")
        .expect("expected FTS blob");

    let searcher = FtsSearcher::from_blob(&blob).expect("FtsSearcher::from_blob failed");
    let hits = searcher.search("fox", 5).expect("search failed");

    assert!(!hits.is_empty(), "expected at least one hit for 'fox'");
    assert_eq!(hits[0].row_id, 0, "row_id=0 ('fox' text) should rank first");
}

/// Blob returned by load_fts_blob must start with AFTS magic.
#[test]
fn fts_blob_has_afts_magic() {
    let bytes = write_file(&["hello world"], Some(fts_cfg(&["chunk_text"])));
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let blob = reader.load_fts_blob().unwrap().expect("expected FTS blob");

    assert_eq!(&blob[..4], b"AFTS", "FTS blob must start with AFTS magic");
}

/// File with FTS section must still be a valid Parquet file.
/// Verified by re-reading Parquet row count from reader.
#[test]
fn fts_file_parquet_rows_readable() {
    let texts = &["doc one", "doc two", "doc three"];
    let bytes = write_file(texts, Some(fts_cfg(&["chunk_text"])));
    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let (batch, _) = reader.read_parquet().expect("read_parquet failed");
    assert_eq!(batch.num_rows(), texts.len(), "row count mismatch");
}

/// FTS search over two columns: indexing both chunk_text and a second column.
#[test]
fn fts_multi_column_finds_term_in_second_column() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![0i64, 1])),
            Arc::new(StringArray::from(vec!["generic content", "other content"])),
            Arc::new(StringArray::from(vec![
                "introduction to rust",
                "python basics",
            ])),
        ],
    )
    .unwrap();
    let embeddings = vec![vec![0.0f32; 4], vec![1.0; 4]];

    let cfg = fts_cfg(&["chunk_text", "title"]);
    let writer = AilakeFileWriter::new(policy()).with_fts(cfg);
    let bytes = writer.write(&batch, &embeddings).unwrap();

    let reader = AilakeFileReader::new(bytes, "embedding", 4);
    let blob = reader.load_fts_blob().unwrap().expect("expected FTS blob");
    let searcher = FtsSearcher::from_blob(&blob).unwrap();

    // "rust" appears only in the `title` column of row 0
    let hits = searcher.search("rust", 5).unwrap();
    assert!(!hits.is_empty(), "expected hit for 'rust' in title column");
    assert_eq!(hits[0].row_id, 0, "row 0 has 'rust' in title");
}
