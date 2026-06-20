// SPDX-License-Identifier: MIT OR Apache-2.0
//! Build a Tantivy inverted index from a RecordBatch and serialize it to a blob.
//!
//! Storage rules:
//!   - `row_id`: only stored field (8 bytes/doc)
//!   - `text`:   WithFreqs (BM25 ok), no position index, NOT stored
//!   - Blob is zstd-compressed via `blob` module
//!   - Opt-in via `FtsConfig`; zero overhead by default

use ailake_core::{AilakeError, AilakeResult};
use arrow_array::{Array, RecordBatch};
use tantivy::schema::{IndexRecordOption, TextFieldIndexing, TextOptions, FAST, STORED};

/// Configuration for per-file FTS index construction.
#[derive(Debug, Clone)]
pub struct FtsConfig {
    /// Columns to concatenate and index (space-separated).
    pub text_columns: Vec<String>,
    /// Tantivy tokenizer name. Default: `"default"`.
    pub tokenizer: String,
    /// Heap budget for the Tantivy writer in bytes. Default: 50 MB.
    pub writer_heap_bytes: usize,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            text_columns: vec!["chunk_text".into()],
            tokenizer: "default".into(),
            writer_heap_bytes: 50_000_000,
        }
    }
}

/// Build an FTS index for one write batch and return the serialized blob.
///
/// Sync; safe to call from both sync and async contexts (no blocking executor).
pub fn build_fts_blob_from_batch(config: &FtsConfig, batch: &RecordBatch) -> AilakeResult<Vec<u8>> {
    use arrow_array::cast::AsArray;
    use tantivy::{doc, Index};

    let mut schema_builder = tantivy::schema::Schema::builder();
    let row_id_field = schema_builder.add_u64_field("row_id", FAST | STORED);
    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer(&config.tokenizer)
        .set_index_option(IndexRecordOption::WithFreqs);
    let text_opts = TextOptions::default().set_indexing_options(text_indexing);
    let text_field = schema_builder.add_text_field("text", text_opts);
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer = index
        .writer(config.writer_heap_bytes)
        .map_err(|e| AilakeError::Fts(format!("writer init: {e}")))?;

    for row_idx in 0..batch.num_rows() {
        let text: String = config
            .text_columns
            .iter()
            .filter_map(|col| {
                batch
                    .column_by_name(col)
                    .and_then(|a| a.as_string_opt::<i32>())
                    .and_then(|sa| sa.is_valid(row_idx).then(|| sa.value(row_idx).to_string()))
            })
            .collect::<Vec<_>>()
            .join(" ");
        if !text.is_empty() {
            writer
                .add_document(doc!(row_id_field => row_idx as u64, text_field => text))
                .map_err(|e| AilakeError::Fts(format!("add_document row {row_idx}: {e}")))?;
        }
    }

    writer
        .commit()
        .map_err(|e| AilakeError::Fts(format!("commit: {e}")))?;

    // Serialize via the index's ManagedDirectory which tracks all live segment files.
    // Single write_batch → single segment; no merge needed for typical batch sizes.
    crate::blob::dir_to_blob(index.directory())
}

/// Re-index a merged batch — used by `CompactionExecutor` to rebuild FTS after compaction.
pub fn merge_fts_blobs(config: &FtsConfig, merged_batch: &RecordBatch) -> AilakeResult<Vec<u8>> {
    build_fts_blob_from_batch(config, merged_batch)
}
