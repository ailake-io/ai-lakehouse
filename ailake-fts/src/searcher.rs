// SPDX-License-Identifier: MIT OR Apache-2.0
//! Query a per-file Tantivy FTS index deserialized from an AILK_FTS blob.

use ailake_core::{AilakeError, AilakeResult};
use tantivy::schema::Value;

/// A single FTS hit returned by `FtsSearcher::search`.
#[derive(Debug)]
pub struct FtsHit {
    /// Row index within the Parquet file (0-based). Matches `row_id` stored in the index.
    pub row_id: u64,
    /// BM25 score from Tantivy. Higher = more relevant.
    pub score: f32,
}

pub struct FtsSearcher {
    index: tantivy::Index,
    reader: tantivy::IndexReader,
    row_id_field: tantivy::schema::Field,
    text_field: tantivy::schema::Field,
}

impl FtsSearcher {
    /// Deserialize a blob (produced by `build_fts_blob_from_batch`) and open it for search.
    pub fn from_blob(blob: &[u8]) -> AilakeResult<Self> {
        let dir = crate::blob::blob_to_ram_dir(blob)?;
        let index =
            tantivy::Index::open(dir).map_err(|e| AilakeError::Fts(format!("open index: {e}")))?;
        let reader = index
            .reader()
            .map_err(|e| AilakeError::Fts(format!("reader: {e}")))?;
        let row_id_field = index
            .schema()
            .get_field("row_id")
            .map_err(|e| AilakeError::Fts(format!("row_id field: {e}")))?;
        let text_field = index
            .schema()
            .get_field("text")
            .map_err(|e| AilakeError::Fts(format!("text field: {e}")))?;
        Ok(Self {
            index,
            reader,
            row_id_field,
            text_field,
        })
    }

    /// Run a BM25 query and return up to `top_k` hits sorted by descending score.
    pub fn search(&self, query_text: &str, top_k: usize) -> AilakeResult<Vec<FtsHit>> {
        use tantivy::collector::TopDocs;
        use tantivy::query::QueryParser;

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);

        // Try full query parser; fall back to stripping special chars; then to
        // quoting each word individually (handles reserved words AND/OR/NOT as literals).
        let query = query_parser
            .parse_query(query_text)
            .or_else(|_| {
                let safe: String = query_text
                    .chars()
                    .map(|c| {
                        if "+-&&||!(){}[]^\"~*?:\\/".contains(c) {
                            ' '
                        } else {
                            c
                        }
                    })
                    .collect();
                query_parser.parse_query(safe.trim())
            })
            .or_else(|_| {
                // Last resort: wrap every word in quotes (phrase query per word = literal match).
                let words: Vec<String> = query_text
                    .split_whitespace()
                    .filter(|w| !w.is_empty())
                    .map(|w| {
                        let escaped = w.replace('\\', "\\\\").replace('"', "\\\"");
                        format!("\"{escaped}\"")
                    })
                    .collect();
                let fallback = words.join(" ");
                if fallback.is_empty() {
                    query_parser.parse_query("")
                } else {
                    query_parser.parse_query(&fallback)
                }
            })
            .map_err(|e| AilakeError::Fts(format!("parse query: {e}")))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k))
            .map_err(|e| AilakeError::Fts(format!("search: {e}")))?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let doc: tantivy::TantivyDocument = searcher
                .doc(doc_addr)
                .map_err(|e| AilakeError::Fts(format!("fetch doc: {e}")))?;
            if let Some(val) = doc.get_first(self.row_id_field) {
                if let Some(row_id) = val.as_u64() {
                    hits.push(FtsHit { row_id, score });
                }
            }
        }
        Ok(hits)
    }
}
