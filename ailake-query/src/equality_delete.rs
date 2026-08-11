// SPDX-License-Identifier: MIT OR Apache-2.0
//! Equality delete filter — Phase H.
//!
//! Loads Iceberg equality delete files from the object store and builds an in-memory
//! predicate set. Applied to each `RecordBatch` during scan to mask logically deleted rows.
//!
//! Scope: single-column equality predicates (most common pattern: document_id, agent_id,
//! session_id). Multi-column AND predicates are supported as long as each column is checked
//! independently (conservative: a row is deleted if ALL delete-file columns match).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ailake_catalog::{read_equality_delete_values, EqualityDeleteFile};
use ailake_core::{AilakeError, AilakeResult};
use ailake_store::Store;
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::DataType;

/// In-memory equality delete filter built from one or more delete files.
///
/// Each entry is one source delete file's own sequence number plus its
/// `column_name → set of string-normalised values to delete`. A data row is
/// deleted by a given entry if, for every column in that entry's predicate,
/// the row's value is a member of that column's set (AND, not OR) — **and**
/// the entry's sequence number is strictly greater than the data file's own
/// (Iceberg spec: a delete only applies to files committed strictly before
/// it). Kept as a `Vec` of per-file predicates rather than one merged map —
/// unlike a plain column→values union, the per-file sequence number can't be
/// collapsed away: two delete files touching the same column need their
/// masks evaluated against different data-file sequence numbers.
pub struct EqualityDeleteFilter {
    /// (delete file's sequence_number, column_name → values to delete)
    filters: Vec<(i64, HashMap<String, HashSet<String>>)>,
}

impl EqualityDeleteFilter {
    /// Build filter from a list of equality delete file references.
    ///
    /// For each file, downloads the Avro payload from `store`, extracts
    /// `(column_name, value)` pairs, and keeps the file's own
    /// `EqualityDeleteFile::sequence_number` alongside — needed later so
    /// `should_delete_row`/`apply` can skip predicates that don't apply to
    /// the data file currently being scanned.
    pub async fn from_files(
        store: &Arc<dyn Store>,
        files: &[EqualityDeleteFile],
    ) -> AilakeResult<Self> {
        let mut filters: Vec<(i64, HashMap<String, HashSet<String>>)> = Vec::new();
        for edf in files {
            let bytes = store.get(&edf.path).await?;
            let pairs = read_equality_delete_values(&bytes)
                .map_err(|e| AilakeError::Catalog(e.to_string()))?;
            let mut cols: HashMap<String, HashSet<String>> = HashMap::new();
            for (col, val) in pairs {
                cols.entry(col).or_default().insert(val);
            }
            filters.push((edf.sequence_number, cols));
        }
        Ok(Self { filters })
    }

    pub fn empty() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Whether `batch`'s row at `row_idx` matches the AND-predicate of a single
    /// delete file's column→values map. Split out of `should_delete_row` so the
    /// sequence-number gate stays the only thing that differs per delete entry.
    fn row_matches(
        cols: &HashMap<String, HashSet<String>>,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> bool {
        let mut any_column_found = false;
        for (col_name, delete_values) in cols {
            let col_idx = match batch.schema().index_of(col_name.as_str()) {
                Ok(i) => i,
                Err(_) => continue, // column absent — skip (schema evolution)
            };
            any_column_found = true;
            let array = batch.column(col_idx);
            if array.is_null(row_idx) {
                return false; // null never matches — AND tuple fails
            }
            let val_str: Option<String> = match array.data_type() {
                DataType::Utf8 => array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row_idx).to_string()),
                DataType::LargeUtf8 => array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .map(|a| a.value(row_idx).to_string()),
                DataType::Int32 => array
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .map(|a| a.value(row_idx).to_string()),
                DataType::Int64 => array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .map(|a| a.value(row_idx).to_string()),
                DataType::Float32 => array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|a| a.value(row_idx).to_string()),
                DataType::Float64 => array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .map(|a| a.value(row_idx).to_string()),
                _ => None,
            };
            match val_str {
                Some(s) if delete_values.contains(&s) => {} // column matches, continue AND check
                Some(_) => return false,                    // column mismatch — AND tuple fails
                None => {}                                  // unknown type — skip column
            }
        }
        any_column_found // true only when all checked columns matched
    }

    /// Check whether a single row (by its physical index in `batch`) matches any
    /// applicable delete predicate.
    ///
    /// Returns `true` if the row should be logically deleted. `data_file_sequence_number`
    /// is the sequence number of the data file `batch` was read from
    /// (`DataFileEntry::sequence_number`) — a delete predicate only applies when its own
    /// sequence number is strictly greater than this, per Iceberg spec. This is what lets
    /// an insert and a delete of the same key committed in the *same* snapshot (equal
    /// sequence numbers) leave the insert visible, instead of the delete masking the row
    /// it was committed alongside.
    pub fn should_delete_row(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
        data_file_sequence_number: i64,
    ) -> bool {
        for (delete_seq, cols) in &self.filters {
            if *delete_seq <= data_file_sequence_number {
                continue; // delete does not apply to this (equal-or-newer) data file
            }
            if Self::row_matches(cols, batch, row_idx) {
                return true;
            }
        }
        false
    }

    /// Apply the filter to `batch`, returning a new batch with matching rows removed.
    ///
    /// `data_file_sequence_number` — see `should_delete_row` doc.
    pub fn apply(
        &self,
        batch: RecordBatch,
        data_file_sequence_number: i64,
    ) -> AilakeResult<RecordBatch> {
        if self.filters.is_empty() {
            return Ok(batch);
        }
        let n = batch.num_rows();
        let keep: Vec<bool> = (0..n)
            .map(|i| !self.should_delete_row(&batch, i, data_file_sequence_number))
            .collect();
        let mask = BooleanArray::from(keep);
        arrow_select::filter::filter_record_batch(&batch, &mask)
            .map_err(|e| AilakeError::Arrow(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::EqualityDeleteFilter;
    use std::collections::{HashMap, HashSet};

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, true),
            Field::new("score", DataType::Int32, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["doc-a", "doc-b", "doc-c", "doc-d"])),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            ],
        )
        .unwrap()
    }

    /// Wraps a single-file predicate at `delete_seq=1` — existing tests below call
    /// `.apply(batch, 0)`/`.should_delete_row(.., 0)` so the delete (seq 1) always
    /// applies to the data (seq 0), preserving their original pre-sequence-scoping
    /// behavior. The scoping behavior itself is covered by the tests further down.
    fn filter_with(filters: HashMap<String, HashSet<String>>) -> EqualityDeleteFilter {
        EqualityDeleteFilter {
            filters: vec![(1, filters)],
        }
    }

    #[test]
    fn empty_filter_is_no_op() {
        let batch = make_batch();
        let f = filter_with(HashMap::new());
        let result = f.apply(batch.clone(), 0).unwrap();
        assert_eq!(result.num_rows(), 4);
    }

    #[test]
    fn single_value_deleted() {
        let mut filters = HashMap::new();
        filters.insert("doc_id".into(), ["doc-b".to_string()].into());
        let f = filter_with(filters);
        let result = f.apply(make_batch(), 0).unwrap();
        assert_eq!(result.num_rows(), 3);
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "doc-a");
        assert_eq!(ids.value(1), "doc-c");
        assert_eq!(ids.value(2), "doc-d");
    }

    #[test]
    fn multiple_values_deleted() {
        let mut filters = HashMap::new();
        filters.insert(
            "doc_id".into(),
            ["doc-a".to_string(), "doc-c".to_string()].into(),
        );
        let f = filter_with(filters);
        let result = f.apply(make_batch(), 0).unwrap();
        assert_eq!(result.num_rows(), 2);
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "doc-b");
        assert_eq!(ids.value(1), "doc-d");
    }

    #[test]
    fn column_absent_from_batch_is_skipped() {
        let mut filters = HashMap::new();
        filters.insert("nonexistent_col".into(), ["x".to_string()].into());
        let f = filter_with(filters);
        let result = f.apply(make_batch(), 0).unwrap();
        assert_eq!(result.num_rows(), 4); // no rows deleted
    }

    #[test]
    fn numeric_column_deletion() {
        let mut filters = HashMap::new();
        filters.insert("score".into(), ["2".to_string(), "4".to_string()].into());
        let f = filter_with(filters);
        let result = f.apply(make_batch(), 0).unwrap();
        assert_eq!(result.num_rows(), 2);
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "doc-a");
        assert_eq!(ids.value(1), "doc-c");
    }

    // ── Sequence-number scoping (the actual bug fix) ────────────────────────────

    #[test]
    fn delete_does_not_mask_a_data_file_with_equal_sequence_number() {
        // Same-snapshot insert+delete of the same key (e.g. an upsert emulated as
        // delete-then-insert in one commit): both land at the same sequence number.
        // Per Iceberg spec (data_seq < delete_seq to mask), an equal sequence number
        // must NOT mask the row — otherwise a real upsert could never make a row
        // reappear after "deleting" the stale version in the same transaction.
        let mut filters = HashMap::new();
        filters.insert("doc_id".into(), ["doc-b".to_string()].into());
        let f = EqualityDeleteFilter {
            filters: vec![(5, filters)],
        };
        let result = f.apply(make_batch(), 5).unwrap();
        assert_eq!(result.num_rows(), 4, "equal sequence number must not mask");
    }

    #[test]
    fn delete_does_not_mask_a_data_file_committed_after_it() {
        // A data file with a HIGHER sequence number than the delete was committed
        // later — e.g. a brand new, unrelated row that happens to reuse a
        // previously-deleted key. The old delete must not reach forward in time
        // and mask it.
        let mut filters = HashMap::new();
        filters.insert("doc_id".into(), ["doc-b".to_string()].into());
        let f = EqualityDeleteFilter {
            filters: vec![(2, filters)],
        };
        let result = f.apply(make_batch(), 5).unwrap();
        assert_eq!(
            result.num_rows(),
            4,
            "a delete must not mask a data file committed after it"
        );
    }

    #[test]
    fn delete_masks_only_data_files_committed_strictly_before_it() {
        let mut filters = HashMap::new();
        filters.insert("doc_id".into(), ["doc-b".to_string()].into());
        let f = EqualityDeleteFilter {
            filters: vec![(5, filters)],
        };
        let result = f.apply(make_batch(), 0).unwrap();
        assert_eq!(result.num_rows(), 3, "older data file must still be masked");
    }

    #[test]
    fn multiple_delete_files_each_scoped_to_their_own_sequence_number() {
        let mut del_a = HashMap::new();
        del_a.insert("doc_id".into(), ["doc-a".to_string()].into());
        let mut del_c = HashMap::new();
        del_c.insert("doc_id".into(), ["doc-c".to_string()].into());
        // doc-a deleted at seq=1 (masks anything committed before seq 1);
        // doc-c deleted at seq=10 (masks anything committed before seq 10).
        let f = EqualityDeleteFilter {
            filters: vec![(1, del_a), (10, del_c)],
        };
        // A data file at seq=5: too new for the doc-a delete (1 <= 5), but the
        // doc-c delete (10 > 5) still applies.
        let result = f.apply(make_batch(), 5).unwrap();
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let survivors: Vec<&str> = (0..result.num_rows()).map(|i| ids.value(i)).collect();
        assert_eq!(survivors, vec!["doc-a", "doc-b", "doc-d"]);
    }
}
