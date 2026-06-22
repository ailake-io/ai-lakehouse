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
/// Each entry is `column_name → set of string-normalised values to delete`.
/// A data row is deleted if, for every column in the filter, the row's value
/// is a member of that column's set.
pub struct EqualityDeleteFilter {
    /// column_name → set of values (string-normalised) to delete
    filters: HashMap<String, HashSet<String>>,
}

impl EqualityDeleteFilter {
    /// Build filter from a list of equality delete file references.
    ///
    /// For each file, downloads the Avro payload from `store` and extracts
    /// `(column_name, value)` pairs. All files are merged into one filter.
    pub async fn from_files(
        store: &Arc<dyn Store>,
        files: &[EqualityDeleteFile],
    ) -> AilakeResult<Self> {
        let mut filters: HashMap<String, HashSet<String>> = HashMap::new();
        for edf in files {
            let bytes = store.get(&edf.path).await?;
            let pairs = read_equality_delete_values(&bytes)
                .map_err(|e| AilakeError::Catalog(e.to_string()))?;
            for (col, val) in pairs {
                filters.entry(col).or_default().insert(val);
            }
        }
        Ok(Self { filters })
    }

    pub fn empty() -> Self {
        Self {
            filters: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Check whether a single row (by its physical index in `batch`) matches the delete predicate.
    ///
    /// Returns `true` if the row should be logically deleted.
    /// Iceberg equality delete semantics: a row is deleted only when ALL columns in the
    /// delete predicate match (AND, not OR). Used in the per-row HNSW result loop.
    pub fn should_delete_row(&self, batch: &RecordBatch, row_idx: usize) -> bool {
        if self.filters.is_empty() {
            return false;
        }
        let mut any_column_found = false;
        for (col_name, delete_values) in &self.filters {
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

    /// Apply the filter to `batch`, returning a new batch with matching rows removed.
    ///
    /// Iceberg equality delete semantics: a row is removed only when ALL columns in the
    /// delete predicate match (AND). Columns absent from the batch are ignored.
    pub fn apply(&self, batch: RecordBatch) -> AilakeResult<RecordBatch> {
        if self.filters.is_empty() {
            return Ok(batch);
        }
        let n = batch.num_rows();
        let keep: Vec<bool> = (0..n).map(|i| !self.should_delete_row(&batch, i)).collect();
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

    fn filter_with(filters: HashMap<String, HashSet<String>>) -> EqualityDeleteFilter {
        EqualityDeleteFilter { filters }
    }

    #[test]
    fn empty_filter_is_no_op() {
        let batch = make_batch();
        let f = filter_with(HashMap::new());
        let result = f.apply(batch.clone()).unwrap();
        assert_eq!(result.num_rows(), 4);
    }

    #[test]
    fn single_value_deleted() {
        let mut filters = HashMap::new();
        filters.insert("doc_id".into(), ["doc-b".to_string()].into());
        let f = filter_with(filters);
        let result = f.apply(make_batch()).unwrap();
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
        let result = f.apply(make_batch()).unwrap();
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
        let result = f.apply(make_batch()).unwrap();
        assert_eq!(result.num_rows(), 4); // no rows deleted
    }

    #[test]
    fn numeric_column_deletion() {
        let mut filters = HashMap::new();
        filters.insert("score".into(), ["2".to_string(), "4".to_string()].into());
        let f = filter_with(filters);
        let result = f.apply(make_batch()).unwrap();
        assert_eq!(result.num_rows(), 2);
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "doc-a");
        assert_eq!(ids.value(1), "doc-c");
    }
}
