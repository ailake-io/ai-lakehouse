// SPDX-License-Identifier: MIT OR Apache-2.0
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult, ColumnFilter, FilterOp, FilterValue};
use ailake_vec::Quantizer;
use arrow_array::builder::BooleanBuilder;
use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, ListArray, RecordBatch, RecordBatchReader, StringArray,
};
use arrow_schema::{ArrowError, DataType, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter};
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics;

/// Compares a decoded cell value against a filter's target value.
///
/// Shared by both halves of predicate pushdown so they can never disagree:
/// the row-group statistics skip (`row_group_may_match`, coarse — operates on
/// min/max) and the exact per-row `RowFilter` (operates on the actual cell).
/// Mismatched `FilterValue` variants (genuine type mismatch between the
/// filter and the column) return `false` for every op — same "no match"
/// behavior SQL gives a comparison across incompatible types.
fn compare(op: FilterOp, cell: &FilterValue, target: &FilterValue) -> bool {
    let ord = match (cell, target) {
        (FilterValue::I64(a), FilterValue::I64(b)) => a.partial_cmp(b),
        (FilterValue::F64(a), FilterValue::F64(b)) => a.partial_cmp(b),
        (FilterValue::Str(a), FilterValue::Str(b)) => a.as_str().partial_cmp(b.as_str()),
        (FilterValue::Bool(a), FilterValue::Bool(b)) => a.partial_cmp(b),
        _ => None,
    };
    match (op, ord) {
        (FilterOp::Eq, Some(Ordering::Equal)) => true,
        (FilterOp::Ne, Some(o)) => o != Ordering::Equal,
        (FilterOp::Lt, Some(Ordering::Less)) => true,
        (FilterOp::Lte, Some(Ordering::Less | Ordering::Equal)) => true,
        (FilterOp::Gt, Some(Ordering::Greater)) => true,
        (FilterOp::Gte, Some(Ordering::Greater | Ordering::Equal)) => true,
        _ => false,
    }
}

/// Extracts a row-group's (min, max) as `FilterValue`s, when the physical
/// type maps to one we filter on and the bounds aren't deprecated/unreliable.
/// `Int96`/`FixedLenByteArray` and non-UTF8 byte arrays return `None` — the
/// row group is then kept unconditionally (never incorrectly skipped).
fn stats_min_max_as_filter_values(stats: &Statistics) -> Option<(FilterValue, FilterValue)> {
    if stats.is_min_max_deprecated() {
        return None;
    }
    match stats {
        Statistics::Int32(s) => Some((
            FilterValue::I64(*s.min() as i64),
            FilterValue::I64(*s.max() as i64),
        )),
        Statistics::Int64(s) => Some((FilterValue::I64(*s.min()), FilterValue::I64(*s.max()))),
        Statistics::Float(s) => Some((
            FilterValue::F64(*s.min() as f64),
            FilterValue::F64(*s.max() as f64),
        )),
        Statistics::Double(s) => Some((FilterValue::F64(*s.min()), FilterValue::F64(*s.max()))),
        Statistics::Boolean(s) => Some((FilterValue::Bool(*s.min()), FilterValue::Bool(*s.max()))),
        Statistics::ByteArray(s) => {
            let min = s.min().as_utf8().ok()?.to_string();
            let max = s.max().as_utf8().ok()?.to_string();
            Some((FilterValue::Str(min), FilterValue::Str(max)))
        }
        _ => None,
    }
}

/// Returns `false` only when the row group is *provably* unable to contain a
/// matching row (safe to skip — never excludes a group that might match).
fn row_group_may_match(stats: &Statistics, filter: &ColumnFilter) -> bool {
    let Some((min, max)) = stats_min_max_as_filter_values(stats) else {
        return true;
    };
    match filter.op {
        FilterOp::Eq => {
            compare(FilterOp::Lte, &min, &filter.value)
                && compare(FilterOp::Gte, &max, &filter.value)
        }
        // Only provably excludable when every value in the group is the same,
        // and equal to the target (min == max == target).
        FilterOp::Ne => {
            !(compare(FilterOp::Eq, &min, &filter.value)
                && compare(FilterOp::Eq, &max, &filter.value))
        }
        FilterOp::Lt => compare(FilterOp::Lt, &min, &filter.value),
        FilterOp::Lte => compare(FilterOp::Lte, &min, &filter.value),
        FilterOp::Gt => compare(FilterOp::Gt, &max, &filter.value),
        FilterOp::Gte => compare(FilterOp::Gte, &max, &filter.value),
    }
}

/// Decodes a single Arrow array cell into a `FilterValue`, or `None` for a
/// null cell or an unsupported physical type (nulls/unsupported types never
/// match any predicate, consistent with SQL's three-valued comparison logic).
fn extract_cell(col: &dyn Array, i: usize) -> Option<FilterValue> {
    if col.is_null(i) {
        return None;
    }
    match col.data_type() {
        DataType::Int32 => col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| FilterValue::I64(a.value(i) as i64)),
        DataType::Int64 => col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| FilterValue::I64(a.value(i))),
        DataType::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| FilterValue::F64(a.value(i) as f64)),
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| FilterValue::F64(a.value(i))),
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| FilterValue::Str(a.value(i).to_string())),
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|a| FilterValue::Str(a.value(i).to_string())),
        DataType::Boolean => col
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| FilterValue::Bool(a.value(i))),
        _ => None,
    }
}

pub struct ParquetVectorReader {
    bytes: Bytes,
    vector_column: String,
}

impl ParquetVectorReader {
    pub fn new(bytes: Bytes, vector_column: &str) -> Self {
        Self {
            bytes,
            vector_column: vector_column.to_string(),
        }
    }

    /// Detects `ailake.pq_only` and `ailake.precision` from the Parquet KV
    /// metadata, shared by `read_all` and `read_all_filtered`.
    fn pq_only_and_precision(builder: &ParquetRecordBatchReaderBuilder<Bytes>) -> (bool, String) {
        let kvs = builder.metadata().file_metadata().key_value_metadata();
        let pq_only = kvs
            .and_then(|kvs| {
                kvs.iter()
                    .find(|kv| kv.key == "ailake.pq_only")
                    .and_then(|kv| kv.value.as_deref())
                    .map(|v| v == "true")
            })
            .unwrap_or(false);

        // `ailake.precision` (written by every AI-Lake writer path — see
        // ailake-parquet/src/writer.rs) tells us how to decode the raw vector
        // bytes for THIS column, which was previously hardcoded to F16
        // regardless of the file's actual stored precision — silently
        // misreading any table written with F32 (or other) precision (wrong
        // element count from a byte-width mismatch, corrupting every
        // downstream read: compaction, scanner's foreign-file flat scan).
        // Absent for raw external source files fed to `ailake insert` (no
        // AI-Lake writer touched them yet) — default to F16 there, preserving
        // that path's existing documented "F16-encoded input" contract.
        let precision = kvs
            .and_then(|kvs| {
                kvs.iter()
                    .find(|kv| kv.key == "ailake.precision")
                    .and_then(|kv| kv.value.clone())
            })
            .unwrap_or_else(|| "f16".to_string());

        (pq_only, precision)
    }

    /// Splits a decoded tabular+vector `RecordBatch` into (tabular columns,
    /// decoded F32 embeddings), shared by `read_all` and `read_all_filtered`.
    ///
    /// PQ-only files (written with `keep_raw_for_reranking = false`) omit the raw
    /// vector column. For those files, the returned embeddings vec is empty and the
    /// returned RecordBatch contains only tabular columns.
    fn decode_vector_column(
        &self,
        batch: RecordBatch,
        pq_only: bool,
        precision: &str,
    ) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        if pq_only || batch.schema().index_of(&self.vector_column).is_err() {
            return Ok((batch, vec![]));
        }

        let vec_idx = batch.schema().index_of(&self.vector_column).map_err(|_| {
            AilakeError::Parquet(format!(
                "vector column '{}' not found in schema",
                self.vector_column
            ))
        })?;

        let vec_col = batch
            .column(vec_idx)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                AilakeError::Parquet(format!(
                    "vector column '{}' is not FixedSizeBinary (expected F16-encoded floats)",
                    self.vector_column
                ))
            })?;

        let decode: fn(&[u8]) -> Vec<f32> = match precision {
            "f16" => Quantizer::f16_bytes_to_f32,
            "f32" => Quantizer::f32_bytes_to_f32,
            other => {
                return Err(AilakeError::Parquet(format!(
                    "vector column '{}': raw embedding decode not supported for precision '{other}' \
                     (only f16/f32 can be losslessly reconstructed from stored bytes)",
                    self.vector_column
                )));
            }
        };
        let embeddings: Vec<Vec<f32>> = (0..vec_col.len())
            .map(|i| decode(vec_col.value(i)))
            .collect();

        // Return tabular batch without the vector column
        let keep: Vec<usize> = (0..batch.num_columns()).filter(|&i| i != vec_idx).collect();
        let tabular_fields: Vec<_> = keep
            .iter()
            .map(|&i| (*batch.schema().field(i)).clone())
            .collect();
        let tabular_schema = Arc::new(Schema::new(tabular_fields));
        let tabular_cols: Vec<_> = keep.iter().map(|&i| batch.column(i).clone()).collect();
        let tabular = RecordBatch::try_new(tabular_schema, tabular_cols)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        Ok((tabular, embeddings))
    }

    /// Read tabular data and decode the vector column back to F32.
    ///
    /// Reads ALL row groups — uses a large batch size so typical single-row-group
    /// files are returned in one pass, and concatenates multiple batches otherwise.
    ///
    /// PQ-only files (written with `keep_raw_for_reranking = false`) omit the raw
    /// vector column. For those files, the returned embeddings vec is empty and the
    /// returned RecordBatch contains only tabular columns.
    pub fn read_all(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let (pq_only, precision) = Self::pq_only_and_precision(&builder);

        // Use a large batch size to avoid splitting single-row-group files across batches.
        // concatenate_all() handles the multi-batch case transparently.
        let record_count = builder.metadata().file_metadata().num_rows() as usize;
        let batch_size = record_count.max(1);
        let reader = builder
            .with_batch_size(batch_size)
            .build()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Collect all batches (usually just one, but handle multi-row-group files too).
        let mut batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        if batches.is_empty() {
            return Err(AilakeError::Parquet(
                "no batches in Parquet file".to_string(),
            ));
        }

        let batch = if batches.len() == 1 {
            batches.remove(0)
        } else {
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
                .map_err(|e| AilakeError::Parquet(e.to_string()))?
        };

        self.decode_vector_column(batch, pq_only, &precision)
    }

    /// Same contract as `read_all`, but pushes `filter` down to the Parquet
    /// read instead of the caller post-filtering the full batch:
    ///
    ///   1. **Row-group skip** (coarse): row groups whose min/max statistics
    ///      prove no row can match are dropped via `.with_row_groups(...)` —
    ///      their pages are never even fetched/decoded.
    ///   2. **Exact row filter**: an Arrow `RowFilter` evaluates `filter`
    ///      against the real cell value of every row in the surviving row
    ///      groups, so only genuinely matching rows appear in the output —
    ///      this is what makes the result *correct*, not just an optimization.
    ///
    /// Both stages share the same `compare()` so they can't disagree.
    ///
    /// If `filter.column` isn't present in this file's physical schema
    /// (e.g. added later via schema evolution, backfilled only at the Arrow
    /// level by `SchemaFiller` *after* this read), the conservative result is
    /// zero rows — the caller must apply `SchemaFiller` after this call if it
    /// needs evolved-column-aware defaults evaluated against the filter.
    pub fn read_all_filtered(
        &self,
        filter: &ColumnFilter,
    ) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let (pq_only, precision) = Self::pq_only_and_precision(&builder);

        let schema_descr = builder.parquet_schema();
        let col_idx = (0..schema_descr.num_columns())
            .find(|&i| schema_descr.column(i).name() == filter.column);

        let Some(col_idx) = col_idx else {
            let empty = RecordBatch::new_empty(builder.schema().clone());
            return self.decode_vector_column(empty, pq_only, &precision);
        };

        let metadata = builder.metadata();
        let surviving_groups: Vec<usize> = (0..metadata.num_row_groups())
            .filter(|&i| {
                metadata
                    .row_group(i)
                    .column(col_idx)
                    .statistics()
                    .map(|s| row_group_may_match(s, filter))
                    .unwrap_or(true)
            })
            .collect();

        if surviving_groups.is_empty() {
            let empty = RecordBatch::new_empty(builder.schema().clone());
            return self.decode_vector_column(empty, pq_only, &precision);
        }

        let record_count = metadata.file_metadata().num_rows() as usize;
        let batch_size = record_count.max(1);

        let projection = ProjectionMask::leaves(builder.parquet_schema(), [col_idx]);
        let filter_owned = filter.clone();
        let predicate = ArrowPredicateFn::new(projection, move |batch: RecordBatch| {
            let col = batch.column(0);
            let mut out = BooleanBuilder::with_capacity(col.len());
            for i in 0..col.len() {
                let matched = extract_cell(col.as_ref(), i)
                    .map(|cell| compare(filter_owned.op, &cell, &filter_owned.value))
                    .unwrap_or(false);
                out.append_value(matched);
            }
            Ok::<BooleanArray, ArrowError>(out.finish())
        });
        let row_filter = RowFilter::new(vec![Box::new(predicate)]);

        let reader = builder
            .with_row_groups(surviving_groups)
            .with_row_filter(row_filter)
            .with_batch_size(batch_size)
            .build()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        let out_schema = reader.schema();

        let mut batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let batch = if batches.is_empty() {
            RecordBatch::new_empty(out_schema)
        } else if batches.len() == 1 {
            batches.remove(0)
        } else {
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
                .map_err(|e| AilakeError::Parquet(e.to_string()))?
        };

        self.decode_vector_column(batch, pq_only, &precision)
    }

    /// Returns the set of original (file-relative) row indices matching `filter`,
    /// without disturbing row identity — unlike `read_all_filtered`, which
    /// compacts the output and is unsafe to use anywhere a row's position must
    /// still line up with an externally-computed index (e.g. an HNSW result's
    /// `row_id`, or a `RoaringBitmap` deletion vector keyed by file position).
    ///
    /// Cheaper than `read_all_filtered` for this use case too: only the filter
    /// column is decoded (via `.with_projection`), and whole row groups proven
    /// not to match by statistics are skipped row-group-by-row-group, tracking
    /// each survivor's true base offset from `ParquetMetaData` directly (no
    /// batch-boundary bookkeeping needed, since each row group is read via its
    /// own single-group reader).
    pub fn matching_row_ids(&self, filter: &ColumnFilter) -> AilakeResult<HashSet<u64>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let schema_descr = builder.parquet_schema();
        let col_idx = (0..schema_descr.num_columns())
            .find(|&i| schema_descr.column(i).name() == filter.column);
        let Some(col_idx) = col_idx else {
            return Ok(HashSet::new());
        };

        let metadata = builder.metadata();
        let mut base_offsets = Vec::with_capacity(metadata.num_row_groups());
        let mut running = 0u64;
        for i in 0..metadata.num_row_groups() {
            base_offsets.push(running);
            running += metadata.row_group(i).num_rows() as u64;
        }

        let surviving_groups: Vec<usize> = (0..metadata.num_row_groups())
            .filter(|&i| {
                metadata
                    .row_group(i)
                    .column(col_idx)
                    .statistics()
                    .map(|s| row_group_may_match(s, filter))
                    .unwrap_or(true)
            })
            .collect();

        let mut matches = HashSet::new();
        for group in surviving_groups {
            let base = base_offsets[group];
            let group_builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
                .map_err(|e| AilakeError::Parquet(e.to_string()))?;
            let projection = ProjectionMask::leaves(group_builder.parquet_schema(), [col_idx]);
            let reader = group_builder
                .with_row_groups(vec![group])
                .with_projection(projection)
                .build()
                .map_err(|e| AilakeError::Parquet(e.to_string()))?;

            let mut local_offset = 0u64;
            for batch in reader {
                let batch = batch.map_err(|e| AilakeError::Parquet(e.to_string()))?;
                let col = batch.column(0);
                for i in 0..col.len() {
                    if let Some(cell) = extract_cell(col.as_ref(), i) {
                        if compare(filter.op, &cell, &filter.value) {
                            matches.insert(base + local_offset + i as u64);
                        }
                    }
                }
                local_offset += col.len() as u64;
            }
        }
        Ok(matches)
    }

    /// Returns true when this file was written in PQ-only mode (raw vector column omitted).
    pub fn is_pq_only(&self) -> AilakeResult<bool> {
        let v = self.kv_metadata("ailake.pq_only")?;
        Ok(v.as_deref() == Some("true"))
    }

    /// Extract a file-level key_value_metadata entry from the Parquet footer.
    pub fn kv_metadata(&self, key: &str) -> AilakeResult<Option<String>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        let kv = builder.metadata().file_metadata().key_value_metadata();
        Ok(kv.and_then(|kvs| {
            kvs.iter()
                .find(|kv| kv.key == key)
                .and_then(|kv| kv.value.clone())
        }))
    }

    pub fn record_count(&self) -> AilakeResult<u64> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        Ok(builder.metadata().file_metadata().num_rows() as u64)
    }

    /// Read a multi-vector column: `List<FixedSizeBinary(bytes_per_vec)>`.
    ///
    /// Returns `(tabular_batch, embeddings_per_row)` where
    /// `embeddings_per_row[i]` is the Vec of F32 embeddings for row `i`.
    /// Each embedding has `dim` elements.
    pub fn read_all_multi_vec(&self) -> AilakeResult<(RecordBatch, Vec<Vec<Vec<f32>>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        let record_count = builder.metadata().file_metadata().num_rows() as usize;
        let reader = builder
            .with_batch_size(record_count.max(1))
            .build()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let mut batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        if batches.is_empty() {
            return Err(AilakeError::Parquet(
                "no batches in Parquet file".to_string(),
            ));
        }
        let batch = if batches.len() == 1 {
            batches.remove(0)
        } else {
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
                .map_err(|e| AilakeError::Parquet(e.to_string()))?
        };

        let vec_idx = batch.schema().index_of(&self.vector_column).map_err(|_| {
            AilakeError::Parquet(format!(
                "multi-vector column '{}' not found",
                self.vector_column
            ))
        })?;

        let field_type = batch.schema().field(vec_idx).data_type().clone();
        if !matches!(field_type, DataType::List(_)) {
            return Err(AilakeError::Parquet(format!(
                "column '{}' is not a List type; use read_all() for single-vector columns",
                self.vector_column
            )));
        }

        let list_col = batch
            .column(vec_idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                AilakeError::Parquet(format!(
                    "multi-vector column '{}' expected ListArray but got incompatible Arrow type",
                    self.vector_column
                ))
            })?;

        let values = list_col
            .values()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                AilakeError::Parquet(format!(
                    "multi-vector column '{}': ListArray values are not FixedSizeBinary (expected F16-encoded floats)",
                    self.vector_column
                ))
            })?;

        let mut embeddings_per_row: Vec<Vec<Vec<f32>>> = Vec::with_capacity(list_col.len());
        for row in 0..list_col.len() {
            let start = list_col.value_offsets()[row] as usize;
            let end = list_col.value_offsets()[row + 1] as usize;
            let row_vecs: Vec<Vec<f32>> = (start..end)
                .map(|vi| Quantizer::f16_bytes_to_f32(values.value(vi)))
                .collect();
            embeddings_per_row.push(row_vecs);
        }

        // Strip vector column → return only tabular columns
        let keep: Vec<usize> = (0..batch.num_columns()).filter(|&i| i != vec_idx).collect();
        let tabular_fields: Vec<_> = keep
            .iter()
            .map(|&i| (*batch.schema().field(i)).clone())
            .collect();
        let tabular_schema = Arc::new(Schema::new(tabular_fields));
        let tabular_cols: Vec<_> = keep.iter().map(|&i| batch.column(i).clone()).collect();
        let tabular = RecordBatch::try_new(tabular_schema, tabular_cols)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        Ok((tabular, embeddings_per_row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::ParquetVectorWriter;
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_policy(dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
        }
    }

    #[test]
    fn roundtrip_embeddings() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![10, 20, 30]))])
                .unwrap();

        let embs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer.write_batch(&batch, &embs).unwrap();

        let reader = ParquetVectorReader::new(bytes, "embedding");
        let (out_batch, out_embs) = reader.read_all().unwrap();

        assert_eq!(out_batch.num_rows(), 3);
        assert_eq!(out_embs.len(), 3);
        // F16 roundtrip should be close for these unit vectors
        for (orig, decoded) in embs.iter().zip(out_embs.iter()) {
            for (a, b) in orig.iter().zip(decoded.iter()) {
                assert!((a - b).abs() < 0.01, "roundtrip mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn kv_metadata_contains_ailake_keys() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer
            .write_batch(&batch, &[vec![0.1, 0.2, 0.3, 0.4]])
            .unwrap();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        assert_eq!(
            reader.kv_metadata("ailake.format_version").unwrap(),
            Some("1".to_string())
        );
        assert_eq!(
            reader.kv_metadata("ailake.precision").unwrap(),
            Some("f16".to_string())
        );
    }

    #[test]
    fn kv_metadata_absent_for_standard_key() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer
            .write_batch(&batch, &[vec![0.1, 0.2, 0.3, 0.4]])
            .unwrap();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        // Standard reader can read this — but ailake.hnsw_offset is absent in Phase 1
        assert!(reader.kv_metadata("ailake.hnsw_offset").unwrap().is_none());
    }

    #[test]
    fn multi_vec_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Int32, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        use arrow_array::StringArray;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["doc_a", "doc_b"])),
            ],
        )
        .unwrap();

        // doc 1 has 3 chunk embeddings, doc 2 has 2
        let embeddings_per_row: Vec<Vec<Vec<f32>>> = vec![
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            vec![vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]],
        ];

        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, count) = writer
            .write_batch_multi_vec(&batch, &embeddings_per_row, &[])
            .unwrap();
        assert_eq!(count, 2);

        let reader = ParquetVectorReader::new(bytes.clone(), "embedding");
        let (out_batch, out_embs) = reader.read_all_multi_vec().unwrap();

        assert_eq!(out_batch.num_rows(), 2);
        assert_eq!(out_embs.len(), 2);
        assert_eq!(out_embs[0].len(), 3, "doc 1 should have 3 embeddings");
        assert_eq!(out_embs[1].len(), 2, "doc 2 should have 2 embeddings");

        // Verify F16 roundtrip accuracy
        for (row_orig, row_dec) in embeddings_per_row.iter().zip(out_embs.iter()) {
            for (orig, dec) in row_orig.iter().zip(row_dec.iter()) {
                for (a, b) in orig.iter().zip(dec.iter()) {
                    assert!((a - b).abs() < 0.01, "roundtrip mismatch: {a} vs {b}");
                }
            }
        }

        // Verify ailake.multi_vec=true in KV metadata
        assert_eq!(
            reader.kv_metadata("ailake.multi_vec").unwrap(),
            Some("true".to_string())
        );

        // Verify tabular columns preserved correctly
        use arrow_array::Int32Array;
        let doc_ids = out_batch
            .column_by_name("doc_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(doc_ids.value(0), 1);
        assert_eq!(doc_ids.value(1), 2);
    }

    fn write_filter_fixture() -> Bytes {
        use arrow_array::{Int32Array, StringArray};
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    "finance", "sports", "finance", "tech", "sports",
                ])),
            ],
        )
        .unwrap();
        let embs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let writer = ParquetVectorWriter::new(make_policy(4));
        writer.write_batch(&batch, &embs).unwrap().0
    }

    #[test]
    fn read_all_filtered_eq_string() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::eq("category", FilterValue::Str("finance".to_string()));
        let (batch, embs) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(embs.len(), 2);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 3]);
    }

    #[test]
    fn read_all_filtered_range_numeric() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::new("id", FilterOp::Gte, FilterValue::I64(3));
        let (batch, embs) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(embs.len(), 3);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[3, 4, 5]);
    }

    #[test]
    fn read_all_filtered_ne() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::new(
            "category",
            FilterOp::Ne,
            FilterValue::Str("finance".to_string()),
        );
        let (batch, _) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn read_all_filtered_no_match_returns_empty_not_error() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::eq("category", FilterValue::Str("nonexistent".to_string()));
        let (batch, embs) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(embs.len(), 0);
    }

    #[test]
    fn read_all_filtered_missing_column_returns_empty_not_error() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::eq("does_not_exist", FilterValue::Str("x".to_string()));
        let (batch, embs) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(embs.len(), 0);
    }

    #[test]
    fn matching_row_ids_preserves_original_file_positions() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        // Fixture rows (0-indexed): 0=finance, 1=sports, 2=finance, 3=tech, 4=sports.
        let filter = ColumnFilter::eq("category", FilterValue::Str("sports".to_string()));
        let ids = reader.matching_row_ids(&filter).unwrap();
        let mut ids: Vec<u64> = ids.into_iter().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 4]);
    }

    #[test]
    fn matching_row_ids_no_match_is_empty() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::eq("category", FilterValue::Str("nonexistent".to_string()));
        assert!(reader.matching_row_ids(&filter).unwrap().is_empty());
    }

    #[test]
    fn matching_row_ids_missing_column_is_empty() {
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let filter = ColumnFilter::eq("does_not_exist", FilterValue::Str("x".to_string()));
        assert!(reader.matching_row_ids(&filter).unwrap().is_empty());
    }

    #[test]
    fn read_all_filtered_matches_manual_post_filter_of_read_all() {
        // Cross-check: read_all_filtered's pushed-down result must equal
        // manually filtering read_all()'s full output — proves the pushdown
        // doesn't silently drop or admit rows relative to the ground truth.
        let bytes = write_filter_fixture();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        let (full_batch, _) = reader.read_all().unwrap();
        let cats = full_batch
            .column_by_name("category")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let expected: usize = (0..cats.len())
            .filter(|&i| cats.value(i) == "sports")
            .count();

        let filter = ColumnFilter::eq("category", FilterValue::Str("sports".to_string()));
        let (filtered_batch, _) = reader.read_all_filtered(&filter).unwrap();
        assert_eq!(filtered_batch.num_rows(), expected);
    }
}
