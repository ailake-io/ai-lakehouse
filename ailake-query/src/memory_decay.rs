// SPDX-License-Identifier: MIT OR Apache-2.0
//! Periodic recency-decay job for `EpisodicMemorySchema` tables.
//!
//! Reads the `last_accessed_at` column from each data file (Timestamp(ns, UTC) or legacy Utf8),
//! recomputes
//! `recency_weight = exp(-lambda * days_since_access)`, rewrites the column,
//! and commits a new Iceberg snapshot replacing the old files.
//!
//! Integrates with the existing `CompactionExecutor` infrastructure: it reads
//! and rewrites individual data files (not a merge), preserving HNSW indexes.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{info, warn};

use ailake_catalog::{
    make_multi_column_data_file_entry, new_snapshot_id, CatalogProvider, ExtraVectorIndex,
    NewSnapshot, SnapshotOperation, TableIdent, VectorIndexInfo,
};
use ailake_core::{AilakeError, AilakeResult, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter, VectorColumnBatch};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::{
    Array, Float32Array, RecordBatch, TimestampMicrosecondArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field};

const LAST_ACCESSED_COL: &str = "last_accessed_at";
const RECENCY_WEIGHT_COL: &str = "recency_weight";

/// Periodic job that updates `recency_weight` for all records in a table.
///
/// The weight decays exponentially with age:
/// `recency_weight = exp(-lambda * days_since_last_access)`
///
/// Where `days_since_last_access` is computed from the `last_accessed_at`
/// column (ISO 8601 string or Unix timestamp string in the record).
///
/// # Usage
///
/// ```ignore
/// let job = MemoryDecayJob::new(catalog, store, policy, lambda: 0.1);
/// let updated = job.run(&table).await?;
/// println!("{updated} files updated");
/// ```
pub struct MemoryDecayJob {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
    /// Exponential decay rate. Higher lambda → faster decay.
    /// Typical values: 0.05 (slow) to 0.5 (aggressive).
    pub decay_lambda: f32,
}

impl MemoryDecayJob {
    pub fn new(
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        policy: VectorStoragePolicy,
        decay_lambda: f32,
    ) -> Self {
        Self {
            catalog,
            store,
            policy,
            decay_lambda,
        }
    }

    /// Run decay update across all data files in the table's current snapshot.
    ///
    /// Returns the number of files that were rewritten (files missing the
    /// `last_accessed_at` column are skipped).
    pub async fn run(&self, table: &TableIdent) -> AilakeResult<usize> {
        let files = self.catalog.list_files(table, None).await?;
        if files.is_empty() {
            return Ok(0);
        }

        let today_day = current_day_since_epoch();
        let mut new_entries = Vec::with_capacity(files.len());
        let mut updated = 0usize;

        for file_entry in &files {
            let file_bytes = self.store.get(&file_entry.path).await?;
            // Clone before moving into the primary reader so extra columns can reuse the bytes.
            let orig_bytes = file_bytes.clone();
            let reader =
                AilakeFileReader::new(file_bytes, &self.policy.column_name, self.policy.dim);

            if !reader.is_ailake_file() {
                // Not an AI-Lake file — carry forward unchanged.
                new_entries.push(file_entry.clone());
                continue;
            }

            let (batch, embeddings) = match reader.read_parquet() {
                Ok(pair) => pair,
                Err(e) => {
                    warn!(
                        "ailake: MemoryDecayJob skipping {} — read error: {}",
                        file_entry.path, e
                    );
                    new_entries.push(file_entry.clone());
                    continue;
                }
            };

            if batch.column_by_name(LAST_ACCESSED_COL).is_none() {
                // Table doesn't have last_accessed_at — nothing to decay.
                new_entries.push(file_entry.clone());
                continue;
            }

            let updated_batch = apply_decay(&batch, today_day, self.decay_lambda)?;

            // Read extra column embeddings from original bytes before rewriting.
            let extra_embeddings: Vec<(String, u32, Vec<Vec<f32>>)> = file_entry
                .extra_vector_indexes
                .iter()
                .filter_map(|xi| {
                    let r = AilakeFileReader::new(orig_bytes.clone(), &xi.column, xi.dim);
                    r.read_parquet()
                        .ok()
                        .map(|(_, embs)| (xi.column.clone(), xi.dim, embs))
                })
                .collect();

            // Rewrite file with updated recency_weight column, preserving all HNSW sections.
            let file_writer = AilakeFileWriter::new(self.policy.clone());
            let new_bytes = if extra_embeddings.is_empty() {
                file_writer.write(&updated_batch, &embeddings)?
            } else {
                // Build minimal policies for secondary columns (metric/precision from primary).
                let extra_policies: Vec<VectorStoragePolicy> = extra_embeddings
                    .iter()
                    .map(|(col, dim, _)| VectorStoragePolicy {
                        column_name: col.clone(),
                        dim: *dim,
                        metric: self.policy.metric,
                        precision: self.policy.precision,
                        pq: None,
                        keep_raw_for_reranking: false,
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
                    })
                    .collect();
                let primary_col = VectorColumnBatch {
                    policy: &self.policy,
                    embeddings: &embeddings,
                };
                let mut col_batches: Vec<VectorColumnBatch<'_>> = vec![primary_col];
                for (policy, (_, _, embs)) in extra_policies.iter().zip(extra_embeddings.iter()) {
                    col_batches.push(VectorColumnBatch {
                        policy,
                        embeddings: embs,
                    });
                }
                file_writer.write_multi(&updated_batch, &col_batches)?
            };
            let new_size = new_bytes.len() as u64;
            // Clone before moving into the primary reader used for header parsing.
            let new_bytes_ref = new_bytes.clone();
            self.store.put(&file_entry.path, new_bytes.clone()).await?;

            let centroid = compute_centroid_and_radius(&embeddings, self.policy.metric);
            let new_reader =
                AilakeFileReader::new(new_bytes, &self.policy.column_name, self.policy.dim);
            let header = new_reader.read_header()?;
            let ailk_start = new_reader.ailk_offset()?;

            // Rebuild ExtraVectorIndex entries from the new file's headers.
            let new_extra: Vec<ExtraVectorIndex> = extra_embeddings
                .iter()
                .filter_map(|(col, dim, _)| {
                    let xr = AilakeFileReader::new(new_bytes_ref.clone(), col, *dim);
                    let xailk = xr.ailk_offset_for_column(col).ok()?;
                    let xhdr = xr.read_header_for_column(col).ok()?;
                    Some(ExtraVectorIndex {
                        column: col.clone(),
                        dim: *dim,
                        hnsw_offset: xailk + xhdr.hnsw_offset,
                        hnsw_len: xhdr.hnsw_len,
                        centroid_b64: None,
                        radius: None,
                    })
                })
                .collect();

            let mut new_entry = make_multi_column_data_file_entry(
                &file_entry.path,
                updated_batch.num_rows() as u64,
                new_size,
                &centroid,
                VectorIndexInfo {
                    column: &self.policy.column_name,
                    dim: self.policy.dim,
                    hnsw_offset: ailk_start + header.hnsw_offset,
                    hnsw_len: header.hnsw_len,
                },
                &new_extra,
            );
            // Decay rewrites the file in place (same row count/order — `apply_decay` only
            // replaces/adds a column, never filters rows), so any existing DV bitmap is
            // still positionally valid and must be carried forward, or the rows it masks
            // reappear on the very next search.
            new_entry.deletion_vector = file_entry.deletion_vector.clone();
            new_entries.push(new_entry);
            updated += 1;
        }

        if updated == 0 {
            info!(
                "ailake: MemoryDecayJob — no files with last_accessed_at column; skipping commit"
            );
            return Ok(0);
        }

        let snap = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: new_entries,
            operation: SnapshotOperation::Overwrite,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
            equality_delete_files: vec![],
        };
        self.catalog.commit_snapshot(table, snap).await?;
        info!(
            "ailake: MemoryDecayJob — updated recency_weight in {} files (lambda={})",
            updated, self.decay_lambda
        );
        Ok(updated)
    }
}

/// Extract days-since-access for each row, supporting Timestamp(ns/us) and legacy Utf8.
fn days_old_vec(col: &Arc<dyn Array>, today_day: i64) -> AilakeResult<Vec<f32>> {
    if let Some(ts) = col.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok((0..ts.len())
            .map(|i| {
                if !ts.is_valid(i) {
                    return 0.0f32;
                }
                let day = ts.value(i) / (86_400 * 1_000_000_000i64);
                (today_day - day).max(0) as f32
            })
            .collect());
    }
    if let Some(ts) = col.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok((0..ts.len())
            .map(|i| {
                if !ts.is_valid(i) {
                    return 0.0f32;
                }
                let day = ts.value(i) / (86_400 * 1_000_000i64);
                (today_day - day).max(0) as f32
            })
            .collect());
    }
    if let Some(sa) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Ok((0..sa.len())
            .map(|i| {
                if !sa.is_valid(i) {
                    return 0.0f32;
                }
                let access_day = parse_iso_date_days(sa.value(i)).unwrap_or(today_day);
                (today_day - access_day).max(0) as f32
            })
            .collect());
    }
    Err(AilakeError::Catalog(
        "last_accessed_at must be Timestamp(Nanosecond/Microsecond) or Utf8".into(),
    ))
}

/// Rewrite the `recency_weight` column in `batch` based on `last_accessed_at`.
fn apply_decay(batch: &RecordBatch, today_day: i64, lambda: f32) -> AilakeResult<RecordBatch> {
    let col = batch
        .column_by_name(LAST_ACCESSED_COL)
        .ok_or_else(|| AilakeError::Catalog("last_accessed_at column not found".into()))?;

    let days_old = days_old_vec(col, today_day)?;
    let new_weights: Vec<f32> = days_old.into_iter().map(|d| (-lambda * d).exp()).collect();

    let new_weight_array = Arc::new(Float32Array::from(new_weights));

    // Rebuild RecordBatch replacing (or adding) the recency_weight column.
    let old_schema = batch.schema();
    let decay_field = Field::new(RECENCY_WEIGHT_COL, DataType::Float32, false);

    let mut new_fields: Vec<arrow_schema::FieldRef> = old_schema.fields().iter().cloned().collect();
    let mut new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect();

    if let Some(pos) = old_schema
        .fields()
        .iter()
        .position(|f| f.name() == RECENCY_WEIGHT_COL)
    {
        new_fields[pos] = Arc::new(decay_field);
        new_columns[pos] = new_weight_array;
    } else {
        new_fields.push(Arc::new(decay_field));
        new_columns.push(new_weight_array);
    }

    let new_schema = Arc::new(arrow_schema::Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns).map_err(|e| AilakeError::Arrow(e.to_string()))
}

/// Parse first 10 chars of an ISO 8601 string as YYYY-MM-DD and return
/// days since Unix epoch (1970-01-01). Returns None on parse failure.
fn parse_iso_date_days(s: &str) -> Option<i64> {
    if s.len() < 10 {
        return None;
    }
    let y: i64 = s[0..4].parse().ok()?;
    let m: i64 = s[5..7].parse().ok()?;
    let d: i64 = s[8..10].parse().ok()?;
    // Julian Day Number (Gregorian calendar formula)
    let a = (14 - m) / 12;
    let y2 = y + 4800 - a;
    let m2 = m + 12 * a - 3;
    let jdn = d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045;
    // Unix epoch = JDN 2440588
    Some(jdn - 2440588)
}

fn current_day_since_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64 / 86400)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso_date_unix_epoch() {
        assert_eq!(parse_iso_date_days("1970-01-01T00:00:00"), Some(0));
    }

    #[test]
    fn parse_iso_date_known_date() {
        // 2024-01-15 — verify against known day count
        let days = parse_iso_date_days("2024-01-15").unwrap();
        // 2024-01-15 is 19737 days after 1970-01-01
        assert_eq!(days, 19737);
    }

    #[test]
    fn parse_iso_date_returns_none_on_short_string() {
        assert!(parse_iso_date_days("2024").is_none());
        assert!(parse_iso_date_days("").is_none());
    }

    #[test]
    fn apply_decay_updates_recency_weight() {
        use arrow_array::StringArray;
        use arrow_schema::{Field, Schema};

        let today = current_day_since_epoch();
        // 10 days ago
        let past_day = today - 10;
        let y = 1970 + past_day / 365; // rough
                                       // Use a fixed known date instead
        let past_str = "2024-01-05T00:00:00"; // 10 days before 2024-01-15

        let schema = Arc::new(Schema::new(vec![
            Field::new(LAST_ACCESSED_COL, DataType::Utf8, true),
            Field::new(RECENCY_WEIGHT_COL, DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![past_str])),
                Arc::new(Float32Array::from(vec![1.0f32])),
            ],
        )
        .unwrap();

        // Use today fixed to 2024-01-15 = day 19737
        let today_day = 19737i64;
        let result = apply_decay(&batch, today_day, 0.1).unwrap();
        let weights = result
            .column_by_name(RECENCY_WEIGHT_COL)
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        let w = weights.value(0);
        // 2024-01-05 = day 19727, so 10 days old: exp(-0.1 * 10) = exp(-1) ≈ 0.368
        let expected = (-0.1f32 * 10.0).exp();
        assert!((w - expected).abs() < 0.001, "expected {expected}, got {w}");
        let _ = y; // suppress unused warning
    }

    #[test]
    fn apply_decay_handles_timestamp_nanosecond() {
        use arrow_schema::{Field, Schema, TimeUnit};

        // 2024-01-05 00:00:00 UTC in nanoseconds = day 19727
        // 2024-01-05 = 19727 days × 86400s × 1e9 ns
        let day_19727_ns: i64 = 19727i64 * 86_400 * 1_000_000_000;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                LAST_ACCESSED_COL,
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                true,
            ),
            Field::new(RECENCY_WEIGHT_COL, DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampNanosecondArray::from(vec![day_19727_ns]).with_timezone("UTC")),
                Arc::new(Float32Array::from(vec![1.0f32])),
            ],
        )
        .unwrap();

        // today = 2024-01-15 = day 19737 → 10 days old → exp(-0.1 * 10) ≈ 0.368
        let today_day = 19737i64;
        let result = apply_decay(&batch, today_day, 0.1).unwrap();
        let weights = result
            .column_by_name(RECENCY_WEIGHT_COL)
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let w = weights.value(0);
        let expected = (-0.1f32 * 10.0).exp();
        assert!((w - expected).abs() < 0.001, "expected {expected}, got {w}");
    }

    #[test]
    fn now_ns_is_recent() {
        // now_ns() must be > 2025-01-01 00:00:00 UTC in nanoseconds
        let floor_2025_ns: i64 = 55 * 365 * 86_400 * 1_000_000_000i64; // ~2025
        let t = ailake_core::now_ns();
        assert!(
            t > floor_2025_ns,
            "now_ns() returned suspiciously small value: {t}"
        );
    }
}
