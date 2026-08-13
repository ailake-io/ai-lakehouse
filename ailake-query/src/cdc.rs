// SPDX-License-Identifier: MIT OR Apache-2.0
//! Change Data Capture (CDC) reader for AI-Lake tables.
//!
//! CDC is implemented as a read-only operation over the existing snapshot history.
//! No table-level flag is required: any AI-Lake table with multiple snapshots can
//! produce a change stream as long as the old snapshots and data files are still
//! reachable.
//!
//! The reader compares two snapshots and emits rows annotated with a change
//! envelope (`_change_type`, `_snapshot_id`, `_sequence_number`, `_commit_timestamp`).

use std::collections::{HashMap, HashSet};
use std::ops::Sub;
use std::sync::Arc;

use ailake_catalog::{
    manifest_commit::{list_equality_deletes_from_metadata, list_files_from_metadata},
    read_equality_delete_values as read_eq_delete_avro, CatalogProvider, DataFileEntry,
    EqualityDeleteFile, IcebergMetadata, IcebergSnapshot, SchemaField, SnapshotId, TableIdent,
};
use ailake_core::{AilakeError, AilakeResult};
use ailake_file::AilakeFileReader;
use ailake_store::Store;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_select::take::take;
use roaring::RoaringBitmap;

use crate::dv::load_deletion_vector;
use crate::schema_filler::SchemaFiller;

/// Type of change for a single row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Insert,
    UpdateBefore,
    UpdateAfter,
    Delete,
}

impl ChangeType {
    fn as_str(&self) -> &'static str {
        match self {
            ChangeType::Insert => "insert",
            ChangeType::UpdateBefore => "update_before",
            ChangeType::UpdateAfter => "update_after",
            ChangeType::Delete => "delete",
        }
    }
}

/// One changed row, carrying the row data plus CDC metadata.
#[derive(Debug, Clone)]
pub struct ChangeRecord {
    pub row: RecordBatch,
    pub change_type: ChangeType,
    pub snapshot_id: SnapshotId,
    pub sequence_number: i64,
    pub timestamp_ms: i64,
}

/// Configuration for `read_changes`.
#[derive(Debug, Clone, Default)]
pub struct ChangeReaderConfig {
    /// Start snapshot (inclusive). When `None`, the parent of `end_snapshot_id` is used.
    pub start_snapshot_id: Option<SnapshotId>,
    /// End snapshot (inclusive). When `None`, the current snapshot is used.
    pub end_snapshot_id: Option<SnapshotId>,
    /// Primary-key column names. Required for update coalescing.
    pub pk_columns: Vec<String>,
    /// When `true`, convert a `DELETE` + `INSERT` pair with the same PK within the
    /// same snapshot into `UPDATE_BEFORE` + `UPDATE_AFTER`.
    pub coalesce_updates: bool,
}

/// Read the change stream between two snapshots of an AI-Lake table.
///
/// Returns a single `RecordBatch` containing all changed rows plus the CDC
/// envelope columns `_change_type`, `_snapshot_id`, `_sequence_number`, and
/// `_commit_timestamp`.
///
/// # Semantics
/// - `INSERT`: rows in data files that appear only in the end snapshot.
/// - `DELETE`: rows removed by equality deletes or deletion vectors that are
///   new in the end snapshot, or rows in data files that exist only in the
///   start snapshot.
/// - `UPDATE_BEFORE` / `UPDATE_AFTER`: emitted only when `coalesce_updates`
///   is enabled and a matching PK is both deleted and inserted within the
///   same end snapshot.
pub async fn read_changes(
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: &TableIdent,
    config: ChangeReaderConfig,
) -> AilakeResult<RecordBatch> {
    let meta = catalog.load_raw_metadata(table).await?;
    let (start_id, end_id) =
        resolve_snapshots(&meta, config.start_snapshot_id, config.end_snapshot_id)?;

    let _start_snap = find_snapshot(&meta, start_id)?;
    let end_snap = find_snapshot(&meta, end_id)?;

    let vector_column = meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".to_string());
    let dim = meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let schema_fields = meta.to_table_metadata().schema_fields;

    let start_files = list_files_from_metadata(&*store, &meta, Some(start_id)).await?;
    let end_files = list_files_from_metadata(&*store, &meta, Some(end_id)).await?;
    let start_deletes = list_equality_deletes_from_metadata(&*store, &meta, Some(start_id)).await?;
    let end_deletes = list_equality_deletes_from_metadata(&*store, &meta, Some(end_id)).await?;

    let start_file_map: HashMap<&str, &DataFileEntry> =
        start_files.iter().map(|f| (f.path.as_str(), f)).collect();
    let end_file_map: HashMap<&str, &DataFileEntry> =
        end_files.iter().map(|f| (f.path.as_str(), f)).collect();

    let mut records: Vec<ChangeRecord> = Vec::new();

    // Files only in end → all surviving rows are INSERT.
    for file in &end_files {
        if !start_file_map.contains_key(file.path.as_str()) {
            let dv = load_dv(&store, file).await?;
            let batch =
                read_file_as_batch(&*store, file, &vector_column, dim, &schema_fields).await?;
            let batch = apply_dv(batch, dv.as_ref())?;
            for row in split_batch(batch)? {
                records.push(ChangeRecord {
                    row,
                    change_type: ChangeType::Insert,
                    snapshot_id: end_snap.snapshot_id,
                    sequence_number: end_snap.sequence_number,
                    timestamp_ms: end_snap.timestamp_ms,
                });
            }
        }
    }

    // Files only in start → all rows are DELETE.
    for file in &start_files {
        if !end_file_map.contains_key(file.path.as_str()) {
            let batch =
                read_file_as_batch(&*store, file, &vector_column, dim, &schema_fields).await?;
            for row in split_batch(batch)? {
                records.push(ChangeRecord {
                    row,
                    change_type: ChangeType::Delete,
                    snapshot_id: end_snap.snapshot_id,
                    sequence_number: end_snap.sequence_number,
                    timestamp_ms: end_snap.timestamp_ms,
                });
            }
        }
    }

    // Files in both → detect newly deleted rows via deletion vectors.
    for file in &end_files {
        if let Some(start_file) = start_file_map.get(file.path.as_str()) {
            let start_dv = load_dv(&store, start_file).await?;
            let end_dv = load_dv(&store, file).await?;
            if dv_bitmap(&start_dv) != dv_bitmap(&end_dv) {
                let batch =
                    read_file_as_batch(&*store, file, &vector_column, dim, &schema_fields).await?;
                let deleted = diff_dv(start_dv.as_ref(), end_dv.as_ref());
                let deleted_batch = take_rows(&batch, &deleted)?;
                for row in split_batch(deleted_batch)? {
                    records.push(ChangeRecord {
                        row,
                        change_type: ChangeType::Delete,
                        snapshot_id: end_snap.snapshot_id,
                        sequence_number: end_snap.sequence_number,
                        timestamp_ms: end_snap.timestamp_ms,
                    });
                }
            }
        }
    }

    // Equality deletes added between snapshots → DELETE full rows that match.
    // We read the raw data files (files present in both snapshots) and emit the
    // actual deleted rows, so coalesced updates get a complete UPDATE_BEFORE.
    let new_eq_predicates =
        collect_equality_predicates(&store, &start_deletes, &end_deletes).await?;
    if !new_eq_predicates.is_empty() {
        for file in &end_files {
            if let Some(start_file) = start_file_map.get(file.path.as_str()) {
                let start_dv = load_dv(&store, start_file).await?;
                let end_dv = load_dv(&store, file).await?;
                let batch =
                    read_file_as_batch(&*store, file, &vector_column, dim, &schema_fields).await?;

                // Rows already emitted as DV deletes.
                let dv_deleted: HashSet<u32> = diff_dv(start_dv.as_ref(), end_dv.as_ref())
                    .into_iter()
                    .collect();

                // Rows matching new equality-delete predicates.
                let eq_deleted =
                    find_equality_deleted_rows(&batch, file.sequence_number, &new_eq_predicates);

                let mut deleted: Vec<u32> = dv_deleted
                    .iter()
                    .copied()
                    .chain(eq_deleted.into_iter().filter(|i| !dv_deleted.contains(i)))
                    .collect();
                deleted.sort_unstable();
                deleted.dedup();

                let deleted_batch = take_rows(&batch, &deleted)?;
                for row in split_batch(deleted_batch)? {
                    records.push(ChangeRecord {
                        row,
                        change_type: ChangeType::Delete,
                        snapshot_id: end_snap.snapshot_id,
                        sequence_number: end_snap.sequence_number,
                        timestamp_ms: end_snap.timestamp_ms,
                    });
                }
            }
        }
    }

    // Optional update coalescing.
    if config.coalesce_updates && !config.pk_columns.is_empty() {
        records = coalesce_updates(records, &config.pk_columns)?;
    }

    build_change_batch(records)
}

/// Resolve start/end snapshot IDs, defaulting to parent/current when omitted.
fn resolve_snapshots(
    meta: &IcebergMetadata,
    start: Option<SnapshotId>,
    end: Option<SnapshotId>,
) -> AilakeResult<(SnapshotId, SnapshotId)> {
    let end_id = match end {
        Some(id) => id,
        None => meta
            .current_snapshot_id
            .ok_or_else(|| AilakeError::Catalog("table has no current snapshot".into()))?,
    };

    let start_id = match start {
        Some(id) => id,
        None => {
            let end_snap = find_snapshot(meta, end_id)?;
            match end_snap.parent_snapshot_id {
                Some(id) => id,
                None => {
                    return Err(AilakeError::Catalog(
                        "no start snapshot provided and end snapshot has no parent".into(),
                    ))
                }
            }
        }
    };

    Ok((start_id, end_id))
}

fn find_snapshot(meta: &IcebergMetadata, id: SnapshotId) -> AilakeResult<IcebergSnapshot> {
    meta.snapshots
        .iter()
        .find(|s| s.snapshot_id == id)
        .cloned()
        .ok_or_else(|| AilakeError::Catalog(format!("snapshot {id} not found")))
}

async fn load_dv(
    store: &Arc<dyn Store>,
    file: &DataFileEntry,
) -> AilakeResult<Option<RoaringBitmap>> {
    match &file.deletion_vector {
        Some(dv) => match load_deletion_vector(store, dv).await {
            Ok(bm) => Ok(Some(bm)),
            Err(e) => {
                tracing::warn!("cdc: failed to load deletion vector for {}: {e}", file.path);
                Ok(None)
            }
        },
        None => Ok(None),
    }
}

fn dv_bitmap(dv: &Option<RoaringBitmap>) -> RoaringBitmap {
    dv.clone().unwrap_or_default()
}

fn diff_dv(start: Option<&RoaringBitmap>, end: Option<&RoaringBitmap>) -> Vec<u32> {
    let start = start.cloned().unwrap_or_default();
    let end = end.cloned().unwrap_or_default();
    (end.sub(start)).iter().collect()
}

async fn read_file_as_batch(
    store: &dyn Store,
    file: &DataFileEntry,
    vector_column: &str,
    dim: u32,
    schema_fields: &[SchemaField],
) -> AilakeResult<RecordBatch> {
    let bytes = store.get(&file.path).await?;
    let reader = AilakeFileReader::new(bytes, vector_column, dim);
    let (mut batch, vectors) = reader.read_parquet()?;
    batch = SchemaFiller::fill(batch, schema_fields)?;

    // Re-append the vector column as a FixedSizeList<Float32> if present.
    if !vectors.is_empty() && dim > 0 {
        batch = append_vector_column(batch, &vectors, vector_column, dim)?;
    }

    Ok(batch)
}

fn append_vector_column(
    batch: RecordBatch,
    vectors: &[Vec<f32>],
    vector_column: &str,
    dim: u32,
) -> AilakeResult<RecordBatch> {
    use arrow_schema::Field;
    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let values_arr = Arc::new(Float32Array::from(flat)) as ArrayRef;
    let vec_col = FixedSizeListArray::new(Arc::clone(&item_field), dim as i32, values_arr, None);
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| (**f).clone())
        .collect();
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    fields.push(Field::new(
        vector_column,
        DataType::FixedSizeList(Arc::clone(&item_field), dim as i32),
        true,
    ));
    cols.push(Arc::new(vec_col));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)
        .map_err(|e| AilakeError::Arrow(e.to_string()))
}

fn apply_dv(batch: RecordBatch, dv: Option<&RoaringBitmap>) -> AilakeResult<RecordBatch> {
    let Some(dv) = dv else {
        return Ok(batch);
    };
    if dv.is_empty() {
        return Ok(batch);
    }
    let keep: Vec<u32> = (0..batch.num_rows() as u32)
        .filter(|i| !dv.contains(*i))
        .collect();
    take_rows(&batch, &keep)
}

fn take_rows(batch: &RecordBatch, indices: &[u32]) -> AilakeResult<RecordBatch> {
    if indices.is_empty() {
        return Ok(RecordBatch::new_empty(batch.schema()));
    }
    let idx_arr = UInt32Array::from(indices.to_vec());
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| {
            take(col.as_ref(), &idx_arr, None).map_err(|e| AilakeError::Arrow(e.to_string()))
        })
        .collect::<AilakeResult<_>>()?;
    RecordBatch::try_new(batch.schema(), cols).map_err(|e| AilakeError::Arrow(e.to_string()))
}

fn split_batch(batch: RecordBatch) -> AilakeResult<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let idx = UInt32Array::from(vec![i as u32]);
        let cols: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| {
                take(col.as_ref(), &idx, None).map_err(|e| AilakeError::Arrow(e.to_string()))
            })
            .collect::<AilakeResult<_>>()?;
        out.push(
            RecordBatch::try_new(batch.schema(), cols)
                .map_err(|e| AilakeError::Arrow(e.to_string()))?,
        );
    }
    Ok(out)
}

async fn read_equality_delete_values(
    store: &Arc<dyn Store>,
    eq_delete: &EqualityDeleteFile,
) -> AilakeResult<Vec<(String, String)>> {
    // Prefer the write-path hint when available (e.g. DuckLake in-memory entries),
    // otherwise load the Avro delete file from the store.
    if let Some((col, vals)) = &eq_delete.inline_values {
        return Ok(vals.iter().map(|v| (col.clone(), v.clone())).collect());
    }
    let bytes = store.get(&eq_delete.path).await?;
    read_eq_delete_avro(&bytes).map_err(|e| {
        AilakeError::Catalog(format!(
            "failed to read equality delete {}: {e}",
            eq_delete.path
        ))
    })
}

/// Build a list of new equality-delete predicates added between the start and
/// end snapshots. Each predicate is `(column, value, delete_sequence_number)`.
async fn collect_equality_predicates(
    store: &Arc<dyn Store>,
    start_deletes: &[EqualityDeleteFile],
    end_deletes: &[EqualityDeleteFile],
) -> AilakeResult<Vec<(String, String, i64)>> {
    let mut predicates = Vec::new();
    for ed in end_deletes {
        if start_deletes.iter().any(|sed| sed.path == ed.path) {
            continue;
        }
        let seq = ed.sequence_number;
        let values = read_equality_delete_values(store, ed).await?;
        for (col, val) in values {
            predicates.push((col, val, seq));
        }
    }
    Ok(predicates)
}

/// Return row indices in `batch` that match any equality-delete predicate whose
/// sequence number is strictly greater than the data file's sequence number.
fn find_equality_deleted_rows(
    batch: &RecordBatch,
    file_sequence_number: i64,
    predicates: &[(String, String, i64)],
) -> Vec<u32> {
    let mut matches = Vec::new();
    let schema = batch.schema();
    for (col, val, delete_seq) in predicates {
        if *delete_seq <= file_sequence_number {
            continue;
        }
        let Ok(col_idx) = schema.index_of(col) else {
            continue;
        };
        let array = batch.column(col_idx);
        for row in 0..batch.num_rows() {
            if array_value_to_string(array, row) == *val {
                matches.push(row as u32);
            }
        }
    }
    matches.sort_unstable();
    matches.dedup();
    matches
}

fn coalesce_updates(
    records: Vec<ChangeRecord>,
    pk_columns: &[String],
) -> AilakeResult<Vec<ChangeRecord>> {
    let mut inserts_by_pk: HashMap<String, Vec<usize>> = HashMap::new();
    let mut deletes_by_pk: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, rec) in records.iter().enumerate() {
        let key = pk_key(&rec.row, pk_columns)?;
        match rec.change_type {
            ChangeType::Insert => inserts_by_pk.entry(key).or_default().push(i),
            ChangeType::Delete => deletes_by_pk.entry(key).or_default().push(i),
            _ => {}
        }
    }

    let mut used = HashSet::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    // First pass: pair each DELETE with the first unpaired matching INSERT.
    for (key, del_idxs) in deletes_by_pk {
        for del_i in del_idxs {
            if used.contains(&del_i) {
                continue;
            }
            if let Some(ins_idxs) = inserts_by_pk.get(&key) {
                if let Some(&ins_i) = ins_idxs.iter().find(|idx| !used.contains(*idx)) {
                    used.insert(del_i);
                    used.insert(ins_i);
                    pairs.push((del_i, ins_i));
                }
            }
        }
    }

    // Build output preserving original order of unpaired records, expanding each
    // paired DELETE+INSERT into UPDATE_BEFORE + UPDATE_AFTER at the DELETE position.
    let mut out: Vec<ChangeRecord> = Vec::with_capacity(records.len());
    let mut pair_idx = 0;
    for (i, rec) in records.iter().enumerate() {
        if used.contains(&i) {
            // If this is the DELETE of a pair, emit the coalesced update records.
            if pair_idx < pairs.len() && pairs[pair_idx].0 == i {
                let (_, ins_i) = pairs[pair_idx];
                pair_idx += 1;
                let delete_rec = &records[i];
                let insert_rec = &records[ins_i];
                out.push(ChangeRecord {
                    row: delete_rec.row.clone(),
                    change_type: ChangeType::UpdateBefore,
                    snapshot_id: delete_rec.snapshot_id,
                    sequence_number: delete_rec.sequence_number,
                    timestamp_ms: delete_rec.timestamp_ms,
                });
                out.push(ChangeRecord {
                    row: insert_rec.row.clone(),
                    change_type: ChangeType::UpdateAfter,
                    snapshot_id: insert_rec.snapshot_id,
                    sequence_number: insert_rec.sequence_number,
                    timestamp_ms: insert_rec.timestamp_ms,
                });
            }
            continue;
        }
        out.push(rec.clone());
    }

    Ok(out)
}

fn pk_key(batch: &RecordBatch, pk_columns: &[String]) -> AilakeResult<String> {
    let mut parts = Vec::with_capacity(pk_columns.len());
    for col in pk_columns {
        let idx = batch
            .schema()
            .index_of(col)
            .map_err(|e| AilakeError::Arrow(format!("pk column {col} not found: {e}")))?;
        let array = batch.column(idx);
        let val = if array.is_null(0) {
            "NULL".to_string()
        } else {
            array_value_to_string(array, 0)
        };
        parts.push(val);
    }
    Ok(parts.join("|"))
}

fn array_value_to_string(array: &ArrayRef, row: usize) -> String {
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    if array.is_null(row) {
        return "NULL".to_string();
    }
    match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(row).to_string(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(row).to_string(),
        DataType::Int32 => array
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        DataType::Int64 => array
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(row)
            .to_string(),
        DataType::Float32 => array
            .as_primitive::<arrow_array::types::Float32Type>()
            .value(row)
            .to_string(),
        DataType::Float64 => array
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(row)
            .to_string(),
        DataType::Boolean => array.as_boolean().value(row).to_string(),
        _ => format!("{:?}", array),
    }
}

fn build_change_batch(records: Vec<ChangeRecord>) -> AilakeResult<RecordBatch> {
    if records.is_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_change_type", DataType::Utf8, false),
            Field::new("_snapshot_id", DataType::Int64, false),
            Field::new("_sequence_number", DataType::Int64, false),
            Field::new("_commit_timestamp", DataType::Int64, false),
        ]));
        return Ok(RecordBatch::new_empty(schema));
    }

    // Records may have heterogeneous schemas (e.g. equality-delete DELETE rows
    // carry only PK columns, while INSERT rows carry the full schema). Build the
    // union schema and pad missing columns with nulls before concatenating.
    let base_schema = union_schema(&records);
    let records = records
        .into_iter()
        .map(|rec| normalize_to_schema(rec, &base_schema))
        .collect::<AilakeResult<Vec<_>>>()?;

    let mut fields: Vec<Field> = base_schema.fields().iter().map(|f| (**f).clone()).collect();
    fields.push(Field::new("_change_type", DataType::Utf8, false));
    fields.push(Field::new("_snapshot_id", DataType::Int64, false));
    fields.push(Field::new("_sequence_number", DataType::Int64, false));
    fields.push(Field::new("_commit_timestamp", DataType::Int64, false));
    let out_schema = Arc::new(Schema::new(fields));

    let mut base_cols: Vec<Vec<ArrayRef>> = vec![vec![]; base_schema.fields().len()];
    let mut types = Vec::with_capacity(records.len());
    let mut snap_ids = Vec::with_capacity(records.len());
    let mut seqs = Vec::with_capacity(records.len());
    let mut timestamps = Vec::with_capacity(records.len());

    for rec in records {
        types.push(rec.change_type.as_str());
        snap_ids.push(rec.snapshot_id);
        seqs.push(rec.sequence_number);
        timestamps.push(rec.timestamp_ms);
        for (i, col) in rec.row.columns().iter().enumerate() {
            base_cols[i].push(col.clone());
        }
    }

    use arrow_select::concat::concat;
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(base_schema.fields().len() + 4);
    for col_parts in base_cols {
        let arrays: Vec<&dyn Array> = col_parts.iter().map(|a| a.as_ref()).collect();
        out_cols.push(concat(&arrays).map_err(|e| AilakeError::Arrow(e.to_string()))?);
    }
    out_cols.push(Arc::new(StringArray::from(types)) as ArrayRef);
    out_cols.push(Arc::new(Int64Array::from(snap_ids)) as ArrayRef);
    out_cols.push(Arc::new(Int64Array::from(seqs)) as ArrayRef);
    out_cols.push(Arc::new(Int64Array::from(timestamps)) as ArrayRef);

    RecordBatch::try_new(out_schema, out_cols).map_err(|e| AilakeError::Arrow(e.to_string()))
}

/// Build the union of all fields across record row schemas, preserving the
/// order of the first occurrence of each field name. A field is made nullable
/// if any record is missing it, because equality-delete rows (PK-only) or
/// schema-drift inserts do not carry every column.
fn union_schema(records: &[ChangeRecord]) -> SchemaRef {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut fields: Vec<Field> = Vec::new();
    for rec in records {
        for f in rec.row.schema().fields() {
            if let Some(&idx) = seen.get(f.name()) {
                if f.is_nullable() && !fields[idx].is_nullable() {
                    fields[idx] = Field::new(f.name(), f.data_type().clone(), true);
                }
            } else {
                seen.insert(f.name().clone(), fields.len());
                fields.push((**f).clone());
            }
        }
    }
    // Mark any field not present in every record as nullable.
    let all_names: HashSet<String> = seen.keys().cloned().collect();
    for rec in records {
        let rec_names: HashSet<String> = rec
            .row
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        for missing in all_names.difference(&rec_names) {
            if let Some(&idx) = seen.get(missing) {
                if !fields[idx].is_nullable() {
                    fields[idx] =
                        Field::new(fields[idx].name(), fields[idx].data_type().clone(), true);
                }
            }
        }
    }
    Arc::new(Schema::new(fields))
}

/// Reorder/extend `rec.row` so it matches `schema`. Missing columns are filled
/// with nulls of the correct Arrow type; extra columns are dropped.
fn normalize_to_schema(rec: ChangeRecord, schema: &SchemaRef) -> AilakeResult<ChangeRecord> {
    if rec.row.schema() == *schema {
        return Ok(rec);
    }
    let n = rec.row.num_rows();
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if let Ok(idx) = rec.row.schema().index_of(field.name()) {
            cols.push(rec.row.column(idx).clone());
        } else {
            cols.push(null_array_for(field.data_type(), n));
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), cols)
        .map_err(|e| AilakeError::Arrow(e.to_string()))?;
    Ok(ChangeRecord {
        row: batch,
        change_type: rec.change_type,
        snapshot_id: rec.snapshot_id,
        sequence_number: rec.sequence_number,
        timestamp_ms: rec.timestamp_ms,
    })
}

fn null_array_for(data_type: &DataType, n: usize) -> ArrayRef {
    use arrow_array::*;
    match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(vec![None::<&str>; n])) as ArrayRef,
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![None::<&str>; n])) as ArrayRef,
        DataType::Int32 => Arc::new(Int32Array::from(vec![None::<i32>; n])) as ArrayRef,
        DataType::Int64 => Arc::new(Int64Array::from(vec![None::<i64>; n])) as ArrayRef,
        DataType::UInt32 => Arc::new(UInt32Array::from(vec![None::<u32>; n])) as ArrayRef,
        DataType::Float32 => Arc::new(Float32Array::from(vec![None::<f32>; n])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![None::<f64>; n])) as ArrayRef,
        DataType::Boolean => Arc::new(BooleanArray::from(vec![None::<bool>; n])) as ArrayRef,
        DataType::FixedSizeList(item, dim) => {
            let nulls = arrow_array::NullArray::new(n);
            let values = Arc::new(null_array_for(item.data_type(), n * *dim as usize));
            Arc::new(FixedSizeListArray::new(
                Arc::clone(item),
                *dim,
                values,
                nulls.nulls().cloned(),
            )) as ArrayRef
        }
        DataType::List(item) => {
            let offsets = arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![
                    0i32;
                    n + 1
                ]));
            let values = null_array_for(item.data_type(), 0);
            Arc::new(ListArray::new(Arc::clone(item), offsets, values, None)) as ArrayRef
        }
        DataType::LargeList(item) => {
            let offsets = arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![
                    0i64;
                    n + 1
                ]));
            let values = null_array_for(item.data_type(), 0);
            Arc::new(LargeListArray::new(Arc::clone(item), offsets, values, None)) as ArrayRef
        }
        _ => Arc::new(arrow_array::NullArray::new(n)) as ArrayRef,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn change_type_strings() {
        assert_eq!(ChangeType::Insert.as_str(), "insert");
        assert_eq!(ChangeType::Delete.as_str(), "delete");
        assert_eq!(ChangeType::UpdateBefore.as_str(), "update_before");
        assert_eq!(ChangeType::UpdateAfter.as_str(), "update_after");
    }
}
