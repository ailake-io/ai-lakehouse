// SPDX-License-Identifier: MIT OR Apache-2.0
// Iceberg Spec v2 Avro manifest writer and reader.
//
// Iceberg uses two layers of Avro files:
//   1. Manifest file   — lists data files (one per batch/commit)
//   2. Manifest list   — lists manifest files (one per snapshot)
//
// Both must be Avro with specific field IDs so PyIceberg, Spark, and Trino
// can read them without the AI-Lake plugin.
//
// AI-Lake specific metadata (centroid, radius, hnsw_offset, hnsw_len, etc.)
// is stored in the `key_metadata` bytes field of each manifest entry.
// Standard Iceberg readers treat this field as opaque encryption metadata
// and skip it; AI-Lake readers parse it as JSON to reconstruct DataFileEntry.

use apache_avro::types::Value;
use bytes::Bytes;

use crate::provider::{DataFileEntry, IndexStatus, PartitionField, PartitionSpec, SnapshotId};

// ---------------------------------------------------------------------------
// Schema constants — field IDs follow Iceberg Spec v2 §3.5
// ---------------------------------------------------------------------------

const MANIFEST_ENTRY_SCHEMA_STR: &str = r#"{
  "type": "record",
  "name": "manifest_entry",
  "fields": [
    {"name": "status",              "type": "int",  "field-id": 0},
    {"name": "snapshot_id",         "type": ["null", "long"],  "default": null, "field-id": 1},
    {"name": "sequence_number",     "type": ["null", "long"],  "default": null, "field-id": 3},
    {"name": "file_sequence_number","type": ["null", "long"],  "default": null, "field-id": 4},
    {"name": "data_file",           "type": {
      "type": "record", "name": "r2",
      "fields": [
        {"name": "content",           "type": "int",    "field-id": 134, "doc": "0=DATA"},
        {"name": "file_path",         "type": "string", "field-id": 100},
        {"name": "file_format",       "type": "string", "field-id": 101},
        {"name": "partition",         "type": {"type": "record", "name": "r102", "fields": []}, "field-id": 102},
        {"name": "record_count",      "type": "long",   "field-id": 103},
        {"name": "file_size_in_bytes","type": "long",   "field-id": 104},
        {"name": "column_sizes",      "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k117_v118","fields":[{"name":"key","type":"int","field-id":117},{"name":"value","type":"long","field-id":118}]},"element-id":119}], "default": null, "field-id": 108},
        {"name": "value_counts",      "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k119_v120","fields":[{"name":"key","type":"int","field-id":119},{"name":"value","type":"long","field-id":120}]},"element-id":121}], "default": null, "field-id": 109},
        {"name": "null_value_counts", "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k121_v122","fields":[{"name":"key","type":"int","field-id":121},{"name":"value","type":"long","field-id":122}]},"element-id":123}], "default": null, "field-id": 110},
        {"name": "nan_value_counts",  "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k138_v139","fields":[{"name":"key","type":"int","field-id":138},{"name":"value","type":"long","field-id":139}]},"element-id":140}], "default": null, "field-id": 137},
        {"name": "lower_bounds",      "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k126_v127","fields":[{"name":"key","type":"int","field-id":126},{"name":"value","type":"bytes","field-id":127}]},"element-id":128}], "default": null, "field-id": 125},
        {"name": "upper_bounds",      "type": ["null", {"type": "array", "logicalType": "map", "items": {"type":"record","name":"k129_v130","fields":[{"name":"key","type":"int","field-id":129},{"name":"value","type":"bytes","field-id":130}]},"element-id":131}], "default": null, "field-id": 128},
        {"name": "key_metadata",      "type": ["null", "bytes"], "default": null, "field-id": 131},
        {"name": "split_offsets",     "type": ["null", {"type": "array", "items": "long", "element-id": 133}], "default": null, "field-id": 132},
        {"name": "equality_ids",      "type": ["null", {"type": "array", "items": "int",  "element-id": 136}], "default": null, "field-id": 135},
        {"name": "sort_order_id",     "type": ["null", "int"],  "default": null, "field-id": 140},
        {"name": "first_row_id",      "type": ["null", "long"], "default": null, "field-id": 141}
      ]
    }, "field-id": 2}
  ]
}"#;

const MANIFEST_LIST_SCHEMA_STR: &str = r#"{
  "type": "record",
  "name": "manifest_file",
  "fields": [
    {"name": "manifest_path",              "type": "string", "field-id": 500},
    {"name": "manifest_length",            "type": "long",   "field-id": 501},
    {"name": "partition_spec_id",          "type": "int",    "field-id": 502},
    {"name": "content",                    "type": "int",    "field-id": 517, "doc": "0=DATA"},
    {"name": "sequence_number",            "type": "long",   "field-id": 515},
    {"name": "min_sequence_number",        "type": "long",   "field-id": 516},
    {"name": "added_snapshot_id",          "type": "long",   "field-id": 503},
    {"name": "added_data_files_count",     "type": "int",    "field-id": 504},
    {"name": "existing_data_files_count",  "type": "int",    "field-id": 505},
    {"name": "deleted_data_files_count",   "type": "int",    "field-id": 506},
    {"name": "added_rows_count",           "type": "long",   "field-id": 512},
    {"name": "existing_rows_count",        "type": "long",   "field-id": 513},
    {"name": "deleted_rows_count",         "type": "long",   "field-id": 514},
    {"name": "partitions", "type": {
      "type": "array",
      "items": {
        "type": "record", "name": "r508",
        "fields": [
          {"name": "contains_null", "type": "boolean", "field-id": 509},
          {"name": "contains_nan",  "type": ["null", "boolean"], "default": null, "field-id": 518},
          {"name": "lower_bound",   "type": ["null", "bytes"],   "default": null, "field-id": 510},
          {"name": "upper_bound",   "type": ["null", "bytes"],   "default": null, "field-id": 511}
        ]
      },
      "element-id": 508
    }, "field-id": 507}
  ]
}"#;

// ---------------------------------------------------------------------------
// Dynamic schema helpers (Phase I)
// ---------------------------------------------------------------------------

/// Build the Avro type string for the partition record (r102) based on the
/// active partition spec.  Returns `{"type":"record","name":"r102","fields":[]}`
/// (empty) when `spec` is None or unpartitioned.
fn build_partition_record_schema(spec: Option<&PartitionSpec>) -> String {
    let fields: Vec<String> = spec
        .map(|s| s.fields.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|f| {
            let avro_type = match f.source_type.as_str() {
                "int" | "integer" => "int",
                "long" => "long",
                _ => "string", // string, uuid, and unknown types → string
            };
            format!(
                r#"{{"name":"{name}","type":["null","{avro_type}"],"default":null,"field-id":{fid}}}"#,
                name = f.name,
                avro_type = avro_type,
                fid = f.field_id
            )
        })
        .collect();
    format!(
        r#"{{"type":"record","name":"r102","fields":[{}]}}"#,
        fields.join(",")
    )
}

/// Build the full manifest entry schema string, injecting the dynamic partition
/// record in place of the static empty r102.
pub fn build_manifest_entry_schema(spec: Option<&PartitionSpec>) -> String {
    let partition_record = build_partition_record_schema(spec);
    // Replace the hard-coded empty r102 placeholder in MANIFEST_ENTRY_SCHEMA_STR.
    MANIFEST_ENTRY_SCHEMA_STR.replace(
        r#"{"type": "record", "name": "r102", "fields": []}"#,
        &partition_record,
    )
}

/// Encode a partition value for one partition field into an Avro binary record.
/// The field type is `["null", T]` so encoding is: union-index (0=null, 1=value) + value.
fn encode_partition_value(field: &PartitionField, value: Option<&str>, buf: &mut Vec<u8>) {
    use crate::avro_raw::{encode_long, encode_string};
    match value {
        None => encode_long(0, buf), // union index 0 = null
        Some(v) => {
            encode_long(1, buf); // union index 1 = non-null
            match field.source_type.as_str() {
                "int" | "integer" => {
                    if let Ok(n) = v.parse::<i32>() {
                        encode_long(n as i64, buf); // Avro int = zigzag varint
                    } else {
                        // Fallback: re-encode as null (malformed value)
                        let last = buf.len() - 1;
                        buf[last] = 0x00; // overwrite union index back to 0
                    }
                }
                "long" => {
                    if let Ok(n) = v.parse::<i64>() {
                        encode_long(n, buf);
                    } else {
                        let last = buf.len() - 1;
                        buf[last] = 0x00;
                    }
                }
                _ => encode_string(v, buf), // string / uuid
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write an Iceberg Spec v2 manifest file (Avro) from a list of DataFileEntry.
/// Returns the raw bytes of the Avro file.
///
/// `partition_spec`: when `Some`, encodes partition values in the native Avro
/// `data_file.partition` record field so Spark/Trino can do partition pruning.
/// Encode one Iceberg `map<int, long>` manifest field (`column_sizes`/`value_counts`/
/// `null_value_counts`) as `["null", array<record{key:int, value:long}>]` — Avro has
/// no native non-string-keyed map type, so Iceberg's manifest schema (see
/// `MANIFEST_ENTRY_SCHEMA_STR`) encodes `map<int,V>` as an array of 2-field records;
/// this must match that shape exactly, mirroring `equality_ids`'s existing
/// union+array+block-count+terminator pattern one level deeper (record instead of
/// bare int per element). Emits `null` when `stats` is `None` or has no entries.
fn encode_int_long_map(
    stats: &Option<std::collections::BTreeMap<i32, crate::column_stats::FieldStats>>,
    get_value: impl Fn(&crate::column_stats::FieldStats) -> Option<i64>,
    rec: &mut Vec<u8>,
) {
    use crate::avro_raw::{encode_int, encode_long, encode_union_null};
    let entries: Vec<(i32, i64)> = stats
        .iter()
        .flatten()
        .filter_map(|(id, s)| get_value(s).map(|v| (*id, v)))
        .collect();
    if entries.is_empty() {
        encode_union_null(rec);
        return;
    }
    encode_long(1, rec); // union: non-null array
    encode_long(entries.len() as i64, rec); // block count
    for (id, v) in entries {
        encode_int(id, rec); // key
        encode_long(v, rec); // value
    }
    encode_long(0, rec); // array end marker
}

/// Same as `encode_int_long_map` but for `lower_bounds`/`upper_bounds` (`map<int,
/// bytes>`) — `get_bound` returns a base64 string (as stored in `FieldStats`), decoded
/// here; malformed base64 is skipped rather than failing the whole manifest write.
fn encode_int_bytes_map(
    stats: &Option<std::collections::BTreeMap<i32, crate::column_stats::FieldStats>>,
    get_bound: impl Fn(&crate::column_stats::FieldStats) -> Option<&str>,
    rec: &mut Vec<u8>,
) {
    use crate::avro_raw::{encode_bytes_field, encode_int, encode_long, encode_union_null};
    use base64::Engine;
    let entries: Vec<(i32, Vec<u8>)> = stats
        .iter()
        .flatten()
        .filter_map(|(id, s)| {
            let b64 = get_bound(s)?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
            Some((*id, bytes))
        })
        .collect();
    if entries.is_empty() {
        encode_union_null(rec);
        return;
    }
    encode_long(1, rec); // union: non-null array
    encode_long(entries.len() as i64, rec); // block count
    for (id, bytes) in entries {
        encode_int(id, rec); // key
        encode_bytes_field(&bytes, rec); // value
    }
    encode_long(0, rec); // array end marker
}

pub fn write_manifest_file(
    files: &[DataFileEntry],
    snapshot_id: SnapshotId,
    sequence_number: i64,
    table_schema_json: &str,
    partition_spec_json: &str,
    format_version: u8,
    partition_spec: Option<&PartitionSpec>,
) -> Bytes {
    use crate::avro_raw::{
        encode_empty_array, encode_int, encode_long, encode_string, encode_union_bytes,
        encode_union_long, encode_union_null, write_avro_container,
    };

    let mut records: Vec<Vec<u8>> = Vec::with_capacity(files.len());
    for f in files {
        // Decoded once per file; feeds the 5 stats fields below (`nan_value_counts`
        // stays null — Parquet doesn't track NaN counts natively, see column_stats.rs).
        let stats: Option<std::collections::BTreeMap<i32, crate::column_stats::FieldStats>> = f
            .column_stats
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());
        let mut rec = Vec::new();
        encode_int(1, &mut rec); // status=ADDED
        encode_union_long(1, snapshot_id, &mut rec); // snapshot_id
        encode_union_long(1, sequence_number, &mut rec); // sequence_number
        encode_union_long(1, sequence_number, &mut rec); // file_sequence_number
                                                         // data_file (nested record — no tag bytes in Avro binary)
        encode_int(0, &mut rec); // content=DATA
        encode_string(&f.path, &mut rec); // file_path
        encode_string("PARQUET", &mut rec); // file_format
                                            // partition r102: encode native partition values when spec is active;
                                            // empty record (0 bytes) for unpartitioned tables.
                                            // For multi-column specs, partition_value is \x1f-separated; each field
                                            // gets its own union slot in the Avro record.
        if let Some(spec) = partition_spec {
            if !spec.fields.is_empty() {
                let raw = f.partition_value.as_deref().unwrap_or("");
                let parts: Vec<&str> = raw.split('\x1f').collect();
                for (i, field) in spec.fields.iter().enumerate() {
                    let val = parts.get(i).copied().filter(|s| !s.is_empty());
                    encode_partition_value(field, val, &mut rec);
                }
            }
        }
        encode_long(f.record_count as i64, &mut rec); // record_count
        encode_long(f.file_size_bytes as i64, &mut rec); // file_size_in_bytes
        encode_int_long_map(&stats, |s| Some(s.column_size), &mut rec); // column_sizes
        encode_int_long_map(&stats, |s| Some(s.value_count), &mut rec); // value_counts
        encode_int_long_map(&stats, |s| Some(s.null_count), &mut rec); // null_value_counts
        encode_union_null(&mut rec); // nan_value_counts — Parquet stats don't track NaN counts
        encode_int_bytes_map(&stats, |s| s.lower_bound_b64.as_deref(), &mut rec); // lower_bounds
        encode_int_bytes_map(&stats, |s| s.upper_bound_b64.as_deref(), &mut rec); // upper_bounds
                                                                                  // serde_json silently encodes NaN/Infinity as JSON `null`, which decodes back to
                                                                                  // `None` on the next read — round-trips as data loss with no error. Drop it
                                                                                  // explicitly and log instead, so it's visible rather than silent.
        let radius = f.radius.filter(|r| {
            if r.is_finite() {
                true
            } else {
                tracing::warn!(
                    "dropping non-finite radius ({r}) for {}: cannot round-trip through JSON key_metadata",
                    f.path
                );
                false
            }
        });
        let ext = AilakeEntryExt {
            centroid_b64: f.centroid_b64.clone(),
            radius,
            hnsw_offset: f.hnsw_offset,
            hnsw_len: f.hnsw_len,
            vector_column: f.vector_column.clone(),
            vector_dim: f.vector_dim,
            extra_vector_indexes: f.extra_vector_indexes.clone(),
            index_status: f.index_status.clone(),
            index_error: f.index_error.clone(),
            batch_id: f.batch_id.clone(),
            embedding_model: f.embedding_model.clone(),
            partition_value: f.partition_value.clone(),
            deletion_vector: f.deletion_vector.clone(),
            first_row_id: f.first_row_id,
        };
        match serde_json::to_vec(&ext) {
            Ok(bytes) => encode_union_bytes(1, &bytes, &mut rec), // key_metadata=bytes
            Err(_) => encode_union_null(&mut rec),                // key_metadata=null
        }
        encode_union_null(&mut rec); // split_offsets
        encode_union_null(&mut rec); // equality_ids
        encode_union_null(&mut rec); // sort_order_id
        match f.first_row_id {
            Some(id) => encode_union_long(1, id, &mut rec),
            None => encode_union_null(&mut rec),
        } // first_row_id (V3 row lineage; null for V2 tables)
        let _ = encode_empty_array; // suppress unused warning
        records.push(rec);
    }

    let fv_str: &[u8] = if format_version >= 3 { b"3" } else { b"2" };
    let spec_id_str = partition_spec
        .map(|s| s.spec_id.to_string())
        .unwrap_or_else(|| "0".to_string());
    let extra_meta: &[(&str, &[u8])] = &[
        ("schema", table_schema_json.as_bytes()),
        ("partition-spec", partition_spec_json.as_bytes()),
        ("partition-spec-id", spec_id_str.as_bytes()),
        ("format-version", fv_str),
        ("content", b"data"),
    ];
    let schema_str = build_manifest_entry_schema(partition_spec);
    Bytes::from(write_avro_container(&schema_str, extra_meta, &records))
}

/// Write an Iceberg Spec v2 manifest list (Avro) pointing to one manifest file.
pub fn write_manifest_list(
    manifest_path: &str,
    manifest_bytes: usize,
    snapshot_id: SnapshotId,
    sequence_number: i64,
    added_rows: i64,
) -> Bytes {
    write_manifest_list_multi(
        &[(manifest_path.to_string(), manifest_bytes as i64)],
        snapshot_id,
        sequence_number,
        added_rows,
    )
}

/// Write an Iceberg Spec v2 manifest list (Avro) pointing to multiple manifest files.
/// `manifests` is a list of (manifest_path, manifest_length_bytes).
pub fn write_manifest_list_multi(
    manifests: &[(String, i64)],
    snapshot_id: SnapshotId,
    sequence_number: i64,
    added_rows: i64,
) -> Bytes {
    use crate::avro_raw::{
        encode_empty_array, encode_int, encode_long, encode_string, write_avro_container,
    };

    let n = manifests.len();
    let mut records: Vec<Vec<u8>> = Vec::with_capacity(n);
    for (i, (path, len)) in manifests.iter().enumerate() {
        let rows = if i + 1 == n { added_rows } else { 0i64 };
        let mut rec = Vec::new();
        encode_string(path, &mut rec); // manifest_path
        encode_long(*len, &mut rec); // manifest_length
        encode_int(0, &mut rec); // partition_spec_id
        encode_int(0, &mut rec); // content=DATA
        encode_long(sequence_number, &mut rec); // sequence_number
        encode_long(sequence_number, &mut rec); // min_sequence_number
        encode_long(snapshot_id, &mut rec); // added_snapshot_id
        encode_int(1, &mut rec); // added_data_files_count
        encode_int(0, &mut rec); // existing_data_files_count
        encode_int(0, &mut rec); // deleted_data_files_count
        encode_long(rows, &mut rec); // added_rows_count
        encode_long(0, &mut rec); // existing_rows_count
        encode_long(0, &mut rec); // deleted_rows_count
        encode_empty_array(&mut rec); // partitions (empty array)
        records.push(rec);
    }

    Bytes::from(write_avro_container(
        MANIFEST_LIST_SCHEMA_STR,
        &[],
        &records,
    ))
}

/// Read DataFileEntry records from an Iceberg manifest file (Avro).
/// AI-Lake metadata is recovered from the `key_metadata` bytes field (JSON-encoded).
pub fn read_manifest_file(data: &[u8]) -> apache_avro::AvroResult<Vec<DataFileEntry>> {
    let reader = apache_avro::Reader::new(data)?;
    let mut results = Vec::new();
    for value in reader {
        let value = value?;
        if let Value::Record(fields) = value {
            // Extract key_metadata bytes for AI-Lake extension fields
            let key_meta_bytes: Option<Vec<u8>> = fields
                .iter()
                .find(|(k, _)| k == "data_file")
                .and_then(|(_, v)| {
                    if let Value::Record(df_fields) = v {
                        df_fields
                            .iter()
                            .find(|(k, _)| k == "key_metadata")
                            .and_then(|(_, v)| match v {
                                Value::Union(_, inner) => {
                                    if let Value::Bytes(b) = inner.as_ref() {
                                        Some(b.clone())
                                    } else {
                                        None
                                    }
                                }
                                Value::Bytes(b) => Some(b.clone()),
                                _ => None,
                            })
                    } else {
                        None
                    }
                });

            let data_file = fields
                .iter()
                .find(|(k, _)| k == "data_file")
                .map(|(_, v)| v);
            if let Some(Value::Record(df_fields)) = data_file {
                let path = df_fields
                    .iter()
                    .find(|(k, _)| k == "file_path")
                    .and_then(|(_, v)| {
                        if let Value::String(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    });
                let record_count = df_fields
                    .iter()
                    .find(|(k, _)| k == "record_count")
                    .and_then(|(_, v)| {
                        if let Value::Long(n) = v {
                            Some(*n as u64)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let file_size_bytes = df_fields
                    .iter()
                    .find(|(k, _)| k == "file_size_in_bytes")
                    .and_then(|(_, v)| {
                        if let Value::Long(n) = v {
                            Some(*n as u64)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                if let Some(path) = path {
                    // Try to recover AI-Lake extension fields from key_metadata
                    let ext: Option<AilakeEntryExt> = key_meta_bytes
                        .as_deref()
                        .and_then(|b| serde_json::from_slice(b).ok());

                    // Parse V3 Deletion Vector from native Avro field (Spark/Trino/PyIceberg).
                    // AI-Lake-written DVs come from key_metadata JSON (Phase C).
                    let native_dv = parse_v3_deletion_vector(df_fields);

                    // Parse V3 first_row_id from native Avro field (Phase D).
                    // Falls back to key_metadata JSON for old AI-Lake manifests.
                    let native_first_row_id = parse_v3_first_row_id(df_fields);

                    // Phase I/K: read native partition values from data_file.partition record.
                    // Multi-column specs → join with \x1f separator; single-column → plain string.
                    // Newly-written manifests carry values here; old manifests fall back to key_metadata JSON.
                    let native_partition_value = df_fields
                        .iter()
                        .find(|(k, _)| k == "partition")
                        .and_then(|(_, v)| {
                            if let Value::Record(parts) = v {
                                if parts.is_empty() {
                                    return None;
                                }
                                let vals: Vec<String> = parts
                                    .iter()
                                    .filter_map(|(_, pv)| match pv {
                                        Value::Union(idx, inner) if *idx > 0 => {
                                            match inner.as_ref() {
                                                Value::String(s) => Some(s.clone()),
                                                Value::Int(n) => Some(n.to_string()),
                                                Value::Long(n) => Some(n.to_string()),
                                                _ => None,
                                            }
                                        }
                                        Value::String(s) => Some(s.clone()),
                                        Value::Int(n) => Some(n.to_string()),
                                        Value::Long(n) => Some(n.to_string()),
                                        _ => None,
                                    })
                                    .collect();
                                if vals.is_empty() {
                                    None
                                } else if vals.len() == 1 {
                                    Some(vals.into_iter().next().unwrap())
                                } else {
                                    Some(vals.join("\x1f"))
                                }
                            } else {
                                None
                            }
                        });

                    results.push(DataFileEntry {
                        path,
                        record_count,
                        file_size_bytes,
                        centroid_b64: ext.as_ref().and_then(|e| e.centroid_b64.clone()),
                        radius: ext.as_ref().and_then(|e| e.radius),
                        hnsw_offset: ext.as_ref().and_then(|e| e.hnsw_offset),
                        hnsw_len: ext.as_ref().and_then(|e| e.hnsw_len),
                        vector_column: ext.as_ref().and_then(|e| e.vector_column.clone()),
                        vector_dim: ext.as_ref().and_then(|e| e.vector_dim),
                        extra_vector_indexes: ext
                            .as_ref()
                            .map(|e| e.extra_vector_indexes.clone())
                            .unwrap_or_default(),
                        index_status: ext
                            .as_ref()
                            .map(|e| e.index_status.clone())
                            .unwrap_or_default(),
                        index_error: ext.as_ref().and_then(|e| e.index_error.clone()),
                        batch_id: ext.as_ref().and_then(|e| e.batch_id.clone()),
                        embedding_model: ext.as_ref().and_then(|e| e.embedding_model.clone()),
                        partition_value: native_partition_value
                            .or_else(|| ext.as_ref().and_then(|e| e.partition_value.clone())),
                        deletion_vector: ext
                            .as_ref()
                            .and_then(|e| e.deletion_vector.clone())
                            .or(native_dv),
                        first_row_id: native_first_row_id
                            .or_else(|| ext.as_ref().and_then(|e| e.first_row_id)),
                        // Write-only field (see `DataFileEntry::column_stats` doc):
                        // encoded as real native Avro fields by `write_manifest_file`,
                        // not read back — nothing downstream consumes it post-read.
                        column_stats: None,
                    });
                }
            }
        }
    }
    Ok(results)
}

/// AI-Lake extension fields encoded as JSON in the Avro `key_metadata` bytes field.
#[derive(serde::Serialize, serde::Deserialize)]
struct AilakeEntryExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw_len: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_dim: Option<u32>,
    #[serde(default)]
    pub extra_vector_indexes: Vec<crate::provider::ExtraVectorIndex>,
    #[serde(default)]
    pub index_status: IndexStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_value: Option<String>,
    /// V3 Deletion Vector written by AI-Lake (Phase C). For DVs written by
    /// external engines (Spark/Trino/PyIceberg), parse from native Avro field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletion_vector: Option<crate::provider::DeletionVector>,
    /// V3 Row Lineage (Phase D) — first_row_id as recorded in key_metadata JSON.
    /// Canonical value comes from the native Avro field; this is a fallback for
    /// manifests written by older AI-Lake versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
}

/// Extract a V3 Deletion Vector reference from the native Avro `data_file.deletion_vector`
/// field written by Spark, Trino, or PyIceberg. Returns None for V2 manifests or when
/// the field is absent / null.
fn parse_v3_deletion_vector(
    df_fields: &[(String, Value)],
) -> Option<crate::provider::DeletionVector> {
    let dv_val = df_fields.iter().find(|(k, _)| k == "deletion_vector")?;

    let dv_record = match &dv_val.1 {
        Value::Union(_, inner) => {
            if let Value::Record(fields) = inner.as_ref() {
                fields
            } else {
                return None;
            }
        }
        Value::Record(fields) => fields,
        _ => return None,
    };

    let get_str = |name: &str| {
        dv_record
            .iter()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
    };
    let get_long = |name: &str| {
        dv_record
            .iter()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| {
                if let Value::Long(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })
    };

    let path = get_str("path")?;
    let offset = get_long("offset")? as u64;
    let length = get_long("length")? as u64;
    let cardinality = get_long("cardinality").unwrap_or(-1);

    Some(crate::provider::DeletionVector {
        path,
        offset,
        length,
        cardinality,
    })
}

/// Extract V3 `first_row_id` from the native Avro `data_file.first_row_id` field.
/// Returns None for V2 manifests or when the field is absent / null.
fn parse_v3_first_row_id(df_fields: &[(String, Value)]) -> Option<i64> {
    df_fields
        .iter()
        .find(|(k, _)| k == "first_row_id")
        .and_then(|(_, v)| match v {
            Value::Union(_, inner) => {
                if let Value::Long(n) = inner.as_ref() {
                    Some(*n)
                } else {
                    None
                }
            }
            Value::Long(n) => Some(*n),
            _ => None,
        })
}

/// Read manifest file paths from an Iceberg manifest list (Avro).
pub fn read_manifest_list(data: &[u8]) -> apache_avro::AvroResult<Vec<String>> {
    Ok(read_manifest_list_typed(data)?
        .into_iter()
        .map(|(p, _)| p)
        .collect())
}

/// Read manifest file paths and content types from an Iceberg manifest list (Avro).
///
/// Returns `Vec<(path, content)>` where `content` follows Iceberg spec:
/// `0` = data manifest, `1` = delete manifest (contains position or equality deletes).
pub fn read_manifest_list_typed(data: &[u8]) -> apache_avro::AvroResult<Vec<(String, i32)>> {
    let reader = apache_avro::Reader::new(data)?;
    let mut results = Vec::new();
    for value in reader {
        let value = value?;
        if let Value::Record(fields) = value {
            let path = fields
                .iter()
                .find(|(k, _)| k == "manifest_path")
                .and_then(|(_, v)| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
            let content: i32 = fields
                .iter()
                .find(|(k, _)| k == "content")
                .and_then(|(_, v)| {
                    if let Value::Int(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            if let Some(p) = path {
                results.push((p, content));
            }
        }
    }
    Ok(results)
}

/// Write a manifest list that carries both data manifests (content=0) and delete manifests
/// (content=1). `manifests` is `(path, length_bytes, content_type)`.
pub fn write_manifest_list_multi_typed(
    manifests: &[(String, i64, i32)],
    snapshot_id: SnapshotId,
    sequence_number: i64,
    added_rows: i64,
) -> Bytes {
    use crate::avro_raw::{
        encode_empty_array, encode_int, encode_long, encode_string, write_avro_container,
    };

    let n = manifests.len();
    let mut records: Vec<Vec<u8>> = Vec::with_capacity(n);
    for (i, (path, len, content)) in manifests.iter().enumerate() {
        let rows = if i + 1 == n { added_rows } else { 0i64 };
        let mut rec = Vec::new();
        encode_string(path, &mut rec);
        encode_long(*len, &mut rec);
        encode_int(0, &mut rec); // partition_spec_id
        encode_int(*content, &mut rec); // content: 0=data, 1=delete
        encode_long(sequence_number, &mut rec);
        encode_long(sequence_number, &mut rec); // min_sequence_number
        encode_long(snapshot_id, &mut rec);
        encode_int(1, &mut rec); // added_data_files_count
        encode_int(0, &mut rec); // existing_data_files_count
        encode_int(0, &mut rec); // deleted_data_files_count
        encode_long(rows, &mut rec);
        encode_long(0, &mut rec);
        encode_long(0, &mut rec);
        encode_empty_array(&mut rec); // partitions
        records.push(rec);
    }

    Bytes::from(write_avro_container(
        MANIFEST_LIST_SCHEMA_STR,
        &[],
        &records,
    ))
}

/// Write an Iceberg equality delete manifest file (Avro) — `content=2` entries.
///
/// Each `EqualityDeleteFile` reference is written as a manifest entry whose
/// `data_file.content = 2` and `equality_ids` carries the matching field IDs.
pub fn write_equality_delete_manifest(
    deletes: &[crate::provider::EqualityDeleteFile],
    snapshot_id: SnapshotId,
    sequence_number: i64,
) -> Bytes {
    use crate::avro_raw::{
        encode_int, encode_long, encode_string, encode_union_long, encode_union_null,
        write_avro_container,
    };

    let mut records: Vec<Vec<u8>> = Vec::with_capacity(deletes.len());
    for d in deletes {
        let mut rec = Vec::new();
        encode_int(1, &mut rec); // status=ADDED
        encode_union_long(1, snapshot_id, &mut rec); // snapshot_id
        encode_union_long(1, sequence_number, &mut rec); // sequence_number
        encode_union_long(1, sequence_number, &mut rec); // file_sequence_number
                                                         // data_file record
        encode_int(2, &mut rec); // content=EQUALITY_DELETES
        encode_string(&d.path, &mut rec); // file_path
        encode_string("AVRO", &mut rec); // file_format
                                         // partition r102: empty record → 0 bytes
        encode_long(d.record_count as i64, &mut rec); // record_count
        encode_long(d.file_size_bytes as i64, &mut rec); // file_size_in_bytes
        encode_union_null(&mut rec); // column_sizes
        encode_union_null(&mut rec); // value_counts
        encode_union_null(&mut rec); // null_value_counts
        encode_union_null(&mut rec); // nan_value_counts
        encode_union_null(&mut rec); // lower_bounds
        encode_union_null(&mut rec); // upper_bounds
        encode_union_null(&mut rec); // key_metadata
        encode_union_null(&mut rec); // split_offsets
                                     // equality_ids: union index 1 (array) + zigzag-encoded array of ints
        encode_long(1, &mut rec); // union: non-null array
        encode_long(d.equality_ids.len() as i64, &mut rec); // block count
        for &id in &d.equality_ids {
            encode_int(id, &mut rec);
        }
        encode_long(0, &mut rec); // array end marker
        encode_union_null(&mut rec); // sort_order_id
        encode_union_null(&mut rec); // first_row_id
        records.push(rec);
    }

    let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
    let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
    let extra_meta: &[(&str, &[u8])] = &[
        ("schema", schema_json.as_bytes()),
        ("partition-spec", partition_spec.as_bytes()),
        ("partition-spec-id", b"0"),
        ("format-version", b"2"),
        ("content", b"deletes"),
    ];
    Bytes::from(write_avro_container(
        MANIFEST_ENTRY_SCHEMA_STR,
        extra_meta,
        &records,
    ))
}

/// Read equality delete file entries from an Iceberg delete manifest.
///
/// Extracts `path`, `equality_ids`, `record_count`, and `file_size_in_bytes`
/// from manifest entries where `data_file.content = 2`.
pub fn read_equality_delete_manifest(
    data: &[u8],
) -> apache_avro::AvroResult<Vec<crate::provider::EqualityDeleteFile>> {
    let reader = apache_avro::Reader::new(data)?;
    let mut results = Vec::new();
    for value in reader {
        let value = value?;
        if let Value::Record(fields) = value {
            let data_file = fields
                .iter()
                .find(|(k, _)| k == "data_file")
                .map(|(_, v)| v);
            if let Some(Value::Record(df_fields)) = data_file {
                let content: i32 = df_fields
                    .iter()
                    .find(|(k, _)| k == "content")
                    .and_then(|(_, v)| {
                        if let Value::Int(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if content != 2 {
                    continue;
                }
                let path = df_fields
                    .iter()
                    .find(|(k, _)| k == "file_path")
                    .and_then(|(_, v)| {
                        if let Value::String(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    });
                let record_count = df_fields
                    .iter()
                    .find(|(k, _)| k == "record_count")
                    .and_then(|(_, v)| {
                        if let Value::Long(n) = v {
                            Some(*n as u64)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let file_size_bytes = df_fields
                    .iter()
                    .find(|(k, _)| k == "file_size_in_bytes")
                    .and_then(|(_, v)| {
                        if let Value::Long(n) = v {
                            Some(*n as u64)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let equality_ids = df_fields
                    .iter()
                    .find(|(k, _)| k == "equality_ids")
                    .and_then(|(_, v)| match v {
                        Value::Union(_, inner) => {
                            if let Value::Array(arr) = inner.as_ref() {
                                Some(
                                    arr.iter()
                                        .filter_map(|item| {
                                            if let Value::Int(n) = item {
                                                Some(*n)
                                            } else {
                                                None
                                            }
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            }
                        }
                        Value::Array(arr) => Some(
                            arr.iter()
                                .filter_map(|item| {
                                    if let Value::Int(n) = item {
                                        Some(*n)
                                    } else {
                                        None
                                    }
                                })
                                .collect(),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default();
                if let Some(path) = path {
                    results.push(crate::provider::EqualityDeleteFile {
                        path,
                        equality_ids,
                        record_count,
                        file_size_bytes,
                        inline_values: None,
                    });
                }
            }
        }
    }
    Ok(results)
}

/// Write an equality delete Avro file containing the predicate rows.
///
/// Each value in `values` becomes one Avro record row `{col_name: value}`.
/// `iceberg_type` controls the Avro type: `"string"` / `"int"` / `"long"` /
/// `"float"` / `"double"`. Unrecognised types default to `"string"`.
/// The Avro schema embeds `field-id` so PyIceberg / Trino can resolve the column.
pub fn write_equality_delete_avro(
    col_name: &str,
    field_id: i32,
    iceberg_type: &str,
    values: &[&str],
) -> apache_avro::AvroResult<Bytes> {
    let avro_type = match iceberg_type.trim() {
        "int" | "integer" => "int",
        "long" => "long",
        "float" => "float",
        "double" => "double",
        _ => "string",
    };
    let schema_str = format!(
        r#"{{"type":"record","name":"eq_delete_entry","fields":[{{"name":"{col_name}","type":"{avro_type}","field-id":{field_id}}}]}}"#
    );
    let schema = apache_avro::Schema::parse_str(&schema_str)?;
    let mut writer = apache_avro::Writer::new(&schema, Vec::new());
    for val in values {
        use apache_avro::types::Value as AV;
        let avro_val = match avro_type {
            "int" => val
                .parse::<i32>()
                .map(AV::Int)
                .unwrap_or(AV::String(val.to_string())),
            "long" => val
                .parse::<i64>()
                .map(AV::Long)
                .unwrap_or(AV::String(val.to_string())),
            "float" => val
                .parse::<f32>()
                .map(AV::Float)
                .unwrap_or(AV::String(val.to_string())),
            "double" => val
                .parse::<f64>()
                .map(AV::Double)
                .unwrap_or(AV::String(val.to_string())),
            _ => AV::String(val.to_string()),
        };
        let record = AV::Record(vec![(col_name.to_string(), avro_val)]);
        writer.append(record)?;
    }
    Ok(Bytes::from(writer.into_inner()?))
}

/// Read (column_name, value_as_string) pairs from an equality delete Avro file.
///
/// Used by `EqualityDeleteFilter` to build the in-memory predicate set.
pub fn read_equality_delete_values(data: &[u8]) -> apache_avro::AvroResult<Vec<(String, String)>> {
    let reader = apache_avro::Reader::new(data)?;
    let mut results = Vec::new();
    for value in reader {
        let value = value?;
        if let Value::Record(fields) = value {
            for (col, val) in fields {
                let s = match &val {
                    Value::String(s) => s.clone(),
                    Value::Int(n) => n.to_string(),
                    Value::Long(n) => n.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Double(d) => d.to_string(),
                    Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
                    _ => continue,
                };
                results.push((col, s));
            }
        }
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Partition statistics Parquet file (Phase J)
// ---------------------------------------------------------------------------

/// Write an Iceberg partition statistics Parquet file.
///
/// Each row represents one distinct partition value with aggregate stats over
/// all data files carrying that value.  Schema:
///
/// ```text
/// message partition_statistics {
///   required group partition {
///     optional <type> <field_name>   -- one field per partition column
///   }
///   required int64 record_count
///   required int64 file_count
///   required int64 total_size_bytes
/// }
/// ```
///
/// Spark and Trino read this file via the `partition-statistics` entry in
/// `metadata.json` to optimise partition-level aggregations without scanning
/// data files.
pub fn write_partition_stats_parquet(
    partition_spec: &PartitionSpec,
    data_files: &[DataFileEntry],
) -> ailake_core::AilakeResult<bytes::Bytes> {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int64Array, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let part_field = match partition_spec.fields.first() {
        Some(f) => f,
        None => {
            return Err(ailake_core::AilakeError::Catalog(
                "write_partition_stats_parquet called on empty spec".into(),
            ))
        }
    };

    // Aggregate: partition_value → (record_count, file_count, total_size_bytes).
    let mut agg: HashMap<String, (i64, i64, i64)> = HashMap::new();
    for f in data_files {
        let key = f.partition_value.clone().unwrap_or_default();
        let e = agg.entry(key).or_insert((0, 0, 0));
        e.0 += f.record_count as i64;
        e.1 += 1;
        e.2 += f.file_size_bytes as i64;
    }

    let n = agg.len();
    let mut part_vals: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut record_counts: Vec<i64> = Vec::with_capacity(n);
    let mut file_counts: Vec<i64> = Vec::with_capacity(n);
    let mut total_sizes: Vec<i64> = Vec::with_capacity(n);

    // Sort by partition value for deterministic output (Spark expects sorted for pruning).
    let mut sorted: Vec<(&String, &(i64, i64, i64))> = agg.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());

    for (key, (rc, fc, ts)) in &sorted {
        // Empty string = null partition (files with no partition value).
        part_vals.push(if key.is_empty() {
            None
        } else {
            Some(key.as_str())
        });
        record_counts.push(*rc);
        file_counts.push(*fc);
        total_sizes.push(*ts);
    }

    // Build the `partition` struct column.  AI-Lake only supports single-field
    // identity partitioning in Phase I, so this is always one field.
    let part_col_field = Field::new(&part_field.name, DataType::Utf8, true);
    let part_struct_field = Field::new(
        "partition",
        DataType::Struct(Fields::from(vec![part_col_field.clone()])),
        false,
    );
    let part_string_arr = Arc::new(StringArray::from(part_vals)) as Arc<dyn arrow_array::Array>;
    let partition_arr = Arc::new(StructArray::new(
        Fields::from(vec![part_col_field]),
        vec![part_string_arr],
        None,
    )) as Arc<dyn arrow_array::Array>;

    let schema = Arc::new(Schema::new(vec![
        part_struct_field,
        Field::new("record_count", DataType::Int64, false),
        Field::new("file_count", DataType::Int64, false),
        Field::new("total_size_bytes", DataType::Int64, false),
    ]));

    let batch = arrow_array::RecordBatch::try_new(
        schema.clone(),
        vec![
            partition_arr,
            Arc::new(Int64Array::from(record_counts)),
            Arc::new(Int64Array::from(file_counts)),
            Arc::new(Int64Array::from(total_sizes)),
        ],
    )
    .map_err(|e| ailake_core::AilakeError::Catalog(e.to_string()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut buf: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))
        .map_err(|e| ailake_core::AilakeError::Catalog(e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| ailake_core::AilakeError::Catalog(e.to_string()))?;
    writer
        .close()
        .map_err(|e| ailake_core::AilakeError::Catalog(e.to_string()))?;

    Ok(bytes::Bytes::from(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{DataFileEntry, IndexStatus};

    #[test]
    fn manifest_list_roundtrip() {
        let bytes = write_manifest_list("warehouse/ns.db/t/metadata/m0.avro", 512, 42, 1, 10);
        let paths = read_manifest_list(&bytes).expect("read_manifest_list failed");
        assert_eq!(paths, vec!["warehouse/ns.db/t/metadata/m0.avro"]);
    }

    #[test]
    fn manifest_file_roundtrip() {
        let file = DataFileEntry {
            path: "data/part-0.parquet".to_string(),
            record_count: 5,
            file_size_bytes: 1024,
            centroid_b64: None,
            radius: Some(0.1),
            hnsw_offset: Some(100),
            hnsw_len: Some(50),
            vector_column: Some("emb".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 99, 1, schema_json, partition_spec, 2, None);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "data/part-0.parquet");
        assert_eq!(entries[0].record_count, 5);
        assert_eq!(entries[0].hnsw_offset, Some(100));
    }

    #[test]
    fn batch_id_roundtrip() {
        let file = DataFileEntry {
            path: "data/part-1.parquet".to_string(),
            record_count: 100,
            file_size_bytes: 4096,
            centroid_b64: None,
            radius: None,
            hnsw_offset: Some(200),
            hnsw_len: Some(80),
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: Some("dag_run_2026-05-28_taskA".to_string()),
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 42, 1, schema_json, partition_spec, 2, None);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(
            entries[0].batch_id.as_deref(),
            Some("dag_run_2026-05-28_taskA")
        );
    }

    #[test]
    fn first_row_id_roundtrip_v3() {
        let file = DataFileEntry {
            path: "data/part-2.parquet".to_string(),
            record_count: 200,
            file_size_bytes: 8192,
            centroid_b64: None,
            radius: None,
            hnsw_offset: Some(300),
            hnsw_len: Some(100),
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: Some(5000),
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 77, 1, schema_json, partition_spec, 3, None);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(entries[0].first_row_id, Some(5000));
    }

    #[test]
    fn first_row_id_none_for_v2() {
        let file = DataFileEntry {
            path: "data/part-3.parquet".to_string(),
            record_count: 100,
            file_size_bytes: 4096,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: None,
            vector_dim: None,
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 88, 1, schema_json, partition_spec, 2, None);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(entries[0].first_row_id, None);
    }

    /// Verifies the hand-written `encode_int_long_map`/`encode_int_bytes_map` binary
    /// matches `MANIFEST_ENTRY_SCHEMA_STR`'s declared `array<record{key,value}>` shape
    /// for `map<int,V>` exactly — bypasses `read_manifest_file` (which never decodes
    /// these fields back into `DataFileEntry`, see its `column_stats` doc) and instead
    /// uses `apache_avro::Reader`'s real schema-driven decoder directly, the same
    /// library Spark/Trino/PyIceberg use to read this file.
    #[test]
    fn column_stats_roundtrip_via_real_avro_reader() {
        use crate::column_stats::FieldStats;
        use base64::Engine;
        use std::collections::BTreeMap;

        let mut stats: BTreeMap<i32, FieldStats> = BTreeMap::new();
        stats.insert(
            1,
            FieldStats {
                value_count: 10,
                null_count: 2,
                column_size: 512,
                lower_bound_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode(5i32.to_le_bytes()),
                ),
                upper_bound_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode(50i32.to_le_bytes()),
                ),
            },
        );

        let file = DataFileEntry {
            path: "data/part-9.parquet".to_string(),
            record_count: 10,
            file_size_bytes: 4096,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: None,
            vector_dim: None,
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: Some(serde_json::to_string(&stats).unwrap()),
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 99, 1, schema_json, partition_spec, 2, None);

        let reader = apache_avro::Reader::new(bytes.as_ref()).expect("valid avro container");
        let value = reader
            .into_iter()
            .next()
            .unwrap()
            .expect("decodable record");
        let Value::Record(fields) = value else {
            panic!("expected top-level record")
        };
        let data_file = fields
            .iter()
            .find(|(k, _)| k == "data_file")
            .map(|(_, v)| v)
            .expect("data_file field");
        let Value::Record(df_fields) = data_file else {
            panic!("expected data_file record")
        };
        let get = |name: &str| {
            df_fields
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("field {name} missing"))
        };

        // Every populated map field decodes as union(non-null) -> array -> one
        // {key,value} record whose key matches the Iceberg field id (1, from the JSON
        // map's key) and whose value round-trips exactly.
        let assert_long_map_entry = |field: &str, expected: i64| match get(field) {
            Value::Union(_, inner) => match inner.as_ref() {
                Value::Array(items) => {
                    assert_eq!(items.len(), 1, "{field}: expected exactly one entry");
                    let Value::Record(kv) = &items[0] else {
                        panic!("{field}: expected record item")
                    };
                    assert_eq!(kv[0], ("key".to_string(), Value::Int(1)));
                    assert_eq!(kv[1], ("value".to_string(), Value::Long(expected)));
                }
                other => panic!("{field}: expected array union payload, got {other:?}"),
            },
            other => panic!("{field}: expected non-null union, got {other:?}"),
        };
        assert_long_map_entry("value_counts", 10);
        assert_long_map_entry("null_value_counts", 2);
        assert_long_map_entry("column_sizes", 512);

        let assert_bytes_map_entry = |field: &str, expected_i32: i32| match get(field) {
            Value::Union(_, inner) => match inner.as_ref() {
                Value::Array(items) => {
                    assert_eq!(items.len(), 1, "{field}: expected exactly one entry");
                    let Value::Record(kv) = &items[0] else {
                        panic!("{field}: expected record item")
                    };
                    assert_eq!(kv[0], ("key".to_string(), Value::Int(1)));
                    let Value::Bytes(b) = &kv[1].1 else {
                        panic!("{field}: expected bytes value")
                    };
                    assert_eq!(
                        i32::from_le_bytes(b.as_slice().try_into().unwrap()),
                        expected_i32
                    );
                }
                other => panic!("{field}: expected array union payload, got {other:?}"),
            },
            other => panic!("{field}: expected non-null union, got {other:?}"),
        };
        assert_bytes_map_entry("lower_bounds", 5);
        assert_bytes_map_entry("upper_bounds", 50);

        // nan_value_counts is deliberately always null (see column_stats.rs doc).
        match get("nan_value_counts") {
            Value::Union(_, inner) => assert_eq!(inner.as_ref(), &Value::Null),
            other => panic!("expected union, got {other:?}"),
        }
    }

    #[test]
    fn equality_delete_manifest_roundtrip() {
        use crate::provider::EqualityDeleteFile;
        let del = EqualityDeleteFile {
            path: "metadata/eq-del-001.avro".to_string(),
            equality_ids: vec![5, 9],
            record_count: 3,
            file_size_bytes: 512,
            inline_values: None,
        };
        let bytes = write_equality_delete_manifest(&[del], 42, 1);
        let entries =
            read_equality_delete_manifest(&bytes).expect("read_equality_delete_manifest failed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "metadata/eq-del-001.avro");
        assert_eq!(entries[0].equality_ids, vec![5, 9]);
        assert_eq!(entries[0].record_count, 3);
        assert_eq!(entries[0].file_size_bytes, 512);
    }

    #[test]
    fn read_manifest_list_typed_returns_content() {
        let manifests = vec![
            ("metadata/m0.avro".to_string(), 1024i64, 0i32),
            ("metadata/m0-eq-del.avro".to_string(), 256i64, 1i32),
        ];
        let bytes = write_manifest_list_multi_typed(&manifests, 99, 1, 10);
        let typed = read_manifest_list_typed(&bytes).expect("read_manifest_list_typed failed");
        assert_eq!(typed.len(), 2);
        assert_eq!(typed[0].0, "metadata/m0.avro");
        assert_eq!(typed[0].1, 0); // data
        assert_eq!(typed[1].0, "metadata/m0-eq-del.avro");
        assert_eq!(typed[1].1, 1); // delete
    }

    #[test]
    fn partition_spec_native_roundtrip() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "agent_id".to_string(),
                transform: "identity".to_string(),
                source_type: "string".to_string(),
            }],
        };
        let file = DataFileEntry {
            path: "data/part-agent-a.parquet".to_string(),
            record_count: 50,
            file_size_bytes: 2048,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: Some("agent-abc-123".to_string()),
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[{"id":1,"name":"agent_id","required":false,"type":"string"}]}"#;
        let partition_spec_json = r#"[{"spec-id":0,"fields":[]},{"spec-id":1,"fields":[{"name":"agent_id","transform":"identity","source-id":1,"field-id":1000}]}]"#;
        let bytes = write_manifest_file(
            &[file],
            200,
            1,
            schema_json,
            partition_spec_json,
            2,
            Some(&spec),
        );
        let entries = read_manifest_file(&bytes).expect("partition roundtrip failed");
        assert_eq!(entries.len(), 1);
        // Native partition value must be read back correctly.
        assert_eq!(entries[0].partition_value.as_deref(), Some("agent-abc-123"));
    }

    #[test]
    fn partition_spec_int_native_roundtrip() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "shard_id".to_string(),
                transform: "identity".to_string(),
                source_type: "int".to_string(),
            }],
        };
        let file = DataFileEntry {
            path: "data/part-shard-7.parquet".to_string(),
            record_count: 10,
            file_size_bytes: 512,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: None,
            vector_dim: None,
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: Some("7".to_string()),
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[{"id":1,"name":"shard_id","required":false,"type":"int"}]}"#;
        let partition_spec_json = r#"[{"spec-id":1,"fields":[{"name":"shard_id","transform":"identity","source-id":1,"field-id":1000}]}]"#;
        let bytes = write_manifest_file(
            &[file],
            201,
            1,
            schema_json,
            partition_spec_json,
            2,
            Some(&spec),
        );
        let entries = read_manifest_file(&bytes).expect("int partition roundtrip failed");
        assert_eq!(entries[0].partition_value.as_deref(), Some("7"));
    }

    #[test]
    fn multi_column_partition_roundtrip() {
        // Phase K: two-column spec (agent_id identity string + ts truncate[4] string).
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![
                PartitionField {
                    source_id: 1,
                    field_id: 1000,
                    name: "agent_id".to_string(),
                    transform: "identity".to_string(),
                    source_type: "string".to_string(),
                },
                PartitionField {
                    source_id: 2,
                    field_id: 1001,
                    name: "ts".to_string(),
                    transform: "truncate[4]".to_string(),
                    source_type: "string".to_string(),
                },
            ],
        };
        let compound = "agt-007\x1f2025"; // agent_id=agt-007, ts (truncated to 4)="2025"
        let file = DataFileEntry {
            path: "data/part-multi.parquet".to_string(),
            record_count: 30,
            file_size_bytes: 1024,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: Some(compound.to_string()),
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[{"id":1,"name":"agent_id","required":false,"type":"string"},{"id":2,"name":"ts","required":false,"type":"string"}]}"#;
        let partition_spec_json = r#"[{"spec-id":0,"fields":[]},{"spec-id":1,"fields":[{"name":"agent_id","transform":"identity","source-id":1,"field-id":1000},{"name":"ts","transform":"truncate[4]","source-id":2,"field-id":1001}]}]"#;
        let bytes = write_manifest_file(
            &[file],
            202,
            1,
            schema_json,
            partition_spec_json,
            2,
            Some(&spec),
        );
        let entries = read_manifest_file(&bytes).expect("multi-column roundtrip failed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].partition_value.as_deref(), Some(compound));
    }

    #[test]
    fn build_manifest_entry_schema_no_spec() {
        let s = build_manifest_entry_schema(None);
        // Empty partition record must be present (no partition fields).
        assert!(s.contains("r102"));
        assert!(s.contains(r#""fields":[]"#));
        assert!(!s.contains("tenant_id")); // no partition fields injected
    }

    #[test]
    fn build_manifest_entry_schema_with_string_spec() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "tenant_id".to_string(),
                transform: "identity".to_string(),
                source_type: "string".to_string(),
            }],
        };
        let s = build_manifest_entry_schema(Some(&spec));
        assert!(s.contains("tenant_id"));
        assert!(s.contains("1000")); // field-id
        assert!(s.contains(r#""null","string""#));
    }

    // ---------- Phase J: partition statistics Parquet ----------

    fn make_file(partition_value: Option<&str>, record_count: u64, size: u64) -> DataFileEntry {
        DataFileEntry {
            path: "data/part.parquet".to_string(),
            record_count,
            file_size_bytes: size,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: None,
            vector_dim: None,
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: partition_value.map(String::from),
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        }
    }

    #[test]
    fn write_partition_stats_parquet_basic() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "agent_id".to_string(),
                transform: "identity".to_string(),
                source_type: "string".to_string(),
            }],
        };
        let files = vec![
            make_file(Some("agent-A"), 100, 4096),
            make_file(Some("agent-A"), 200, 8192),
            make_file(Some("agent-B"), 50, 2048),
        ];
        let bytes = write_partition_stats_parquet(&spec, &files).expect("should not fail");
        assert!(!bytes.is_empty(), "output must be non-empty");

        // Read back with parquet crate and verify row count.
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let reader =
            SerializedFileReader::new(bytes::Bytes::from(bytes.to_vec())).expect("valid parquet");
        let row_count: usize = reader.get_row_iter(None).expect("iter").count();
        // 2 distinct partition values → 2 rows
        assert_eq!(row_count, 2, "expected one row per partition value");
    }

    #[test]
    fn write_partition_stats_parquet_empty_files() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "tenant_id".to_string(),
                transform: "identity".to_string(),
                source_type: "string".to_string(),
            }],
        };
        // No data files → zero rows in stats file (still valid Parquet).
        let bytes = write_partition_stats_parquet(&spec, &[]).expect("should not fail");
        assert!(!bytes.is_empty());

        use parquet::file::reader::{FileReader, SerializedFileReader};
        let reader =
            SerializedFileReader::new(bytes::Bytes::from(bytes.to_vec())).expect("valid parquet");
        let row_count = reader.get_row_iter(None).expect("iter").count();
        assert_eq!(row_count, 0);
    }

    #[test]
    fn write_partition_stats_parquet_aggregates_correctly() {
        use crate::provider::{PartitionField, PartitionSpec};
        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                name: "region".to_string(),
                transform: "identity".to_string(),
                source_type: "string".to_string(),
            }],
        };
        let files = vec![
            make_file(Some("us-east"), 1000, 10000),
            make_file(Some("us-east"), 2000, 20000),
            make_file(Some("eu-west"), 500, 5000),
        ];
        let bytes = write_partition_stats_parquet(&spec, &files).expect("aggregation ok");

        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::RowAccessor;
        let reader =
            SerializedFileReader::new(bytes::Bytes::from(bytes.to_vec())).expect("valid parquet");

        let mut rc_us = 0i64;
        let mut rc_eu = 0i64;
        for row in reader.get_row_iter(None).expect("iter") {
            let row = row.expect("row");
            // Column 1 = record_count (int64)
            let rc = row.get_long(1).expect("record_count");
            if rc == 3000 {
                rc_us = rc; // us-east: 1000+2000
            } else if rc == 500 {
                rc_eu = rc; // eu-west
            }
        }
        assert_eq!(rc_us, 3000, "us-east record_count should aggregate to 3000");
        assert_eq!(rc_eu, 500, "eu-west record_count should be 500");
    }

    #[test]
    fn index_failed_roundtrip() {
        let file = DataFileEntry {
            path: "data/part-failed.parquet".to_string(),
            record_count: 10,
            file_size_bytes: 1024,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Failed,
            index_error: Some("k-means did not converge".to_string()),
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 55, 1, schema_json, partition_spec, 2, None);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(entries[0].index_status, IndexStatus::Failed);
        assert_eq!(
            entries[0].index_error.as_deref(),
            Some("k-means did not converge")
        );
    }

    // ── Proptest: Avro manifest round-trip fuzzing ───────────────────

    // The proptest! macro must be at the top level of the cfg(test) module
    // due to macro hygiene. Tests are defined outside fuzz_tests but share
    // the same strategies.
    mod fuzz_utils {
        use crate::provider::DataFileEntry;

        pub const AVRO_SCHEMA: &str = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        pub const AVRO_PARTITION: &str = r#"[{"spec-id":0,"fields":[]}]"#;

        pub fn assert_entry_eq(a: &DataFileEntry, b: &DataFileEntry) {
            assert_eq!(a.path, b.path);
            assert_eq!(a.record_count, b.record_count);
            assert_eq!(a.file_size_bytes, b.file_size_bytes);
            assert_eq!(a.centroid_b64, b.centroid_b64);
            // Non-finite radius can't round-trip through JSON key_metadata (serde_json
            // encodes it as `null`) — write_manifest_file drops it explicitly instead of
            // letting that happen silently, so it comes back as `None`, not the original.
            match a.radius {
                Some(r) if !r.is_finite() => assert_eq!(b.radius, None),
                _ => assert_eq!(a.radius, b.radius),
            }
            assert_eq!(a.hnsw_offset, b.hnsw_offset);
            assert_eq!(a.hnsw_len, b.hnsw_len);
            assert_eq!(a.vector_column, b.vector_column);
            assert_eq!(a.vector_dim, b.vector_dim);
            assert_eq!(a.extra_vector_indexes.len(), b.extra_vector_indexes.len());
            for (i, (ea, eb)) in a
                .extra_vector_indexes
                .iter()
                .zip(b.extra_vector_indexes.iter())
                .enumerate()
            {
                assert_eq!(ea.column, eb.column, "extra_index[{i}].column");
                assert_eq!(ea.dim, eb.dim, "extra_index[{i}].dim");
                assert_eq!(
                    ea.hnsw_offset, eb.hnsw_offset,
                    "extra_index[{i}].hnsw_offset"
                );
                assert_eq!(ea.hnsw_len, eb.hnsw_len, "extra_index[{i}].hnsw_len");
            }
            assert_eq!(a.index_status, b.index_status);
            assert_eq!(a.index_error, b.index_error);
            assert_eq!(a.batch_id, b.batch_id);
            assert_eq!(a.embedding_model, b.embedding_model);
            assert_eq!(a.partition_value, b.partition_value);
            assert_eq!(
                a.deletion_vector.as_ref().map(|d| &d.path),
                b.deletion_vector.as_ref().map(|d| &d.path),
            );
            assert_eq!(
                a.deletion_vector.as_ref().map(|d| d.offset),
                b.deletion_vector.as_ref().map(|d| d.offset),
            );
            assert_eq!(a.first_row_id, b.first_row_id);
            assert!(
                b.column_stats.is_none(),
                "column_stats must be None after Avro round-trip"
            );
        }
    }

    mod fuzz_strategies {
        use crate::provider::{
            DataFileEntry, DeletionVector, EqualityDeleteFile, ExtraVectorIndex, IndexStatus,
        };
        use proptest::prelude::*;

        pub fn arb_dv() -> impl Strategy<Value = Option<DeletionVector>> {
            proptest::option::of(
                ("[a-z]{4,20}/dv.bin", 0u64..10_000, 1u64..4096, 1i64..1000).prop_map(
                    |(path, offset, length, cardinality)| DeletionVector {
                        path,
                        offset,
                        length,
                        cardinality,
                    },
                ),
            )
        }

        pub fn arb_extra() -> impl Strategy<Value = Vec<ExtraVectorIndex>> {
            proptest::collection::vec(
                ("[a-z_]{3,12}", 2u32..8, 0u64..1000, 0u64..500).prop_map(
                    |(column, dim, hnsw_offset, hnsw_len)| ExtraVectorIndex {
                        column,
                        dim,
                        hnsw_offset,
                        hnsw_len,
                        centroid_b64: None,
                        radius: None,
                    },
                ),
                0..3,
            )
        }

        pub fn arb_entry() -> impl Strategy<Value = DataFileEntry> {
            let group1 = (
                "[a-zA-Z0-9_/.-]{5,40}\\.parquet",
                0u64..10_000,
                0u64..1_000_000,
                proptest::option::of("[A-Za-z0-9+/=]{10,100}"),
                // Includes NaN/Infinity deliberately — assert_entry_eq documents and
                // exercises the intentional non-finite-radius-drops-to-None behavior.
                proptest::option::of(proptest::num::f32::ANY),
                proptest::option::of(0u64..10_000_000),
                proptest::option::of(0u64..5_000_000),
                proptest::option::of("[a-zA-Z_][a-zA-Z0-9_]{2,15}"),
            );
            let group2 = (
                proptest::option::of(2u32..4096),
                arb_extra(),
                proptest::option::of("[a-zA-Z0-9_-]{1,30}"),
                proptest::option::of("[a-zA-Z0-9_.-]{2,30}"),
                proptest::option::of("\\w{1,20}"),
                proptest::option::of(proptest::num::i64::ANY),
                arb_dv(),
            );
            (group1, group2).prop_map(
                |(
                    (
                        path,
                        record_count,
                        file_size_bytes,
                        centroid_b64,
                        radius,
                        hnsw_offset,
                        hnsw_len,
                        vector_column,
                    ),
                    (
                        vector_dim,
                        extra_vector_indexes,
                        batch_id,
                        embedding_model,
                        partition_value,
                        first_row_id,
                        deletion_vector,
                    ),
                )| {
                    DataFileEntry {
                        path,
                        record_count,
                        file_size_bytes,
                        centroid_b64,
                        radius,
                        hnsw_offset,
                        hnsw_len,
                        vector_column,
                        vector_dim,
                        extra_vector_indexes,
                        index_status: IndexStatus::Ready,
                        index_error: None,
                        batch_id,
                        embedding_model,
                        partition_value,
                        deletion_vector,
                        first_row_id,
                        column_stats: None,
                    }
                },
            )
        }

        pub fn arb_eq_del() -> impl Strategy<Value = EqualityDeleteFile> {
            (
                "[a-zA-Z0-9_/.-]{5,40}\\.avro",
                0u64..10_000,
                0u64..1_000_000,
                proptest::collection::vec(proptest::num::i32::ANY, 0..5),
            )
                .prop_map(|(path, record_count, file_size_bytes, equality_ids)| {
                    EqualityDeleteFile {
                        path,
                        equality_ids,
                        record_count,
                        file_size_bytes,
                        inline_values: None,
                    }
                })
        }
    }

    use fuzz_strategies::*;
    use fuzz_utils::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_manifest_file_roundtrip(
            entry in arb_entry(),
        ) {
            let bytes = crate::avro_manifest::write_manifest_file(
                std::slice::from_ref(&entry), 99, 1,
                AVRO_SCHEMA, AVRO_PARTITION, 2, None,
            );
            let entries = crate::avro_manifest::read_manifest_file(&bytes)
                .expect("read_manifest_file should succeed");
            assert_eq!(entries.len(), 1, "should read back 1 entry");
            assert_entry_eq(&entry, &entries[0]);
        }

        #[test]
        fn prop_manifest_file_multi_entry_roundtrip(
            entries in proptest::collection::vec(arb_entry(), 0..5),
        ) {
            let count = entries.len();
            let bytes = crate::avro_manifest::write_manifest_file(
                &entries, 99, 1,
                AVRO_SCHEMA, AVRO_PARTITION, 2, None,
            );
            let decoded = crate::avro_manifest::read_manifest_file(&bytes)
                .expect("read_manifest_file should succeed");
            assert_eq!(
                decoded.len(), count,
                "entry count mismatch: {count} written, {} read", decoded.len()
            );
            for (orig, dec) in entries.iter().zip(decoded.iter()) {
                assert_entry_eq(orig, dec);
            }
        }

        #[test]
        fn prop_manifest_list_roundtrip(
            path in "[a-zA-Z0-9_/.-]{5,60}\\.avro",
            manifest_len in 100u64..1_000_000,
            snapshot_id in proptest::num::i64::ANY,
        ) {
            let bytes = crate::avro_manifest::write_manifest_list(&path, manifest_len as usize, snapshot_id, 1, 10);
            let paths = crate::avro_manifest::read_manifest_list(&bytes)
                .expect("read_manifest_list should succeed");
            assert_eq!(paths.len(), 1, "manifest list should have 1 entry");
            assert!(
                paths[0].contains(".avro"),
                "manifest list path should be .avro, got: {}", paths[0]
            );
        }

        #[test]
        fn prop_manifest_list_multi_typed_roundtrip(
            manifests in proptest::collection::vec(
                ("[a-zA-Z0-9_/.-]{5,40}\\.avro", 0i64..1_000_000, 0i32..2),
                0..4,
            ),
            snapshot_id in proptest::num::i64::ANY,
        ) {
            let bytes = crate::avro_manifest::write_manifest_list_multi_typed(&manifests, snapshot_id, 1, 10);
            let decoded = crate::avro_manifest::read_manifest_list_typed(&bytes)
                .expect("read_manifest_list_typed should succeed");
            assert_eq!(
                decoded.len(), manifests.len(),
                "multi-typed manifest list entry count mismatch"
            );
        }

        #[test]
        fn prop_equality_delete_manifest_roundtrip(
            del in arb_eq_del(),
            snapshot_id in proptest::num::i64::ANY,
        ) {
            let bytes = crate::avro_manifest::write_equality_delete_manifest(std::slice::from_ref(&del), snapshot_id, 1);
            let decoded = crate::avro_manifest::read_equality_delete_manifest(&bytes)
                .expect("read_equality_delete_manifest should succeed");
            assert_eq!(decoded.len(), 1, "equality delete manifest should have 1 entry");
            assert_eq!(decoded[0].path, del.path, "equality delete path mismatch");
            assert_eq!(decoded[0].record_count, del.record_count, "equality delete record_count mismatch");
            assert_eq!(decoded[0].file_size_bytes, del.file_size_bytes, "equality delete file_size_bytes mismatch");
        }
    }

    #[test]
    fn prop_empty_manifest_roundtrip() {
        let entries =
            crate::avro_manifest::read_manifest_file(&crate::avro_manifest::write_manifest_file(
                &[],
                99,
                1,
                AVRO_SCHEMA,
                AVRO_PARTITION,
                2,
                None,
            ))
            .expect("empty manifest should decode successfully");
        assert!(entries.is_empty(), "empty manifest should yield 0 entries");
    }
}
