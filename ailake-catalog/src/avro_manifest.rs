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

use crate::provider::{DataFileEntry, IndexStatus, SnapshotId};

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
        {"name": "sort_order_id",     "type": ["null", "int"],  "default": null, "field-id": 140}
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
// Public API
// ---------------------------------------------------------------------------

/// Write an Iceberg Spec v2 manifest file (Avro) from a list of DataFileEntry.
/// Returns the raw bytes of the Avro file.
pub fn write_manifest_file(
    files: &[DataFileEntry],
    snapshot_id: SnapshotId,
    sequence_number: i64,
    table_schema_json: &str,
    partition_spec_json: &str,
) -> Bytes {
    use crate::avro_raw::{
        encode_empty_array, encode_int, encode_long, encode_string, encode_union_bytes,
        encode_union_long, encode_union_null, write_avro_container,
    };

    let mut records: Vec<Vec<u8>> = Vec::with_capacity(files.len());
    for f in files {
        let mut rec = Vec::new();
        encode_int(1, &mut rec); // status=ADDED
        encode_union_long(1, snapshot_id, &mut rec); // snapshot_id
        encode_union_long(1, sequence_number, &mut rec); // sequence_number
        encode_union_long(1, sequence_number, &mut rec); // file_sequence_number
                                                         // data_file (nested record — no tag bytes in Avro binary)
        encode_int(0, &mut rec); // content=DATA
        encode_string(&f.path, &mut rec); // file_path
        encode_string("PARQUET", &mut rec); // file_format
                                            // partition r102: empty record → 0 bytes
        encode_long(f.record_count as i64, &mut rec); // record_count
        encode_long(f.file_size_bytes as i64, &mut rec); // file_size_in_bytes
        encode_union_null(&mut rec); // column_sizes
        encode_union_null(&mut rec); // value_counts
        encode_union_null(&mut rec); // null_value_counts
        encode_union_null(&mut rec); // nan_value_counts
        encode_union_null(&mut rec); // lower_bounds
        encode_union_null(&mut rec); // upper_bounds
        let ext = AilakeEntryExt {
            centroid_b64: f.centroid_b64.clone(),
            radius: f.radius,
            hnsw_offset: f.hnsw_offset,
            hnsw_len: f.hnsw_len,
            vector_column: f.vector_column.clone(),
            vector_dim: f.vector_dim,
            extra_vector_indexes: f.extra_vector_indexes.clone(),
            index_status: f.index_status.clone(),
            batch_id: f.batch_id.clone(),
            embedding_model: f.embedding_model.clone(),
        };
        match serde_json::to_vec(&ext) {
            Ok(bytes) => encode_union_bytes(1, &bytes, &mut rec), // key_metadata=bytes
            Err(_) => encode_union_null(&mut rec),                // key_metadata=null
        }
        encode_union_null(&mut rec); // split_offsets
        encode_union_null(&mut rec); // equality_ids
        encode_union_null(&mut rec); // sort_order_id
                                     // (encode_empty_array not needed here — only arrays that aren't union-wrapped)
        let _ = encode_empty_array; // suppress unused warning
        records.push(rec);
    }

    let extra_meta: &[(&str, &[u8])] = &[
        ("schema", table_schema_json.as_bytes()),
        ("partition-spec", partition_spec_json.as_bytes()),
        ("partition-spec-id", b"0"),
        ("format-version", b"2"),
        ("content", b"data"),
    ];
    Bytes::from(write_avro_container(
        MANIFEST_ENTRY_SCHEMA_STR,
        extra_meta,
        &records,
    ))
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
                        batch_id: ext.as_ref().and_then(|e| e.batch_id.clone()),
                        embedding_model: ext.as_ref().and_then(|e| e.embedding_model.clone()),
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
    pub batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
}

/// Read manifest file paths from an Iceberg manifest list (Avro).
pub fn read_manifest_list(data: &[u8]) -> apache_avro::AvroResult<Vec<String>> {
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
            if let Some(p) = path {
                results.push(p);
            }
        }
    }
    Ok(results)
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
            batch_id: None,
            embedding_model: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 99, 1, schema_json, partition_spec);
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
            batch_id: Some("dag_run_2026-05-28_taskA".to_string()),
            embedding_model: None,
        };
        let schema_json = r#"{"schema-id":0,"type":"struct","fields":[]}"#;
        let partition_spec = r#"[{"spec-id":0,"fields":[]}]"#;
        let bytes = write_manifest_file(&[file], 42, 1, schema_json, partition_spec);
        let entries = read_manifest_file(&bytes).expect("read_manifest_file failed");
        assert_eq!(
            entries[0].batch_id.as_deref(),
            Some("dag_run_2026-05-28_taskA")
        );
    }
}
