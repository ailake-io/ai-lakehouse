// SPDX-License-Identifier: MIT OR Apache-2.0
// Iceberg Spec v2 metadata.json read/write.
// Only the fields needed by AI-Lake are modelled — the rest are passed through as JSON.

use std::collections::HashMap;

use ailake_core::{AilakeError, AilakeResult, EmbeddingModelInfo, VectorStoragePolicy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::provider::{SnapshotId, TableMetadata};

#[derive(Debug, Serialize, Deserialize)]
pub struct IcebergMetadata {
    #[serde(rename = "format-version")]
    pub format_version: i32,
    #[serde(rename = "table-uuid")]
    pub table_uuid: String,
    pub location: String,
    #[serde(rename = "last-sequence-number", default)]
    pub last_sequence_number: i64,
    /// Iceberg V3 Row Lineage: next available globally-unique row ID.
    /// Incremented by record_count for each new data file at commit time.
    /// Absent (0) in V2 tables — field is ignored when format-version < 3.
    #[serde(rename = "next-row-id", default)]
    pub next_row_id: i64,
    #[serde(rename = "last-updated-ms")]
    pub last_updated_ms: i64,
    #[serde(rename = "last-column-id", default)]
    pub last_column_id: i32,
    #[serde(default)]
    pub schemas: Vec<Value>,
    #[serde(rename = "current-schema-id", default)]
    pub current_schema_id: i32,
    #[serde(rename = "partition-specs", default)]
    pub partition_specs: Vec<Value>,
    #[serde(rename = "default-spec-id", default)]
    pub default_spec_id: i32,
    #[serde(rename = "last-partition-id", default)]
    pub last_partition_id: i32,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(rename = "current-snapshot-id", default)]
    pub current_snapshot_id: Option<SnapshotId>,
    #[serde(default)]
    pub snapshots: Vec<IcebergSnapshot>,
    #[serde(rename = "snapshot-log", default)]
    pub snapshot_log: Vec<Value>,
    #[serde(rename = "metadata-log", default)]
    pub metadata_log: Vec<Value>,
    #[serde(rename = "sort-orders", default)]
    pub sort_orders: Vec<Value>,
    #[serde(rename = "default-sort-order-id", default)]
    pub default_sort_order_id: i32,
    #[serde(rename = "refs", default)]
    pub refs: HashMap<String, Value>,
    /// Iceberg V3 statistics files (Puffin). Only written for format-version=3.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub statistics: Vec<IcebergStatisticsRef>,
}

/// Iceberg V3 statistics file reference stored in `metadata.json`.
/// Points to a Puffin file containing table/snapshot statistics blobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergStatisticsRef {
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(rename = "statistics-path")]
    pub statistics_path: String,
    #[serde(rename = "file-size-in-bytes")]
    pub file_size_in_bytes: u64,
    #[serde(rename = "file-footer-size-in-bytes")]
    pub file_footer_size_in_bytes: u64,
    /// Blob descriptors within the Puffin file. May be empty — readers can
    /// always parse the Puffin footer directly for full blob metadata.
    #[serde(rename = "blob-file-references", default, skip_serializing_if = "Vec::is_empty")]
    pub blob_file_references: Vec<BlobRef>,
}

/// Describes one blob within an Iceberg Puffin statistics file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobRef {
    #[serde(rename = "type")]
    pub blob_type: String,
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(default)]
    pub fields: Vec<i32>,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IcebergSnapshot {
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: SnapshotId,
    #[serde(rename = "parent-snapshot-id", skip_serializing_if = "Option::is_none")]
    pub parent_snapshot_id: Option<SnapshotId>,
    #[serde(rename = "sequence-number")]
    pub sequence_number: i64,
    #[serde(rename = "timestamp-ms")]
    pub timestamp_ms: i64,
    #[serde(rename = "manifest-list")]
    pub manifest_list: String,
    pub summary: HashMap<String, String>,
    #[serde(rename = "schema-id")]
    pub schema_id: Option<i32>,
}

impl IcebergMetadata {
    /// Create a new metadata.json for a fresh AI-Lake table.
    ///
    /// `format_version`: 2 = Iceberg V2 (default); 3 = Iceberg V3 opt-in.
    /// V3 tables are append/update compatible; equality deletes and partition
    /// statistics require future phases (see docs/specs/ICEBERG_V3.md).
    pub fn new(location: &str, policy: &VectorStoragePolicy, format_version: u8) -> Self {
        let mut properties = HashMap::new();
        properties.insert("ailake.format-version".to_string(), "1".to_string());
        properties.insert(
            "ailake.vector-column".to_string(),
            policy.column_name.clone(),
        );
        properties.insert("ailake.vector-dim".to_string(), policy.dim.to_string());
        properties.insert(
            "ailake.vector-metric".to_string(),
            format!("{:?}", policy.metric).to_lowercase(),
        );
        properties.insert(
            "ailake.vector-precision".to_string(),
            format!("{:?}", policy.precision).to_lowercase(),
        );
        if let Some(m) = policy.hnsw_m {
            properties.insert("ailake.hnsw-m".to_string(), m.to_string());
        }
        if let Some(ef) = policy.hnsw_ef_construction {
            properties.insert("ailake.hnsw-ef-construction".to_string(), ef.to_string());
        }
        if let Some(model) = &policy.embedding_model {
            properties.insert(
                EmbeddingModelInfo::property_key().to_string(),
                model.to_property_value(),
            );
            if let Some(dim) = model.dim {
                properties.insert("ailake.embedding-model-dim".to_string(), dim.to_string());
            }
            if let Some(metric) = model.metric {
                properties.insert(
                    "ailake.embedding-model-metric".to_string(),
                    format!("{:?}", metric).to_lowercase(),
                );
            }
        }
        if let Some(modality) = policy.modality {
            properties.insert(
                format!("ailake.modality-{}", policy.column_name),
                modality.as_str().to_string(),
            );
        }
        if let Some(col) = &policy.partition_by {
            properties.insert("ailake.partition-by".to_string(), col.clone());
        }

        // When partition_by is set, emit an Iceberg identity partition spec so
        // Iceberg-aware engines (Spark, Trino, PyIceberg) can push down filters.
        let (partition_specs, default_spec_id, last_partition_id) =
            if let Some(col) = &policy.partition_by {
                let spec = serde_json::json!({
                    "spec-id": 1,
                    "fields": [{
                        "name": col,
                        "transform": "identity",
                        "source-id": 1000,
                        "field-id": 1000
                    }]
                });
                (vec![serde_json::json!({"spec-id": 0, "fields": []}), spec], 1, 1000)
            } else {
                (vec![serde_json::json!({"spec-id": 0, "fields": []})], 0, 999)
            };

        let format_version = format_version.max(2) as i32;
        if format_version >= 3 {
            eprintln!(
                "[ailake] WARN: creating Iceberg V3 table at {location} — \
                 append/update workloads fully supported; \
                 equality deletes not implemented"
            );
        }
        let now_ms = now_ms();
        IcebergMetadata {
            format_version,
            table_uuid: Uuid::new_v4().to_string(),
            location: location.to_string(),
            last_sequence_number: 0,
            next_row_id: 0,
            last_updated_ms: now_ms,
            last_column_id: 0,
            schemas: vec![serde_json::json!({"schema-id": 0, "type": "struct", "fields": []})],
            current_schema_id: 0,
            partition_specs,
            default_spec_id,
            last_partition_id,
            properties,
            current_snapshot_id: None,
            snapshots: vec![],
            snapshot_log: vec![],
            metadata_log: vec![],
            sort_orders: vec![serde_json::json!({"order-id": 0, "fields": []})],
            default_sort_order_id: 0,
            refs: HashMap::new(),
            statistics: vec![],
        }
    }

    pub fn to_json(&self) -> AilakeResult<String> {
        serde_json::to_string_pretty(self).map_err(AilakeError::Json)
    }

    pub fn from_json(s: &str) -> AilakeResult<Self> {
        serde_json::from_str(s).map_err(AilakeError::Json)
    }

    pub fn to_table_metadata(&self) -> TableMetadata {
        // Find the Puffin stats file for the current snapshot (most recent wins).
        let current_statistics_path = self.current_snapshot_id.and_then(|snap_id| {
            self.statistics
                .iter()
                .rev()
                .find(|s| s.snapshot_id == snap_id)
                .map(|s| s.statistics_path.clone())
        });

        // Phase G: parse schema fields from the current schema entry.
        let schema_fields = self
            .schemas
            .iter()
            .find(|s| s["schema-id"].as_i64() == Some(self.current_schema_id as i64))
            .and_then(|s| s["fields"].as_array().cloned())
            .map(|fields| {
                fields
                    .into_iter()
                    .filter_map(|f| {
                        let id = f["id"].as_i64()? as i32;
                        let name = f["name"].as_str()?.to_string();
                        let required = f["required"].as_bool().unwrap_or(false);
                        let iceberg_type = match &f["type"] {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        let initial_default = f
                            .get("initial-default")
                            .filter(|v| !v.is_null())
                            .cloned();
                        let write_default = f
                            .get("write-default")
                            .filter(|v| !v.is_null())
                            .cloned();
                        Some(crate::provider::SchemaField {
                            id,
                            name,
                            required,
                            iceberg_type,
                            initial_default,
                            write_default,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        TableMetadata {
            table_uuid: self.table_uuid.clone(),
            format_version: self.format_version,
            location: self.location.clone(),
            properties: self.properties.clone(),
            current_snapshot_id: self.current_snapshot_id,
            current_statistics_path,
            schema_fields,
            equality_delete_files: vec![], // populated lazily via list_equality_deletes
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_core::{VectorMetric, VectorPrecision};

    fn make_policy() -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim: 4,
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
        }
    }

    #[test]
    fn roundtrip_json() {
        let meta = IcebergMetadata::new("s3://my-lake/my_table", &make_policy(), 2);
        let json = meta.to_json().unwrap();
        let meta2 = IcebergMetadata::from_json(&json).unwrap();
        assert_eq!(meta2.format_version, 2);
        assert_eq!(
            meta2.properties.get("ailake.vector-column"),
            Some(&"embedding".to_string())
        );
    }

    #[test]
    fn properties_contain_ailake_keys() {
        let meta = IcebergMetadata::new("file:///tmp/tbl", &make_policy(), 2);
        assert!(meta.properties.contains_key("ailake.format-version"));
        assert!(meta.properties.contains_key("ailake.vector-dim"));
    }

    #[test]
    fn embedding_model_stored_in_properties() {
        use ailake_core::EmbeddingModelInfo;
        let mut policy = make_policy();
        policy.embedding_model =
            Some(EmbeddingModelInfo::new("text-embedding-3-small").with_version("2024-01"));
        let meta = IcebergMetadata::new("file:///tmp/tbl", &policy, 2);
        assert_eq!(
            meta.properties
                .get("ailake.embedding-model")
                .map(|s| s.as_str()),
            Some("text-embedding-3-small@2024-01")
        );
    }

    #[test]
    fn format_version_v3_emitted() {
        let meta = IcebergMetadata::new("s3://my-lake/v3_table", &make_policy(), 3);
        let json = meta.to_json().unwrap();
        let meta2 = IcebergMetadata::from_json(&json).unwrap();
        assert_eq!(meta2.format_version, 3);
    }

    #[test]
    fn format_version_defaults_to_v2() {
        let meta = IcebergMetadata::new("s3://my-lake/v2_table", &make_policy(), 2);
        assert_eq!(meta.format_version, 2);
    }
}
