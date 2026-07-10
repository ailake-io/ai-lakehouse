// SPDX-License-Identifier: MIT OR Apache-2.0
// Snapshot manifest builder.
// Phase 1: writes a simple JSON manifest (not Avro) to keep deps minimal.
// Phase 2: replace with apache-avro for Iceberg spec compliance.

use ailake_core::AilakeResult;
use serde::{Deserialize, Serialize};

use crate::provider::{DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId};

/// Phase 1 manifest: a JSON list of DataFileEntry records.
/// Iceberg-compatible Avro manifest is Phase 2.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub snapshot_id: SnapshotId,
    pub files: Vec<DataFileEntry>,
    /// Equality delete files active as of this snapshot (accumulated the same
    /// way `files` is: Append/Delete inherit the previous snapshot's list and
    /// extend it, Overwrite/Replace treat the caller-supplied list as complete).
    /// `#[serde(default)]` keeps old manifests (written before this field
    /// existed) readable.
    #[serde(default)]
    pub equality_delete_files: Vec<EqualityDeleteFile>,
}

impl Manifest {
    pub fn to_json(&self) -> AilakeResult<String> {
        serde_json::to_string_pretty(self).map_err(ailake_core::AilakeError::Json)
    }

    pub fn from_json(s: &str) -> AilakeResult<Self> {
        serde_json::from_str(s).map_err(ailake_core::AilakeError::Json)
    }
}

pub fn build_manifest(snapshot: &NewSnapshot) -> Manifest {
    Manifest {
        snapshot_id: snapshot.snapshot_id,
        files: snapshot.files.clone(),
        equality_delete_files: snapshot.equality_delete_files.clone(),
    }
}

pub fn manifest_path(snapshot_id: SnapshotId) -> String {
    format!("metadata/snap-{snapshot_id}.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::DataFileEntry;

    #[test]
    fn manifest_roundtrip() {
        let manifest = Manifest {
            snapshot_id: 12345,
            files: vec![DataFileEntry {
                path: "data/part-00001.parquet".to_string(),
                record_count: 100,
                file_size_bytes: 65536,
                centroid_b64: None,
                radius: Some(0.5),
                hnsw_offset: Some(4096),
                hnsw_len: Some(8192),
                vector_column: Some("embedding".to_string()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: crate::provider::IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            }],
            equality_delete_files: vec![],
        };
        let json = manifest.to_json().unwrap();
        let m2 = Manifest::from_json(&json).unwrap();
        assert_eq!(m2.snapshot_id, 12345);
        assert_eq!(m2.files[0].path, "data/part-00001.parquet");
    }
}
