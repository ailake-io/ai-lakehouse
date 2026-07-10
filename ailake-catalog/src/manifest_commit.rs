// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared Iceberg V2/V3 manifest read/write logic for every backend that owns
//! its own `metadata.json` (Hadoop, Glue, Jdbc) — real Avro manifests + manifest-
//! list chaining, V3 `first_row_id` assignment, Phase F Puffin stats, Phase J
//! partition stats, Phase I partition-spec source-id remap. Extracted from
//! `HadoopCatalog` (the original, most complete implementation) so Glue/Jdbc
//! can't silently drift from it — before this, they wrote a much simpler flat-
//! JSON manifest (`crate::snapshot::Manifest`) that never adopted any of these
//! V3/partition features and silently dropped `equality_delete_files` entirely.
//!
//! Backends that delegate metadata management to a server (Rest/Nessie) don't
//! use this — they have their own protocol-native equivalent (`rest.rs`).

use base64::Engine as _;

use crate::avro_manifest::{
    read_equality_delete_manifest, read_manifest_file, read_manifest_list_typed,
    write_equality_delete_manifest, write_manifest_file, write_manifest_list_multi_typed,
    write_partition_stats_parquet,
};
use crate::metadata::{IcebergMetadata, IcebergPartitionStatsRef, IcebergSnapshot};
use crate::provider::{
    DataFileEntry, EqualityDeleteFile, NewSnapshot, SnapshotId, SnapshotOperation,
};
use ailake_core::{AilakeError, AilakeResult};
use ailake_store::Store;

/// Extract centroid + radius from each DataFileEntry for Phase F Puffin stats.
/// Files without centroid metadata (e.g. Indexing status) are skipped.
pub(crate) fn collect_vector_stats(files: &[DataFileEntry]) -> Vec<crate::puffin::VectorStatEntry> {
    files
        .iter()
        .filter_map(|f| {
            let b64 = f.centroid_b64.as_ref()?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
            let centroid: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| {
                    f32::from_le_bytes(
                        b.try_into()
                            .expect("chunks_exact(4) guarantees 4-byte slices"),
                    )
                })
                .collect();
            let radius = f.radius?;
            Some(crate::puffin::VectorStatEntry {
                path: f.path.clone(),
                centroid,
                radius,
            })
        })
        .collect()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/// Apply one commit to `meta` in place: writes the Avro data manifest (+ delete
/// manifest, if any), chains the manifest list, assigns V3 `first_row_id`, writes
/// Phase F Puffin stats + Phase J partition stats, applies any `iceberg_schema`/
/// `extra_properties` update, and pushes the new `IcebergSnapshot`. Does **not**
/// persist `meta` — the caller owns that. Each backend has its own pointer-write
/// and OCC/CAS mechanism: `HadoopCatalog::save_metadata`, Glue's version-id-
/// guarded `update_table`, Jdbc's conditional `UPDATE` on `metadata_location`.
pub(crate) async fn commit_into_metadata(
    store: &dyn Store,
    table_root: &str,
    warehouse: &str,
    meta: &mut IcebergMetadata,
    snapshot: NewSnapshot,
) -> AilakeResult<SnapshotId> {
    let snap_id = snapshot.snapshot_id;
    let seq = meta.last_sequence_number + 1;

    // Serialize the actual Iceberg schema for this table (phase I: may contain partition column).
    let current_schema = meta
        .schemas
        .iter()
        .find(|s| s["schema-id"].as_i64() == Some(meta.current_schema_id as i64))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"schema-id":0,"type":"struct","fields":[]}));
    let table_schema_json = serde_json::to_string(&current_schema)
        .unwrap_or_else(|_| r#"{"schema-id":0,"type":"struct","fields":[]}"#.to_string());
    let partition_spec_json = serde_json::to_string(&meta.partition_specs)
        .unwrap_or_else(|_| r#"[{"spec-id":0,"fields":[]}]"#.to_string());

    // Parse the active partition spec for native partition value encoding.
    let active_partition_spec = meta.to_table_metadata().partition_spec;

    // Write Avro manifest file for the new data files.
    // Iceberg spec requires absolute file paths in manifests. Prefix relative paths
    // with the warehouse root only when the warehouse is itself an absolute path
    // (starts with '/' or contains a URI scheme). Relative warehouse names (e.g. in
    // unit tests) are left as-is so the store can resolve them normally.
    let warehouse_prefix: Option<&str> = if warehouse.starts_with('/') || warehouse.contains("://")
    {
        Some(warehouse)
    } else {
        None
    };
    // Build absolute-path file list. For V3 tables, assign first_row_id from
    // the table's next-row-id counter so every row has a globally unique ID.
    let mut abs_files: Vec<DataFileEntry> = snapshot
        .files
        .iter()
        .map(|f| {
            let path = if f.path.starts_with('/') || f.path.contains("://") {
                f.path.clone()
            } else if let Some(prefix) = warehouse_prefix {
                format!("{}/{}", prefix, f.path)
            } else {
                f.path.clone()
            };
            DataFileEntry { path, ..f.clone() }
        })
        .collect();

    if meta.format_version >= 3 {
        let mut next_id = meta.next_row_id;
        for f in abs_files.iter_mut() {
            // Compaction pre-sets first_row_id from source files — respect it.
            // Only allocate fresh IDs (and advance the counter) for brand-new files.
            if f.first_row_id.is_none() {
                f.first_row_id = Some(next_id);
                next_id += f.record_count as i64;
            }
        }
        meta.next_row_id = next_id;
    }
    let added_rows: i64 = abs_files.iter().map(|f| f.record_count as i64).sum();
    let manifest_file_path = format!("{}/metadata/{}-m0.avro", table_root, snap_id);
    let manifest_bytes = write_manifest_file(
        &abs_files,
        snap_id,
        seq,
        &table_schema_json,
        &partition_spec_json,
        meta.format_version as u8,
        active_partition_spec.as_ref(),
    );
    let manifest_len = manifest_bytes.len();
    store.put(&manifest_file_path, manifest_bytes).await?;

    // Collect manifest paths from the previous snapshot (if any) for the manifest list.
    // Replace/Overwrite: new manifest IS the complete state — don't inherit old manifests.
    // Append/Delete: inherit previous manifests so old files remain visible.
    // Manifests carry content: 0=data, 1=delete.
    let mut all_manifests: Vec<(String, i64, i32)> = Vec::new();
    if matches!(
        snapshot.operation,
        SnapshotOperation::Append | SnapshotOperation::Delete
    ) {
        if let Some(prev_snap) = meta.snapshots.last() {
            if let Ok(ml_bytes) = store.get(&prev_snap.manifest_list).await {
                if let Ok(prev_manifests) = read_manifest_list_typed(&ml_bytes) {
                    for (prev_path, content) in prev_manifests {
                        let len = store.file_size(&prev_path).await.unwrap_or(0) as i64;
                        all_manifests.push((prev_path, len, content));
                    }
                }
            }
        }
    }
    all_manifests.push((manifest_file_path.clone(), manifest_len as i64, 0));

    // Phase H: write delete manifest for equality delete files (if any).
    let abs_eq_deletes: Vec<EqualityDeleteFile> = snapshot
        .equality_delete_files
        .iter()
        .map(|d| EqualityDeleteFile {
            path: if d.path.starts_with('/') || d.path.contains("://") {
                d.path.clone()
            } else {
                format!("{}/{}", table_root, d.path)
            },
            equality_ids: d.equality_ids.clone(),
            record_count: d.record_count,
            file_size_bytes: d.file_size_bytes,
        })
        .collect();
    if !abs_eq_deletes.is_empty() {
        let del_manifest_path = format!("{}/metadata/{}-eq-del.avro", table_root, snap_id);
        let del_manifest_bytes = write_equality_delete_manifest(&abs_eq_deletes, snap_id, seq);
        let del_manifest_len = del_manifest_bytes.len();
        store.put(&del_manifest_path, del_manifest_bytes).await?;
        all_manifests.push((del_manifest_path, del_manifest_len as i64, 1));
    }

    // Write Avro manifest list for this snapshot
    let manifest_list_path = format!("{}/metadata/snap-{}-1.avro", table_root, snap_id);
    let ml_bytes = write_manifest_list_multi_typed(&all_manifests, snap_id, seq, added_rows);
    store.put(&manifest_list_path, ml_bytes).await?;

    let commit_now_ms = now_ms();
    let operation = snapshot.operation.clone();
    let iceberg_snap = IcebergSnapshot {
        snapshot_id: snap_id,
        parent_snapshot_id: snapshot.parent_snapshot_id,
        sequence_number: seq,
        timestamp_ms: commit_now_ms,
        manifest_list: manifest_list_path,
        summary: std::collections::HashMap::from([
            (
                "operation".to_string(),
                format!("{:?}", operation).to_lowercase(),
            ),
            (
                "added-data-files".to_string(),
                snapshot.files.len().to_string(),
            ),
        ]),
        schema_id: Some(0),
    };
    meta.last_sequence_number = seq;
    meta.last_updated_ms = commit_now_ms;
    meta.current_snapshot_id = Some(snap_id);
    meta.snapshots.push(iceberg_snap);

    if let Some(schema_update) = snapshot.iceberg_schema {
        // Bootstrap metadata (IcebergMetadata::new, for partition_by /
        // partition_fields tables) assigns the partition column field-id
        // assuming it is the *only* schema field at table-creation time.
        // The real first write replaces that bootstrap schema with the
        // full column set in actual Arrow order, so the partition column
        // can land at a different field-id (e.g. "topic_id" written
        // second behind "text" ends up id=2, not id=1). Remap every
        // partition-spec's source-id to whatever id the matching column
        // name now has — otherwise readers (Trino/Spark iceberg-java)
        // reject the table with "Cannot create identity partition
        // sourced from different field in schema".
        let new_id_by_name: std::collections::HashMap<&str, i64> = schema_update
            .fields
            .iter()
            .filter_map(|f| Some((f["name"].as_str()?, f["id"].as_i64()?)))
            .collect();
        for spec in meta.partition_specs.iter_mut() {
            if let Some(fields) = spec["fields"].as_array_mut() {
                for pf in fields.iter_mut() {
                    if let Some(name) = pf["name"].as_str() {
                        if let Some(&new_id) = new_id_by_name.get(name) {
                            pf["source-id"] = serde_json::json!(new_id);
                        }
                    }
                }
            }
        }

        if let Some(schema) = meta.schemas.first_mut() {
            schema["fields"] = serde_json::Value::Array(schema_update.fields);
        }
        meta.last_column_id = schema_update.last_column_id;
        meta.properties.insert(
            "schema.name-mapping.default".to_string(),
            schema_update.name_mapping_json,
        );
    }

    // Merge secondary-column properties (ailake.dim-<col>, ailake.metric-<col>).
    for (k, v) in snapshot.extra_properties {
        meta.properties.insert(k, v);
    }

    // Phase F: write Puffin stats file for V3 tables (vector stats + BM25 bloom).
    if meta.format_version >= 3 {
        let vector_stats = collect_vector_stats(&abs_files);
        let bm25_blooms: Vec<crate::puffin::BM25BloomEntry> = snapshot
            .bloom_filters
            .iter()
            .map(|(path, bytes)| crate::puffin::BM25BloomEntry {
                path: path.clone(),
                bloom_bytes: bytes.clone(),
            })
            .collect();

        if !vector_stats.is_empty() {
            match crate::puffin::AilakePuffinWriter::write_stats(
                &vector_stats,
                &bm25_blooms,
                snap_id,
            ) {
                Ok(result) => {
                    let puffin_path = format!("{table_root}/metadata/stats-{snap_id}.puffin");
                    let puffin_len = result.bytes.len() as u64;
                    if let Err(e) = store.put(&puffin_path, result.bytes).await {
                        tracing::warn!("ailake: Phase F — failed to write Puffin stats: {e}");
                    } else {
                        use crate::metadata::{BlobRef, IcebergStatisticsRef};
                        let mut blob_refs = vec![BlobRef {
                            blob_type: crate::puffin::BLOB_TYPE_VECTOR_STATS.to_string(),
                            snapshot_id: snap_id,
                            sequence_number: seq,
                            fields: vec![],
                            offset: result.vector_stats_blob.0,
                            length: result.vector_stats_blob.1,
                        }];
                        if let Some((off, len)) = result.bm25_bloom_blob {
                            blob_refs.push(BlobRef {
                                blob_type: crate::puffin::BLOB_TYPE_BM25_BLOOM.to_string(),
                                snapshot_id: snap_id,
                                sequence_number: seq,
                                fields: vec![],
                                offset: off,
                                length: len,
                            });
                        }
                        meta.statistics.push(IcebergStatisticsRef {
                            snapshot_id: snap_id,
                            statistics_path: puffin_path,
                            file_size_in_bytes: puffin_len,
                            file_footer_size_in_bytes: result.footer_size as u64,
                            blob_file_references: blob_refs,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("ailake: Phase F — Puffin stats encode error: {e}");
                }
            }
        }
    }

    // Phase J: write partition statistics Parquet file for partitioned tables.
    // Covers ALL data files in this snapshot (reads every data manifest) so that
    // Spark/Trino can do partition-level aggregations without scanning data files.
    if let Some(spec) = &active_partition_spec {
        if !spec.is_unpartitioned() {
            let mut all_data_entries: Vec<DataFileEntry> = Vec::new();
            for (mpath, _len, content) in &all_manifests {
                if *content != 0 {
                    continue;
                }
                match store.get(mpath).await {
                    Ok(mb) => match read_manifest_file(&mb) {
                        Ok(entries) => all_data_entries.extend(entries),
                        Err(e) => {
                            tracing::warn!("ailake: Phase J — manifest read error {mpath}: {e}")
                        }
                    },
                    Err(e) => {
                        tracing::warn!("ailake: Phase J — store get error {mpath}: {e}")
                    }
                }
            }

            match write_partition_stats_parquet(spec, &all_data_entries) {
                Ok(stats_bytes) => {
                    let stats_path =
                        format!("{table_root}/metadata/partition-stats-{snap_id}.parquet");
                    let stats_len = stats_bytes.len() as u64;
                    match store.put(&stats_path, stats_bytes).await {
                        Ok(()) => {
                            meta.partition_statistics.push(IcebergPartitionStatsRef {
                                snapshot_id: snap_id,
                                statistics_path: stats_path,
                                file_size_in_bytes: stats_len,
                            });
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ailake: Phase J — failed to write partition stats: {e}"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("ailake: Phase J — partition stats encode error: {e}");
                }
            }
        }
    }

    Ok(snap_id)
}

/// Read the active (or given) snapshot's data manifests back into a flat
/// `DataFileEntry` list — skips delete manifests (`content=1`).
pub(crate) async fn list_files_from_metadata(
    store: &dyn Store,
    meta: &IcebergMetadata,
    snapshot_id: Option<SnapshotId>,
) -> AilakeResult<Vec<DataFileEntry>> {
    let snap_id = match snapshot_id.or(meta.current_snapshot_id) {
        Some(id) => id,
        None => return Ok(vec![]), // new table — no snapshots yet, no committed files
    };

    let snap = meta
        .snapshots
        .iter()
        .find(|s| s.snapshot_id == snap_id)
        .ok_or_else(|| AilakeError::Catalog(format!("snapshot {snap_id} not found")))?;

    let ml_bytes = store.get(&snap.manifest_list).await?;
    let manifest_entries =
        read_manifest_list_typed(&ml_bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;

    let mut entries: Vec<DataFileEntry> = Vec::new();
    for (mpath, content) in manifest_entries {
        if content != 0 {
            continue; // skip delete manifests (content=1)
        }
        let mf_bytes = store.get(&mpath).await?;
        let file_entries =
            read_manifest_file(&mf_bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;
        entries.extend(file_entries);
    }
    Ok(entries)
}

/// Read the active (or given) snapshot's delete manifests (`content=1`) back
/// into a flat `EqualityDeleteFile` list.
pub(crate) async fn list_equality_deletes_from_metadata(
    store: &dyn Store,
    meta: &IcebergMetadata,
    snapshot_id: Option<SnapshotId>,
) -> AilakeResult<Vec<EqualityDeleteFile>> {
    let snap_id = match snapshot_id.or(meta.current_snapshot_id) {
        Some(id) => id,
        None => return Ok(vec![]),
    };
    let snap = meta
        .snapshots
        .iter()
        .find(|s| s.snapshot_id == snap_id)
        .ok_or_else(|| AilakeError::Catalog(format!("snapshot {snap_id} not found")))?;

    let ml_bytes = store.get(&snap.manifest_list).await?;
    let manifest_entries =
        read_manifest_list_typed(&ml_bytes).map_err(|e| AilakeError::Catalog(e.to_string()))?;

    let mut deletes: Vec<EqualityDeleteFile> = Vec::new();
    for (mpath, content) in manifest_entries {
        if content != 1 {
            continue; // only delete manifests
        }
        let mf_bytes = store.get(&mpath).await?;
        let entries = read_equality_delete_manifest(&mf_bytes)
            .map_err(|e| AilakeError::Catalog(e.to_string()))?;
        deletes.extend(entries);
    }
    Ok(deletes)
}
