// SPDX-License-Identifier: MIT OR Apache-2.0
// Iceberg V3 Deletion Vector write support — Phase C.
//
// Produces Roaring Bitmap blobs in minimal Puffin `.dvd` files and updates
// the manifest entry so scanners (Phase B) automatically mask deleted rows
// from HNSW and flat-scan results.
//
// Phase B (read) is independent: existing DVs written by Spark / Trino /
// PyIceberg are consumed without requiring Phase C.

use std::sync::Arc;

use bytes::Bytes;
use roaring::RoaringBitmap;

use ailake_catalog::{
    provider::{
        new_snapshot_id, CatalogProvider, DeletionVector, NewSnapshot, SnapshotOperation,
        TableIdent,
    },
    DataFileEntry, EqualityDeleteFile,
};
use ailake_core::{AilakeError, AilakeResult};
use ailake_store::Store;

use crate::dv::load_deletion_vector;

// ── Puffin writer ─────────────────────────────────────────────────────────────

/// Puffin magic bytes — per Iceberg Puffin spec §2.
const PUFFIN_MAGIC: &[u8] = b"PFAc";

/// Minimal single-blob Puffin file writer for Deletion Vectors.
///
/// Puffin format (simplified, one blob):
/// ```text
/// [4 bytes magic "PFAc"] [blob bytes] [footer JSON] [4 bytes footer_len LE] [4 bytes magic "PFAc"]
/// ```
/// The DV manifest entry stores `offset=4` (after magic) and `length=blob.len()`, so
/// readers skip the Puffin header/footer and fetch only the bitmap bytes via range GET.
pub struct PuffinWriter;

impl PuffinWriter {
    /// Serialize `bitmap` into a single-blob Puffin file.
    ///
    /// Returns `(file_bytes, blob_offset, blob_length)`.
    pub fn write_single_dv(
        bitmap: &RoaringBitmap,
        snapshot_id: i64,
    ) -> AilakeResult<(Bytes, u64, u64)> {
        let mut blob = Vec::new();
        bitmap.serialize_into(&mut blob).map_err(|e| {
            AilakeError::Io(std::io::Error::other(format!("DV serialize: {e}")))
        })?;

        let blob_offset = PUFFIN_MAGIC.len() as u64;
        let blob_length = blob.len() as u64;

        // Footer JSON per Iceberg Puffin spec §4.
        let footer_json = serde_json::json!({
            "blobs": [{
                "type": "deletion-vector-v1",
                "snapshot-id": snapshot_id,
                "sequence-number": 0,
                "offset": blob_offset,
                "length": blob_length
            }],
            "properties": {}
        })
        .to_string();
        let footer_bytes = footer_json.as_bytes();
        let footer_len = (footer_bytes.len() as u32).to_le_bytes();

        let mut out = Vec::with_capacity(
            PUFFIN_MAGIC.len() * 2 + blob.len() + footer_bytes.len() + 4,
        );
        out.extend_from_slice(PUFFIN_MAGIC);
        out.extend_from_slice(&blob);
        out.extend_from_slice(footer_bytes);
        out.extend_from_slice(&footer_len);
        out.extend_from_slice(PUFFIN_MAGIC);

        Ok((Bytes::from(out), blob_offset, blob_length))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Logically delete rows from a V3 AI-Lake table using Iceberg Deletion Vectors.
///
/// # What this does
/// 1. Verifies the table is `format-version=3` (DVs require V3).
/// 2. Reads the current file list from the catalog.
/// 3. Finds `file_path` in the snapshot (exact match or suffix match for
///    tables where the catalog prefixes absolute paths).
/// 4. Merges `row_ids` into the existing DV bitmap for that file (or creates
///    a new one if the file has no DV yet).
/// 5. Writes a new Puffin `.dvd` file to `{table_location}/metadata/dv-{snap_id}.dvd`.
/// 6. Commits a `Replace` snapshot so all readers see the updated DV immediately.
///
/// After the call, `scanner.rs` (Phase B) will automatically exclude the
/// deleted rows from HNSW and flat-scan results. The data file is not modified.
///
/// # Arguments
/// * `catalog` — catalog for manifest reads and snapshot commits.
/// * `store` — object store for Puffin file I/O.
/// * `table` — fully-qualified table identifier (`namespace.name`).
/// * `file_path` — path of the data file whose rows are being deleted.
///   May be a relative path (e.g. `"data/part-00001.parquet"`) or an absolute
///   path as returned by `catalog.list_files()`. Suffix matching is applied.
/// * `row_ids` — 0-based row positions to delete (within the data file).
///
/// # Errors
/// * `InvalidArgument` if the table is `format-version < 3`.
/// * `Catalog` if the table has no current snapshot or `file_path` is not found.
pub async fn delete_rows(
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: &TableIdent,
    file_path: &str,
    row_ids: &[u32],
) -> AilakeResult<()> {
    if row_ids.is_empty() {
        return Ok(());
    }

    // Verify table is V3.
    let meta = catalog.load_table(table).await?;
    if meta.format_version < 3 {
        return Err(AilakeError::InvalidArgument(format!(
            "Deletion Vectors require Iceberg V3 table (got format-version={}). \
             Recreate the table with format_version=3.",
            meta.format_version
        )));
    }

    // Load current file list.
    let mut files: Vec<DataFileEntry> = catalog.list_files(table, None).await?;

    // Find target file (exact match or suffix match for absolute-path manifests).
    let target_idx = files
        .iter()
        .position(|f| f.path == file_path || f.path.ends_with(file_path))
        .ok_or_else(|| {
            AilakeError::Catalog(format!(
                "file '{file_path}' not found in current snapshot"
            ))
        })?;

    // Build bitmap: merge existing DV with new row_ids.
    let mut bitmap = if let Some(ref dv) = files[target_idx].deletion_vector {
        load_deletion_vector(&store, dv)
            .await
            .unwrap_or_default()
    } else {
        RoaringBitmap::new()
    };
    for &id in row_ids {
        bitmap.insert(id);
    }
    let cardinality = bitmap.len() as i64;

    // Write Puffin .dvd file alongside table metadata.
    let snap_id = new_snapshot_id();
    let (puffin_bytes, blob_offset, blob_length) =
        PuffinWriter::write_single_dv(&bitmap, snap_id)?;
    let table_root = meta.location.trim_end_matches('/');
    let dv_path = format!("{table_root}/metadata/dv-{snap_id}.dvd");
    store.put(&dv_path, puffin_bytes).await?;

    // Patch the target entry with the new DV pointer.
    files[target_idx].deletion_vector = Some(DeletionVector {
        path: dv_path,
        offset: blob_offset,
        length: blob_length,
        cardinality,
    });

    // Replace snapshot: carries all files with the updated DV entry.
    // Replace does not inherit old manifests — the full file list is the new state.
    let snapshot = NewSnapshot {
        snapshot_id: snap_id,
        parent_snapshot_id: meta.current_snapshot_id,
        files,
        operation: SnapshotOperation::Replace,
        iceberg_schema: None,
        extra_properties: std::collections::HashMap::new(),
        bloom_filters: vec![],
                equality_delete_files: vec![],
    };
    catalog.commit_snapshot(table, snapshot).await?;
    Ok(())
}

// ── Equality Delete (Phase H) ──────────────────────────────────────────────────

/// Logically delete all rows where `column_name` equals any value in `values`.
///
/// Writes an Iceberg equality delete Avro file containing one row per value,
/// then commits a `Delete` snapshot that inherits existing data manifests and
/// appends a new delete manifest (`content=1`) pointing to that file.
///
/// Scanners that load equality delete files (AI-Lake, Spark, Trino with plugin)
/// will automatically mask matching rows at read time without rewriting data files.
///
/// # Arguments
/// * `column_name` — column to match against (must exist in the table schema)
/// * `values` — values that identify rows to delete
pub async fn delete_where(
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: &TableIdent,
    column_name: &str,
    values: &[&str],
) -> AilakeResult<()> {
    if values.is_empty() {
        return Ok(());
    }

    let meta = catalog.load_table(table).await?;
    let table_root = meta.location.trim_end_matches('/');

    // Look up field-id and iceberg_type from schema_fields. Fall back to id=0 / "string"
    // for tables without schema_fields (old format) — the column name in the Avro file
    // is still sufficient for AI-Lake's own scanner.
    let (field_id, iceberg_type) = meta
        .schema_fields
        .iter()
        .find(|sf| sf.name == column_name)
        .map(|sf| (sf.id, sf.iceberg_type.clone()))
        .unwrap_or((0, "string".to_string()));

    // Write equality delete Avro file.
    let snap_id = new_snapshot_id();
    let eq_del_avro =
        ailake_catalog::write_equality_delete_avro(column_name, field_id, &iceberg_type, values)
            .map_err(|e| AilakeError::Catalog(e.to_string()))?;
    let file_size = eq_del_avro.len() as u64;
    let eq_del_path = format!("{table_root}/metadata/eq-del-{snap_id}.avro");
    store.put(&eq_del_path, eq_del_avro).await?;

    let eq_del_file = ailake_catalog::EqualityDeleteFile {
        path: eq_del_path,
        equality_ids: vec![field_id],
        record_count: values.len() as u64,
        file_size_bytes: file_size,
    };

    // Commit Delete snapshot — inherits previous data manifests, appends delete manifest.
    let snapshot = NewSnapshot {
        snapshot_id: snap_id,
        parent_snapshot_id: meta.current_snapshot_id,
        files: vec![],
        operation: SnapshotOperation::Delete,
        iceberg_schema: None,
        extra_properties: std::collections::HashMap::new(),
        bloom_filters: vec![],
        equality_delete_files: vec![eq_del_file],
    };
    catalog.commit_snapshot(table, snapshot).await?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::{
        provider::{IndexStatus, TableProperties},
        HadoopCatalog,
    };
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use ailake_store::LocalStore;

    fn make_props(format_version: u8) -> TableProperties {
        TableProperties {
            policy: VectorStoragePolicy {
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
            },
            extra: std::collections::HashMap::new(),
            format_version,
        }
    }

    fn make_file_entry(path: &str) -> DataFileEntry {
        DataFileEntry {
            path: path.to_string(),
            record_count: 100,
            file_size_bytes: 4096,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: Some("embedding".to_string()),
            vector_dim: Some(4),
            extra_vector_indexes: vec![],
            index_status: IndexStatus::Ready,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
        }
    }

    async fn setup_v3_table(
        warehouse: &str,
        store: Arc<dyn Store>,
    ) -> (Arc<dyn CatalogProvider>, TableIdent) {
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(Arc::clone(&store), warehouse));
        let table = TableIdent::new("default", "docs");
        catalog.create_table(&table, &make_props(3)).await.unwrap();

        let snap = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![make_file_entry("data/part-00001.parquet")],
            operation: SnapshotOperation::Append,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
                equality_delete_files: vec![],
        };
        catalog.commit_snapshot(&table, snap).await.unwrap();
        (catalog, table)
    }

    #[tokio::test]
    async fn writes_dv_and_manifest_reflects_cardinality() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let (catalog, table) = setup_v3_table("", Arc::clone(&store)).await;

        delete_rows(
            Arc::clone(&catalog),
            Arc::clone(&store),
            &table,
            "data/part-00001.parquet",
            &[5, 10, 42],
        )
        .await
        .unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files.len(), 1);
        let dv = files[0].deletion_vector.as_ref().expect("DV should be present");
        assert_eq!(dv.cardinality, 3);

        // Verify Puffin file was created and bitmap is correct.
        let bm = load_deletion_vector(&store, dv).await.unwrap();
        assert!(bm.contains(5));
        assert!(bm.contains(10));
        assert!(bm.contains(42));
        assert!(!bm.contains(0));
        assert_eq!(bm.len(), 3);
    }

    #[tokio::test]
    async fn merges_with_existing_dv_across_calls() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let (catalog, table) = setup_v3_table("", Arc::clone(&store)).await;

        // First delete batch.
        delete_rows(
            Arc::clone(&catalog),
            Arc::clone(&store),
            &table,
            "data/part-00001.parquet",
            &[1, 2],
        )
        .await
        .unwrap();

        // Second delete batch — should accumulate.
        delete_rows(
            Arc::clone(&catalog),
            Arc::clone(&store),
            &table,
            "data/part-00001.parquet",
            &[3, 4],
        )
        .await
        .unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        let dv = files[0].deletion_vector.as_ref().unwrap();
        let bm = load_deletion_vector(&store, dv).await.unwrap();
        assert!(bm.contains(1) && bm.contains(2) && bm.contains(3) && bm.contains(4));
        assert_eq!(bm.len(), 4);
    }

    #[tokio::test]
    async fn rejects_v2_table() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));
        let table = TableIdent::new("default", "docs");
        catalog.create_table(&table, &make_props(2)).await.unwrap();

        let err = delete_rows(
            Arc::clone(&catalog),
            Arc::clone(&store),
            &table,
            "data/part-00001.parquet",
            &[0],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("format-version=2"));
    }

    #[tokio::test]
    async fn noop_when_row_ids_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let (catalog, table) = setup_v3_table("", Arc::clone(&store)).await;

        // Should return Ok immediately, no DV written.
        delete_rows(
            Arc::clone(&catalog),
            Arc::clone(&store),
            &table,
            "data/part-00001.parquet",
            &[],
        )
        .await
        .unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert!(files[0].deletion_vector.is_none());
    }

    #[tokio::test]
    async fn puffin_magic_and_structure_valid() {
        let mut bm = RoaringBitmap::new();
        bm.insert(7);
        bm.insert(99);
        let (bytes, offset, length) = PuffinWriter::write_single_dv(&bm, 42).unwrap();

        // Starts and ends with magic.
        assert_eq!(&bytes[..4], PUFFIN_MAGIC);
        assert_eq!(&bytes[bytes.len() - 4..], PUFFIN_MAGIC);

        // Bitmap bytes are at the declared offset.
        let blob_slice = &bytes[offset as usize..(offset + length) as usize];
        let recovered = RoaringBitmap::deserialize_from(blob_slice).unwrap();
        assert!(recovered.contains(7) && recovered.contains(99));
    }
}
