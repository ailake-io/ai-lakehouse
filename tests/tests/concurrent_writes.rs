// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stress tests for catalog concurrent-write safety.
//!
//! HadoopCatalog: tokio::sync::Mutex serializes intra-process commits.
//! JdbcCatalog:   CAS UPDATE + retry handles concurrent SQLite writers.

mod fixtures;

use std::collections::HashMap;
use std::sync::Arc;

use ailake_catalog::JdbcCatalog;
use ailake_catalog::{
    new_snapshot_id, CatalogProvider, DataFileEntry, HadoopCatalog, IndexStatus, NewSnapshot,
    SnapshotOperation, TableIdent, TableProperties,
};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::TableWriter;
use ailake_store::LocalStore;
use tempfile::TempDir;

fn make_policy(dim: u32) -> VectorStoragePolicy {
    VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
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
    }
}

fn make_table_props(policy: VectorStoragePolicy) -> TableProperties {
    TableProperties {
        policy,
        format_version: 2,
        extra: Default::default(),
        partition_column_type: None,
    }
}

fn fake_file(path: &str) -> DataFileEntry {
    DataFileEntry {
        path: path.to_string(),
        record_count: 30,
        file_size_bytes: 4096,
        centroid_b64: None,
        radius: None,
        hnsw_offset: None,
        hnsw_len: None,
        vector_column: Some("embedding".to_string()),
        vector_dim: Some(16),
        extra_vector_indexes: vec![],
        index_status: IndexStatus::Ready,
        index_error: None,
        batch_id: None,
        embedding_model: None,
        partition_value: None,
        deletion_vector: None,
        first_row_id: None,
    }
}

// ── HadoopCatalog: 8 concurrent tokio tasks each write one batch ──────────────
//
// The tokio::sync::Mutex in HadoopCatalog serializes all commit_snapshot calls
// within the process. Without it, concurrent reads of version-hint.text could
// produce the same version number for two writers → lost update.

#[tokio::test]
async fn hadoop_8_concurrent_appends_no_lost_update() {
    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "concurrent_test");
    let policy = make_policy(16);

    catalog
        .create_table(&table, &make_table_props(policy.clone()))
        .await
        .unwrap();

    const TASKS: usize = 8;
    let mut handles = Vec::with_capacity(TASKS);
    for i in 0..TASKS {
        let catalog2 = Arc::clone(&catalog);
        let store2 = Arc::clone(&store);
        let table2 = table.clone();
        let policy2 = policy.clone();
        handles.push(tokio::spawn(async move {
            let (batch, embeddings) = fixtures::generate_batch(50, 16);
            let mut writer =
                TableWriter::create_or_open(catalog2, store2, policy2, table2, i as u8 + 1)
                    .await
                    .unwrap();
            writer.write_batch(&batch, &embeddings).await.unwrap();
            writer.commit().await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let files = catalog.list_files(&table, None).await.unwrap();
    assert_eq!(
        files.len(),
        TASKS,
        "expected {TASKS} data files after {TASKS} concurrent appends, got {}",
        files.len()
    );
}

// ── HadoopCatalog: Overwrite tasks race with Append tasks ────────────────────
//
// Simulates MemoryDecayJob (Overwrite) racing with TableWriter (Append).
// The Mutex ensures both operations run atomically — no corrupted metadata.

#[tokio::test]
async fn hadoop_overwrite_and_append_no_corruption() {
    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "race_test");
    let policy = make_policy(16);

    catalog
        .create_table(&table, &make_table_props(policy.clone()))
        .await
        .unwrap();

    // Seed one file so Overwrite has a non-empty snapshot to build on
    {
        let (batch, embeddings) = fixtures::generate_batch(20, 16);
        let mut writer = TableWriter::create_or_open(
            Arc::clone(&catalog) as Arc<dyn CatalogProvider>,
            Arc::clone(&store),
            policy.clone(),
            table.clone(),
            1,
        )
        .await
        .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    }

    let mut handles = Vec::new();

    // 4 Append tasks via TableWriter
    for i in 0..4usize {
        let c = Arc::clone(&catalog) as Arc<dyn CatalogProvider>;
        let s = Arc::clone(&store);
        let t = table.clone();
        let p = policy.clone();
        handles.push(tokio::spawn(async move {
            let (batch, embeddings) = fixtures::generate_batch(10, 16);
            let mut writer = TableWriter::create_or_open(c, s, p, t, 10 + i as u8)
                .await
                .unwrap();
            writer.write_batch(&batch, &embeddings).await.unwrap();
            writer.commit().await.unwrap();
        }));
    }

    // 2 Overwrite tasks: call commit_snapshot directly with a synthetic file entry.
    // Represents decay-job style full-replace commits (no actual parquet file needed —
    // manifest records the path; store.get on missing path is only called on search).
    for j in 0..2usize {
        let c = Arc::clone(&catalog) as Arc<dyn CatalogProvider>;
        let t = table.clone();
        handles.push(tokio::spawn(async move {
            let snap_id = new_snapshot_id();
            let snap = NewSnapshot {
                snapshot_id: snap_id,
                parent_snapshot_id: None,
                files: vec![fake_file(&format!(
                    "warehouse/default/race_test/data/decay-{j}.parquet"
                ))],
                operation: SnapshotOperation::Overwrite,
                iceberg_schema: None,
                extra_properties: HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            };
            c.commit_snapshot(&t, snap).await.unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // Metadata must be in valid state regardless of ordering
    let meta = catalog.load_table(&table).await.unwrap();
    assert!(
        meta.current_snapshot_id.is_some(),
        "table must have a current snapshot after concurrent Append+Overwrite"
    );
}

// ── JdbcCatalog: CAS retry under SQLite concurrent writers ───────────────────
//
// Each task calls commit_snapshot directly and returns the snap_id it committed.
// After all tasks finish, we verify every snap_id is individually findable —
// proving no commit was silently lost to a concurrent writer overwriting the
// metadata_location pointer.
//
// Note: JdbcCatalog uses a simple per-snapshot JSON manifest (no cross-snapshot
// accumulation), so list_files(None) returns only the current snapshot's files.
// To verify all N commits persisted we query each snap_id individually.

#[tokio::test]
async fn jdbc_4_concurrent_commits_no_lost_update() {
    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let db_path = dir.path().join("catalog.db");
    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

    let catalog: Arc<dyn CatalogProvider> = Arc::new(
        JdbcCatalog::connect(&db_url, "test_catalog", "warehouse", Arc::clone(&store))
            .await
            .unwrap(),
    );
    let table = TableIdent::new("default", "jdbc_concurrent");
    let policy = make_policy(16);

    catalog
        .create_table(&table, &make_table_props(policy.clone()))
        .await
        .unwrap();

    const TASKS: usize = 4;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<i64>(TASKS);
    let mut handles = Vec::with_capacity(TASKS);

    for i in 0..TASKS {
        let catalog2 = Arc::clone(&catalog);
        let table2 = table.clone();
        let tx2 = tx.clone();
        handles.push(tokio::spawn(async move {
            let snap_id = new_snapshot_id();
            let snap = NewSnapshot {
                snapshot_id: snap_id,
                parent_snapshot_id: None,
                files: vec![fake_file(&format!(
                    "warehouse/default/jdbc_concurrent/data/file-{i}.parquet"
                ))],
                operation: SnapshotOperation::Append,
                iceberg_schema: None,
                extra_properties: HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            };
            catalog2.commit_snapshot(&table2, snap).await.unwrap();
            tx2.send(snap_id).await.unwrap();
        }));
    }
    drop(tx);
    for h in handles {
        h.await.expect("JDBC concurrent writer task panicked");
    }

    let mut snap_ids: Vec<i64> = Vec::with_capacity(TASKS);
    while let Some(id) = rx.recv().await {
        snap_ids.push(id);
    }
    assert_eq!(
        snap_ids.len(),
        TASKS,
        "expected {TASKS} committed snapshots, channel received {}",
        snap_ids.len()
    );

    // Each snap_id must be individually findable — a concurrent commit must not corrupt
    // or drop a sibling commit's manifest pointer.
    for &snap_id in &snap_ids {
        catalog
            .list_files(&table, Some(snap_id))
            .await
            .unwrap_or_else(|e| panic!("snap {snap_id} not findable after commit: {e}"));
    }

    // The real "no lost update" check: Append inherits the previous snapshot's file list
    // (see `JdbcCatalog::commit_snapshot`), so the *current* state after all 4 concurrent
    // Appends land must contain all 4 files — not just whichever commit's manifest happened
    // to write last. (A per-snapshot manifest legitimately holds fewer files the earlier it
    // was committed in the retry race — that's expected, not a lost update.)
    let current_files = catalog.list_files(&table, None).await.unwrap();
    assert_eq!(
        current_files.len(),
        TASKS,
        "expected {TASKS} files in the current snapshot after {TASKS} concurrent appends, got {}",
        current_files.len()
    );
    let mut paths: Vec<&str> = current_files.iter().map(|f| f.path.as_str()).collect();
    paths.sort();
    paths.dedup();
    assert_eq!(
        paths.len(),
        TASKS,
        "expected {TASKS} distinct file paths, got {} (duplicate or overwritten entry?)",
        paths.len()
    );
}
