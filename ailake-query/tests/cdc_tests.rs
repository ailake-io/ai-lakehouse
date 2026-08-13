// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for AI-Lake CDC (`read_changes`).

use std::sync::Arc;

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{delete_where, read_changes, ChangeReaderConfig, TableWriter};
use ailake_store::LocalStore;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

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
        partition_column_type: None,
        partition_fields: vec![],
    }
}

fn make_batch(ids: Vec<i32>, texts: Vec<&str>, _embeddings: Vec<Vec<f32>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)) as Arc<dyn arrow_array::Array>,
            Arc::new(StringArray::from(texts)) as Arc<dyn arrow_array::Array>,
        ],
    )
    .unwrap()
}

async fn setup_table() -> (
    Arc<dyn ailake_catalog::CatalogProvider>,
    Arc<dyn ailake_store::Store>,
    TableIdent,
) {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog: Arc<dyn ailake_catalog::CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));
    let ident = TableIdent::new("default", "cdc_docs");

    catalog
        .create_table(
            &ident,
            &ailake_catalog::TableProperties {
                policy: make_policy(),
                extra: Default::default(),
                format_version: 2,
                partition_column_type: None,
            },
        )
        .await
        .unwrap();

    (catalog, store, ident)
}

async fn current_snapshot_id(
    catalog: Arc<dyn ailake_catalog::CatalogProvider>,
    ident: &TableIdent,
) -> i64 {
    catalog
        .load_table(ident)
        .await
        .unwrap()
        .current_snapshot_id
        .unwrap()
}

#[tokio::test]
async fn cdc_insert_between_snapshots() {
    let (catalog, store, ident) = setup_table().await;

    // Snapshot 1: insert two rows.
    let mut w1 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch1 = make_batch(
        vec![1, 2],
        vec!["hello", "world"],
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
    );
    w1.write_batch(&batch1, &batch1_to_embeddings(&batch1))
        .await
        .unwrap();
    w1.commit().await.unwrap();
    let snap1 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    // Snapshot 2: insert one more row.
    let mut w2 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch2 = make_batch(vec![3], vec!["foo"], vec![vec![0.0, 0.0, 1.0, 0.0]]);
    w2.write_batch(&batch2, &batch2_to_embeddings(&batch2))
        .await
        .unwrap();
    w2.commit().await.unwrap();
    let snap2 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    let batch = read_changes(
        catalog,
        store,
        &ident,
        ChangeReaderConfig {
            start_snapshot_id: Some(snap1),
            end_snapshot_id: Some(snap2),
            pk_columns: vec!["id".to_string()],
            coalesce_updates: false,
        },
    )
    .await
    .unwrap();

    assert_eq!(batch.num_rows(), 1, "expected one inserted row");
    let change_col = batch
        .column_by_name("_change_type")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(change_col.value(0), "insert");
    let id_col = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(id_col.value(0), 3);
}

#[tokio::test]
async fn cdc_delete_via_equality_delete() {
    let (catalog, store, ident) = setup_table().await;

    // Snapshot 1: insert rows.
    let mut w1 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch1 = make_batch(
        vec![1, 2],
        vec!["hello", "world"],
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
    );
    w1.write_batch(&batch1, &batch1_to_embeddings(&batch1))
        .await
        .unwrap();
    w1.commit().await.unwrap();
    let snap1 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    // Snapshot 2: equality-delete row id=1.
    delete_where(
        Arc::clone(&catalog),
        Arc::clone(&store),
        &ident,
        "id",
        &["1"],
    )
    .await
    .unwrap();
    let snap2 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    let batch = read_changes(
        catalog,
        store,
        &ident,
        ChangeReaderConfig {
            start_snapshot_id: Some(snap1),
            end_snapshot_id: Some(snap2),
            pk_columns: vec!["id".to_string()],
            coalesce_updates: false,
        },
    )
    .await
    .unwrap();

    assert_eq!(batch.num_rows(), 1, "expected one delete record");
    let change_col = batch
        .column_by_name("_change_type")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(change_col.value(0), "delete");
}

#[tokio::test]
async fn cdc_no_changes_between_same_snapshot() {
    let (catalog, store, ident) = setup_table().await;

    let mut w1 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch1 = make_batch(vec![1], vec!["hello"], vec![vec![1.0, 0.0, 0.0, 0.0]]);
    w1.write_batch(&batch1, &batch1_to_embeddings(&batch1))
        .await
        .unwrap();
    w1.commit().await.unwrap();
    let snap1 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    let batch = read_changes(
        catalog,
        store,
        &ident,
        ChangeReaderConfig {
            start_snapshot_id: Some(snap1),
            end_snapshot_id: Some(snap1),
            pk_columns: vec![],
            coalesce_updates: false,
        },
    )
    .await
    .unwrap();

    assert_eq!(batch.num_rows(), 0);
}

#[tokio::test]
async fn cdc_coalesce_update() {
    let (catalog, store, ident) = setup_table().await;

    // Snapshot 1: insert row id=1.
    let mut w1 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch1 = make_batch(vec![1], vec!["v1"], vec![vec![1.0, 0.0, 0.0, 0.0]]);
    w1.write_batch(&batch1, &batch1_to_embeddings(&batch1))
        .await
        .unwrap();
    w1.commit().await.unwrap();
    let snap1 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    // Snapshot 2: delete id=1 and insert id=1 with new text (simulated update).
    delete_where(
        Arc::clone(&catalog),
        Arc::clone(&store),
        &ident,
        "id",
        &["1"],
    )
    .await
    .unwrap();

    let mut w2 = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        make_policy(),
        ident.clone(),
        2,
    )
    .await
    .unwrap();
    let batch2 = make_batch(vec![1], vec!["v2"], vec![vec![0.0, 1.0, 0.0, 0.0]]);
    w2.write_batch(&batch2, &batch2_to_embeddings(&batch2))
        .await
        .unwrap();
    w2.commit().await.unwrap();
    let snap2 = current_snapshot_id(Arc::clone(&catalog), &ident).await;

    let batch = read_changes(
        Arc::clone(&catalog),
        Arc::clone(&store),
        &ident,
        ChangeReaderConfig {
            start_snapshot_id: Some(snap1),
            end_snapshot_id: Some(snap2),
            pk_columns: vec!["id".to_string()],
            coalesce_updates: true,
        },
    )
    .await
    .unwrap();

    let change_col = batch
        .column_by_name("_change_type")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(change_col.len(), 2);
    assert_eq!(change_col.value(0), "update_before");
    assert_eq!(change_col.value(1), "update_after");

    let text_col = batch
        .column_by_name("text")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        text_col.value(0),
        "v1",
        "UPDATE_BEFORE must carry the old row"
    );
    assert_eq!(text_col.value(1), "v2");
}

fn batch1_to_embeddings(batch: &RecordBatch) -> Vec<Vec<f32>> {
    // Helper: deterministic embeddings based on row count.
    let n = batch.num_rows();
    (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            v
        })
        .collect()
}

fn batch2_to_embeddings(batch: &RecordBatch) -> Vec<Vec<f32>> {
    batch1_to_embeddings(batch)
}
