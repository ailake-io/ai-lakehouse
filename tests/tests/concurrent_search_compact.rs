// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stress test: concurrent search() and compaction on the same table.
//!
//! Verifies that:
//! - Search never panics or returns corrupt results while compaction runs.
//! - Compaction never corrupts the table metadata.
//! - The final table is readable and returns expected results.

mod fixtures;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent, TableProperties};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::compaction::{CompactionConfig, CompactionPlanner, CompactionExecutor};
use ailake_query::scanner::{search, SearchConfig};
use ailake_store::{LocalStore, Store};
use tempfile::TempDir;
use tokio::time::sleep;

fn make_policy(dim: u32) -> VectorStoragePolicy {
    VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
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

/// Stress test: 1 compaction task + 4 search tasks running concurrently.
///
/// Compaction: runs `compact()` on all current files, commits via Replace,
/// then repeats — up to 5 passes.
///
/// Search: each search task runs `search()` in a loop with random-ish queries,
/// counts successes and tracks errors.
#[tokio::test]
async fn concurrent_search_and_compact() {
    let dir = TempDir::new().unwrap();
    let dim = 8u32;
    let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "stress_test");
    let policy = make_policy(dim);

    catalog
        .create_table(
            &table,
            &TableProperties {
                policy: policy.clone(),
                format_version: 2,
                extra: Default::default(),
                partition_column_type: None,
            },
        )
        .await
        .unwrap();

    // Seed 4 small data files (each with distinct unit-basis embeddings)
    for i in 0..4 {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int32, false),
        ]));
        let ids: Vec<i32> = (0..5).map(|j| (i * 5 + j) as i32).collect();
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::Int32Array::from(ids))],
        )
        .unwrap();
        let embeddings: Vec<Vec<f32>> = (0..5)
            .map(|j| {
                let mut v = vec![0.0f32; dim as usize];
                v[(i * 2 + j as usize) % dim as usize] = 1.0;
                v
            })
            .collect();

        let mut writer = ailake_query::TableWriter::create_or_open(
            Arc::clone(&catalog),
            Arc::clone(&store),
            policy.clone(),
            table.clone(),
            (i + 1) as u8,
        )
        .await
        .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    }

    // Verify initial state
    let initial_files = catalog.list_files(&table, None).await.unwrap();
    assert_eq!(
        initial_files.len(),
        4,
        "expected 4 seed data files, got {}",
        initial_files.len()
    );

    // Shared state for coordinating tasks
    let stop_flag = Arc::new(AtomicBool::new(false));
    let search_errors = Arc::new(AtomicUsize::new(0));
    let compact_passes = Arc::new(AtomicUsize::new(0));
    let search_attempts = Arc::new(AtomicUsize::new(0));

    // ── Compaction task ──
    let comp_store = Arc::clone(&store);
    let comp_cat: Arc<dyn CatalogProvider> = Arc::clone(&catalog);
    let comp_table = table.clone();
    let comp_policy = policy.clone();
    let comp_stop = Arc::clone(&stop_flag);
    let comp_passes = Arc::clone(&compact_passes);
    let comp_handle = tokio::spawn(async move {
        let config = CompactionConfig {
            min_files_to_compact: 2,
            target_file_size_bytes: 10 * 1024 * 1024,
            ..Default::default()
        };
        let planner = CompactionPlanner::new(config);

        for _pass in 0..5 {
            if comp_stop.load(Ordering::Acquire) {
                break;
            }

            // Wait a bit for search tasks to start
            sleep(Duration::from_millis(50)).await;

            let files = comp_cat.list_files(&comp_table, None).await;
            if files.is_err() {
                continue;
            }
            let files = files.unwrap();
            let to_compact = planner.plan(&files);
            if to_compact.is_empty() {
                continue;
            }

            let executor = CompactionExecutor::new(Arc::clone(&comp_store), comp_policy.clone());
            let merged = executor.compact(&to_compact, "data/merged.parquet").await;
            if merged.is_err() {
                continue;
            }
            let merged = merged.unwrap();

            // Commit the merge
            let meta = comp_cat.load_table(&comp_table).await;
            if meta.is_err() {
                continue;
            }
            let meta = meta.unwrap();

            let merged_entry = ailake_catalog::DataFileEntry {
                path: merged.path.clone(),
                record_count: merged.record_count,
                file_size_bytes: merged.file_size_bytes,
                centroid_b64: merged.centroid_b64,
                radius: merged.radius,
                hnsw_offset: merged.hnsw_offset,
                hnsw_len: merged.hnsw_len,
                vector_column: merged.vector_column,
                vector_dim: merged.vector_dim,
                extra_vector_indexes: merged.extra_vector_indexes,
                index_status: merged.index_status,
                index_error: merged.index_error,
                batch_id: merged.batch_id,
                embedding_model: merged.embedding_model,
                partition_value: merged.partition_value,
                deletion_vector: merged.deletion_vector,
                first_row_id: None,
                column_stats: None,
            };

            let result = comp_cat
                .commit_snapshot(
                    &comp_table,
                    ailake_catalog::NewSnapshot {
                        snapshot_id: 1000 + _pass as i64,
                        parent_snapshot_id: meta.current_snapshot_id,
                        files: vec![merged_entry],
                        operation: ailake_catalog::SnapshotOperation::Replace,
                        iceberg_schema: None,
                        extra_properties: std::collections::HashMap::new(),
                        bloom_filters: vec![],
                        equality_delete_files: vec![],
                    },
                )
                .await;
            if result.is_ok() {
                comp_passes.fetch_add(1, Ordering::Release);
            }

            // Brief pause between compaction passes
            sleep(Duration::from_millis(100)).await;
        }
    });

    // ── Search tasks ──
    const NUM_SEARCHERS: usize = 4;
    let mut search_handles = Vec::with_capacity(NUM_SEARCHERS);
    for seeker_id in 0..NUM_SEARCHERS {
        let s_store = Arc::clone(&store);
        let s_cat: Arc<dyn CatalogProvider> = Arc::clone(&catalog);
        let s_table = table.clone();
        let s_stop = Arc::clone(&stop_flag);
        let s_errors = Arc::clone(&search_errors);
        let s_attempts = Arc::clone(&search_attempts);

        search_handles.push(tokio::spawn(async move {
            for _iter in 0..10 {
                if s_stop.load(Ordering::Acquire) {
                    break;
                }

                // Use the seeker_id to generate a deterministic-ish query
                let query_idx = seeker_id % dim as usize;
                let mut query = vec![0.0f32; dim as usize];
                query[query_idx] = 1.0;

                let config = SearchConfig {
                    top_k: 3,
                    ef_search: 50,
                    pruning_threshold: f32::INFINITY,
                    ..Default::default()
                };

                s_attempts.fetch_add(1, Ordering::Release);

                let results = search(
                    &s_table,
                    &query,
                    config,
                    "embedding",
                    dim,
                    Arc::clone(&s_cat),
                    Arc::clone(&s_store),
                )
                .await;

                match results {
                    Ok(results) => {
                        // Results may be empty if compaction removed/replaced files
                        // in an intermediate state — that's acceptable.
                        for r in &results {
                            assert!(
                                r.distance.is_finite(),
                                "non-finite distance returned by search"
                            );
                            assert!(
                                r.distance >= 0.0,
                                "negative distance returned by search"
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "search error (seeker {seeker_id}, iter {_iter}): {e:?}"
                        );
                        s_errors.fetch_add(1, Ordering::Release);
                    }
                }

                sleep(Duration::from_millis(20)).await;
            }
        }));
    }

    // Wait for all search tasks
    for h in search_handles {
        h.await.unwrap();
    }

    // Signal compaction to stop and wait
    stop_flag.store(true, Ordering::Release);
    comp_handle.await.unwrap();

    // ── Final assertions ──
    let attempts = search_attempts.load(Ordering::Acquire);
    let errors = search_errors.load(Ordering::Acquire);
    let passes = compact_passes.load(Ordering::Acquire);

    eprintln!(
        "concurrent_search_and_compact: \
         search_attempts={attempts}, search_errors={errors}, compact_passes={passes}"
    );

    // Search should have a high success rate — allow some errors due to
    // transient catalog states during compaction (file replaced mid-read).
    assert!(
        errors < attempts / 2 || attempts == 0,
        "search error rate too high: {errors}/{attempts}"
    );

    // Final table must be readable
    let final_files = catalog.list_files(&table, None).await.unwrap();
    assert!(
        !final_files.is_empty(),
        "final table must have at least one data file"
    );

    // Final search must succeed
    let final_query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let final_config = SearchConfig {
        top_k: 5,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        ..Default::default()
    };
    let final_results = search(
        &table,
        &final_query,
        final_config,
        "embedding",
        dim,
        catalog,
        store,
    )
    .await;
    assert!(final_results.is_ok(), "final search after stress test failed");
    let final_results = final_results.unwrap();
    assert!(
        !final_results.is_empty(),
        "final search returned no results"
    );
}
