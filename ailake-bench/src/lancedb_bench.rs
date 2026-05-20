use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use aa58::{FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
use as58::{DataType, Field, Schema};
use futures::{StreamExt, TryStreamExt};
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::DistanceType;

use crate::bench_result::BenchResult;
use crate::dataset::Dataset;
use crate::metrics;

const WRITE_BATCH: usize = 100_000;

pub async fn run(
    ds: &Dataset,
    top_k: usize,
    nprobes: usize,
    num_partitions: u32,
    ef_construction: u32,
    concurrency: usize,
) -> anyhow::Result<BenchResult> {
    let tmp = tempfile::TempDir::new().context("create temp dir")?;
    let uri = tmp.path().to_str().expect("valid utf8").to_string();
    let dim = ds.dim as i32;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    // ── Write phase ───────────────────────────────────────────────────────────
    eprintln!("\nLanceDB write phase …");
    let write_start = Instant::now();

    let db = lancedb::connect(&uri).execute().await?;
    let total = ds.base.len();
    let batches = make_batches(&ds.base, schema.clone(), WRITE_BATCH);
    let reader: Box<dyn aa58::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone()));
    let table = db
        .create_table("sift1m", reader)
        .execute()
        .await
        .context("create lancedb table")?;

    let write_elapsed = write_start.elapsed();
    let write_vec_per_sec = total as f64 / write_elapsed.as_secs_f64();
    eprintln!(
        "  wrote {} vectors in {:.1}s  ({:.0} vec/s)",
        total,
        write_elapsed.as_secs_f64(),
        write_vec_per_sec,
    );

    // ── Index build ───────────────────────────────────────────────────────────
    eprintln!("\nLanceDB index build (IVF-HNSW-SQ, partitions={num_partitions}, ef_construction={ef_construction}) …");
    let index_start = Instant::now();

    table
        .create_index(
            &["vector"],
            Index::IvfHnswSq(
                IvfHnswSqIndexBuilder::default()
                    .distance_type(DistanceType::L2)
                    .num_partitions(num_partitions)
                    .num_edges(16)
                    .ef_construction(ef_construction),
            ),
        )
        .execute()
        .await
        .context("create ivf-hnsw-sq index")?;

    let index_elapsed = index_start.elapsed();
    eprintln!("  index built in {:.1}s", index_elapsed.as_secs_f64());

    // ── Warm-up (first query, measure connection/cache overhead separately) ───
    let load_start = Instant::now();
    let _ = table
        .query()
        .nearest_to(ds.queries[0].as_slice())?
        .distance_type(DistanceType::L2)
        .limit(1)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let load_elapsed = load_start.elapsed();

    // ── Search phase ──────────────────────────────────────────────────────────
    eprintln!("\nLanceDB search phase (top_k={top_k}, nprobes={nprobes}, concurrency={concurrency}) …");

    let num_queries = ds.queries.len();
    let done_count = Arc::new(AtomicUsize::new(0));
    let search_wall_start = Instant::now();

    let query_results: Vec<anyhow::Result<(usize, u64, Vec<u32>)>> =
        futures::stream::iter(ds.queries.iter().enumerate())
            .map(|(qi, query)| {
                let table = table.clone();
                let query_vec = query.clone();
                let done_count = done_count.clone();
                async move {
                    let t0 = Instant::now();
                    let result_batches: Vec<RecordBatch> = table
                        .query()
                        .nearest_to(query_vec.as_slice())?
                        .distance_type(DistanceType::L2)
                        .limit(top_k)
                        .nprobes(nprobes)
                        .execute()
                        .await?
                        .try_collect()
                        .await?;
                    let latency_us = t0.elapsed().as_micros() as u64;
                    let result_ids = collect_ids(&result_batches);
                    let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 1000 == 0 {
                        eprint!("\r  {done}/{num_queries} queries …");
                    }
                    anyhow::Ok((qi, latency_us, result_ids))
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

    eprintln!("\r  {num_queries}/{num_queries} queries done");
    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;

    let mut latencies_us: Vec<u64> = vec![0u64; num_queries];
    let mut recall_sum = 0.0f64;
    for result in query_results {
        let (qi, latency_us, result_ids) = result?;
        latencies_us[qi] = latency_us;
        recall_sum += metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], top_k);
    }

    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);

    Ok(BenchResult {
        engine: "LanceDB 0.29 (IVF-HNSW-SQ)".to_string(),
        write_secs: write_elapsed.as_secs_f64(),
        write_vec_per_sec,
        index_build_secs: index_elapsed.as_secs_f64(),
        load_secs: load_elapsed.as_secs_f64(),
        recall: recall_sum / num_queries as f64,
        qps: lat.qps,
        mean_ms: lat.mean_ms,
        p50_ms: lat.p50_ms,
        p95_ms: lat.p95_ms,
        p99_ms: lat.p99_ms,
    })
}

fn make_batches(vecs: &[Vec<f32>], schema: Arc<Schema>, batch_size: usize) -> Vec<RecordBatch> {
    let dim = vecs.first().map(|v| v.len()).unwrap_or(0) as i32;
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let mut batches = Vec::new();

    for (chunk_idx, chunk) in vecs.chunks(batch_size).enumerate() {
        let base = chunk_idx * batch_size;
        let ids: Vec<i64> = (base as i64..(base + chunk.len()) as i64).collect();

        let flat: Vec<f32> = chunk.iter().flat_map(|v| v.iter().cloned()).collect();
        let values = Arc::new(Float32Array::from(flat));
        let vector_col =
            Arc::new(FixedSizeListArray::new(item_field.clone(), dim, values, None)) as _;

        batches.push(
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(ids)) as _, vector_col],
            )
            .expect("valid batch"),
        );
    }
    batches
}

fn collect_ids(batches: &[RecordBatch]) -> Vec<u32> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .map(|a| a.values().iter().map(|&v| v as u32).collect::<Vec<_>>())
                .unwrap_or_default()
        })
        .collect()
}
