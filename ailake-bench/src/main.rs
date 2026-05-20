//! AI-Lake public benchmark — SIFT-1M
//!
//! Measures write throughput, search QPS, and recall@10 on real ANN data.
//!
//! Usage:
//!   ailake-bench --dataset-dir /data/sift1m [--top-k 10] [--ef 50] [--shard-size 100000]
//!
//! Download the dataset first:
//!   ./scripts/download_sift1m.sh /data/sift1m

mod dataset;
mod metrics;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{SearchConfig, SearchSession, TableWriter};
use ailake_store::{LocalStore, Store};

#[derive(Parser, Debug)]
#[command(
    name = "ailake-bench",
    about = "AI-Lake public benchmark on SIFT-1M (128-dim Euclidean)"
)]
struct Args {
    /// Path to directory containing sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
    #[arg(long)]
    dataset_dir: PathBuf,

    /// Directory where the AI-Lake table will be written (default: system temp)
    #[arg(long)]
    table_dir: Option<PathBuf>,

    /// Vectors per shard file
    #[arg(long, default_value_t = 100_000)]
    shard_size: usize,

    /// Nearest neighbors to retrieve (top-k)
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// HNSW ef_search parameter
    #[arg(long, default_value_t = 50)]
    ef: usize,

    /// Truncate base set (useful for smoke-tests, e.g. --limit 10000)
    #[arg(long)]
    limit: Option<usize>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Load dataset ──────────────────────────────────────────────────────────
    let ds = dataset::load(&args.dataset_dir, args.limit)?;

    // ── Set up AI-Lake table ──────────────────────────────────────────────────
    let tmp;
    let table_root = match &args.table_dir {
        Some(p) => p.clone(),
        None => {
            tmp = tempfile::TempDir::new().context("create temp dir")?;
            tmp.path().to_path_buf()
        }
    };

    let store: Arc<dyn Store> = Arc::new(LocalStore::new(&table_root));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
    let table = TableIdent::new("default", "sift1m");

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim: ds.dim as u32,
        metric: VectorMetric::Euclidean,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
    };

    // ── Write phase ───────────────────────────────────────────────────────────
    eprintln!("\nWrite phase …");
    let write_start = Instant::now();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let total_base = ds.base.len();
    let shard_size = args.shard_size;
    let num_shards = total_base.div_ceil(shard_size);

    // Track shard base offsets so recall can map local RowId → global id
    let mut shard_offsets: Vec<u64> = Vec::with_capacity(num_shards);

    let mut writer = TableWriter::create_or_open(
        catalog.clone(),
        store.clone(),
        policy.clone(),
        table.clone(),
    )
    .await
    .context("create table")?;

    for shard_idx in 0..num_shards {
        let base_offset = shard_idx * shard_size;
        let end = (base_offset + shard_size).min(total_base);
        let shard_vecs = &ds.base[base_offset..end];
        let n = shard_vecs.len();

        let ids: Vec<i64> = (base_offset as i64..(base_offset + n) as i64).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids))],
        )?;

        writer
            .write_batch(&batch, shard_vecs)
            .await
            .with_context(|| format!("write shard {shard_idx}"))?;

        shard_offsets.push(base_offset as u64);

        eprint!(
            "\r  shard {}/{} ({} vectors)",
            shard_idx + 1,
            num_shards,
            end
        );
    }
    eprintln!();

    let snapshot_id = writer.commit().await.context("commit")?;
    let write_elapsed = write_start.elapsed();
    let write_throughput = total_base as f64 / write_elapsed.as_secs_f64();

    eprintln!(
        "  committed snapshot {}  ({:.1}s  {:.0} vec/s)",
        snapshot_id,
        write_elapsed.as_secs_f64(),
        write_throughput,
    );

    // ── Search phase ──────────────────────────────────────────────────────────
    eprintln!("\nSearch phase (top_k={} ef={}) …", args.top_k, args.ef);

    let search_cfg = SearchConfig {
        top_k: args.top_k,
        ef_search: args.ef,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
    };

    eprintln!("  Loading indexes into memory …");
    let load_start = Instant::now();
    let session = SearchSession::load(
        &table,
        "embedding",
        ds.dim as u32,
        catalog.clone(),
        store.clone(),
        false,
    )
    .await
    .context("load search session")?;
    let load_elapsed = load_start.elapsed();
    eprintln!(
        "  Loaded {} shard(s) in {:.2}s",
        session.shard_count(),
        load_elapsed.as_secs_f64()
    );

    let num_queries = ds.queries.len();
    let mut latencies_us: Vec<u64> = Vec::with_capacity(num_queries);
    let mut recall_sum = 0.0f64;

    let search_wall_start = Instant::now();

    for (qi, query) in ds.queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = session.search_query(query, &search_cfg);
        let elapsed_us = t0.elapsed().as_micros() as u64;
        latencies_us.push(elapsed_us);

        // Convert SearchResult → global IDs
        let result_ids: Vec<u32> = results
            .iter()
            .map(|r| {
                let part_num = parse_part_num(&r.file_path);
                (part_num as u64 * shard_size as u64 + r.row_id.as_u64()) as u32
            })
            .collect();

        let recall = metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], args.top_k);
        recall_sum += recall;

        if (qi + 1) % 1000 == 0 {
            eprint!("\r  {}/{} queries …", qi + 1, num_queries);
        }
    }
    eprintln!("\r  {num_queries}/{num_queries} queries done");

    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;
    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);
    let mean_recall = recall_sum / num_queries as f64;

    // ── Report ────────────────────────────────────────────────────────────────
    println!();
    println!("AI-Lake Benchmark — SIFT-1M ({}D, Euclidean, F16)", ds.dim);
    println!("{}", "=".repeat(58));
    println!(
        "Dataset    {:>10} base  |  {:>6} queries  |  {} GT neighbors",
        fmt_int(total_base),
        fmt_int(num_queries),
        ds.ground_truth.first().map(|v| v.len()).unwrap_or(0),
    );
    println!();
    println!("Write phase");
    println!("  Shards        : {} × {} vectors", num_shards, shard_size);
    println!("  Wall time     : {:.1} s", write_elapsed.as_secs_f64());
    println!("  Throughput    : {:.0} vec/s", write_throughput);
    println!();
    println!("Index load");
    println!("  Shards loaded : {}", session.shard_count());
    println!("  Load time     : {:.2} s", load_elapsed.as_secs_f64());
    println!();
    println!("Search phase  (top_k={}, ef={})  [indexes pre-loaded]", args.top_k, args.ef);
    println!("  Recall@{}     : {:.4}", args.top_k, mean_recall);
    println!("  QPS           : {:.0}", lat.qps);
    println!("  Latency mean  : {:.3} ms", lat.mean_ms);
    println!("  Latency p50   : {:.3} ms", lat.p50_ms);
    println!("  Latency p95   : {:.3} ms", lat.p95_ms);
    println!("  Latency p99   : {:.3} ms", lat.p99_ms);
    println!();

    Ok(())
}

/// Extract the part number from a path like "data/part-00007.parquet" → 7.
fn parse_part_num(file_path: &str) -> usize {
    file_path
        .rsplit('/')
        .next()
        .and_then(|name| name.strip_prefix("part-"))
        .and_then(|rest| rest.strip_suffix(".parquet"))
        .and_then(|num_str| num_str.parse().ok())
        .unwrap_or(0)
}

fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}
