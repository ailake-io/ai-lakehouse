//! AI-Lake public benchmark — SIFT-1M
//!
//! Measures write throughput, search QPS, and recall@10 on real ANN data.
//!
//! Usage:
//!   ailake-bench --dataset-dir /data/sift1m [--engine ailake|ailake-ivf-pq|lancedb|pgvector|all]
//!
//! LanceDB comparison requires: --features lancedb-bench
//! pgvector comparison requires: --features pgvector-bench  +  --pgvector-url <conn>
//!
//! Download the dataset first:
//!   ./scripts/download_sift1m.sh /data/sift1m

mod bench_result;
mod dataset;
mod metrics;

#[cfg(feature = "lancedb-bench")]
mod lancedb_bench;

#[cfg(feature = "pgvector-bench")]
mod pgvector_bench;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;

use ailake_catalog::{CatalogProvider, HadoopCatalog, IndexStatus, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_index::HardwareProfile;
use ailake_query::{IvfPqConfig, SearchConfig, SearchSession, TableWriter};
use ailake_store::{LocalStore, Store};

use bench_result::BenchResult;

#[derive(clap::ValueEnum, Clone, Debug, Default)]
enum Engine {
    #[default]
    Ailake,
    /// AI-Lake with IVF-PQ index (smaller index, sequential-scan, better S3 throughput)
    AilakeIvfPq,
    /// AI-Lake with auto-selected index (detects GPU / CPU cores at runtime)
    AilakeAuto,
    Lancedb,
    Pgvector,
    All,
}

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

    /// Which engine(s) to benchmark
    #[arg(long, default_value = "ailake")]
    engine: Engine,

    /// Vectors per AI-Lake shard file
    #[arg(long, default_value_t = 100_000)]
    shard_size: usize,

    /// Nearest neighbors to retrieve (top-k)
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// AI-Lake HNSW ef_search parameter
    #[arg(long, default_value_t = 50)]
    ef: usize,

    /// Truncate base set (useful for smoke-tests, e.g. --limit 10000)
    #[arg(long)]
    limit: Option<usize>,

    /// IVF-PQ coarse clusters (nlist)
    #[arg(long, default_value_t = 256)]
    ivf_nlist: usize,

    /// IVF-PQ clusters probed per query (nprobe)
    #[arg(long, default_value_t = 8)]
    ivf_nprobe: usize,

    /// IVF-PQ sub-vectors (pq_m, must divide 128 for SIFT)
    #[arg(long, default_value_t = 8)]
    ivf_pq_m: usize,

    /// LanceDB IVF nprobes during search (requires --features lancedb-bench)
    #[arg(long, default_value_t = 20)]
    lancedb_nprobes: usize,

    /// LanceDB IVF number of partitions (requires --features lancedb-bench)
    #[arg(long, default_value_t = 256)]
    lancedb_partitions: u32,

    /// LanceDB HNSW ef_construction during index build (requires --features lancedb-bench)
    #[arg(long, default_value_t = 100)]
    lancedb_ef_construction: u32,

    /// Number of concurrent LanceDB search queries (requires --features lancedb-bench)
    #[arg(long, default_value_t = 32)]
    lancedb_concurrency: usize,

    /// PostgreSQL connection string for pgvector benchmark (requires --features pgvector-bench)
    /// Example: "host=localhost user=postgres password=postgres dbname=postgres"
    #[arg(long)]
    pgvector_url: Option<String>,

    /// pgvector HNSW m parameter (requires --features pgvector-bench)
    #[arg(long, default_value_t = 16)]
    pgvector_m: u32,

    /// pgvector HNSW ef_construction parameter (requires --features pgvector-bench)
    #[arg(long, default_value_t = 64)]
    pgvector_ef_construction: u32,

    /// pgvector HNSW ef_search parameter (requires --features pgvector-bench)
    #[arg(long, default_value_t = 50)]
    pgvector_ef_search: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let ds = dataset::load(&args.dataset_dir, args.limit)?;

    match args.engine {
        Engine::Ailake => {
            let r = run_ailake(&args, &ds).await?;
            bench_result::print_single(&r, ds.base.len(), args.top_k);
        }
        Engine::AilakeIvfPq => {
            let r = run_ailake_ivf_pq(&args, &ds).await?;
            bench_result::print_single(&r, ds.base.len(), args.top_k);
        }
        Engine::AilakeAuto => {
            let r = run_ailake_auto(&args, &ds).await?;
            bench_result::print_single(&r, ds.base.len(), args.top_k);
        }
        Engine::Lancedb => {
            #[cfg(not(feature = "lancedb-bench"))]
            anyhow::bail!("--engine lancedb requires recompiling with --features lancedb-bench");
            #[cfg(feature = "lancedb-bench")]
            {
                let r = lancedb_bench::run(
                    &ds,
                    args.top_k,
                    args.lancedb_nprobes,
                    args.lancedb_partitions,
                    args.lancedb_ef_construction,
                    args.lancedb_concurrency,
                )
                .await?;
                bench_result::print_single(&r, ds.base.len(), args.top_k);
            }
        }
        Engine::Pgvector => {
            #[cfg(not(feature = "pgvector-bench"))]
            anyhow::bail!("--engine pgvector requires recompiling with --features pgvector-bench");
            #[cfg(feature = "pgvector-bench")]
            {
                let pg_url = args
                    .pgvector_url
                    .as_deref()
                    .context("--pgvector-url required for --engine pgvector")?;
                let r = pgvector_bench::run(
                    &ds,
                    pg_url,
                    args.top_k,
                    args.pgvector_m,
                    args.pgvector_ef_construction,
                    args.pgvector_ef_search,
                )
                .await?;
                bench_result::print_single(&r, ds.base.len(), args.top_k);
            }
        }
        Engine::All => {
            #[cfg(not(feature = "lancedb-bench"))]
            anyhow::bail!("--engine all requires recompiling with --features lancedb-bench");
            #[cfg(feature = "lancedb-bench")]
            {
                let ailake = run_ailake(&args, &ds).await?;
                let ailake_ivf = run_ailake_ivf_pq(&args, &ds).await?;
                let lancedb = lancedb_bench::run(
                    &ds,
                    args.top_k,
                    args.lancedb_nprobes,
                    args.lancedb_partitions,
                    args.lancedb_ef_construction,
                    args.lancedb_concurrency,
                )
                .await?;

                let results: Vec<BenchResult> = vec![ailake, ailake_ivf, lancedb];

                #[cfg(feature = "pgvector-bench")]
                if let Some(ref pg_url) = args.pgvector_url {
                    let pgvec = pgvector_bench::run(
                        &ds,
                        pg_url,
                        args.top_k,
                        args.pgvector_m,
                        args.pgvector_ef_construction,
                        args.pgvector_ef_search,
                    )
                    .await?;
                    results.push(pgvec);
                }

                let refs: Vec<&BenchResult> = results.iter().collect();
                bench_result::print_multi_comparison(&refs, args.top_k);
            }
        }
    }

    Ok(())
}

async fn run_ailake(args: &Args, ds: &dataset::Dataset) -> anyhow::Result<BenchResult> {
    let tmp;
    let table_root: PathBuf = match &args.table_dir {
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
    eprintln!("\nAI-Lake write phase …");
    let write_start = Instant::now();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let total_base = ds.base.len();
    let shard_size = args.shard_size;
    let num_shards = total_base.div_ceil(shard_size);

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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ids))])?;

        writer
            .write_batch_deferred(&batch, shard_vecs)
            .await
            .with_context(|| format!("write shard {shard_idx}"))?;

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
    let write_vec_per_sec = total_base as f64 / write_elapsed.as_secs_f64();

    eprintln!(
        "  committed snapshot {}  ({:.1}s  {:.0} vec/s) — Parquet-only, HNSW building …",
        snapshot_id,
        write_elapsed.as_secs_f64(),
        write_vec_per_sec,
    );

    // ── Wait for background HNSW builds ──────────────────────────────────────
    eprintln!("  Waiting for {num_shards} background HNSW build(s) …");
    let index_start = Instant::now();
    loop {
        let files = catalog
            .list_files(&table, None)
            .await
            .context("list files")?;
        let ready = files
            .iter()
            .filter(|f| f.index_status == IndexStatus::Ready)
            .count();
        if ready >= num_shards {
            break;
        }
        eprint!("\r  Indexing: {ready}/{num_shards} shards ready …");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    let index_build_elapsed = index_start.elapsed();
    eprintln!(
        "\r  All {num_shards} shards indexed in {:.1}s                    ",
        index_build_elapsed.as_secs_f64()
    );

    // ── Load indexes ──────────────────────────────────────────────────────────
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

    // ── Search phase ──────────────────────────────────────────────────────────
    eprintln!(
        "\nAI-Lake search phase (top_k={}, ef={}) …",
        args.top_k, args.ef
    );

    let search_cfg = SearchConfig {
        top_k: args.top_k,
        ef_search: args.ef,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
    };

    let num_queries = ds.queries.len();
    let mut latencies_us: Vec<u64> = Vec::with_capacity(num_queries);
    let mut recall_sum = 0.0f64;
    let search_wall_start = Instant::now();

    for (qi, query) in ds.queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = session.search_query(query, &search_cfg);
        latencies_us.push(t0.elapsed().as_micros() as u64);

        let result_ids: Vec<u32> = results
            .iter()
            .map(|r| {
                let part_num = parse_part_num(&r.file_path);
                (part_num as u64 * shard_size as u64 + r.row_id.as_u64()) as u32
            })
            .collect();

        recall_sum += metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], args.top_k);

        if (qi + 1) % 1000 == 0 {
            eprint!("\r  {}/{} queries …", qi + 1, num_queries);
        }
    }
    eprintln!("\r  {num_queries}/{num_queries} queries done");

    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;
    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);

    Ok(BenchResult {
        engine: format!("AI-Lake 0.2 ({num_shards} shards, deferred HNSW+F16)"),
        write_secs: write_elapsed.as_secs_f64(),
        write_vec_per_sec,
        index_build_secs: index_build_elapsed.as_secs_f64(),
        load_secs: load_elapsed.as_secs_f64(),
        recall: recall_sum / num_queries as f64,
        qps: lat.qps,
        mean_ms: lat.mean_ms,
        p50_ms: lat.p50_ms,
        p95_ms: lat.p95_ms,
        p99_ms: lat.p99_ms,
    })
}

async fn run_ailake_ivf_pq(args: &Args, ds: &dataset::Dataset) -> anyhow::Result<BenchResult> {
    let tmp;
    let table_root: PathBuf = match &args.table_dir {
        Some(p) => {
            // Use a sub-dir so it doesn't collide with the HNSW bench run
            let mut p = p.clone();
            p.push("ivf_pq");
            p
        }
        None => {
            tmp = tempfile::TempDir::new().context("create temp dir")?;
            tmp.path().to_path_buf()
        }
    };

    let store: Arc<dyn Store> = Arc::new(LocalStore::new(&table_root));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
    let table = TableIdent::new("default", "sift1m_ivf");

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim: ds.dim as u32,
        metric: VectorMetric::Euclidean,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
    };

    let ivf_config = IvfPqConfig {
        nlist: args.ivf_nlist,
        nprobe: args.ivf_nprobe,
        pq_m: args.ivf_pq_m,
        pq_k: 256,
        max_iter: 25,
    };

    eprintln!(
        "\nAI-Lake IVF-PQ write phase (nlist={}, nprobe={}, pq_m={}) …",
        ivf_config.nlist, ivf_config.nprobe, ivf_config.pq_m
    );
    let write_start = Instant::now();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let total_base = ds.base.len();
    let shard_size = args.shard_size;
    let num_shards = total_base.div_ceil(shard_size);

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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ids))])?;

        writer
            .write_batch_ivf_pq(&batch, shard_vecs, ivf_config.clone())
            .await
            .with_context(|| format!("write shard {shard_idx}"))?;

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
    let write_vec_per_sec = total_base as f64 / write_elapsed.as_secs_f64();

    eprintln!(
        "  committed snapshot {}  ({:.1}s  {:.0} vec/s) — IVF-PQ built inline",
        snapshot_id,
        write_elapsed.as_secs_f64(),
        write_vec_per_sec,
    );

    // ── Load indexes ──────────────────────────────────────────────────────────
    eprintln!("  Loading IVF-PQ indexes into memory …");
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

    // ── Search phase ──────────────────────────────────────────────────────────
    eprintln!(
        "\nAI-Lake IVF-PQ search phase (top_k={}, nprobe={}) …",
        args.top_k, ivf_config.nprobe
    );

    let search_cfg = SearchConfig {
        top_k: args.top_k,
        ef_search: args.ef,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
    };

    let num_queries = ds.queries.len();
    let mut latencies_us: Vec<u64> = Vec::with_capacity(num_queries);
    let mut recall_sum = 0.0f64;
    let search_wall_start = Instant::now();

    for (qi, query) in ds.queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = session.search_query(query, &search_cfg);
        latencies_us.push(t0.elapsed().as_micros() as u64);

        let result_ids: Vec<u32> = results
            .iter()
            .map(|r| {
                let part_num = parse_part_num(&r.file_path);
                (part_num as u64 * shard_size as u64 + r.row_id.as_u64()) as u32
            })
            .collect();

        recall_sum += metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], args.top_k);

        if (qi + 1) % 1000 == 0 {
            eprint!("\r  {}/{} queries …", qi + 1, num_queries);
        }
    }
    eprintln!("\r  {num_queries}/{num_queries} queries done");

    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;
    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);

    Ok(BenchResult {
        engine: format!(
            "AI-Lake IVF-PQ ({num_shards} shards, nlist={}, nprobe={}, pq_m={})",
            ivf_config.nlist, ivf_config.nprobe, ivf_config.pq_m
        ),
        write_secs: write_elapsed.as_secs_f64(),
        write_vec_per_sec,
        index_build_secs: 0.0, // inline — already counted in write_secs
        load_secs: load_elapsed.as_secs_f64(),
        recall: recall_sum / num_queries as f64,
        qps: lat.qps,
        mean_ms: lat.mean_ms,
        p50_ms: lat.p50_ms,
        p95_ms: lat.p95_ms,
        p99_ms: lat.p99_ms,
    })
}

async fn run_ailake_auto(args: &Args, ds: &dataset::Dataset) -> anyhow::Result<BenchResult> {
    // Print hardware profile so the user can verify detection is correct.
    let hw = HardwareProfile::detect();
    let backend_label = match hw.backend {
        ailake_index::HardwareBackend::NvidiaCuda => "NVIDIA CUDA",
        ailake_index::HardwareBackend::AmdRocm => "AMD ROCm",
        ailake_index::HardwareBackend::CpuSimd => "CPU (no GPU)",
    };
    eprintln!("\nHardware detection:");
    eprintln!("  Backend      : {backend_label}");
    eprintln!("  CUDA GPU     : {}", hw.has_cuda);
    eprintln!("  ROCm GPU     : {}", hw.has_rocm);
    eprintln!("  CPU cores    : {}", hw.cpu_logical_cores);
    eprintln!("  AVX2         : {}", hw.has_avx2);
    eprintln!("  AVX-512F     : {}", hw.has_avx512);
    let shard_n = args.shard_size;
    let index_choice = if hw.recommend_ivf_pq(shard_n) {
        "IVF-PQ"
    } else {
        "HNSW"
    };
    eprintln!("  Index chosen : {index_choice}  (shard_size={shard_n})");

    let tmp;
    let table_root: PathBuf = match &args.table_dir {
        Some(p) => {
            let mut p = p.clone();
            p.push("auto");
            p
        }
        None => {
            tmp = tempfile::TempDir::new().context("create temp dir")?;
            tmp.path().to_path_buf()
        }
    };

    let store: Arc<dyn Store> = Arc::new(LocalStore::new(&table_root));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
    let table = TableIdent::new("default", "sift1m_auto");

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim: ds.dim as u32,
        metric: VectorMetric::Euclidean,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
    };

    eprintln!("\nAI-Lake Auto write phase …");
    let write_start = Instant::now();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let total_base = ds.base.len();
    let shard_size = args.shard_size;
    let num_shards = total_base.div_ceil(shard_size);

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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ids))])?;

        writer
            .write_batch_auto(&batch, shard_vecs)
            .await
            .with_context(|| format!("write shard {shard_idx}"))?;

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
    let write_vec_per_sec = total_base as f64 / write_elapsed.as_secs_f64();

    eprintln!(
        "  committed snapshot {}  ({:.1}s  {:.0} vec/s) — {index_choice} built inline",
        snapshot_id,
        write_elapsed.as_secs_f64(),
        write_vec_per_sec,
    );

    eprintln!("  Loading {index_choice} indexes into memory …");
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

    eprintln!("\nAI-Lake Auto search phase (top_k={}) …", args.top_k);

    let search_cfg = SearchConfig {
        top_k: args.top_k,
        ef_search: args.ef,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
    };

    let num_queries = ds.queries.len();
    let mut latencies_us: Vec<u64> = Vec::with_capacity(num_queries);
    let mut recall_sum = 0.0f64;
    let search_wall_start = Instant::now();

    for (qi, query) in ds.queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = session.search_query(query, &search_cfg);
        latencies_us.push(t0.elapsed().as_micros() as u64);

        let result_ids: Vec<u32> = results
            .iter()
            .map(|r| {
                let part_num = parse_part_num(&r.file_path);
                (part_num as u64 * shard_size as u64 + r.row_id.as_u64()) as u32
            })
            .collect();

        recall_sum += metrics::recall_at_k(&result_ids, &ds.ground_truth[qi], args.top_k);

        if (qi + 1) % 1000 == 0 {
            eprint!("\r  {}/{} queries …", qi + 1, num_queries);
        }
    }
    eprintln!("\r  {num_queries}/{num_queries} queries done");

    let search_wall_ns = search_wall_start.elapsed().as_nanos() as u64;
    let lat = metrics::LatencyStats::compute(&mut latencies_us, search_wall_ns);

    Ok(BenchResult {
        engine: format!("AI-Lake Auto/{index_choice} ({num_shards} shards)"),
        write_secs: write_elapsed.as_secs_f64(),
        write_vec_per_sec,
        index_build_secs: 0.0,
        load_secs: load_elapsed.as_secs_f64(),
        recall: recall_sum / num_queries as f64,
        qps: lat.qps,
        mean_ms: lat.mean_ms,
        p50_ms: lat.p50_ms,
        p95_ms: lat.p95_ms,
        p99_ms: lat.p99_ms,
    })
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
