// SPDX-License-Identifier: MIT OR Apache-2.0
mod serve;

use std::sync::Arc;

use ailake_catalog::{
    hadoop::HadoopCatalog,
    provider::{CatalogProvider, TableIdent, TableProperties},
};
use ailake_core::{
    AilakeError, EmbeddingModelInfo, VectorMetric, VectorModality, VectorPrecision,
    VectorStoragePolicy,
};
use ailake_query::{
    CompactionConfig, CompactionExecutor, CompactionPlanner, EmbedFn, MigrationJob,
    MigrationProgress, MigrationStrategy, MultiVectorBatch, ProgressFn, SearchConfig, TableWriter,
};
use ailake_store::store_from_url;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "ailake",
    about = "AI-Lake Format — administrative CLI",
    version,
    propagate_version = true
)]
struct Cli {
    /// Storage URL (s3://bucket/prefix, gs://bucket/prefix, az://container/prefix, /local/path)
    #[arg(long, global = true, env = "AILAKE_STORE_URL", default_value = ".")]
    store: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new AI-Lake table
    Create {
        /// Table name (namespace.table or just table — defaults to namespace "default")
        table: String,
        /// Vector column dimensionality
        #[arg(long)]
        dim: u32,
        /// Distance metric
        #[arg(long, value_enum, default_value = "cosine")]
        metric: Metric,
        /// Vector precision
        #[arg(long, value_enum, default_value = "f16")]
        precision: Precision,
        /// Vector column name
        #[arg(long, default_value = "embedding")]
        column: String,
        /// Normalize vectors to unit L2 at write time (recommended for cosine).
        /// Enables the NormalizedCosine fast path: 1-dot(a,b) instead of full
        /// cosine — no sqrt in the HNSW hot loop. ~12-20% faster search on
        /// high-dim embeddings (OpenAI, Cohere). No-op for euclidean/dot.
        #[arg(long, default_value_t = false)]
        pre_normalize: bool,
        /// HNSW M — connections per node (default: 16).
        /// Higher → better recall, more memory. Range: 4–64.
        #[arg(long)]
        hnsw_m: Option<u32>,
        /// HNSW ef_construction — candidate pool during build (default: 150).
        /// Higher → better graph quality, slower build. Range: 40–400.
        #[arg(long)]
        hnsw_ef: Option<u32>,
        /// PQ-only mode: omit raw vector column from Parquet files.
        /// Reduces vector storage by ~98% (index BLOB only; no raw F16 column).
        /// Trade-off: reranking after HNSW/IVF-PQ is disabled — recall@10 ~93-95%.
        /// Requires `--metric cosine` (or euclidean) with an IVF-PQ or HNSW index.
        #[arg(long, default_value_t = false)]
        pq_only: bool,
        /// Residual PQ: encode (vec - coarse_centroid) per IVF cell instead of raw vec.
        /// Zero storage overhead; improves recall@10 by ~2-4 pp on typical embeddings.
        /// Only effective when the auto index path selects IVF-PQ.
        #[arg(long, default_value_t = false)]
        ivf_residual: bool,
        /// Modality tag for the primary vector column (text, image, audio, video).
        /// Stored as ailake.modality-<column> in Iceberg properties.
        #[arg(long, value_enum)]
        modality: Option<ModalityArg>,
    },
    /// Insert a Parquet file (with an embedding column) into a table
    Insert {
        /// Table name
        table: String,
        /// Path to source Parquet file on the local filesystem
        file: String,
        /// Name of the embeddings column in the source file (single-column mode)
        #[arg(long, default_value = "embedding")]
        embeddings: String,
        /// Multi-column mode: comma-separated column specs, each as col:dim:metric[:modality].
        /// Example: "embedding:1536:cosine,image_embedding:512:cosine:image"
        /// When set, --embeddings is ignored.
        #[arg(long)]
        vector_cols: Option<String>,
        /// Idempotency key — no-op if this batch_id was already committed (safe for Airflow retries)
        #[arg(long)]
        batch_id: Option<String>,
    },
    /// Search a table by vector similarity
    Search {
        /// Table name
        table: String,
        /// Query vector as comma-separated floats (e.g. "0.1,0.2,0.3")
        #[arg(long, conflicts_with = "query_file")]
        query: Option<String>,
        /// Path to a binary file containing the query vector (little-endian f32 array)
        #[arg(long, conflicts_with = "query")]
        query_file: Option<String>,
        /// Number of results to return
        #[arg(long, default_value = "10")]
        top_k: usize,
        /// Geometric pruning threshold (0.0–1.0; lower = more aggressive)
        #[arg(long, default_value = "0.8")]
        pruning_threshold: f32,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
    /// Compact small files in a table into a larger merged file
    Compact {
        /// Table name
        table: String,
        /// Target file size in bytes (default: 512 MiB)
        #[arg(long, default_value = "536870912")]
        target_size: u64,
        /// Minimum number of small files required to trigger compaction
        #[arg(long, default_value = "4")]
        min_files: usize,
    },
    /// Start an HTTP server exposing search, write, compact and info over JSON
    Serve {
        /// Table name
        table: String,
        /// Port to listen on
        #[arg(long, default_value = "7700")]
        port: u16,
        /// Vector column name
        #[arg(long, default_value = "embedding")]
        column: String,
    },
    /// Print table statistics
    Info {
        /// Table name
        table: String,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
    /// Migrate embedding column to a new model by re-embedding all chunks via an external command
    Migrate {
        /// Table name (namespace.table or just table)
        table: String,
        /// Name of the existing embedding column
        #[arg(long, default_value = "embedding")]
        old_column: String,
        /// Name for the migrated column (may equal --old-column for in-place upgrade)
        #[arg(long, default_value = "embedding_v2")]
        new_column: String,
        /// Parquet column that holds the raw text to re-embed
        #[arg(long, default_value = "chunk_text")]
        text_column: String,
        /// Shell command that reads a JSON array of strings from stdin and writes
        /// a JSON array of float arrays to stdout. Example:
        ///   python3 embed.py
        #[arg(long)]
        embed_cmd: String,
        /// Migration strategy: atomic_replace (lower storage) or dual_write_then_cutover (zero downtime)
        #[arg(long, value_enum, default_value = "dual-write-then-cutover")]
        strategy: MigrateStrategy,
        /// Number of texts per embed-cmd call
        #[arg(long, default_value = "512")]
        batch_size: usize,
        /// Model identifier stored in ailake.embedding-model after migration
        #[arg(long)]
        model_name: Option<String>,
        /// Optional version tag appended to --model-name (stored as "<name>@<version>")
        #[arg(long)]
        model_version: Option<String>,
    },
    /// Estimate storage usage before writing (no I/O — pure math)
    Estimate {
        /// Number of vectors (supports K/M/B suffixes: 1M, 500K, 1B)
        #[arg(long)]
        rows: String,
        /// Vector dimensionality
        #[arg(long)]
        dim: u32,
        /// HNSW M parameter — connections per node (default: 16)
        #[arg(long, default_value = "16")]
        hnsw_m: u32,
        /// PQ sub-vectors M — used for PQ-only and IVF-PQ estimates (default: dim/32, min 8)
        #[arg(long)]
        pq_m: Option<u32>,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
}

#[derive(ValueEnum, Clone)]
enum MigrateStrategy {
    AtomicReplace,
    DualWriteThenCutover,
}

impl From<MigrateStrategy> for MigrationStrategy {
    fn from(s: MigrateStrategy) -> Self {
        match s {
            MigrateStrategy::AtomicReplace => MigrationStrategy::AtomicReplace,
            MigrateStrategy::DualWriteThenCutover => MigrationStrategy::DualWriteThenCutover,
        }
    }
}

#[derive(ValueEnum, Clone)]
enum Metric {
    Cosine,
    Euclidean,
    Dot,
}

impl From<Metric> for VectorMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Cosine => VectorMetric::Cosine,
            Metric::Euclidean => VectorMetric::Euclidean,
            Metric::Dot => VectorMetric::DotProduct,
        }
    }
}

#[derive(ValueEnum, Clone)]
enum Precision {
    F32,
    F16,
    I8,
}

impl From<Precision> for VectorPrecision {
    fn from(p: Precision) -> Self {
        match p {
            Precision::F32 => VectorPrecision::F32,
            Precision::F16 => VectorPrecision::F16,
            Precision::I8 => VectorPrecision::I8,
        }
    }
}

#[derive(ValueEnum, Clone)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(ValueEnum, Clone)]
enum ModalityArg {
    Text,
    Image,
    Audio,
    Video,
}

impl From<ModalityArg> for VectorModality {
    fn from(m: ModalityArg) -> Self {
        match m {
            ModalityArg::Text => VectorModality::Text,
            ModalityArg::Image => VectorModality::Image,
            ModalityArg::Audio => VectorModality::Audio,
            ModalityArg::Video => VectorModality::Video,
        }
    }
}

/// Parse "namespace.table" → (namespace, table).
/// Plain "table" → ("default", "table").
fn parse_table_ident(s: &str) -> TableIdent {
    match s.split_once('.') {
        Some((ns, name)) => TableIdent::new(ns, name),
        None => TableIdent::new("default", s),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let cli = Cli::parse();

    if let Err(e) = run(cli).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    // Estimate is pure math — handle before store/catalog creation.
    if let Commands::Estimate {
        rows,
        dim,
        hnsw_m,
        pq_m,
        format,
    } = cli.command
    {
        return run_estimate(&rows, dim, hnsw_m, pq_m, &format);
    }

    let store = store_from_url(&cli.store).map_err(|e| e.to_string())?;
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));

    match cli.command {
        Commands::Create {
            table,
            dim,
            metric,
            precision,
            column,
            pre_normalize,
            hnsw_m,
            hnsw_ef,
            pq_only,
            ivf_residual,
            modality,
        } => {
            let ident = parse_table_ident(&table);
            let policy = VectorStoragePolicy {
                column_name: column,
                dim,
                metric: metric.into(),
                precision: precision.into(),
                pq: None,
                keep_raw_for_reranking: !pq_only,
                pre_normalize,
                hnsw_m,
                hnsw_ef_construction: hnsw_ef,
                ivf_residual,
                embedding_model: None,
                modality: modality.map(VectorModality::from),
            };

            catalog
                .create_table(
                    &ident,
                    &TableProperties {
                        policy,
                        extra: std::collections::HashMap::new(),
                    },
                )
                .await
                .map_err(|e| e.to_string())?;

            println!("created table {table}");
            Ok(())
        }

        Commands::Insert {
            table,
            file,
            embeddings,
            vector_cols,
            batch_id,
        } => {
            let ident = parse_table_ident(&table);

            // Read source Parquet from local disk.
            let raw = std::fs::read(&file).map_err(|e| format!("failed to read {file}: {e}"))?;
            let bytes = bytes::Bytes::from(raw);

            if let Some(cols_spec) = vector_cols {
                // Multi-column mode: col:dim:metric[:modality],...
                let col_specs = parse_vector_cols(&cols_spec)?;
                if col_specs.is_empty() {
                    return Err("--vector-cols must have at least one column spec".into());
                }

                // Read tabular data + all embedding columns from source Parquet.
                let first_col = &col_specs[0].0;
                let reader = ailake_parquet::ParquetVectorReader::new(bytes.clone(), first_col);
                let (batch, first_embs) = reader.read_all().map_err(|e| e.to_string())?;
                let rows = first_embs.len();

                // Build policies + embeddings for each column.
                let mut mv_owned: Vec<(VectorStoragePolicy, Vec<Vec<f32>>)> =
                    Vec::with_capacity(col_specs.len());
                let (first_name, first_dim, first_metric, first_modality) = col_specs[0].clone();
                let first_policy = VectorStoragePolicy {
                    column_name: first_name,
                    dim: first_dim,
                    metric: first_metric,
                    precision: VectorPrecision::F16,
                    pq: None,
                    keep_raw_for_reranking: true,
                    pre_normalize: false,
                    hnsw_m: None,
                    hnsw_ef_construction: None,
                    ivf_residual: false,
                    embedding_model: None,
                    modality: first_modality,
                };
                mv_owned.push((first_policy, first_embs));

                for (col_name, dim, metric, modality) in &col_specs[1..] {
                    let reader =
                        ailake_parquet::ParquetVectorReader::new(bytes.clone(), col_name);
                    let (_, embs) = reader.read_all().map_err(|e| e.to_string())?;
                    let policy = VectorStoragePolicy {
                        column_name: col_name.clone(),
                        dim: *dim,
                        metric: *metric,
                        precision: VectorPrecision::F16,
                        pq: None,
                        keep_raw_for_reranking: true,
                        pre_normalize: false,
                        hnsw_m: None,
                        hnsw_ef_construction: None,
                        ivf_residual: false,
                        embedding_model: None,
                        modality: *modality,
                    };
                    mv_owned.push((policy, embs));
                }

                // Use first policy as the table-level policy for create_or_open.
                let table_policy = mv_owned[0].0.clone();
                let mut writer =
                    TableWriter::create_or_open(catalog, Arc::clone(&store), table_policy, ident)
                        .await
                        .map_err(|e| e.to_string())?;

                let batches: Vec<MultiVectorBatch<'_>> = mv_owned
                    .iter()
                    .map(|(policy, embs)| MultiVectorBatch {
                        policy: policy.clone(),
                        embeddings: embs.as_slice(),
                    })
                    .collect();

                writer
                    .write_batch_multi(&batch, &batches)
                    .await
                    .map_err(|e| e.to_string())?;
                writer.commit().await.map_err(|e| e.to_string())?;

                println!("inserted {rows} rows into {table} ({} vector columns)", col_specs.len());
            } else {
                // Single-column mode (original behavior).
                let reader = ailake_parquet::ParquetVectorReader::new(bytes, &embeddings);
                let (batch, embs) = reader.read_all().map_err(|e| e.to_string())?;

                let dim = embs.first().map(|v| v.len() as u32).unwrap_or(0);
                if dim == 0 {
                    return Err("source file has no embedding rows".into());
                }

                // Load existing policy from catalog, or default to cosine/f16.
                let policy = match catalog.load_table(&ident).await {
                    Ok(meta) => VectorStoragePolicy {
                        column_name: embeddings.clone(),
                        dim,
                        metric: meta
                            .properties
                            .get("ailake.vector-metric")
                            .map(|m| match m.as_str() {
                                "euclidean" => VectorMetric::Euclidean,
                                "dot" => VectorMetric::DotProduct,
                                _ => VectorMetric::Cosine,
                            })
                            .unwrap_or(VectorMetric::Cosine),
                        precision: VectorPrecision::F16,
                        pq: None,
                        keep_raw_for_reranking: true,
                        pre_normalize: false,
                        hnsw_m: None,
                        hnsw_ef_construction: None,
                        ivf_residual: false,
                        embedding_model: None,
                        modality: None,
                    },
                    Err(_) => VectorStoragePolicy {
                        column_name: embeddings.clone(),
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
                    },
                };

                let mut writer =
                    TableWriter::create_or_open(catalog, Arc::clone(&store), policy, ident)
                        .await
                        .map_err(|e| e.to_string())?;

                let rows = embs.len();
                match batch_id {
                    Some(ref id) => writer
                        .write_batch_idempotent(&batch, &embs, id)
                        .await
                        .map_err(|e| e.to_string())?,
                    None => writer
                        .write_batch(&batch, &embs)
                        .await
                        .map_err(|e| e.to_string())?,
                }
                writer.commit().await.map_err(|e| e.to_string())?;

                println!("inserted {rows} rows into {table}");
            }
            Ok(())
        }

        Commands::Search {
            table,
            query,
            query_file,
            top_k,
            pruning_threshold,
            format,
        } => {
            let ident = parse_table_ident(&table);

            let query_vec: Vec<f32> = if let Some(file) = query_file {
                let raw = std::fs::read(&file)
                    .map_err(|e| format!("failed to read query file {file}: {e}"))?;
                if raw.len() % 4 != 0 {
                    return Err(format!(
                        "query file size {} not a multiple of 4 (expected little-endian f32 array)",
                        raw.len()
                    ));
                }
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            } else if let Some(q) = query {
                q.split(',')
                    .map(|s| s.trim().parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?
            } else {
                return Err("either --query or --query-file is required".into());
            };

            let dim = query_vec.len() as u32;

            let config = SearchConfig {
                top_k,
                ef_search: top_k * 5,
                pruning_threshold,
                rerank_factor: None,
            };

            let results = ailake_query::search(
                &ident,
                &query_vec,
                config,
                "embedding",
                dim,
                catalog as Arc<dyn CatalogProvider>,
                store,
            )
            .await
            .map_err(|e| e.to_string())?;

            match format {
                OutputFormat::Json => {
                    let json_results: Vec<serde_json::Value> = results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            serde_json::json!({
                                "rank": i + 1,
                                "row_id": r.row_id.0,
                                "distance": r.distance,
                                "file_path": r.file_path,
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string(&serde_json::json!({ "results": json_results }))
                            .map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => {
                    if results.is_empty() {
                        println!("no results");
                        return Ok(());
                    }
                    println!("{:<6} {:<12} file", "rank", "distance");
                    for (i, r) in results.iter().enumerate() {
                        println!("{:<6} {:<12.6} {}", i + 1, r.distance, r.file_path);
                    }
                }
            }
            Ok(())
        }

        Commands::Compact {
            table,
            target_size,
            min_files,
        } => {
            let ident = parse_table_ident(&table);

            let meta = catalog
                .load_table(&ident)
                .await
                .map_err(|e| e.to_string())?;

            let dim = meta
                .properties
                .get("ailake.vector-dim")
                .and_then(|v| v.parse::<u32>().ok())
                .ok_or("table missing ailake.vector-dim property")?;
            let column = meta
                .properties
                .get("ailake.vector-column")
                .cloned()
                .unwrap_or_else(|| "embedding".to_string());

            let policy = VectorStoragePolicy {
                column_name: column,
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
            };

            let files = catalog
                .list_files(&ident, None)
                .await
                .map_err(|e| e.to_string())?;

            let config = CompactionConfig {
                min_files_to_compact: min_files,
                target_file_size_bytes: target_size,
                index_strategy: Default::default(),
            };
            let planner = CompactionPlanner::new(config);
            let to_compact = planner.plan(&files);

            if to_compact.is_empty() {
                println!("nothing to compact ({} files below threshold)", files.len());
                return Ok(());
            }

            println!(
                "compacting {} of {} files...",
                to_compact.len(),
                files.len()
            );

            let executor = CompactionExecutor::new(Arc::clone(&store), policy);
            let output_path = format!(
                "data/compacted-{}.parquet",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            let new_entry = executor
                .compact(&to_compact, &output_path)
                .await
                .map_err(|e| e.to_string())?;

            // Build replacement file list: keep files not compacted + add merged.
            let compacted_paths: std::collections::HashSet<&str> =
                to_compact.iter().map(|f| f.path.as_str()).collect();
            let mut remaining: Vec<_> = files
                .into_iter()
                .filter(|f| !compacted_paths.contains(f.path.as_str()))
                .collect();
            remaining.push(new_entry);

            let snap = ailake_catalog::provider::NewSnapshot {
                snapshot_id: ailake_catalog::provider::new_snapshot_id(),
                parent_snapshot_id: meta.current_snapshot_id,
                files: remaining,
                operation: ailake_catalog::provider::SnapshotOperation::Replace,
                iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
            };
            catalog
                .commit_snapshot(&ident, snap)
                .await
                .map_err(|e| e.to_string())?;

            println!("compacted into {output_path}");
            Ok(())
        }

        Commands::Serve {
            table,
            port,
            column,
        } => {
            let ident = parse_table_ident(&table);
            let meta = catalog
                .load_table(&ident)
                .await
                .map_err(|e| e.to_string())?;
            let dim = meta
                .properties
                .get("ailake.vector-dim")
                .and_then(|v| v.parse::<u32>().ok())
                .ok_or("table missing ailake.vector-dim property")?;
            let metric = meta
                .properties
                .get("ailake.vector-metric")
                .map(|m| match m.as_str() {
                    "euclidean" => VectorMetric::Euclidean,
                    "dot" => VectorMetric::DotProduct,
                    _ => VectorMetric::Cosine,
                })
                .unwrap_or(VectorMetric::Cosine);
            let policy = VectorStoragePolicy {
                column_name: column,
                dim,
                metric,
                precision: VectorPrecision::F16,
                pq: None,
                keep_raw_for_reranking: true,
                pre_normalize: false,
                hnsw_m: None,
                hnsw_ef_construction: None,
                ivf_residual: false,
                embedding_model: None,
                modality: None,
            };
            serve::run(
                catalog as Arc<dyn CatalogProvider>,
                store,
                ident,
                policy,
                port,
            )
            .await
        }

        Commands::Info { table, format } => {
            let ident = parse_table_ident(&table);

            let meta = catalog
                .load_table(&ident)
                .await
                .map_err(|e| e.to_string())?;
            let files = catalog
                .list_files(&ident, None)
                .await
                .map_err(|e| e.to_string())?;

            let file_count = files.len();
            let row_count: u64 = files.iter().map(|f| f.record_count).sum();
            let size_bytes: u64 = files.iter().map(|f| f.file_size_bytes).sum();
            let ready = files
                .iter()
                .filter(|f| f.index_status == ailake_catalog::provider::IndexStatus::Ready)
                .count();

            let location = meta
                .properties
                .get("ailake.location")
                .cloned()
                .unwrap_or_else(|| meta.location.clone());
            let vector_column = meta
                .properties
                .get("ailake.vector-column")
                .map(String::as_str)
                .unwrap_or("-")
                .to_string();
            let vector_dim = meta
                .properties
                .get("ailake.vector-dim")
                .map(String::as_str)
                .unwrap_or("-")
                .to_string();
            let vector_metric = meta
                .properties
                .get("ailake.vector-metric")
                .map(String::as_str)
                .unwrap_or("-")
                .to_string();

            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string(&serde_json::json!({
                            "table": table,
                            "location": location,
                            "vector_column": vector_column,
                            "vector_dim": vector_dim,
                            "vector_metric": vector_metric,
                            "files": file_count,
                            "indexed_files": ready,
                            "rows": row_count,
                            "size_bytes": size_bytes,
                            "snapshot_id": meta.current_snapshot_id,
                        }))
                        .map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => {
                    println!("table:       {table}");
                    println!("location:    {location}");
                    println!(
                        "vector:      col={vector_column} dim={vector_dim} metric={vector_metric}"
                    );
                    println!("files:       {file_count} ({ready} indexed)");
                    println!("rows:        {row_count}");
                    println!("size:        {}", format_bytes(size_bytes));
                    if let Some(snap_id) = meta.current_snapshot_id {
                        println!("snapshot:    {snap_id}");
                    }
                }
            }
            Ok(())
        }

        Commands::Migrate {
            table,
            old_column,
            new_column,
            text_column,
            embed_cmd,
            strategy,
            batch_size,
            model_name,
            model_version,
        } => {
            let ident = parse_table_ident(&table);

            let new_model = model_name.map(|name| {
                let mut info = EmbeddingModelInfo::new(name);
                if let Some(v) = model_version {
                    info = info.with_version(v);
                }
                info
            });

            // Wrap external embed command as a sync Fn closure.
            // stdin: JSON array of strings; stdout: JSON array of float arrays.
            let embed_fn: EmbedFn = {
                let embed_cmd = embed_cmd.clone();
                std::sync::Arc::new(move |texts: &[String]| {
                    use std::io::Write;
                    let input = serde_json::to_string(texts)
                        .map_err(|e| AilakeError::InvalidArgument(e.to_string()))?;
                    let output = std::process::Command::new("sh")
                        .args(["-c", &embed_cmd])
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::inherit())
                        .spawn()
                        .and_then(|mut child| {
                            child.stdin.take().unwrap().write_all(input.as_bytes())?;
                            child.wait_with_output()
                        })
                        .map_err(|e| {
                            AilakeError::InvalidArgument(format!("embed-cmd spawn error: {e}"))
                        })?;
                    if !output.status.success() {
                        return Err(AilakeError::InvalidArgument(format!(
                            "embed-cmd exited with status {}",
                            output.status
                        )));
                    }
                    serde_json::from_slice::<Vec<Vec<f32>>>(&output.stdout).map_err(|e| {
                        AilakeError::InvalidArgument(format!("embed-cmd stdout parse error: {e}"))
                    })
                })
            };

            let on_progress: Option<ProgressFn> = Some(Arc::new(|p: MigrationProgress| {
                eprintln!(
                    "migration: {}/{} files done, {} rows migrated",
                    p.files_done, p.files_total, p.rows_migrated
                );
            }));

            let job = MigrationJob {
                table: ident,
                old_column,
                new_column,
                text_column,
                embed_fn,
                strategy: strategy.into(),
                batch_size,
                new_model,
                on_progress,
            };

            job.run(catalog as std::sync::Arc<dyn CatalogProvider>, store)
                .await
                .map_err(|e| e.to_string())?;

            println!("migration complete");
            Ok(())
        }

        // Handled before store/catalog creation — unreachable here.
        Commands::Estimate { .. } => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse "--vector-cols" spec: "col:dim:metric[:modality],..."
/// Returns Vec<(column_name, dim, metric, modality)>.
fn parse_vector_cols(
    spec: &str,
) -> Result<Vec<(String, u32, VectorMetric, Option<VectorModality>)>, String> {
    spec.split(',')
        .map(|part| {
            let parts: Vec<&str> = part.trim().splitn(4, ':').collect();
            if parts.len() < 3 {
                return Err(format!(
                    "invalid vector-cols spec '{part}' — expected col:dim:metric[:modality]"
                ));
            }
            let col = parts[0].to_string();
            let dim: u32 = parts[1]
                .parse()
                .map_err(|_| format!("invalid dim '{}' in vector-cols spec '{part}'", parts[1]))?;
            let metric = match parts[2] {
                "cosine" => VectorMetric::Cosine,
                "euclidean" => VectorMetric::Euclidean,
                "dot" | "dot_product" | "dotproduct" => VectorMetric::DotProduct,
                "normalized_cosine" => VectorMetric::NormalizedCosine,
                other => {
                    return Err(format!(
                        "unknown metric '{other}' in vector-cols spec '{part}'"
                    ))
                }
            };
            let modality = parts.get(3).and_then(|m| VectorModality::from_str(m));
            Ok((col, dim, metric, modality))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ailake estimate — pure storage math, no I/O
// ---------------------------------------------------------------------------

fn parse_rows(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1_000u64),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1_000_000u64),
        Some('B') | Some('b') | Some('G') | Some('g') => (&s[..s.len() - 1], 1_000_000_000u64),
        _ => (s, 1u64),
    };
    let n: f64 = num.replace('_', "").parse().map_err(|_| {
        format!("invalid row count '{s}' — use a number with optional K/M/B suffix")
    })?;
    Ok((n * mult as f64) as u64)
}

#[derive(Debug)]
struct EstimateRow {
    label: &'static str,
    vectors_bytes: u64,
    index_bytes: u64,
    recall: &'static str,
    note: &'static str,
}

fn run_estimate(
    rows_str: &str,
    dim: u32,
    hnsw_m: u32,
    pq_m_opt: Option<u32>,
    format: &OutputFormat,
) -> Result<(), String> {
    let rows = parse_rows(rows_str)?;
    let dim = dim as u64;

    // Default PQ M: dim/32, clamped to [8, dim].
    let pq_m = pq_m_opt
        .map(|m| m as u64)
        .unwrap_or_else(|| (dim / 32).max(8).min(dim));

    // Raw vector bytes per row per precision.
    let vec_f32 = rows * dim * 4;
    let vec_f16 = rows * dim * 2;
    let vec_i8 = rows * dim;

    // HNSW index: each node stores ~M×2 neighbor IDs (u32 × 2 per layer).
    // Approximation: rows × hnsw_m × 2 × 4 bytes (two u32 per neighbor slot).
    // Real bincode overhead adds ~10-15%; use 18 bytes/neighbor as empirical factor.
    let hnsw_bytes = rows * hnsw_m as u64 * 2 * 9; // ≈ M×2 neighbors × 9 bytes avg

    // IVF-PQ codes: rows × pq_m bytes (1 byte per sub-quantizer code).
    // Codebook: negligible vs row data for any practical table size.
    let pq_bytes = rows * pq_m;

    // Recall estimates (literature + empirical for text embeddings dim=768-3072).
    let rows_table = vec![
        EstimateRow {
            label: "F32 (baseline)",
            vectors_bytes: vec_f32,
            index_bytes: hnsw_bytes,
            recall: "~99%",
            note: "",
        },
        EstimateRow {
            label: "F16 (default)",
            vectors_bytes: vec_f16,
            index_bytes: hnsw_bytes,
            recall: "~99%",
            note: "",
        },
        EstimateRow {
            label: "I8",
            vectors_bytes: vec_i8,
            index_bytes: hnsw_bytes,
            recall: "~97%",
            note: "",
        },
        EstimateRow {
            label: "F16 + IVF-PQ index",
            vectors_bytes: vec_f16,
            index_bytes: pq_bytes,
            recall: "~99%",
            note: "reranks with raw F16",
        },
        EstimateRow {
            label: "I8  + IVF-PQ index",
            vectors_bytes: vec_i8,
            index_bytes: pq_bytes,
            recall: "~97%",
            note: "reranks with raw I8",
        },
        EstimateRow {
            label: "PQ-only (--pq-only)",
            vectors_bytes: 0,
            index_bytes: pq_bytes,
            recall: "~94%",
            note: "no reranking",
        },
    ];

    let baseline_total = vec_f32 + hnsw_bytes;

    match format {
        OutputFormat::Json => {
            let entries: Vec<serde_json::Value> = rows_table
                .iter()
                .map(|r| {
                    let total = r.vectors_bytes + r.index_bytes;
                    let reduction = baseline_total as f64 / total.max(1) as f64;
                    serde_json::json!({
                        "mode": r.label,
                        "vectors_bytes": r.vectors_bytes,
                        "index_bytes": r.index_bytes,
                        "total_bytes": total,
                        "reduction_factor": format!("{reduction:.1}×"),
                        "recall_at_10": r.recall,
                        "note": r.note,
                    })
                })
                .collect();
            let out = serde_json::json!({
                "rows": rows,
                "dim": dim,
                "hnsw_m": hnsw_m,
                "pq_m": pq_m,
                "estimates": entries,
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
        }

        OutputFormat::Text => {
            println!(
                "\nStorage estimate — {} rows × dim={} (HNSW M={}, PQ M={})\n",
                format_count(rows),
                dim,
                hnsw_m,
                pq_m
            );
            println!(
                "  {:<26} {:>10}  {:>12}  {:>10}  {:>8}  Recall@10",
                "Mode", "Vectors", "Index", "Total", "Reduct."
            );
            println!("  {}", "-".repeat(82));

            for r in &rows_table {
                let total = r.vectors_bytes + r.index_bytes;
                let reduction = baseline_total as f64 / total.max(1) as f64;
                let note = if r.note.is_empty() {
                    String::new()
                } else {
                    format!("  ← {}", r.note)
                };
                println!(
                    "  {:<26} {:>10}  {:>12}  {:>10}  {:>7.1}×  {}{}",
                    r.label,
                    format_bytes(r.vectors_bytes),
                    format_bytes(r.index_bytes),
                    format_bytes(total),
                    reduction,
                    r.recall,
                    note,
                );
            }
            println!();
        }
    }

    Ok(())
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

fn format_bytes(b: u64) -> String {
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;
    if b >= GB {
        format!("{:.2} GiB", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{:.2} MiB", b as f64 / MB as f64)
    } else {
        format!("{b} B")
    }
}
