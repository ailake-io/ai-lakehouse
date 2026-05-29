// SPDX-License-Identifier: MIT OR Apache-2.0
mod serve;

use std::sync::Arc;

use ailake_catalog::{
    hadoop::HadoopCatalog,
    provider::{CatalogProvider, TableIdent, TableProperties},
};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{
    CompactionConfig, CompactionExecutor, CompactionPlanner, SearchConfig, TableWriter,
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
    },
    /// Insert a Parquet file (with an embedding column) into a table
    Insert {
        /// Table name
        table: String,
        /// Path to source Parquet file on the local filesystem
        file: String,
        /// Name of the embeddings column in the source file
        #[arg(long, default_value = "embedding")]
        embeddings: String,
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
    let store = store_from_url(&cli.store).map_err(|e| e.to_string())?;
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));

    match cli.command {
        Commands::Create {
            table,
            dim,
            metric,
            precision,
            column,
        } => {
            let ident = parse_table_ident(&table);
            let policy = VectorStoragePolicy {
                column_name: column,
                dim,
                metric: metric.into(),
                precision: precision.into(),
                pq: None,
                keep_raw_for_reranking: false,
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
            batch_id,
        } => {
            let ident = parse_table_ident(&table);

            // Read source Parquet from local disk.
            let raw = std::fs::read(&file).map_err(|e| format!("failed to read {file}: {e}"))?;
            let bytes = bytes::Bytes::from(raw);

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
                    keep_raw_for_reranking: false,
                },
                Err(_) => VectorStoragePolicy {
                    column_name: embeddings.clone(),
                    dim,
                    metric: VectorMetric::Cosine,
                    precision: VectorPrecision::F16,
                    pq: None,
                    keep_raw_for_reranking: false,
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
                keep_raw_for_reranking: false,
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
                keep_raw_for_reranking: false,
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
