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
    delete_rows as rs_delete_rows, CompactionConfig, CompactionExecutor, CompactionPlanner,
    EmbedFn, HybridConfig, MemoryDecayJob, MigrationJob, MigrationProgress, MigrationStrategy,
    MultiVectorBatch, ProgressFn, SearchConfig, TableWriter,
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

    /// Catalog backend. `ducklake` requires a local filesystem `--store` (no
    /// s3://gs://az:// scheme) and the `catalog-ducklake` build feature — see
    /// docs/guides/DUCKLAKE_CATALOG.md.
    #[arg(long, global = true, value_enum, default_value = "hadoop")]
    catalog: CatalogBackendArg,

    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone)]
enum CatalogBackendArg {
    Hadoop,
    Ducklake,
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
        /// Iceberg format version: 2 (default, V2) or 3 (opt-in V3).
        /// V3 enables format-version=3 in metadata.json and manifests.
        /// Append/update workloads fully supported; equality deletes not implemented.
        #[arg(long, default_value = "2")]
        format_version: u8,
        /// Comma-separated text columns to index with Tantivy FTS.
        /// When set, every inserted file embeds a per-file inverted index (AILK_FTS section).
        /// Enables the search_text() fast path (O(log N) vs O(N) brute-force BM25).
        /// Example: --fts-columns "chunk_text,document_title"
        #[arg(long)]
        fts_columns: Option<String>,
        /// Tantivy tokenizer for FTS. Default: "default" (standard English tokenizer).
        #[arg(long, default_value = "default")]
        fts_tokenizer: String,
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
        /// Distance metric (single-column mode only; default: table's existing metric,
        /// or cosine for a brand-new table). Multi-column mode carries metric per
        /// column in --vector-cols instead.
        #[arg(long, value_enum)]
        metric: Option<Metric>,
        /// Vector precision (single-column mode only; default: f16).
        #[arg(long, value_enum)]
        precision: Option<Precision>,
        /// Model identifier stored in ailake.embedding-model Iceberg property
        /// (single-column mode only).
        #[arg(long)]
        embedding_model: Option<String>,
        /// Multi-column mode: comma-separated column specs, each as col:dim:metric[:modality].
        /// Example: "embedding:1536:cosine,image_embedding:512:cosine:image"
        /// When set, --embeddings is ignored.
        #[arg(long)]
        vector_cols: Option<String>,
        /// Idempotency key — no-op if this batch_id was already committed (safe for Airflow retries)
        #[arg(long)]
        batch_id: Option<String>,
        /// Comma-separated text columns to index with Tantivy FTS.
        /// Embeds a per-file inverted index (AILK_FTS section) enabling fast full-text search.
        #[arg(long)]
        fts_columns: Option<String>,
        /// Tantivy tokenizer for FTS (default: "default").
        #[arg(long, default_value = "default")]
        fts_tokenizer: String,
        /// Single-column identity partition column name (legacy; prefer --partition-fields).
        #[arg(long)]
        partition_by: Option<String>,
        /// Value for --partition-by. Must be set when --partition-by is set.
        #[arg(long)]
        partition_value: Option<String>,
        /// Multi-column Iceberg partition spec as a JSON array, e.g.
        /// '[{"column":"topic_id","transform":"identity","column_type":"int"}]'.
        /// Takes precedence over --partition-by when set.
        #[arg(long)]
        partition_fields: Option<String>,
        /// Iceberg format version: 2 (default, V2) or 3 (opt-in V3).
        #[arg(long, default_value = "2")]
        format_version: u8,
        /// HNSW M — connections per node (default: 16). Only applies when the table
        /// is created by this insert; ignored on writes to an already-created table.
        #[arg(long)]
        hnsw_m: Option<u32>,
        /// HNSW ef_construction — candidate pool during build (default: 150). Same
        /// only-applies-at-creation caveat as --hnsw-m.
        #[arg(long)]
        hnsw_ef: Option<u32>,
        /// Normalize vectors to unit L2 at write time (recommended for cosine).
        #[arg(long, default_value_t = false)]
        pre_normalize: bool,
        /// Write Parquet immediately and build the HNSW index in the background
        /// instead of blocking until it's fully built. Not combinable with --batch-id
        /// (deferred writes don't yet carry an idempotency tag).
        #[arg(long, default_value_t = false, conflicts_with = "batch_id")]
        deferred: bool,
    },
    /// Search a table by vector similarity or full-text (mutually exclusive)
    Search {
        /// Table name
        table: String,
        /// Query vector as comma-separated floats (e.g. "0.1,0.2,0.3")
        #[arg(long, conflicts_with_all = &["query_file", "text"])]
        query: Option<String>,
        /// Path to a binary file containing the query vector (little-endian f32 array)
        #[arg(long, conflicts_with_all = &["query", "text"])]
        query_file: Option<String>,
        /// Full-text query (uses Tantivy FTS if available, else BM25 brute-force).
        /// Exclusive with --query / --query-file.
        #[arg(long, conflicts_with_all = &["query", "query_file"])]
        text: Option<String>,
        /// Comma-separated text columns to search when using --text.
        /// Falls back to "chunk_text" if not specified.
        #[arg(long)]
        text_columns: Option<String>,
        /// Enables hybrid BM25+vector search: text query fused with --query/--query-file
        /// via Reciprocal Rank Fusion. Requires a vector query too.
        #[arg(long)]
        hybrid_text: Option<String>,
        /// Parquet column BM25-scored for --hybrid-text (default: "chunk_text")
        #[arg(long, default_value = "chunk_text")]
        text_column: String,
        /// BM25 weight in hybrid RRF fusion (0.0 = pure vector, 1.0 = pure BM25)
        #[arg(long, default_value = "0.5")]
        bm25_weight: f32,
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
        /// Maximum files merged in one pass (bounds peak RAM / HNSW rebuild cost)
        #[arg(long, default_value = "20")]
        max_files_per_pass: usize,
        /// Write merged Parquet immediately and build the HNSW index in the
        /// background instead of blocking until it's fully built
        #[arg(long)]
        deferred: bool,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
    /// Recompute recency weights across all memory files in a table
    /// (exp(-lambda * days_since_access), Phase 9 agent memory)
    DecayMemories {
        /// Table name
        table: String,
        /// Exponential decay rate. Higher = faster decay. Typical: 0.05 (slow) to 0.5 (aggressive)
        #[arg(long, default_value = "0.1")]
        lambda: f32,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
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
    /// Mark rows as deleted in a V3 table using Iceberg Deletion Vectors
    DeleteRows {
        /// Table name (namespace.table or just table)
        table: String,
        /// Path of the Parquet data file containing the rows to delete
        /// (as reported by `ailake info`, e.g. "data/part-00001.parquet")
        #[arg(long)]
        file: String,
        /// Comma-separated 0-based row positions to delete (e.g. "0,5,42")
        #[arg(long)]
        rows: String,
    },
    /// Logically delete rows matching an equality predicate (Phase H).
    ///
    /// Writes an Iceberg equality delete file and commits a Delete snapshot.
    /// Matching rows are masked at scan time without rewriting data files.
    DeleteWhere {
        /// Table name (namespace.table or just table)
        table: String,
        /// Column to match against (e.g. document_id, agent_id)
        #[arg(long)]
        col: String,
        /// Comma-separated values to delete (e.g. "doc-abc,doc-def")
        #[arg(long)]
        vals: String,
    },
    /// Evolve the table schema without rewriting data files (Phase G).
    ///
    /// Adds or renames columns in `metadata.json`. Old files missing new columns
    /// will return `initial-default` (or null) at read time — no compaction needed.
    Evolve {
        /// Table name (namespace.table or just table)
        table: String,
        /// Add a column: "name:iceberg_type" e.g. "score:float" or "label:string"
        /// May be specified multiple times for multiple additions.
        #[arg(long = "add", value_name = "NAME:TYPE")]
        adds: Vec<String>,
        /// Initial default for the most recently listed --add column.
        /// JSON literal: 0, 0.0, "unknown", true, null.
        /// Repeated values align positionally with --add occurrences.
        #[arg(long = "initial-default", value_name = "JSON")]
        initial_defaults: Vec<String>,
        /// Rename a column: "old:new" e.g. "old_name:new_name"
        /// May be specified multiple times.
        #[arg(long = "rename", value_name = "OLD:NEW")]
        renames: Vec<String>,
    },
    /// Add a new vector column to an existing table schema (no data files rewritten).
    ///
    /// Stores ailake.dim-<col>, ailake.metric-<col>, ailake.precision-<col> in metadata.json.
    /// Old files return null for the new column until BackfillVectorColumn is run.
    AddVectorColumn {
        /// Table name (namespace.table or just table)
        table: String,
        /// New vector column name
        #[arg(long)]
        column: String,
        /// Vector dimensionality for the new column
        #[arg(long)]
        dim: u32,
        /// Distance metric for the new column
        #[arg(long, value_enum, default_value = "cosine")]
        metric: Metric,
        /// Vector precision for the new column
        #[arg(long, value_enum, default_value = "f16")]
        precision: Precision,
        /// Normalize vectors to unit L2 at write time (enables NormalizedCosine fast path)
        #[arg(long, default_value_t = false)]
        pre_normalize: bool,
        /// HNSW M — connections per node (default: 16)
        #[arg(long)]
        hnsw_m: Option<u32>,
        /// HNSW ef_construction — candidate pool during build (default: 150)
        #[arg(long)]
        hnsw_ef: Option<u32>,
    },
    /// Backfill a new vector column in all existing files (rewrite each file with new embeddings).
    ///
    /// Reads text from --text-column, calls an embed command for each batch, and rewrites
    /// files to include both the original vector column and the new one (via write_multi).
    /// Idempotent: files already containing the column are skipped.
    BackfillVectorColumn {
        /// Table name (namespace.table or just table)
        table: String,
        /// New vector column to backfill (must already exist via add-vector-column)
        #[arg(long)]
        column: String,
        /// Parquet column that holds the raw text to embed
        #[arg(long, default_value = "chunk_text")]
        text_column: String,
        /// Shell command that reads a JSON array of strings from stdin and writes
        /// a JSON array of float arrays to stdout.
        #[arg(long)]
        embed_cmd: String,
        /// How many texts to embed per embed-cmd call
        #[arg(long, default_value = "512")]
        batch_size: usize,
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

/// Build a `DuckLakeCatalog` rooted at `--store` (must be a local filesystem path —
/// DuckLake's metadata store isn't wired for object storage in this CLI; see
/// docs/guides/DUCKLAKE_CATALOG.md). Layout: `<store>/catalog/{ailake_root,ducklake_meta}.db`,
/// `<store>/data/` — created on first use.
#[cfg(feature = "catalog-ducklake")]
async fn build_ducklake_catalog(store_arg: &str) -> Result<Arc<dyn CatalogProvider>, String> {
    if store_arg.contains("://") {
        return Err(
            "--catalog ducklake only supports a local filesystem --store (no s3://, gs://, \
             az:// — see docs/guides/DUCKLAKE_CATALOG.md)"
                .to_string(),
        );
    }
    let warehouse = std::path::Path::new(store_arg);
    let catalog_dir = warehouse.join("catalog");
    let data_dir = warehouse.join("data");
    std::fs::create_dir_all(&catalog_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
    let root_db = catalog_dir.join("ailake_root.db");
    let meta_db = catalog_dir.join("ducklake_meta.db");
    let catalog = ailake_catalog::DuckLakeCatalog::connect(
        root_db.to_str().ok_or("--store path is not valid UTF-8")?,
        meta_db.to_str().ok_or("--store path is not valid UTF-8")?,
        data_dir.to_str().ok_or("--store path is not valid UTF-8")?,
        "lake",
        store_arg,
    )
    .await
    .map_err(|e| e.to_string())?;
    Ok(Arc::new(catalog) as Arc<dyn CatalogProvider>)
}

#[cfg(not(feature = "catalog-ducklake"))]
async fn build_ducklake_catalog(_store_arg: &str) -> Result<Arc<dyn CatalogProvider>, String> {
    Err(
        "this `ailake` binary was built without the catalog-ducklake feature — rebuild with \
         `cargo build --features catalog-ducklake` (or the equivalent release asset)"
            .to_string(),
    )
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
    let catalog: Arc<dyn CatalogProvider> = match cli.catalog {
        CatalogBackendArg::Hadoop => Arc::new(HadoopCatalog::new(Arc::clone(&store), "")),
        CatalogBackendArg::Ducklake => build_ducklake_catalog(&cli.store).await?,
    };

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
            format_version,
            fts_columns,
            fts_tokenizer,
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
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
            };

            let mut extra = std::collections::HashMap::new();
            if let Some(ref cols) = fts_columns {
                extra.insert("ailake.fts.enabled".to_string(), "true".to_string());
                extra.insert("ailake.fts.text-columns".to_string(), cols.clone());
                extra.insert("ailake.fts.tokenizer".to_string(), fts_tokenizer);
            }

            catalog
                .create_table(
                    &ident,
                    &TableProperties {
                        policy,
                        extra,
                        format_version,
                        partition_column_type: None,
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
            metric,
            precision,
            embedding_model,
            vector_cols,
            batch_id,
            fts_columns,
            fts_tokenizer,
            partition_by,
            partition_value,
            partition_fields,
            format_version,
            hnsw_m,
            hnsw_ef,
            pre_normalize,
            deferred,
        } => {
            if deferred && !catalog.supports_in_place_rewrite() {
                return Err("--deferred is not supported with this catalog backend: the \
                            background index build patches the data file in place at its \
                            committed path, which DuckLake cannot re-register (it trusts \
                            the stats and footer size recorded at registration). Use a \
                            blocking insert instead."
                    .into());
            }
            let ident = parse_table_ident(&table);
            let fts_cfg: Option<ailake_fts::FtsConfig> =
                fts_columns.map(|cols| ailake_fts::FtsConfig {
                    text_columns: cols.split(',').map(str::trim).map(String::from).collect(),
                    tokenizer: fts_tokenizer,
                    writer_heap_bytes: 50 * 1024 * 1024,
                });
            let partition_fields: Vec<ailake_core::PartitionDef> = match partition_fields {
                Some(json) => serde_json::from_str(&json)
                    .map_err(|e| format!("invalid --partition-fields JSON: {e}"))?,
                None => vec![],
            };

            // Read source Parquet from local disk.
            let raw = std::fs::read(&file).map_err(|e| format!("failed to read {file}: {e}"))?;
            let bytes = bytes::Bytes::from(raw);

            if let Some(cols_spec) = vector_cols {
                // Multi-column mode: col:dim:metric[:modality],...
                let col_specs = parse_vector_cols(&cols_spec)?;
                if col_specs.is_empty() {
                    return Err("--vector-cols requires at least one column spec (format: col:dim:metric[:modality],...)".into());
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
                    pre_normalize,
                    hnsw_m,
                    hnsw_ef_construction: hnsw_ef,
                    ivf_residual: false,
                    embedding_model: None,
                    modality: first_modality,
                    partition_by: partition_by.clone(),
                    partition_value: partition_value.clone(),
                    partition_column_type: None,
                    partition_fields: partition_fields.clone(),
                };
                mv_owned.push((first_policy, first_embs));

                for (col_name, dim, metric, modality) in &col_specs[1..] {
                    let reader = ailake_parquet::ParquetVectorReader::new(bytes.clone(), col_name);
                    let (_, embs) = reader.read_all().map_err(|e| e.to_string())?;
                    let policy = VectorStoragePolicy {
                        column_name: col_name.clone(),
                        dim: *dim,
                        metric: *metric,
                        precision: VectorPrecision::F16,
                        pq: None,
                        keep_raw_for_reranking: true,
                        pre_normalize,
                        hnsw_m,
                        hnsw_ef_construction: hnsw_ef,
                        ivf_residual: false,
                        embedding_model: None,
                        modality: *modality,
                        partition_by: None,
                        partition_value: None,
                        partition_column_type: None,
                        partition_fields: vec![],
                    };
                    mv_owned.push((policy, embs));
                }

                // Use first policy as the table-level policy for create_or_open.
                let table_policy = mv_owned[0].0.clone();
                let mut writer = {
                    let w = TableWriter::create_or_open(
                        catalog,
                        Arc::clone(&store),
                        table_policy,
                        ident,
                        format_version,
                    )
                    .await
                    .map_err(|e| e.to_string())?;
                    if let Some(cfg) = fts_cfg {
                        w.with_fts_config(cfg)
                    } else {
                        w
                    }
                };

                let batches: Vec<MultiVectorBatch<'_>> = mv_owned
                    .iter()
                    .map(|(policy, embs)| MultiVectorBatch {
                        policy: policy.clone(),
                        embeddings: embs.as_slice(),
                    })
                    .collect();

                if deferred {
                    writer
                        .write_batch_multi_deferred(&batch, &batches)
                        .await
                        .map_err(|e| e.to_string())?;
                } else {
                    writer
                        .write_batch_multi(&batch, &batches)
                        .await
                        .map_err(|e| e.to_string())?;
                }
                writer.commit().await.map_err(|e| e.to_string())?;

                println!(
                    "inserted {rows} rows into {table} ({} vector columns)",
                    col_specs.len()
                );
            } else {
                // Single-column mode (original behavior).
                let reader = ailake_parquet::ParquetVectorReader::new(bytes, &embeddings);
                let (batch, embs) = reader.read_all().map_err(|e| e.to_string())?;

                let dim = embs.first().map(|v| v.len() as u32).unwrap_or(0);
                if dim == 0 {
                    return Err(format!(
                        "embedding column '{}' is empty or contains no vectors in source file",
                        embeddings
                    ));
                }

                let embedding_model_info = embedding_model.map(EmbeddingModelInfo::new);

                // Load existing policy from catalog, or default to cosine/f16.
                let policy = match catalog.load_table(&ident).await {
                    Ok(meta) => VectorStoragePolicy {
                        column_name: embeddings.clone(),
                        dim,
                        metric: metric.clone().map(VectorMetric::from).unwrap_or_else(|| {
                            meta.properties
                                .get("ailake.vector-metric")
                                .map(|m| match m.as_str() {
                                    "euclidean" => VectorMetric::Euclidean,
                                    "dot" => VectorMetric::DotProduct,
                                    _ => VectorMetric::Cosine,
                                })
                                .unwrap_or(VectorMetric::Cosine)
                        }),
                        precision: precision
                            .clone()
                            .map(VectorPrecision::from)
                            .unwrap_or(VectorPrecision::F16),
                        pq: None,
                        keep_raw_for_reranking: true,
                        pre_normalize,
                        hnsw_m,
                        hnsw_ef_construction: hnsw_ef,
                        ivf_residual: false,
                        embedding_model: embedding_model_info.clone(),
                        modality: None,
                        partition_by: partition_by.clone(),
                        partition_value: partition_value.clone(),
                        partition_column_type: None,
                        partition_fields: partition_fields.clone(),
                    },
                    Err(_) => VectorStoragePolicy {
                        column_name: embeddings.clone(),
                        dim,
                        metric: metric
                            .map(VectorMetric::from)
                            .unwrap_or(VectorMetric::Cosine),
                        precision: precision
                            .map(VectorPrecision::from)
                            .unwrap_or(VectorPrecision::F16),
                        pq: None,
                        keep_raw_for_reranking: true,
                        pre_normalize,
                        hnsw_m,
                        hnsw_ef_construction: hnsw_ef,
                        ivf_residual: false,
                        embedding_model: embedding_model_info,
                        modality: None,
                        partition_by,
                        partition_value,
                        partition_column_type: None,
                        partition_fields,
                    },
                };

                let mut writer = {
                    let w = TableWriter::create_or_open(
                        catalog,
                        Arc::clone(&store),
                        policy,
                        ident,
                        format_version,
                    )
                    .await
                    .map_err(|e| e.to_string())?;
                    if let Some(cfg) = fts_cfg {
                        w.with_fts_config(cfg)
                    } else {
                        w
                    }
                };

                let rows = embs.len();
                match batch_id {
                    Some(ref id) => writer
                        .write_batch_idempotent(&batch, &embs, id)
                        .await
                        .map_err(|e| e.to_string())?,
                    None if deferred => writer
                        .write_batch_deferred(&batch, &embs)
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
            text,
            text_columns,
            hybrid_text,
            text_column,
            bm25_weight,
            top_k,
            pruning_threshold,
            format,
        } => {
            let ident = parse_table_ident(&table);

            // Full-text search path
            if let Some(ref txt) = text {
                let cols_str = text_columns.as_deref().unwrap_or("chunk_text");
                let cols: Vec<&str> = cols_str.split(',').map(str::trim).collect();
                let results = ailake_query::search_text(
                    &ident,
                    txt,
                    &cols,
                    top_k,
                    catalog as Arc<dyn CatalogProvider>,
                    store,
                    None,
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
                                    "score": -r.distance,
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
                        } else {
                            for (i, r) in results.iter().enumerate() {
                                println!(
                                    "{}: row_id={} score={:.4} file={}",
                                    i + 1,
                                    r.row_id.0,
                                    -r.distance,
                                    r.file_path
                                );
                            }
                        }
                    }
                }
                return Ok(());
            }

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
                return Err("either --query, --query-file, or --text is required".into());
            };

            let dim = query_vec.len() as u32;

            let column = match catalog.load_table(&ident).await {
                Ok(meta) => meta
                    .properties
                    .get("ailake.vector-column")
                    .cloned()
                    .unwrap_or_else(|| "embedding".to_string()),
                Err(_) => "embedding".to_string(),
            };

            let hybrid = hybrid_text.map(|txt| HybridConfig {
                query_text: txt,
                text_columns: vec![text_column],
                bm25_weight,
                ..Default::default()
            });

            let config = SearchConfig {
                top_k,
                ef_search: top_k * 5,
                pruning_threshold,
                rerank_factor: None,
                score_fn: None,
                partition_filter: None,
                hybrid,
            };

            let results = ailake_query::search(
                &ident,
                &query_vec,
                config,
                &column,
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
            max_files_per_pass,
            deferred,
            format,
        } => {
            if deferred && !catalog.supports_in_place_rewrite() {
                return Err("--deferred is not supported with this catalog backend: the \
                            background index build patches the merged file in place at its \
                            committed path, which DuckLake cannot re-register (it trusts \
                            the stats and footer size recorded at registration). Run a \
                            blocking compact instead."
                    .into());
            }
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
            let metric = match meta
                .properties
                .get("ailake.vector-metric")
                .map(|s| s.as_str())
                .unwrap_or("cosine")
            {
                "euclidean" => VectorMetric::Euclidean,
                "dotproduct" | "dot_product" => VectorMetric::DotProduct,
                "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
                _ => VectorMetric::Cosine,
            };
            let pre_normalize = meta
                .properties
                .get("ailake.pre-normalize")
                .map(|s| s == "true")
                .unwrap_or(false);
            let hnsw_m = meta
                .properties
                .get("ailake.hnsw-m")
                .and_then(|s| s.parse().ok());
            let hnsw_ef_construction = meta
                .properties
                .get("ailake.hnsw-ef-construction")
                .and_then(|s| s.parse().ok());

            let policy = VectorStoragePolicy {
                column_name: column,
                dim,
                metric,
                precision: VectorPrecision::F16,
                pq: None,
                keep_raw_for_reranking: true,
                pre_normalize,
                hnsw_m,
                hnsw_ef_construction,
                ivf_residual: false,
                embedding_model: None,
                modality: None,
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
            };

            let config = CompactionConfig {
                min_files_to_compact: min_files,
                target_file_size_bytes: target_size,
                index_strategy: Default::default(),
                max_files_per_pass,
            };
            let planner = CompactionPlanner::new(config);
            let executor = CompactionExecutor::new(Arc::clone(&store), policy);

            let result = if deferred {
                executor
                    .run_deferred(
                        &planner,
                        &ident,
                        catalog as Arc<dyn CatalogProvider>,
                        "data",
                    )
                    .await
                    .map_err(|e| e.to_string())?
            } else {
                executor
                    .run(
                        &planner,
                        &ident,
                        catalog as Arc<dyn CatalogProvider>,
                        "data",
                    )
                    .await
                    .map_err(|e| e.to_string())?
            };

            match format {
                OutputFormat::Json => {
                    let json = match &result {
                        Some(entry) => serde_json::json!({
                            "ok": true,
                            "files_compacted": 1,
                            "output_path": entry.path,
                        }),
                        None => serde_json::json!({
                            "ok": true,
                            "files_compacted": 0,
                        }),
                    };
                    println!(
                        "{}",
                        serde_json::to_string(&json).map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => match &result {
                    Some(entry) => println!("compacted into {}", entry.path),
                    None => println!("nothing to compact (no files eligible)"),
                },
            }
            Ok(())
        }

        Commands::DecayMemories {
            table,
            lambda,
            format,
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
            let metric = match meta
                .properties
                .get("ailake.vector-metric")
                .map(|s| s.as_str())
                .unwrap_or("cosine")
            {
                "euclidean" => VectorMetric::Euclidean,
                "dotproduct" | "dot_product" => VectorMetric::DotProduct,
                "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
                _ => VectorMetric::Cosine,
            };
            let policy = VectorStoragePolicy::default_f16(&column, dim, metric);

            let job = MemoryDecayJob::new(
                Arc::clone(&catalog) as Arc<dyn CatalogProvider>,
                Arc::clone(&store),
                policy,
                lambda,
            );
            let files_updated = job.run(&ident).await.map_err(|e| e.to_string())?;

            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string(&serde_json::json!({
                            "ok": true,
                            "files_updated": files_updated,
                        }))
                        .map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => {
                    println!("files_updated: {files_updated}");
                }
            }
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
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
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
            let failed = files
                .iter()
                .filter(|f| f.index_status == ailake_catalog::provider::IndexStatus::Failed)
                .count();
            // Files with no centroid were never written by the AI-Lake SDK — likely a
            // generic Iceberg engine (Spark/Trino OPTIMIZE, DuckDB) rewrote them with no
            // knowledge of AI-Lake. Every query against these degrades to an O(N) flat
            // scan until `ailake compact` repairs them (see CompactionPlanner::plan).
            let foreign: Vec<&str> = files
                .iter()
                .filter(|f| f.is_foreign())
                .map(|f| f.path.as_str())
                .collect();

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
                            "failed_files": failed,
                            "foreign_files": foreign.len(),
                            "foreign_file_paths": foreign,
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
                    if failed > 0 {
                        println!("files:       {file_count} ({ready} indexed, {failed} failed — compaction will rebuild)");
                    } else {
                        println!("files:       {file_count} ({ready} indexed)");
                    }
                    if !foreign.is_empty() {
                        println!(
                            "foreign:     {} file(s) with no AI-Lake index (external rewrite \
                             suspected) — degraded to flat scan, run `ailake compact` to repair",
                            foreign.len()
                        );
                        for path in &foreign {
                            println!("               - {path}");
                        }
                    }
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
                            let mut stdin = child.stdin.take().ok_or_else(|| {
                                std::io::Error::new(
                                    std::io::ErrorKind::BrokenPipe,
                                    "embed-cmd stdin unavailable",
                                )
                            })?;
                            stdin.write_all(input.as_bytes())?;
                            drop(stdin); // close stdin so the child sees EOF
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

        Commands::DeleteRows { table, file, rows } => {
            let ident = parse_table_ident(&table);
            let row_ids: Vec<u32> = rows
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<u32>()
                        .map_err(|e| format!("invalid row id '{}': {e}", s.trim()))
                })
                .collect::<Result<_, _>>()?;

            rs_delete_rows(
                catalog as Arc<dyn CatalogProvider>,
                store,
                &ident,
                &file,
                &row_ids,
            )
            .await
            .map_err(|e| e.to_string())?;

            println!("deleted {} rows from {table} file {file}", row_ids.len());
            Ok(())
        }

        Commands::DeleteWhere { table, col, vals } => {
            use ailake_query::delete_where as rs_delete_where;
            let ident = parse_table_ident(&table);
            let values: Vec<&str> = vals.split(',').map(str::trim).collect();
            rs_delete_where(
                catalog as Arc<dyn CatalogProvider>,
                store,
                &ident,
                &col,
                &values,
            )
            .await
            .map_err(|e| e.to_string())?;
            println!(
                "delete-where committed: {} predicates on column '{col}'",
                values.len()
            );
            Ok(())
        }

        Commands::Evolve {
            table,
            adds,
            initial_defaults,
            renames,
        } => {
            use ailake_catalog::{AddColumnRequest, SchemaEvolution};
            let ident = parse_table_ident(&table);
            let mut evolution = SchemaEvolution::new();

            for (i, add_spec) in adds.iter().enumerate() {
                let (name, iceberg_type) = add_spec
                    .split_once(':')
                    .ok_or_else(|| format!("--add value '{add_spec}' must be NAME:TYPE"))?;
                let initial_default: Option<serde_json::Value> =
                    initial_defaults.get(i).and_then(|s| {
                        serde_json::from_str(s)
                            .map_err(|e| {
                                eprintln!(
                                    "warn: could not parse --initial-default '{}' as JSON: {e}; \
                                     using null",
                                    s
                                );
                                e
                            })
                            .ok()
                    });
                evolution = evolution.add_column(AddColumnRequest {
                    name: name.to_string(),
                    iceberg_type: iceberg_type.to_string(),
                    required: false,
                    initial_default: initial_default.clone(),
                    write_default: initial_default,
                    doc: None,
                });
            }

            for rename_spec in &renames {
                let (old_name, new_name) = rename_spec
                    .split_once(':')
                    .ok_or_else(|| format!("--rename value '{rename_spec}' must be OLD:NEW"))?;
                evolution = evolution.rename_column(old_name, new_name);
            }

            let new_schema_id = catalog
                .evolve_schema(&ident, evolution)
                .await
                .map_err(|e| e.to_string())?;

            println!("new_schema_id: {new_schema_id}");
            Ok(())
        }

        Commands::AddVectorColumn {
            table,
            column,
            dim,
            metric,
            precision,
            pre_normalize,
            hnsw_m,
            hnsw_ef,
        } => {
            use ailake_core::VectorColSpec;
            let ident = parse_table_ident(&table);
            let spec = VectorColSpec {
                column_name: column,
                dim,
                metric: metric.into(),
                precision: precision.into(),
                pre_normalize,
                hnsw_m,
                hnsw_ef_construction: hnsw_ef,
            };
            let new_schema_id = catalog
                .add_vector_column(&ident, &spec)
                .await
                .map_err(|e| e.to_string())?;
            println!(
                "vector column '{}' added — new_schema_id: {new_schema_id}",
                spec.column_name
            );
            Ok(())
        }

        Commands::BackfillVectorColumn {
            table,
            column,
            text_column,
            embed_cmd,
            batch_size,
        } => {
            use ailake_core::VectorColSpec;
            use ailake_query::{BackfillJob, EmbedFn};
            let ident = parse_table_ident(&table);

            // Load new column spec from table properties.
            let table_meta = catalog
                .load_table(&ident)
                .await
                .map_err(|e| e.to_string())?;
            let dim_key = format!("ailake.dim-{column}");
            let metric_key = format!("ailake.metric-{column}");
            let precision_key = format!("ailake.precision-{column}");

            let dim: u32 = table_meta
                .properties
                .get(&dim_key)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("column '{column}' not found — run add-vector-column first")
                })?;

            let metric: VectorMetric = match table_meta
                .properties
                .get(&metric_key)
                .map(|s| s.as_str())
                .unwrap_or("cosine")
            {
                "euclidean" => VectorMetric::Euclidean,
                "dotproduct" | "dot_product" => VectorMetric::DotProduct,
                "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
                _ => VectorMetric::Cosine,
            };

            let precision: VectorPrecision = match table_meta
                .properties
                .get(&precision_key)
                .map(|s| s.as_str())
                .unwrap_or("f16")
            {
                "f32" => VectorPrecision::F32,
                "i8" => VectorPrecision::I8,
                _ => VectorPrecision::F16,
            };

            let new_col = VectorColSpec {
                column_name: column.clone(),
                dim,
                metric,
                precision,
                pre_normalize: false,
                hnsw_m: None,
                hnsw_ef_construction: None,
            };

            let cmd = embed_cmd.clone();
            let embed_fn: EmbedFn = Arc::new(move |texts: &[String]| {
                use std::io::Write;
                use std::process::{Command, Stdio};
                let input = serde_json::to_string(texts)
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let mut child = Command::new("sh")
                    .arg("-c")
                    .arg(&cmd)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(input.as_bytes())
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let out = child
                    .wait_with_output()
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                serde_json::from_slice::<Vec<Vec<f32>>>(&out.stdout).map_err(|e| {
                    ailake_core::AilakeError::InvalidArgument(format!(
                        "embed-cmd output parse error: {e}"
                    ))
                })
            });

            let job = BackfillJob {
                table: ident,
                text_column,
                new_col,
                embed_fn,
                batch_size,
                on_progress: Some(Arc::new(|p: ailake_query::BackfillProgress| {
                    eprintln!(
                        "backfill: {}/{} files done ({} skipped), {} rows",
                        p.files_done, p.files_total, p.files_skipped, p.rows_backfilled
                    );
                })),
            };

            job.run(catalog as Arc<dyn CatalogProvider>, store)
                .await
                .map_err(|e| e.to_string())?;

            println!("backfill complete for column '{column}'");
            Ok(())
        }

        // Handled before store/catalog creation — unreachable here.
        Commands::Estimate { .. } => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type ColSpec = (String, u32, VectorMetric, Option<VectorModality>);

/// Parse "--vector-cols" spec: "col:dim:metric[:modality],..."
/// Returns Vec<(column_name, dim, metric, modality)>.
fn parse_vector_cols(spec: &str) -> Result<Vec<ColSpec>, String> {
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
            let modality = parts.get(3).and_then(|m| m.parse::<VectorModality>().ok());
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

    // Raw vector bytes per row per precision (saturate on overflow — display only).
    let vec_f32 = rows.saturating_mul(dim).saturating_mul(4);
    let vec_f16 = rows.saturating_mul(dim).saturating_mul(2);
    let vec_i8 = rows.saturating_mul(dim);

    // HNSW index: each node stores ~M×2 neighbor IDs (u32 × 2 per layer).
    // Approximation: rows × hnsw_m × 2 × 4 bytes (two u32 per neighbor slot).
    // Real bincode overhead adds ~10-15%; use 18 bytes/neighbor as empirical factor.
    let hnsw_bytes = rows
        .saturating_mul(hnsw_m as u64)
        .saturating_mul(2)
        .saturating_mul(9); // ≈ M×2 neighbors × 9 bytes avg

    // IVF-PQ codes: rows × pq_m bytes (1 byte per sub-quantizer code).
    // Codebook: negligible vs row data for any practical table size.
    let pq_bytes = rows.saturating_mul(pq_m);

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
