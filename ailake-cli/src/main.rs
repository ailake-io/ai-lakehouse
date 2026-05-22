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
    #[arg(long, global = true, env = "AILAKE_STORE_URL")]
    store: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new AI-Lake table
    Create {
        /// Table name
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
    },
    /// Insert a Parquet file into a table
    Insert {
        /// Table name
        table: String,
        /// Path to source Parquet file
        file: String,
        /// Name of the embeddings column in the source file
        #[arg(long, default_value = "embedding")]
        embeddings: String,
    },
    /// Search a table by vector similarity
    Search {
        /// Table name
        table: String,
        /// Query vector as comma-separated floats (e.g. "0.1,0.2,0.3")
        #[arg(long)]
        query: String,
        /// Number of results to return
        #[arg(long, default_value = "10")]
        top_k: usize,
        /// Pruning threshold (geometric; lower = more aggressive pruning)
        #[arg(long, default_value = "0.8")]
        pruning_threshold: f32,
    },
    /// Compact small files in a table into a single larger file
    Compact {
        /// Table name
        table: String,
        /// Target file size in bytes
        #[arg(long, default_value = "536870912")]
        target_size: u64,
    },
    /// Print table statistics (file count, row count, index type)
    Info {
        /// Table name
        table: String,
    },
}

#[derive(ValueEnum, Clone)]
enum Metric {
    Cosine,
    Euclidean,
    Dot,
}

#[derive(ValueEnum, Clone)]
enum Precision {
    F32,
    F16,
    I8,
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

    let result = run(cli).await;
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Create {
            table,
            dim,
            metric,
            precision,
        } => {
            let metric_s = match metric {
                Metric::Cosine => "cosine",
                Metric::Euclidean => "euclidean",
                Metric::Dot => "dot",
            };
            let prec_s = match precision {
                Precision::F32 => "f32",
                Precision::F16 => "f16",
                Precision::I8 => "i8",
            };
            eprintln!(
                "not yet implemented: create table={table} dim={dim} metric={metric_s} precision={prec_s}"
            );
            Err("not yet implemented".into())
        }
        Commands::Insert {
            table,
            file,
            embeddings,
        } => {
            eprintln!(
                "not yet implemented: insert table={table} file={file} embeddings={embeddings}"
            );
            Err("not yet implemented".into())
        }
        Commands::Search {
            table,
            query,
            top_k,
            pruning_threshold,
        } => {
            eprintln!(
                "not yet implemented: search table={table} top_k={top_k} \
                 pruning_threshold={pruning_threshold} query_len={}",
                query.split(',').count()
            );
            Err("not yet implemented".into())
        }
        Commands::Compact { table, target_size } => {
            eprintln!("not yet implemented: compact table={table} target_size={target_size}");
            Err("not yet implemented".into())
        }
        Commands::Info { table } => {
            eprintln!("not yet implemented: info table={table}");
            Err("not yet implemented".into())
        }
    }
}
