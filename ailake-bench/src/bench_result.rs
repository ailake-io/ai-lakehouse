/// Results from one benchmark engine run.
pub struct BenchResult {
    pub engine: String,
    /// Seconds to write all vectors (including per-shard HNSW build for AI-Lake).
    pub write_secs: f64,
    pub write_vec_per_sec: f64,
    /// Seconds to build a separate index step (0.0 if built during write).
    pub index_build_secs: f64,
    /// Seconds to load / open indexes into memory before the search loop.
    pub load_secs: f64,
    pub recall: f64,
    pub qps: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Print a single-engine result block.
pub fn print_single(r: &BenchResult, total: usize, top_k: usize) {
    println!("{} — SIFT-1M ({}D, Euclidean)", r.engine, 128);
    println!("{}", "=".repeat(58));
    println!("Dataset    {:>10} base vectors", fmt_int(total));
    println!();
    println!("Write phase");
    println!("  Wall time     : {:.1} s", r.write_secs);
    println!("  Throughput    : {:.0} vec/s", r.write_vec_per_sec);
    if r.index_build_secs > 0.0 {
        println!("  Index build   : {:.1} s (separate step)", r.index_build_secs);
    }
    println!();
    println!("Index load");
    println!("  Load time     : {:.2} s", r.load_secs);
    println!();
    println!("Search phase  (top_k={top_k})");
    println!("  Recall@{top_k}     : {:.4}", r.recall);
    println!("  QPS           : {:.0}", r.qps);
    println!("  Latency mean  : {:.3} ms", r.mean_ms);
    println!("  Latency p50   : {:.3} ms", r.p50_ms);
    println!("  Latency p95   : {:.3} ms", r.p95_ms);
    println!("  Latency p99   : {:.3} ms", r.p99_ms);
    println!();
}

/// Print a side-by-side comparison of two results.
#[cfg(feature = "lancedb-bench")]
pub fn print_comparison(a: &BenchResult, b: &BenchResult, top_k: usize) {
    let w = 22usize;
    let cw = 20usize;

    println!();
    println!(
        "Comparison — SIFT-1M (128D Euclidean, top_k={top_k})"
    );
    println!("{}", "═".repeat(66));
    println!(
        "{:<w$}  {:>cw$}  {:>cw$}",
        "Metric", &truncate(&a.engine, cw), &truncate(&b.engine, cw)
    );
    println!("{}", "─".repeat(66));

    let write_a = format!("{:.0} vec/s", a.write_vec_per_sec);
    let write_b = format!("{:.0} vec/s", b.write_vec_per_sec);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Write throughput", write_a, write_b);

    // Index build: only show if any engine has a separate build step
    if a.index_build_secs > 0.0 || b.index_build_secs > 0.0 {
        let ib_a = if a.index_build_secs > 0.0 {
            format!("{:.1} s", a.index_build_secs)
        } else {
            "incl. write".to_string()
        };
        let ib_b = if b.index_build_secs > 0.0 {
            format!("{:.1} s", b.index_build_secs)
        } else {
            "incl. write".to_string()
        };
        println!("{:<w$}  {:>cw$}  {:>cw$}", "Index build", ib_a, ib_b);
    }

    let load_a = format!("{:.2} s", a.load_secs);
    let load_b = format!("{:.2} s", b.load_secs);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Index load", load_a, load_b);

    let recall_a = format!("{:.4}", a.recall);
    let recall_b = format!("{:.4}", b.recall);
    println!("{:<w$}  {:>cw$}  {:>cw$}", &format!("Recall@{top_k}"), recall_a, recall_b);

    let qps_a = format!("{:.0}", a.qps);
    let qps_b = format!("{:.0}", b.qps);
    let qps_delta = delta_str(a.qps, b.qps);
    println!(
        "{:<w$}  {:>cw$}  {:>cw$}  {}",
        "QPS", qps_a, qps_b, qps_delta
    );

    let mean_a = format!("{:.3} ms", a.mean_ms);
    let mean_b = format!("{:.3} ms", b.mean_ms);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Latency mean", mean_a, mean_b);

    let p50_a = format!("{:.3} ms", a.p50_ms);
    let p50_b = format!("{:.3} ms", b.p50_ms);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Latency p50", p50_a, p50_b);

    let p95_a = format!("{:.3} ms", a.p95_ms);
    let p95_b = format!("{:.3} ms", b.p95_ms);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Latency p95", p95_a, p95_b);

    let p99_a = format!("{:.3} ms", a.p99_ms);
    let p99_b = format!("{:.3} ms", b.p99_ms);
    println!("{:<w$}  {:>cw$}  {:>cw$}", "Latency p99", p99_a, p99_b);

    println!("{}", "─".repeat(66));
    println!();
    println!("Notes:");
    println!("  - Same hardware, same 10k queries, same ground truth");
    println!("  - AI-Lake: 10 shards, HNSW built during write, indexes pre-loaded");
    println!("  - LanceDB: single table, IvfHnswSq, index built separately");
    println!("  - AI-Lake QPS = sequential; LanceDB QPS = concurrent (see --lancedb-concurrency)");
    println!();
}

#[cfg(feature = "lancedb-bench")]
fn delta_str(a: f64, b: f64) -> String {
    if b == 0.0 {
        return String::new();
    }
    let ratio = a / b;
    if ratio > 1.0 {
        format!("(AI-Lake {:.1}×)", ratio)
    } else {
        format!("(LanceDB {:.1}×)", 1.0 / ratio)
    }
}

#[cfg(feature = "lancedb-bench")]
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

pub fn fmt_int(n: usize) -> String {
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
