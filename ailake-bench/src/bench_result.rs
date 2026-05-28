// SPDX-License-Identifier: MIT OR Apache-2.0
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
        println!(
            "  Index build   : {:.1} s (separate step)",
            r.index_build_secs
        );
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

/// Print a side-by-side comparison of two results (AI-Lake vs one other engine).
#[cfg(feature = "lancedb-bench")]
#[allow(dead_code)]
pub fn print_comparison(a: &BenchResult, b: &BenchResult, top_k: usize) {
    print_multi_comparison(&[a, b], top_k);
}

/// Print a multi-engine comparison table for N ≥ 2 results.
#[cfg(feature = "lancedb-bench")]
pub fn print_multi_comparison(results: &[&BenchResult], top_k: usize) {
    let n = results.len();
    let lw = 22usize; // label column width
    let cw = 20usize; // value column width
    let total_w = lw + n * (cw + 2);

    println!();
    println!("Comparison — SIFT-1M (128D Euclidean, top_k={top_k})");
    println!("{}", "═".repeat(total_w));

    // Header
    let mut header = format!("{:<lw$}", "Metric");
    for r in results {
        header.push_str(&format!("  {:>cw$}", truncate(&r.engine, cw)));
    }
    println!("{header}");
    println!("{}", "─".repeat(total_w));

    // Rows
    let write_vals: Vec<String> = results
        .iter()
        .map(|r| format!("{:.0} vec/s", r.write_vec_per_sec))
        .collect();
    print_row("Write throughput", &write_vals, lw, cw);

    if results.iter().any(|r| r.index_build_secs > 0.0) {
        let ib: Vec<String> = results
            .iter()
            .map(|r| {
                if r.index_build_secs > 0.0 {
                    format!("{:.1} s", r.index_build_secs)
                } else {
                    "incl. write".to_string()
                }
            })
            .collect();
        print_row("Index build", &ib, lw, cw);
    }

    let load: Vec<String> = results
        .iter()
        .map(|r| format!("{:.2} s", r.load_secs))
        .collect();
    print_row("Index load", &load, lw, cw);

    let recall: Vec<String> = results.iter().map(|r| format!("{:.4}", r.recall)).collect();
    print_row(&format!("Recall@{top_k}"), &recall, lw, cw);

    // QPS: highlight fastest engine
    let max_qps = results.iter().map(|r| r.qps).fold(0.0_f64, f64::max);
    let qps_vals: Vec<String> = results
        .iter()
        .map(|r| {
            let s = format!("{:.0}", r.qps);
            if results.len() > 1 && (r.qps - max_qps).abs() < 0.5 {
                format!("{s} ◀")
            } else {
                s
            }
        })
        .collect();
    print_row("QPS", &qps_vals, lw, cw);

    let mean: Vec<String> = results
        .iter()
        .map(|r| format!("{:.3} ms", r.mean_ms))
        .collect();
    print_row("Latency mean", &mean, lw, cw);

    let p50: Vec<String> = results
        .iter()
        .map(|r| format!("{:.3} ms", r.p50_ms))
        .collect();
    print_row("Latency p50", &p50, lw, cw);

    let p95: Vec<String> = results
        .iter()
        .map(|r| format!("{:.3} ms", r.p95_ms))
        .collect();
    print_row("Latency p95", &p95, lw, cw);

    let p99: Vec<String> = results
        .iter()
        .map(|r| format!("{:.3} ms", r.p99_ms))
        .collect();
    print_row("Latency p99", &p99, lw, cw);

    println!("{}", "─".repeat(total_w));
    println!();
    println!("Notes:");
    println!("  - Same hardware, same 10k queries, same ground truth");
    for r in results {
        if r.engine.starts_with("AI-Lake") {
            println!("  - AI-Lake: 10 shards, HNSW deferred async, indexes pre-loaded");
        } else if r.engine.starts_with("LanceDB") {
            println!("  - LanceDB: IvfHnswSq, concurrent queries (see --lancedb-concurrency)");
        } else if r.engine.starts_with("pgvector") {
            println!("  - pgvector: HNSW index, sequential queries, single connection");
        }
    }
    println!();
}

#[cfg(feature = "lancedb-bench")]
fn print_row(label: &str, vals: &[String], lw: usize, cw: usize) {
    let mut line = format!("{:<lw$}", label);
    for v in vals {
        line.push_str(&format!("  {:>cw$}", v));
    }
    println!("{line}");
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
