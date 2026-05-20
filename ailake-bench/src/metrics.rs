//! Recall@k and latency statistics.

use std::collections::HashSet;

/// Fraction of true top-k neighbors found in `result_ids[..k]`.
///
/// Both slices may be longer than k; only the first k elements are used.
pub fn recall_at_k(result_ids: &[u32], ground_truth: &[u32], k: usize) -> f64 {
    let found: HashSet<u32> = result_ids.iter().take(k).cloned().collect();
    let truth: HashSet<u32> = ground_truth.iter().take(k).cloned().collect();
    if truth.is_empty() {
        return 0.0;
    }
    found.intersection(&truth).count() as f64 / truth.len() as f64
}

#[derive(Debug)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    /// Queries per second measured over the full query wall time.
    pub qps: f64,
}

impl LatencyStats {
    /// `latencies_us` — per-query latencies in microseconds (modified in place for sorting).
    /// `total_wall_ns` — total elapsed wall time in nanoseconds for all queries.
    pub fn compute(latencies_us: &mut [u64], total_wall_ns: u64) -> Self {
        latencies_us.sort_unstable();
        let n = latencies_us.len();
        let mean_us = if n == 0 {
            0.0
        } else {
            latencies_us.iter().sum::<u64>() as f64 / n as f64
        };
        let p50_us = latencies_us[n / 2] as f64;
        let p95_us = latencies_us[n * 95 / 100] as f64;
        let p99_us = latencies_us[n * 99 / 100] as f64;
        let total_sec = total_wall_ns as f64 / 1e9;
        Self {
            mean_ms: mean_us / 1000.0,
            p50_ms: p50_us / 1000.0,
            p95_ms: p95_us / 1000.0,
            p99_ms: p99_us / 1000.0,
            qps: n as f64 / total_sec,
        }
    }
}
