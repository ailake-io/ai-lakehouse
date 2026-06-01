// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU data integration tests.
//!
//! Skipped automatically when AILAKE_GPU_BACKEND=none (or unset).
//! On a runner with CUDA or ROCm installed, these tests fire real cuBLAS /
//! hipBLAS kernels against synthetic but realistic-sized datasets and assert
//! that GPU results match CPU brute-force ground truth within a tight recall
//! threshold.

use std::collections::HashSet;

use ailake_core::{RowId, VectorMetric};
use ailake_index::gpu::{try_nvidia_kmeans, try_nvidia_search_batch, try_rocm_kmeans, try_rocm_search_batch};

fn gpu_backend() -> String {
    std::env::var("AILAKE_GPU_BACKEND").unwrap_or_else(|_| "none".into())
}

// Deterministic LCG vectors in [-1, 1].
fn gen_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

// CPU brute-force cosine nearest-neighbours — ground truth for recall.
fn cpu_topk_cosine(query: &[f32], db: &[Vec<f32>], top_k: usize) -> HashSet<u64> {
    let qnorm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut scored: Vec<(usize, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            let vnorm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = if qnorm > 1e-8 && vnorm > 1e-8 { dot / (qnorm * vnorm) } else { 0.0 };
            (i, 1.0 - cos) // cosine distance
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(i, _)| i as u64).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Fire cuBLAS / hipBLAS SGEMM on 2 000 vectors × dim 128, 20 queries.
/// GPU recall@10 vs CPU brute-force must reach ≥ 99 %.
#[test]
fn gpu_search_recall_vs_cpu_baseline() {
    let backend = gpu_backend();
    if backend == "none" {
        println!("AILAKE_GPU_BACKEND=none — skipping gpu_search_recall_vs_cpu_baseline");
        return;
    }

    let n = 2_000usize;
    let dim = 128usize;
    let n_queries = 20usize;
    let top_k = 10usize;

    let db = gen_vecs(n, dim, 42);
    let query_vecs = gen_vecs(n_queries, dim, 99);
    let flat: Vec<f32> = db.iter().flat_map(|v| v.iter().copied()).collect();
    let row_ids: Vec<u64> = (0..n as u64).collect();
    let query_slices: Vec<&[f32]> = query_vecs.iter().map(|q| q.as_slice()).collect();

    let gpu_results = match backend.as_str() {
        "cuda" => try_nvidia_search_batch(&query_slices, &row_ids, &flat, dim, VectorMetric::Cosine, top_k),
        "rocm" => try_rocm_search_batch(&query_slices, &row_ids, &flat, dim, VectorMetric::Cosine, top_k),
        other => panic!("unknown AILAKE_GPU_BACKEND={other}"),
    }
    .expect("GPU search returned None — is the GPU driver running?");

    assert_eq!(gpu_results.len(), n_queries, "one result list per query");

    let mut total_hits = 0usize;
    for (qi, gpu_top) in gpu_results.iter().enumerate() {
        let cpu_set = cpu_topk_cosine(&query_vecs[qi], &db, top_k);
        let gpu_set: HashSet<u64> = gpu_top.iter().map(|(r, _)| r.as_u64()).collect();
        total_hits += gpu_set.intersection(&cpu_set).count();
    }

    let recall = total_hits as f32 / (n_queries * top_k) as f32;
    println!(
        "GPU recall@{top_k} vs CPU brute-force: {:.1}% ({total_hits}/{} hits, n={n}, dim={dim})",
        recall * 100.0,
        n_queries * top_k,
    );
    assert!(
        recall >= 0.99,
        "GPU batch search recall@{top_k} must be ≥99%, got {:.1}%",
        recall * 100.0,
    );
}

/// Query == a database vector → GPU top-1 must be that exact row, dist ≈ 0.
#[test]
fn gpu_search_exact_hit_in_large_db() {
    let backend = gpu_backend();
    if backend == "none" {
        println!("AILAKE_GPU_BACKEND=none — skipping gpu_search_exact_hit_in_large_db");
        return;
    }

    let n = 5_000usize;
    let dim = 64usize;
    let anchor = 1_337usize; // row we'll use as query

    let db = gen_vecs(n, dim, 7);
    let flat: Vec<f32> = db.iter().flat_map(|v| v.iter().copied()).collect();
    let row_ids: Vec<u64> = (0..n as u64).collect();
    let q = db[anchor].clone();
    let queries: &[&[f32]] = &[q.as_slice()];

    let got = match backend.as_str() {
        "cuda" => try_nvidia_search_batch(queries, &row_ids, &flat, dim, VectorMetric::Cosine, 5),
        "rocm" => try_rocm_search_batch(queries, &row_ids, &flat, dim, VectorMetric::Cosine, 5),
        other => panic!("unknown AILAKE_GPU_BACKEND={other}"),
    }
    .expect("GPU exact-hit search returned None");

    let (top_row, top_dist) = got[0][0];
    assert_eq!(
        top_row,
        RowId::new(anchor as u64),
        "top-1 must be the anchor row {anchor}, got {top_row:?}",
    );
    assert!(
        top_dist < 1e-3,
        "cosine dist to self must be ≈0, got {top_dist}",
    );
}

/// GPU k-means on 8 well-separated clusters × 50 vectors, dim 32.
/// Each returned centroid must match a distinct cluster mean within ε = 1.0.
#[test]
fn gpu_kmeans_converges_on_clustered_data() {
    let backend = gpu_backend();
    if backend == "none" {
        println!("AILAKE_GPU_BACKEND=none — skipping gpu_kmeans_converges_on_clustered_data");
        return;
    }

    let k = 8usize;
    let dim = 32usize;
    let per_cluster = 50usize;

    // Clusters are spaced 100 units apart in the first dimension — trivially separable.
    let vecs: Vec<Vec<f32>> = (0..k)
        .flat_map(|c| {
            (0..per_cluster).map(move |j| {
                (0..dim)
                    .map(|d| c as f32 * 100.0 + d as f32 * 0.01 + (c * per_cluster + j + d) as f32 * 0.001)
                    .collect()
            })
        })
        .collect();

    let cluster_means: Vec<Vec<f32>> = (0..k)
        .map(|c| {
            let mut mean = vec![0.0f32; dim];
            for v in &vecs[c * per_cluster..(c + 1) * per_cluster] {
                for (d, &x) in v.iter().enumerate() {
                    mean[d] += x;
                }
            }
            mean.iter_mut().for_each(|x| *x /= per_cluster as f32);
            mean
        })
        .collect();

    let centroids = match backend.as_str() {
        "cuda" => try_nvidia_kmeans(&vecs, k, 30),
        "rocm" => try_rocm_kmeans(&vecs, k, 30),
        other => panic!("unknown AILAKE_GPU_BACKEND={other}"),
    }
    .expect("GPU k-means returned None");

    assert_eq!(centroids.len(), k, "expected {k} centroids, got {}", centroids.len());

    // Every centroid must uniquely map to a cluster mean within ε.
    let mut matched: HashSet<usize> = HashSet::new();
    for c in &centroids {
        let (best_idx, best_dist) = cluster_means
            .iter()
            .enumerate()
            .map(|(i, mean)| {
                let d: f32 = c.iter().zip(mean).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                (i, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(
            best_dist < 1.0,
            "centroid is not close to any cluster mean (min dist = {best_dist:.3})",
        );
        matched.insert(best_idx);
    }
    assert_eq!(
        matched.len(),
        k,
        "each GPU centroid must map to a distinct cluster, got {} unique matches",
        matched.len(),
    );
}
