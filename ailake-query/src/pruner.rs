// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::HashMap;

use ailake_catalog::{decode_centroid, DataFileEntry};
use ailake_core::VectorMetric;
use ailake_vec::{cosine_distance, dot_product, euclidean_distance};
use tracing::debug;

pub struct VectorPruner;

impl VectorPruner {
    /// Remove files whose centroid is geometrically guaranteed to contain no vectors
    /// within `threshold` distance of `query`.
    ///
    /// Pruning condition: `distance(query, centroid) - radius > threshold`
    /// Files without centroid metadata are kept (conservative fallback).
    pub fn prune(
        files: Vec<DataFileEntry>,
        query: &[f32],
        metric: VectorMetric,
        threshold: f32,
    ) -> Vec<DataFileEntry> {
        files
            .into_iter()
            .filter(|entry| {
                match decode_centroid(entry, metric) {
                    Some(centroid) => {
                        // Centroid is stored for the primary column. When searching a
                        // secondary column with a different dimension (multimodal), dims
                        // won't match — skip pruning and keep the file conservatively.
                        if centroid.values.len() != query.len() {
                            debug!(
                                "ailake: pruner {} — centroid dim={} != query dim={}, skipping (secondary column)",
                                entry.path,
                                centroid.values.len(),
                                query.len(),
                            );
                            return true;
                        }
                        let dist = compute_distance(query, &centroid.values, metric);
                        let keep = dist - centroid.radius <= threshold;
                        debug!(
                            "ailake: pruner {} — dist={:.4} radius={:.4} edge={:.4} threshold={:.4} → {}",
                            entry.path,
                            dist,
                            centroid.radius,
                            dist - centroid.radius,
                            threshold,
                            if keep { "KEEP" } else { "PRUNE" }
                        );
                        keep
                    }
                    None => {
                        debug!(
                            "ailake: pruner {} — no centroid metadata, keeping (conservative fallback)",
                            entry.path
                        );
                        true // no centroid → keep (safe fallback)
                    }
                }
            })
            .collect()
    }
}

fn compute_distance(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    match metric {
        VectorMetric::Cosine | VectorMetric::NormalizedCosine => cosine_distance(a, b),
        VectorMetric::Euclidean => euclidean_distance(a, b),
        VectorMetric::DotProduct => -dot_product(a, b),
    }
}

/// File-level BM25 Bloom filter pruner (Phase F).
///
/// Given a map of `file_path → BloomFilter` loaded from the Puffin stats file,
/// removes files where no query term can possibly appear. Zero false negatives:
/// if a term is in the file, the Bloom filter will return `true`. Files without
/// a Bloom filter entry are kept (conservative fallback for V2 tables or files
/// written before Phase F).
pub struct BloomPruner;

impl BloomPruner {
    /// Skip files whose Bloom filter guarantees no query term is present.
    ///
    /// Returns the subset of `files` that *may* contain at least one query term.
    /// Files absent from `bloom_map` are always kept.
    pub fn prune(
        files: Vec<DataFileEntry>,
        query_text: &str,
        bloom_map: &HashMap<String, crate::bloom::BloomFilter>,
    ) -> Vec<DataFileEntry> {
        let query_terms: Vec<String> = crate::bm25::tokenize(query_text);
        if query_terms.is_empty() || bloom_map.is_empty() {
            return files;
        }
        let before = files.len();
        let surviving: Vec<DataFileEntry> = files
            .into_iter()
            .filter(|entry| match bloom_map.get(&entry.path) {
                Some(bloom) => {
                    let keep = query_terms.iter().any(|t| bloom.may_contain(t));
                    debug!(
                        "ailake: bloom pruner {} — {} query terms, keep={}",
                        entry.path,
                        query_terms.len(),
                        keep
                    );
                    keep
                }
                None => true,
            })
            .collect();
        debug!(
            "ailake: bloom pruning — {}/{} files survive",
            surviving.len(),
            before
        );
        surviving
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::{make_data_file_entry, VectorIndexInfo};
    use ailake_core::VectorMetric;
    use ailake_vec::compute_centroid_and_radius;

    fn make_entry(path: &str, vecs: &[Vec<f32>], metric: VectorMetric) -> DataFileEntry {
        let centroid = compute_centroid_and_radius(vecs, metric);
        make_data_file_entry(
            path,
            vecs.len() as u64,
            1024,
            &centroid,
            VectorIndexInfo {
                column: "embedding",
                dim: vecs[0].len() as u32,
                hnsw_offset: 0,
                hnsw_len: 0,
            },
        )
    }

    #[test]
    fn prunes_far_file() {
        // File centroid near [1,0,0], query near [0,0,1] — orthogonal → prune
        let vecs = vec![vec![1.0f32, 0.0, 0.0], vec![0.9, 0.1, 0.0]];
        let entry = make_entry("far.parquet", &vecs, VectorMetric::Cosine);
        let query = vec![0.0f32, 0.0, 1.0];
        let pruned = VectorPruner::prune(vec![entry], &query, VectorMetric::Cosine, 0.1);
        assert!(pruned.is_empty(), "far file should be pruned");
    }

    #[test]
    fn keeps_nearby_file() {
        let vecs = vec![vec![1.0f32, 0.0, 0.0], vec![0.99, 0.1, 0.0]];
        let entry = make_entry("near.parquet", &vecs, VectorMetric::Cosine);
        let query = vec![1.0f32, 0.0, 0.0];
        let kept = VectorPruner::prune(vec![entry], &query, VectorMetric::Cosine, 0.5);
        assert_eq!(kept.len(), 1, "nearby file should be kept");
    }

    #[test]
    fn no_centroid_always_kept() {
        let entry = DataFileEntry {
            path: "unknown.parquet".into(),
            record_count: 10,
            file_size_bytes: 512,
            centroid_b64: None,
            radius: None,
            hnsw_offset: None,
            hnsw_len: None,
            vector_column: None,
            vector_dim: None,
            extra_vector_indexes: vec![],
            index_status: ailake_catalog::IndexStatus::Ready,
            index_error: None,
            batch_id: None,
            embedding_model: None,
            partition_value: None,
            deletion_vector: None,
            first_row_id: None,
            column_stats: None,
        };
        let query = vec![0.0f32, 0.0, 1.0];
        let kept = VectorPruner::prune(vec![entry], &query, VectorMetric::Cosine, 0.0);
        assert_eq!(kept.len(), 1);
    }
}
