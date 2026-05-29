// SPDX-License-Identifier: MIT OR Apache-2.0
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
        VectorMetric::Cosine => cosine_distance(a, b),
        VectorMetric::Euclidean => euclidean_distance(a, b),
        VectorMetric::DotProduct => -dot_product(a, b),
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
            batch_id: None,
        };
        let query = vec![0.0f32, 0.0, 1.0];
        let kept = VectorPruner::prune(vec![entry], &query, VectorMetric::Cosine, 0.0);
        assert_eq!(kept.len(), 1);
    }
}
