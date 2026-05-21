use ailake_core::{AilakeError, AilakeResult, VectorMetric};
use serde::{Deserialize, Serialize};

use crate::hnsw::{HnswConfig, HnswIndex};

#[derive(Serialize, Deserialize)]
struct HnswSnapshot {
    m: usize,
    ef_construction: usize,
    max_elements: usize,
    metric: u8,
    dim: u32,
    /// Row IDs parallel to flat_vecs (one entry per vector).
    row_ids: Vec<u64>,
    /// Contiguous vector storage: flat_vecs[i*dim..(i+1)*dim] = vector i.
    flat_vecs: Vec<f32>,
    // Graph structure (empty = old format, triggers brute-force fallback)
    neighbors: Vec<Vec<Vec<usize>>>,
    node_levels: Vec<usize>,
    entry_point: Option<usize>,
    max_layer: usize,
}

fn metric_to_u8(m: VectorMetric) -> u8 {
    match m {
        VectorMetric::Cosine => 0,
        VectorMetric::Euclidean => 1,
        VectorMetric::DotProduct => 2,
    }
}

fn u8_to_metric(v: u8) -> AilakeResult<VectorMetric> {
    match v {
        0 => Ok(VectorMetric::Cosine),
        1 => Ok(VectorMetric::Euclidean),
        2 => Ok(VectorMetric::DotProduct),
        _ => Err(AilakeError::Catalog(format!("unknown metric byte: {v}"))),
    }
}

pub struct HnswSerializer;

impl HnswSerializer {
    pub fn to_bytes(index: &HnswIndex) -> AilakeResult<Vec<u8>> {
        let snap = HnswSnapshot {
            m: index.config.m,
            ef_construction: index.config.ef_construction,
            max_elements: index.config.max_elements,
            metric: metric_to_u8(index.metric),
            dim: index.dim,
            row_ids: index.row_ids.clone(),
            flat_vecs: index.flat_vecs.clone(),
            neighbors: index.neighbors.clone(),
            node_levels: index.node_levels.clone(),
            entry_point: index.entry_point,
            max_layer: index.max_layer,
        };
        bincode::serialize(&snap).map_err(|e| AilakeError::Bincode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<HnswIndex> {
        let snap: HnswSnapshot =
            bincode::deserialize(bytes).map_err(|e| AilakeError::Bincode(e.to_string()))?;
        let metric = u8_to_metric(snap.metric)?;
        Ok(HnswIndex {
            config: HnswConfig {
                m: snap.m,
                ef_construction: snap.ef_construction,
                max_elements: snap.max_elements,
            },
            metric,
            dim: snap.dim,
            row_ids: snap.row_ids,
            flat_vecs: snap.flat_vecs,
            flat_vecs_f16: None, // populated at runtime by quantize_to_f16() if needed
            neighbors: snap.neighbors,
            node_levels: snap.node_levels,
            entry_point: snap.entry_point,
            max_layer: snap.max_layer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswBuilder;
    use ailake_core::RowId;

    #[test]
    fn serialize_roundtrip() {
        let mut b = HnswBuilder::new(3, VectorMetric::Cosine, Default::default());
        b.insert(RowId::new(0), vec![1.0, 0.0, 0.0]);
        b.insert(RowId::new(1), vec![0.0, 1.0, 0.0]);
        let idx = b.build();
        let bytes = HnswSerializer::to_bytes(&idx).unwrap();
        let idx2 = HnswSerializer::from_bytes(&bytes).unwrap();
        assert_eq!(idx2.node_count(), 2);
        assert_eq!(idx2.dim(), 3);
        let r = idx2.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(r[0].0, RowId::new(0));
    }

    #[test]
    fn serialize_preserves_graph() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        let mut b = HnswBuilder::new(8, VectorMetric::Euclidean, Default::default());
        for i in 0..50u64 {
            let v: Vec<f32> = (0..8).map(|_| rng.gen::<f32>()).collect();
            b.insert(RowId::new(i), v);
        }
        let idx = b.build();
        let query: Vec<f32> = (0..8).map(|_| rng.gen::<f32>()).collect();
        let r1 = idx.search(&query, 5, 50);

        let bytes = HnswSerializer::to_bytes(&idx).unwrap();
        let idx2 = HnswSerializer::from_bytes(&bytes).unwrap();
        let r2 = idx2.search(&query, 5, 50);

        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0);
        }
    }
}
