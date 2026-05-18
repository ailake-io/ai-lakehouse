// Serialize HnswIndex as (config, metric, dim, vectors) via bincode.
// Phase 2: replace with hnsw_rs graph dump for faster load times.

use ailake_core::{AilakeError, AilakeResult, RowId, VectorMetric};
use serde::{Deserialize, Serialize};

use crate::hnsw::{HnswConfig, HnswIndex};

#[derive(Serialize, Deserialize)]
struct HnswSnapshot {
    m: usize,
    ef_construction: usize,
    max_elements: usize,
    metric: u8,
    dim: u32,
    // (row_id_u64, flat_f32_bytes)
    vectors: Vec<(u64, Vec<f32>)>,
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
            vectors: index
                .vectors
                .iter()
                .map(|(id, v)| (id.as_u64(), v.clone()))
                .collect(),
        };
        bincode::serialize(&snap).map_err(|e| AilakeError::Bincode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<HnswIndex> {
        let snap: HnswSnapshot =
            bincode::deserialize(bytes).map_err(|e| AilakeError::Bincode(e.to_string()))?;
        let metric = u8_to_metric(snap.metric)?;
        let config = HnswConfig {
            m: snap.m,
            ef_construction: snap.ef_construction,
            max_elements: snap.max_elements,
        };
        let vectors: Vec<(RowId, Vec<f32>)> = snap
            .vectors
            .into_iter()
            .map(|(id, v)| (RowId::new(id), v))
            .collect();
        Ok(HnswIndex {
            config,
            metric,
            dim: snap.dim,
            vectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswBuilder;

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
}
