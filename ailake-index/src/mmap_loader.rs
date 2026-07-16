// SPDX-License-Identifier: MIT OR Apache-2.0
use std::io::Write;

use ailake_core::{AilakeError, AilakeResult};
use memmap2::Mmap;
use tracing::debug;

use crate::hnsw::HnswIndex;
use crate::serialize::HnswSerializer;

pub struct MmapLoader;

impl MmapLoader {
    /// Write `bytes` to a temporary file, mmap it, and deserialize the HNSW index.
    /// Using mmap lets the OS lazily page in only the graph nodes touched during search —
    /// critical for large indexes (>1 GB) where loading the full file would waste RAM.
    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<HnswIndex> {
        debug!(
            "ailake: loading HNSW index via mmap ({} bytes)",
            bytes.len()
        );
        let mut tmp = tempfile::tempfile().map_err(|e| {
            AilakeError::Store(format!("failed to create tempfile for HNSW mmap: {e}"))
        })?;
        tmp.write_all(bytes).map_err(|e| {
            AilakeError::Store(format!(
                "failed to write {} bytes to HNSW tempfile: {e}",
                bytes.len()
            ))
        })?;
        // SAFETY: the backing file is not modified after mmap is created.
        // The mmap is dropped before the function returns (index owns its data).
        let mmap = unsafe { Mmap::map(&tmp) }.map_err(|e| {
            AilakeError::Store(format!(
                "mmap failed for HNSW tempfile ({} bytes): {e}",
                bytes.len()
            ))
        })?;
        let idx = HnswSerializer::from_bytes(&mmap)?;
        debug!(
            "ailake: HNSW index loaded — {} nodes via mmap",
            idx.node_count()
        );
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswBuilder, HnswConfig};
    use crate::serialize::HnswSerializer;
    use ailake_core::{RowId, VectorMetric};
    use proptest::prelude::*;
    use rand::Rng;

    #[test]
    fn mmap_roundtrip() {
        let mut b = HnswBuilder::new(4, VectorMetric::Cosine, HnswConfig::default());
        b.insert(RowId::new(0), vec![1.0, 0.0, 0.0, 0.0]);
        b.insert(RowId::new(1), vec![0.0, 1.0, 0.0, 0.0]);
        let idx = b.build();
        let bytes = HnswSerializer::to_bytes(&idx).unwrap();

        let loaded = MmapLoader::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.node_count(), 2);
        let r = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1, 50);
        assert_eq!(r[0].0, RowId::new(0));
    }

    // ── Fuzz-style: property tests for mmap loader ────────────────────────

    fn arb_query(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        let val = (-10.0f32..10.0).prop_filter("no NaN/Inf", |x| x.is_finite());
        proptest::collection::vec(val, dim)
    }

    proptest! {
        #[test]
        fn prop_mmap_roundtrip_search(
            n_nodes in 2usize..10,
            m in 4usize..8,
            query in arb_query(4),
        ) {
            let dim = 4u32;
            let m = m.max(2);
            let mut rng = rand::thread_rng();

            let mut b = HnswBuilder::new(dim, VectorMetric::Cosine, HnswConfig { m, ef_construction: 50, max_elements: n_nodes.max(10) });
            let mut inserted: Vec<(RowId, Vec<f32>)> = Vec::new();
            for i in 0..n_nodes {
                let v: Vec<f32> = (0..dim as usize).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let rid = RowId::new(i as u64);
                b.insert(rid, v.clone());
                inserted.push((rid, v));
            }
            let idx = b.build();
            let bytes = HnswSerializer::to_bytes(&idx)
                .expect("serialization should succeed");

            let loaded = MmapLoader::from_bytes(&bytes)
                .expect("mmap load should succeed");
            prop_assert_eq!(loaded.node_count(), n_nodes as u64);

            let results = loaded.search(&query, 1, 50);
            if !results.is_empty() {
                prop_assert!(results[0].1 >= 0.0, "distance must be non-negative");
            }
        }

        #[test]
        fn prop_mmap_empty_index(
            dim in 2u32..8,
            m in 2usize..8,
        ) {
            let b = HnswBuilder::new(dim, VectorMetric::Cosine, HnswConfig { m, ef_construction: 50, max_elements: 10 });
            let idx = b.build();
            let bytes = HnswSerializer::to_bytes(&idx)
                .expect("empty index should serialize");

            let loaded = MmapLoader::from_bytes(&bytes)
                .expect("empty index should load via mmap");
            prop_assert_eq!(loaded.node_count(), 0);
        }
    }
}
