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
        debug!("ailake: loading HNSW index via mmap ({} bytes)", bytes.len());
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
}
