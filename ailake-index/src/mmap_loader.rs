// Phase 1: loads from in-memory bytes — no actual mmap.
// Phase 2: use memmap2::Mmap on a tempfile backed by partial S3 GET.

use ailake_core::AilakeResult;

use crate::hnsw::HnswIndex;
use crate::serialize::HnswSerializer;

pub struct MmapLoader;

impl MmapLoader {
    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<HnswIndex> {
        HnswSerializer::from_bytes(bytes)
    }
}
