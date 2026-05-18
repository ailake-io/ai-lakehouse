// Phase 1 stub — pass-through compression.
// Phase 2: replace with Snappy/Zstd block compression for vector data.
pub struct BlockCompressor;

impl BlockCompressor {
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }
}
