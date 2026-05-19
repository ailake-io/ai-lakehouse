#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    None,
    Lz4,
    Zstd,
}

pub struct BlockCompressor {
    codec: CompressionCodec,
    zstd_level: i32,
}

impl BlockCompressor {
    pub fn none() -> Self {
        Self {
            codec: CompressionCodec::None,
            zstd_level: 3,
        }
    }

    pub fn lz4() -> Self {
        Self {
            codec: CompressionCodec::Lz4,
            zstd_level: 3,
        }
    }

    pub fn zstd(level: i32) -> Self {
        Self {
            codec: CompressionCodec::Zstd,
            zstd_level: level,
        }
    }

    pub fn codec(&self) -> CompressionCodec {
        self.codec
    }

    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        match self.codec {
            CompressionCodec::None => data.to_vec(),
            CompressionCodec::Lz4 => lz4_flex::compress_prepend_size(data),
            CompressionCodec::Zstd => {
                zstd::bulk::compress(data, self.zstd_level).unwrap_or_else(|_| data.to_vec())
            }
        }
    }

    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        match self.codec {
            CompressionCodec::None => data.to_vec(),
            CompressionCodec::Lz4 => {
                lz4_flex::decompress_size_prepended(data).unwrap_or_else(|_| data.to_vec())
            }
            CompressionCodec::Zstd => {
                zstd::bulk::decompress(data, 64 * 1024 * 1024).unwrap_or_else(|_| data.to_vec())
            }
        }
    }
}

impl Default for BlockCompressor {
    fn default() -> Self {
        Self::zstd(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(codec: BlockCompressor, data: &[u8]) {
        let compressed = codec.compress(data);
        let decompressed = codec.decompress(&compressed);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_roundtrip() {
        let data: Vec<u8> = (0u8..200).cycle().take(4096).collect();
        roundtrip(BlockCompressor::lz4(), &data);
    }

    #[test]
    fn zstd_roundtrip() {
        let data: Vec<u8> = (0u8..200).cycle().take(4096).collect();
        roundtrip(BlockCompressor::zstd(3), &data);
    }

    #[test]
    fn none_passthrough() {
        let data = b"hello ailake";
        roundtrip(BlockCompressor::none(), data);
    }

    #[test]
    fn zstd_compresses_repetitive_data() {
        // Repetitive float data (like zero vectors) should compress well
        let data = vec![0u8; 8192];
        let c = BlockCompressor::zstd(3);
        let compressed = c.compress(&data);
        assert!(
            compressed.len() < data.len() / 4,
            "expected >4x compression ratio"
        );
    }
}
