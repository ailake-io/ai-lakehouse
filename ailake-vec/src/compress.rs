// SPDX-License-Identifier: MIT OR Apache-2.0
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

    /// Decompresses `data` written by [`compress`](Self::compress) with the same codec.
    ///
    /// Returns an error on truncated/corrupted input instead of silently substituting
    /// the still-compressed bytes as if they were the decompressed payload — a caller
    /// has no way to detect corruption if a failed decompress looks like success.
    pub fn decompress(&self, data: &[u8]) -> ailake_core::AilakeResult<Vec<u8>> {
        match self.codec {
            CompressionCodec::None => Ok(data.to_vec()),
            CompressionCodec::Lz4 => lz4_flex::decompress_size_prepended(data).map_err(|e| {
                ailake_core::AilakeError::Io(std::io::Error::other(format!(
                    "ailake: LZ4 block decompression failed ({} bytes input): {e}",
                    data.len()
                )))
            }),
            CompressionCodec::Zstd => zstd::bulk::decompress(data, 64 * 1024 * 1024).map_err(|e| {
                ailake_core::AilakeError::Io(std::io::Error::other(format!(
                    "ailake: Zstd block decompression failed ({} bytes input): {e}",
                    data.len()
                )))
            }),
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
        let decompressed = codec.decompress(&compressed).unwrap();
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

    /// Regression: `decompress()` used to swallow the error and return the still-compressed
    /// bytes verbatim on truncated/corrupted input, masking corruption as if it were a
    /// successful decompress instead of surfacing it to the caller.
    #[test]
    fn lz4_decompress_of_corrupt_data_errors() {
        let c = BlockCompressor::lz4();
        let garbage = vec![0xFFu8; 8];
        assert!(c.decompress(&garbage).is_err());
    }

    #[test]
    fn zstd_decompress_of_corrupt_data_errors() {
        let c = BlockCompressor::zstd(3);
        let garbage = vec![0xFFu8; 8];
        assert!(c.decompress(&garbage).is_err());
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
