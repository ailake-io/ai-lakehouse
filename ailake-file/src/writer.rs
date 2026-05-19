use ailake_core::{AilakeResult, Centroid, VectorStoragePolicy};
use ailake_index::{HnswBuilder, HnswConfig, HnswSerializer};
use ailake_parquet::ParquetVectorWriter;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use bytes::{BufMut, Bytes, BytesMut};

use crate::footer::{
    AilakeHeader, AilakeTrailer, DistanceMetric, Precision, AILAKE_FORMAT_VERSION, HEADER_SIZE,
    TRAILER_SIZE,
};

pub struct AilakeFileWriter {
    policy: VectorStoragePolicy,
    hnsw_config: HnswConfig,
}

impl AilakeFileWriter {
    pub fn new(policy: VectorStoragePolicy) -> Self {
        Self {
            policy,
            hnsw_config: HnswConfig::default(),
        }
    }

    pub fn with_hnsw_config(mut self, config: HnswConfig) -> Self {
        self.hnsw_config = config;
        self
    }

    /// Write RecordBatch + embeddings into a single AI-Lake file.
    ///
    /// Layout:
    ///   [PAR1][row groups][AILK header+centroid+HNSW+trailer][Parquet footer][footer_len][PAR1]
    ///
    /// Standard Parquet readers find PAR1 at the end, read the footer, skip directly to row
    /// group offsets. The AILK section sits between row groups and footer and is never touched.
    /// AI-Lake readers find the AILK section via `ailake.footer_offset` in the Parquet footer KV.
    pub fn write(&self, batch: &RecordBatch, embeddings: &[Vec<f32>]) -> AilakeResult<Bytes> {
        let parquet_writer = ParquetVectorWriter::new(self.policy.clone());

        // Pass 1 – write Parquet without AILK location KV to measure the data section size.
        let (parquet_v1, record_count) = parquet_writer.write_batch(batch, embeddings)?;
        let footer_start = parquet_footer_start(&parquet_v1)?;
        let ailk_offset = footer_start as u64; // AILK will live right before the footer

        // Build centroid section
        let centroid: Centroid = compute_centroid_and_radius(embeddings, self.policy.metric);
        let centroid_bytes = encode_centroid(&centroid);

        // Build HNSW
        let mut builder = HnswBuilder::new(
            self.policy.dim,
            self.policy.metric,
            self.hnsw_config.clone(),
        );
        for (i, v) in embeddings.iter().enumerate() {
            builder.insert(ailake_core::RowId::new(i as u64), v.clone());
        }
        let index = builder.build();
        let hnsw_bytes = HnswSerializer::to_bytes(&index)?;

        // Compute AILK section layout
        let centroid_offset = HEADER_SIZE as u64;
        let centroid_len = centroid_bytes.len() as u64;
        let hnsw_offset_in_ailk = centroid_offset + centroid_len;
        let hnsw_len = hnsw_bytes.len() as u64;
        let ailk_total_len = HEADER_SIZE as u64 + centroid_len + hnsw_len + TRAILER_SIZE as u64;

        let header = AilakeHeader {
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
            dim: self.policy.dim,
            precision: Precision::from(self.policy.precision),
            distance_metric: DistanceMetric::from(self.policy.metric),
            record_count,
            centroid_offset,
            centroid_len,
            hnsw_offset: hnsw_offset_in_ailk,
            hnsw_len,
        };
        let trailer = AilakeTrailer {
            footer_offset: ailk_offset,
            footer_len: ailk_total_len,
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
        };

        let mut ailk_section = BytesMut::with_capacity(ailk_total_len as usize);
        ailk_section.put_slice(&header.to_bytes());
        ailk_section.put_slice(&centroid_bytes);
        ailk_section.put_slice(&hnsw_bytes);
        ailk_section.put_slice(&trailer.to_bytes());

        // Pass 2 – write Parquet with `ailake.footer_offset` in KV so the AI-Lake reader can
        // locate the AILK section without external metadata.
        let ailk_offset_str = ailk_offset.to_string();
        let (parquet_v2, _) = parquet_writer.write_batch_with_kv(
            batch,
            embeddings,
            &[("ailake.footer_offset", ailk_offset_str.as_str())],
        )?;
        let footer_start_v2 = parquet_footer_start(&parquet_v2)?;

        // Splice: data section + AILK section + Parquet footer (unchanged offsets) + PAR1
        let total = footer_start_v2 + ailk_section.len() + (parquet_v2.len() - footer_start_v2);
        let mut out = BytesMut::with_capacity(total);
        out.put_slice(&parquet_v2[..footer_start_v2]); // PAR1 + row groups
        out.put(ailk_section.freeze());
        out.put_slice(&parquet_v2[footer_start_v2..]); // footer thrift + footer_len + PAR1

        Ok(out.freeze())
    }
}

/// Returns the byte offset in `buf` where the Parquet footer thrift starts.
/// Layout of buf tail: [...footer_thrift...][footer_len u32 LE][PAR1 4 bytes]
fn parquet_footer_start(buf: &[u8]) -> AilakeResult<usize> {
    use ailake_core::AilakeError;
    let len = buf.len();
    if len < 8 {
        return Err(AilakeError::Parquet("file too small".into()));
    }
    if &buf[len - 4..] != b"PAR1" {
        return Err(AilakeError::Parquet("missing PAR1 footer magic".into()));
    }
    let footer_thrift_len = u32::from_le_bytes(buf[len - 8..len - 4].try_into().unwrap()) as usize;
    let start = len
        .checked_sub(8 + footer_thrift_len)
        .ok_or_else(|| AilakeError::Parquet("footer length overflow".into()))?;
    Ok(start)
}

fn encode_centroid(c: &Centroid) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(c.values.len() * 4 + 4);
    for &v in &c.values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&c.radius.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_core::{VectorMetric, VectorPrecision};
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_policy(dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: false,
        }
    }

    #[test]
    fn write_ends_with_par1() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();

        let writer = AilakeFileWriter::new(make_policy(4));
        let file = writer.write(&batch, &embs).unwrap();

        // Standard Parquet readers require PAR1 as the last 4 bytes
        assert_eq!(&file[file.len() - 4..], b"PAR1");
        // File must also start with PAR1
        assert_eq!(&file[..4], b"PAR1");
        // AILK magic must appear somewhere inside (in the embedded AILK section)
        assert!(file.windows(4).any(|w| w == b"AILK"));
    }
}
