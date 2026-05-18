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

    /// Write RecordBatch + embeddings into a single AI-Lake file (Parquet + AILK footer).
    /// Returns the complete file as Bytes.
    pub fn write(&self, batch: &RecordBatch, embeddings: &[Vec<f32>]) -> AilakeResult<Bytes> {
        let n = embeddings.len();

        // 1. Write Parquet section
        let parquet_writer = ParquetVectorWriter::new(self.policy.clone());
        let (parquet_bytes, record_count) = parquet_writer.write_batch(batch, embeddings)?;

        // 2. Compute centroid + radius
        let centroid: Centroid = compute_centroid_and_radius(embeddings, self.policy.metric);

        // 3. Build HNSW from embeddings
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

        // 4. Build centroid section: [f32; dim] + f32 radius (little-endian)
        let centroid_bytes = encode_centroid(&centroid);

        // 5. Calculate offsets within the AI-Lake footer extension
        // Layout: [AILK header (64)] [centroid section] [HNSW section] [trailer (24)]
        let centroid_offset = HEADER_SIZE as u64;
        let centroid_len = centroid_bytes.len() as u64;
        let hnsw_offset = centroid_offset + centroid_len;
        let hnsw_len = hnsw_bytes.len() as u64;
        let footer_len = HEADER_SIZE as u64 + centroid_len + hnsw_len + TRAILER_SIZE as u64;
        let footer_offset = parquet_bytes.len() as u64;

        // 6. Assemble AI-Lake header
        let header = AilakeHeader {
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
            dim: self.policy.dim,
            precision: Precision::from(self.policy.precision),
            distance_metric: DistanceMetric::from(self.policy.metric),
            record_count: record_count,
            centroid_offset,
            centroid_len,
            hnsw_offset,
            hnsw_len,
        };

        // 7. Assemble trailer
        let trailer = AilakeTrailer {
            footer_offset,
            footer_len,
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
        };

        // 8. Concatenate: Parquet bytes + header + centroid + HNSW + trailer
        let total = parquet_bytes.len()
            + HEADER_SIZE
            + centroid_bytes.len()
            + hnsw_bytes.len()
            + TRAILER_SIZE;
        let mut out = BytesMut::with_capacity(total);
        out.put(parquet_bytes);
        out.put(&header.to_bytes()[..]);
        out.put(&centroid_bytes[..]);
        out.put(&hnsw_bytes[..]);
        out.put(&trailer.to_bytes()[..]);

        let _ = n; // suppress unused warning
        Ok(out.freeze())
    }
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
    fn write_ends_with_ailk() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        let embs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();

        let writer = AilakeFileWriter::new(make_policy(4));
        let file = writer.write(&batch, &embs).unwrap();

        // File must end with AILK magic
        assert_eq!(&file[file.len() - 4..], b"AILK");
        // File must also contain PAR1 somewhere (Parquet marker)
        assert!(file.windows(4).any(|w| w == b"PAR1"));
    }
}
