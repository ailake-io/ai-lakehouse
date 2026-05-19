use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult, VectorStoragePolicy};
use arrow_array::{Array, FixedSizeBinaryArray, RecordBatch};
use arrow_schema::{Field, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::format::KeyValue;

use ailake_vec::Quantizer;

use crate::schema::{metric_str, precision_str, vector_field};

pub struct ParquetVectorWriter {
    policy: VectorStoragePolicy,
}

impl ParquetVectorWriter {
    pub fn new(policy: VectorStoragePolicy) -> Self {
        Self { policy }
    }

    /// Write a batch of tabular rows + embeddings into Parquet bytes.
    /// Returns (parquet_bytes, record_count).
    pub fn write_batch(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<(Bytes, u64)> {
        self.write_batch_with_kv(batch, embeddings, &[])
    }

    /// Like `write_batch` but appends extra key-value pairs to the Parquet footer metadata.
    pub fn write_batch_with_kv(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        extra_kv: &[(&str, &str)],
    ) -> AilakeResult<(Bytes, u64)> {
        let n = batch.num_rows();
        if embeddings.len() != n {
            return Err(AilakeError::DimensionMismatch {
                expected: n as u32,
                actual: embeddings.len() as u32,
            });
        }

        let bytes_per_vec = self.policy.dim as usize * self.policy.precision.bytes_per_element();

        // Encode each vector to F16 bytes, concatenate
        let flat: Vec<u8> = embeddings
            .iter()
            .flat_map(|v| Quantizer::f32_to_f16_bytes(v))
            .collect();

        // Build FixedSizeBinary array from contiguous bytes
        let chunks: Vec<Option<&[u8]>> = flat.chunks_exact(bytes_per_vec).map(Some).collect();
        let vec_array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            chunks.into_iter(),
            bytes_per_vec as i32,
        )
        .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Extend schema with vector column
        let vec_f = vector_field(
            &self.policy.column_name,
            self.policy.dim,
            self.policy.metric,
            self.policy.precision,
        );
        let new_fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| (**f).clone())
            .chain(std::iter::once(vec_f))
            .collect();
        let extended_schema = Arc::new(Schema::new_with_metadata(
            new_fields,
            batch.schema().metadata().clone(),
        ));

        // Build extended RecordBatch
        let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        cols.push(Arc::new(vec_array));
        let extended = RecordBatch::try_new(extended_schema.clone(), cols)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        let mut kv = vec![
            KeyValue::new("ailake.format_version".to_string(), Some("1".to_string())),
            KeyValue::new(
                "ailake.precision".to_string(),
                Some(precision_str(self.policy.precision).to_string()),
            ),
            KeyValue::new(
                "ailake.metric".to_string(),
                Some(metric_str(self.policy.metric).to_string()),
            ),
            KeyValue::new("ailake.record_count".to_string(), Some(n.to_string())),
            KeyValue::new(
                "ailake.vector_column".to_string(),
                Some(self.policy.column_name.clone()),
            ),
            KeyValue::new("ailake.dim".to_string(), Some(self.policy.dim.to_string())),
        ];
        for (k, v) in extra_kv {
            kv.push(KeyValue::new(k.to_string(), Some(v.to_string())));
        }

        let props = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_key_value_metadata(Some(kv))
            .build();

        let mut buf: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, extended_schema, Some(props))
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        writer
            .write(&extended)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        writer
            .close()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        Ok((Bytes::from(buf), n as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_core::{VectorMetric, VectorPrecision};
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

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
    fn write_produces_nonempty_bytes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let embeddings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1f32, 0.2, 0.3, 0.4]).collect();

        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, count) = writer.write_batch(&batch, &embeddings).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(count, 3);
    }
}
