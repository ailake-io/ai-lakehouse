use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use ailake_vec::Quantizer;
use arrow_array::{Array, FixedSizeBinaryArray, RecordBatch};
use arrow_schema::Schema;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub struct ParquetVectorReader {
    bytes: Bytes,
    vector_column: String,
}

impl ParquetVectorReader {
    pub fn new(bytes: Bytes, vector_column: &str) -> Self {
        Self {
            bytes,
            vector_column: vector_column.to_string(),
        }
    }

    /// Read tabular data and decode the vector column back to F32.
    ///
    /// Reads ALL row groups — uses a large batch size so typical single-row-group
    /// files are returned in one pass, and concatenates multiple batches otherwise.
    pub fn read_all(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Use a large batch size to avoid splitting single-row-group files across batches.
        // concatenate_all() handles the multi-batch case transparently.
        let record_count = builder.metadata().file_metadata().num_rows() as usize;
        let batch_size = record_count.max(1);
        let mut reader = builder
            .with_batch_size(batch_size)
            .build()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Collect all batches (usually just one, but handle multi-row-group files too).
        let mut batches: Vec<RecordBatch> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        if batches.is_empty() {
            return Err(AilakeError::Parquet(
                "no batches in Parquet file".to_string(),
            ));
        }

        let batch = if batches.len() == 1 {
            batches.remove(0)
        } else {
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
                .map_err(|e| AilakeError::Parquet(e.to_string()))?
        };

        // Find and extract vector column
        let vec_idx = batch.schema().index_of(&self.vector_column).map_err(|_| {
            AilakeError::Parquet(format!(
                "vector column '{}' not found in schema",
                self.vector_column
            ))
        })?;

        let vec_col = batch
            .column(vec_idx)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                AilakeError::Parquet("vector column is not FixedSizeBinary".to_string())
            })?;

        let embeddings: Vec<Vec<f32>> = (0..vec_col.len())
            .map(|i| Quantizer::f16_bytes_to_f32(vec_col.value(i)))
            .collect();

        // Return tabular batch without the vector column
        let keep: Vec<usize> = (0..batch.num_columns()).filter(|&i| i != vec_idx).collect();
        let tabular_fields: Vec<_> = keep
            .iter()
            .map(|&i| (*batch.schema().field(i)).clone())
            .collect();
        let tabular_schema = Arc::new(Schema::new(tabular_fields));
        let tabular_cols: Vec<_> = keep.iter().map(|&i| batch.column(i).clone()).collect();
        let tabular = RecordBatch::try_new(tabular_schema, tabular_cols)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        Ok((tabular, embeddings))
    }

    /// Extract a file-level key_value_metadata entry from the Parquet footer.
    pub fn kv_metadata(&self, key: &str) -> AilakeResult<Option<String>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        let kv = builder.metadata().file_metadata().key_value_metadata();
        Ok(kv.and_then(|kvs| {
            kvs.iter()
                .find(|kv| kv.key == key)
                .and_then(|kv| kv.value.clone())
        }))
    }

    pub fn record_count(&self) -> AilakeResult<u64> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        Ok(builder.metadata().file_metadata().num_rows() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::ParquetVectorWriter;
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use arrow_array::Int32Array;
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
    fn roundtrip_embeddings() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![10, 20, 30]))])
                .unwrap();

        let embs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer.write_batch(&batch, &embs).unwrap();

        let reader = ParquetVectorReader::new(bytes, "embedding");
        let (out_batch, out_embs) = reader.read_all().unwrap();

        assert_eq!(out_batch.num_rows(), 3);
        assert_eq!(out_embs.len(), 3);
        // F16 roundtrip should be close for these unit vectors
        for (orig, decoded) in embs.iter().zip(out_embs.iter()) {
            for (a, b) in orig.iter().zip(decoded.iter()) {
                assert!((a - b).abs() < 0.01, "roundtrip mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn kv_metadata_contains_ailake_keys() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer
            .write_batch(&batch, &[vec![0.1, 0.2, 0.3, 0.4]])
            .unwrap();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        assert_eq!(
            reader.kv_metadata("ailake.format_version").unwrap(),
            Some("1".to_string())
        );
        assert_eq!(
            reader.kv_metadata("ailake.precision").unwrap(),
            Some("f16".to_string())
        );
    }

    #[test]
    fn kv_metadata_absent_for_standard_key() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, _) = writer
            .write_batch(&batch, &[vec![0.1, 0.2, 0.3, 0.4]])
            .unwrap();
        let reader = ParquetVectorReader::new(bytes, "embedding");
        // Standard reader can read this — but ailake.hnsw_offset is absent in Phase 1
        assert!(reader.kv_metadata("ailake.hnsw_offset").unwrap().is_none());
    }
}
