// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult, VectorStoragePolicy};
use arrow_array::{Array, FixedSizeBinaryArray, ListArray, RecordBatch};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{Field, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::format::KeyValue;

use ailake_vec::Quantizer;

use crate::schema::{metric_str, multi_vector_field, precision_str, vector_field};

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

        let pq_only = !self.policy.keep_raw_for_reranking;

        // Stamp Iceberg-aligned field IDs on tabular columns.
        let tabular_fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| stamp_field_id((**f).clone(), i + 1))
            .collect();

        let (extended_schema, extended_cols) = if pq_only {
            // PQ-only mode: omit the raw vector column entirely.
            // The AILK section still contains the index; search works via the index blob.
            // Callers that need raw vectors for reranking should use keep_raw_for_reranking=true.
            let schema = Arc::new(Schema::new_with_metadata(
                tabular_fields,
                batch.schema().metadata().clone(),
            ));
            let cols = batch.columns().to_vec();
            (schema, cols)
        } else {
            let bytes_per_vec =
                self.policy.dim as usize * self.policy.precision.bytes_per_element();
            let flat: Vec<u8> = embeddings
                .iter()
                .flat_map(|v| Quantizer::f32_to_f16_bytes(v))
                .collect();
            let chunks: Vec<Option<&[u8]>> = flat.chunks_exact(bytes_per_vec).map(Some).collect();
            let vec_array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                chunks.into_iter(),
                bytes_per_vec as i32,
            )
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

            let vec_f = vector_field(
                &self.policy.column_name,
                self.policy.dim,
                self.policy.metric,
                self.policy.precision,
            );
            let vec_field_id = batch.schema().fields().len() + 1;
            let new_fields: Vec<Field> = tabular_fields
                .into_iter()
                .chain(std::iter::once(stamp_field_id(vec_f, vec_field_id)))
                .collect();
            let schema = Arc::new(Schema::new_with_metadata(
                new_fields,
                batch.schema().metadata().clone(),
            ));
            let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
            cols.push(Arc::new(vec_array));
            (schema, cols)
        };

        let extended = RecordBatch::try_new(extended_schema.clone(), extended_cols)
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
        if pq_only {
            kv.push(KeyValue::new(
                "ailake.pq_only".to_string(),
                Some("true".to_string()),
            ));
        }
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

    /// Write a batch where each row has a variable number of embeddings.
    ///
    /// `embeddings_per_row[i]` is the list of embeddings for row `i`.
    /// Physical column type: `List<FixedSizeBinary(bytes_per_vec)>`.
    /// Standard Parquet/Iceberg readers see a List column of byte arrays.
    /// AI-Lake SDK decodes each inner binary as an F16 vector of `dim` floats.
    pub fn write_batch_multi_vec(
        &self,
        batch: &RecordBatch,
        embeddings_per_row: &[Vec<Vec<f32>>],
        extra_kv: &[(&str, &str)],
    ) -> AilakeResult<(Bytes, u64)> {
        let n = batch.num_rows();
        if embeddings_per_row.len() != n {
            return Err(AilakeError::DimensionMismatch {
                expected: n as u32,
                actual: embeddings_per_row.len() as u32,
            });
        }

        let bytes_per_vec = self.policy.dim as usize * self.policy.precision.bytes_per_element();

        // Encode all vectors flat: [row0_vec0, row0_vec1, ..., row1_vec0, ...]
        let flat_bytes: Vec<u8> = embeddings_per_row
            .iter()
            .flat_map(|row| row.iter().flat_map(|v| Quantizer::f32_to_f16_bytes(v)))
            .collect();

        let total_vecs: usize = embeddings_per_row.iter().map(|r| r.len()).sum();
        let chunks: Vec<Option<&[u8]>> = flat_bytes.chunks_exact(bytes_per_vec).map(Some).collect();
        if chunks.len() != total_vecs {
            return Err(AilakeError::Parquet(format!(
                "multi-vec encoding produced {} chunks but expected {} (dim={} precision={} bytes)",
                chunks.len(),
                total_vecs,
                self.policy.dim,
                bytes_per_vec,
            )));
        }

        // Build inner FixedSizeBinaryArray (all vectors concatenated)
        let values = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            chunks.into_iter(),
            bytes_per_vec as i32,
        )
        .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Build offsets: [0, len(row0), len(row0)+len(row1), ...]
        let mut off: i32 = 0;
        let offsets: Vec<i32> = std::iter::once(0)
            .chain(embeddings_per_row.iter().map(|r| {
                off += r.len() as i32;
                off
            }))
            .collect();
        let offsets_buf = OffsetBuffer::new(offsets.into());

        let inner_field = Arc::new(arrow_schema::Field::new_list_field(
            arrow_schema::DataType::FixedSizeBinary(bytes_per_vec as i32),
            false,
        ));
        let list_array = ListArray::new(inner_field, offsets_buf, Arc::new(values), None);

        // Extend schema with multi-vector column; stamp Iceberg-aligned field IDs.
        let vec_f = multi_vector_field(
            &self.policy.column_name,
            self.policy.dim,
            self.policy.metric,
            self.policy.precision,
        );
        let vec_field_id = batch.schema().fields().len() + 1;
        let new_fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| stamp_field_id((**f).clone(), i + 1))
            .chain(std::iter::once(stamp_field_id(vec_f, vec_field_id)))
            .collect();
        let extended_schema = Arc::new(Schema::new_with_metadata(
            new_fields,
            batch.schema().metadata().clone(),
        ));

        let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        cols.push(Arc::new(list_array));
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
            KeyValue::new("ailake.multi_vec".to_string(), Some("true".to_string())),
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

/// Stamp `PARQUET:field_id` on an Arrow field so ArrowWriter embeds the Iceberg-aligned
/// field ID in the Parquet schema. `id` must be 1-based and match the Iceberg schema.
fn stamp_field_id(field: Field, id: usize) -> Field {
    let mut meta = field.metadata().clone();
    meta.insert(
        parquet::arrow::PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    );
    field.with_metadata(meta)
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
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
        }
    }

    fn make_pq_only_policy(dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: false,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
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

    #[test]
    fn pq_only_omits_vector_column() {
        use crate::reader::ParquetVectorReader;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let embeddings: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1f32, 0.2, 0.3, 0.4]).collect();

        let writer = ParquetVectorWriter::new(make_pq_only_policy(4));
        let (bytes, count) = writer.write_batch(&batch, &embeddings).unwrap();
        assert_eq!(count, 2);

        let reader = ParquetVectorReader::new(bytes, "embedding");
        assert!(reader.is_pq_only().unwrap(), "ailake.pq_only must be true");

        let (tabular, embs) = reader.read_all().unwrap();
        assert!(embs.is_empty(), "PQ-only: no raw embeddings returned");
        // Tabular columns preserved
        assert_eq!(tabular.num_rows(), 2);
        assert!(tabular.schema().index_of("id").is_ok());
        assert!(tabular.schema().index_of("text").is_ok());
        // Vector column absent from schema
        assert!(tabular.schema().index_of("embedding").is_err());
    }

    #[test]
    fn non_pq_only_preserves_vector_column() {
        use crate::reader::ParquetVectorReader;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let embeddings = vec![vec![1.0f32, 0.0, 0.0, 0.0]];

        // keep_raw_for_reranking = true (default)
        let mut policy = make_pq_only_policy(4);
        policy.keep_raw_for_reranking = true;
        let writer = ParquetVectorWriter::new(policy);
        let (bytes, _) = writer.write_batch(&batch, &embeddings).unwrap();

        let reader = ParquetVectorReader::new(bytes, "embedding");
        assert!(!reader.is_pq_only().unwrap());
        let (_, embs) = reader.read_all().unwrap();
        assert_eq!(embs.len(), 1, "raw embeddings present when keep_raw=true");
    }
}
