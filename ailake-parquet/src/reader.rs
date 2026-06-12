// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use ailake_vec::Quantizer;
use arrow_array::{Array, FixedSizeBinaryArray, ListArray, RecordBatch};
use arrow_schema::{DataType, Schema};
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
    ///
    /// PQ-only files (written with `keep_raw_for_reranking = false`) omit the raw
    /// vector column. For those files, the returned embeddings vec is empty and the
    /// returned RecordBatch contains only tabular columns.
    pub fn read_all(&self) -> AilakeResult<(RecordBatch, Vec<Vec<f32>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        // Detect PQ-only before consuming the builder.
        let pq_only = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .and_then(|kvs| {
                kvs.iter()
                    .find(|kv| kv.key == "ailake.pq_only")
                    .and_then(|kv| kv.value.as_deref())
                    .map(|v| v == "true")
            })
            .unwrap_or(false);

        // Use a large batch size to avoid splitting single-row-group files across batches.
        // concatenate_all() handles the multi-batch case transparently.
        let record_count = builder.metadata().file_metadata().num_rows() as usize;
        let batch_size = record_count.max(1);
        let reader = builder
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

        // PQ-only: no raw vector column present. Return tabular batch with empty embeddings.
        if pq_only || batch.schema().index_of(&self.vector_column).is_err() {
            return Ok((batch, vec![]));
        }

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

    /// Returns true when this file was written in PQ-only mode (raw vector column omitted).
    pub fn is_pq_only(&self) -> AilakeResult<bool> {
        let v = self.kv_metadata("ailake.pq_only")?;
        Ok(v.as_deref() == Some("true"))
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

    /// Read a multi-vector column: `List<FixedSizeBinary(bytes_per_vec)>`.
    ///
    /// Returns `(tabular_batch, embeddings_per_row)` where
    /// `embeddings_per_row[i]` is the Vec of F32 embeddings for row `i`.
    /// Each embedding has `dim` elements.
    pub fn read_all_multi_vec(&self) -> AilakeResult<(RecordBatch, Vec<Vec<Vec<f32>>>)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;
        let record_count = builder.metadata().file_metadata().num_rows() as usize;
        let reader = builder
            .with_batch_size(record_count.max(1))
            .build()
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

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

        let vec_idx = batch.schema().index_of(&self.vector_column).map_err(|_| {
            AilakeError::Parquet(format!(
                "multi-vector column '{}' not found",
                self.vector_column
            ))
        })?;

        let field_type = batch.schema().field(vec_idx).data_type().clone();
        if !matches!(field_type, DataType::List(_)) {
            return Err(AilakeError::Parquet(format!(
                "column '{}' is not a List type; use read_all() for single-vector columns",
                self.vector_column
            )));
        }

        let list_col = batch
            .column(vec_idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| AilakeError::Parquet("failed to downcast to ListArray".to_string()))?;

        let values = list_col
            .values()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                AilakeError::Parquet(
                    "ListArray values are not FixedSizeBinary — unexpected column type".to_string(),
                )
            })?;

        let mut embeddings_per_row: Vec<Vec<Vec<f32>>> = Vec::with_capacity(list_col.len());
        for row in 0..list_col.len() {
            let start = list_col.value_offsets()[row] as usize;
            let end = list_col.value_offsets()[row + 1] as usize;
            let row_vecs: Vec<Vec<f32>> = (start..end)
                .map(|vi| Quantizer::f16_bytes_to_f32(values.value(vi)))
                .collect();
            embeddings_per_row.push(row_vecs);
        }

        // Strip vector column → return only tabular columns
        let keep: Vec<usize> = (0..batch.num_columns()).filter(|&i| i != vec_idx).collect();
        let tabular_fields: Vec<_> = keep
            .iter()
            .map(|&i| (*batch.schema().field(i)).clone())
            .collect();
        let tabular_schema = Arc::new(Schema::new(tabular_fields));
        let tabular_cols: Vec<_> = keep.iter().map(|&i| batch.column(i).clone()).collect();
        let tabular = RecordBatch::try_new(tabular_schema, tabular_cols)
            .map_err(|e| AilakeError::Parquet(e.to_string()))?;

        Ok((tabular, embeddings_per_row))
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
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
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

    #[test]
    fn multi_vec_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Int32, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        use arrow_array::StringArray;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["doc_a", "doc_b"])),
            ],
        )
        .unwrap();

        // doc 1 has 3 chunk embeddings, doc 2 has 2
        let embeddings_per_row: Vec<Vec<Vec<f32>>> = vec![
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            vec![vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]],
        ];

        let writer = ParquetVectorWriter::new(make_policy(4));
        let (bytes, count) = writer
            .write_batch_multi_vec(&batch, &embeddings_per_row, &[])
            .unwrap();
        assert_eq!(count, 2);

        let reader = ParquetVectorReader::new(bytes.clone(), "embedding");
        let (out_batch, out_embs) = reader.read_all_multi_vec().unwrap();

        assert_eq!(out_batch.num_rows(), 2);
        assert_eq!(out_embs.len(), 2);
        assert_eq!(out_embs[0].len(), 3, "doc 1 should have 3 embeddings");
        assert_eq!(out_embs[1].len(), 2, "doc 2 should have 2 embeddings");

        // Verify F16 roundtrip accuracy
        for (row_orig, row_dec) in embeddings_per_row.iter().zip(out_embs.iter()) {
            for (orig, dec) in row_orig.iter().zip(row_dec.iter()) {
                for (a, b) in orig.iter().zip(dec.iter()) {
                    assert!((a - b).abs() < 0.01, "roundtrip mismatch: {a} vs {b}");
                }
            }
        }

        // Verify ailake.multi_vec=true in KV metadata
        assert_eq!(
            reader.kv_metadata("ailake.multi_vec").unwrap(),
            Some("true".to_string())
        );

        // Verify tabular columns preserved correctly
        use arrow_array::Int32Array;
        let doc_ids = out_batch
            .column_by_name("doc_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(doc_ids.value(0), 1);
        assert_eq!(doc_ids.value(1), 2);
    }
}
