// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verifies row N in Parquet == HNSW node N.

mod fixtures;

use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};

#[tokio::test]
async fn positional_invariant_holds_for_1k_rows() {
    let dim = 16u32;
    let policy = VectorStoragePolicy {
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
        embedding_model: None,
        modality: None,
        partition_by: None,
        partition_value: None,
    partition_column_type: None,
        partition_fields: vec![],
};
    let (batch, embs) = fixtures::generate_batch(1000, dim as usize);
    let writer = AilakeFileWriter::new(policy);
    let file_bytes = writer.write(&batch, &embs).unwrap();

    let reader = AilakeFileReader::new(file_bytes, "embedding", dim);
    reader.verify_integrity().unwrap();
    assert_eq!(reader.load_index().unwrap().node_count(), 1000);
}
