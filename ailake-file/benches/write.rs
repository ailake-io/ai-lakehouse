// SPDX-License-Identifier: MIT OR Apache-2.0
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_write(c: &mut Criterion) {
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use ailake_file::AilakeFileWriter;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let dim = 64u32;
    let n = 256usize;

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids: Vec<i32> = (0..n as i32).collect();
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();

    let embeddings: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; dim as usize];
            v[i % dim as usize] = 1.0;
            v
        })
        .collect();

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
        rabitq: None,
        binary: None,
    };
    let writer = AilakeFileWriter::new(policy);

    c.bench_function("ailake_file_write_256x64", |b| {
        b.iter(|| writer.write(&batch, &embeddings).unwrap());
    });
}

criterion_group!(benches, bench_write);
criterion_main!(benches);
