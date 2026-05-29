// SPDX-License-Identifier: MIT OR Apache-2.0
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_hnsw_search(c: &mut Criterion) {
    use ailake_core::{RowId, VectorMetric};
    use ailake_index::{HnswBuilder, HnswConfig};

    let dim = 64u32;
    let n = 1000usize;
    let mut builder = HnswBuilder::new(dim, VectorMetric::Cosine, HnswConfig::default());
    for i in 0..n {
        let mut v = vec![0.0f32; dim as usize];
        v[i % dim as usize] = 1.0;
        builder.insert(RowId::new(i as u64), v);
    }
    let index = builder.build();

    let query: Vec<f32> = (0..dim as usize)
        .map(|i| if i == 0 { 1.0 } else { 0.0 })
        .collect();
    c.bench_function("hnsw_search_top10", |b| {
        b.iter(|| index.search(&query, 10, 50));
    });
}

criterion_group!(benches, bench_hnsw_search);
criterion_main!(benches);
