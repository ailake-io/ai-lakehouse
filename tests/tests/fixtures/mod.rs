// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared test data generators.
//! Used by all integration tests.

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

#[allow(dead_code)]
pub fn generate_batch(rows: usize, dim: usize) -> (RecordBatch, Vec<Vec<f32>>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let ids: Vec<i32> = (0..rows as i32).collect();
    let texts: Vec<&str> = (0..rows).map(|_| "test").collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();

    let embeddings: Vec<Vec<f32>> = (0..rows)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (i * dim + j).hash(&mut h);
                    let bits = (h.finish() & 0x3FFF_FFFF) as u32;
                    (bits as f32 / 0x3FFF_FFFF as f32) * 2.0 - 1.0
                })
                .collect();
            // Normalize to unit sphere for cosine distance tests
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect();

    (batch, embeddings)
}

/// Generate `count` vectors near `center` by adding gaussian-like noise.
#[allow(dead_code)]
pub fn cluster_around(center: &[f32], dim: usize, count: usize, noise: f32) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (i * 1000 + j).hash(&mut h);
                    let bits = (h.finish() & 0x3FFF_FFFF) as u32;
                    let delta = (bits as f32 / 0x3FFF_FFFF as f32) * 2.0 - 1.0;
                    if j < center.len() {
                        center[j] + delta * noise
                    } else {
                        delta * noise
                    }
                })
                .collect();
            // Normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect()
}
