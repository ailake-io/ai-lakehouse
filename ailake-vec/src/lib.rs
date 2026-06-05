// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-vec — vector data transformations
//!
//! No I/O. Pure computation: quantization, distance functions, centroid computation, PQ.

pub mod compress;
pub mod distance;
pub mod pq;
pub mod quantize;
pub mod rabitq;

pub use compress::{BlockCompressor, CompressionCodec};
pub use distance::{
    compute_centroid_and_radius, cosine_distance, cosine_distance_f16, dot_product,
    dot_product_f16, euclidean_distance, euclidean_distance_f16, exact_distance, normalize_l2,
    normalized_cosine_distance, normalized_cosine_distance_f16,
};
pub use pq::{kmeans_centroids, PQCodebook};
pub use quantize::{Quantizer, ScalingParams};
