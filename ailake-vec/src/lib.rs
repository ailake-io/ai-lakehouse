//! ailake-vec — vector data transformations
//!
//! No I/O. Pure computation: quantization, distance functions, centroid computation, PQ.

pub mod compress;
pub mod distance;
pub mod pq;
pub mod quantize;

pub use compress::{BlockCompressor, CompressionCodec};
pub use distance::{
    compute_centroid_and_radius, cosine_distance, dot_product, euclidean_distance, exact_distance,
};
pub use pq::PQCodebook;
pub use quantize::{Quantizer, ScalingParams};
