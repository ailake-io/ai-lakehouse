//! ailake-vec — vector data transformations
//!
//! No I/O. Pure computation: quantization, distance functions, centroid computation.

pub mod quantize;
pub mod distance;
pub mod compress;

pub use quantize::{Quantizer, ScalingParams};
pub use distance::{cosine_distance, euclidean_distance, dot_product, compute_centroid_and_radius};
pub use compress::BlockCompressor;
