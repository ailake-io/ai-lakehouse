//! ailake-file — unified file format
//!
//! Owns the combined Parquet + AI-Lake footer file.
//! The single file that Iceberg manifests point to.
//!
//! Layout: [PAR1][row groups][parquet footer][PAR1] [AILK header][centroid][HNSW][AILK trailer]
//!
//! See docs/specs/FILE_FORMAT.md for the binary specification.

pub mod footer;
pub mod writer;
pub mod reader;

pub use footer::{
    AilakeHeader, AilakeTrailer, Precision, DistanceMetric,
    AILAKE_MAGIC, AILAKE_FORMAT_VERSION, TRAILER_SIZE, HEADER_SIZE,
};
pub use writer::AilakeFileWriter;
pub use reader::AilakeFileReader;
