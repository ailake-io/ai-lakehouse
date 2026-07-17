# CODING_STANDARDS.md — Rust Conventions for AI-Lake

## General principles

- **Correctness first**: the positional invariant (row N == HNSW node N) is sacred. Any code path that could violate it must be rejected in review regardless of performance benefit.
- **Explicit over implicit**: prefer verbose, readable code over clever one-liners. This codebase will be read by contributors unfamiliar with the domain.
- **Fail loudly at the boundary**: validate inputs at public API entry points. Trust internal invariants inside modules.

---

## Error handling

### Use `thiserror` for all error types. One error enum per crate.

```rust
// ailake-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AilakeError {
    #[error("unsupported format version: {0}")]
    UnsupportedFormatVersion(u16),

    #[error("AI-Lake footer magic mismatch: expected AILK, got {0:?}")]
    InvalidAilakeMagic([u8; 4]),

    #[error("Parquet footer magic mismatch: expected PAR1, got {0:?}")]
    InvalidParquetMagic([u8; 4]),

    #[error("positional invariant violated: parquet row count {parquet} != HNSW node count {hnsw}")]
    RowCountMismatch { parquet: u64, hnsw: u64 },

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    #[error("centroid length mismatch: expected {expected_dim} dims, got {actual} bytes")]
    InvalidCentroidLength { expected_dim: u32, actual: usize },

    #[error("file is not a valid AI-Lake file (no AILK trailer)")]
    NotAnAilakeFile,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("bincode serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("catalog error: {0}")]
    Catalog(String),
}

pub type AilakeResult<T> = Result<T, AilakeError>;
```

### Never use `.unwrap()` or `.expect()` in library code.

```rust
// WRONG
let header = read_header(&mut file).unwrap();

// RIGHT
let header = read_header(&mut file)?;
```

`.expect()` is allowed only in:
- `main.rs` of CLI binaries
- Test code (`#[cfg(test)]`)
- Const contexts where failure is truly impossible

### Never use `panic!()` in library code.

Return `Err(AilakeError::...)` instead. If the error case seems impossible, add a comment explaining why and still return `Err`.

### Propagate with `?`, add context with `.map_err()`

```rust
// Add context when the upstream error is too generic
let data = store.get(&path)
    .await
    .map_err(|e| AilakeError::Catalog(format!("reading manifest {path}: {e}")))?;
```

---

## Async

- All I/O functions are `async`. Use `tokio` as the runtime.
- Never block inside an async function. CPU-heavy work (HNSW build, quantization of large batches) goes in `tokio::task::spawn_blocking`.
- Use `tokio::io::AsyncReadExt` / `AsyncWriteExt` for file I/O, not `std::io`.

```rust
// WRONG — blocks async executor
pub async fn read_footer(&mut self) -> AilakeResult<AilakeHeader> {
    let mut buf = [0u8; 64];
    self.file.read_exact(&mut buf)?;  // std::io::Read — BLOCKS
    AilakeHeader::parse(&buf)
}

// RIGHT
pub async fn read_footer(&mut self) -> AilakeResult<AilakeHeader> {
    use tokio::io::AsyncReadExt;
    let mut buf = [0u8; 64];
    self.file.read_exact(&mut buf).await?;
    AilakeHeader::parse(&buf)
}
```

### `spawn_blocking` for HNSW construction

`HnswBuilder::build()` is CPU-bound and not async-aware. Always wrap in `spawn_blocking`:

```rust
pub async fn build_hnsw(
    row_ids: Vec<RowId>,
    vectors: Vec<Vec<f32>>,
    dim: u32,
    metric: VectorMetric,
) -> AilakeResult<HnswIndex> {
    tokio::task::spawn_blocking(move || {
        let mut builder = HnswBuilder::new(dim, metric, HnswConfig::default());
        for (id, v) in row_ids.into_iter().zip(vectors) {
            builder.insert(id, v);
        }
        Ok(builder.build())
    })
    .await
    .map_err(|e| AilakeError::Catalog(format!("HNSW build task panicked: {e}")))?
}
```

---

## `unsafe` policy

`unsafe` is banned in all crates except `ailake-vec`, `ailake-index`, `ailake-file`, and `ailake-jni`.

In those crates:
- Every `unsafe` block requires a `// SAFETY:` comment explaining the invariant being upheld.
- `unsafe` is permitted for:
  - Memory layout operations (casting `&[f32]` to `&[u8]`, mmap operations)
  - SIMD intrinsics in `ailake-vec/src/distance.rs` — guarded by `#[target_feature(enable = "avx2")]` or `#[target_feature(enable = "neon")]`; callers check CPU features via `is_x86_feature_detected!` / `is_aarch64_feature_detected!` before calling
  - FFI calls
- No raw pointer arithmetic without bounds proof in the safety comment.

```rust
// SAFETY: f16 and u8 have the same alignment requirement (1 byte for u8,
// 2 bytes for f16 but the cast goes to u8). The slice length in bytes is
// vectors.len() * dim * 2, which we use as the slice length. This cast is safe.
let bytes: &[u8] = unsafe {
    std::slice::from_raw_parts(
        vectors.as_ptr() as *const u8,
        vectors.len() * dim * std::mem::size_of::<half::f16>(),
    )
};

// SAFETY: memmap2::Mmap requires file to outlive the mapping. The file handle
// is held in the same struct as the Mmap and dropped together. The mapped
// bytes are read-only for our usage. The bytes are valid UTF-8 byte sequence
// for bincode (which doesn't require alignment beyond u8).
let mmap = unsafe { memmap2::Mmap::map(&file)? };
```

---

## Types and naming

### `RowId` is a newtype, not a raw `u64`

```rust
// ailake-core/src/types.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct RowId(pub u64);

impl RowId {
    pub fn new(n: u64) -> Self { Self(n) }
    pub fn as_u64(self) -> u64 { self.0 }
    pub fn as_usize(self) -> usize { self.0 as usize }
}
```

Never pass raw `u64` where a `RowId` is expected. This prevents accidentally mixing row indices with record counts or byte offsets.

### Dimension and offset types

```rust
pub type Dim = u32;        // vector dimensionality
pub type ByteOffset = u64; // file offset in bytes
pub type ByteLen = u64;    // length in bytes
```

### Naming conventions

| Thing | Convention | Example |
|---|---|---|
| File paths (local FS) | `std::path::Path` / `PathBuf` | `file_path: &Path` |
| Object storage paths | `object_store::path::Path` | `store_path: object_store::path::Path` |
| Vector data in memory | `&[f32]` (not owned unless needed) | `fn search(query: &[f32])` |
| Block-level buffers | `Bytes` (from `bytes` crate) | `compressed: Bytes` |
| Distances | `f32` (never `f64`) | `let dist: f32 = cosine(a, b)` |

---

## Testing

### Unit tests: in the same file as the code

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ailake_trailer_round_trip() {
        let trailer = AilakeTrailer {
            footer_offset: 12_582_912,
            footer_len: 4_194_304,
            format_version: 1,
            flags: 0,
        };
        let bytes = trailer.to_bytes();
        let decoded = AilakeTrailer::from_bytes(&bytes).unwrap();
        assert_eq!(trailer.footer_offset, decoded.footer_offset);
        assert_eq!(trailer.footer_len, decoded.footer_len);
    }
}
```

### Integration tests: `tests/` directory at workspace root

```
tests/
├── write_read_roundtrip.rs       # Write unified file, read back, assert equality
├── iceberg_compat.rs             # Write table, verify PyIceberg can read (Phase 2)
├── parquet_trailing_bytes.rs     # Verify Parquet readers ignore AI-Lake footer
├── vector_pruning.rs             # Insert known vectors, verify pruning correctness
├── positional_invariant.rs       # Insert N rows, verify HNSW returns matching row IDs
└── context_assembler.rs          # Known chunks, assert dedup and ordering
```

### Property-based tests with `proptest`

For quantization, centroid computation, and serialization round-trips:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn f32_to_f16_round_trip_within_tolerance(
        values in prop::collection::vec(-1.0f32..1.0f32, 1..1536)
    ) {
        let f16_values: Vec<half::f16> = values.iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let restored: Vec<f32> = f16_values.iter().map(|v| v.to_f32()).collect();
        for (orig, rest) in values.iter().zip(restored.iter()) {
            prop_assert!((orig - rest).abs() < 0.001);
        }
    }

    #[test]
    fn centroid_radius_invariant(
        vectors in prop::collection::vec(
            prop::collection::vec(-1.0f32..1.0f32, 128),
            10..100
        )
    ) {
        let (centroid, radius) = compute_centroid_and_radius(&vectors, VectorMetric::Cosine);
        // Every input vector must be within radius of the centroid
        for v in &vectors {
            let d = cosine_distance(v, &centroid);
            prop_assert!(d <= radius + 1e-5);
        }
    }
}
```

### Benchmarks: `benches/` per crate, using `criterion`

```rust
// ailake-file/benches/write_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_write_unified_file_100k_vectors(c: &mut Criterion) {
    c.bench_function("write_100k_f16_with_hnsw", |b| {
        b.iter(|| {
            // write 100k × 1536-dim F16 vectors + build HNSW
            // to a tempfile, measure end-to-end time
        })
    });
}

criterion_group!(benches, bench_write_unified_file_100k_vectors);
criterion_main!(benches);
```

---

## Module structure

Each crate's `lib.rs` should only re-export the public API. Internal modules stay private.

```rust
// ailake-file/src/lib.rs
mod footer;
mod writer;
mod reader;

pub use footer::{AilakeHeader, AilakeTrailer, Precision, DistanceMetric,
                  AILAKE_MAGIC, AILAKE_FORMAT_VERSION};
pub use writer::AilakeFileWriter;
pub use reader::AilakeFileReader;
```

---

## Logging and tracing

Use `tracing` macros, not `println!` or `eprintln!`.

```rust
use tracing::{debug, info, warn, error, instrument};

#[instrument(skip(vectors), fields(dim = vectors[0].len(), count = vectors.len()))]
pub async fn build_index(vectors: &[Vec<f32>]) -> AilakeResult<HnswIndex> {
    info!("building HNSW index");
    // ...
    debug!(node_count = index.len(), "index built");
    Ok(index)
}
```

Log levels:
- `error`: unrecoverable, requires operator attention
- `warn`: recoverable, but unexpected (e.g. compaction skipped because another is running)
- `info`: significant lifecycle events (file written, snapshot committed, compaction started)
- `debug`: per-operation detail useful during development
- `trace`: per-block/per-vector detail, enabled only in debugging sessions

---

## Commit and PR conventions

- Commits: `<crate>: <verb> <what>` — e.g. `ailake-file: add footer rewrite on close`
- One logical change per commit.
- PRs must include: what changed, why, and which invariants are affected.
- Every PR that touches `ailake-file`, `ailake-index`, or `ailake-catalog` must include a note on how the positional invariant is preserved.
- Breaking changes to public APIs require a version bump in the affected crate's `Cargo.toml`.

---

## What Claude Code should never do

- Add `unwrap()` in library code without a comment and a test that covers the None/Err case.
- Mix `RowId` semantics with raw integers without explicit conversion.
- Write to the Iceberg `metadata.json` root outside of `ailake-catalog`.
- Skip the integrity checks in `AilakeFileReader::open()` (magic, version, record count match).
- Use `f64` for distances or vector elements — the entire pipeline is `f32`.
- Store the AI-Lake footer in a separate file — it MUST be appended to the Parquet file.
- Break the Parquet `PAR1` end marker — the Parquet section must be a valid Parquet file before the AI-Lake footer is appended.
- Reference the AI-Lake footer from the Iceberg Avro manifest as a separate file path — only the `.parquet` file path is in the manifest; the footer location is in `custom-properties` as a byte offset.
- Add a DataFusion dependency to any crate — see ADR-009.
