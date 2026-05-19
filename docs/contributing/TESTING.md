# TESTING.md — Test Strategy

## Philosophy

- **Every public function has at least one unit test.**
- **Every invariant has a property-based test.**
- **Every compatibility claim has an integration test that actually runs the target system.**
- A test that only checks "no panic" is not sufficient. Assert the specific output.
- Flaky tests are treated as bugs. If a test fails intermittently, it is disabled until fixed.

---

## Test categories

| Category | Location | Runs in | What it covers |
|---|---|---|---|
| Unit | `src/` inline `#[cfg(test)]` | `cargo test` | Single function, no I/O |
| Integration | `tests/` at workspace root | `cargo test --features integration` | Multiple crates, local FS |
| Property-based | `src/` inline or `tests/` | `cargo test` | Invariants across random inputs |
| Benchmark | `benches/` per crate | `cargo bench` | Performance regressions |
| Compat (Phase 2) | `tests/compat/` | Docker Compose | External engines read AI-Lake files |
| Compat (Phase 3) | `tests/compat/engines/` | Docker Compose | Spark / Trino / Beam |

---

## Unit tests

### Conventions

Place unit tests in the same file as the code under test, inside a `mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn centroid_of_identical_vectors_is_that_vector() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let vectors = vec![v.clone(), v.clone(), v.clone()];
        let (centroid, radius) = compute_centroid_and_radius(&vectors, VectorMetric::Cosine);
        assert!((centroid[0] - 1.0).abs() < 1e-6);
        assert!(radius < 1e-6);
    }

    #[test]
    fn footer_magic_round_trip() {
        let trailer = AilakeTrailer {
            footer_offset: 12_582_912,
            footer_len: 4_194_304,
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
        };
        let bytes = trailer.to_bytes();
        assert_eq!(bytes.len(), TRAILER_SIZE);
        let decoded = AilakeTrailer::from_bytes(&bytes).unwrap();
        assert_eq!(trailer.footer_offset, decoded.footer_offset);
        assert_eq!(trailer.footer_len, decoded.footer_len);
    }
}
```

### Async unit tests

Use `#[tokio::test]` for async functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_store_put_get_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        let path = object_store::path::Path::from("test/file.bin");
        store.put(&path, b"hello world".to_vec().into()).await.unwrap();
        let data = store.get(&path).await.unwrap();
        assert_eq!(data.as_ref(), b"hello world");
    }
}
```

---

## Integration tests (`tests/` at workspace root)

These run without Docker. They use `LocalStore` and `HadoopCatalog` (filesystem).

### `tests/write_read_roundtrip.rs`

The most critical test. Covers the full write → read pipeline end-to-end.

```rust
#[tokio::test]
async fn write_10k_rows_search_top10() {
    let dir = tempfile::tempdir().unwrap();
    let table_uri = dir.path().to_str().unwrap();

    // Generate 10,000 random 128-dim F32 vectors + fake tabular data
    let (record_batch, embeddings) = fixtures::generate_batch(10_000, 128);

    // Write
    let mut writer = TableWriter::new(table_uri, WriterConfig::default_f16()).await.unwrap();
    writer.write_batch(record_batch, &embeddings).await.unwrap();
    let snapshot_id = writer.commit().await.unwrap();

    // Pick a known query vector (row 42's embedding)
    let query = embeddings.row(42).to_vec();

    // Search
    let results = search(table_uri, &query, 10, None).await.unwrap();

    // Row 42 must be in top-10
    let row_ids: Vec<u64> = results.iter().map(|r| r.row_id.as_u64()).collect();
    assert!(row_ids.contains(&42), "query vector's own row not in top-10");

    // All distances must be in [0, 1] for cosine
    for r in &results {
        assert!(r.distance >= 0.0 && r.distance <= 1.0 + 1e-6);
    }

    // Results must be sorted by distance ascending
    for window in results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}
```

### `tests/positional_invariant.rs`

Verifies that row N in Parquet == HNSW node N.

```rust
#[tokio::test]
async fn positional_invariant_holds_for_1k_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (record_batch, embeddings) = fixtures::generate_batch(1_000, 64);

    let mut writer = TableWriter::new(dir.path().to_str().unwrap(), WriterConfig::default_f16()).await.unwrap();
    writer.write_batch(record_batch.clone(), &embeddings).await.unwrap();
    writer.commit().await.unwrap();

    // Open the file directly and verify
    let file_path = find_parquet_file(dir.path());
    let reader = AilakeFileReader::open(&file_path, &LocalStore::new(dir.path())).await.unwrap();

    let parquet_count = reader.parquet_record_count().await.unwrap();
    let hnsw = reader.load_index().await.unwrap();

    assert_eq!(parquet_count, 1_000);
    assert_eq!(hnsw.node_count(), 1_000,
        "HNSW node count must match Parquet row count");

    // For each row, searching its exact vector must return itself as top-1
    for i in 0..1_000usize {
        let query = embeddings.row(i).to_vec();
        let results = hnsw.search(&query, 1, 50);
        assert_eq!(results[0].0.as_usize(), i,
            "row {i}: exact vector search did not return itself as top-1");
    }
}
```

### `tests/parquet_trailing_bytes.rs`

Verifies that standard Parquet readers ignore the AI-Lake footer.

```rust
#[test]
fn pyarrow_ignores_ailake_footer() {
    // This test calls Python via subprocess to avoid PyO3 setup complexity in unit tests
    let dir = tempfile::tempdir().unwrap();
    let (batch, embeddings) = fixtures::generate_batch(100, 32);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = TableWriter::new(dir.path().to_str().unwrap(), WriterConfig::default_f16()).await.unwrap();
        writer.write_batch(batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    });

    let file_path = find_parquet_file(dir.path());
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            &format!(
                "import pyarrow.parquet as pq; t = pq.read_table('{}'); print(len(t))",
                file_path.display()
            ),
        ])
        .output()
        .expect("python3 not found — required for compat test");

    assert!(output.status.success(),
        "PyArrow raised an error reading AI-Lake parquet: {}",
        String::from_utf8_lossy(&output.stderr));

    let row_count: usize = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap();
    assert_eq!(row_count, 100);
}
```

### `tests/vector_pruning.rs`

Places known vectors into two groups in separate files, verifies only the correct file survives pruning.

```rust
#[tokio::test]
async fn pruning_eliminates_distant_file() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().to_str().unwrap();

    let store = Arc::new(LocalStore::new(root));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), root));
    let table = TableIdent::new("default", "prune_test");

    let policy = VectorStoragePolicy { column_name: "embedding".into(), dim: 4,
        metric: VectorMetric::Cosine, precision: VectorPrecision::F16,
        pq: None, keep_raw_for_reranking: false };

    // File A: vectors near [1, 0, 0, 0]
    // File B: vectors near [0, 0, 0, 1]  (far from query)
    // ... write both files via TableWriter ...

    // Query near [1, 0, 0, 0] — file B should be pruned
    let results = search(
        &table, &[1.0f32, 0.0, 0.0, 0.0],
        SearchConfig { top_k: 5, ef_search: 50, pruning_threshold: 0.5 },
        "embedding", 4, catalog, store,
    ).await.unwrap();

    // All results must come from file A (part-00000.parquet)
    for r in &results {
        assert!(r.file_path.contains("part-00000"));
    }
}
```

### `tests/context_assembler.rs`

Verifies deduplication and document grouping.

```rust
#[test]
fn dedup_removes_near_identical_chunks() {
    let config = ContextAssemblerConfig { dedup_threshold: 0.05, ..Default::default() };
    let assembler = ContextAssembler::new(config);

    let emb = vec![1.0f32, 0.0, 0.0];
    let mut chunk_a = Chunk {
        document_id: "doc-1".into(), chunk_index: 0,
        chunk_text: "The gross margin improved significantly.".into(),
        embedding: Some(emb.clone()), distance: 0.1, ..Default::default()
    };
    let mut chunk_b = Chunk {
        document_id: "doc-1".into(), chunk_index: 1,
        chunk_text: "Gross margin saw significant improvement.".into(),
        embedding: Some(emb.clone()), // exact same embedding → duplicate
        distance: 0.2, ..Default::default()
    };
    let chunk_c = Chunk {
        document_id: "doc-2".into(), chunk_index: 0,
        chunk_text: "Revenue grew 15% year over year.".into(),
        embedding: None, distance: 0.3, ..Default::default()
    };

    let results = assembler.assemble_chunks(vec![chunk_a, chunk_b, chunk_c]);

    assert_eq!(results.chunk_count, 2, "near-duplicate should be removed");
}

#[test]
fn grouping_restores_chunk_order() {
    let config = ContextAssemblerConfig::default();
    let assembler = ContextAssembler::new(config);

    // Chunks returned by search in arbitrary order
    let chunks = vec![
        Chunk { document_id: "doc-1".into(), chunk_index: 2, chunk_text: "third chunk".into(), distance: 0.3, ..Default::default() },
        Chunk { document_id: "doc-1".into(), chunk_index: 0, chunk_text: "first chunk".into(),  distance: 0.1, ..Default::default() },
        Chunk { document_id: "doc-1".into(), chunk_index: 1, chunk_text: "second chunk".into(), distance: 0.2, ..Default::default() },
    ];

    let results = assembler.assemble_chunks(chunks);
    let text = &results.text;

    let pos_first  = text.find("first chunk").unwrap();
    let pos_second = text.find("second chunk").unwrap();
    let pos_third  = text.find("third chunk").unwrap();

    assert!(pos_first < pos_second && pos_second < pos_third,
        "chunks should be in document order in the assembled context");
}
```

---

## Property-based tests (`proptest`)

### `ailake-vec` — quantization invariants

```rust
use proptest::prelude::*;

proptest! {
    /// F32 → F16 round-trip stays within tolerance for values in [-1, 1]
    #[test]
    fn f32_f16_roundtrip_tolerance(
        v in prop::collection::vec(-1.0f32..=1.0f32, 1..=1536)
    ) {
        let f16s: Vec<half::f16> = v.iter().map(|&x| half::f16::from_f32(x)).collect();
        let back: Vec<f32> = f16s.iter().map(|x| x.to_f32()).collect();
        for (orig, restored) in v.iter().zip(back.iter()) {
            prop_assert!((orig - restored).abs() < 0.001,
                "F16 round-trip error too large: {} → {}", orig, restored);
        }
    }

    /// Centroid is always within radius of every input vector
    #[test]
    fn centroid_radius_covers_all_vectors(
        vectors in prop::collection::vec(
            prop::collection::vec(-1.0f32..=1.0f32, 4usize..=64),
            2usize..=50
        )
    ) {
        let (centroid, radius) = compute_centroid_and_radius(&vectors, VectorMetric::Cosine);
        for v in &vectors {
            let d = cosine_distance(v, &centroid);
            prop_assert!(d <= radius + 1e-5,
                "vector distance {} exceeds radius {}", d, radius);
        }
    }
}
```

### `ailake-file` — binary layout invariants

```rust
proptest! {
    /// Any valid header serializes and deserializes to the same value
    #[test]
    fn ailake_header_round_trip(
        dim in 1u32..=4096,
        record_count in 0u64..=10_000_000,
        hnsw_offset in 1000u64..=1_000_000_000,
        hnsw_len in 100u64..=100_000_000,
    ) {
        let header = AilakeHeader {
            format_version: AILAKE_FORMAT_VERSION,
            flags: 0,
            dim,
            precision: Precision::F16,
            distance_metric: DistanceMetric::Cosine,
            record_count,
            centroid_offset: 64,
            centroid_len: dim as u64 * 4 + 4,
            hnsw_offset,
            hnsw_len,
        };
        let bytes = header.to_bytes();
        prop_assert_eq!(bytes.len(), HEADER_SIZE);
        let decoded = AilakeHeader::from_bytes(&bytes).unwrap();
        prop_assert_eq!(header.dim, decoded.dim);
        prop_assert_eq!(header.record_count, decoded.record_count);
        prop_assert_eq!(header.hnsw_offset, decoded.hnsw_offset);
    }
}
```

---

## Test fixtures (`tests/fixtures/`)

Shared test data generators used across all integration tests.

```rust
// tests/fixtures/mod.rs

pub fn generate_batch(rows: usize, dim: usize) -> (RecordBatch, Array2<f32>) {
    let mut rng = rand::thread_rng();
    let embeddings = Array2::from_shape_fn((rows, dim), |_| rng.gen::<f32>() * 2.0 - 1.0);
    let record_batch = build_record_batch(rows, &embeddings);
    (record_batch, embeddings)
}

pub fn cluster_around(center: &[f32], dim: usize, count: usize, noise: f32) -> Array2<f32> {
    // Returns `count` vectors clustered around `center` with Gaussian noise `noise`
    ...
}

pub fn chunk(doc_id: &str, index: u32, text: &str) -> Chunk {
    // Creates a Chunk with deterministic embedding derived from text hash
    ...
}

pub fn find_parquet_file(dir: &Path) -> PathBuf {
    // Finds the first .parquet file in dir/data/
    ...
}

pub async fn write_file(dir: &Path, name: &str, embeddings: &Array2<f32>) {
    // Convenience: write a minimal AI-Lake file with given embeddings
    ...
}
```

---

## Compatibility tests (Phase 2 — Docker required)

Run with:
```bash
docker compose -f tests/docker/compose.yml up -d
cargo test --workspace --features integration -- --test-threads=1
docker compose -f tests/docker/compose.yml down
```

### `tests/compat/minio_s3.rs`

Writes a table to MinIO, reads it back via AI-Lake SDK. Validates S3 `get_range` partial reads.

```rust
#[tokio::test]
#[cfg(feature = "integration")]
async fn write_to_minio_read_back() {
    let store = S3Store::new(S3Config {
        bucket: "test-bucket".to_string(),
        region: "us-east-1".to_string(),
        endpoint: Some("http://localhost:9000".to_string()),
        path_style: true,
        credentials: S3Credentials::Static {
            key_id: "minioadmin".to_string(),
            secret: "minioadmin".to_string(),
        },
        ..Default::default()
    });
    // ... write + search + assert
}
```

### `tests/compat/nessie_catalog.rs`

Writes a table with Nessie as catalog, verifies snapshot appears in Nessie API.

### `tests/compat/pyiceberg_read.rs`

Writes a table via Rust SDK, reads with PyIceberg. Asserts row count, schema, no errors.

### `tests/compat/pyarrow_parquet.rs`

Writes via Rust SDK, reads the raw Parquet file with `pyarrow.parquet`. Asserts trailing bytes are ignored.

---

## Compatibility tests (Phase 3 — Engines, requires Docker + JVM)

```bash
docker compose -f tests/docker/compose-engines.yml up -d
./tests/compat/run_all_engines.sh
```

### `compose-engines.yml` services

```yaml
services:
  spark:
    image: apache/spark:3.5.0
    environment:
      - SPARK_MODE=master
    ports: ["8080:8080", "7077:7077"]

  spark-worker:
    image: apache/spark:3.5.0
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark:7077

  trino:
    image: trinodb/trino:432
    ports: ["8081:8080"]
    volumes:
      - ./tests/docker/trino-catalog:/etc/trino/catalog

  beam-direct:
    image: apache/beam_python3.11_sdk:2.59.0
    # Used for Beam SDK compat tests
```

### Engine compat test matrix

Each engine test follows the same script:

```
1. Rust SDK writes a 10k-row AI-Lake table to MinIO
2. Engine reads the table (no AI-Lake plugin)
3. Assert: row count == 10,000
4. Assert: schema contains expected columns
5. Assert: 'embedding' column is BINARY/BYTES type
6. Assert: a filter query returns correct filtered rows
7. Assert: no error messages about unrecognized format
```

Scripts:
- `tests/compat/spark_read.py` — PySpark
- `tests/compat/trino_read.sql` — Trino SQL via JDBC
- `tests/compat/beam_read.py` — Beam Python SDK with `Managed.ICEBERG`
- `tests/compat/pyiceberg_read.py` — PyIceberg
- `tests/compat/duckdb_read.sql` — DuckDB

---

## Benchmarks

### `ailake-file/benches/write.rs`

```rust
fn bench_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");

    for &rows in &[1_000usize, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("f16_lz4_with_hnsw", rows),
            &rows,
            |b, &rows| {
                let (batch, embeddings) = fixtures::generate_batch(rows, 1536);
                b.iter(|| {
                    let dir = tempfile::tempdir().unwrap();
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let mut writer = TableWriter::new(
                            dir.path().to_str().unwrap(),
                            WriterConfig::default_f16()
                        ).await.unwrap();
                        writer.write_batch(batch.clone(), &embeddings).await.unwrap();
                        writer.commit().await.unwrap();
                    });
                });
            },
        );
    }
    group.finish();
}
```

### `ailake-index/benches/search.rs`

```rust
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for &rows in &[10_000usize, 100_000, 1_000_000] {
        // Build index once, bench search only
        let index = build_index(rows, 1536);
        let query: Vec<f32> = (0..1536).map(|_| rand::random()).collect();

        group.bench_with_input(
            BenchmarkId::new("top10_cosine", rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    index.search(&query, 10, 50)
                });
            },
        );
    }
    group.finish();
}
```

Run benchmarks with:
```bash
cargo bench --workspace
# HTML reports in target/criterion/
```

---

## CI matrix (GitHub Actions)

```yaml
# .github/workflows/ci.yml

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - cargo test --workspace

  clippy:
    runs-on: ubuntu-latest
    steps:
      - cargo clippy --workspace --all-targets -- -D warnings

  fmt:
    runs-on: ubuntu-latest
    steps:
      - cargo fmt --workspace --check

  integration:
    runs-on: ubuntu-latest
    services:
      minio:
        image: minio/minio
        ...
      nessie:
        image: projectnessie/nessie:latest
        ...
    steps:
      - cargo test --workspace --features integration -- --test-threads=1

  compat-parquet:
    runs-on: ubuntu-latest
    steps:
      - pip install pyarrow pyiceberg
      - cargo test --test parquet_trailing_bytes
      - cargo test --test pyiceberg_read

  bench-regression:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - cargo bench --workspace -- --output-format bencher | tee bench_output.txt
      # Compare against baseline — fail if >10% regression

  compat-engines:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'   # only on main, expensive
    steps:
      - docker compose -f tests/docker/compose-engines.yml up -d
      - ./tests/compat/run_all_engines.sh
      - docker compose down
```

### Failure policy

| Test suite | Failure blocks |
|---|---|
| `unit` | Every PR |
| `clippy` | Every PR |
| `fmt` | Every PR |
| `integration` | Every PR |
| `compat-parquet` | Every PR |
| `bench-regression` | Every PR (>10% regression) |
| `compat-engines` | Release only (runs on main merge) |
