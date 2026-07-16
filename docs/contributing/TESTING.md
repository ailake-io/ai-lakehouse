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
|---|---|---|---|---|
| Unit | `src/` inline `#[cfg(test)]` | `cargo test` | Single function, no I/O |
| Integration | `tests/` at workspace root | `cargo test -p ailake-tests` | Multiple crates, local FS |
| Property-based | `src/` inline or `tests/` | `cargo test` | Invariants across random inputs |
| Benchmark | external [`ailake-benchmarks`](https://github.com/ThiagoLange/ailake-benchmarks) repo | `cargo run --release` (in that repo) | SIFT-1M write/index/search throughput + recall vs. LanceDB/pgvector/Deep Lake |
| UB detection (Miri) | `src/` inline `#[cfg(miri)]` | `ci-safety.yml` — every PR | `get_unchecked_mut`, SIMD intrinsics, CStr FFI, scalar edge cases |
| Concurrency model (Loom) | `src/` inline `#[cfg(feature = "loom")]` | `ci-safety.yml` — every PR | JNI table locks, shared codebooks, atomic counters |
| Compat (Python/DuckDB) | `tests/compat/` | `ci.yml` — every PR | PyArrow, DuckDB, PyIceberg, ailake-py SDK |
| Compat (Spark/Trino/JVM) | `tests/compat/` + Gradle | `compat-heavy.yml` — push to main + weekly | Spark+Iceberg, Trino+REST, Flink/Spark/Trino JVM plugins |

---

## JVM plugin tests (Kotlin/Scala) — the "native lib absent" trap

`spark-plugin`/`trino-plugin`/`ailake-flink` unit tests (`AilakeNativeTest`, `AilakeCatalogTest`, `AilakePageSinkTest`, ...) run against a real JNA-loaded `libailake_jni.so` in every CI job that touches them (`test-jvm` in `ci.yml`, `compat-jvm-plugins` in `compat-heavy.yml` — both `cargo build --release -p ailake-jni` first and set `LD_LIBRARY_PATH`/`AILAKE_LIB_PATH` or `-Dailake.native.lib=...` before `gradle test`, per `CONTRIBUTING.md` §4). Only a bare local `./gradlew test` with none of those set actually runs with the lib absent.

This has produced the same bug three times (see `CHANGELOG.md` "Fixed" entries for `finishPassesTextColumnsAsColumnsMapToNativeWriteBatch`, `writeBatchMultiDoesNotThrowWhenNativeLibAbsent`, `AilakeCatalogTest`'s `alterTable` test): a test names itself `...WhenNativeLibAbsent` and asserts a strict `null`/empty result, but with the lib present the native call actually executes — and two things make it *succeed* instead of failing like it would against real infra:

1. `LocalStore::new()` (`ailake-store/src/local.rs`) only strips a `file://` prefix. A fake `"s3://bucket/t/"` warehouse falls through as a literal relative path, so the "write" lands on local disk instead of failing like real S3 (no credentials/network) would.
2. `file:///tmp/test-table` is a real writable path reused across many tests *and* across plugins — `test-jvm` runs `gradle -p trino-plugin test` then `gradle -p spark-plugin test` in the same job, sharing `/tmp` on the runner, so a table one plugin's test wrote can already exist by the time another plugin's test runs against the same path.

**Rule**: never assert a JVM-plugin native call returns `null`/empty just because "the lib is absent in tests." Assert it returns *either* `null`/empty (lib truly absent or the call genuinely fails) *or* a valid success value (lib present and the call succeeds) — e.g. `assertTrue(result == null || result > 0)`. If a test needs a guaranteed-null result regardless of environment, force it structurally (empty `ids`, etc. — see `AilakeNative.writeBatch`'s `if (ids.isEmpty()) return null` guard), not by relying on the call failing.

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

### `ailake-query` — Iceberg schema mapping unit tests

`ailake-query/src/writer.rs` contains 7 inline unit tests for `arrow_schema_to_iceberg_update`:

| Test | What it checks |
|---|---|
| `schema_fields_match_arrow_columns` | Simple `Int64 + Utf8` → Iceberg `"long"` + `"string"`, field-ids start at 1 |
| `timestamp_with_tz_maps_to_timestamptz` | `Timestamp(Microsecond, Some("UTC"))` → `"timestamptz"` |
| `list_type_generates_nested_json` | `List<Utf8>` → `{"type":"list","element-id":N,"element":"string",...}` |
| `struct_type_generates_nested_json` | `Struct<f32>` → `{"type":"struct","fields":[...]}` |
| `vector_column_not_duplicated` | When vector column already in batch schema, not appended twice |
| `multi_vec_extra_policies` | Second vector column gets correct field-id (`N+2`) |
| `top_level_field_ids_align_with_parquet` | Iceberg field-ids match `PARQUET:field_id` stamps on all columns |

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

### SIFT-1M end-to-end (external repo)

Benchmarks live at **https://github.com/ThiagoLange/ailake-benchmarks**.

```bash
git clone https://github.com/ThiagoLange/ailake-benchmarks.git
cd ailake-benchmarks
bash scripts/download_sift1m.sh /data/sift1m
cargo run --release -- --dataset-dir /data/sift1m
```

What it measures:
- **Write phase**: 10 shards × 100k vectors, wall time + vec/s throughput
- **Index build**: time for background HNSW/IVF-PQ builds to reach `IndexStatus::Ready`
- **Index load**: time to `SearchSession::load()` all shards into memory
- **Search phase** (top_k=10, ef=50): Recall@10, QPS, mean/p50/p95/p99 latency

Reference results (SIFT-1M, x86_64 AVX2, 8 cores):

| Engine | Write | Index build | Recall@10 | QPS | p99 |
|--------|-------|-------------|-----------|-----|-----|
| `ailake` (HNSW deferred) | 199k vec/s | 165s (async) | 0.9963 | 1365 | 1.96ms |
| `ailake-ivf-pq-deferred` | 251k vec/s | 42.7s (async) | 0.9065 | 252 | 5.53ms |
| `ailake-auto` (HNSW) | 6.3k vec/s | 159s (inline) | 0.9960 | 1485 | 1.67ms |
| `lancedb` | 530k vec/s | 55s (inline) | 0.8805 | 745 | 63.34ms |

Engine selection guide:
- `ailake` — best recall, streaming ingestion (deferred build)
- `ailake-ivf-pq-deferred` — 100× smaller index; use when RAM is limited
- `ailake-auto` — hardware-adaptive; use in heterogeneous deployments
- `lancedb` — comparison baseline only

No in-repo `criterion` microbenchmarks exist today (no crate depends on `criterion`,
no `benches/` directory in the workspace) — `cargo bench` is not a working command
here. The SIFT-1M benchmark above is the only current benchmarking mechanism.

---

## CI matrix (GitHub Actions)

### `ci.yml` — manual dispatch (`workflow_dispatch`)

| Job | Command | What it covers |
|---|---|---|
| `fmt` | `cargo fmt --all -- --check` | Formatting |
| `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` | Lints |
| `deny` | `cargo deny check licenses advisories sources` | License + advisory audit |
| `unit` | `cargo test --workspace --lib --bins` | All unit tests |
| `integration` | `cargo test -p ailake-tests -- --test-threads=1` | End-to-end write/read/search, iceberg_compat |
| `index-cpu-fallback` | `cargo test -p ailake-index -- --nocapture` | Verifies `hardware::detect_backend()` returns `CpuSimd` and all index tests pass when no CUDA/ROCm libraries are present on the Linux runner |
| `compat-parquet` | `cargo test -p ailake-tests --test parquet_trailing_bytes --test positional_invariant` | Parquet spec compliance |
| `compat-pyarrow` | `write_fixture` + `pip install pyarrow` + `check_pyarrow.py` | PyArrow reads AI-Lake Parquet |
| `compat-duckdb` | `write_fixture` + `pip install duckdb` + `check_duckdb.py` | DuckDB reads via `parquet_scan` |
| `compat-pyiceberg` | `write_fixture` + `pip install pyiceberg[pyarrow]` + `check_pyiceberg.py` | PyIceberg `StaticTable.scan()` |
| `test-airflow-provider` | `pip install apache-airflow pytest` + `pytest tests/` | Airflow provider unit tests (2.x/3.x) |
| `compat-ailake-py` | `maturin build` (Python 3.12) + `check_ailake_py.py` | Python SDK write→search→assemble_context; `fts_text_columns` write + `search_text()` (Tantivy fast path); `search_multimodal` RRF |

### `ci-gpu.yml` — manual dispatch (`workflow_dispatch`)

Runs `ailake-index` unit + integration tests on GPU runners.

> **Note (v0.0.25):** Linux GPU jobs (`index-gpu-linux-cuda`, `index-gpu-linux-rocm`) are disabled (`if: false`) — no Linux GPU runner is currently registered. Only the Windows bare-metal job is active. Docker images (`docker/gpu-cuda/Dockerfile`, `docker/gpu-rocm/Dockerfile`) and `docker-compose.gpu.yml` are provided for local use.

| Job | Runner | Status | What it covers |
|---|---|---|---|
| `index-gpu-windows` | `[self-hosted, Windows, X64]` | **Active** | Detects CUDA (`cudart64_*.dll`) or ROCm (`amdhip64.dll`) via `Find-Dll` (PATH search); uses composite action `locate-rust-windows`; runs full `ailake-index` test suite including GPU unit tests in `src/gpu.rs` |
| `index-gpu-linux-cuda` | `[self-hosted, Linux, X64, gpu-nvidia]` | Disabled (`if: false`) | Builds `docker/gpu-cuda/Dockerfile` (FROM `nvidia/cuda:12.6.0-runtime-ubuntu22.04`); runs `cargo test -p ailake-index -- --nocapture` with `--gpus all`; exercises the Linux `libcuda.so.1` / `libcublas.so.12` libloading path |
| `index-gpu-linux-rocm` | `[self-hosted, Linux, X64, gpu-amd]` | Disabled (`if: false`) | Builds `docker/gpu-rocm/Dockerfile` (FROM `rocm/dev-ubuntu-22.04:6.2`); passes `--device /dev/kfd --device /dev/dri --group-add video`; exercises the Linux `libamdhip64.so` / `libhipblas.so` libloading path |

**GPU unit tests** (`ailake-index/src/gpu.rs`, gated on `AILAKE_GPU_BACKEND`):

| Test | Dataset | Assert |
|---|---|---|
| `gpu_search_batch_cosine_top1_exact` | 64 vecs × dim 16 | top-1 == query at dist ≈ 0 |
| `gpu_search_batch_euclidean_top1_exact` | 32 vecs × dim 8 | top-1 euclidean == anchor at dist = 0 |
| `gpu_kmeans_returns_k_centroids` | 4 clusters × 10 vecs, dim 8 | returns exactly k centroids of correct dim |

All three skip (not fail) when `AILAKE_GPU_BACKEND=none`.

**Runner requirements**:

- Windows job: Windows 10/11 or Server 2019+, NVIDIA CUDA Toolkit 11/12 or AMD ROCm for Windows, Rust stable toolchain.
- Linux/CUDA job: `nvidia-container-toolkit` installed and Docker configured; verify with `docker run --gpus all --rm nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`.
- Linux/ROCm job: AMD GPU with `amdgpu` kernel module, `/dev/kfd` and `/dev/dri` accessible; verify with `docker run --device /dev/kfd --device /dev/dri --group-add video --rm rocm/dev-ubuntu-22.04:6.2 rocm-smi`.

**Local developer usage** (no runner required):

```bash
# NVIDIA
docker compose -f docker-compose.gpu.yml run --rm gpu-cuda

# AMD
docker compose -f docker-compose.gpu.yml run --rm gpu-rocm

# Run only the gpu_data integration tests
docker compose -f docker-compose.gpu.yml run --rm gpu-cuda \
  cargo test -p ailake-index --test gpu_data -- --nocapture
```

> **Note (v0.0.25):** `ci-gpu-data.yml` was deleted — its `gpu_data` test target is a strict subset of `cargo test -p ailake-index` already run by `ci-gpu.yml`. The `gpu_data` tests are documented below for reference; they run in the Windows job.

**GPU data integration tests** (`ailake-index/tests/gpu_data.rs`, gated on `AILAKE_GPU_BACKEND`):

| Test | Dataset | Assert |
|---|---|---|
| `gpu_search_recall_vs_cpu_baseline` | 2 000 vecs × dim 128, 20 queries | GPU recall@10 ≥ 99% vs CPU brute-force |
| `gpu_search_exact_hit_in_large_db` | 5 000 vecs × dim 64, query == db[1337] | top-1 == row 1337, cosine dist ≈ 0 |
| `gpu_kmeans_converges_on_clustered_data` | 8 clusters × 50 vecs, dim 32 | each centroid maps unique cluster, dist < 1.0 |

All three skip when `AILAKE_GPU_BACKEND=none`.

### Composite action: `locate-rust-windows`

`.github/actions/locate-rust-windows/action.yml` — reusable composite action used by the `ci-gpu.yml` Windows job. Finds `cargo.exe` on a self-hosted Windows runner with three fallback levels:

1. Real toolchain binary inside `~\.rustup\toolchains\*\bin\`
2. Rustup shim at `~\.cargo\bin\`
3. `cargo` already on `PATH`

Fails the step with a descriptive error if cargo is not found. Adding the found directory to `$env:GITHUB_PATH` makes it available to all subsequent steps in the job.

### `ci-go.yml` — manual dispatch (`workflow_dispatch`)

| Job | Command | What it covers |
|---|---|---|
| `build` | `go build ./...` + `go vet ./...` | Go SDK compiles and passes vet |

### `ci-cpp.yml` — manual dispatch (`workflow_dispatch`)

| Job | Command | What it covers |
|---|---|---|
| `build` | `cmake -S ailake-cpp -B ailake-cpp/build` + `cmake --build` | C++17 SDK configures and builds (CPU-only, no CUDA) |

### `secret-scan.yml` — manual dispatch (`workflow_dispatch`) *(while repository is private)*

> **Note**: automatic `push` and `pull_request` triggers are commented out while the repository is private. Re-enable both in `secret-scan.yml` when the repository goes public so every push and external PR is scanned automatically.

| Job | Command | What it covers |
|---|---|---|
| `trufflehog` | `trufflesecurity/trufflehog@main --only-verified` | Secret scanning — blocks on verified credential leaks |

### `ci-safety.yml` — every PR and push

| Job | Command | What it covers |
|---|---|---|
| `miri` | `cargo miri test -p ailake-vec -p ailake-index -p ailake-jni -- miri_` | Miri (nightly): detects UB in `get_unchecked_mut` (visited tracker), scalar SIMD fallback paths, CStr FFI boundary. Each crate's `#[cfg(miri)]` tests exercise edge cases (zero vectors, dimension mismatch, null-terminated strings). |
| `loom` | `cargo test --features loom -p ailake-query -- loom_` | Loom (stable): explores all thread interleavings for JNI table-lock pattern, once-init flag, and `AtomicU32` batch counter. Limited to 2 threads / `LOOM_MAX_BRANCHES=10000` for bounded model checking. |

### `compat-heavy.yml` — manual dispatch (`workflow_dispatch`)

| Job | What it covers |
|---|---|
| `compat-spark` | PySpark: direct Parquet read + Spark+Iceberg HadoopCatalog SQL (`COUNT`, `MIN`/`MAX`, schema) |
| `compat-trino` | Trino: `tabulario/iceberg-rest` REST catalog + `trinodb/trino:436`; PyIceberg REST scan + Trino Python client |
| `compat-jvm-plugins` | `libailake_jni.so` C-ABI + Flink, Spark, Trino Gradle integration tests; includes FTS write (`fts_columns[]`) + `ailake_search_text_json` round-trip for Spark and Trino |
| `compat-bigquery` | BigQuery: `fsouza/fake-gcs-server` + `goccy/bigquery-emulator:0.6.6`; pyarrow reads AILK Parquet + BQ streaming inserts (`insertAll`); validates row count, schema, `MIN`/`MAX(id)` |

### `publish-jvm.yml` — manual fallback (`workflow_dispatch`)

Re-builds and re-uploads JVM plugin fat-JARs + `libailake_jni.so` to an existing GitHub Release **without** rerunning the full release pipeline. The canonical publish-jvm job now lives inside `release.yml` (see below).

| Input | Description |
|---|---|
| `tag` | Release tag to attach artifacts to (e.g. `v0.1.0`). Optional — derived from `Cargo.toml` when omitted. |

### `publish-pypi.yml` — manual fallback (`workflow_dispatch`)

Re-builds and re-publishes `ailake` wheels to PyPI + attaches to an existing GitHub Release **without** rerunning the full release pipeline. The canonical build+publish chain lives inside `release.yml`.

### `publish-airflow-provider.yml` — manual fallback (`workflow_dispatch`)

Re-builds and re-publishes `apache-airflow-providers-ailake` to PyPI + attaches to an existing GitHub Release. The canonical publish-airflow job lives inside `release.yml`.

### Failure policy

| Test suite | Failure blocks |
|---|---|
| `fmt`, `clippy` | Every PR |
| `unit`, `integration`, `compat-parquet` | Every PR |
| `compat-pyarrow`, `compat-duckdb`, `compat-pyiceberg`, `compat-ailake-py` | Every PR |
| `miri` (UB), `loom` (concurrency model) | Every PR |
| `compat-spark`, `compat-trino`, `compat-jvm-plugins`, `compat-bigquery` | Release (manual dispatch before triggering `release.yml`) |

---

## Manual Actions trigger order (pre-release)

All CI workflows are `workflow_dispatch`. Trigger in this order — each step must succeed before the next.

| Step | Workflow | What it does | Blocks on |
|---|---|---|---|
| 1 | **CI** (`ci.yml`) | Rust fmt/clippy/deny, unit, integration, compat Python/DuckDB/PyIceberg/ailake-py, Airflow provider tests | Must pass |
| 1b | **CI Safety** (`ci-safety.yml`) | Miri UB detection (nightly) + Loom concurrency model checking (stable). Runs in parallel with CI. | Must pass |
| 2 | **CI Go** (`ci-go.yml`) | Go SDK build + vet | Must pass |
| 3 | **CI C++** (`ci-cpp.yml`) | C++17 cmake build | Must pass |
| 4 | **CI GPU** (`ci-gpu.yml`) | GPU unit + data integration tests on Windows bare-metal; Linux jobs disabled (`if: false`); skips gracefully if `AILAKE_GPU_BACKEND=none` | Must pass (on GPU runner) |
| 5 | **Compat Heavy** (`compat-heavy.yml`) | Spark+Iceberg, Trino+REST, JVM plugins (Gradle), BigQuery emulator — Docker required | Must pass |
| 6 | **Release** (`release.yml`) | Triggered automatically on merge to `main` — runs all publishing steps sequentially (see chain below). Can also be triggered manually via `workflow_dispatch`. | Steps 1–5 green |

Step 4 requires the Windows GPU runner — can run in parallel with steps 2 and 3. Only the Windows job is currently active; Linux jobs (`index-gpu-linux-cuda`, `index-gpu-linux-rocm`) are disabled until a Linux GPU runner is registered.

### `release.yml` sequential chain

`release.yml` triggers on `push: branches: [main]` (automatic) and `workflow_dispatch` (manual).

The `release` job auto-bumps the patch version before tagging — **no manual version edits required**:

1. Reads the latest semver tag (`v*.*.*`) and increments the patch component (`v0.0.11` → `v0.0.12`).
2. Updates every `Cargo.toml` (crate version + inter-crate deps) via `sed`.
3. Commits the bump with `[skip ci]` and pushes to `main` — `[skip ci]` prevents a second workflow run.
4. Creates the git tag and GitHub Release on the bumped commit.
5. Runs the full publish chain sequentially.

```
merge develop → main  (or workflow_dispatch)
  └── release job
        ├── patch+1 from latest tag → bump all Cargo.toml → commit [skip ci] → push main
        ├── git tag vX.Y.Z → push
        ├── gh release create
        └── publish-crates → publish-jvm → publish-airflow
              └── pypi-linux (x86_64 → aarch64) → pypi-macos [disabled] → pypi-windows
                    └── pypi-sdist → pypi-publish
```

If any publish job fails, re-run only that job and its dependents — the tag and GitHub Release already exist.

**Fallback workflows** (re-publish without rerunning the full chain):

| Workflow | When to use |
|---|---|
| `publish-jvm.yml` | Re-upload JARs to existing release |
| `publish-airflow-provider.yml` | Re-publish Airflow provider to existing release |
| `publish-pypi.yml` | Re-build + re-publish Python wheels to existing release |
