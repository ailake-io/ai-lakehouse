# SETUP.md — Testing AI-Lake Format Locally

Guide for running the file format locally: writing batches, vector search with geometric pruning, compaction, ContextAssembler, Python bindings, layout inspection, and Parquet compatibility verification.

---

## Fastest path — Docker demo (no Rust toolchain required)

```bash
# From the repository root — builds ailake-py wheel on first run (~3-5 min, cached after)
docker compose -f tests/docker/compose-demo.yml up -d
```

Open **http://localhost:8888** — JupyterLab starts with 500 synthetic documents
already indexed and ready for vector search.

For engine demos (Trino + BigQuery emulator):

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines up -d
```

See [`tests/docker/`](./tests/docker/) for full details.

---

## Prerequisites

| Tool | Minimum version | Installation |
|---|---|---|
| Rust + Cargo | 1.75+ (stable) | `curl https://sh.rustup.rs -sSf \| sh` |
| Python 3 | 3.9+ | system / conda |
| PyArrow | any | `pip install pyarrow` |
| maturin | 1.4+ | `pip install maturin` *(only for Python bindings testing)* |

Verify:

```bash
rustc --version   # rustc 1.75+
cargo --version
python3 -c "import pyarrow; print(pyarrow.__version__)"
```

---

## 1. Clone and build

```bash
git clone https://github.com/ThiagoLange/ai-lakehouse.git
cd ai-lakehouse

# Default build — parallel CPU (rayon), no CUDA dependency
cargo build --workspace
```

First compilation takes ~2-3 min (downloads Arrow/Parquet dependencies).

### Build variants

```bash
# Cloud storage
cargo build --workspace --features store-s3      # Amazon S3
cargo build --workspace --features store-gcs     # Google Cloud Storage
cargo build --workspace --features store-azure   # Azure Blob
```

**Golden rule**: the default build (`cargo build`) compiles and runs on any
machine — CPU-only, NVIDIA, or AMD. There is no `ailake-index/gpu` flag anymore.
Both GPUs are detected at runtime via `libloading` with no build dependency.

---

## 2. Full test suite

```bash
# Unit tests for all crates
cargo test --workspace --lib

# Integration tests (write + read + search end-to-end)
cargo test -p tests

# All at once
cargo test --workspace
```

Should finish with `112 passed` (2 ignored — doctests requiring live credentials or runtime context).

### Tests by crate

| Crate | What it covers |
|---|---|
| `ailake-vec` | F32→F16 quantization, PQ (encode/decode/ADC), BlockCompressor (zstd/lz4), centroids, `exact_distance`, SIMD (AVX2/NEON), RaBitQ encoding, binary sign quantization (MSB-first) |
| `ailake-index` | HNSW build/search, IVF-PQ (train/search/serialize), RaBitQ (encode/search/serialize), Binary Hamming (encode/search/serialize), GPU k-means dispatch, bincode serialization, MmapLoader round-trip |
| `ailake-file` | Unified file write/read, AILK layout (HNSW, IVF-PQ, RaBitQ, Binary Hamming), integrity |
| `ailake-query` | ContextAssembler, geometric pruning, post-PQ reranking, MemTableWriter, write_batch_ivf_pq; `arrow_schema_to_iceberg_update` (automatic schema propagation on commit) |
| `ailake-parquet` | write_batch_multi_vec / read_all_multi_vec (multi-vector columns) |
| `tests` (integration) | write→read→search end-to-end, positional invariant, PyArrow compatibility, pruning, context assembler |

---

## 3. Phase 2 tests in detail

### 3A. Product Quantization (PQ)

```bash
cargo test -p ailake-vec -- pq
```

Tests:
- `encode_decode_roundtrip_approx` — encode + decode preserves dimension
- `adc_distance_non_negative` — ADC distance ≥ 0 always
- `nearest_neighbor_rank_preserved` — q1 closer to cluster 1 than cluster 2
- `dim_not_divisible_errors` — error if `dim % M != 0`

### 3B. BlockCompressor (zstd/lz4)

```bash
cargo test -p ailake-vec -- compress
```

Tests compression/decompression round-trip for `None`, `Lz4`, and `Zstd` codecs.

### 3C. MmapLoader

```bash
cargo test -p ailake-index -- mmap
```

Tests that HNSW bytes written to tempfile and opened via mmap deserialize correctly.

### 3D. Geometric pruning (integration)

```bash
cargo test -p tests --test vector_pruning
```

Creates two files:
- **File A**: vectors near `[1, 0, 0, 0]`
- **File B**: vectors near `[0, 0, 0, 1]`

Search with query `[1, 0, 0, 0]` and `pruning_threshold = 0.5`. File B should be eliminated — all results come from `part-00000.parquet`.

### 3E. ContextAssembler (integration)

```bash
cargo test -p tests --test context_assembler
```

- `dedup_removes_near_identical_chunks` — identical embeddings → only 1 chunk survives
- `grouping_restores_chunk_order` — out-of-order chunks → XML with ascending `chunk_index`

### 3F. Cloud storage — credential builders

`ailake-store` exposes typed per-cloud builders to configure credentials explicitly. For quick use via env, `store_from_url` uses each cloud's default credential chain.

#### S3 — credential variants

```rust
use ailake_store::{s3_store, S3Config, S3Credentials};

// Development / MinIO / LocalStack — explicit key
let store = s3_store(S3Config {
    bucket: "my-bucket".into(),
    region: "us-east-1".into(),
    endpoint: Some("http://localhost:9000".into()),
    allow_http: true,
    credentials: S3Credentials::Static {
        access_key_id: "minioadmin".into(),
        secret_access_key: "minioadmin".into(),
        session_token: None,
    },
}, "warehouse/")?;

// EC2 with IAM Instance Profile (IMDSv2)
let store = s3_store(S3Config {
    bucket: "prod-bucket".into(),
    region: "us-east-1".into(),
    endpoint: None,
    allow_http: false,
    credentials: S3Credentials::InstanceProfile,
}, "warehouse/")?;

// EKS with IRSA (AWS_WEB_IDENTITY_TOKEN_FILE + AWS_ROLE_ARN injected by EKS controller)
let store = s3_store(S3Config {
    bucket: "prod-bucket".into(),
    region: "us-east-1".into(),
    endpoint: None,
    allow_http: false,
    credentials: S3Credentials::WebIdentity,
}, "warehouse/")?;

// Full automatic chain: env vars → ~/.aws → WebIdentity → IMDSv2
let store = s3_store(S3Config {
    bucket: "prod-bucket".into(),
    region: "us-east-1".into(),
    endpoint: None,
    allow_http: false,
    credentials: S3Credentials::Default,
}, "warehouse/")?;

// URL-based — uses S3Credentials::Default, region read from AWS_DEFAULT_REGION
let store = ailake_store::store_from_url("s3://prod-bucket/warehouse")?;
```

Requires feature `store-s3`:
```bash
cargo build --features ailake-store/store-s3
```

#### GCS — credential variants

```rust
use ailake_store::{gcs_store, GcsConfig, GcsCredentials};

// Service account JSON file
let store = gcs_store(GcsConfig {
    bucket: "my-gcs-bucket".into(),
    credentials: GcsCredentials::ServiceAccountFile("/secrets/sa.json".into()),
}, "warehouse/")?;

// Inline JSON (from secrets manager / env var)
let store = gcs_store(GcsConfig {
    bucket: "my-gcs-bucket".into(),
    credentials: GcsCredentials::ServiceAccountJson(
        std::env::var("GCP_SA_JSON")?,
    ),
}, "warehouse/")?;

// GKE Workload Identity / Cloud Run / GOOGLE_APPLICATION_CREDENTIALS
// → metadata server used automatically when env var absent
let store = gcs_store(GcsConfig {
    bucket: "my-gcs-bucket".into(),
    credentials: GcsCredentials::ApplicationDefault,
}, "warehouse/")?;

// URL-based — uses ApplicationDefault
let store = ailake_store::store_from_url("gs://my-gcs-bucket/warehouse")?;
```

Requires feature `store-gcs`:
```bash
cargo build --features ailake-store/store-gcs
```

#### Azure Blob / ADLS Gen2 — credential variants

```rust
use ailake_store::{azure_store, AzureConfig, AzureCredentials};

// Service principal (Entra app registration) — production
let store = azure_store(AzureConfig {
    account_name: "mystorageaccount".into(),
    container: "my-container".into(),
    credentials: AzureCredentials::ClientSecret {
        tenant_id: std::env::var("AZURE_TENANT_ID")?,
        client_id: std::env::var("AZURE_CLIENT_ID")?,
        client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
    },
}, "warehouse/")?;

// Managed Identity — system-assigned (no client_id)
let store = azure_store(AzureConfig {
    account_name: "mystorageaccount".into(),
    container: "my-container".into(),
    credentials: AzureCredentials::ManagedIdentity { client_id: None },
}, "warehouse/")?;

// Managed Identity — user-assigned
let store = azure_store(AzureConfig {
    account_name: "mystorageaccount".into(),
    container: "my-container".into(),
    credentials: AzureCredentials::ManagedIdentity {
        client_id: Some("00000000-0000-0000-0000-000000000000".into()),
    },
}, "warehouse/")?;

// Storage account access key (dev / admin)
let store = azure_store(AzureConfig {
    account_name: "mystorageaccount".into(),
    container: "my-container".into(),
    credentials: AzureCredentials::AccessKey(
        std::env::var("AZURE_STORAGE_KEY")?,
    ),
}, "warehouse/")?;

// URL-based — uses ManagedIdentity, account read from AZURE_STORAGE_ACCOUNT_NAME
let store = ailake_store::store_from_url("az://my-container/warehouse")?;
```

Requires feature `store-azure`:
```bash
cargo build --features ailake-store/store-azure
```

#### Local MinIO for S3 testing

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create bucket via console at http://localhost:9001
```

Use `S3Credentials::Static` with `endpoint: Some("http://localhost:9000")` and `allow_http: true` as shown above.

---

## 4. Full demo — write, search, inspect

The `demo` example (in `ailake-query/examples/demo.rs`) runs the complete flow on the local filesystem:

1. Creates an AI-Lake table with 2 files (500 rows each)
2. Prints the binary layout of the file (offsets of PAR1, AILK, HNSW)
3. Searches top-5 by cosine similarity (no pruning — `pruning_threshold = f32::INFINITY`)
4. Verifies integrity of both files
5. Lists the Iceberg catalog

```bash
cargo run --example demo -p ailake-query
```

Expected output:

```
Workspace: /tmp/ailakeXXXXXX

=== Writing 2 batches (500 rows each) ===
  part-00000.parquet written
  part-00001.parquet written
  Committed snapshot id=1

=== File layout inspection (part-00000.parquet) ===
  File layout (NNNNN bytes):
    PAR1 #1 at byte 0
    AILK magic at byte XXXXX
    AILK magic at byte XXXXX
    PAR1 #2 at byte NNNNN-4
    AILK section    : XXXXX..XXXXX
    Centroid section: XXXXX..XXXXX
    HNSW section    : XXXXX..XXXXX (YYYYY bytes)
    Record count    : 500
    Dim             : 16

=== Search: query = embs1[0] (should be top result) ===
  Top-5 results:
    #1: row_id=0 distance=0.000000  file=data/part-00000.parquet
    ...

PASS: top result distance = X.XXe-XX < 0.01

=== Integrity check on both files ===
  data/part-00000.parquet — 500 nodes, integrity OK
  data/part-00001.parquet — 500 nodes, integrity OK

=== Catalog: list_files ===
  data/part-00000.parquet — 500 rows, hnsw_offset=XXXXX, hnsw_len=XXXXX
  data/part-00001.parquet — 500 rows, hnsw_offset=XXXXX, hnsw_len=XXXXX

Phase 1 demo completed successfully.
```

---

## 5. Testing geometric pruning in the demo

To see pruning in action, create two files with vectors in opposite directions and search with a low threshold:

```rust
// SearchConfig with active pruning
let results = search(
    &table, &query,
    SearchConfig {
        top_k: 5,
        ef_search: 50,
        pruning_threshold: 0.5,  // f32::INFINITY = no pruning
    },
    "embedding", dim, catalog, store,
).await?;
```

`pruning_threshold` controls aggressiveness: lower = more files eliminated = faster, potentially lower recall.

---

## 6. Testing Python bindings (ailake-py)

```bash
cd ailake-py
pip install maturin pyarrow numpy

# Compile and install in current Python env
maturin develop

# Verify import
python3 -c "import ailake; print(dir(ailake))"
```

Using the Python SDK:

```python
import ailake
import numpy as np

# Write — fluent API (recommended)
table = ailake.open_table("/tmp/ailake-test", dim=64, metric="cosine")
table.insert(
    texts=["text chunk 1", "text chunk 2"],
    embeddings=np.random.rand(2, 64).astype(np.float32),
)
snapshot_id = table.commit()
print(f"Snapshot: {snapshot_id}")

# Search — chainable, materialise to pandas / polars / list
query = np.random.rand(64).astype(np.float32)
df = ailake.search("/tmp/ailake-test", query, top_k=5).to_pandas()
print(df)

# Async (optional)
import asyncio
async def run():
    df = await table.search(query).limit(5).to_pandas_async()
asyncio.run(run())

# ContextAssembler
ctx = ailake.assemble_context(
    chunks=[
        {"document_id": "doc-1", "chunk_index": 0, "chunk_text": "Text...", "distance": 0.1},
    ],
    max_tokens=4096,
    dedup_threshold=0.05,
)
print(ctx)
```

> `ailake.TableWriter(path, ...)` is still supported for backward compatibility.

---

## 7. Verify PyArrow compatibility manually

```bash
# Generate temporary file via test
cargo test -p tests --test parquet_trailing_bytes -- --ignored --nocapture 2>&1 | grep -i "path\|file\|ok\|FAILED"
```

Or point to a file generated by the demo:

```python
import pyarrow.parquet as pq

# Replace with path printed by demo ("Workspace: ...")
path = "/tmp/ailakeXXXXXX/warehouse/default/demo_table/data/part-00000.parquet"

table = pq.read_table(path)
print(f"Rows: {table.num_rows}")
print(f"Schema: {table.schema}")
print(table.to_pandas().head())
```

PyArrow should read normally — columns `id`, `text`, and `embedding` (as bytes). No magic or footer errors.

---

## 7B. Verify PyIceberg compatibility (StaticTable scan)

Requires PyIceberg installed:
```bash
pip install "pyiceberg[pyarrow]"
```

Generate fixture and run verification:
```bash
# Generate fixture (1000 rows, dim=8, F16)
cargo run --example write_fixture -p ailake-query

# Verify compatibility
python tests/compat/check_pyiceberg.py ./compat-fixture
```

Expected output:
```
PASS (StaticTable): PyIceberg read 1000 rows, schema=['id', 'text', 'embedding']
```

What the test validates:
- `StaticTable.from_metadata` reads `vN.metadata.json` via `version-hint.text`
- Avro OCF manifests are read correctly (field-ids preserved by `avro_raw.rs`)
- `table.scan().to_arrow()` returns 1000 rows with schema `[id, text, embedding]`
- `embedding` column read as `fixed_size_binary[16]` (F16, dim=8)

---

## 8. Running benchmarks

Benchmarks live in a separate repository:
**https://github.com/ThiagoLange/ailake-benchmarks**

### 8A. Setup

```bash
git clone https://github.com/ThiagoLange/ailake-benchmarks.git
cd ailake-benchmarks

# Dataset download (~164 MB)
bash scripts/download_sift1m.sh /data/sift1m
```

### 8B. Run

```bash
# Default: AI-Lake HNSW deferred
cargo run --release -- --dataset-dir /data/sift1m

# Smoke-test (10k vectors)
cargo run --release -- --dataset-dir /data/sift1m --limit 10000

# IVF-PQ deferred
cargo run --release -- --dataset-dir /data/sift1m --engine ailake-ivf-pq-deferred

# HNSW M/ef comparison (M=8 vs M=16 default vs M=32)
cargo run --release -- --dataset-dir /data/sift1m --engine ailake-hnsw-compare

# Compare vs LanceDB
cargo run --release --features lancedb-bench -- \
    --dataset-dir /data/sift1m --engine all

# Compare vs pgvector
cargo run --release --features pgvector-bench -- \
    --dataset-dir /data/sift1m \
    --engine pgvector \
    --pgvector-url "host=localhost user=postgres password=postgres dbname=postgres"
```

Expected results (SIFT-1M, top_k=10, x86_64 AVX2 8-core):

```
Engine                    Write         Index build   Recall@10   QPS      p99
──────────────────────────────────────────────────────────────────────────────
AI-Lake HNSW deferred     199k vec/s    165s defer    0.9963      1365     1.96ms
AI-Lake IVF-PQ deferred   251k vec/s    42.7s defer   0.9065      252      5.53ms
AI-Lake Auto/HNSW         6.3k vec/s    159s inline   0.9960      1485     1.67ms
LanceDB IVF-HNSW-SQ       530k vec/s    55s inline    0.8805      745     63.34ms
```

HNSW M/ef comparison (SIFT-1M, top_k=10):

```
Config              Index build   Recall@10   QPS    p99     Use case
────────────────────────────────────────────────────────────────────
M=8,  ef=100         55s          0.9850      2053   0.86ms  Low latency / high QPS
M=16, ef=150 (def)  139s          0.9960      1366   1.28ms  General purpose
M=32, ef=400        435s          0.9985       908   1.91ms  Max recall (medical, legal)
```

Optional pgvector parameters:
```bash
--pgvector-m 16              # HNSW m (default: 16)
--pgvector-ef-construction 64 # ef_construction (default: 64)
--pgvector-ef-search 50      # ef_search at query time (default: 50)
```

### 8D. Deep Lake benchmark (Python, optional)

Deep Lake free tier only supports exact search (brute-force). The script below measures write and exact search throughput on a subset:

```bash
pip install deeplake numpy
python3 scripts/deeplake_bench.py  # run from ailake-benchmarks repo \
    --dataset-dir data/sift1m \
    --limit 10000
```

> **Note**: Approximate ANN (Deep Memory) requires a paid Activeloop plan. Recall comparison with AI-Lake/pgvector/LanceDB is not direct.

### 8E. Criterion microbenchmarks

```bash
# HNSW search benchmark (ailake-index)
cargo bench -p ailake-index

# Write benchmark (ailake-file)
cargo bench -p ailake-file
```

---

## 8F. GPU search — NVIDIA CUDA and AMD ROCm

### Licensing note — third-party GPU SDKs

NVIDIA CUDA Toolkit and AMD ROCm are **proprietary software owned by their respective vendors**.
The AI-Lake repository (MIT OR Apache-2.0) does not bundle, redistribute, or statically link
these SDKs in its default configuration.

| SDK | Owner | License | How AI-Lake uses it |
|---|---|---|---|
| NVIDIA CUDA Toolkit (`libcudart`, `libcublas`) | NVIDIA Corporation | [CUDA EULA](https://docs.nvidia.com/cuda/eula/) | **Rust/Go**: runtime dlopen via `libloading` — zero build dependency. **C++ SDK**: opt-in static link when `-DAILAKE_CUDA=ON` |
| AMD ROCm (`libamdhip64`, `libhipblas`) | Advanced Micro Devices | [ROCm License](https://rocm.docs.amd.com/en/latest/about/license.html) | **Rust/Go**: runtime dlopen via `libloading` — zero build dependency |

Binary distributions of this SDK must not bundle NVIDIA or AMD proprietary libraries.
Vendors distributing `ailake-cpp` compiled with `-DAILAKE_CUDA=ON` are responsible for
complying with the NVIDIA CUDA EULA.

---

### Rust + Go SDK: runtime detection (no build dependency)

Two independent GPU backends. Both use `libloading` dlopen at runtime —
**zero build dependency** for any GPU. The same binary runs on
CPU-only, NVIDIA, and AMD without recompilation.

| Backend | Detection | Compute | Build requirement |
|---------|----------|---------|-------------------|
| **NVIDIA CUDA** | `libcuda.so.1` + `cuDeviceGetCount` | cuBLAS SGEMM via libloading | **none** — runtime only |
| **AMD ROCm** | `libamdhip64.so` + `hipGetDeviceCount` | hipBLAS SGEMM via libloading | **none** — runtime only |
| **CPU** | fallback | parallel rayon | none |

**Detection priority**: AMD ROCm → NVIDIA CUDA → CPU SIMD.
AMD is checked first because ROCm systems often expose a CUDA compatibility layer (`libcuda.so.1`), which would incorrectly identify as NVIDIA without this priority.

---

### 8F-1. NVIDIA CUDA

#### Runtime prerequisites (none at build time)

| Requirement | Minimum version |
|-----------|---------------|
| NVIDIA GPU | Maxwell+ architecture (sm_50) |
| NVIDIA Driver | ≥ 450 |
| `libcudart.so` | CUDA 11.x or 12.x in `LD_LIBRARY_PATH` |
| `libcublas.so` | CUDA 11.x or 12.x in `LD_LIBRARY_PATH` |

CUDA Toolkit, `nvcc`, or headers are not required — only the runtime libraries.

#### Build — no additional flags needed

```bash
# Default build — NVIDIA detected and used at runtime automatically
cargo build --release
```

The binary detects CUDA via `libloading` (dlopen of `libcuda.so.1`,
`libcudart.so`, `libcublas.so`). Without GPU: silently falls back to AMD ROCm or CPU.

#### Verify CUDA detection

```rust
use ailake_index::{detect_backend, HardwareBackend};
assert_eq!(detect_backend(), HardwareBackend::NvidiaCuda);
```

---

### 8F-2. AMD ROCm

#### Prerequisites

| Requirement | Minimum version |
|-----------|---------------|
| AMD GPU | GCN4+ (Polaris) or RDNA1+ recommended |
| ROCm | 5.0+ (`libamdhip64.so` + `libhipblas.so`) |
| amdgpu driver | included in ROCm |

#### Build — no additional flags needed

```bash
# Default build — ROCm detected and used at runtime automatically
cargo build --release

# ROCm + CUDA at the same time (ROCm takes priority as it is detected first)
cargo build --release --features ailake-index/gpu
```

The ROCm backend uses `libhipblas.so` via `libloading` — no build dependency. The binary runs on any machine and only uses ROCm when `libamdhip64.so` + `libhipblas.so` are installed.

#### Verify ROCm detection

```bash
# Via HardwareProfile::detect()
use ailake_index::{detect_backend, detect_rocm, HardwareBackend};
assert!(detect_rocm());
assert_eq!(detect_backend(), HardwareBackend::AmdRocm);
```

---

### 8F-3. Verify detection via benchmark

The `ailake-auto` engine prints the hardware profile before running:

```bash
cargo run --release -- \
    --dataset-dir data/sift1m --engine ailake-auto --limit 50000
```

Expected output on NVIDIA machine:
```
Hardware detection:
  Backend      : NVIDIA CUDA
  CUDA GPU     : true
  ROCm GPU     : false
  CPU cores    : 16
  AVX2         : true
  AVX-512F     : false
  Index chosen : IVF-PQ  (shard_size=100000)
```

Expected output on AMD ROCm machine:
```
Hardware detection:
  Backend      : AMD ROCm
  CUDA GPU     : false
  ROCm GPU     : true
  CPU cores    : 16
  AVX2         : true
  AVX-512F     : false
  Index chosen : IVF-PQ  (shard_size=100000)
```

Expected output on CPU-only:
```
Hardware detection:
  Backend      : CPU (no GPU)
  CUDA GPU     : false
  ROCm GPU     : false
  CPU cores    : 8
  AVX2         : true
  AVX-512F     : false
  Index chosen : HNSW  (shard_size=100000)
```

---

### 8F-4. When GPU beats CPU

| File (vectors) | dim | CPU rayon | NVIDIA/AMD GPU |
|----------------|-----|-----------|----------------|
| 10k | 1536 | ~2 ms | ~3 ms (overhead dominates) |
| 100k | 1536 | ~20 ms | ~2 ms |
| 500k | 1536 | ~100 ms | ~4 ms |

GPU wins from ~50k vectors/file for dim=1536. Applies to both vendors.

---

### 8F-5. Supported metrics on GPU

| Metric | NVIDIA (cuBLAS SGEMM) | AMD (hipBLAS SGEMM) |
|---------|-----------------------|---------------------|
| **Cosine** | normalize CPU → SGEMM → `1 + raw` | normalize CPU → SGEMM → `1 + raw` |
| **Euclidean** | SGEMM (−2·q·d) + norms CPU → sqrt | SGEMM (−2·q·d) + norms CPU → sqrt |
| **DotProduct** | SGEMM with alpha=−1 | SGEMM with alpha=−1 |

Both backends use identical SGEMM formulation (`C[N×Q col-major] = alpha · db^T · queries`); they differ only in operation constants (`CUBLAS_OP_T=1` vs `HIPBLAS_OP_T=112`) and library names.

---

### 8F-6. GPU k-means for IVF-PQ

`IvfPqIndex` training (k-means for coarse centroids and `PQCodebook`) uses GPU via `kmeans_dispatch` with NVIDIA → AMD → CPU priority:

```
NVIDIA (runtime)  →  try_nvidia_kmeans (cuBLAS SGEMM via libloading)
AMD ROCm (runtime)  →  try_rocm_kmeans (hipBLAS SGEMM via libloading)
CPU fallback        →  kmeans_centroids (parallel rayon)
```

- **No GPU at runtime**: `kmeans_dispatch` silently falls back to CPU.
- **NVIDIA without libcublas installed**: even if `libcuda.so.1` exists, `try_nvidia_kmeans` returns `None` if `libcublas.so` is not found → tries AMD → CPU.
- **AMD without hipBLAS installed**: even if `libamdhip64.so` exists, `try_rocm_kmeans` returns `None` if `libhipblas.so` is not found → CPU fallback.

```bash
# Default build — GPU accelerated automatically if available (NVIDIA or AMD)
cargo build --release
```

---

## 8G. HNSW M/ef tuning (`--hnsw-m`, `--hnsw-ef`)

HNSW M and ef_construction are now per-table parameters stored in Iceberg metadata.

```bash
# Set at table creation time via CLI
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine \
    --hnsw-m 32 --hnsw-ef 400

# Or via Python
writer = ailake.TableWriter("s3://my-lake/docs/",
    metric="cosine", hnsw_m=32, hnsw_ef_construction=400)
```

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `--hnsw-m` | 16 | 4–64 | Connections per node. Higher → better recall, more memory, slower build |
| `--hnsw-ef` | 150 | 40–500 | Build candidate pool. Higher → better graph quality, slower build |

Quick guide:

| Goal | M | ef_construction |
|---|---|---|
| Low latency / max QPS | 8 | 100 |
| General purpose (default) | 16 | 150 |
| High recall (production RAG) | 24 | 200 |
| Max recall (medical, legal) | 32 | 400 |

`ef_search` (query-time) is separate — set via `SearchConfig.ef_search` (Rust) or the `--ef` flag in the benchmark.

## 8H. IVF-PQ benchmark (`--engine ailake-ivf-pq`)

IVF-PQ uses an index with inverted lists encoded by Product Quantization, instead of the HNSW graph. Index is 10-100× smaller — a good choice for S3 where sequential access is cheaper.

```bash
# Basic benchmark (SIFT-1M, defaults: nlist=256, nprobe=8, pq_m=8)
cargo run --release -- --dataset-dir data/sift1m --engine ailake-ivf-pq

# Adjust parameters
cargo run --release -- \
    --dataset-dir data/sift1m \
    --engine ailake-ivf-pq \
    --ivf-nlist 512 \
    --ivf-nprobe 16 \
    --ivf-pq-m 16

# Full comparison AI-Lake HNSW + IVF-PQ + LanceDB + pgvector
cargo run --release --features lancedb-bench,pgvector-bench -- \
    --dataset-dir data/sift1m \
    --engine all \
    --pgvector-url "host=localhost user=postgres password=postgres dbname=postgres"
```

IVF-PQ parameters:

| Flag | Default | Description |
|---|---|---|
| `--ivf-nlist` | 256 | Voronoi cells (inverted lists). Rule: `sqrt(n)` |
| `--ivf-nprobe` | 8 | Cells queried per query. Higher = better recall |
| `--ivf-pq-m` | 8 | PQ sub-vectors. Must divide 128 for SIFT |

Expected result (SIFT-1M, CPU, nlist=256, nprobe=8):
```
AI-Lake IVF-PQ write phase (nlist=256, nprobe=8, pq_m=8) …
  Throughput    : ~1800 vec/s  (k-means slower than HNSW insert)

Search phase  (top_k=10)
  Recall@10     : ~0.94
  QPS           : ~2100
  Index size    : ~5 MB  (vs ~80 MB HNSW for 1M vectors dim=128)
```

---

## 8I. RaBitQ flat index (`--rabitq`)

RaBitQ uses 1 bit/dim (packed sign bits) after a **modified Gram-Schmidt orthonormal random rotation** — 16× smaller than F16 with better recall than naive binary quantization via an unbiased XOR/popcount IP estimator. No graph construction: write is one-pass O(n), making it the fastest index to build. Search is sequential O(N) flat scan with O(N) partial select; outer shard parallelism handles concurrency.

### When to use RaBitQ

| Criterion | RaBitQ | HNSW | IVF-PQ |
|---|---|---|---|
| Write throughput | **~163k vec/s** (SIFT-1M measured) | ~50k vec/s | ~200k vec/s |
| Storage (dim=1536) | **200 bytes/vec** | ~10 MB/50k vecs | ~2 MB/50k vecs |
| Recall@10 cosine (rerank≥3) | 0.85–0.95 | ≥0.95 | 0.90–0.95 |
| Recall@10 Euclidean (rerank=3) | ~0.67 | ≥0.95 | 0.90–0.95 |
| Search QPS (SIFT-1M, 10 shards) | ~101 | ~1400 | ~380 |
| Graph build overhead | **None** | O(n log n) | O(n) k-means |
| Best use case | High-insert, extreme compression, cosine | Online search | S3 cold storage |

RaBitQ is designed for **cosine** workloads. Euclidean recall is lower because the IP estimator adds approximation noise when converting to L2. Use `rerank_factor ≥ 10` for complex datasets; `rerank_factor ≥ 3` is sufficient for most real-world embedding models.

Use RaBitQ when storage is the primary constraint and you can afford a second pass for reranking.

### CLI usage

```bash
# Create table with RaBitQ flat index
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine --rabitq

# With custom seed (for reproducibility across shards)
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine \
    --rabitq --rabitq-seed 42

# Without raw F16 storage (extreme compression, no reranking)
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine \
    --rabitq --rabitq-seed 42 --no-rabitq-keep-raw
```

### Python usage

```python
import ailake

writer = ailake.TableWriter(
    "s3://my-lake/docs/",
    dim=1536,
    metric="cosine",
    rabitq=True,
    rabitq_seed=42,       # same seed across all shards → comparable distances
    rabitq_keep_raw=True, # keep F16 for reranking (recommended)
)
writer.write_batch(texts, embeddings)
writer.commit()

# Search with reranking: fetch top_30, rerank with raw F16, return top_10
results = ailake.search(
    path="s3://my-lake/docs/",
    query=query_embedding,
    top_k=10,
    rerank_factor=10,  # recommended: ≥ 3 for most datasets, ≥ 10 for complex/Euclidean
)
```

### Storage comparison (dim=1536, 1M vectors)

| Index | Bytes/vector | Total (1M vecs) |
|---|---|---|
| F32 raw | 6 144 | 6 GB |
| F16 raw | 3 072 | 3 GB |
| HNSW graph (F16) | ~3 200 | ~3.2 GB |
| IVF-PQ (M=48) | ~50 | ~50 MB |
| **RaBitQ (no raw)** | **192** | **192 MB** |
| RaBitQ + raw F16 | 3 264 | ~3.3 GB (codes + F16) |

**Note**: with `keep_raw=False`, only the binary codes and norms are stored — 192 bytes/vec for dim=1536, with no reranking possible. With `keep_raw=True` (default), raw F16 vectors are also stored for reranking — total is similar to HNSW but with faster writes.

---

## 8J. Binary Hamming flat index (`--binary`)

Binary Hamming uses 1 bit/dim packed MSB-first — no rotation matrix, no k-means. Quantization rule: `bit_i = (x_i >= 0.0)`. Storage: 192 bytes/vector at dim=1536 (32× smaller than F32). Write throughput exceeds 200k vec/s. Search is sequential Hamming scan with SIMD dispatch (AVX2+SSSE3 → NEON → scalar u64 popcnt). Lower recall than RaBitQ (no orthonormal rotation to spread signs); best for cosine with reranking.

### CLI usage

```bash
# Create table with Binary Hamming flat index
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine --binary

# Without raw F16 storage (maximum compression, no reranking)
ailake create s3://my-lake/docs/ --dim 1536 --metric cosine --binary --no-binary-keep-raw
```

### Python usage

```python
import ailake

writer = ailake.TableWriter(
    "s3://my-lake/docs/",
    dim=1536,
    metric="cosine",
    binary=True,
    binary_keep_raw=True,  # keep F16 for reranking (recommended)
)
writer.write_batch(texts, embeddings)
writer.commit()

results = ailake.search(
    path="s3://my-lake/docs/",
    query=query_embedding,
    top_k=10,
    rerank_factor=3,  # ≥ 3 for cosine; ≥ 10 for Euclidean/complex
)
```

### Storage comparison (dim=1536, 1M vectors)

| Index | Bytes/vector | Total (1M vecs) | Recall@10 cosine |
|---|---|---|---|
| HNSW (F16) | ~3 200 | ~3.2 GB | ≥ 0.95 |
| IVF-PQ (M=48) | ~50 | ~50 MB | 0.90–0.95 |
| RaBitQ (no raw) | 192 | 192 MB | 0.70–0.85 |
| RaBitQ + raw F16 | 3 264 | ~3.3 GB | 0.85–0.95 |
| **Binary (no raw)** | **192** | **192 MB** | 0.50–0.70 |
| Binary + raw F16 | 3 264 | ~3.3 GB | 0.80–0.92 |

C++ SDK: 14 unit tests in `ailake-cpp/tests/test_binary.cpp` cover `f32_to_bits`, `hamming_distance`, and `binary_search` (empty, zero top_k, top-k cap, F16 rerank).

---

## 9. Testing RestCatalog — multi-cloud

`RestCatalog` implements the [Iceberg REST Catalog spec](https://iceberg.apache.org/spec/#rest-catalog) and works with Polaris, Nessie, S3 Tables, AWS BigLake, and Unity Catalog.

### 9A. Unit tests (no external server)

```bash
cargo test -p ailake-catalog
```

Covers URL building, `CommitTableRequest` serialization, storage root derivation, and Databricks configs.

### 9B. RestCatalog locally with Nessie

```bash
# Start Nessie (Project Nessie — catalog with branching)
docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest

# Run integration test (requires server)
cargo test -p tests --test rest_nessie -- --ignored
```

Manual Rust configuration:

```rust
use ailake_catalog::{RestCatalog, RestCatalogAuth, RestCatalogConfig};
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "http://localhost:19120/api".into(),
        prefix: Some("main".into()),
        warehouse: Some("/tmp/warehouse".into()),
        auth: RestCatalogAuth::None,
    },
    store,
);
```

### 9C. RestCatalog locally with Apache Polaris

```bash
docker run -p 8181:8181 apache/polaris:latest

cargo test -p tests --test rest_polaris -- --ignored
```

Configuration:

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "http://localhost:8181".into(),
        prefix: Some("my_polaris_catalog".into()),
        warehouse: Some("s3://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::Bearer("my-bootstrap-token".into()),
    },
    store,
);
```

### 9D. AWS S3 Tables (native REST on AWS)

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://s3tables.us-east-1.amazonaws.com/iceberg".into(),
        prefix: Some("arn:aws:s3tables:us-east-1:123456789012:bucket/my-bucket".into()),
        warehouse: None,
        auth: RestCatalogAuth::Bearer(aws_access_token),
    },
    s3_store,
);
```

### 9E. GCP BigLake Metastore

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://biglake.googleapis.com/iceberg/v1beta".into(),
        prefix: Some("projects/my-project/locations/us-central1/catalogs/my-catalog".into()),
        warehouse: Some("gs://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::Bearer(gcp_access_token),
    },
    gcs_store,
);
```

### 9F. Azure Blob + Apache Polaris (production Azure)

```rust
use ailake_store::{azure_store, AzureConfig, AzureCredentials};

let store = Arc::new(azure_store(AzureConfig {
    account_name: "myaccount".into(),
    container: "mycontainer".into(),
    credentials: AzureCredentials::AccessKey(std::env::var("AZURE_STORAGE_KEY")?),
}, "warehouse/")?)

let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://my-polaris.azuredatabricks.net/polaris/api/catalog".into(),
        prefix: Some("my_catalog".into()),
        warehouse: Some("abfss://mycontainer@myaccount.dfs.core.windows.net/warehouse".into()),
        auth: RestCatalogAuth::OAuth2 {
            token_endpoint: "https://login.microsoftonline.com/TENANT/oauth2/v2.0/token".into(),
            client_id: "CLIENT_ID".into(),
            client_secret: "CLIENT_SECRET".into(),
            scope: Some("api://POLARIS_APP_ID/.default".into()),
        },
    },
    store,
);
```

---

## 10. Testing Databricks Unity Catalog

The `databricks_azure` / `databricks_aws` / `databricks_gcp` helpers build the correct `RestCatalogConfig` for each cloud. They require a real Databricks workspace — there is no local emulator.

### 10A. Azure (service principal)

```rust
use ailake_catalog::{databricks_azure, DatabricksAuth, RestCatalog};
use ailake_store::{azure_store, AzureConfig, AzureCredentials};
use std::sync::Arc;

let store = Arc::new(azure_store(AzureConfig {
    account_name: "myaccount".into(),
    container: "mycontainer".into(),
    credentials: AzureCredentials::ClientSecret {
        tenant_id: std::env::var("AZURE_TENANT_ID")?,
        client_id: std::env::var("AZURE_CLIENT_ID")?,
        client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
    },
}, "warehouse/")?)

let catalog = RestCatalog::new(
    databricks_azure(
        "myworkspace.azuredatabricks.net",
        "my_unity_catalog",
        "abfss://mycontainer@myaccount.dfs.core.windows.net/warehouse",
        DatabricksAuth::AzureServicePrincipal {
            tenant_id: std::env::var("AZURE_TENANT_ID")?,
            client_id: std::env::var("AZURE_CLIENT_ID")?,
            client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
        },
    ),
    store,
);
```

For dev/CI with PAT:

```rust
DatabricksAuth::Pat(std::env::var("DATABRICKS_TOKEN")?)
```

Token endpoint used: `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`
Scope: `2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default` (Databricks resource in Azure AD)

### 10B. AWS (M2M OAuth2)

```rust
use ailake_catalog::{databricks_aws, DatabricksAuth};
use ailake_store::{s3_store, S3Config, S3Credentials};

let store = Arc::new(s3_store(S3Config {
    bucket: "my-bucket".into(),
    region: "us-east-1".into(),
    endpoint: None,
    allow_http: false,
    credentials: S3Credentials::Default,
}, "warehouse/")?)

let catalog = RestCatalog::new(
    databricks_aws(
        "myworkspace.cloud.databricks.com",
        "my_unity_catalog",
        "s3://my-bucket/warehouse",
        DatabricksAuth::AwsOAuth2 {
            client_id: std::env::var("DATABRICKS_CLIENT_ID")?,
            client_secret: std::env::var("DATABRICKS_CLIENT_SECRET")?,
        },
    ),
    store,
);
```

Token endpoint used: `https://myworkspace.cloud.databricks.com/oidc/v1/token`
Scope: `all-apis`

### 10C. GCP (Bearer token)

```bash
# Get token via gcloud
export GCP_TOKEN=$(gcloud auth print-access-token)
```

```rust
use ailake_catalog::{databricks_gcp, DatabricksAuth};
use ailake_store::{gcs_store, GcsConfig, GcsCredentials};

let store = Arc::new(gcs_store(GcsConfig {
    bucket: "my-bucket".into(),
    credentials: GcsCredentials::ApplicationDefault,
}, "warehouse/")?)

let catalog = RestCatalog::new(
    databricks_gcp(
        "myworkspace.gcp.databricks.com",
        "my_unity_catalog",
        "gs://my-bucket/warehouse",
        DatabricksAuth::GcpBearer(std::env::var("GCP_TOKEN")?),
    ),
    store,
);
```

### 10D. Unity Catalog hierarchy

Unity Catalog uses 3 levels: `catalog.schema.table`.

```rust
// Table: my_unity_catalog.prod_schema.embeddings
let table = TableIdent::new("prod_schema", "embeddings");
// catalog = prefix from RestCatalogConfig (defined in databricks_*)
// schema  = TableIdent.namespace
// table   = TableIdent.name
```

Resulting URL:
```
GET https://myworkspace.azuredatabricks.net/api/2.1/unity-catalog/iceberg
    /v1/my_unity_catalog/namespaces/prod_schema/tables/embeddings
```

### 10E. Multi-cloud search flow (same code for all backends)

```rust
use ailake_query::{search, SearchConfig};
use ailake_catalog::{TableIdent, CatalogProvider};
use std::sync::Arc;

// catalog can be HadoopCatalog, RestCatalog, or any backend
let catalog: Arc<dyn CatalogProvider> = Arc::new(/* any backend */);

let table = TableIdent::new("prod_schema", "embeddings");
let query = vec![0.1_f32; 1536];

let results = search(
    &table, &query,
    SearchConfig { top_k: 10, ef_search: 50, pruning_threshold: 0.8 },
    "embedding", 1536, catalog, store,
).await?;
```

Geometric pruning works identically for all backends — centroid and radius are in the manifest, not on the catalog server.

---

## 11. NessieCatalog — branching operations

`NessieCatalog` wraps `RestCatalog` for all `CatalogProvider` operations and adds the Nessie v2 branching API.

### 11A. Unit tests (no external server)

```bash
cargo test -p ailake-catalog --features catalog-nessie
```

Covers URL construction (`trees_url`, `ref_url`, `merge_url`) and JSON deserialization of the Nessie API.

### 11B. Integration tests (requires Nessie server)

```bash
docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest

cargo test -p tests --test rest_nessie -- --ignored
```

### 11C. Configuration and usage

```rust
use ailake_catalog::{NessieCatalog, NessieCatalogConfig, RestCatalogAuth};
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = NessieCatalog::new(
    NessieCatalogConfig {
        uri: "http://localhost:19120/api".into(),
        default_branch: "main".into(),
        warehouse: Some("/tmp/warehouse".into()),
        auth: RestCatalogAuth::None,
    },
    store,
);

// CatalogProvider → delegates to inner RestCatalog (branch "main")
catalog.create_table(&table, &props).await?;

// Branching operations — Nessie-specific
let branches = catalog.list_branches().await?;
catalog.create_branch("feature-rag-v2", "main").await?;

// work on feature branch...

catalog.merge_branch("feature-rag-v2", "main").await?;
catalog.delete_branch("feature-rag-v2").await?;
```

Auth via PAT:
```rust
auth: RestCatalogAuth::Bearer("my-nessie-token".into())
```

Auth via OAuth2 (Nessie with OIDC):
```rust
auth: RestCatalogAuth::OAuth2 {
    token_endpoint: "https://my-oidc/token".into(),
    client_id: "client-id".into(),
    client_secret: "secret".into(),
    scope: None,
}
```

---

## 12. JdbcCatalog — PostgreSQL / MySQL

Stores the `metadata_location` pointer in a relational database. Ideal for self-hosted deployments without AWS Glue.

### 12A. Unit tests + SQLite e2e (no external DB)

```bash
cargo test -p ailake-catalog --features catalog-jdbc
```

Includes a complete end-to-end test with in-process SQLite (`catalog-jdbc` feature enables the SQLite driver via sqlx).

### 12B. PostgreSQL via Docker

```bash
docker run --name pg-ailake -e POSTGRES_PASSWORD=test -p 5432:5432 -d postgres:16
```

```rust
use ailake_catalog::JdbcCatalog;
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = JdbcCatalog::connect(
    "postgres://postgres:test@localhost:5432/postgres",
    "prod-catalog",      // catalog name (partitions iceberg_tables)
    "/tmp/warehouse",    // warehouse root
    store,
).await?;

// Schema created automatically (CREATE TABLE IF NOT EXISTS iceberg_tables)
catalog.create_table(&table, &props).await?;
let snap_id = catalog.commit_snapshot(&table, snapshot).await?;
let files = catalog.list_files(&table, Some(snap_id)).await?;
```

### 12C. MySQL via Docker

```bash
docker run --name mysql-ailake \
  -e MYSQL_ROOT_PASSWORD=test -e MYSQL_DATABASE=ailake \
  -p 3306:3306 -d mysql:8
```

```rust
let catalog = JdbcCatalog::connect(
    "mysql://root:test@localhost:3306/ailake",
    "prod-catalog",
    "s3://my-bucket/warehouse",
    store,
).await?;
```

### 12D. SQLite local (dev / tests)

```rust
let catalog = JdbcCatalog::connect(
    "sqlite:///tmp/catalog.db?mode=rwc",
    "dev-catalog",
    "/tmp/warehouse",
    store,
).await?;
```

Note: `sqlite::memory:` does not work with pools (each connection has a separate DB). Use a file.

### 12E. Schema created automatically

```sql
CREATE TABLE IF NOT EXISTS iceberg_tables (
    catalog_name      VARCHAR(255) NOT NULL,
    table_namespace   VARCHAR(255) NOT NULL,
    table_name        VARCHAR(255) NOT NULL,
    metadata_location VARCHAR(1000) NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name)
);
```

Each `commit_snapshot` writes a new `{uuid}.metadata.json` to the Store and `UPDATE`s the pointer in the database. Assumption: single-writer.

---

## 13. GlueCatalog — AWS Glue Data Catalog

Stores `metadata_location` in Glue. Tables become visible in Athena, EMR, Glue ETL, and Redshift Spectrum.

### 13A. Unit tests (no AWS)

```bash
cargo test -p ailake-catalog --features catalog-glue
```

Covers Glue parameter encoding and path format.

### 13B. Configuration

```rust
use ailake_catalog::{GlueCatalog, GlueCatalogConfig};
use ailake_store::{s3_store, S3Config, S3Credentials};
use std::sync::Arc;

// Automatic chain: env vars → ~/.aws → IMDSv2 → WebIdentity (IRSA)
let store = Arc::new(s3_store(S3Config {
    bucket: "my-bucket".into(),
    region: "us-east-1".into(),
    endpoint: None,
    allow_http: false,
    credentials: S3Credentials::Default,
}, "warehouse/")?);

let catalog = GlueCatalog::from_env(
    GlueCatalogConfig {
        database: "my_glue_database".into(),
        warehouse: "s3://my-bucket/warehouse".into(),
        region: Some("us-east-1".into()),
    },
    store,
).await;

catalog.create_table(&table, &props).await?;
```

Explicit client (when you already have an `aws_sdk_glue::Client`):

```rust
use aws_config::BehaviorVersion;
use aws_sdk_glue::config::Region;

let sdk_config = aws_config::defaults(BehaviorVersion::latest())
    .region(Region::new("us-east-1"))
    .load()
    .await;
let client = aws_sdk_glue::Client::new(&sdk_config);
let catalog = GlueCatalog::from_client(client, config, store);
```

### 13C. Parameters created in Glue

```
table_type        = "ICEBERG"
metadata_location = "s3://bucket/warehouse/ns/table/metadata/{uuid}.metadata.json"
```

Compatible with `SHOW TBLPROPERTIES` in Athena and with the Iceberg connector in AWS Glue ETL.

### 13D. Test with Localstack (optional)

```bash
pip install localstack awscli-local
localstack start -d

# create database in local Glue
awslocal glue create-database --database-input '{"Name": "test_db"}'

# test
AWS_ENDPOINT_URL=http://localhost:4566 \
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
  cargo test -p tests --test glue_localstack -- --ignored
```

---

## 14. Clippy and formatting

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Both should finish without errors or warnings.

---

## 15. Trino plugin — VectorScanConnector

The `trino-plugin` exposes AI-Lake tables as a native Trino connector. Requires the native library `libailake_jni.so` (built by Cargo) and the plugin fat-jar (built by Gradle).

### 15A. Additional prerequisites

| Tool | Version | Installation |
|---|---|---|
| JDK | 17+ | `sudo apt install openjdk-17-jdk` |
| Gradle | 8+ | `sdk install gradle` (or use wrapper) |
| Trino server | 430+ | [trino.io/download](https://trino.io/download.html) |

### 15B. Step 1 — Compile the native library

```bash
# From the project root
cargo build --release -p ailake-jni

# Linux:
ls -lh target/release/libailake_jni.so

# macOS:
ls -lh target/release/libailake_jni.dylib
```

### 15C. Step 2 — Compile the plugin fat-jar

```bash
cd trino-plugin

# Create Gradle wrapper (first time only)
gradle wrapper

# Compile
./gradlew shadowJar

# Output:
ls build/libs/trino-plugin-0.1.0-plugin.jar
```

### 15D. Step 3 — Install in Trino

```bash
# Trino installation directory (adjust for your environment)
TRINO_HOME=/opt/trino

# Create plugin directory and copy jar
mkdir -p $TRINO_HOME/plugin/ailake
cp build/libs/trino-plugin-0.1.0-plugin.jar $TRINO_HOME/plugin/ailake/

# Place native library in Trino's library path
# Option A: copy to Trino's lib/
cp ../target/release/libailake_jni.so $TRINO_HOME/lib/

# Option B: define in Trino's jvm.config
echo "-Djava.library.path=/path/to/target/release" >> $TRINO_HOME/etc/jvm.config
```

### 15E. Step 4 — Configure the catalog

Create the file `$TRINO_HOME/etc/catalog/ailake.properties`:

```properties
# AI-Lake connector
connector.name=ailake

# AI-Lake table URI (local filesystem or s3://)
ailake.table-uri=/tmp/ailake-demo

# Vector column and dimension (must match the table)
ailake.vector-column=embedding
ailake.vector-dim=64
```

To use with the table generated by the demo (section 4):

```bash
# Generate demo table first
cargo run --example demo -p ailake-query 2>&1 | grep "Workspace:"
# Workspace: /tmp/ailakeXXXXXX

# Use the path printed above in properties:
# ailake.table-uri=/tmp/ailakeXXXXXX/warehouse/default/demo_table
```

### 15F. Step 5 — Start Trino and run the first search

```bash
# Start the server
$TRINO_HOME/bin/launcher start

# Connect via CLI
$TRINO_HOME/bin/trino
```

At the Trino prompt:

```sql
-- 1. Verify the connector is active
SHOW SCHEMAS FROM ailake;
-- default

SHOW TABLES FROM ailake.default;
-- search

DESCRIBE ailake.default.search;
-- row_id    | bigint  | ...
-- distance  | double  | ...
-- file_path | varchar | ...

-- 2. Set the query vector (comma-separated values, 64 dims for demo)
SET SESSION ailake.query_vector = '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4';

SET SESSION ailake.top_k = 5;

-- 3. Execute vector search
SELECT row_id, distance, file_path
FROM ailake.default.search
ORDER BY distance;

-- Expected result:
--  row_id | distance  | file_path
-- --------+-----------+------------------------------
--       0 | 0.000000  | data/part-00000.parquet
--      12 | 0.031456  | data/part-00000.parquet
--     ...

-- 4. Combine with Iceberg (JOIN with tabular data via standard Iceberg connector)
-- The AI-Lake table returns row_ids; JOIN with the Iceberg table to get text
SELECT s.row_id, s.distance, i.chunk_text
FROM ailake.default.search s
JOIN iceberg.default.demo_table i ON s.row_id = i.id
ORDER BY s.distance
LIMIT 5;
```

### 15G. Trino plugin tests

```bash
cd trino-plugin
./gradlew test

# Expected output:
# VectorScanMetadataTest: 7 tests passed
# VectorScanConnectorTest: 7 tests passed
# VectorScanSplitManagerTest: 5 tests passed
# VectorScanRecordSetTest: 9 tests passed
# AilakeNativeTest: 5 tests passed
```

Tests run without a Trino server or native library — graceful degradation guaranteed.

---

## 16. Spark plugin — VectorScanStrategy

The `spark-plugin` registers a custom `SparkStrategy` in the Spark 3.5 Catalyst planner. It converts `VectorSearchPlan` plans into `VectorScanExec` that calls the native library via JNA.

### 16A. Additional prerequisites

Same JDK 17+ and Gradle from section 15A. Local Spark 3.5 (for testing), or a pre-configured Spark cluster.

```bash
# Local Spark (optional — for testing)
curl -LO https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar xf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=$(pwd)/spark-3.5.0-bin-hadoop3
```

### 16B. Step 1 — Native library (same as section 15B)

```bash
cargo build --release -p ailake-jni
# target/release/libailake_jni.so  (Linux)
```

### 16C. Step 2 — Compile the plugin fat-jar

```bash
cd spark-plugin
gradle wrapper   # first time only
./gradlew shadowJar

ls build/libs/spark-plugin-0.1.0-plugin.jar
```

### 16D. Step 3 — Generate demo table

Use the table generated by the demo in section 4:

```bash
# Generate table and note the path
cargo run --example demo -p ailake-query 2>&1 | grep "Workspace:"
# Workspace: /tmp/ailakeXXXXXX

export AILAKE_TABLE=/tmp/ailakeXXXXXX/warehouse/default/demo_table
```

### 16E. Step 4 — spark-shell (interactive Scala)

```bash
$SPARK_HOME/bin/spark-shell \
  --jars $(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$(pwd)/target/release" \
  --conf spark.ui.enabled=false
```

At the `scala>` prompt:

```scala
import io.ailake.spark.implicits._

// Table generated by demo (dim=64, 1000 rows)
val tableUri = sys.env("AILAKE_TABLE")

// Query vector — same as the first embedding in the demo
val query: Array[Float] = Array.fill(64)(0.5f)

// Vector search — returns DataFrame with (row_id, distance, file_path)
val results = spark.ailakeSearch(
  tableUri    = tableUri,
  queryVector = query,
  topK        = 10,
)

results.show(10, truncate = false)
// +------+-----------+----------------------------+
// |row_id|distance   |file_path                   |
// +------+-----------+----------------------------+
// |0     |0.0        |data/part-00000.parquet     |
// |12    |0.031456   |data/part-00000.parquet     |
// |87    |0.044123   |data/part-00001.parquet     |
// ...

// Sort by distance and show top-5
results.orderBy("distance").limit(5).show()

// Returned schema
results.printSchema()
// root
//  |-- row_id: long (nullable = false)
//  |-- distance: double (nullable = false)
//  |-- file_path: string (nullable = false)
```

### 16F. Step 4 alternative — PySpark

```bash
$SPARK_HOME/bin/pyspark \
  --jars $(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$(pwd)/target/release"
```

```python
# PySpark — call Scala logic via py4j
jvm = spark._jvm

# Instantiate VectorSearchPlan directly via py4j
float_array = jvm.Array(jvm.Float.TYPE, 64)
for i in range(64):
    float_array[i] = 0.5

table_uri = "/tmp/ailakeXXXXXX/warehouse/default/demo_table"

# Call AilakeNative directly (bypassing Spark planner)
native = jvm.io.ailake.spark.AilakeNative
results_java = native.search(table_uri, float_array, 10)

# Convert to Python
results = [
    {"row_id": r.rowId(), "distance": r.distance(), "file_path": r.filePath()}
    for r in results_java
]
for r in results:
    print(f"row_id={r['row_id']}  distance={r['distance']:.6f}  file={r['file_path']}")

# Or use ailake-py (recommended for Python):
import ailake
query = [0.5] * 64
results = ailake.search(path=table_uri, query=query, top_k=10)
```

### 16G. Step 5 — submit Spark job (cluster)

```scala
// MyVectorSearchJob.scala
import io.ailake.spark.implicits._
import org.apache.spark.sql.SparkSession

object MyVectorSearchJob {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ailake-vector-search")
      .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
      .getOrCreate()

    val tableUri  = args(0)  // s3://my-lake/docs/
    val topK      = args(1).toInt
    val queryJson = args(2)  // "[0.1, -0.2, 0.3, ...]"

    val query: Array[Float] =
      ujson.read(queryJson).arr.map(_.num.toFloat).toArray

    spark.ailakeSearch(tableUri, query, topK)
      .write.parquet(args(3))  // output path
  }
}
```

```bash
spark-submit \
  --jars spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  my-vector-search-job.jar \
  "s3://my-lake/docs/" 100 "[0.021, -0.043, ...]" "s3://results/out/"
```

**Important**: `libailake_jni.so` must be available on all executors. Add it to spark-submit's `--files` or install via a bootstrap script.

### 16H. Spark plugin tests

```bash
cd spark-plugin
./gradlew test

# Expected output:
# VectorSearchPlanTest: 8 tests passed
# VectorScanStrategyTest: 6 tests passed
# AilakeNativeTest: 4 tests passed
# AilakeSparkExtensionsTest: 5 tests passed  (requires local SparkSession)
```

Integration tests (`AilakeSparkExtensionsTest`) start a local SparkSession — takes ~15s on first run.

---

## 17. Flink plugin — AilakeVectorConnector

`ailake-flink` is a Flink Table API connector (Kotlin/Gradle) that uses the native library `libailake_jni.so` via JNA for vector read and write operations on AI-Lake tables.

### 17A. Additional prerequisites

| Tool | Version | Installation |
|---|---|---|
| JDK | 17+ | `sudo apt install openjdk-17-jdk` |
| Gradle | 8+ | `sdk install gradle` (or use wrapper) |
| Apache Flink | 1.18+ | [flink.apache.org](https://flink.apache.org/downloads/) |

### 17B. Step 1 — Compile the native library

```bash
# From the project root (same lib as Trino/Spark plugin)
cargo build --release -p ailake-jni

# Linux:
ls -lh target/release/libailake_jni.so
```

### 17C. Step 2 — Compile the fat-jar

```bash
cd ailake-flink
gradle wrapper   # first time only
./gradlew shadowJar

ls build/libs/ailake-flink-0.1.0-plugin.jar
```

The shadow jar (`-plugin`) includes JNA and Jackson. Flink dependencies are outside (`compileOnly`).

### 17D. Register the connector in Flink (SQL Client or DataStream)

```sql
-- Flink SQL Client
ADD JAR '/path/to/ailake-flink-0.1.0-plugin.jar';

-- Source + sink table
CREATE TABLE docs (
  id        BIGINT,
  text      STRING,
  embedding BYTES,
  _distance FLOAT   -- filled by vector search, ignored on writes
) WITH (
  'connector'        = 'ailake',
  'warehouse'        = 's3://my-lake/',
  'namespace'        = 'default',
  'table-name'       = 'docs',
  'vector.column'    = 'embedding',
  'vector.dim'       = '1536',
  'vector.metric'    = 'cosine',
  'vector.precision' = 'f16',
  'search.top-k'     = '10',
  'search.ef'        = '50'
);

SELECT id, text, _distance FROM docs;
```

The query vector is passed as a job parameter (`ailake.query.vector` — floats separated by commas):

```bash
flink run \
  -D "pipeline.global-job-parameters=ailake.query.vector=0.021,-0.043,0.118,..." \
  my-pipeline.jar
```

For streaming ingestion (sink):

```sql
INSERT INTO docs
SELECT id, chunk_text, embedding FROM kafka_source;
```

The sink (`AilakeSinkFunction`) accumulates 10,000 rows and calls `AilakeNativeLoader.writeBatch()` on flush.

### 17E. Flink plugin tests

```bash
cd ailake-flink
./gradlew test

# Expected output:
# AilakeVectorConnectorFactoryTest: tests passed
```

Tests run without a Flink server or native library — graceful degradation via mock JNA.

---

## 18. Multi-vector columns (`write_batch_multi_vec`)

Stores multiple embeddings per row as `List<FixedSizeBinary>` — useful when a document has multiple chunks and you want to avoid row explosion.

### 18A. Write with multiple vectors per row

```rust
use ailake_parquet::writer::write_batch_multi_vec;

// 3 documents, each with 2 embeddings of dim=64
let texts = vec!["doc A".to_string(), "doc B".to_string(), "doc C".to_string()];
let multi_embeddings: Vec<Vec<Vec<f32>>> = vec![
    vec![vec![0.1_f32; 64], vec![0.2_f32; 64]],  // doc A: 2 vecs
    vec![vec![0.3_f32; 64]],                       // doc B: 1 vec
    vec![vec![0.4_f32; 64], vec![0.5_f32; 64]],   // doc C: 2 vecs
];

let batch = write_batch_multi_vec(&texts, &multi_embeddings, "embedding", 64)?;
```

### 18B. Read back

```rust
use ailake_parquet::reader::read_all_multi_vec;

let (texts, multi_vecs) = read_all_multi_vec(&parquet_bytes, "embedding", 64)?;
// multi_vecs[i] = Vec<Vec<f32>> for document i
```

### 18C. What standard Parquet readers see

Readers without the AI-Lake plugin (Spark, Trino, DuckDB) read the column as `List<BINARY>` — opaque bytes, no error. Vector semantics are only activated by the SDK.

---

## 19. MemTableWriter — buffer for streaming ingestion

`MemTableWriter` accumulates micro-batches in RAM and flushes to a single Parquet shard when size/row/time thresholds are reached. Reduces HNSW build frequency in Flink/Spark Streaming pipelines.

### 19A. Basic usage

```rust
use ailake_query::mem_table::{MemTableWriter, MemTableConfig};
use std::time::Duration;

let config = MemTableConfig {
    flush_size_bytes: 128 * 1024 * 1024,  // flush after 128 MiB
    flush_max_rows:   200_000,             // flush after 200k rows
    flush_interval:   Duration::from_secs(60), // flush after 60s idle
};

let mut mt = MemTableWriter::new(catalog, store, policy, table, config);

// Streaming ingestion loop
loop {
    let (batch, embeddings) = receive_micro_batch().await;
    mt.insert(&batch, &embeddings).await?;

    // Automatic flush if threshold reached
    if let Some(snap_id) = mt.flush_if_due().await? {
        println!("Flushed snapshot {snap_id}");
    }
}

// Final flush when shutting down the job
let snap_id = mt.flush().await?;
```

### 19B. Default thresholds

| Threshold | Default | Trigger |
|---|---|---|
| `flush_size_bytes` | 64 MiB | Accumulated embedding bytes |
| `flush_max_rows` | 100,000 | Accumulated rows |
| `flush_interval` | 30s | Time since last flush |

`flush_if_due()` checks all three. `flush()` forces regardless of thresholds.

### 19C. Inspect buffer state

```rust
println!("Buffered: {} rows, {} bytes", mt.buffered_rows(), mt.buffered_bytes());
if mt.is_full() { mt.flush().await?; }
```

---

## Crate structure

```
ailake-core/      base types: VectorMetric, VectorPrecision, RowId, AilakeError
ailake-parquet/   Parquet read/write with VECTOR column; write_batch_multi_vec / read_all_multi_vec
ailake-vec/       F32→F16 quantization, PQ (PQCodebook), BlockCompressor, SIMD distances (AVX2/NEON), centroids
ailake-index/     HNSW + IVF-PQ (AnyIndex enum); hardware detection (HardwareBackend: NvidiaCuda/AmdRocm/CpuSimd);
                  NVIDIA via cuBLAS libloading (runtime, no build flag); AMD ROCm via hipBLAS libloading (runtime, no build flag);
                  kmeans_dispatch: NVIDIA → ROCm → CPU; bincode, MmapLoader (memmap2)
ailake-file/      unified file: AILK supports IndexType::Hnsw and IndexType::IvfPq
ailake-catalog/   Iceberg catalog: HadoopCatalog, RestCatalog, NessieCatalog, JdbcCatalog, GlueCatalog
ailake-store/     storage abstraction: LocalStore + ObjectStoreBackend (S3/GCS/Azure via object_store)
ailake-query/     TableWriter (write_batch, write_batch_ivf_pq, write_batch_multi), MemTableWriter,
                  search() with geometric pruning, ContextAssembler, CompactionExecutor
ailake-benchmarks  SIFT-1M benchmark (separate repo): https://github.com/ThiagoLange/ailake-benchmarks
ailake-py/        PyO3 bindings (outside workspace — compile with maturin)
ailake-jni/       C-ABI cdylib via JNA for Spark, Trino, and Flink
spark-plugin/     Spark 3.5 Catalyst strategy (Kotlin/Gradle)
trino-plugin/     Trino connector (Java/Gradle)
ailake-flink/     Flink Table API connector (Kotlin/Gradle)
tests/            integration and compatibility tests
```

---

## Troubleshooting

**`error: linker 'cc' not found`**
```bash
# Ubuntu/Debian
sudo apt install build-essential
```

**`import pyarrow` fails**
```bash
pip install pyarrow
# or with conda:
conda install pyarrow
```

**`import ailake` fails after `maturin develop`**
```bash
# Verify you are in the ailake-py directory and the correct venv
cd ailake-py
maturin develop --release
python3 -c "import ailake"
```

**`cargo test` fails on `pyarrow_ignores_ailake_footer`**
This test requires `python3` + `pyarrow`. Run with `--ignored`:
```bash
cargo test -p tests --test parquet_trailing_bytes -- --ignored
```

**Benchmark fails with `E0601`**
Make sure you are on the `main` or `develop` branch (empty benches were fixed in `e382e83`).

**`pruning_threshold` removes all results**
Threshold too low cuts legitimate files. Use `f32::INFINITY` to disable pruning and debug:
```rust
SearchConfig { top_k: 10, ef_search: 50, pruning_threshold: f32::INFINITY }
```

**Trino plugin: `UnsatisfiedLinkError: libailake_jni.so`**
The native library is not in Trino's `java.library.path`.
```bash
# Check where Trino looks
grep "java.library.path" $TRINO_HOME/etc/jvm.config

# Add the path
echo "-Djava.library.path=/path/to/target/release" >> $TRINO_HOME/etc/jvm.config

# Restart Trino
$TRINO_HOME/bin/launcher restart
```

**Trino plugin: `ailake.table-uri is required`**
The property was not defined in the catalog file.
```bash
cat $TRINO_HOME/etc/catalog/ailake.properties
# Verify it contains: ailake.table-uri=...
```

**Trino plugin: `SELECT` returns 0 rows**
Session property `query_vector` is empty. Verify:
```sql
SHOW SESSION LIKE 'ailake%';
SET SESSION ailake.query_vector = '0.1,0.2,0.3,...';
```

**Spark plugin: `ClassNotFoundException: io.ailake.spark.AilakeSparkExtensions`**
The plugin jar was not passed to Spark.
```bash
# Verify --jars includes the plugin
spark-shell --jars /path/to/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions
```

**Spark plugin: `ailakeSearch` not found**
The implicits import is missing.
```scala
// Add at the beginning of the script
import io.ailake.spark.implicits._
```

**Spark plugin: VectorScanExec returns empty DataFrame**
Expected behavior in an environment without `libailake_jni.so` — graceful degradation.
To enable real search, ensure the lib is in `java.library.path`:
```bash
spark-shell \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/path/to/target/release" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/path/to/target/release"
```

**`SearchConfig` compilation fails with `missing field rerank_factor`**
Add the missing field (introduced in Phase 4):
```rust
SearchConfig {
    top_k: 10,
    ef_search: 50,
    pruning_threshold: 0.8,
    rerank_factor: None,  // Some(3) to enable reranking
}
```

**NVIDIA GPU available but search uses CPU (`try_nvidia_search_batch` returns `None`)**
Runtime libraries missing or not in `LD_LIBRARY_PATH`. CUDA Toolkit is not required — only runtime libs:
```bash
# Ubuntu — install runtime only (no nvcc/headers)
sudo apt install libcuda1 libcublas-12-x

# Or via CUDA keyring (canonical)
sudo apt install cuda-libraries-12-x

# Verify presence
ldconfig -p | grep -E "libcuda|libcublas"
ls /usr/local/cuda/lib64/libcudart.so*

# Export if not in default path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**GPU build compiles but search uses CPU (log does not show "NVIDIA" or "AMD")**
`detect_backend()` returned `CpuSimd`. Verify:
```bash
nvidia-smi                        # NVIDIA GPU visible?
rocm-smi                          # AMD GPU visible?
ls /dev/nvidia*                   # NVIDIA devices present?
ls /dev/kfd                       # AMD KFD device present?
echo $CUDA_VISIBLE_DEVICES        # Must not be "NoDevFiles"
```
For NVIDIA: `libcuda.so.1` must exist (`ldconfig -p | grep libcuda`).
For AMD: `libamdhip64.so` and `libhipblas.so` must exist (`ldconfig -p | grep -E "hip|hsa"`).

**AMD ROCm detected but search uses CPU**
`detect_rocm()` returns `true` (HIP runtime ok) but `try_rocm_search_batch` returns `None`.
Most common cause: `libhipblas.so` not installed (ROCm Math Libraries are optional).
```bash
# Verify hipBLAS presence
ldconfig -p | grep hipblas

# Install hipBLAS (Ubuntu/ROCm PPA)
sudo apt install hipblas
# or
sudo apt install rocm-libs
```

**`ailake-auto` shows `Backend: NVIDIA CUDA` on an AMD machine**
ROCm with CUDA compatibility layer installed — `libcuda.so.1` exists (provided by `hip-runtime-amd`). In this case, if `libamdhip64.so` also exists, the correct `AmdRocm` backend is already chosen (AMD is checked first). If NVIDIA still appears, verify:
```bash
ldconfig -p | grep -E "libamdhip|libcuda"
# libamdhip64.so must appear for ROCm to be detected as AmdRocm
```

**Flink plugin: `ClassNotFoundException: io.ailake.flink.AilakeVectorConnectorFactory`**
The plugin jar was not added to Flink.
```bash
# SQL Client — add before CREATE TABLE
ADD JAR '/path/to/ailake-flink-0.1.0-plugin.jar';

# or via flink-conf.yaml
classloader.parent-first-patterns.additional: io.ailake
```

**Flink plugin: connector registered but sink does not persist**
`libailake_jni.so` is not in the TaskManager's `java.library.path`:
```bash
# Add in flink-conf.yaml
env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib
```

**`IvfPqConfig` — `pq_m` must divide `dim`**
Build fails with `"pq_m X does not divide dim Y"`. Use `IvfPqConfig::for_dim(dim)` to derive valid values automatically:
```rust
let cfg = IvfPqConfig::for_dim(1536);  // pq_m=96, nlist=256, nprobe=8
```

**`MemTableWriter::flush()` returns error after empty `insert`**
Calling `flush()` without any prior `insert` returns `Err(EmptyBatch)`. Check `buffered_rows() > 0` first:
```rust
if mt.buffered_rows() > 0 {
    mt.flush().await?;
}
```
