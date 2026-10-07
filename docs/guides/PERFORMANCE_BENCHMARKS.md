# Performance, load and regression checks

AI-Lake has a lightweight in-repository performance harness and a separate
workflow for load tests. The workflow is intentionally split from the regular
correctness CI: the CPU checks run on pull requests, while GPU and emulator
matrix jobs run on pushes, nightly builds or manual dispatch.

## CPU microbenchmark

The dependency-free vector-distance benchmark reports median nanoseconds per
operation for cosine, Euclidean and dot-product kernels at 128, 768 and 1536
dimensions. Each case warms up and records seven timed samples, including the
sample range:

```bash
cargo bench -p ailake-vec --bench distance
cargo bench -p ailake-vec --bench distance -- --json > perf/cpu-distance.json
python3 scripts/ci/compare_benchmark.py perf/cpu-distance.json
```

JSON output is suitable for artifacts and external dashboards. Absolute
numbers are not portable across CPUs, SIMD features, compiler versions or
thermal state. To compare against a same-runner baseline:

```bash
python3 scripts/ci/compare_benchmark.py \
  perf/cpu-distance.json \
  --baseline perf/cpu-distance-baseline.json \
  --max-regression 0.25
```

The default tolerance is 25%. A baseline must come from the same runner class
and compiler profile; do not compare a laptop measurement with a GitHub-hosted
runner measurement.

## Recall and search latency

The deterministic integration test writes 10,000 128-dimensional vectors,
runs four warm-up queries, compares 32 HNSW queries with an exact brute-force
oracle and measures p95 search latency. On the current development machine the
debug run took about 103 seconds; expect the duration to vary with hardware and
build cache state:

```bash
AILAKE_MIN_RECALL=0.95 AILAKE_MAX_P95_MS=500 \
  cargo test -p ailake-tests --test performance_regression -- --nocapture
```

The test uses a fixed fixture and seed, emits `PERF_RECALL_JSON=...` and fails
when Recall@10 or p95 exceeds the configured limit. Larger SIFT-1M comparisons
remain in the external [`ailake-benchmarks`](https://github.com/ThiagoLange/ailake-benchmarks)
repository and should be used for release reports, not every pull request.

### Coverage boundaries and next performance work

These checks do not yet provide a full production performance profile:

- The recall/p95 case uses one local table at 10,000 rows and 128 dimensions.
  It does not measure large multi-file/cloud scans, cold-cache behavior, peak
  memory, or concurrent searches across many shards.
- The distance benchmark measures kernels. Pull request runs compare with the
  base branch on the same runner and use a 25% tolerance; push, scheduled and
  manual runs validate the output schema without a baseline comparison.
- A search limits its file fan-out to 32, while HTTP serve accepts up to 64
  requests concurrently. A process-wide semaphore now budgets file searches by
  manifest size in 16 MiB units, up to an estimated 512 MiB in aggregate. The
  budget is fixed and does not account for backend latency or actual decoded
  memory; a single file larger than the budget runs alone.
- Bounded top-K reduces intermediate result memory, but brute-force fallback
  remains O(rows × dimensions); it should remain a fallback for small or
  unindexed shards, with compaction/reindexing monitored operationally.
- Parquet delete filtering reuses footer metadata, but still opens readers for
  surviving row groups. Further batching should follow measurements on realistic
  files rather than assumed gains.
- FTS payloads are capped at 64 MiB and BM25 stats use a size check plus bounded
  range read. These limits are fixed constants; tuning them by workload remains
  a follow-up if real indexes approach those ceilings.

For the HTTP harness, also vary request concurrency, file count, object-store
latency and table size; report p95/p99 and memory alongside QPS. Keep benchmark
baselines tied to the same runner and compiler profile.

## Real HTTP load and concurrent writers

Build a release server, create a test table and run the dependency-free Python
harness:

```bash
cargo build --release -p ailake-cli
target/release/ailake --store /tmp/ailake-perf create default.load --dim 32
target/release/ailake --store /tmp/ailake-perf serve default.load --port 7700
python3 scripts/ci/http_load.py \
  --base-url http://127.0.0.1:7700 \
  --requests 100 --concurrency 16 --write-workers 4 \
  --output perf/http-load.json
```

The harness performs concurrent `/write` requests, then concurrent `/search`
requests, and records QPS, mean, p50, p95 and p99. It fails on non-2xx
responses or when p95 exceeds `--max-p95-ms`.

The multi-process fencing check confirms that a second `ailake serve` instance
cannot acquire the same table lease:

```bash
bash scripts/ci/check_multi_process_lock.sh \
  target/release/ailake /tmp/ailake-perf default.load
```

The Rust `concurrent_writes` integration suite remains the catalog-level
baseline for Hadoop and JDBC writers.

## CI workflow and matrix

`.github/workflows/performance.yml` contains these jobs:

| Job | Trigger | Coverage |
|---|---|---|
| `benchmark-cpu` | PR, push, nightly, manual | Structured CPU microbenchmark and artifact |
| `recall-latency` | PR, push, nightly, manual | Deterministic Recall@10 and p95 gate |
| `http-load-and-writers` | PR, push, nightly, manual | Real server, HTTP load, concurrent writes and process fencing |
| `catalog-matrix` | PR, push, nightly, manual | Local catalog, MinIO/LocalStack and Iceberg REST emulator smoke |
| `gpu-performance-matrix` | push, nightly, manual | CPU fallback plus NVIDIA CUDA and AMD ROCm runners when registered |

Linux GPU jobs require self-hosted runners labelled `gpu-nvidia` and
`gpu-amd`. Until those runners exist, the matrix keeps those entries visible
but they remain queued only for non-PR triggers. The existing `ci-gpu.yml`
continues to provide the dedicated manual GPU correctness workflow.

The emulator matrix is safe for pull requests. Real AWS S3, GCS and Azure
performance runs should be added as scheduled/manual jobs using short-lived
OIDC credentials, never as mandatory PR jobs with long-lived secrets.
