# Security Policy

## Supported versions

Only the latest release receives security fixes.

| Version | Supported |
|---------|-----------|
| 0.0.x (latest) | ✓ |
| < 0.0.5 | ✗ |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Send a report to: **thiago.egon@gmail.com**

Include:
- Description of the vulnerability and its impact
- Steps to reproduce or proof-of-concept code
- Affected versions
- Any suggested fix (optional)

You will receive an acknowledgement within **48 hours** and a status update within **7 days**.

## Disclosure policy

- We follow coordinated disclosure: fixes are prepared before public announcement.
- CVEs are requested by the maintainer once a fix is ready.
- Credit is given to reporters in the release notes unless anonymity is requested.

## Scope

In scope:
- Memory safety issues in Rust unsafe blocks (`ailake-index`, `ailake-file`, `ailake-vec`)
- Path traversal or arbitrary file write via warehouse/table paths
- Credential exposure via `ailake-store` cloud backends
- Malformed AI-Lake or Parquet file causing panic or memory corruption
- JNI/C-ABI input validation bypass (null pointers, oversized buffers, non-UTF-8)
- Auth bypass in REST catalog backend (OAuth2 token leakage, bearer token mishandling)
- Concurrency race conditions across JNI/catalog backends

Out of scope:
- Denial-of-service via intentionally large inputs (no SLA for resource exhaustion)
- Issues in transitive dependencies not triggered by AI-Lake code paths
- Vulnerabilities in benchmark or test tooling ([`ailake-benchmarks`](https://github.com/ThiagoLange/ailake-benchmarks), `tests/`)

---

## Security hardening

### Fuzz testing

| Tool | Scope | Detail |
|------|-------|--------|
| **Proptest** (18 strategies) | `ailake-vec` (6), `ailake-index` (2), `ailake-query` (2), `ailake-catalog` (6), `ailake-jni` (8) | Property-based: NaN/Inf queries, zero-dim vectors, corrupt Avro manifests, arbitrary C-ABI strings, null pointers, extreme lengths, round-trip fuzzing |
| **FailStore** | `ailake-store` | I/O error injection store wrapper — 10 tests covering partial writes, read failures, missing files |
| **Corrupt file tests** | `ailake-file` | 7 tests: AILK magic corruption, HNSW bincode corruption, truncated Parquet, empty file, missing footer, zero query, dimension mismatch |

### CI security checks

| Check | Scope | Detail |
|-------|-------|--------|
| **Miri** | `ailake-core`, `ailake-vec`, `ailake-index` | Nightly Rust; detects undefined behavior (UB) in unsafe code. 30 min timeout. No SIMD/FFI coverage |
| **ASan + LSan** | 7 core crates | Nightly + `-Zbuild-std`. Address and leak sanitizers. No UBSan: rustc's `-Z sanitizer=` never accepted `undefined` as a value — Rust doesn't expose a standalone UBSan pass the way Clang does |
| **cargo-deny** | All crates | License compliance, security advisories, source checks |
| **TruffleHog** | Entire repo | Secret scanning on push to `main`/`develop` and all PRs. `--only-verified` |
| **RUST_BACKTRACE=1** | CI env | Full backtrace on panic for bug repro |

### JNI / C-ABI (`ailake-jni/src/lib.rs`)

All 11 `#[no_mangle]` exports follow the same safety pattern:

```
catch_ffi_panic(|params| {
    if ptr.is_null() { return error_json; }
    let s = CStr::from_ptr(ptr).to_str()?;  // UTF-8 validation
    ... operation ...
})
```

| Guard | Functions | Detail |
|-------|-----------|--------|
| Null pointer check | All 11 exports | Returns `{"ok":false,"error":"..."}` on null input |
| Panic boundary | All 11 exports | `catch_ffi_panic` — `std::panic::catch_unwind` prevents unwind across FFI |
| UTF-8 validation | All string-receiving exports | `CStr::from_ptr().to_str()` — non-UTF-8 returns error JSON |
| `query_len > 65536` | `ailake_vector_search_json`:375 | Rejected before heap allocation |
| `ef_search ≤ 100000` | `ailake_search_json`:521, `ailake_scan_json`:2080 | `.min(100000)` prevents runaway HNSW candidate heap |
| IPC len < 0 | `ailake_write_batch_ipc`:1024 | Negative `i64` rejected before `from_raw_parts` |
| Per-table mutex | `write_batch_json`, `ipc`, `multi_json`, `delete_where`, `evolve_schema`, `compact_json` | `jni_table_lock()` — prevents concurrent HadoopCatalog commits |
| Memory ownership | `ailake_free_string`:1465 | Every `*mut c_char` returned must be freed; `ailake_version` returns `*const c_char` (static, never free) |

### HTTP server (`ailake-cli/src/serve.rs`)

| Guard | Detail |
|-------|--------|
| `DefaultBodyLimit::max(32 MB)` | Axum middleware; rejects oversized bodies before handler |
| Empty query → 400 | `handle_search` — `ApiError("query must not be empty")` before any I/O |
| `top_k.clamp(1, 10_000)` | `handle_search` — prevents unbounded HNSW result sets |
| Startup warning | `eprintln!("WARNING: ailake serve has no authentication …")` on start |

> **Note:** `ailake serve` is for internal/trusted networks only. No rate limiting, no auth. Do not expose publicly without authenticating proxy (API gateway, mTLS sidecar).

### CLI input validation (`ailake-cli/src/`)

| Command | Guard | Detail |
|---------|-------|--------|
| `insert` | `--deferred` conflicts with `--batch-id` | Clap `conflicts_with` |
| `insert` | Empty `--vector-cols` rejected | L812 |
| `insert` | Zero dim rejected (`dim == 0`) | L920 |
| `search` | `--query`, `--query-file`, `--text` mutually exclusive | Clap `conflicts_with_all` |
| `search` | Query file size %4 != 0 rejected | L1097 (binary embedding files must be f32-aligned) |
| `compact` | `--deferred` requires catalog support | `catalog.supports_in_place_rewrite()` check |
| Catalog `rest` | `--rest-auth bearer` requires `--rest-token` | L606 |
| Catalog `rest` | `--rest-auth oauth2` requires endpoint + client_id + client_secret | L613 |
| Catalog `ducklake` | Rejects non-local store URLs (s3://, gs://, az://) | L563 |

### File format safety (`ailake-file/src/reader.rs`)

All offsets use `checked_add` to prevent integer overflow:

| Location | Check |
|----------|-------|
| `ailk_offset_from_trailer` | `footer_start < TRAILER_SIZE` → early error (underflow prevention) |
| Header read | `header_end > bytes.len()` → `NotAnAilakeFile` |
| Centroid read | `centroid_end > bytes.len()` → `NotAnAilakeFile` |
| Centroid parse | `centroid_data.len() != expected_len` → `InvalidCentroidLength` |
| HNSW blob read | `hnsw_end > bytes.len()` → `NotAnAilakeFile` |
| FTS blob read | `fts_abs + AILK_FTS_HEADER_SIZE > bytes.len()` → error |

Magic byte verification: AILK header (magic `b"AILK"`), FTS header (magic `b"AFTS"`), Parquet footer (`b"PAR1"`).

### Vector distance kernels (`ailake-vec/src/distance.rs`)

| Guard | Detail |
|-------|--------|
| Dimension mismatch | `debug_assert_eq!(a.len(), b.len())` in `dot_product`, `euclidean_distance`, `cosine_distance` |
| NaN/Inf rejection | Proptest filters NaN/Inf queries; zero-vector L2 normalize returns error |
| SIMD runtime detection | AVX-512 (`avx512f`), AVX2, NEON — checked at runtime via `is_x86_feature_detected!` / `is_aarch64_feature_detected!`. Fallback scalar always available |
| Pre-normalize | `VectorMetric::NormalizedCosine` → L2 unit normalization on write; query auto-normalized. Skips sqrt in hot loop |

> **Known gap**: `debug_assert_eq` is compiled out in release builds. A dimension mismatch between query and stored vectors in release mode causes out-of-bounds reads in the SIMD distance kernels. Mitigation: callers (`scanner.rs`, `TableWriter`) validate dim before calling distance functions. In release, this is the only barrier. See `docs/architecture/THREAT_MODEL.md`.

### Query engine (`ailake-query/src/`)

| Guard | Location | Detail |
|-------|----------|--------|
| Dimension validation | `scanner.rs:218` | Query dim must match stored column dim; returns `ModelMismatch` error |
| Geometric pruning | `scanner.rs:272` | `distance(query, centroid) - radius > threshold` → file skipped before I/O |
| Bloom filter | `scanner.rs:282` | Per-file Bloom filter checked before fetch |
| Predicate pushdown | `scanner.rs:628` | `matching_row_ids()` — empty filter set skips file entirely |
| Concurrent search | `scanner.rs:368` | `try_join_all` over surviving files; no shared mutable state (per-file `FileSearchOutcome`) |
| Foreign file warning | `scanner.rs:681` | Files without AILK footer detected; falls back to flat scan O(N), search still correct but degraded |
| Safe DV failure | `scanner.rs:545` | Deletion vector parse failure → `warn!` + continue (shows deleted rows rather than fails the search) |
| Safe eq-delete failure | `scanner.rs:296` | Equality delete file parse failure → `warn!` + continue |
| BM25 vocab cap | `bm25.rs:28` | `MAX_VOCAB = 50_000` terms; lowest-DF terms pruned after merge |
| BM25 weight clamp | `bm25.rs:221` | `w.clamp(0.0, 1.0)` |
| MemTable flush limits | `mem_table.rs:20` | 64 MiB / 100k rows default |
| Episodic importance clamp | `mem_table.rs:252` | `importance.clamp(0.0, 1.0)` |
| ScoreFn bounds | `scanner.rs:1014` | `row_id < batch.num_rows()` before indexing |
| Fetch rows bounds | `scanner.rs:1585` | `idx >= batch.num_rows()` → warn + skip |
| Write batch idempotent | `writer.rs:339` | Checks current snapshot before I/O; skips on duplicate `batch_id` |
| Deferred index | `writer.rs:243` | Background build via `tokio::spawn`; errors logged, not silently ignored |

### REST catalog (`ailake-catalog/src/rest.rs`)

| Guard | Detail |
|-------|--------|
| Auth modes | `None`, `Bearer(token)`, `OAuth2` |
| Namespace auto-create | `ensure_namespace()` — idempotent `POST /v1/{prefix}/namespaces`; 409 treated as success |
| Schema patch dedup | `add_schema`/`set_current_schema` skipped when schema unchanged |
| Partition spec dedup | `add_partition_spec` skipped when spec unchanged |
| Snapshot null handling | `-1` sentinel treated as `null` (no snapshot) before commit requirement |
| Token storage | Bearer token and OAuth2 client_secret held in-memory `String` (plaintext) |

### Credential environment variables

| Variable | Source | Risk |
|----------|--------|------|
| `AILAKE_REST_TOKEN` | User / CI | Bearer token in env — may leak via `/proc` or core dumps |
| `AILAKE_REST_OAUTH_CLIENT_SECRET` | User / CI | OAuth2 secret in env |
| `AILAKE_REST_OAUTH_CLIENT_ID` | User / CI | OAuth2 client ID (not secret, but identifying) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | SDK standard | Cloud credentials via `object_store` |
| `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_CLIENT_*` | SDK standard | Azure credentials via `object_store` |
| `GOOGLE_APPLICATION_CREDENTIALS` | SDK standard | GCP credentials via `object_store` |
| `CARGO_REGISTRY_TOKEN`, `PYPI_API_TOKEN` | CI only | CI release tokens; stored in GitHub secrets |

### Airflow provider (`airflow-providers-ailake/`)

| Guard | Detail |
|-------|--------|
| stderr/stdout truncated to 4096 chars | Prevents cloud SDK error messages (which may include credential-adjacent context) from flooding Airflow task logs |

---

## Known gaps

| # | Severity | Gap | Location | Impact |
|---|----------|-----|----------|--------|
| 1 | **Critical** | `debug_assert_eq` for dim check — ineffective in release | `ailake-vec/src/distance.rs:8,33,58` | OOB read on dim mismatch in release builds; mitigated by caller-side validation |
| 2 | **High** | No path traversal guard in `LocalStore::full_path` | `ailake-store/src/local.rs:30` | `../../etc/passwd` resolves outside root; LocalStore only for trusted paths |
| 3 | **Medium** | No max `ipc_len` limit in `ailake_write_batch_ipc` | `ailake-jni/src/lib.rs:1044` | Malicious JNA caller can request ~16 GB allocation via `from_raw_parts` |
| 4 | **Medium** | No rate limiting on HTTP server | `ailake-cli/src/serve.rs` | No DoS protection via request concurrency |
| 5 | **Low** | Secrets in plaintext in-memory (REST token, OAuth2 secret) | `ailake-catalog/src/rest.rs` | Potential leak via `tracing` or core dumps |
| 6 | **Low** | `AILAKE_REST_*` env vars not masked in CLI help/error output | `ailake-cli/src/main.rs` | Secret may appear in shell history if passed as `--rest-token` flag |

## Disclosure history

| Date | CVE | Summary | Fixed in |
|------|-----|---------|----------|
| — | — | No CVEs issued yet | — |
