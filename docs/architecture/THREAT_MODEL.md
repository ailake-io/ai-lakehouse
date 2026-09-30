# THREAT_MODEL.md — AI-Lakehouse Threat Model

## Approach

STRIDE per component. Trust boundary: **every C-ABI call boundary** (JNA from JVM, dlopen from C++, FFI from Python) and **every network boundary** (S3/GCS/Azure, REST catalog, HTTP serve).

### Assumptions

1. JVM/CPython/C++ callers are **untrusted** — they may send arbitrary pointers, lengths, JSON
2. S3/GCS/Azure providers are **trusted** for transport security (TLS)
3. REST catalog server is **semi-trusted** — must authenticate, may serve malicious metadata
4. Local filesystem is **trusted** (only used for development/testing)
5. Network between server and object store is **trusted** (VPC/internal)
6. Disk-level attacks are **out of scope**

---

## 1. JNI/C-ABI Layer (`ailake-jni`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: caller passes wrong table/warehouse | Low | No auth at C-ABI layer — access control delegated to JVM caller |
| **T**ampering: malicious JSON, null pointers, non-UTF-8 | Medium | Every export: null check + `catch_ffi_panic` + `CStr::to_str()` (UTF-8 validation). Returns error JSON, never UB |
| **R**epudiation: no audit log | Low | No mitigation; callers should log at JVM layer |
| **I**nformation disclosure: oversized IPC buffer reads beyond allocation | Low | `ipc_len` is validated as non-negative and capped at 512 MiB before `from_raw_parts`; caller still owns the pointed-to allocation |
| **D**enial of service: `query_len = 65536`, oversized IPC, `top_k = u32::MAX` | Low | `query_len` capped at 65536; IPC capped at 512 MiB; `top_k` capped at `MAX_TOP_K = 100_000` across all search entry points |
| **E**levation of privilege: panic across FFI → JVM compromise | **High** | `catch_ffi_panic` on every export — panic becomes error JSON, never unwinds across FFI |

### Key finding: Release-build dim mismatch (Critical, fixed)

Distance kernels now use an unconditional assertion before SIMD dispatch. A JVM caller sending a mismatched vector receives the normal FFI error envelope through `catch_ffi_panic`; callers still validate dimensions before invoking the kernels.

### Key finding: unbounded `top_k` → process abort (High, fixed)

None of the 5 C-ABI search entry points (`ailake_vector_search_json`,
`ailake_search_json`, `ailake_search_text_json`, `ailake_search_multimodal_json`,
`ailake_scan_json`) validated `top_k` before this fix. It flows into
`ef.max(top_k)` and then `BinaryHeap::with_capacity(ef * 2)` in the HNSW search —
`top_k = u32::MAX` requests a ~8.6 billion-element allocation. Allocation failure
calls Rust's global alloc-error handler, which **aborts the process** — this does
not unwind, so `catch_ffi_panic`'s `catch_unwind` cannot catch it. A single
attacker-controlled `top_k` value crashed the entire embedding host process (Spark
executor, Trino worker, etc.), not just the call. Fixed: `MAX_TOP_K = 100_000`
validated at the top of all 5 entry points, before any allocation-sizing math
(THR-014).

### Test coverage

- 8 proptest functions: arbitrary bytes, null pointers, extreme dims, extreme top_k, IPC garbage buffers, round-trip write+search (20 I/O cases)
- 6 existing tests: null guards, bad warehouse, dim mismatch, IPC corruption, parse-only

---

## 2. Object Store Layer (`ailake-store`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: fake S3 endpoint | Medium | `allow_http` flag; default is HTTPS. Endpoint configured by caller |
| **T**ampering: path traversal in `LocalStore` | Low | `LocalStore` rejects `..` components and absolute paths outside the configured root before any filesystem operation |
| **R**epudiation: no operation audit | Low | No mitigation |
| **I**nformation disclosure: `file://` prefix stripping reveals local paths | Low | `strip_prefix("file://")` only — no path sanitization |
| **D**enial of service: `get_range` with extreme offsets | Low | `object_store` handles range validation server-side |
| **E**levation of privilege: cloud credential theft | Medium | Credentials in env vars (`AWS_*`, `AZURE_*`, `GOOGLE_*`). `object_store` handles auth; AI-Lake never logs credentials |

### Key finding: LocalStore path traversal (High, fixed)

`LocalStore::full_path(path)` now rejects parent-directory components and absolute paths outside the configured root. Regression tests cover both forms.

### Test coverage

- 10 FailStore tests: I/O error injection
- Path traversal tests in LocalStore for `../` and external `file://` paths

---

## 3. REST Catalog (`ailake-catalog/src/rest.rs`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: fake REST server | Medium | TLS via HTTPS; no certificate pinning. Auth token sent over TLS |
| **T**ampering: malicious metadata from server | Medium | OCC (optimistic concurrency control) — 5 retries on `CommitFailedException`. Schema comparison before `AddSchema` |
| **R**epudiation: no audit | Low | Iceberg snapshot history provides immutable audit trail |
| **I**nformation disclosure: token in env/logs | Low | `AILAKE_REST_TOKEN`, `AILAKE_REST_OAUTH_CLIENT_SECRET` in env. CLI flags visible in `ps aux`. No masking |
| **D**enial of service: slow catalog responses | Low | No timeout configuration exposed; Tokio runtime handles timeouts at OS level |
| **E**levation of privilege: OAuth2 token reuse | Low | Token refresh handled by client; scope limited by server |

### Known bugs (fixed in Phase 17)

1. Missing namespace creation before table creation → `NoSuchNamespaceException`
2. `-1` sentinel for "no snapshot" sent as integer instead of JSON `null` → commit failure
3. Duplicate `AddPartitionSpec` on every commit → server 500
4. Inconditional `AddSchema`+`SetCurrentSchema` on every commit → schema ID drift

### Test coverage

- 9 unit tests (URL construction, auth parsing, config building)
- 2 live tests (`#[ignore]` by default, require `apache/iceberg-rest-fixture` container)
- Round-trip create→insert→commit→search verified via `ailake-py` against live container

---

## 4. File Format (`ailake-file`, `ailake-vec`, `ailake-index`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: fake AILK magic | Low | Magic bytes verified (`b"AILK"`, `b"AFTS"`, `b"PAR1"`) |
| **T**ampering: corrupt HNSW blob → panic | **High** | All offsets use `checked_add`; out-of-bounds → `NotAnAilakeFile` error. 7 corruption tests |
| **R**epudiation: N/A | — | — |
| **I**nformation disclosure: malformed centroid → wrong pruning | Low | Centroid parse validated (`InvalidCentroidLength`). Missing centroid → conservative (keep file) |
| **D**enial of service: extremely large HNSW blob | Low | Blob size bounded by `u64`; reader validates against file length |
| **E**levation of privilege: N/A | — | — |

### Key finding: Release dim mismatch (Critical, see §1)

### Key finding: HNSW graph-structure validation (High, fixed)

The file-format layer's `checked_add`/`NotAnAilakeFile` mitigation (row above) only
validates *byte offsets into the file* — it says nothing about whether a
successfully-decoded HNSW graph is internally consistent. `HnswSerializer::from_bytes`
(`ailake-index/src/serialize.rs`) used to bincode-deserialize `neighbors`/`entry_point`
straight into `HnswIndex` with no check that node indices stay within
`row_ids.len()`; the only downstream guard was a `debug_assert!` in
`VisitedTracker::visit` (`ailake-index/src/hnsw.rs`), compiled out in release, ahead of
an `unsafe { get_unchecked_mut(idx) }`. A structurally-valid-bincode-but-corrupt graph
(bit flip, truncation that still decodes, malicious IPC/S3 payload) caused
undefined behavior in release builds. Fixed: `from_bytes` now validates
`entry_point`/`neighbors`/`node_levels`/`flat_vecs` bounds and returns
`AilakeError::Bincode` instead of constructing an invalid index (THR-013).

### Test coverage

- 7 corrupt file format tests (Phase 3E)
- 2 mmap proptest (Phase 1A)
- 10 distance proptests + 12 edge case tests (Phase 1A + 3B)
- 3 `HnswSerializer::from_bytes` bounds-validation tests (out-of-bounds neighbor,
  out-of-bounds entry_point, flat_vecs length mismatch)

---

## 5. Query Engine (`ailake-query`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: wrong table/warehouse | Low | Caller provides identifiers; no auth |
| **T**ampering: malicious `ScoreFn` | Low | `ScoreFn` is Rust closure, not user-supplied at runtime; only configured at compile time |
| **R**epudiation: N/A | — | Search is stateless |
| **I**nformation disclosure: equality delete file read failure | Low | `warn!` + continue — shows deleted rows rather than failing. Deliberate: safety over correctness |
| **D**enial of service: `top_k = 100000`, no geometric pruning | Low | `ef_search.clamp(1, 100000)`. Files without centroid always included (no pruning) |
| **E**levation of privilege: N/A | — | — |

### Test coverage

- 2 FailStore proptest (scanner + compaction, 256 cases each)
- Compaction recall parity test
- Concurrent stress test (1 compactor + 4 searchers, 5 passes)
- Dimension mismatch rejection test
- 41 unit tests (scanner, writer, compaction, pruner, bm25, mem_table)
- 3 Loom models (`ailake-query/src/loom_tests.rs`) — see coverage gap below

### Known gap: Loom models don't cover the JNI lock+block_on pattern (Open)

The 3 Loom models verify: per-key mutex acquisition from a shared map doesn't
deadlock (`jni_table_lock_model`), a generic once-init race pattern in isolation
(`once_init_flag_model` — doesn't correspond to a real call site; actual once-init
uses `std::sync::OnceLock`), and the real `part_counter` increment ordering
(`writer_batch_counter_model`, `Ordering::SeqCst`, matches `writer.rs:387` exactly).

None of them model that `ailake-jni/src/lib.rs` (lines 787, 1149, 1409, 2208, 2355,
2456) takes a `std::sync::Mutex` per-table lock and then calls
`rt().block_on(async { ... })` **while holding it** — blocking an OS thread on a
shared Tokio runtime from inside a lock the runtime's own tasks might need. Loom
models thread interleavings, not async runtime scheduling, so this class of runtime
starvation is out of reach for Loom as used here. Not fixed in this pass (THR-012)
— would need either restructuring those call sites to not block while holding the
lock, or a different verification approach (e.g. `tokio-console`, load testing under
contention) than Loom can provide.

---

## 6. HTTP Serve (`ailake-cli/src/serve.rs`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: no auth | **High** | Optional Bearer token on every endpoint except `/healthz`; startup warning when disabled. Use a gateway/mTLS for production |
| **T**ampering: oversized body | Low | `DefaultBodyLimit::max(32 MB)` |
| **R**epudiation: no access log | Low | Prometheus counters and request duration are exposed at `/metrics`; access-log correlation remains a deployment concern |
| **I**nformation disclosure: error messages may reveal paths | Low | `ApiError` surfaces Rust error messages |
| **D**enial of service: request flooding | Medium | 32 MB body limit, 64 in-flight request cap, and optional Redis/Valkey rate limiting by bearer token/IP with separate read/write quotas; keep gateway/mTLS controls for internet-facing deployments |
| **E**levation of privilege: N/A | — | — |

### Test coverage

- Health, metrics authentication, cache invalidation and auto-compaction paths are covered by dedicated server tests.
- HTTP load, multi-writer fencing, cache invalidation and rate-limit decisions are exercised by server tests and `performance.yml`; production deployments should still validate gateway behavior and real Redis/Valkey failure policy.

---

## 7. CI/CD Pipeline

| Threat | Risk | Mitigation |
|--------|------|------------|
| Compromised dependency → supply chain attack | Medium | `cargo-deny` + `dependabot` |
| Secret leak in CI logs | Low | Tokens in GitHub secrets, not inline |
| Malicious PR → UB in unsafe code | Medium | Miri + sanitizers in CI; `--only-verified` TruffleHog on all PRs |
| Docker container escape (compat-heavy) | Low | Docker cleanup on exit (always, even on failure); fixed IP for Spark |

---

## Risk register

| ID | Severity | Component | Finding | Status |
|----|----------|-----------|---------|--------|
| THR-001 | **Critical** | ailake-vec | Release-build dim mismatch → OOB read in SIMD | Fixed. Unconditional assertion + FFI panic boundary |
| THR-002 | **High** | ailake-store | LocalStore path traversal | Fixed. Root containment validation + regression tests |
| THR-003 | **Medium** | ailake-jni | No max IPC len cap | Fixed. 512 MiB cap before slice creation |
| THR-004 | **Medium** | ailake-serve | Distributed rate limiting unavailable | Fixed in-process and with Redis/Valkey backend; trust proxy headers is opt-in and an external gateway remains recommended |
| THR-005 | **Low** | ailake-catalog | Secret material exposure | Mitigated. Provider inputs use `SecretString`; temporary buffers are zeroized and errors/debug output redact secret contents |
| THR-006 | **Low** | ailake-cli | Inline REST secret flags visible in `ps aux` | Mitigated. Use `env://`, mounted files, Vault, Kubernetes Secrets or AWS Secrets Manager references; direct flags remain compatibility-only |
| THR-012 | **Medium** | local catalogs | Cross-process commit race | Fixed for LocalStore-backed Hadoop and DuckLake with exclusive lock files; deferred index jobs use conditional object-store leases, while catalog commits on object stores continue to rely on REST/Nessie OCC |
| THR-007 | **Fixed** | ailake-catalog | REST commit: sent -1 instead of null | Fixed Phase 17 |
| THR-008 | **Fixed** | ailake-catalog | REST commit: unconditional AddSchema | Fixed Phase 17 |
| THR-009 | **Fixed** | ailake-catalog | REST commit: AddPartitionSpec on every commit | Fixed Phase 17 |
| THR-010 | **Fixed** | ailake-catalog | Missing namespace before create_table | Fixed Phase 17 |
| THR-011 | **Fixed** | ailake-jni | C-ABI panic across FFI | Fixed v0.0.12: catch_ffi_panic on all exports |
| THR-012 | **Medium** | ailake-jni | Mutex held across `block_on` (Tokio starvation) | Open. Not modeled by Loom; needs restructuring or a different verification tool |
| THR-013 | **Fixed** | ailake-index | Unvalidated HNSW neighbor/entry_point indices → UB in release | Fixed: `HnswSerializer::from_bytes` bounds-validates before constructing the index |
| THR-014 | **Fixed** | ailake-jni | Unbounded `top_k` → alloc failure aborts process | Fixed: `MAX_TOP_K = 100_000` at all 5 search entry points |
| THR-015 | **Fixed** | ailake-vec, ailake-catalog | Non-finite `radius` silently lost on manifest round-trip | Fixed: excluded from the max-fold in `compute_centroid_and_radius`; explicit `warn!` + drop as defense in depth in `avro_manifest.rs` |
