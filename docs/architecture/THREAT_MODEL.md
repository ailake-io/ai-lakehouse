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
| **I**nformation disclosure: oversized IPC buffer reads beyond allocation | Medium | `ipc_len ≤ 0` check; no max cap (gap #3). Attacker controls IPC buffer content fully |
| **D**enial of service: `query_len = 65536`, `ipc_len = i64::MAX` | Low | `query_len` capped at 65536; IPC len rejected if ≤ 0 but no upper cap (gap #3, 16 GB theoretical) |
| **E**levation of privilege: panic across FFI → JVM compromise | **High** | `catch_ffi_panic` on every export — panic becomes error JSON, never unwinds across FFI |

### Key finding: Release-build dim mismatch (Critical)

`debug_assert_eq` (distance.rs:8,33,58) is compiled out in release. A JVM caller sending `query_len=128` to a table with `dim=256` produces OOB read in SIMD kernels. Mitigated by caller-side validation in `scanner.rs:218` and `writer.rs:206`, but any code path that bypasses these (e.g., direct `ailake_vector_search_json` binary API) is vulnerable.

### Test coverage

- 8 proptest functions: arbitrary bytes, null pointers, extreme dims, extreme top_k, IPC garbage buffers, round-trip write+search (20 I/O cases)
- 6 existing tests: null guards, bad warehouse, dim mismatch, IPC corruption, parse-only

---

## 2. Object Store Layer (`ailake-store`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: fake S3 endpoint | Medium | `allow_http` flag; default is HTTPS. Endpoint configured by caller |
| **T**ampering: path traversal in `LocalStore` | **High** | `full_path()` does `root.join(path)` without canonicalization (gap #2). `../../etc/passwd` resolves outside root. Mitigation: LocalStore only for dev/test |
| **R**epudiation: no operation audit | Low | No mitigation |
| **I**nformation disclosure: `file://` prefix stripping reveals local paths | Low | `strip_prefix("file://")` only — no path sanitization |
| **D**enial of service: `get_range` with extreme offsets | Low | `object_store` handles range validation server-side |
| **E**levation of privilege: cloud credential theft | Medium | Credentials in env vars (`AWS_*`, `AZURE_*`, `GOOGLE_*`). `object_store` handles auth; AI-Lake never logs credentials |

### Key finding: LocalStore path traversal (High)

`LocalStore::full_path(path)` returns `self.root.join(path)` with no `..` check. Any code path accepting user-controlled path can read/write outside the warehouse directory. This affects:
- `ailake insert --file <path>` — write arbitrary local files
- `SearchSession::load` — read arbitrary local files

### Test coverage

- 10 FailStore tests: I/O error injection
- No path traversal test in LocalStore

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

### Test coverage

- 7 corrupt file format tests (Phase 3E)
- 2 mmap proptest (Phase 1A)
- 10 distance proptests + 12 edge case tests (Phase 1A + 3B)

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

---

## 6. HTTP Serve (`ailake-cli/src/serve.rs`)

### STRIDE

| Threat | Risk | Mitigation |
|--------|------|------------|
| **S**poofing: no auth | **High** | Explicit warning on start: "no authentication". Must use authenticating proxy |
| **T**ampering: oversized body | Low | `DefaultBodyLimit::max(32 MB)` |
| **R**epudiation: no access log | Low | No mitigation |
| **I**nformation disclosure: error messages may reveal paths | Low | `ApiError` surfaces Rust error messages |
| **D**enial of service: no rate limit | Medium | No mitigation (gap #4) |
| **E**levation of privilege: N/A | — | — |

### Test coverage

- No dedicated security tests for serve endpoint
- Relies on Axum framework safety

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
| THR-001 | **Critical** | ailake-vec | Release-build dim mismatch → OOB read in SIMD | Open. Mitigated by caller validation |
| THR-002 | **High** | ailake-store | LocalStore path traversal | Open. Documented, not fixed |
| THR-003 | **Medium** | ailake-jni | No max IPC len cap | Open. 16 GB theoretical max |
| THR-004 | **Medium** | ailake-serve | No rate limiting | Open. Serve is internal-only |
| THR-005 | **Low** | ailake-catalog | Secrets in plaintext memory | Open. No tracing of sensitive fields |
| THR-006 | **Low** | ailake-cli | `AILAKE_REST_*` flags visible in `ps aux` | Open. Recommend env vars over flags |
| THR-007 | **Fixed** | ailake-catalog | REST commit: sent -1 instead of null | Fixed Phase 17 |
| THR-008 | **Fixed** | ailake-catalog | REST commit: unconditional AddSchema | Fixed Phase 17 |
| THR-009 | **Fixed** | ailake-catalog | REST commit: AddPartitionSpec on every commit | Fixed Phase 17 |
| THR-010 | **Fixed** | ailake-catalog | Missing namespace before create_table | Fixed Phase 17 |
| THR-011 | **Fixed** | ailake-jni | C-ABI panic across FFI | Fixed v0.0.12: catch_ffi_panic on all exports |
