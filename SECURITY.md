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

Out of scope:
- Denial-of-service via intentionally large inputs (no SLA for resource exhaustion)
- Issues in transitive dependencies not triggered by AI-Lake code paths
- Vulnerabilities in benchmark or test tooling ([`ailake-benchmarks`](https://github.com/ThiagoLange/ailake-benchmarks), `tests/`)

---

## Security hardening (v0.0.27+)

Mitigations applied in the security audit of June 2026:

### JNI / C-ABI

| Guard | Location | Detail |
|---|---|---|
| `query_len > 65 536` rejected | `ailake-jni/src/lib.rs` — `ailake_vector_search_json` | Returns `{"ok":false,"error":"..."}` immediately; prevents oversized heap allocation from untrusted callers |
| `ef_search` capped at 100 000 | `ailake-jni/src/lib.rs` — `ailake_search_json` + `ailake_scan_json` | `req.ef_search.min(100_000)` prevents runaway HNSW candidate heap under adversarial input |

### HTTP server (`ailake serve`)

| Guard | Location | Detail |
|---|---|---|
| `DefaultBodyLimit::max(32 MB)` | `ailake-cli/src/serve.rs` | Axum middleware; rejects oversized request bodies before handler |
| Empty query → HTTP 400 | `ailake-cli/src/serve.rs` — `handle_search` | Returns `ApiError("query must not be empty")` before any I/O |
| `top_k.clamp(1, 10_000)` | `ailake-cli/src/serve.rs` — `handle_search` | Prevents unbounded HNSW result sets |
| Startup warning | `ailake-cli/src/serve.rs` | Prints `eprintln!("WARNING: ailake serve has no authentication …")` to stderr on start |

> **Note**: `ailake serve` is intended for internal/trusted networks. Do not expose it directly to the public internet without an authenticating proxy (API gateway, mTLS sidecar, etc.).

### Airflow provider

| Guard | Location | Detail |
|---|---|---|
| stderr/stdout truncated to 4 096 chars | `airflow-providers-ailake/airflow_providers_ailake/hooks/ailake.py` — `run_cli` | Prevents cloud SDK verbose error messages (which may include credential-adjacent context) from flooding Airflow task logs |
