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
- Vulnerabilities in benchmark or test tooling (`ailake-bench`, `tests/`)
