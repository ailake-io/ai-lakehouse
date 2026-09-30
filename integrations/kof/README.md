# AI-Lake + Kof

This directory contains a production-oriented Kof integration for the AI-Lake
service. The HTTP client exposes typed DTOs on Kof JVM and JS, while the Native
target uses the stable C-ABI binding for in-process calls.

## Run

Start AI-Lake against an existing table:

```bash
cargo run -p ailake-cli -- \
  --store ./warehouse serve default.docs \
  --host 127.0.0.1 --port 7700
```

Then execute the typed Kof HTTP client:

```bash
kof check integrations/kof/client --target jvm
kof run integrations/kof/client/main.kf --target jvm
```

The `search.kf` example demonstrates the typed-input `AilakeClient.search()` method:

```bash
kof check integrations/kof/search --target jvm
kof run integrations/kof/search/main.kf --target jvm
```

For a non-local deployment, start the server with `--auth-token`. The client
sends `Authorization: Bearer ...`, `Accept`, `Content-Type` and
`X-Ailake-Client-Version` headers. Configure resilience with
`client.configure(timeoutSeconds, retries, circuitTrips)`.

The paginated methods use the server-side page envelope:

```kof
client.setBearerToken(token)
client.configure(15, 2, 5)
var page = client.listJobsPage(0, 100)
println(page.has_more)
```

`GET /jobs` and `GET /index-jobs` remain backward-compatible arrays without
query parameters. With `offset` and `limit`, they return `items`, `offset`,
`limit`, `total` and `has_more`.

## Native C-ABI binding

```bash
cargo build -p ailake-jni
kof check integrations/kof/native --target native
kof run integrations/kof/native/main.kf --target native
```

`AilakeNative` uses `ailake_kof_*` borrowed adapters. They release the Rust
allocation with the AI-Lake allocator and let Kof copy the response into a
managed string, avoiding an ownership mismatch. The borrowed pointer is
thread-local and valid until the next adapter call on that thread.

The Native Kof JSON decoder currently does not accept nested DTO fields such as
`List<SearchResult>`. Therefore the Native path intentionally exposes the
versioned JSON C-ABI response, while nested typed DTOs are used by the JVM/JS
HTTP client until that Kof runtime limitation is closed upstream.

## Contract

The service exposes:

```text
GET  /healthz
GET  /readyz
GET  /info
POST /search
POST /write
POST /compact
POST /jobs/compact
GET  /jobs
GET  /jobs/{id}
POST /jobs/{id}/cancel
POST /jobs/{id}/retry
GET  /index-jobs
GET  /index-jobs/{id}
POST /index-jobs/{id}/cancel
POST /index-jobs/{id}/retry
GET  /metrics
```

`/healthz` is intentionally unauthenticated for liveness probes. All other
endpoints require `Authorization: Bearer <token>` when `--auth-token` is set.
`/jobs/compact` returns `202` and persists the job state under
`metadata/ailake_jobs.json`; poll `/jobs/{id}` until it reaches `succeeded` or
`failed`. Use `/jobs/{id}/cancel` for queued/running jobs and
`/jobs/{id}/retry` for failed/cancelled jobs. Deferred HNSW and IVF-PQ builds
are persisted independently under `metadata/index-jobs/`; `/index-jobs/{id}`
reports `progress`, `attempts`, and status, and the cancel/retry endpoints
control those builds. `/metrics` exposes Prometheus text format.
