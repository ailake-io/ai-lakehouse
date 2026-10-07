// SPDX-License-Identifier: MIT OR Apache-2.0
// HTTP server for AI-Lake — exposes search, write, compact, and info over JSON.
//
// Endpoints:
//   POST /search   {"query":[f32...], "top_k":10, "pruning_threshold":0.8}
//   POST /write    {"texts":["..."], "embeddings":[[f32...]], "batch_id":"..."}
//   POST /compact  {}
//   GET  /info
//
// SECURITY: This server has no authentication. It is designed for trusted-network
// deployments (localhost, VPC-internal, sidecar). Do NOT expose it on a public
// interface without an authenticating reverse proxy (e.g., nginx + mTLS, API gateway).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, Semaphore};
use tracing::{info, warn};
use uuid::Uuid;

use ailake_cache::{
    CacheConfig, CacheKind, CacheManager, CachingStore, CircuitBreaker, CircuitBreakingStore,
    RateLimitClass, RateLimitConfig, RateLimiter,
};
use ailake_catalog::provider::{
    new_snapshot_id, CatalogProvider, IndexStatus, NewSnapshot, SnapshotOperation, TableIdent,
};
use ailake_catalog::DataFileEntry;
use ailake_core::{AilakeError, VectorStoragePolicy};
use ailake_query::{
    handle_for, list_index_jobs, load_index_job, resume_index_jobs, CompactionConfig,
    CompactionExecutor, CompactionPlanner, SearchConfig, TableWriter,
};
use ailake_store::Store;

/// Minimum time between foreign-file probes (see `maybe_probe_auto_compact`).
/// Bounds the extra `list_files` catalog call to at most once per window,
/// server-wide, regardless of query rate.
const AUTO_COMPACT_CHECK_COOLDOWN_MS: u64 = 60_000;
const JOBS_PREFIX: &str = "metadata/ailake-jobs";
const LEGACY_JOBS_PATH: &str = "metadata/ailake_jobs.json";
const JOBS_GC_LOCK_PATH: &str = "metadata/ailake-jobs/gc.lock";
const MAX_PERSISTED_JOBS: usize = 1_000;
const SERVE_LOCK_PATH_PREFIX: &str = "metadata/ailake-serve";
const SERVE_LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub(crate) struct AppState {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: TableIdent,
    policy: VectorStoragePolicy,
    /// Guards against overlapping auto-compact passes (ADR-018 gap fix: a
    /// long-running server sees the same foreign/externally-rewritten file
    /// (Spark/Trino `OPTIMIZE`, DuckDB) degrade to O(N) flat scan on every
    /// query until someone runs `ailake compact` by hand — see
    /// `maybe_probe_auto_compact`).
    auto_compact_inflight: Arc<AtomicBool>,
    /// Unix-ms timestamp of the last foreign-file probe; rate-limits how
    /// often `handle_search` re-lists files to check for them.
    auto_compact_last_check_ms: Arc<AtomicU64>,
    /// Optional Bearer token. `None` preserves the trusted-network mode.
    auth_token: Option<SecretString>,
    /// Bounds concurrent searches/writes/compactions in the process.
    inflight: Arc<Semaphore>,
    /// Shared two-tier cache for query results, metadata and index reads.
    cache: Arc<CacheManager>,
    /// Distributed quotas keyed by authenticated token and trusted client IP.
    rate_limiter: Arc<RateLimiter>,
    /// Catalog dependency circuit breaker.
    catalog_breaker: Arc<CircuitBreaker>,
    /// Object-storage dependency circuit breaker.
    storage_breaker: Arc<CircuitBreaker>,
    /// Fail queries when delete metadata cannot be loaded.
    strict_deletes: bool,
    /// Accept client IP headers only when explicitly enabled for a trusted proxy.
    trust_proxy_headers: bool,
    /// Process and job metrics exposed through `/metrics`.
    metrics: Arc<ServerMetrics>,
    /// Durable asynchronous job registry.
    jobs: Arc<JobManager>,
}

pub(crate) struct ServeConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) auth_token: Option<SecretString>,
    pub(crate) cache_url: Option<String>,
    pub(crate) cache_max_bytes: usize,
    pub(crate) cache_ttl_secs: u64,
    pub(crate) strict_deletes: bool,
    pub(crate) rate_limit_url: Option<String>,
    pub(crate) rate_window_secs: u64,
    pub(crate) search_quota_token: u64,
    pub(crate) write_quota_token: u64,
    pub(crate) search_quota_ip: u64,
    pub(crate) write_quota_ip: u64,
    pub(crate) trust_proxy_headers: bool,
    pub(crate) rate_limit_fail_closed: bool,
    pub(crate) circuit_failure_threshold: u32,
    pub(crate) circuit_cooldown_secs: u64,
}

struct CircuitBreakingCatalog {
    inner: Arc<dyn CatalogProvider>,
    breaker: Arc<CircuitBreaker>,
}

impl CircuitBreakingCatalog {
    fn new(inner: Arc<dyn CatalogProvider>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }

    fn open_error() -> AilakeError {
        AilakeError::Catalog("catalog circuit breaker is open".into())
    }
}

#[async_trait::async_trait]
impl CatalogProvider for CircuitBreakingCatalog {
    async fn create_table(
        &self,
        name: &TableIdent,
        props: &ailake_catalog::provider::TableProperties,
    ) -> ailake_core::AilakeResult<()> {
        self.breaker
            .execute(Self::open_error(), self.inner.create_table(name, props))
            .await
    }

    async fn load_table(
        &self,
        name: &TableIdent,
    ) -> ailake_core::AilakeResult<ailake_catalog::provider::TableMetadata> {
        self.breaker
            .execute(Self::open_error(), self.inner.load_table(name))
            .await
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> ailake_core::AilakeResult<ailake_catalog::provider::SnapshotId> {
        self.breaker
            .execute(
                Self::open_error(),
                self.inner.commit_snapshot(table, snapshot),
            )
            .await
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<ailake_catalog::provider::SnapshotId>,
    ) -> ailake_core::AilakeResult<Vec<DataFileEntry>> {
        self.breaker
            .execute(
                Self::open_error(),
                self.inner.list_files(table, snapshot_id),
            )
            .await
    }

    async fn drop_table(&self, name: &TableIdent) -> ailake_core::AilakeResult<()> {
        self.breaker
            .execute(Self::open_error(), self.inner.drop_table(name))
            .await
    }

    fn retires_files_physically(&self) -> bool {
        self.inner.retires_files_physically()
    }

    fn supports_in_place_rewrite(&self) -> bool {
        self.inner.supports_in_place_rewrite()
    }

    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: ailake_catalog::schema_evolution::SchemaEvolution,
    ) -> ailake_core::AilakeResult<i32> {
        self.breaker
            .execute(
                Self::open_error(),
                self.inner.evolve_schema(table, evolution),
            )
            .await
    }

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        snapshot_id: Option<ailake_catalog::provider::SnapshotId>,
    ) -> ailake_core::AilakeResult<Vec<ailake_catalog::provider::EqualityDeleteFile>> {
        self.breaker
            .execute(
                Self::open_error(),
                self.inner.list_equality_deletes(table, snapshot_id),
            )
            .await
    }

    async fn load_raw_metadata(
        &self,
        table: &TableIdent,
    ) -> ailake_core::AilakeResult<ailake_catalog::provider::IcebergMetadata> {
        self.breaker
            .execute(Self::open_error(), self.inner.load_raw_metadata(table))
            .await
    }

    async fn list_snapshots(
        &self,
        table: &TableIdent,
    ) -> ailake_core::AilakeResult<Vec<ailake_catalog::provider::IcebergSnapshot>> {
        self.breaker
            .execute(Self::open_error(), self.inner.list_snapshots(table))
            .await
    }

    async fn add_vector_column(
        &self,
        table: &TableIdent,
        spec: &ailake_core::VectorColSpec,
    ) -> ailake_core::AilakeResult<i32> {
        self.breaker
            .execute(
                Self::open_error(),
                self.inner.add_vector_column(table, spec),
            )
            .await
    }
}

#[derive(Default)]
struct ServerMetrics {
    requests_total: AtomicU64,
    requests_failed: AtomicU64,
    rate_limited: AtomicU64,
    search_total: AtomicU64,
    write_total: AtomicU64,
    compact_total: AtomicU64,
    request_duration_ms: AtomicU64,
    jobs_submitted: AtomicU64,
    jobs_succeeded: AtomicU64,
    jobs_failed: AtomicU64,
}

impl ServerMetrics {
    fn render(
        &self,
        inflight: usize,
        cache: ailake_cache::CacheStats,
        rate: ailake_cache::RateLimitStats,
        catalog_open: bool,
        storage_open: bool,
    ) -> String {
        let flat_scan = ailake_query::scanner::flat_scan_stats();
        format!(
            "# TYPE ailake_http_requests_total counter\n\
             ailake_http_requests_total {}\n\
             # TYPE ailake_http_requests_failed_total counter\n\
             ailake_http_requests_failed_total {}\n\
             # TYPE ailake_http_rate_limited_total counter\n\
             ailake_http_rate_limited_total {}\n\
             # TYPE ailake_http_request_duration_ms_total counter\n\
             ailake_http_request_duration_ms_total {}\n\
             # TYPE ailake_http_search_total counter\n\
             ailake_http_search_total {}\n\
             # TYPE ailake_http_write_total counter\n\
             ailake_http_write_total {}\n\
             # TYPE ailake_http_compact_total counter\n\
             ailake_http_compact_total {}\n\
             # TYPE ailake_search_flat_scan_deferred_files_total counter\n\
             ailake_search_flat_scan_deferred_files_total {}\n\
             # TYPE ailake_search_flat_scan_unexpected_files_total counter\n\
             ailake_search_flat_scan_unexpected_files_total {}\n\
             # TYPE ailake_search_flat_scan_rows_total counter\n\
             ailake_search_flat_scan_rows_total {}\n\
             # TYPE ailake_search_flat_scan_elapsed_micros_total counter\n\
             ailake_search_flat_scan_elapsed_micros_total {}\n\
             # TYPE ailake_cache_hits_total counter\n\
             ailake_cache_hits_total {}\n\
             # TYPE ailake_cache_misses_total counter\n\
             ailake_cache_misses_total {}\n\
             # TYPE ailake_cache_inserts_total counter\n\
             ailake_cache_inserts_total {}\n\
             # TYPE ailake_cache_evictions_total counter\n\
             ailake_cache_evictions_total {}\n\
             # TYPE ailake_cache_invalidations_total counter\n\
             ailake_cache_invalidations_total {}\n\
             # TYPE ailake_cache_redis_hits_total counter\n\
             ailake_cache_redis_hits_total {}\n\
             # TYPE ailake_cache_redis_errors_total counter\n\
             ailake_cache_redis_errors_total {}\n\
             # TYPE ailake_cache_entries gauge\n\
             ailake_cache_entries {}\n\
             # TYPE ailake_cache_bytes_in_use gauge\n\
             ailake_cache_bytes_in_use {}\n\
             # TYPE ailake_cache_bytes_limit gauge\n\
             ailake_cache_bytes_limit {}\n\
             # TYPE ailake_rate_allowed_total counter\n\
             ailake_rate_allowed_total {}\n\
             # TYPE ailake_rate_limited_total counter\n\
             ailake_rate_limited_total {}\n\
             # TYPE ailake_rate_backend_errors_total counter\n\
             ailake_rate_backend_errors_total {}\n\
             # TYPE ailake_rate_redis_requests_total counter\n\
             ailake_rate_redis_requests_total {}\n\
             # TYPE ailake_catalog_circuit_open gauge\n\
             ailake_catalog_circuit_open {}\n\
             # TYPE ailake_storage_circuit_open gauge\n\
             ailake_storage_circuit_open {}\n\
             # TYPE ailake_jobs_submitted_total counter\n\
             ailake_jobs_submitted_total {}\n\
             # TYPE ailake_jobs_succeeded_total counter\n\
             ailake_jobs_succeeded_total {}\n\
             # TYPE ailake_jobs_failed_total counter\n\
             ailake_jobs_failed_total {}\n\
             # TYPE ailake_http_inflight_limit gauge\n\
             ailake_http_inflight_limit {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.requests_failed.load(Ordering::Relaxed),
            self.rate_limited.load(Ordering::Relaxed),
            self.request_duration_ms.load(Ordering::Relaxed),
            self.search_total.load(Ordering::Relaxed),
            self.write_total.load(Ordering::Relaxed),
            self.compact_total.load(Ordering::Relaxed),
            flat_scan.deferred_files_total,
            flat_scan.unexpected_files_total,
            flat_scan.rows_total,
            flat_scan.elapsed_micros_total,
            cache.hits_total,
            cache.misses_total,
            cache.inserts_total,
            cache.evictions_total,
            cache.invalidations_total,
            cache.redis_hits_total,
            cache.redis_errors_total,
            cache.entries,
            cache.bytes_in_use,
            cache.bytes_limit,
            rate.allowed_total,
            rate.limited_total,
            rate.backend_errors_total,
            rate.redis_requests_total,
            u8::from(catalog_open),
            u8::from(storage_open),
            self.jobs_submitted.load(Ordering::Relaxed),
            self.jobs_succeeded.load(Ordering::Relaxed),
            self.jobs_failed.load(Ordering::Relaxed),
            inflight,
        )
    }
}

struct RequestMetricGuard {
    metrics: Arc<ServerMetrics>,
    route: &'static str,
    started: Instant,
    succeeded: bool,
}

impl RequestMetricGuard {
    fn new(metrics: &Arc<ServerMetrics>, route: &'static str) -> Self {
        Self {
            metrics: Arc::clone(metrics),
            route,
            started: Instant::now(),
            succeeded: false,
        }
    }

    fn success(&mut self) {
        self.succeeded = true;
    }
}

impl Drop for RequestMetricGuard {
    fn drop(&mut self) {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .request_duration_ms
            .fetch_add(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
        if !self.succeeded {
            self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        match self.route {
            "search" => self.metrics.search_total.fetch_add(1, Ordering::Relaxed),
            "write" => self.metrics.write_total.fetch_add(1, Ordering::Relaxed),
            "compact" => self.metrics.compact_total.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobRecord {
    id: String,
    kind: String,
    status: String,
    created_at: u64,
    updated_at: u64,
    error: Option<String>,
    result: Option<String>,
    #[serde(default)]
    request: Option<CompactRequest>,
    #[serde(default)]
    cancel_requested: bool,
}

#[derive(Debug, Deserialize, Default)]
struct PaginationQuery {
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct PageResponse<T> {
    items: Vec<T>,
    offset: usize,
    limit: usize,
    total: usize,
    has_more: bool,
}

const DEFAULT_PAGE_LIMIT: usize = 100;
const MAX_PAGE_LIMIT: usize = 1_000;

fn page_bounds(query: &PaginationQuery, total: usize) -> (usize, usize) {
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query
        .limit
        .unwrap_or(DEFAULT_PAGE_LIMIT)
        .clamp(1, MAX_PAGE_LIMIT);
    (offset, limit)
}

struct JobManager {
    store: Arc<dyn Store>,
}

impl JobManager {
    async fn load(store: Arc<dyn Store>) -> Result<Arc<Self>, String> {
        let manager = Arc::new(Self { store });
        manager.migrate_legacy_registry().await?;
        for job in manager.list().await {
            if job.status == "running" || job.status == "cancel_requested" {
                manager
                    .update(
                        &job.id,
                        "failed",
                        None,
                        Some("process restarted before the job completed".into()),
                    )
                    .await?;
            }
        }
        manager.prune().await?;
        Ok(manager)
    }

    fn job_path(id: &str) -> String {
        format!("{JOBS_PREFIX}/{id}.json")
    }

    fn job_lock_path(id: &str) -> String {
        format!("{JOBS_PREFIX}/{id}.lock")
    }

    async fn acquire_lock(&self, path: &str) -> Result<u64, String> {
        for attempt in 0..40u32 {
            match self
                .store
                .try_acquire_lock_fenced(path)
                .await
                .map_err(|e| e.to_string())?
            {
                Some(fence) => return Ok(fence),
                None => {
                    tokio::time::sleep(Duration::from_millis(25 + u64::from(attempt))).await;
                }
            }
        }
        Err(format!(
            "job registry lock is held by another instance: {path}"
        ))
    }

    async fn mutate_job<T, F>(&self, id: &str, mutate: F) -> Result<T, String>
    where
        F: FnOnce(&mut JobRecord) -> Result<T, String>,
    {
        let lock_path = Self::job_lock_path(id);
        let fence = self.acquire_lock(&lock_path).await?;
        let result = async {
            let bytes = self
                .store
                .get(&Self::job_path(id))
                .await
                .map_err(|e| e.to_string())?;
            let mut job: JobRecord = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            let value = mutate(&mut job)?;
            if !self
                .store
                .check_lock_fence(&lock_path, fence)
                .await
                .map_err(|e| e.to_string())?
            {
                return Err("job fencing token is no longer valid".into());
            }
            job.updated_at = unix_seconds();
            let json = serde_json::to_vec(&job).map_err(|e| e.to_string())?;
            self.store
                .put(&Self::job_path(id), bytes::Bytes::from(json))
                .await
                .map_err(|e| e.to_string())?;
            Ok(value)
        }
        .await;
        let release = self
            .store
            .release_lock_fenced(&lock_path, fence)
            .await
            .map_err(|e| e.to_string());
        match (result, release) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    async fn create_job(&self, record: &JobRecord) -> Result<(), String> {
        let lock_path = Self::job_lock_path(&record.id);
        let fence = self.acquire_lock(&lock_path).await?;
        let result = async {
            if self
                .store
                .exists(&Self::job_path(&record.id))
                .await
                .map_err(|e| e.to_string())?
            {
                return Err(format!("job already exists: {}", record.id));
            }
            if !self
                .store
                .check_lock_fence(&lock_path, fence)
                .await
                .map_err(|e| e.to_string())?
            {
                return Err("job fencing token is no longer valid".into());
            }
            let json = serde_json::to_vec(record).map_err(|e| e.to_string())?;
            self.store
                .put(&Self::job_path(&record.id), bytes::Bytes::from(json))
                .await
                .map_err(|e| e.to_string())
        }
        .await;
        let release = self
            .store
            .release_lock_fenced(&lock_path, fence)
            .await
            .map_err(|e| e.to_string());
        match (result, release) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }

    async fn submit(&self, kind: &str, request: CompactRequest) -> Result<JobRecord, String> {
        let now = unix_seconds();
        let record = JobRecord {
            id: Uuid::new_v4().to_string(),
            kind: kind.into(),
            status: "queued".into(),
            created_at: now,
            updated_at: now,
            error: None,
            result: None,
            request: Some(request),
            cancel_requested: false,
        };
        self.create_job(&record).await?;
        self.prune().await?;
        Ok(record)
    }

    async fn update(
        &self,
        id: &str,
        status: &str,
        result: Option<String>,
        error: Option<String>,
    ) -> Result<(), String> {
        self.mutate_job(id, |job| {
            job.status = status.into();
            job.result = result;
            job.error = error;
            if status == "running" {
                job.cancel_requested = false;
            }
            Ok(())
        })
        .await
    }

    async fn get(&self, id: &str) -> Option<JobRecord> {
        let bytes = self.store.get(&Self::job_path(id)).await.ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    async fn list(&self) -> Vec<JobRecord> {
        let mut jobs = Vec::new();
        let Ok(paths) = self.store.list(JOBS_PREFIX).await else {
            return jobs;
        };
        for path in paths {
            if !path.ends_with(".json") || path == LEGACY_JOBS_PATH {
                continue;
            }
            if let Ok(bytes) = self.store.get(&path).await {
                if let Ok(job) = serde_json::from_slice::<JobRecord>(&bytes) {
                    jobs.push(job);
                }
            }
        }
        jobs.sort_by_key(|job| std::cmp::Reverse(job.created_at));
        jobs
    }

    async fn cancel(&self, id: &str) -> Result<JobRecord, String> {
        self.mutate_job(id, |job| {
            match job.status.as_str() {
                "queued" => job.status = "cancelled".into(),
                "running" => {
                    job.status = "cancel_requested".into();
                    job.cancel_requested = true;
                }
                "cancel_requested" | "cancelled" => return Ok(job.clone()),
                _ => return Err(format!("job {id} is already terminal")),
            }
            Ok(job.clone())
        })
        .await
    }

    async fn retry(&self, id: &str) -> Result<JobRecord, String> {
        self.mutate_job(id, |job| {
            if !matches!(job.status.as_str(), "failed" | "cancelled") {
                return Err(format!(
                    "job {id} is not retryable in status {}",
                    job.status
                ));
            }
            if job.request.is_none() {
                return Err(format!("job {id} has no persisted request payload"));
            }
            job.status = "queued".into();
            job.error = None;
            job.result = None;
            job.cancel_requested = false;
            Ok(job.clone())
        })
        .await
    }

    async fn is_cancel_requested(&self, id: &str) -> bool {
        self.get(id)
            .await
            .map(|job| job.cancel_requested)
            .unwrap_or(false)
    }

    async fn migrate_legacy_registry(&self) -> Result<(), String> {
        let existing = self
            .list()
            .await
            .into_iter()
            .map(|job| job.id)
            .collect::<std::collections::HashSet<_>>();
        let Ok(bytes) = self.store.get(LEGACY_JOBS_PATH).await else {
            return Ok(());
        };
        let legacy: HashMap<String, JobRecord> =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        for job in legacy.values() {
            if !existing.contains(&job.id) {
                self.create_job(job).await?;
            }
        }
        Ok(())
    }

    async fn prune(&self) -> Result<(), String> {
        let Some(fence) = self
            .store
            .try_acquire_lock_fenced(JOBS_GC_LOCK_PATH)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(());
        };
        let result = async {
            let mut jobs = self.list().await;
            if jobs.len() <= MAX_PERSISTED_JOBS {
                return Ok(());
            }
            jobs.sort_by_key(|job| job.updated_at);
            let remove = jobs
                .iter()
                .filter(|job| matches!(job.status.as_str(), "succeeded" | "failed" | "cancelled"))
                .take(jobs.len().saturating_sub(MAX_PERSISTED_JOBS))
                .map(|job| job.id.clone())
                .collect::<Vec<_>>();
            if !self
                .store
                .check_lock_fence(JOBS_GC_LOCK_PATH, fence)
                .await
                .map_err(|e| e.to_string())?
            {
                return Err("job GC fencing token is no longer valid".into());
            }
            for id in remove {
                let _ = self.store.delete(&Self::job_path(&id)).await;
            }
            Ok(())
        }
        .await;
        let release = self
            .store
            .release_lock_fenced(JOBS_GC_LOCK_PATH, fence)
            .await
            .map_err(|e| e.to_string());
        match (result, release) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn serve_lock_path(table: &TableIdent) -> String {
    format!(
        "{SERVE_LOCK_PATH_PREFIX}/{}/{}.lock",
        table.namespace, table.name
    )
}

fn cache_scope(table: &TableIdent) -> String {
    format!("{}.{}", table.namespace, table.name)
}

fn query_cache_identity(body: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    body.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

async fn load_table_cached(state: &AppState) -> ApiResult<ailake_catalog::provider::TableMetadata> {
    let scope = cache_scope(&state.table);
    if let Some(bytes) = state
        .cache
        .get(CacheKind::Metadata, &scope, None, "table")
        .await
        .map_err(ApiError::from)?
    {
        if let Ok(meta) = serde_json::from_slice::<ailake_catalog::provider::TableMetadata>(&bytes)
        {
            if let Some(snapshot) = meta.current_snapshot_id {
                state.cache.observe_snapshot(&scope, snapshot).await;
            }
            return Ok(meta);
        }
    }
    let meta = state
        .catalog
        .load_table(&state.table)
        .await
        .map_err(ApiError::from)?;
    let bytes = serde_json::to_vec(&meta).map_err(ApiError::from)?;
    state
        .cache
        .put(
            CacheKind::Metadata,
            &scope,
            None,
            "table",
            bytes::Bytes::from(bytes),
        )
        .await
        .map_err(ApiError::from)?;
    if let Some(snapshot) = meta.current_snapshot_id {
        state.cache.observe_snapshot(&scope, snapshot).await;
    }
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

struct ApiError {
    status: StatusCode,
    message: String,
    retry_after_secs: Option<u64>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": self.message}).to_string();
        let mut response = (self.status, body).into_response();
        if let Some(seconds) = self.retry_after_secs {
            if let Ok(value) = seconds.to_string().parse() {
                response
                    .headers_mut()
                    .insert(axum::http::header::RETRY_AFTER, value);
            }
        }
        response
    }
}

impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
            retry_after_secs: None,
        }
    }
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "missing or invalid bearer token".into(),
            retry_after_secs: None,
        }
    }

    fn too_many_requests() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "server concurrency limit reached".into(),
            retry_after_secs: None,
        }
    }

    fn rate_limited(retry_after_secs: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "rate limit exceeded".into(),
            retry_after_secs: Some(retry_after_secs),
        }
    }

    fn dependency_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            retry_after_secs: Some(1),
        }
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

const MAX_TOP_K: usize = 10_000;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024; // 32 MB
const MAX_INFLIGHT_REQUESTS: usize = 64;

fn bearer_token_matches(expected: &str, provided: &str) -> bool {
    let expected = expected.as_bytes();
    let provided = provided.as_bytes();
    let mut difference = expected.len() ^ provided.len();
    for index in 0..expected.len().max(provided.len()) {
        difference |= usize::from(
            expected.get(index).copied().unwrap_or(0) ^ provided.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

#[derive(Clone, Copy)]
enum RateLimitRequestClass {
    Search,
    Write,
    Other,
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

fn client_ip(headers: &HeaderMap, trust_proxy_headers: bool) -> Option<String> {
    if !trust_proxy_headers {
        return None;
    }
    let parse_header = |name| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(',').next())
            .map(str::trim)
            .and_then(|value| value.parse::<IpAddr>().ok())
            .map(|address| address.to_string())
    };
    parse_header("x-forwarded-for").or_else(|| parse_header("x-real-ip"))
}

async fn authorize_and_acquire(
    state: &AppState,
    headers: &HeaderMap,
    class: RateLimitRequestClass,
) -> ApiResult<tokio::sync::OwnedSemaphorePermit> {
    if let Some(expected) = &state.auth_token {
        let valid = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(|value| {
                value
                    .strip_prefix("Bearer ")
                    .is_some_and(|token| bearer_token_matches(expected.expose_secret(), token))
            })
            .unwrap_or(false);
        if !valid {
            return Err(ApiError::unauthorized());
        }
    }
    let rate_class = match class {
        RateLimitRequestClass::Search => Some(RateLimitClass::Search),
        RateLimitRequestClass::Write => Some(RateLimitClass::Write),
        RateLimitRequestClass::Other => None,
    };
    if let Some(rate_class) = rate_class {
        let ip = client_ip(headers, state.trust_proxy_headers);
        let decision = state
            .rate_limiter
            .check(rate_class, bearer_token(headers), ip.as_deref())
            .await
            .map_err(|error| ApiError::dependency_unavailable(error.to_string()))?;
        if !decision.allowed {
            state.metrics.rate_limited.fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::rate_limited(decision.retry_after_secs));
        }
    }
    state
        .inflight
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::too_many_requests())
}

#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_pruning")]
    pruning_threshold: f32,
}

fn default_top_k() -> usize {
    10
}
fn default_pruning() -> f32 {
    0.8
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Serialize)]
struct SearchResult {
    rank: usize,
    row_id: u64,
    distance: f32,
    file_path: String,
}

#[derive(Deserialize)]
struct WriteRequest {
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    batch_id: Option<String>,
}

#[derive(Serialize)]
struct WriteResponse {
    snapshot_id: i64,
    rows: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct CompactRequest {
    #[serde(default = "default_target_size")]
    target_size: u64,
    #[serde(default = "default_min_files")]
    min_files: usize,
}

fn default_target_size() -> u64 {
    536_870_912
}
fn default_min_files() -> usize {
    4
}

#[derive(Serialize)]
struct CompactResponse {
    message: String,
    compacted_files: usize,
}

#[derive(Serialize)]
struct InfoResponse {
    table: String,
    location: String,
    vector_column: String,
    vector_dim: String,
    vector_metric: String,
    files: usize,
    indexed_files: usize,
    failed_files: usize,
    rows: u64,
    size_bytes: u64,
    estimated_search_budget_bytes: u64,
    snapshot_id: Option<i64>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handle_search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "search");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Search).await?;
    let req: SearchRequest = serde_json::from_str(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))?;

    if req.query.is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    // Read the current catalog snapshot before consulting the query cache. The
    // snapshot id is part of the key, so writes from another process cannot
    // reuse an answer produced by an older table version.
    let meta = state
        .catalog
        .load_table(&state.table)
        .await
        .map_err(ApiError::from)?;
    let snapshot_id = meta.current_snapshot_id.unwrap_or(-1);
    let scope = cache_scope(&state.table);
    state.cache.observe_snapshot(&scope, snapshot_id).await;
    let identity = query_cache_identity(&body);
    if let Some(cached) = state
        .cache
        .get(CacheKind::Query, &scope, Some(snapshot_id), &identity)
        .await
        .map_err(ApiError::from)?
    {
        let cached = String::from_utf8(cached.to_vec()).map_err(ApiError::from)?;
        request_metrics.success();
        return Ok((StatusCode::OK, cached));
    }
    let top_k = req.top_k.clamp(1, MAX_TOP_K);
    let dim = req.query.len() as u32;
    let config = SearchConfig {
        top_k,
        ef_search: top_k.saturating_mul(5),
        pruning_threshold: req.pruning_threshold,
        rerank_factor: None,
        score_fn: None,
        partition_filter: None,
        hybrid: None,
        column_filter: None,
        strict_deletes: state.strict_deletes,
    };

    let results = ailake_query::search(
        &state.table,
        &req.query,
        config,
        &state.policy.column_name,
        dim,
        Arc::clone(&state.catalog) as Arc<dyn CatalogProvider>,
        Arc::clone(&state.store),
    )
    .await
    .map_err(ApiError::from)?;

    // Self-healing: schedule a (rate-limited, non-blocking) probe for foreign files —
    // see `maybe_probe_auto_compact` — without adding latency to this response.
    maybe_probe_auto_compact(&state);

    let resp = SearchResponse {
        results: results
            .iter()
            .enumerate()
            .map(|(i, r)| SearchResult {
                rank: i + 1,
                row_id: r.row_id.0,
                distance: r.distance,
                file_path: r.file_path.clone(),
            })
            .collect(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    state
        .cache
        .put(
            CacheKind::Query,
            &scope,
            Some(snapshot_id),
            &identity,
            bytes::Bytes::from(json.clone()),
        )
        .await
        .map_err(ApiError::from)?;
    request_metrics.success();
    Ok((StatusCode::OK, json))
}

/// Schedule a background foreign-file probe, at most once per
/// `AUTO_COMPACT_CHECK_COOLDOWN_MS`, without blocking the caller.
///
/// `CompactionPlanner::plan()` already prioritizes foreign files (files a
/// generic Iceberg engine rewrote with no knowledge of AI-Lake — see
/// `DataFileEntry::is_foreign`) over the normal size/count thresholds, but
/// only when `ailake compact` actually runs. On a long-running `serve`
/// instance nothing ever calls it unless an operator does so by hand, so a
/// foreign file's O(N) flat-scan degradation (see `flat_scan_unexpected` in
/// `ailake-query/src/scanner.rs`) persists indefinitely. This closes that
/// gap by piggybacking a cheap, rate-limited check on the search path.
fn maybe_probe_auto_compact(state: &Arc<AppState>) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let last = state.auto_compact_last_check_ms.load(Ordering::Acquire);
    if now_ms.saturating_sub(last) < AUTO_COMPACT_CHECK_COOLDOWN_MS {
        return;
    }
    // CAS claims this check window so concurrent requests don't all re-probe at once;
    // losers just skip — the winner's probe covers them.
    if state
        .auto_compact_last_check_ms
        .compare_exchange(last, now_ms, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let state = Arc::clone(state);
    tokio::spawn(async move {
        if let Err(e) = probe_and_auto_compact(&state).await {
            warn!("ailake: auto-compact (foreign file repair) probe failed: {e}");
        }
    });
}

/// Lists files (metadata only — no data file bytes fetched) and, if any is
/// foreign, runs one blocking compaction pass in this background task.
/// `auto_compact_inflight` prevents a second probe from starting a
/// concurrent pass while one is still running.
async fn probe_and_auto_compact(state: &AppState) -> Result<(), String> {
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(|e| e.to_string())?;
    if !files.iter().any(DataFileEntry::is_foreign) {
        return Ok(());
    }
    if state
        .auto_compact_inflight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(()); // a pass triggered by an earlier probe is already running
    }

    let planner = CompactionPlanner::new(CompactionConfig::default());
    let executor = CompactionExecutor::new(Arc::clone(&state.store), state.policy.clone());
    let result = executor
        .run(&planner, &state.table, Arc::clone(&state.catalog), "data")
        .await;
    state.auto_compact_inflight.store(false, Ordering::Release);

    match result {
        Ok(Some(entry)) => {
            info!(
                "ailake: auto-compact repaired foreign file(s) — merged into {}",
                entry.path
            );
            Ok(())
        }
        Ok(None) => Ok(()), // nothing left to compact (raced with a manual compact)
        Err(e) => Err(e.to_string()),
    }
}

async fn handle_write(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "write");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let req: WriteRequest = serde_json::from_str(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))?;

    if req.texts.len() != req.embeddings.len() {
        return Err(ApiError::bad_request(format!(
            "texts length {} != embeddings length {}",
            req.texts.len(),
            req.embeddings.len()
        )));
    }

    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "text",
        arrow_schema::DataType::Utf8,
        false,
    )]));
    let text_arr = arrow_array::StringArray::from(req.texts.clone());
    let batch = arrow_array::RecordBatch::try_new(schema, vec![std::sync::Arc::new(text_arr)])
        .map_err(|e| ApiError::bad_request(format!("RecordBatch error: {e}")))?;

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&state.catalog),
        Arc::clone(&state.store),
        state.policy.clone(),
        state.table.clone(),
        2,
    )
    .await
    .map_err(ApiError::from)?;

    let rows = req.embeddings.len();
    match req.batch_id {
        Some(ref id) => writer
            .write_batch_idempotent(&batch, &req.embeddings, id)
            .await
            .map_err(ApiError::from)?,
        None => writer
            .write_batch(&batch, &req.embeddings)
            .await
            .map_err(ApiError::from)?,
    }
    let snapshot_id = writer.commit().await.map_err(ApiError::from)?;

    let resp = WriteResponse { snapshot_id, rows };
    state
        .cache
        .observe_snapshot(&cache_scope(&state.table), snapshot_id)
        .await;
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

async fn handle_compact(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "compact");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let req: CompactRequest = if body.trim().is_empty() {
        CompactRequest::default()
    } else {
        serde_json::from_str(&body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))?
    };

    let resp = execute_compaction(&state, req).await?;
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

async fn execute_compaction(state: &AppState, req: CompactRequest) -> ApiResult<CompactResponse> {
    let meta = state
        .catalog
        .load_table(&state.table)
        .await
        .map_err(ApiError::from)?;
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(ApiError::from)?;

    let config = CompactionConfig {
        min_files_to_compact: req.min_files,
        target_file_size_bytes: req.target_size,
        index_strategy: Default::default(),
        max_files_per_pass: 20,
    };
    let planner = CompactionPlanner::new(config);
    let to_compact = planner.plan(&files);

    if to_compact.is_empty() {
        return Ok(CompactResponse {
            message: format!("nothing to compact ({} files below threshold)", files.len()),
            compacted_files: 0,
        });
    }

    let n = to_compact.len();
    let executor = CompactionExecutor::new(Arc::clone(&state.store), state.policy.clone());
    let output_path = format!("data/compacted-{}.parquet", unix_seconds());
    let new_entry = executor
        .compact(&to_compact, &output_path)
        .await
        .map_err(ApiError::from)?;

    let compacted_paths: std::collections::HashSet<&str> =
        to_compact.iter().map(|f| f.path.as_str()).collect();
    let mut remaining: Vec<_> = files
        .into_iter()
        .filter(|f| !compacted_paths.contains(f.path.as_str()))
        .collect();
    remaining.push(new_entry);

    let new_snapshot_id = new_snapshot_id();
    state
        .catalog
        .commit_snapshot(
            &state.table,
            NewSnapshot {
                snapshot_id: new_snapshot_id,
                parent_snapshot_id: meta.current_snapshot_id,
                files: remaining,
                operation: SnapshotOperation::Replace,
                iceberg_schema: None,
                extra_properties: std::collections::HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            },
        )
        .await
        .map_err(ApiError::from)?;
    state
        .cache
        .observe_snapshot(&cache_scope(&state.table), new_snapshot_id)
        .await;
    state
        .cache
        .invalidate_scope(&cache_scope(&state.table))
        .await;

    Ok(CompactResponse {
        message: format!("compacted into {output_path}"),
        compacted_files: n,
    })
}

async fn handle_info(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "info");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let meta = load_table_cached(&state).await?;
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(ApiError::from)?;

    let file_count = files.len();
    let row_count: u64 = files.iter().map(|f| f.record_count).sum();
    let size_bytes: u64 = files.iter().map(|f| f.file_size_bytes).sum();
    let estimated_search_budget_bytes = ailake_query::scanner::estimate_file_search_budget_bytes(
        files.iter().map(|file| file.file_size_bytes),
    );
    let ready = files
        .iter()
        .filter(|f| f.index_status == IndexStatus::Ready)
        .count();
    let failed = files
        .iter()
        .filter(|f| f.index_status == IndexStatus::Failed)
        .count();

    let resp = InfoResponse {
        table: format!("{}.{}", state.table.namespace, state.table.name),
        location: meta
            .properties
            .get("ailake.location")
            .cloned()
            .unwrap_or_else(|| meta.location.clone()),
        vector_column: meta
            .properties
            .get("ailake.vector-column")
            .cloned()
            .unwrap_or_else(|| state.policy.column_name.clone()),
        vector_dim: meta
            .properties
            .get("ailake.vector-dim")
            .cloned()
            .unwrap_or_else(|| state.policy.dim.to_string()),
        vector_metric: meta
            .properties
            .get("ailake.vector-metric")
            .cloned()
            .unwrap_or_else(|| "-".to_string()),
        files: file_count,
        indexed_files: ready,
        failed_files: failed,
        rows: row_count,
        size_bytes,
        estimated_search_budget_bytes,
        snapshot_id: meta.current_snapshot_id,
    };
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

async fn handle_health() -> impl IntoResponse {
    (StatusCode::OK, r#"{"ok":true}"#)
}

async fn handle_ready(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "ready");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let _ = load_table_cached(&state).await?;
    request_metrics.success();
    Ok((StatusCode::OK, r#"{"ok":true}"#))
}

async fn handle_metrics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "metrics");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let body = state.metrics.render(
        MAX_INFLIGHT_REQUESTS,
        state.cache.stats().await,
        state.rate_limiter.stats().await,
        state.catalog_breaker.is_open().await,
        state.storage_breaker.is_open().await,
    );
    request_metrics.success();
    Ok((
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    ))
}

async fn handle_submit_compact(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "compact");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let req: CompactRequest = if body.trim().is_empty() {
        CompactRequest::default()
    } else {
        serde_json::from_str(&body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))?
    };
    let job = state
        .jobs
        .submit("compact", req)
        .await
        .map_err(ApiError::from)?;
    state.metrics.jobs_submitted.fetch_add(1, Ordering::Relaxed);
    spawn_compaction_job(Arc::clone(&state), job.clone());
    request_metrics.success();
    Ok((
        StatusCode::ACCEPTED,
        serde_json::to_string(&job).unwrap_or_else(|_| "{}".into()),
    ))
}

fn spawn_compaction_job(state: Arc<AppState>, job: JobRecord) {
    let job_id = job.id.clone();
    let job_manager = Arc::clone(&state.jobs);
    tokio::spawn(async move {
        if job_manager.is_cancel_requested(&job_id).await {
            let _ = job_manager
                .update(
                    &job_id,
                    "cancelled",
                    None,
                    Some("cancelled before start".into()),
                )
                .await;
            return;
        }
        let _ = job_manager.update(&job_id, "running", None, None).await;
        let request = job.request.unwrap_or_default();
        match execute_compaction(&state, request).await {
            Ok(_result) if job_manager.is_cancel_requested(&job_id).await => {
                let _ = job_manager
                    .update(
                        &job_id,
                        "cancelled",
                        None,
                        Some("cancellation requested".into()),
                    )
                    .await;
            }
            Ok(result) => {
                let serialized = serde_json::to_string(&result).unwrap_or_default();
                if job_manager
                    .update(&job_id, "succeeded", Some(serialized), None)
                    .await
                    .is_ok()
                {
                    state.metrics.jobs_succeeded.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(error) => {
                let _ = job_manager
                    .update(&job_id, "failed", None, Some(error.message))
                    .await;
                state.metrics.jobs_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
}

async fn handle_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let job = state.jobs.get(&job_id).await.ok_or_else(|| ApiError {
        status: StatusCode::NOT_FOUND,
        message: format!("job not found: {job_id}"),
        retry_after_secs: None,
    })?;
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&job).unwrap()))
}

async fn handle_jobs(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PaginationQuery>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "jobs");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let jobs = state.jobs.list().await;
    if query.offset.is_some() || query.limit.is_some() {
        let total = jobs.len();
        let (offset, limit) = page_bounds(&query, total);
        let items = jobs
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();
        let has_more = offset.saturating_add(items.len()) < total;
        request_metrics.success();
        return Ok((
            StatusCode::OK,
            serde_json::to_string(&PageResponse {
                items,
                offset,
                limit,
                total,
                has_more,
            })
            .unwrap(),
        ));
    }
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&jobs).unwrap()))
}

async fn handle_cancel_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let job = state
        .jobs
        .cancel(&job_id)
        .await
        .map_err(|message| ApiError {
            status: if message.starts_with("job not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::CONFLICT
            },
            message,
            retry_after_secs: None,
        })?;
    request_metrics.success();
    Ok((StatusCode::ACCEPTED, serde_json::to_string(&job).unwrap()))
}

async fn handle_retry_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let job = state
        .jobs
        .retry(&job_id)
        .await
        .map_err(|message| ApiError {
            status: if message.starts_with("job not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::CONFLICT
            },
            message,
            retry_after_secs: None,
        })?;
    state.metrics.jobs_submitted.fetch_add(1, Ordering::Relaxed);
    spawn_compaction_job(Arc::clone(&state), job.clone());
    request_metrics.success();
    Ok((StatusCode::ACCEPTED, serde_json::to_string(&job).unwrap()))
}

async fn handle_index_jobs(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PaginationQuery>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "index-jobs");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let jobs = list_index_jobs(Arc::clone(&state.store))
        .await
        .map_err(ApiError::from)?;
    if query.offset.is_some() || query.limit.is_some() {
        let total = jobs.len();
        let (offset, limit) = page_bounds(&query, total);
        let items = jobs
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();
        let has_more = offset.saturating_add(items.len()) < total;
        request_metrics.success();
        return Ok((
            StatusCode::OK,
            serde_json::to_string(&PageResponse {
                items,
                offset,
                limit,
                total,
                has_more,
            })
            .unwrap(),
        ));
    }
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&jobs).unwrap()))
}

async fn handle_index_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "index-job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Other).await?;
    let job = load_index_job(Arc::clone(&state.store), &job_id)
        .await
        .map_err(|error| ApiError {
            status: StatusCode::NOT_FOUND,
            message: error.to_string(),
            retry_after_secs: None,
        })?;
    request_metrics.success();
    Ok((StatusCode::OK, serde_json::to_string(&job).unwrap()))
}

async fn handle_cancel_index_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "index-job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let job = handle_for(Arc::clone(&state.store), job_id);
    let record = job.request_cancel().await.map_err(|error| ApiError {
        status: if error.to_string().contains("No such file") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::CONFLICT
        },
        message: error.to_string(),
        retry_after_secs: None,
    })?;
    request_metrics.success();
    Ok((
        StatusCode::ACCEPTED,
        serde_json::to_string(&record).unwrap(),
    ))
}

async fn handle_retry_index_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    headers: HeaderMap,
) -> ApiResult<impl IntoResponse> {
    let mut request_metrics = RequestMetricGuard::new(&state.metrics, "index-job");
    let _permit = authorize_and_acquire(&state, &headers, RateLimitRequestClass::Write).await?;
    let job = handle_for(Arc::clone(&state.store), job_id);
    let record = job.retry().await.map_err(|error| ApiError {
        status: if error.to_string().contains("No such file") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::CONFLICT
        },
        message: error.to_string(),
        retry_after_secs: None,
    })?;
    resume_index_jobs(
        Arc::clone(&state.store),
        Arc::clone(&state.catalog),
        state.policy.clone(),
        state.table.clone(),
    )
    .await
    .map_err(ApiError::from)?;
    request_metrics.success();
    Ok((
        StatusCode::ACCEPTED,
        serde_json::to_string(&record).unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) async fn run(
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: TableIdent,
    policy: VectorStoragePolicy,
    config: ServeConfig,
) -> Result<(), String> {
    let ServeConfig {
        host,
        port,
        auth_token,
        cache_url,
        cache_max_bytes,
        cache_ttl_secs,
        strict_deletes,
        rate_limit_url,
        rate_window_secs,
        search_quota_token,
        write_quota_token,
        search_quota_ip,
        write_quota_ip,
        trust_proxy_headers,
        rate_limit_fail_closed,
        circuit_failure_threshold,
        circuit_cooldown_secs,
    } = config;
    validate_bind_auth(&host, auth_token.is_some())?;
    let table_meta = catalog
        .load_table(&table)
        .await
        .map_err(|e| format!("catalog preflight failed: {e}"))?;
    store
        .validate_location(&table_meta.location)
        .map_err(|e| format!("catalog/storage location mismatch: {e}"))?;
    let serve_lock_path = serve_lock_path(&table);
    let serve_fence = store
        .try_acquire_lock_fenced(&serve_lock_path)
        .await
        .map_err(|e| format!("serve instance lock failed: {e}"))?
        .ok_or_else(|| {
            format!(
                "another ailake serve instance already owns table {}.{}",
                table.namespace, table.name
            )
        })?;
    let catalog_breaker = match CircuitBreaker::new(
        circuit_failure_threshold,
        Duration::from_secs(circuit_cooldown_secs.max(1)),
    ) {
        Ok(breaker) => breaker,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(format!("catalog circuit configuration failed: {error}"));
        }
    };
    let storage_breaker = match CircuitBreaker::new(
        circuit_failure_threshold,
        Duration::from_secs(circuit_cooldown_secs.max(1)),
    ) {
        Ok(breaker) => breaker,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(format!("storage circuit configuration failed: {error}"));
        }
    };
    let guarded_catalog: Arc<dyn CatalogProvider> = Arc::new(CircuitBreakingCatalog::new(
        Arc::clone(&catalog),
        Arc::clone(&catalog_breaker),
    ));
    let guarded_store: Arc<dyn Store> = Arc::new(CircuitBreakingStore::new(
        Arc::clone(&store),
        Arc::clone(&storage_breaker),
    ));
    let cache = match CacheManager::new(CacheConfig {
        namespace: "ailake".into(),
        max_bytes: cache_max_bytes,
        ttl: Duration::from_secs(cache_ttl_secs.max(1)),
        redis_url: cache_url.clone(),
    }) {
        Ok(cache) => cache,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(format!("cache initialization failed: {error}"));
        }
    };
    let rate_limiter = match RateLimiter::new(RateLimitConfig {
        namespace: "ailake:rate".into(),
        window: Duration::from_secs(rate_window_secs.max(1)),
        search_per_token: search_quota_token,
        write_per_token: write_quota_token,
        search_per_ip: search_quota_ip,
        write_per_ip: write_quota_ip,
        redis_url: rate_limit_url.or_else(|| cache_url.clone()),
        fail_closed: rate_limit_fail_closed,
    }) {
        Ok(limiter) => limiter,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(format!("rate limiter initialization failed: {error}"));
        }
    };
    if let Err(error) = resume_index_jobs(
        Arc::clone(&guarded_store),
        Arc::clone(&guarded_catalog),
        policy.clone(),
        table.clone(),
    )
    .await
    {
        let _ = store
            .release_lock_fenced(&serve_lock_path, serve_fence)
            .await;
        return Err(format!("index job recovery failed: {error}"));
    }
    let jobs = match JobManager::load(Arc::clone(&store)).await {
        Ok(jobs) => jobs,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(error);
        }
    };
    let query_store: Arc<dyn Store> = Arc::new(CachingStore::new(
        Arc::clone(&guarded_store),
        Arc::clone(&cache),
        cache_scope(&table),
    ));
    let state = Arc::new(AppState {
        catalog: guarded_catalog,
        store: query_store,
        table,
        policy,
        auto_compact_inflight: Arc::new(AtomicBool::new(false)),
        auto_compact_last_check_ms: Arc::new(AtomicU64::new(0)),
        auth_token,
        inflight: Arc::new(Semaphore::new(MAX_INFLIGHT_REQUESTS)),
        cache,
        rate_limiter,
        catalog_breaker,
        storage_breaker,
        strict_deletes,
        trust_proxy_headers,
        metrics: Arc::new(ServerMetrics::default()),
        jobs,
    });

    // Resume jobs that were durably queued before the previous process exited.
    for job in state.jobs.list().await {
        if job.status == "queued" && job.kind == "compact" {
            spawn_compaction_job(Arc::clone(&state), job);
        }
    }

    let app = build_router(Arc::clone(&state));

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(error) => {
            let _ = store
                .release_lock_fenced(&serve_lock_path, serve_fence)
                .await;
            return Err(format!("bind {addr}: {error}"));
        }
    };

    eprintln!("ailake server listening on http://{addr}");
    if state.auth_token.is_none() {
        eprintln!("WARNING: no authentication — use --auth-token for non-local deployments");
    }
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let renewal_store = Arc::clone(&store);
    let renewal_path = serve_lock_path.clone();
    let renewal_task = tokio::spawn(async move {
        let mut shutdown_tx = Some(shutdown_tx);
        loop {
            tokio::time::sleep(SERVE_LEASE_RENEW_INTERVAL).await;
            match renewal_store
                .renew_lock_fenced(&renewal_path, serve_fence)
                .await
            {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    if let Some(tx) = shutdown_tx.take() {
                        let _ = tx.send(());
                    }
                    break;
                }
            }
        }
    });
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
            eprintln!("ailake serve lease lost; shutting down");
        })
        .await
        .map_err(|e| e.to_string());
    renewal_task.abort();
    let _ = store
        .release_lock_fenced(&serve_lock_path, serve_fence)
        .await;
    result
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn validate_bind_auth(host: &str, has_auth: bool) -> Result<(), String> {
    if !has_auth && !is_loopback_host(host) {
        return Err(format!(
            "refusing to bind unauthenticated server to non-loopback host '{host}'; configure --auth-token or bind to localhost"
        ));
    }
    Ok(())
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/search", post(handle_search))
        .route("/write", post(handle_write))
        .route("/compact", post(handle_compact))
        .route("/info", get(handle_info))
        .route("/metrics", get(handle_metrics))
        .route("/healthz", get(handle_health))
        .route("/readyz", get(handle_ready))
        .route("/jobs/compact", post(handle_submit_compact))
        .route("/jobs", get(handle_jobs))
        .route("/jobs/:job_id/retry", post(handle_retry_job))
        .route("/jobs/:job_id/cancel", post(handle_cancel_job))
        .route("/jobs/:job_id", get(handle_job))
        .route("/index-jobs", get(handle_index_jobs))
        .route("/index-jobs/:job_id/retry", post(handle_retry_index_job))
        .route("/index-jobs/:job_id/cancel", post(handle_cancel_index_job))
        .route("/index-jobs/:job_id", get(handle_index_job))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_tokens_match_without_early_byte_exit() {
        assert!(bearer_token_matches("secret-token", "secret-token"));
        assert!(!bearer_token_matches("secret-token", "secret-tokeN"));
        assert!(!bearer_token_matches("secret-token", "short"));
    }

    #[test]
    fn unauthenticated_bind_is_limited_to_loopback_hosts() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.internal"));
        assert!(validate_bind_auth("127.0.0.1", false).is_ok());
        assert!(validate_bind_auth("0.0.0.0", true).is_ok());
        assert!(validate_bind_auth("0.0.0.0", false).is_err());
    }

    #[test]
    fn proxy_ip_headers_are_opt_in_and_must_contain_an_ip_address() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.7, 10.0.0.2".parse().unwrap());
        assert_eq!(client_ip(&headers, false), None);
        assert_eq!(client_ip(&headers, true).as_deref(), Some("203.0.113.7"));

        headers.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        headers.insert("x-real-ip", "2001:db8::1".parse().unwrap());
        assert_eq!(client_ip(&headers, true).as_deref(), Some("2001:db8::1"));
    }
    use ailake_catalog::HadoopCatalog;
    use ailake_core::{VectorMetric, VectorPrecision};
    use ailake_query::TableWriter;
    use ailake_store::LocalStore;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use axum::body::Body;
    use axum::http::Request;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn test_policy() -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim: 4,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
        }
    }

    fn test_state(catalog: Arc<dyn CatalogProvider>, store: Arc<dyn Store>) -> AppState {
        AppState {
            catalog,
            store: Arc::clone(&store),
            table: TableIdent::new("default", "table"),
            policy: test_policy(),
            auto_compact_inflight: Arc::new(AtomicBool::new(false)),
            auto_compact_last_check_ms: Arc::new(AtomicU64::new(0)),
            auth_token: None,
            inflight: Arc::new(Semaphore::new(MAX_INFLIGHT_REQUESTS)),
            cache: CacheManager::new(CacheConfig::default()).unwrap(),
            rate_limiter: RateLimiter::new(RateLimitConfig::default()).unwrap(),
            catalog_breaker: CircuitBreaker::new(5, Duration::from_secs(15)).unwrap(),
            storage_breaker: CircuitBreaker::new(5, Duration::from_secs(15)).unwrap(),
            strict_deletes: false,
            trust_proxy_headers: false,
            metrics: Arc::new(ServerMetrics::default()),
            jobs: Arc::new(JobManager { store }),
        }
    }

    /// Real round-trip, no mocks: writes a genuine AI-Lake file via `TableWriter`,
    /// then overwrites its manifest entry to strip `centroid_b64` — the same
    /// after-the-fact state a generic Iceberg engine's `OPTIMIZE`/`rewrite_data_files`
    /// leaves behind (see `DataFileEntry::is_foreign`, `CompactionPlanner::plan`).
    /// `read_parquet()` never touches the AILK footer (see `compaction.rs`
    /// `read_files_parallel`), so this reproduces exactly what `probe_and_auto_compact`
    /// has to detect and repair without needing a second, footerless physical file.
    #[tokio::test]
    async fn probe_and_auto_compact_repairs_foreign_file() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");
        let policy = test_policy();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![0i32, 1]))]).unwrap();
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let mut writer = TableWriter::create_or_open(
            catalog.clone(),
            store.clone(),
            policy.clone(),
            table.clone(),
            2,
        )
        .await
        .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files.len(), 1);
        assert!(
            !files[0].is_foreign(),
            "sanity: TableWriter must produce a native (non-foreign) entry"
        );

        // Simulate the foreign rewrite: same physical bytes, manifest entry stripped
        // of the AI-Lake-only metadata a generic engine's rewrite would never populate.
        let mut foreign_entry = files[0].clone();
        foreign_entry.centroid_b64 = None;
        foreign_entry.hnsw_offset = None;
        foreign_entry.hnsw_len = None;
        let parent_snapshot_id = catalog
            .load_table(&table)
            .await
            .unwrap()
            .current_snapshot_id;
        catalog
            .commit_snapshot(
                &table,
                NewSnapshot {
                    snapshot_id: new_snapshot_id(),
                    parent_snapshot_id,
                    files: vec![foreign_entry],
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                    equality_delete_files: vec![],
                },
            )
            .await
            .unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert!(
            files.iter().any(DataFileEntry::is_foreign),
            "setup sanity: file must now read as foreign"
        );

        let state = test_state(catalog.clone(), store.clone());
        probe_and_auto_compact(&state).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files_after.len(),
            1,
            "auto-compact should merge the single foreign file into one repaired file"
        );
        assert!(
            !files_after[0].is_foreign(),
            "repaired file must carry a real centroid again"
        );
        assert_eq!(
            files_after[0].record_count, 2,
            "both rows must survive the repair"
        );
    }

    /// No foreign files present — the probe must not touch the catalog beyond the
    /// one `list_files` check (no compaction pass, no snapshot commit).
    #[tokio::test]
    async fn probe_and_auto_compact_is_noop_when_nothing_foreign() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");
        let policy = test_policy();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![0i32]))]).unwrap();
        let mut writer =
            TableWriter::create_or_open(catalog.clone(), store.clone(), policy, table.clone(), 2)
                .await
                .unwrap();
        writer
            .write_batch(&batch, &[vec![1.0, 0.0, 0.0, 0.0]])
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let state = test_state(catalog.clone(), store.clone());
        probe_and_auto_compact(&state).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files_after.len(), 1, "no compaction should have run");
        assert!(!state.auto_compact_inflight.load(Ordering::Acquire));
    }

    /// `maybe_probe_auto_compact` must skip re-listing files (and thus skip spawning
    /// a probe task) when called again inside the cooldown window.
    #[tokio::test]
    async fn maybe_probe_auto_compact_respects_cooldown() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let state = Arc::new(test_state(catalog, store));

        // First call claims the window (sets the timestamp to "now").
        maybe_probe_auto_compact(&state);
        let first = state.auto_compact_last_check_ms.load(Ordering::Acquire);
        assert!(first > 0, "first call must claim the cooldown window");

        // Immediate second call must NOT reset the timestamp (still inside cooldown).
        maybe_probe_auto_compact(&state);
        let second = state.auto_compact_last_check_ms.load(Ordering::Acquire);
        assert_eq!(first, second, "second call within cooldown must be a no-op");
    }

    #[tokio::test]
    async fn http_health_metrics_and_auth_contract() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let mut state = test_state(catalog, store);
        state.auth_token = Some("test-token".into());
        let app = build_router(Arc::new(state));

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("ailake_search_flat_scan_unexpected_files_total"));
        assert!(body.contains("ailake_search_flat_scan_elapsed_micros_total"));
    }

    #[tokio::test]
    async fn query_cache_is_snapshot_scoped_and_invalidated() {
        let cache = CacheManager::new(CacheConfig::default()).unwrap();
        cache
            .put(
                CacheKind::Query,
                "default.table",
                Some(1),
                "same-query",
                bytes::Bytes::from_static(b"cached-result"),
            )
            .await
            .unwrap();
        assert_eq!(
            cache
                .get(CacheKind::Query, "default.table", Some(1), "same-query")
                .await
                .unwrap(),
            Some(bytes::Bytes::from_static(b"cached-result"))
        );
        cache.invalidate_snapshot("default.table", 2).await;
        assert!(cache
            .get(CacheKind::Query, "default.table", Some(1), "same-query")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn jobs_support_list_cancel_and_retry() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let manager = JobManager::load(store.clone()).await.unwrap();
        let job = manager
            .submit("compact", CompactRequest::default())
            .await
            .unwrap();
        assert_eq!(manager.list().await.len(), 1);
        let cancelled = manager.cancel(&job.id).await.unwrap();
        assert_eq!(cancelled.status, "cancelled");
        let retried = manager.retry(&job.id).await.unwrap();
        assert_eq!(retried.status, "queued");
        let reloaded = JobManager::load(store).await.unwrap();
        assert_eq!(reloaded.get(&job.id).await.unwrap().status, "queued");
    }

    #[tokio::test]
    async fn concurrent_job_managers_merge_into_shared_registry() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let first = JobManager::load(store.clone()).await.unwrap();
        let second = JobManager::load(store.clone()).await.unwrap();

        let (left, right) = tokio::join!(
            first.submit("compact", CompactRequest::default()),
            second.submit("compact", CompactRequest::default()),
        );
        left.unwrap();
        right.unwrap();

        let reloaded = JobManager::load(store).await.unwrap();
        assert_eq!(reloaded.list().await.len(), 2);
    }

    #[tokio::test]
    async fn legacy_aggregate_registry_is_migrated_to_per_job_records() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let record = JobRecord {
            id: Uuid::new_v4().to_string(),
            kind: "compact".into(),
            status: "queued".into(),
            created_at: unix_seconds(),
            updated_at: unix_seconds(),
            error: None,
            result: None,
            request: Some(CompactRequest::default()),
            cancel_requested: false,
        };
        let mut legacy = HashMap::new();
        legacy.insert(record.id.clone(), record.clone());
        store
            .put(
                LEGACY_JOBS_PATH,
                bytes::Bytes::from(serde_json::to_vec(&legacy).unwrap()),
            )
            .await
            .unwrap();

        let manager = JobManager::load(store.clone()).await.unwrap();
        assert_eq!(manager.get(&record.id).await.unwrap().id, record.id);
        assert!(store
            .exists(&JobManager::job_path(&record.id))
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn serve_lock_is_exclusive_per_table() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let table = TableIdent::new("default", "table");
        let path = serve_lock_path(&table);
        let first = store.try_acquire_lock_fenced(&path).await.unwrap();
        assert_eq!(first, Some(0));
        assert_eq!(store.try_acquire_lock_fenced(&path).await.unwrap(), None);
        store
            .release_lock_fenced(&path, first.unwrap())
            .await
            .unwrap();
        assert_eq!(store.try_acquire_lock_fenced(&path).await.unwrap(), Some(0));
    }

    #[test]
    fn pagination_bounds_are_safe_and_clamped() {
        let (offset, limit) = page_bounds(
            &PaginationQuery {
                offset: Some(99),
                limit: Some(10_000),
            },
            12,
        );
        assert_eq!((offset, limit), (12, MAX_PAGE_LIMIT));

        let (offset, limit) = page_bounds(&PaginationQuery::default(), 0);
        assert_eq!((offset, limit), (0, DEFAULT_PAGE_LIMIT));
    }
}
