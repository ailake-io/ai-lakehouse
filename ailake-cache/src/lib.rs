//! Production cache primitives for AI-Lake.
//!
//! The cache is deliberately independent from the catalog. Query keys carry a
//! snapshot id, so an old result is never reused after a new snapshot is
//! observed. Redis/Valkey is an optional shared tier; the local tier remains a
//! bounded, process-wide LRU so a remote cache outage does not stop queries.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ailake_core::{AilakeError, AilakeResult};
use ailake_store::Store;
use async_trait::async_trait;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

#[cfg(feature = "redis")]
use redis::{AsyncCommands, Script};

/// Logical cache partitions. The memory limit is shared by all partitions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheKind {
    Query,
    Metadata,
    Index,
}

impl CacheKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Metadata => "metadata",
            Self::Index => "index",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub namespace: String,
    pub max_bytes: usize,
    pub ttl: Duration,
    pub redis_url: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            namespace: "ailake".into(),
            max_bytes: 256 * 1024 * 1024,
            ttl: Duration::from_secs(2),
            redis_url: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub hits_total: u64,
    pub misses_total: u64,
    pub inserts_total: u64,
    pub evictions_total: u64,
    pub invalidations_total: u64,
    pub redis_hits_total: u64,
    pub redis_errors_total: u64,
    pub entries: usize,
    pub bytes_in_use: usize,
    pub bytes_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RateLimitClass {
    Search,
    Write,
}

#[derive(Clone, Debug)]
pub struct RateLimitConfig {
    pub namespace: String,
    pub window: Duration,
    pub search_per_token: u64,
    pub write_per_token: u64,
    pub search_per_ip: u64,
    pub write_per_ip: u64,
    pub redis_url: Option<String>,
    /// If true, a Redis outage rejects requests instead of failing open.
    pub fail_closed: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            namespace: "ailake:rate".into(),
            window: Duration::from_secs(60),
            search_per_token: 600,
            write_per_token: 60,
            search_per_ip: 1_200,
            write_per_ip: 120,
            redis_url: None,
            fail_closed: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RateLimitStats {
    pub allowed_total: u64,
    pub limited_total: u64,
    pub backend_errors_total: u64,
    pub redis_requests_total: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RateLimitDecision {
    pub allowed: bool,
    pub retry_after_secs: u64,
    pub dimension: Option<String>,
}

#[derive(Debug)]
pub enum RateLimitError {
    Configuration(String),
    Backend(String),
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration(e) => write!(f, "rate limit configuration error: {e}"),
            Self::Backend(e) => write!(f, "rate limit backend error: {e}"),
        }
    }
}

impl std::error::Error for RateLimitError {}

#[derive(Debug)]
pub enum CacheError {
    Configuration(String),
    Serialization(String),
    Backend(String),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration(e) => write!(f, "cache configuration error: {e}"),
            Self::Serialization(e) => write!(f, "cache serialization error: {e}"),
            Self::Backend(e) => write!(f, "cache backend error: {e}"),
        }
    }
}

impl std::error::Error for CacheError {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LocalKey(String);

#[derive(Clone)]
struct Entry {
    value: Bytes,
    expires_at: Instant,
    size: usize,
}

#[derive(Default)]
struct LocalState {
    entries: HashMap<LocalKey, Entry>,
    order: VecDeque<LocalKey>,
    bytes: usize,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
    invalidations: AtomicU64,
    redis_hits: AtomicU64,
    redis_errors: AtomicU64,
}

#[derive(Default)]
struct RateCounters {
    allowed: AtomicU64,
    limited: AtomicU64,
    backend_errors: AtomicU64,
    redis_requests: AtomicU64,
}

#[cfg(feature = "redis")]
struct RedisBackend {
    client: redis::Client,
}

#[cfg(feature = "redis")]
struct RateRedisBackend {
    client: redis::Client,
}

struct LocalRateWindow {
    started: Instant,
    count: u64,
}

/// Fixed-window quota limiter. Redis uses atomic INCR+EXPIRE, so all server
/// instances sharing the same namespace enforce the same counters.
pub struct RateLimiter {
    config: RateLimitConfig,
    local: Mutex<HashMap<String, LocalRateWindow>>,
    counters: RateCounters,
    #[cfg(feature = "redis")]
    redis: Option<RateRedisBackend>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Result<Arc<Self>, RateLimitError> {
        if config.window.is_zero() {
            return Err(RateLimitError::Configuration(
                "window must be greater than zero".into(),
            ));
        }
        #[cfg(feature = "redis")]
        let redis = config
            .redis_url
            .as_deref()
            .map(redis::Client::open)
            .transpose()
            .map_err(|e| RateLimitError::Configuration(e.to_string()))?
            .map(|client| RateRedisBackend { client });
        #[cfg(not(feature = "redis"))]
        if config.redis_url.is_some() {
            return Err(RateLimitError::Configuration(
                "Redis/Valkey URL requires the ailake-cache `redis` feature".into(),
            ));
        }
        Ok(Arc::new(Self {
            config,
            local: Mutex::new(HashMap::new()),
            counters: RateCounters::default(),
            #[cfg(feature = "redis")]
            redis,
        }))
    }

    fn limits(&self, class: RateLimitClass) -> (u64, u64) {
        match class {
            RateLimitClass::Search => (self.config.search_per_token, self.config.search_per_ip),
            RateLimitClass::Write => (self.config.write_per_token, self.config.write_per_ip),
        }
    }

    pub async fn check(
        &self,
        class: RateLimitClass,
        token: Option<&str>,
        ip: Option<&str>,
    ) -> Result<RateLimitDecision, RateLimitError> {
        let (token_limit, ip_limit) = self.limits(class);
        let mut dimensions = Vec::new();
        if token_limit > 0 {
            if let Some(token) = token {
                dimensions.push((format!("token:{}", digest(token)), token_limit));
            }
        }
        if ip_limit > 0 {
            if let Some(ip) = ip {
                dimensions.push((format!("ip:{}", digest(ip)), ip_limit));
            }
        }
        if dimensions.is_empty() {
            self.counters.allowed.fetch_add(1, Ordering::Relaxed);
            return Ok(RateLimitDecision {
                allowed: true,
                retry_after_secs: 0,
                dimension: None,
            });
        }

        #[cfg(feature = "redis")]
        if let Some(redis) = &self.redis {
            self.counters.redis_requests.fetch_add(1, Ordering::Relaxed);
            let mut denied = None;
            for (key, limit) in &dimensions {
                match redis_increment(redis, &self.config.namespace, key, self.config.window).await
                {
                    Ok(count) if count > i64::try_from(*limit).unwrap_or(i64::MAX) => {
                        denied = Some(key.clone());
                    }
                    Ok(_) => {}
                    Err(error) => {
                        self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                        if self.config.fail_closed {
                            return Err(RateLimitError::Backend(error.to_string()));
                        }
                        denied = None;
                        break;
                    }
                }
            }
            if let Some(dimension) = denied {
                self.counters.limited.fetch_add(1, Ordering::Relaxed);
                return Ok(RateLimitDecision {
                    allowed: false,
                    retry_after_secs: self.config.window.as_secs().max(1),
                    dimension: Some(dimension),
                });
            }
            self.counters.allowed.fetch_add(1, Ordering::Relaxed);
            return Ok(RateLimitDecision {
                allowed: true,
                retry_after_secs: 0,
                dimension: None,
            });
        }

        let mut state = self.local.lock().await;
        let now = Instant::now();
        // A caller can present a new token/IP on every request. Reap expired
        // identities on the hot path so the local fallback stays bounded by
        // identities seen during the active window rather than process uptime.
        state.retain(|_, window| now.duration_since(window.started) < self.config.window);
        let mut denied = None;
        for (key, limit) in &dimensions {
            let window = state.entry(key.clone()).or_insert(LocalRateWindow {
                started: now,
                count: 0,
            });
            if now.duration_since(window.started) >= self.config.window {
                window.started = now;
                window.count = 0;
            }
            window.count = window.count.saturating_add(1);
            if window.count > *limit {
                denied = Some(key.clone());
            }
        }
        if let Some(dimension) = denied {
            self.counters.limited.fetch_add(1, Ordering::Relaxed);
            return Ok(RateLimitDecision {
                allowed: false,
                retry_after_secs: self.config.window.as_secs().max(1),
                dimension: Some(dimension),
            });
        }
        self.counters.allowed.fetch_add(1, Ordering::Relaxed);
        Ok(RateLimitDecision {
            allowed: true,
            retry_after_secs: 0,
            dimension: None,
        })
    }

    pub async fn stats(&self) -> RateLimitStats {
        RateLimitStats {
            allowed_total: self.counters.allowed.load(Ordering::Relaxed),
            limited_total: self.counters.limited.load(Ordering::Relaxed),
            backend_errors_total: self.counters.backend_errors.load(Ordering::Relaxed),
            redis_requests_total: self.counters.redis_requests.load(Ordering::Relaxed),
        }
    }
}

fn digest(value: &str) -> String {
    // A stable cryptographic digest gives every server the same Redis key and
    // prevents chosen inputs from cheaply colliding with another identity.
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

#[cfg(feature = "redis")]
async fn redis_increment(
    backend: &RateRedisBackend,
    namespace: &str,
    dimension: &str,
    window: Duration,
) -> Result<i64, redis::RedisError> {
    let mut connection = backend.client.get_multiplexed_async_connection().await?;
    let key = format!("{namespace}:{dimension}");
    Script::new(
        "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],ARGV[1]) end; return n",
    )
    .key(key)
    .arg(window.as_secs().max(1))
    .invoke_async(&mut connection)
    .await
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BreakerState {
    Closed { failures: u32 },
    Open { opened_at: Instant },
    HalfOpen,
}

/// Async circuit breaker shared by a dependency wrapper. After the threshold
/// of consecutive failures, calls fail fast until the cooldown elapses; one
/// probe is then allowed to decide whether the circuit closes again.
pub struct CircuitBreaker {
    state: Mutex<BreakerState>,
    failure_threshold: u32,
    cooldown: Duration,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown: Duration) -> Result<Arc<Self>, CacheError> {
        if failure_threshold == 0 || cooldown.is_zero() {
            return Err(CacheError::Configuration(
                "circuit breaker threshold and cooldown must be greater than zero".into(),
            ));
        }
        Ok(Arc::new(Self {
            state: Mutex::new(BreakerState::Closed { failures: 0 }),
            failure_threshold,
            cooldown,
        }))
    }

    pub async fn execute<T, F>(&self, open_error: AilakeError, operation: F) -> AilakeResult<T>
    where
        F: std::future::Future<Output = AilakeResult<T>>,
    {
        {
            let mut state = self.state.lock().await;
            match *state {
                BreakerState::Closed { .. } => {}
                BreakerState::Open { opened_at } if opened_at.elapsed() >= self.cooldown => {
                    *state = BreakerState::HalfOpen
                }
                BreakerState::Open { .. } | BreakerState::HalfOpen => return Err(open_error),
            }
        }
        match operation.await {
            Ok(value) => {
                *self.state.lock().await = BreakerState::Closed { failures: 0 };
                Ok(value)
            }
            Err(error) => {
                let mut state = self.state.lock().await;
                let failures = match *state {
                    BreakerState::HalfOpen => self.failure_threshold,
                    BreakerState::Closed { failures } => failures.saturating_add(1),
                    BreakerState::Open { .. } => self.failure_threshold,
                };
                *state = if failures >= self.failure_threshold {
                    BreakerState::Open {
                        opened_at: Instant::now(),
                    }
                } else {
                    BreakerState::Closed { failures }
                };
                Err(error)
            }
        }
    }

    pub async fn is_open(&self) -> bool {
        matches!(*self.state.lock().await, BreakerState::Open { .. })
    }
}

/// Two-tier cache. Redis is best-effort: failures increment a metric and fall
/// back to the object store, preserving availability while exposing the fault.
pub struct CacheManager {
    config: CacheConfig,
    local: Mutex<LocalState>,
    snapshots: Mutex<HashMap<String, i64>>,
    counters: Counters,
    #[cfg(feature = "redis")]
    redis: Option<RedisBackend>,
}

impl CacheManager {
    pub fn new(config: CacheConfig) -> Result<Arc<Self>, CacheError> {
        if config.max_bytes == 0 {
            return Err(CacheError::Configuration(
                "max_bytes must be greater than zero".into(),
            ));
        }
        #[cfg(feature = "redis")]
        let redis = config
            .redis_url
            .as_deref()
            .map(redis::Client::open)
            .transpose()
            .map_err(|e| CacheError::Configuration(e.to_string()))?
            .map(|client| RedisBackend { client });
        #[cfg(not(feature = "redis"))]
        if config.redis_url.is_some() {
            return Err(CacheError::Configuration(
                "Redis/Valkey URL requires the ailake-cache `redis` feature".into(),
            ));
        }
        Ok(Arc::new(Self {
            config,
            local: Mutex::new(LocalState::default()),
            snapshots: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            #[cfg(feature = "redis")]
            redis,
        }))
    }

    fn key(&self, kind: CacheKind, scope: &str, snapshot: Option<i64>, identity: &str) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.config.namespace,
            kind.as_str(),
            scope,
            snapshot.map_or_else(|| "current".into(), |id| id.to_string()),
            identity
        )
    }

    pub async fn get(
        &self,
        kind: CacheKind,
        scope: &str,
        snapshot: Option<i64>,
        identity: &str,
    ) -> Result<Option<Bytes>, CacheError> {
        let key = self.key(kind, scope, snapshot, identity);
        {
            let mut state = self.local.lock().await;
            if let Some(entry) = state.entries.get(&LocalKey(key.clone())).cloned() {
                if entry.expires_at > Instant::now() {
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                    state.order.retain(|item| item.0 != key);
                    state.order.push_back(LocalKey(key));
                    return Ok(Some(entry.value));
                }
                remove_local(&mut state, &key);
            }
        }

        #[cfg(feature = "redis")]
        if let Some(redis) = &self.redis {
            match redis_get(redis, &key).await {
                Ok(Some(value)) => {
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                    self.counters.redis_hits.fetch_add(1, Ordering::Relaxed);
                    self.insert_local(LocalKey(key), value.clone()).await;
                    return Ok(Some(value));
                }
                Ok(None) => {}
                Err(_) => {
                    self.counters.redis_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    pub async fn put(
        &self,
        kind: CacheKind,
        scope: &str,
        snapshot: Option<i64>,
        identity: &str,
        value: Bytes,
    ) -> Result<(), CacheError> {
        let key = self.key(kind, scope, snapshot, identity);
        self.insert_local(LocalKey(key.clone()), value.clone())
            .await;
        #[cfg(feature = "redis")]
        if let Some(redis) = &self.redis {
            if redis_put(redis, &key, &value, self.config.ttl)
                .await
                .is_err()
            {
                self.counters.redis_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.counters.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn insert_local(&self, key: LocalKey, value: Bytes) {
        let mut state = self.local.lock().await;
        if let Some(old) = state.entries.remove(&key) {
            state.bytes = state.bytes.saturating_sub(old.size);
        }
        state.order.retain(|item| item != &key);
        let size = value.len();
        state.entries.insert(
            key.clone(),
            Entry {
                value,
                expires_at: Instant::now() + self.config.ttl,
                size,
            },
        );
        state.order.push_back(key);
        state.bytes = state.bytes.saturating_add(size);
        while state.bytes > self.config.max_bytes {
            let Some(oldest) = state.order.pop_front() else {
                break;
            };
            if let Some(old) = state.entries.remove(&oldest) {
                state.bytes = state.bytes.saturating_sub(old.size);
                self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Drops local entries from older snapshots and makes current metadata
    /// stale. Redis keys remain versioned and expire naturally, avoiding a
    /// cluster-wide SCAN on every write.
    pub async fn invalidate_snapshot(&self, scope: &str, snapshot: i64) {
        let prefix = format!("{}:", scope);
        let mut state = self.local.lock().await;
        let keys: Vec<String> = state
            .entries
            .keys()
            .filter(|key| key.0.contains(&prefix) && !key.0.contains(&format!(":{snapshot}:")))
            .map(|key| key.0.clone())
            .collect();
        for key in keys {
            remove_local(&mut state, &key);
        }
        self.counters.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the snapshot observed by the catalog and evicts local entries
    /// from older versions. File/index reads can then use the same versioned
    /// key as query results without making the Store depend on the catalog.
    pub async fn observe_snapshot(&self, scope: &str, snapshot: i64) {
        self.snapshots
            .lock()
            .await
            .insert(scope.to_string(), snapshot);
        self.invalidate_snapshot(scope, snapshot).await;
    }

    async fn snapshot_for(&self, scope: &str) -> Option<i64> {
        self.snapshots.lock().await.get(scope).copied()
    }

    pub async fn invalidate_scope(&self, scope: &str) {
        let prefix = format!(":{scope}:");
        let mut state = self.local.lock().await;
        let keys: Vec<String> = state
            .entries
            .keys()
            .filter(|key| key.0.contains(&prefix))
            .map(|key| key.0.clone())
            .collect();
        for key in keys {
            remove_local(&mut state, &key);
        }
        drop(state);
        #[cfg(feature = "redis")]
        if let Some(redis) = &self.redis {
            let current_metadata = self.key(CacheKind::Metadata, scope, None, "table");
            if redis_del(redis, &current_metadata).await.is_err() {
                self.counters.redis_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.counters.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn invalidate_identity(&self, scope: &str, identity: &str) {
        let marker = format!(":{scope}:");
        let mut state = self.local.lock().await;
        let keys: Vec<String> = state
            .entries
            .keys()
            .filter(|key| key.0.contains(&marker) && key.0.ends_with(identity))
            .map(|key| key.0.clone())
            .collect();
        for key in keys {
            remove_local(&mut state, &key);
        }
    }

    pub async fn stats(&self) -> CacheStats {
        let state = self.local.lock().await;
        CacheStats {
            hits_total: self.counters.hits.load(Ordering::Relaxed),
            misses_total: self.counters.misses.load(Ordering::Relaxed),
            inserts_total: self.counters.inserts.load(Ordering::Relaxed),
            evictions_total: self.counters.evictions.load(Ordering::Relaxed),
            invalidations_total: self.counters.invalidations.load(Ordering::Relaxed),
            redis_hits_total: self.counters.redis_hits.load(Ordering::Relaxed),
            redis_errors_total: self.counters.redis_errors.load(Ordering::Relaxed),
            entries: state.entries.len(),
            bytes_in_use: state.bytes,
            bytes_limit: self.config.max_bytes,
        }
    }
}

fn remove_local(state: &mut LocalState, key: &str) {
    if let Some(old) = state.entries.remove(&LocalKey(key.to_string())) {
        state.bytes = state.bytes.saturating_sub(old.size);
    }
    state.order.retain(|item| item.0 != key);
}

#[cfg(feature = "redis")]
async fn redis_get(backend: &RedisBackend, key: &str) -> Result<Option<Bytes>, redis::RedisError> {
    let mut connection = backend.client.get_multiplexed_async_connection().await?;
    let value: Option<Vec<u8>> = connection.get(key).await?;
    Ok(value.map(Bytes::from))
}

#[cfg(feature = "redis")]
async fn redis_put(
    backend: &RedisBackend,
    key: &str,
    value: &Bytes,
    ttl: Duration,
) -> Result<(), redis::RedisError> {
    let mut connection = backend.client.get_multiplexed_async_connection().await?;
    let ttl_secs = ttl.as_secs().max(1);
    let _: () = connection.set_ex(key, value.as_ref(), ttl_secs).await?;
    Ok(())
}

#[cfg(feature = "redis")]
async fn redis_del(backend: &RedisBackend, key: &str) -> Result<(), redis::RedisError> {
    let mut connection = backend.client.get_multiplexed_async_connection().await?;
    let _: u64 = connection.del(key).await?;
    Ok(())
}

/// Store decorator used by the query path. It caches immutable data and
/// manifest/index reads while explicitly bypassing mutable coordination files.
pub struct CachingStore {
    inner: Arc<dyn Store>,
    cache: Arc<CacheManager>,
    scope: String,
}

impl CachingStore {
    pub fn new(inner: Arc<dyn Store>, cache: Arc<CacheManager>, scope: impl Into<String>) -> Self {
        Self {
            inner,
            cache,
            scope: scope.into(),
        }
    }

    fn cacheable(path: &str) -> bool {
        !(path.contains("ailake-serve")
            || path.contains("ailake-jobs")
            || path.contains("ailake_jobs")
            || path.contains("index-jobs")
            || path.ends_with(".lock")
            || path.ends_with(".cancel"))
    }

    fn kind(path: &str) -> CacheKind {
        if path.starts_with("metadata/") && !path.contains("manifest") {
            CacheKind::Metadata
        } else {
            CacheKind::Index
        }
    }

    async fn read(&self, path: &str, range: Option<Range<u64>>) -> AilakeResult<Bytes> {
        if !Self::cacheable(path) {
            return match range {
                Some(range) => self.inner.get_range(path, range).await,
                None => self.inner.get(path).await,
            };
        }
        let identity = match range {
            Some(ref range) => format!("{path}#{}-{}", range.start, range.end),
            None => path.to_string(),
        };
        let kind = Self::kind(path);
        let snapshot = self.cache.snapshot_for(&self.scope).await;
        if let Some(value) = self
            .cache
            .get(kind, &self.scope, snapshot, &identity)
            .await
            .map_err(cache_store_error)?
        {
            return Ok(value);
        }
        let value = match range {
            Some(range) => self.inner.get_range(path, range).await?,
            None => self.inner.get(path).await?,
        };
        self.cache
            .put(kind, &self.scope, snapshot, &identity, value.clone())
            .await
            .map_err(cache_store_error)?;
        Ok(value)
    }
}

fn cache_store_error(error: CacheError) -> AilakeError {
    AilakeError::Store(error.to_string())
}

#[async_trait]
impl Store for CachingStore {
    fn validate_location(&self, location: &str) -> AilakeResult<()> {
        self.inner.validate_location(location)
    }

    async fn try_acquire_lock(&self, path: &str) -> AilakeResult<bool> {
        self.inner.try_acquire_lock(path).await
    }

    async fn try_acquire_lock_fenced(&self, path: &str) -> AilakeResult<Option<u64>> {
        self.inner.try_acquire_lock_fenced(path).await
    }

    async fn release_lock(&self, path: &str) -> AilakeResult<()> {
        self.inner.release_lock(path).await
    }

    async fn release_lock_fenced(&self, path: &str, token: u64) -> AilakeResult<()> {
        self.inner.release_lock_fenced(path, token).await
    }

    async fn renew_lock(&self, path: &str) -> AilakeResult<bool> {
        self.inner.renew_lock(path).await
    }

    async fn renew_lock_fenced(&self, path: &str, token: u64) -> AilakeResult<bool> {
        self.inner.renew_lock_fenced(path, token).await
    }

    async fn check_lock_fence(&self, path: &str, token: u64) -> AilakeResult<bool> {
        self.inner.check_lock_fence(path, token).await
    }

    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        self.read(path, None).await
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        self.read(path, Some(range)).await
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        self.inner.put(path, data).await?;
        self.cache.invalidate_identity(&self.scope, path).await;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn file_size(&self, path: &str) -> AilakeResult<u64> {
        self.inner.file_size(path).await
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        self.inner.exists(path).await
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        self.inner.delete(path).await?;
        self.cache.invalidate_identity(&self.scope, path).await;
        Ok(())
    }
}

/// Store decorator that fails fast while the object store is unhealthy. Lock
/// operations are deliberately delegated directly because lock contention is a
/// coordination result, not a storage outage.
pub struct CircuitBreakingStore {
    inner: Arc<dyn Store>,
    breaker: Arc<CircuitBreaker>,
}

impl CircuitBreakingStore {
    pub fn new(inner: Arc<dyn Store>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }

    fn open_error() -> AilakeError {
        AilakeError::Store("object storage circuit breaker is open".into())
    }
}

#[async_trait]
impl Store for CircuitBreakingStore {
    fn validate_location(&self, location: &str) -> AilakeResult<()> {
        self.inner.validate_location(location)
    }

    async fn try_acquire_lock(&self, path: &str) -> AilakeResult<bool> {
        self.inner.try_acquire_lock(path).await
    }

    async fn try_acquire_lock_fenced(&self, path: &str) -> AilakeResult<Option<u64>> {
        self.inner.try_acquire_lock_fenced(path).await
    }

    async fn release_lock(&self, path: &str) -> AilakeResult<()> {
        self.inner.release_lock(path).await
    }

    async fn release_lock_fenced(&self, path: &str, token: u64) -> AilakeResult<()> {
        self.inner.release_lock_fenced(path, token).await
    }

    async fn renew_lock(&self, path: &str) -> AilakeResult<bool> {
        self.inner.renew_lock(path).await
    }

    async fn renew_lock_fenced(&self, path: &str, token: u64) -> AilakeResult<bool> {
        self.inner.renew_lock_fenced(path, token).await
    }

    async fn check_lock_fence(&self, path: &str, token: u64) -> AilakeResult<bool> {
        self.inner.check_lock_fence(path, token).await
    }

    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        self.breaker
            .execute(Self::open_error(), self.inner.get(path))
            .await
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        self.breaker
            .execute(Self::open_error(), self.inner.get_range(path, range))
            .await
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        self.breaker
            .execute(Self::open_error(), self.inner.put(path, data))
            .await
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        self.breaker
            .execute(Self::open_error(), self.inner.list(prefix))
            .await
    }

    async fn file_size(&self, path: &str) -> AilakeResult<u64> {
        self.breaker
            .execute(Self::open_error(), self.inner.file_size(path))
            .await
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        self.breaker
            .execute(Self::open_error(), self.inner.exists(path))
            .await
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        self.breaker
            .execute(Self::open_error(), self.inner.delete(path))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn global_limit_evicts_oldest_entry() {
        let cache = CacheManager::new(CacheConfig {
            max_bytes: 5,
            ..CacheConfig::default()
        })
        .unwrap();
        cache
            .put(
                CacheKind::Query,
                "t",
                Some(1),
                "a",
                Bytes::from_static(b"1234"),
            )
            .await
            .unwrap();
        cache
            .put(
                CacheKind::Index,
                "t",
                Some(1),
                "b",
                Bytes::from_static(b"12"),
            )
            .await
            .unwrap();
        assert!(cache
            .get(CacheKind::Query, "t", Some(1), "a")
            .await
            .unwrap()
            .is_none());
        assert_eq!(cache.stats().await.evictions_total, 1);
    }

    #[tokio::test]
    async fn local_rate_limit_enforces_token_and_ip_quotas() {
        let limiter = RateLimiter::new(RateLimitConfig {
            search_per_token: 2,
            search_per_ip: 3,
            ..RateLimitConfig::default()
        })
        .unwrap();
        assert!(
            limiter
                .check(RateLimitClass::Search, Some("token"), Some("10.0.0.1"))
                .await
                .unwrap()
                .allowed
        );
        assert!(
            limiter
                .check(RateLimitClass::Search, Some("token"), Some("10.0.0.1"))
                .await
                .unwrap()
                .allowed
        );
        let decision = limiter
            .check(RateLimitClass::Search, Some("token"), Some("10.0.0.1"))
            .await
            .unwrap();
        assert!(!decision.allowed);
        assert!(decision
            .dimension
            .as_deref()
            .is_some_and(|dimension| dimension.starts_with("token:")));
        assert_eq!(limiter.stats().await.limited_total, 1);
    }

    #[test]
    fn rate_limit_digest_is_stable() {
        assert_eq!(digest("token-a"), digest("token-a"));
        assert_ne!(digest("token-a"), digest("token-b"));
        assert_eq!(digest("token-a").len(), 64);
        assert!(!digest("token-a").contains("token-a"));
    }

    #[tokio::test]
    async fn circuit_breaker_fails_fast_after_threshold() {
        let breaker = CircuitBreaker::new(2, Duration::from_secs(60)).unwrap();
        for _ in 0..2 {
            assert!(breaker
                .execute(AilakeError::Store("circuit open".into()), async {
                    Err::<(), _>(AilakeError::Store("backend down".into()))
                },)
                .await
                .is_err());
        }
        let error = breaker
            .execute(AilakeError::Store("circuit open".into()), async {
                Ok::<_, AilakeError>(())
            })
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "store error: circuit open");
        assert!(breaker.is_open().await);
    }

    #[tokio::test]
    async fn snapshot_invalidation_keeps_current_snapshot() {
        let cache = CacheManager::new(CacheConfig::default()).unwrap();
        cache
            .put(
                CacheKind::Query,
                "t",
                Some(1),
                "a",
                Bytes::from_static(b"old"),
            )
            .await
            .unwrap();
        cache
            .put(
                CacheKind::Query,
                "t",
                Some(2),
                "b",
                Bytes::from_static(b"new"),
            )
            .await
            .unwrap();
        cache.invalidate_snapshot("t", 2).await;
        assert!(cache
            .get(CacheKind::Query, "t", Some(1), "a")
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            cache
                .get(CacheKind::Query, "t", Some(2), "b")
                .await
                .unwrap(),
            Some(Bytes::from_static(b"new"))
        );
    }

    #[tokio::test]
    async fn redis_round_trip_when_configured() {
        let Ok(redis_url) = std::env::var("AILAKE_TEST_REDIS_URL") else {
            return;
        };
        let namespace = format!(
            "cache-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let config = CacheConfig {
            namespace,
            ttl: Duration::from_secs(30),
            redis_url: Some(redis_url),
            ..CacheConfig::default()
        };
        let writer = CacheManager::new(config.clone()).unwrap();
        writer
            .put(
                CacheKind::Index,
                "integration.table",
                Some(7),
                "object#0-16",
                Bytes::from_static(b"redis-value"),
            )
            .await
            .unwrap();

        let reader = CacheManager::new(config).unwrap();
        assert_eq!(
            reader
                .get(
                    CacheKind::Index,
                    "integration.table",
                    Some(7),
                    "object#0-16"
                )
                .await
                .unwrap(),
            Some(Bytes::from_static(b"redis-value"))
        );
        assert_eq!(reader.stats().await.redis_hits_total, 1);
    }

    #[tokio::test]
    async fn redis_rate_limit_is_shared_when_configured() {
        let Ok(redis_url) = std::env::var("AILAKE_TEST_REDIS_URL") else {
            return;
        };
        let config = RateLimitConfig {
            namespace: format!(
                "rate-test-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            search_per_token: 1,
            search_per_ip: 0,
            redis_url: Some(redis_url),
            ..RateLimitConfig::default()
        };
        let first = RateLimiter::new(config.clone()).unwrap();
        let second = RateLimiter::new(config).unwrap();
        assert!(
            first
                .check(RateLimitClass::Search, Some("shared-token"), None)
                .await
                .unwrap()
                .allowed
        );
        assert!(
            !second
                .check(RateLimitClass::Search, Some("shared-token"), None)
                .await
                .unwrap()
                .allowed
        );
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn redis_outage_obeys_rate_limit_fail_closed_policy() {
        let redis_url = "redis://127.0.0.1:1/".to_string();
        let fail_closed = RateLimiter::new(RateLimitConfig {
            redis_url: Some(redis_url.clone()),
            fail_closed: true,
            ..RateLimitConfig::default()
        })
        .unwrap();
        let error = fail_closed
            .check(RateLimitClass::Search, Some("token"), None)
            .await
            .unwrap_err();
        assert!(matches!(error, RateLimitError::Backend(_)));
        assert_eq!(fail_closed.stats().await.backend_errors_total, 1);

        let fail_open = RateLimiter::new(RateLimitConfig {
            redis_url: Some(redis_url),
            fail_closed: false,
            ..RateLimitConfig::default()
        })
        .unwrap();
        assert!(
            fail_open
                .check(RateLimitClass::Search, Some("token"), None)
                .await
                .unwrap()
                .allowed
        );
        assert_eq!(fail_open.stats().await.backend_errors_total, 1);
    }
}
