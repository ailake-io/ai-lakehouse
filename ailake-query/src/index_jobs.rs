// SPDX-License-Identifier: MIT OR Apache-2.0
//! Durable lifecycle for deferred vector-index builds.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ailake_catalog::TableIdent;
use ailake_core::{AilakeError, AilakeResult, VectorStoragePolicy};
use ailake_index::IvfPqConfig;
use ailake_store::Store;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const JOBS_PREFIX: &str = "metadata/index-jobs";
const MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum IndexAlgorithm {
    Hnsw,
    IvfPq,
    MultiHnsw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexJobRecord {
    pub id: String,
    pub table: String,
    pub file_path: String,
    pub algorithm: IndexAlgorithm,
    pub ivf_config: Option<IvfPqConfig>,
    pub status: String,
    pub progress: u8,
    pub attempts: u32,
    pub created_at: u64,
    pub updated_at: u64,
    pub error: Option<String>,
    #[serde(default)]
    pub cancel_requested: bool,
    #[serde(default)]
    pub multi_policies: Option<Vec<VectorStoragePolicy>>,
    #[serde(default)]
    pub payload_path: Option<String>,
}

#[derive(Clone)]
pub struct IndexJobHandle {
    store: Arc<dyn Store>,
    id: String,
    fencing_token: Arc<tokio::sync::Mutex<Option<u64>>>,
}

impl IndexJobHandle {
    pub async fn create(
        store: Arc<dyn Store>,
        table: &TableIdent,
        file_path: &str,
        algorithm: IndexAlgorithm,
        ivf_config: Option<IvfPqConfig>,
    ) -> AilakeResult<Self> {
        let now = now_ms();
        let handle = Self {
            store,
            id: Uuid::new_v4().to_string(),
            fencing_token: Arc::new(tokio::sync::Mutex::new(None)),
        };
        let record = IndexJobRecord {
            id: handle.id.clone(),
            table: table_key(table),
            file_path: file_path.to_string(),
            algorithm,
            ivf_config,
            status: "queued".into(),
            progress: 0,
            attempts: 0,
            created_at: now,
            updated_at: now,
            error: None,
            cancel_requested: false,
            multi_policies: None,
            payload_path: None,
        };
        handle.write(&record).await?;
        Ok(handle)
    }

    pub async fn create_multi(
        store: Arc<dyn Store>,
        table: &TableIdent,
        file_path: &str,
        policies: Vec<VectorStoragePolicy>,
        embeddings: Vec<Vec<Vec<f32>>>,
    ) -> AilakeResult<Self> {
        let handle = Self::create(store, table, file_path, IndexAlgorithm::MultiHnsw, None).await?;
        let payload_path = format!("{JOBS_PREFIX}/{}.payload.json", handle.id);
        let payload = serde_json::to_vec(&(policies.clone(), embeddings))
            .map_err(|e| AilakeError::Store(e.to_string()))?;
        handle
            .store
            .put(&payload_path, Bytes::from(payload))
            .await?;
        let mut record = handle.record().await?;
        record.multi_policies = Some(policies);
        record.payload_path = Some(payload_path);
        handle.write(&record).await?;
        Ok(handle)
    }

    pub async fn multi_payload(
        &self,
        record: &IndexJobRecord,
    ) -> AilakeResult<(Vec<VectorStoragePolicy>, Vec<Vec<Vec<f32>>>)> {
        let path = record
            .payload_path
            .as_deref()
            .ok_or_else(|| AilakeError::Store("multi-column job payload is missing".into()))?;
        let bytes = self.store.get(path).await?;
        serde_json::from_slice(&bytes).map_err(|e| AilakeError::Store(e.to_string()))
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn record_path(&self) -> String {
        format!("{JOBS_PREFIX}/{}.json", self.id)
    }

    fn run_lock_path(&self) -> String {
        format!("{JOBS_PREFIX}/{}.run.lock", self.id)
    }

    fn cancel_path(&self) -> String {
        format!("{JOBS_PREFIX}/{}.cancel", self.id)
    }

    pub async fn record(&self) -> AilakeResult<IndexJobRecord> {
        let bytes = self.store.get(&self.record_path()).await?;
        serde_json::from_slice(&bytes).map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn write(&self, record: &IndexJobRecord) -> AilakeResult<()> {
        let mut updated = record.clone();
        updated.updated_at = now_ms();
        let data = serde_json::to_vec(&updated).map_err(|e| AilakeError::Store(e.to_string()))?;
        self.store.put(&self.record_path(), Bytes::from(data)).await
    }

    async fn write_fenced(&self, record: &IndexJobRecord) -> AilakeResult<()> {
        let fence = *self.fencing_token.lock().await;
        let Some(fence) = fence else {
            return Err(AilakeError::Store("index job is not fenced".into()));
        };
        if !self
            .store
            .check_lock_fence(&self.run_lock_path(), fence)
            .await?
        {
            return Err(AilakeError::Store(
                "index job fencing token is no longer valid".into(),
            ));
        }
        self.write(record).await
    }

    /// Claim a job for one worker. The run lock is separate from the JSON record,
    /// so two processes cannot build the same file concurrently on LocalStore.
    pub async fn try_claim(&self) -> AilakeResult<bool> {
        let Some(fence) = self
            .store
            .try_acquire_lock_fenced(&self.run_lock_path())
            .await?
        else {
            return Ok(false);
        };
        *self.fencing_token.lock().await = Some(fence);
        let mut record = self.record().await?;
        if record.status == "succeeded"
            || record.status == "cancelled"
            || record.attempts >= MAX_ATTEMPTS
        {
            self.release_claim().await?;
            return Ok(false);
        }
        if self.is_cancel_requested().await? {
            record.status = "cancelled".into();
            record.error = Some("cancelled before execution".into());
            record.cancel_requested = true;
            self.write_fenced(&record).await?;
            self.release_claim().await?;
            return Ok(false);
        }
        record.status = "running".into();
        record.attempts += 1;
        record.error = None;
        if let Err(error) = self.write_fenced(&record).await {
            let _ = self.release_claim().await;
            return Err(error);
        }
        Ok(true)
    }

    pub async fn release_claim(&self) -> AilakeResult<()> {
        let fence = self.fencing_token.lock().await.take();
        if let Some(fence) = fence {
            self.store
                .release_lock_fenced(&self.run_lock_path(), fence)
                .await
        } else {
            Ok(())
        }
    }

    pub async fn renew_claim(&self) -> AilakeResult<bool> {
        let Some(fence) = *self.fencing_token.lock().await else {
            return Ok(false);
        };
        self.store
            .renew_lock_fenced(&self.run_lock_path(), fence)
            .await
    }

    pub async fn update_progress(&self, progress: u8) -> AilakeResult<()> {
        let mut record = self.record().await?;
        if record.status == "running" {
            record.progress = progress.min(99);
            self.write_fenced(&record).await?;
        }
        Ok(())
    }

    pub async fn succeed(&self) -> AilakeResult<()> {
        let mut record = self.record().await?;
        record.status = "succeeded".into();
        record.progress = 100;
        record.error = None;
        self.write_fenced(&record).await
    }

    pub async fn mark_cancelled(&self, reason: impl Into<String>) -> AilakeResult<()> {
        let mut record = self.record().await?;
        record.status = "cancelled".into();
        record.error = Some(reason.into());
        record.cancel_requested = true;
        if self.fencing_token.lock().await.is_some() {
            self.write_fenced(&record).await
        } else {
            self.write(&record).await
        }
    }

    /// Persist failure and return whether the caller should retry.
    pub async fn fail_or_retry(&self, error: &str) -> AilakeResult<bool> {
        let mut record = self.record().await?;
        record.error = Some(error.to_string());
        if record.attempts < MAX_ATTEMPTS && !record.cancel_requested {
            record.status = "queued".into();
            record.progress = 0;
            self.write_fenced(&record).await?;
            Ok(true)
        } else {
            record.status = "failed".into();
            self.write_fenced(&record).await?;
            Ok(false)
        }
    }

    pub async fn request_cancel(&self) -> AilakeResult<IndexJobRecord> {
        let mut record = self.record().await?;
        if matches!(record.status.as_str(), "succeeded" | "failed" | "cancelled") {
            return Err(AilakeError::InvalidArgument(format!(
                "index job {} is already {}",
                self.id, record.status
            )));
        }
        record.cancel_requested = true;
        self.store.put(&self.cancel_path(), Bytes::new()).await?;
        self.write(&record).await?;
        Ok(record)
    }

    pub async fn is_cancel_requested(&self) -> AilakeResult<bool> {
        let record = self.record().await?;
        Ok(record.cancel_requested || self.store.exists(&self.cancel_path()).await?)
    }

    pub async fn retry(&self) -> AilakeResult<IndexJobRecord> {
        let mut record = self.record().await?;
        if !matches!(record.status.as_str(), "failed" | "cancelled") {
            return Err(AilakeError::InvalidArgument(format!(
                "index job {} cannot be retried from {}",
                self.id, record.status
            )));
        }
        record.status = "queued".into();
        record.progress = 0;
        record.attempts = 0;
        record.error = None;
        record.cancel_requested = false;
        let _ = self.store.delete(&self.cancel_path()).await;
        self.write(&record).await?;
        Ok(record)
    }

    /// Turn an abandoned running job back into queued after process restart.
    pub async fn recover_after_restart(&self) -> AilakeResult<bool> {
        let mut record = self.record().await?;
        if record.status != "running" {
            return Ok(record.status == "queued");
        }
        if record.attempts >= MAX_ATTEMPTS {
            record.status = "failed".into();
            record.error = Some("worker stopped after the retry limit was reached".into());
            self.write(&record).await?;
            return Ok(false);
        }
        if self.is_cancel_requested().await? {
            record.status = "cancelled".into();
            record.cancel_requested = true;
            record.error = Some("cancelled while the previous worker stopped".into());
        } else {
            record.status = "queued".into();
            record.error = Some("worker recovered after process restart".into());
        }
        self.write(&record).await?;
        Ok(record.status == "queued")
    }
}

pub async fn load_index_job(store: Arc<dyn Store>, id: &str) -> AilakeResult<IndexJobRecord> {
    IndexJobHandle {
        store,
        id: id.to_string(),
        fencing_token: Arc::new(tokio::sync::Mutex::new(None)),
    }
    .record()
    .await
}

pub async fn list_index_jobs(store: Arc<dyn Store>) -> AilakeResult<Vec<IndexJobRecord>> {
    let mut result = Vec::new();
    for path in store.list(JOBS_PREFIX).await? {
        if !path.ends_with(".json") {
            continue;
        }
        if let Ok(bytes) = store.get(&path).await {
            if let Ok(record) = serde_json::from_slice::<IndexJobRecord>(&bytes) {
                result.push(record);
            }
        }
    }
    result.sort_by_key(|job| std::cmp::Reverse(job.created_at));
    Ok(result)
}

pub fn handle_for(store: Arc<dyn Store>, id: impl Into<String>) -> IndexJobHandle {
    IndexJobHandle {
        store,
        id: id.into(),
        fencing_token: Arc::new(tokio::sync::Mutex::new(None)),
    }
}

pub fn table_key(table: &TableIdent) -> String {
    format!("{}.{}", table.namespace, table.name)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_store::LocalStore;
    use tempfile::TempDir;

    #[tokio::test]
    async fn durable_job_lifecycle_supports_cancel_and_retry() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let table = TableIdent::new("ns", "items");
        let job = IndexJobHandle::create(
            Arc::clone(&store),
            &table,
            "data/a.parquet",
            IndexAlgorithm::Hnsw,
            None,
        )
        .await
        .unwrap();
        assert!(job.try_claim().await.unwrap());
        job.update_progress(42).await.unwrap();
        job.release_claim().await.unwrap();
        job.request_cancel().await.unwrap();
        assert!(job.is_cancel_requested().await.unwrap());
        job.mark_cancelled("test cancellation").await.unwrap();
        job.retry().await.unwrap();
        let record = job.record().await.unwrap();
        assert_eq!(record.status, "queued");
        assert_eq!(record.progress, 0);
        assert_eq!(list_index_jobs(store).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn running_job_is_requeued_after_restart() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let job = IndexJobHandle::create(
            Arc::clone(&store),
            &TableIdent::new("ns", "items"),
            "data/a.parquet",
            IndexAlgorithm::Hnsw,
            None,
        )
        .await
        .unwrap();
        assert!(job.try_claim().await.unwrap());
        job.release_claim().await.unwrap();
        assert!(job.recover_after_restart().await.unwrap());
        assert_eq!(job.record().await.unwrap().status, "queued");
        assert!(job.try_claim().await.unwrap());
        job.release_claim().await.unwrap();
    }
}
