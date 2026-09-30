// SPDX-License-Identifier: MIT OR Apache-2.0
//! Secret resolution with bounded exposure and refreshable credentials.
//!
//! Providers return [`secrecy::SecretString`] and never include secret values in
//! errors. [`RotatingSecret`] caches a value for a bounded interval and fetches
//! a replacement on the next access after that interval, which makes mounted
//! Kubernetes secrets, Vault values, and AWS Secrets Manager values rotatable
//! without restarting the process.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
#[cfg(any(feature = "vault", feature = "aws-secrets-manager", test))]
use secrecy::ExposeSecret;
use secrecy::SecretString;
use thiserror::Error;
use tokio::sync::Mutex;
use zeroize::Zeroize;

#[derive(Debug, Error)]
pub enum SecretError {
    #[error("invalid secret reference: {0}")]
    InvalidReference(String),
    #[error("secret provider is not enabled: {0}")]
    ProviderDisabled(&'static str),
    #[error("secret I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("secret response could not be parsed: {0}")]
    Parse(String),
    #[error("secret provider request failed: {0}")]
    Request(String),
}

/// A provider is deliberately read-only. Rotation is implemented by fetching
/// the value again, so the provider never needs to own a mutable plaintext.
#[async_trait]
pub trait SecretProvider: Send + Sync {
    async fn fetch(&self) -> Result<SecretString, SecretError>;
}

#[derive(Clone)]
pub struct RotatingSecret {
    provider: Arc<dyn SecretProvider>,
    refresh_after: Duration,
    state: Arc<Mutex<Option<CachedSecret>>>,
}

struct CachedSecret {
    value: SecretString,
    refreshed_at: Instant,
}

impl RotatingSecret {
    pub fn new(provider: Arc<dyn SecretProvider>, refresh_after: Duration) -> Self {
        Self {
            provider,
            refresh_after: refresh_after.max(Duration::from_secs(1)),
            state: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn get(&self) -> Result<SecretString, SecretError> {
        {
            let state = self.state.lock().await;
            if let Some(cached) = &*state {
                if cached.refreshed_at.elapsed() < self.refresh_after {
                    return Ok(cached.value.clone());
                }
            }
        }

        // Serialize refreshes so a burst of concurrent requests performs only
        // one provider call. The second check handles the race after waiting.
        let mut state = self.state.lock().await;
        if let Some(cached) = &*state {
            if cached.refreshed_at.elapsed() < self.refresh_after {
                return Ok(cached.value.clone());
            }
        }
        let value = self.provider.fetch().await?;
        let result = value.clone();
        *state = Some(CachedSecret {
            value,
            refreshed_at: Instant::now(),
        });
        Ok(result)
    }

    pub fn refresh_after(&self) -> Duration {
        self.refresh_after
    }
}

#[async_trait]
impl SecretProvider for RotatingSecret {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        self.get().await
    }
}

#[derive(Clone, Debug)]
pub struct SecretResolverConfig {
    pub vault_address: Option<String>,
    pub vault_token: Option<SecretString>,
    pub aws_region: Option<String>,
    pub refresh_after: Duration,
}

impl Default for SecretResolverConfig {
    fn default() -> Self {
        Self {
            vault_address: None,
            vault_token: None,
            aws_region: None,
            refresh_after: Duration::from_secs(300),
        }
    }
}

#[derive(Clone)]
pub struct SecretResolver {
    config: SecretResolverConfig,
}

impl SecretResolver {
    pub fn new(config: SecretResolverConfig) -> Self {
        Self { config }
    }

    pub fn rotating(&self, reference: &str) -> Result<RotatingSecret, SecretError> {
        let provider = self.provider(reference)?;
        Ok(RotatingSecret::new(provider, self.config.refresh_after))
    }

    pub async fn resolve(&self, reference: &str) -> Result<SecretString, SecretError> {
        self.rotating(reference)?.get().await
    }

    fn provider(&self, reference: &str) -> Result<Arc<dyn SecretProvider>, SecretError> {
        let (scheme, remainder) = reference
            .split_once("://")
            .ok_or_else(|| SecretError::InvalidReference("expected scheme://value".into()))?;
        let (target, key) = split_fragment(remainder);
        match scheme {
            "env" => {
                if target.is_empty() || key.is_some() {
                    return Err(SecretError::InvalidReference(
                        "env:// requires only a variable name".into(),
                    ));
                }
                Ok(Arc::new(EnvProvider {
                    name: target.to_owned(),
                }))
            }
            "file" => Ok(Arc::new(FileProvider {
                path: absolute_path(target),
                key: key.map(str::to_owned),
            })),
            "k8s" => Ok(Arc::new(KubernetesProvider {
                path: absolute_path(target),
                key: key.map(str::to_owned),
            })),
            "vault" => {
                #[cfg(feature = "vault")]
                {
                    let address = self.config.vault_address.clone().ok_or_else(|| {
                        SecretError::InvalidReference(
                            "vault provider requires vault_address".into(),
                        )
                    })?;
                    let token = self.config.vault_token.clone().ok_or_else(|| {
                        SecretError::InvalidReference("vault provider requires vault_token".into())
                    })?;
                    Ok(Arc::new(VaultProvider {
                        client: reqwest::Client::new(),
                        address,
                        token,
                        path: target.trim_start_matches('/').to_owned(),
                        key: key.map(str::to_owned),
                    }))
                }
                #[cfg(not(feature = "vault"))]
                {
                    let _ = (target, key);
                    Err(SecretError::ProviderDisabled("vault"))
                }
            }
            "aws-sm" | "aws-secretsmanager" => {
                #[cfg(feature = "aws-secrets-manager")]
                {
                    Ok(Arc::new(AwsSecretsManagerProvider::new(
                        target.to_owned(),
                        key.map(str::to_owned),
                        self.config.aws_region.clone(),
                    )))
                }
                #[cfg(not(feature = "aws-secrets-manager"))]
                {
                    let _ = (target, key);
                    Err(SecretError::ProviderDisabled("aws-secrets-manager"))
                }
            }
            _ => Err(SecretError::InvalidReference(format!(
                "unsupported scheme '{scheme}'"
            ))),
        }
    }
}

fn split_fragment(value: &str) -> (&str, Option<&str>) {
    match value.split_once('#') {
        Some((target, key)) if !key.is_empty() => (target, Some(key)),
        Some((target, _)) => (target, None),
        None => (value, None),
    }
}

fn absolute_path(value: &str) -> PathBuf {
    PathBuf::from(format!("/{}", value.trim_start_matches('/')))
}

struct EnvProvider {
    name: String,
}

#[async_trait]
impl SecretProvider for EnvProvider {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        let value = std::env::var(&self.name).map_err(|_| {
            SecretError::InvalidReference(format!(
                "environment variable '{}' is not set",
                self.name
            ))
        })?;
        into_secret(value)
    }
}

struct FileProvider {
    path: PathBuf,
    key: Option<String>,
}

#[async_trait]
impl SecretProvider for FileProvider {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        let mut raw = tokio::fs::read_to_string(&self.path).await?;
        let value = extract_value(&raw, self.key.as_deref())?;
        raw.zeroize();
        into_secret(value)
    }
}

struct KubernetesProvider {
    path: PathBuf,
    key: Option<String>,
}

#[async_trait]
impl SecretProvider for KubernetesProvider {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        let mounted_key = match &self.key {
            Some(key) if self.path.is_dir() => self.path.join(key),
            _ => self.path.clone(),
        };
        let key = if self.path.is_dir() {
            None
        } else {
            self.key.as_deref()
        };
        let mut raw = tokio::fs::read_to_string(mounted_key).await?;
        let value = extract_value(&raw, key)?;
        raw.zeroize();
        into_secret(value)
    }
}

fn extract_value(raw: &str, key: Option<&str>) -> Result<String, SecretError> {
    let trimmed = raw.trim();
    if let Some(key) = key {
        let json: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| SecretError::Parse(format!("JSON secret: {e}")))?;
        let value = json
            .get("data")
            .and_then(|data| data.get(key))
            .or_else(|| json.get(key))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SecretError::Parse(format!("secret key '{key}' is missing")))?;
        return Ok(value.to_owned());
    }
    if trimmed.is_empty() {
        return Err(SecretError::Parse("secret is empty".into()));
    }
    Ok(trimmed.to_owned())
}

fn into_secret(value: String) -> Result<SecretString, SecretError> {
    if value.is_empty() {
        return Err(SecretError::Parse("secret is empty".into()));
    }
    Ok(SecretString::from(value))
}

#[cfg(feature = "vault")]
struct VaultProvider {
    client: reqwest::Client,
    address: String,
    token: SecretString,
    path: String,
    key: Option<String>,
}

#[cfg(feature = "vault")]
#[async_trait]
impl SecretProvider for VaultProvider {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        let url = format!(
            "{}/v1/{}",
            self.address.trim_end_matches('/'),
            self.path.trim_start_matches('/')
        );
        let response = self
            .client
            .get(url)
            .header("X-Vault-Token", self.token.expose_secret())
            .send()
            .await
            .map_err(|e| SecretError::Request(e.to_string()))?;
        if !response.status().is_success() {
            return Err(SecretError::Request(format!(
                "Vault returned HTTP {}",
                response.status()
            )));
        }
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SecretError::Parse(format!("Vault JSON: {e}")))?;
        let data = body.get("data").unwrap_or(&body);
        let data = data.get("data").unwrap_or(data);
        let key = self
            .key
            .as_deref()
            .ok_or_else(|| SecretError::InvalidReference("vault://path requires #key".into()))?;
        let value = data
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SecretError::Parse(format!("Vault key '{key}' is missing")))?;
        into_secret(value.to_owned())
    }
}

#[cfg(feature = "aws-secrets-manager")]
struct AwsSecretsManagerProvider {
    secret_id: String,
    key: Option<String>,
    region: Option<String>,
    client: tokio::sync::OnceCell<aws_sdk_secretsmanager::Client>,
}

#[cfg(feature = "aws-secrets-manager")]
impl AwsSecretsManagerProvider {
    fn new(secret_id: String, key: Option<String>, region: Option<String>) -> Self {
        Self {
            secret_id,
            key,
            region,
            client: tokio::sync::OnceCell::const_new(),
        }
    }
}

#[cfg(feature = "aws-secrets-manager")]
#[async_trait]
impl SecretProvider for AwsSecretsManagerProvider {
    async fn fetch(&self) -> Result<SecretString, SecretError> {
        let client = self
            .client
            .get_or_try_init(|| async {
                let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
                if let Some(region) = &self.region {
                    loader = loader.region(aws_types::region::Region::new(region.clone()));
                }
                let config = loader.load().await;
                Ok::<_, SecretError>(aws_sdk_secretsmanager::Client::new(&config))
            })
            .await?;
        let response = client
            .get_secret_value()
            .secret_id(&self.secret_id)
            .send()
            .await
            .map_err(|e| {
                SecretError::Request(format!("AWS Secrets Manager request failed: {e}"))
            })?;
        let raw = response
            .secret_string()
            .ok_or_else(|| SecretError::Parse("AWS secret has no SecretString payload".into()))?;
        extract_value(raw, self.key.as_deref()).and_then(into_secret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingProvider {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl SecretProvider for CountingProvider {
        async fn fetch(&self) -> Result<SecretString, SecretError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(SecretString::from(format!("secret-{n}")))
        }
    }

    #[tokio::test]
    async fn rotating_secret_caches_until_deadline() {
        let provider = Arc::new(CountingProvider {
            calls: AtomicUsize::new(0),
        });
        let secret = RotatingSecret::new(provider.clone(), Duration::from_secs(60));
        assert_eq!(secret.get().await.unwrap().expose_secret(), "secret-0");
        assert_eq!(secret.get().await.unwrap().expose_secret(), "secret-0");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn extracts_kubernetes_json_key() {
        assert_eq!(
            extract_value(r#"{"token":"abc"}"#, Some("token")).unwrap(),
            "abc"
        );
        assert_eq!(extract_value("abc\n", None).unwrap(), "abc");
    }
}
