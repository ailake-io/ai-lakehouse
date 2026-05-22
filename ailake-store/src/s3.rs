use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use object_store::aws::AmazonS3Builder;

use crate::ObjectStoreBackend;

pub enum S3Credentials {
    /// Explicit key + secret — dev, CI, MinIO, LocalStack.
    Static {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// IRSA / EKS Pod Identity — reads `AWS_WEB_IDENTITY_TOKEN_FILE` + `AWS_ROLE_ARN`
    /// from env (injected by the EKS controller). Fails at build time if either var is absent.
    WebIdentity,
    /// EC2 IMDSv2 instance profile — no env vars required.
    InstanceProfile,
    /// Full SDK chain: env vars → `~/.aws/credentials` → WebIdentity → IMDSv2.
    Default,
}

pub struct S3Config {
    pub bucket: String,
    /// AWS region. For `Default` credentials, also readable from `AWS_DEFAULT_REGION` / `AWS_REGION`.
    pub region: String,
    /// Custom endpoint — MinIO, LocalStack, etc. (`"http://localhost:9000"`)
    pub endpoint: Option<String>,
    /// Allow plain-text HTTP (required for MinIO without TLS).
    pub allow_http: bool,
    pub credentials: S3Credentials,
}

/// Build an [`ObjectStoreBackend`] pointing at an S3 bucket.
///
/// `prefix` is prepended to every path — set to the table root key
/// (e.g. `"warehouse/my_table/"`) so callers use relative paths.
#[cfg(feature = "store-s3")]
pub fn s3_store(config: S3Config, prefix: impl Into<String>) -> AilakeResult<ObjectStoreBackend> {
    let mut b = match &config.credentials {
        S3Credentials::Default => AmazonS3Builder::from_env(),
        _ => AmazonS3Builder::new(),
    };

    b = b
        .with_bucket_name(&config.bucket)
        .with_region(&config.region);

    if let Some(ep) = &config.endpoint {
        b = b.with_endpoint(ep);
    }
    if config.allow_http {
        b = b.with_allow_http(true);
    }

    match config.credentials {
        S3Credentials::Static {
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            b = b
                .with_access_key_id(access_key_id)
                .with_secret_access_key(secret_access_key);
            if let Some(tok) = session_token {
                b = b.with_token(tok);
            }
        }
        S3Credentials::WebIdentity => {
            if std::env::var("AWS_WEB_IDENTITY_TOKEN_FILE").is_err()
                || std::env::var("AWS_ROLE_ARN").is_err()
            {
                return Err(AilakeError::Store(
                    "S3Credentials::WebIdentity requires AWS_WEB_IDENTITY_TOKEN_FILE \
                     and AWS_ROLE_ARN env vars (injected by EKS/IRSA)"
                        .into(),
                ));
            }
        }
        S3Credentials::InstanceProfile | S3Credentials::Default => {}
    }

    let store = b
        .build()
        .map_err(|e| AilakeError::Store(e.to_string()))?;
    Ok(ObjectStoreBackend::new(Arc::new(store), prefix))
}
