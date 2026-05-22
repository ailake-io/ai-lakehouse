use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};

use crate::ObjectStoreBackend;

pub enum AzureCredentials {
    /// Service principal (Entra app registration) — primary auth for ADLS Gen2 in production.
    /// `client_id` + `client_secret` + `tenant_id`.
    ClientSecret {
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },
    /// Azure Managed Identity.
    /// `client_id = None` → system-assigned; `Some(id)` → user-assigned.
    ManagedIdentity { client_id: Option<String> },
    /// Storage account access key — dev / admin use only.
    AccessKey(String),
    /// SAS token string (e.g. `"sv=2021-10-04&ss=b&..."`).
    /// object_store parses the query-pair format internally.
    SasToken(String),
    /// Azure CLI (`az login`) — local development.
    AzureCli,
}

pub struct AzureConfig {
    pub account_name: String,
    pub container: String,
    pub credentials: AzureCredentials,
}

/// Build an [`ObjectStoreBackend`] pointing at an Azure Blob / ADLS Gen2 container.
///
/// `prefix` is prepended to every path — set to the table root key
/// (e.g. `"warehouse/my_table/"`) so callers use relative paths.
#[cfg(feature = "store-azure")]
pub fn azure_store(
    config: AzureConfig,
    prefix: impl Into<String>,
) -> AilakeResult<ObjectStoreBackend> {
    let mut b = MicrosoftAzureBuilder::new()
        .with_account(&config.account_name)
        .with_container_name(&config.container);

    match config.credentials {
        AzureCredentials::ClientSecret {
            tenant_id,
            client_id,
            client_secret,
        } => {
            b = b.with_client_secret_authorization(client_id, client_secret, tenant_id);
        }
        AzureCredentials::ManagedIdentity { client_id } => {
            if let Some(id) = client_id {
                // User-assigned: set client_id; no secret/tenant → builder routes to IMDS MSI.
                b = b.with_client_id(id);
            }
            // System-assigned: no fields needed; builder falls to ImdsManagedIdentityProvider.
        }
        AzureCredentials::AccessKey(key) => {
            b = b.with_access_key(key);
        }
        AzureCredentials::SasToken(token) => {
            // with_config(SasKey, …) lets object_store parse the raw query-pair string.
            b = b.with_config(AzureConfigKey::SasKey, token);
        }
        AzureCredentials::AzureCli => {
            b = b.with_use_azure_cli(true);
        }
    }

    let store = b.build().map_err(|e| AilakeError::Store(e.to_string()))?;
    Ok(ObjectStoreBackend::new(Arc::new(store), prefix))
}
