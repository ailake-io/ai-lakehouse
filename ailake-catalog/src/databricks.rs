// Databricks Unity Catalog — configuration helpers for RestCatalog.
//
// Unity Catalog exposes an Iceberg REST Catalog endpoint, so RestCatalog
// handles all protocol work. This module only provides convenience
// constructors that wire up the correct URIs and auth for each Databricks
// cloud (Azure, AWS, GCP).
//
// URL layout:
//   base_uri = https://{workspace_host}/api/2.1/unity-catalog/iceberg
//   prefix   = {unity_catalog_name}           ← catalog name
//   namespace = {schema_name}                  ← in TableIdent
//   table    = {table_name}                    ← in TableIdent
//
// Full path example:
//   GET https://adb-xxx.azuredatabricks.net/api/2.1/unity-catalog/iceberg
//       /v1/my_catalog/namespaces/my_schema/tables/my_table
//
// Auth:
//   Pat                — all clouds, simplest (dev / CI)
//   AzureServicePrincipal — Azure production (service principal + Azure AD)
//   AwsOAuth2          — AWS production (Databricks M2M OAuth)
//   GcpBearer          — GCP production (short-lived GCP access token)

use crate::rest::{RestCatalogAuth, RestCatalogConfig};

/// Authentication strategy for Databricks.
#[derive(Debug, Clone)]
pub enum DatabricksAuth {
    /// Personal Access Token (all clouds). Simple; avoid in production for
    /// long-running services (tokens don't expire on a machine-friendly schedule).
    Pat(String),

    /// Azure AD service principal via OAuth2 client credentials.
    /// Requires an app registration with Databricks resource permission.
    /// Token endpoint: https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
    /// Scope: 2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default  (Azure Databricks resource)
    AzureServicePrincipal {
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },

    /// Databricks OAuth2 M2M on AWS.
    /// Token endpoint: https://{workspace_host}/oidc/v1/token
    /// Create a service principal in the Databricks account console.
    AwsOAuth2 {
        client_id: String,
        client_secret: String,
    },

    /// GCP — pre-obtained Google/Databricks access token.
    /// Obtain via `gcloud auth print-access-token` or Workload Identity Federation.
    /// For long-running services prefer a token refresh loop outside the SDK.
    GcpBearer(String),
}

/// Configuration builder for Databricks Unity Catalog on Azure.
///
/// ```rust,no_run
/// use ailake_catalog::databricks::{DatabricksAuth, databricks_azure};
/// use ailake_catalog::RestCatalog;
/// use ailake_store::LocalStore;
/// use std::sync::Arc;
///
/// let config = databricks_azure(
///     "myworkspace.azuredatabricks.net",
///     "my_unity_catalog",
///     "abfss://container@account.dfs.core.windows.net/warehouse",
///     DatabricksAuth::AzureServicePrincipal {
///         tenant_id: "00000000-0000-0000-0000-000000000000".into(),
///         client_id: "app-client-id".into(),
///         client_secret: "app-client-secret".into(),
///     },
/// );
/// let catalog = RestCatalog::new(config, Arc::new(LocalStore::new("/tmp")));
/// ```
pub fn databricks_azure(
    workspace_host: &str,
    unity_catalog: &str,
    warehouse: &str,
    auth: DatabricksAuth,
) -> RestCatalogConfig {
    let uri = format!(
        "https://{}/api/2.1/unity-catalog/iceberg",
        workspace_host.trim_end_matches('/')
    );
    RestCatalogConfig {
        uri,
        prefix: Some(unity_catalog.to_string()),
        warehouse: Some(warehouse.to_string()),
        auth: to_rest_auth_azure(auth),
    }
}

/// Configuration builder for Databricks Unity Catalog on AWS.
///
/// ```rust,no_run
/// use ailake_catalog::databricks::{DatabricksAuth, databricks_aws};
///
/// let config = databricks_aws(
///     "myworkspace.cloud.databricks.com",
///     "my_unity_catalog",
///     "s3://my-bucket/warehouse",
///     DatabricksAuth::AwsOAuth2 {
///         client_id: "sp-client-id".into(),
///         client_secret: "sp-client-secret".into(),
///     },
/// );
/// ```
pub fn databricks_aws(
    workspace_host: &str,
    unity_catalog: &str,
    warehouse: &str,
    auth: DatabricksAuth,
) -> RestCatalogConfig {
    let host = workspace_host.trim_end_matches('/');
    let uri = format!("https://{host}/api/2.1/unity-catalog/iceberg");
    RestCatalogConfig {
        uri,
        prefix: Some(unity_catalog.to_string()),
        warehouse: Some(warehouse.to_string()),
        auth: to_rest_auth_aws(auth, host),
    }
}

/// Configuration builder for Databricks Unity Catalog on GCP.
///
/// ```rust,no_run
/// use ailake_catalog::databricks::{DatabricksAuth, databricks_gcp};
///
/// let token = std::process::Command::new("gcloud")
///     .args(["auth", "print-access-token"])
///     .output().unwrap();
/// let token = String::from_utf8(token.stdout).unwrap().trim().to_string();
///
/// let config = databricks_gcp(
///     "myworkspace.gcp.databricks.com",
///     "my_unity_catalog",
///     "gs://my-bucket/warehouse",
///     DatabricksAuth::GcpBearer(token),
/// );
/// ```
pub fn databricks_gcp(
    workspace_host: &str,
    unity_catalog: &str,
    warehouse: &str,
    auth: DatabricksAuth,
) -> RestCatalogConfig {
    let uri = format!(
        "https://{}/api/2.1/unity-catalog/iceberg",
        workspace_host.trim_end_matches('/')
    );
    RestCatalogConfig {
        uri,
        prefix: Some(unity_catalog.to_string()),
        warehouse: Some(warehouse.to_string()),
        auth: databricks_auth_to_bearer(auth),
    }
}

// ── Auth conversion helpers ───────────────────────────────────────────────────

fn to_rest_auth_azure(auth: DatabricksAuth) -> RestCatalogAuth {
    match auth {
        DatabricksAuth::Pat(token) => RestCatalogAuth::Bearer(token),
        DatabricksAuth::AzureServicePrincipal {
            tenant_id,
            client_id,
            client_secret,
        } => RestCatalogAuth::OAuth2 {
            token_endpoint: format!(
                "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            ),
            client_id,
            client_secret,
            // Databricks resource ID in Azure AD (constant across all tenants)
            scope: Some("2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default".into()),
        },
        other => databricks_auth_to_bearer(other),
    }
}

fn to_rest_auth_aws(auth: DatabricksAuth, workspace_host: &str) -> RestCatalogAuth {
    match auth {
        DatabricksAuth::Pat(token) => RestCatalogAuth::Bearer(token),
        DatabricksAuth::AwsOAuth2 {
            client_id,
            client_secret,
        } => RestCatalogAuth::OAuth2 {
            token_endpoint: format!("https://{workspace_host}/oidc/v1/token"),
            client_id,
            client_secret,
            scope: Some("all-apis".into()),
        },
        other => databricks_auth_to_bearer(other),
    }
}

fn databricks_auth_to_bearer(auth: DatabricksAuth) -> RestCatalogAuth {
    match auth {
        DatabricksAuth::Pat(token) => RestCatalogAuth::Bearer(token),
        DatabricksAuth::GcpBearer(token) => RestCatalogAuth::Bearer(token),
        DatabricksAuth::AzureServicePrincipal { .. } => {
            // If caller uses AzureServicePrincipal on a non-Azure builder, treat as no-auth
            // and let them hit the 401 — better than silently misconfiguring.
            RestCatalogAuth::None
        }
        DatabricksAuth::AwsOAuth2 { .. } => RestCatalogAuth::None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_pat_config() {
        let cfg = databricks_azure(
            "adb-123.azuredatabricks.net",
            "my_catalog",
            "abfss://container@account.dfs.core.windows.net/wh",
            DatabricksAuth::Pat("dapiabc".into()),
        );
        assert_eq!(
            cfg.uri,
            "https://adb-123.azuredatabricks.net/api/2.1/unity-catalog/iceberg"
        );
        assert_eq!(cfg.prefix.as_deref(), Some("my_catalog"));
        assert_eq!(
            cfg.warehouse.as_deref(),
            Some("abfss://container@account.dfs.core.windows.net/wh")
        );
        assert!(matches!(cfg.auth, RestCatalogAuth::Bearer(t) if t == "dapiabc"));
    }

    #[test]
    fn azure_service_principal_uses_oauth2() {
        let cfg = databricks_azure(
            "adb-123.azuredatabricks.net",
            "my_catalog",
            "abfss://container@account.dfs.core.windows.net/wh",
            DatabricksAuth::AzureServicePrincipal {
                tenant_id: "tenant-uuid".into(),
                client_id: "client-uuid".into(),
                client_secret: "secret".into(),
            },
        );
        match &cfg.auth {
            RestCatalogAuth::OAuth2 {
                token_endpoint,
                scope,
                ..
            } => {
                assert!(token_endpoint.contains("tenant-uuid"));
                assert!(token_endpoint.contains("login.microsoftonline.com"));
                assert_eq!(
                    scope.as_deref(),
                    Some("2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default")
                );
            }
            other => panic!("expected OAuth2, got {:?}", other),
        }
    }

    #[test]
    fn aws_oauth2_uses_workspace_oidc_endpoint() {
        let cfg = databricks_aws(
            "myworkspace.cloud.databricks.com",
            "my_catalog",
            "s3://my-bucket/warehouse",
            DatabricksAuth::AwsOAuth2 {
                client_id: "sp-id".into(),
                client_secret: "sp-secret".into(),
            },
        );
        match &cfg.auth {
            RestCatalogAuth::OAuth2 {
                token_endpoint,
                scope,
                ..
            } => {
                assert_eq!(
                    token_endpoint.as_str(),
                    "https://myworkspace.cloud.databricks.com/oidc/v1/token"
                );
                assert_eq!(scope.as_deref(), Some("all-apis"));
            }
            other => panic!("expected OAuth2, got {:?}", other),
        }
    }

    #[test]
    fn gcp_bearer_config() {
        let cfg = databricks_gcp(
            "myworkspace.gcp.databricks.com",
            "my_catalog",
            "gs://my-bucket/warehouse",
            DatabricksAuth::GcpBearer("ya29.token".into()),
        );
        assert!(matches!(cfg.auth, RestCatalogAuth::Bearer(t) if t == "ya29.token"));
        assert_eq!(
            cfg.uri,
            "https://myworkspace.gcp.databricks.com/api/2.1/unity-catalog/iceberg"
        );
    }

    #[test]
    fn trailing_slash_stripped_from_host() {
        let cfg = databricks_azure(
            "adb-123.azuredatabricks.net/", // trailing slash
            "catalog",
            "s3://bucket/wh",
            DatabricksAuth::Pat("token".into()),
        );
        assert!(
            !cfg.uri.contains("//api"),
            "double slash in URI: {}",
            cfg.uri
        );
    }
}
