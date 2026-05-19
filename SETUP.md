# SETUP.md — Testando o AI-Lake Format localmente

Guia para rodar o formato de arquivo localmente: escrever batches, busca vetorial com pruning geométrico, compaction, ContextAssembler, bindings Python, inspeção de layout e verificação de compatibilidade Parquet.

---

## Pré-requisitos

| Ferramenta | Versão mínima | Instalação |
|---|---|---|
| Rust + Cargo | 1.75+ (stable) | `curl https://sh.rustup.rs -sSf \| sh` |
| Python 3 | 3.9+ | sistema / conda |
| PyArrow | qualquer | `pip install pyarrow` |
| maturin | 1.4+ | `pip install maturin` *(só para testar bindings Python)* |

Verificar:

```bash
rustc --version   # rustc 1.75+
cargo --version
python3 -c "import pyarrow; print(pyarrow.__version__)"
```

---

## 1. Clone e build

```bash
git clone https://github.com/ThiagoLange/iceberg-ai-deltalakehouse.git
cd iceberg-ai-deltalakehouse

# Compilar todos os crates do workspace
cargo build --workspace
```

Primeira compilação leva ~2-3 min (baixa dependências Arrow/Parquet).

Para habilitar backends de cloud storage:

```bash
# S3
cargo build --workspace --features store-s3

# GCS
cargo build --workspace --features store-gcs

# Azure Blob
cargo build --workspace --features store-azure
```

---

## 2. Suite de testes completa

```bash
# Testes unitários de todos os crates (60 testes, ~0.5s)
cargo test --workspace --lib

# Testes de integração (write + read + search end-to-end)
cargo test -p tests

# Todos de uma vez
cargo test --workspace
```

Deve terminar com `60 passed, 1 ignored`.

### Testes por crate

| Crate | O que cobre |
|---|---|
| `ailake-vec` | Quantização F32→F16, PQ (encode/decode/ADC), BlockCompressor (zstd/lz4), centróides |
| `ailake-index` | HNSW build/search, serialização bincode, MmapLoader round-trip |
| `ailake-file` | Escrita/leitura do arquivo unificado, layout AILK, integridade |
| `ailake-query` | ContextAssembler (dedup, grouping, XML, budget), pruning geométrico |
| `tests` (integração) | write→read→search end-to-end, invariante posicional, compatibilidade PyArrow, pruning, context assembler |

---

## 3. Testes de Fase 2 em detalhe

### 3A. Product Quantization (PQ)

```bash
cargo test -p ailake-vec -- pq
```

Testa:
- `encode_decode_roundtrip_approx` — encode + decode preserva dimensão
- `adc_distance_non_negative` — distância ADC ≥ 0 sempre
- `nearest_neighbor_rank_preserved` — q1 mais próximo do cluster 1 do que do cluster 2
- `dim_not_divisible_errors` — erro se `dim % M != 0`

### 3B. BlockCompressor (zstd/lz4)

```bash
cargo test -p ailake-vec -- compress
```

Testa round-trip de compressão/decompressão para codecs `None`, `Lz4` e `Zstd`.

### 3C. MmapLoader

```bash
cargo test -p ailake-index -- mmap
```

Testa que bytes HNSW escritos em tempfile e abertos via mmap desserializam corretamente.

### 3D. Pruning geométrico (integração)

```bash
cargo test -p tests --test vector_pruning
```

Cria dois arquivos:
- **File A**: vetores próximos de `[1, 0, 0, 0]`
- **File B**: vetores próximos de `[0, 0, 0, 1]`

Busca com query `[1, 0, 0, 0]` e `pruning_threshold = 0.5`. File B deve ser eliminado — todos os resultados vêm de `part-00000.parquet`.

### 3E. ContextAssembler (integração)

```bash
cargo test -p tests --test context_assembler
```

- `dedup_removes_near_identical_chunks` — embeddings idênticos → só 1 chunk sobrevive
- `grouping_restores_chunk_order` — chunks fora de ordem → XML com `chunk_index` crescente

### 3F. ObjectStoreBackend (cloud storage)

Os backends S3/GCS/Azure não têm testes automáticos sem Docker (deferred para Fase 3). Para testar manualmente com MinIO local:

```bash
# Subir MinIO
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Criar bucket via mc ou console (http://localhost:9001)
```

```rust
use object_store::aws::AmazonS3Builder;
use ailake_store::ObjectStoreBackend;

let s3 = AmazonS3Builder::new()
    .with_bucket_name("test-bucket")
    .with_region("us-east-1")
    .with_endpoint("http://localhost:9000")
    .with_access_key_id("minioadmin")
    .with_secret_access_key("minioadmin")
    .with_allow_http(true)
    .build()?;

let store = ObjectStoreBackend::new(Arc::new(s3), "warehouse/");
```

---

## 4. Demo completo — escrever, buscar, inspecionar

O exemplo `demo` (em `ailake-query/examples/demo.rs`) faz o fluxo completo em filesystem local:

1. Cria tabela AI-Lake com 2 arquivos (500 linhas cada)
2. Imprime layout binário do arquivo (offsets de PAR1, AILK, HNSW)
3. Busca top-5 por similaridade cosine (sem pruning — `pruning_threshold = f32::INFINITY`)
4. Verifica integridade dos dois arquivos
5. Lista o catálogo Iceberg

```bash
cargo run --example demo -p ailake-query
```

Saída esperada:

```
Workspace: /tmp/ailakeXXXXXX

=== Writing 2 batches (500 rows each) ===
  part-00000.parquet written
  part-00001.parquet written
  Committed snapshot id=1

=== File layout inspection (part-00000.parquet) ===
  File layout (NNNNN bytes):
    PAR1 #1 at byte 0
    AILK magic at byte XXXXX
    AILK magic at byte XXXXX
    PAR1 #2 at byte NNNNN-4
    AILK section    : XXXXX..XXXXX
    Centroid section: XXXXX..XXXXX
    HNSW section    : XXXXX..XXXXX (YYYYY bytes)
    Record count    : 500
    Dim             : 16

=== Search: query = embs1[0] (should be top result) ===
  Top-5 results:
    #1: row_id=0 distance=0.000000  file=data/part-00000.parquet
    ...

PASS: top result distance = X.XXe-XX < 0.01

=== Integrity check on both files ===
  data/part-00000.parquet — 500 nodes, integrity OK
  data/part-00001.parquet — 500 nodes, integrity OK

=== Catalog: list_files ===
  data/part-00000.parquet — 500 rows, hnsw_offset=XXXXX, hnsw_len=XXXXX
  data/part-00001.parquet — 500 rows, hnsw_offset=XXXXX, hnsw_len=XXXXX

Fase 1 demo concluída com sucesso.
```

---

## 5. Testar pruning geométrico no demo

Para ver pruning em ação, crie dois arquivos com vetores em direções opostas e busque com threshold baixo:

```rust
// SearchConfig com pruning ativo
let results = search(
    &table, &query,
    SearchConfig {
        top_k: 5,
        ef_search: 50,
        pruning_threshold: 0.5,  // f32::INFINITY = sem pruning
    },
    "embedding", dim, catalog, store,
).await?;
```

`pruning_threshold` controla agressividade: menor = mais arquivos eliminados = mais rápido, potencialmente menor recall.

---

## 6. Testar bindings Python (ailake-py)

```bash
cd ailake-py
pip install maturin pyarrow numpy

# Compilar e instalar no env Python atual
maturin develop

# Verificar importação
python3 -c "import ailake; print(dir(ailake))"
```

Usar o SDK Python:

```python
import ailake
import numpy as np

# Escrever
writer = ailake.TableWriter(
    path="/tmp/ailake-test",
    vector_column="embedding",
    dim=64,
    metric="cosine",
)
writer.write_batch(
    texts=["chunk de texto 1", "chunk de texto 2"],
    embeddings=np.random.rand(2, 64).astype(np.float32),
)
snapshot_id = writer.commit()
print(f"Snapshot: {snapshot_id}")

# Buscar
query = np.random.rand(64).astype(np.float32)
results = ailake.search(path="/tmp/ailake-test", query=query.tolist(), top_k=5)
print(results)

# ContextAssembler
ctx = ailake.assemble_context(
    chunks=[
        {"document_id": "doc-1", "chunk_index": 0, "chunk_text": "Texto...", "distance": 0.1},
    ],
    max_tokens=4096,
    dedup_threshold=0.05,
)
print(ctx)
```

---

## 7. Verificar compatibilidade PyArrow manualmente

```bash
# Gerar arquivo temporário via teste
cargo test -p tests --test parquet_trailing_bytes -- --ignored --nocapture 2>&1 | grep -i "path\|file\|ok\|FAILED"
```

Ou apontar para arquivo gerado pelo demo:

```python
import pyarrow.parquet as pq

# Substituir pelo caminho impresso pelo demo ("Workspace: ...")
path = "/tmp/ailakeXXXXXX/warehouse/default/demo_table/data/part-00000.parquet"

table = pq.read_table(path)
print(f"Rows: {table.num_rows}")
print(f"Schema: {table.schema}")
print(table.to_pandas().head())
```

PyArrow deve ler normalmente — colunas `id`, `text`, e `embedding` (como bytes). Sem erros de magic ou footer.

---

## 8. Rodar benchmarks

```bash
# HNSW search benchmark (ailake-index)
cargo bench -p ailake-index

# Write benchmark (ailake-file)
cargo bench -p ailake-file
```

---

## 9. Testar RestCatalog — multi-cloud

O `RestCatalog` implementa o [Iceberg REST Catalog spec](https://iceberg.apache.org/spec/#rest-catalog) e funciona com Polaris, Nessie, S3 Tables, AWS BigLake e Unity Catalog.

### 9A. Testes unitários (sem servidor externo)

```bash
cargo test -p ailake-catalog
```

Cobre URL building, serialização do `CommitTableRequest`, storage root derivation e configs Databricks.

### 9B. RestCatalog local com Nessie

```bash
# Subir Nessie (Project Nessie — catálogo com branching)
docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest

# Rodar teste de integração (necessita servidor)
cargo test -p tests --test rest_nessie -- --ignored
```

Configuração manual em Rust:

```rust
use ailake_catalog::{RestCatalog, RestCatalogAuth, RestCatalogConfig};
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "http://localhost:19120/api".into(),
        prefix: Some("main".into()),
        warehouse: Some("/tmp/warehouse".into()),
        auth: RestCatalogAuth::None,
    },
    store,
);
```

### 9C. RestCatalog local com Apache Polaris

```bash
docker run -p 8181:8181 apache/polaris:latest

cargo test -p tests --test rest_polaris -- --ignored
```

Configuração:

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "http://localhost:8181".into(),
        prefix: Some("my_polaris_catalog".into()),
        warehouse: Some("s3://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::Bearer("my-bootstrap-token".into()),
    },
    store,
);
```

### 9D. AWS S3 Tables (REST nativo na AWS)

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://s3tables.us-east-1.amazonaws.com/iceberg".into(),
        prefix: Some("arn:aws:s3tables:us-east-1:123456789012:bucket/my-bucket".into()),
        warehouse: None,
        auth: RestCatalogAuth::Bearer(aws_access_token),
    },
    s3_store,
);
```

### 9E. GCP BigLake Metastore

```rust
let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://biglake.googleapis.com/iceberg/v1beta".into(),
        prefix: Some("projects/my-project/locations/us-central1/catalogs/my-catalog".into()),
        warehouse: Some("gs://my-bucket/warehouse".into()),
        auth: RestCatalogAuth::Bearer(gcp_access_token),
    },
    gcs_store,
);
```

### 9F. Azure Blob + Apache Polaris (produção Azure)

```rust
use object_store::azure::MicrosoftAzureBuilder;
use ailake_store::ObjectStoreBackend;

let azure = MicrosoftAzureBuilder::new()
    .with_account("myaccount")
    .with_access_key("my-access-key")
    .with_container("mycontainer")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(azure), "warehouse/"));

let catalog = RestCatalog::new(
    RestCatalogConfig {
        uri: "https://my-polaris.azuredatabricks.net/polaris/api/catalog".into(),
        prefix: Some("my_catalog".into()),
        warehouse: Some("abfss://mycontainer@myaccount.dfs.core.windows.net/warehouse".into()),
        auth: RestCatalogAuth::OAuth2 {
            token_endpoint: "https://login.microsoftonline.com/TENANT/oauth2/v2.0/token".into(),
            client_id: "CLIENT_ID".into(),
            client_secret: "CLIENT_SECRET".into(),
            scope: Some("api://POLARIS_APP_ID/.default".into()),
        },
    },
    store,
);
```

---

## 10. Testar Databricks Unity Catalog

Os helpers `databricks_azure` / `databricks_aws` / `databricks_gcp` constroem o `RestCatalogConfig` correto para cada cloud. Requerem workspace Databricks real — não tem emulador local.

### 10A. Azure (service principal)

```rust
use ailake_catalog::{databricks_azure, DatabricksAuth, RestCatalog};
use object_store::azure::MicrosoftAzureBuilder;
use ailake_store::ObjectStoreBackend;
use std::sync::Arc;

let azure = MicrosoftAzureBuilder::new()
    .with_account("myaccount")
    .with_access_key("my-access-key")
    .with_container("mycontainer")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(azure), "warehouse/"));

let catalog = RestCatalog::new(
    databricks_azure(
        "myworkspace.azuredatabricks.net",
        "my_unity_catalog",
        "abfss://mycontainer@myaccount.dfs.core.windows.net/warehouse",
        DatabricksAuth::AzureServicePrincipal {
            tenant_id: std::env::var("AZURE_TENANT_ID")?,
            client_id: std::env::var("AZURE_CLIENT_ID")?,
            client_secret: std::env::var("AZURE_CLIENT_SECRET")?,
        },
    ),
    store,
);
```

Para dev/CI com PAT:

```rust
DatabricksAuth::Pat(std::env::var("DATABRICKS_TOKEN")?)
```

Token endpoint usado: `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`
Scope: `2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default` (recurso Databricks no Azure AD)

### 10B. AWS (M2M OAuth2)

```rust
use ailake_catalog::{databricks_aws, DatabricksAuth};
use object_store::aws::AmazonS3Builder;

let s3 = AmazonS3Builder::new()
    .with_bucket_name("my-bucket")
    .with_region("us-east-1")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(s3), "warehouse/"));

let catalog = RestCatalog::new(
    databricks_aws(
        "myworkspace.cloud.databricks.com",
        "my_unity_catalog",
        "s3://my-bucket/warehouse",
        DatabricksAuth::AwsOAuth2 {
            client_id: std::env::var("DATABRICKS_CLIENT_ID")?,
            client_secret: std::env::var("DATABRICKS_CLIENT_SECRET")?,
        },
    ),
    store,
);
```

Token endpoint usado: `https://myworkspace.cloud.databricks.com/oidc/v1/token`
Scope: `all-apis`

### 10C. GCP (Bearer token)

```bash
# Obter token via gcloud
export GCP_TOKEN=$(gcloud auth print-access-token)
```

```rust
use ailake_catalog::{databricks_gcp, DatabricksAuth};
use object_store::gcp::GoogleCloudStorageBuilder;

let gcs = GoogleCloudStorageBuilder::new()
    .with_bucket_name("my-bucket")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(gcs), "warehouse/"));

let catalog = RestCatalog::new(
    databricks_gcp(
        "myworkspace.gcp.databricks.com",
        "my_unity_catalog",
        "gs://my-bucket/warehouse",
        DatabricksAuth::GcpBearer(std::env::var("GCP_TOKEN")?),
    ),
    store,
);
```

### 10D. Hierarquia Unity Catalog

Unity Catalog usa 3 níveis: `catalog.schema.table`.

```rust
// Tabela: my_unity_catalog.prod_schema.embeddings
let table = TableIdent::new("prod_schema", "embeddings");
// catalog = prefixo do RestCatalogConfig (definido no databricks_*)
// schema  = TableIdent.namespace
// table   = TableIdent.name
```

URL resultante:
```
GET https://myworkspace.azuredatabricks.net/api/2.1/unity-catalog/iceberg
    /v1/my_unity_catalog/namespaces/prod_schema/tables/embeddings
```

### 10E. Fluxo de busca multi-cloud (mesmo código para todos os backends)

```rust
use ailake_query::{search, SearchConfig};
use ailake_catalog::{TableIdent, CatalogProvider};
use std::sync::Arc;

// catalog pode ser HadoopCatalog, RestCatalog, ou qualquer backend
let catalog: Arc<dyn CatalogProvider> = Arc::new(/* qualquer backend */);

let table = TableIdent::new("prod_schema", "embeddings");
let query = vec![0.1_f32; 1536];

let results = search(
    &table, &query,
    SearchConfig { top_k: 10, ef_search: 50, pruning_threshold: 0.8 },
    "embedding", 1536, catalog, store,
).await?;
```

Pruning geométrico funciona identicamente para todos os backends — centróide e raio ficam no manifesto, não no servidor de catálogo.

---

## 11. NessieCatalog — branching operations

`NessieCatalog` wraps `RestCatalog` para todas as operações `CatalogProvider` e adiciona a API de branching do Nessie v2.

### 11A. Testes unitários (sem servidor externo)

```bash
cargo test -p ailake-catalog --features catalog-nessie
```

Cobre URL construction (`trees_url`, `ref_url`, `merge_url`) e desserialização JSON da API Nessie.

### 11B. Testes de integração (requer servidor Nessie)

```bash
docker run -p 19120:19120 ghcr.io/projectnessie/nessie:latest

cargo test -p tests --test rest_nessie -- --ignored
```

### 11C. Configuração e uso

```rust
use ailake_catalog::{NessieCatalog, NessieCatalogConfig, RestCatalogAuth};
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = NessieCatalog::new(
    NessieCatalogConfig {
        uri: "http://localhost:19120/api".into(),
        default_branch: "main".into(),
        warehouse: Some("/tmp/warehouse".into()),
        auth: RestCatalogAuth::None,
    },
    store,
);

// CatalogProvider → delega para inner RestCatalog (branch "main")
catalog.create_table(&table, &props).await?;

// Branching operations — específicas do Nessie
let branches = catalog.list_branches().await?;
catalog.create_branch("feature-rag-v2", "main").await?;

// trabalhar na branch feature...

catalog.merge_branch("feature-rag-v2", "main").await?;
catalog.delete_branch("feature-rag-v2").await?;
```

Auth via PAT:
```rust
auth: RestCatalogAuth::Bearer("my-nessie-token".into())
```

Auth via OAuth2 (Nessie com OIDC):
```rust
auth: RestCatalogAuth::OAuth2 {
    token_endpoint: "https://my-oidc/token".into(),
    client_id: "client-id".into(),
    client_secret: "secret".into(),
    scope: None,
}
```

---

## 12. JdbcCatalog — PostgreSQL / MySQL

Armazena o ponteiro `metadata_location` em banco de dados relacional. Ideal para deploys self-hosted sem AWS Glue.

### 12A. Testes unitários + SQLite e2e (sem DB externo)

```bash
cargo test -p ailake-catalog --features catalog-jdbc
```

Inclui teste end-to-end completo com SQLite in-process (`catalog-jdbc` feature ativa o driver SQLite via sqlx).

### 12B. PostgreSQL via Docker

```bash
docker run --name pg-ailake -e POSTGRES_PASSWORD=test -p 5432:5432 -d postgres:16
```

```rust
use ailake_catalog::JdbcCatalog;
use ailake_store::LocalStore;
use std::sync::Arc;

let store = Arc::new(LocalStore::new("/tmp/warehouse"));
let catalog = JdbcCatalog::connect(
    "postgres://postgres:test@localhost:5432/postgres",
    "prod-catalog",      // catalog name (partitions iceberg_tables)
    "/tmp/warehouse",    // warehouse root
    store,
).await?;

// Schema criado automaticamente (CREATE TABLE IF NOT EXISTS iceberg_tables)
catalog.create_table(&table, &props).await?;
let snap_id = catalog.commit_snapshot(&table, snapshot).await?;
let files = catalog.list_files(&table, Some(snap_id)).await?;
```

### 12C. MySQL via Docker

```bash
docker run --name mysql-ailake \
  -e MYSQL_ROOT_PASSWORD=test -e MYSQL_DATABASE=ailake \
  -p 3306:3306 -d mysql:8
```

```rust
let catalog = JdbcCatalog::connect(
    "mysql://root:test@localhost:3306/ailake",
    "prod-catalog",
    "s3://my-bucket/warehouse",
    store,
).await?;
```

### 12D. SQLite local (dev / testes)

```rust
let catalog = JdbcCatalog::connect(
    "sqlite:///tmp/catalog.db?mode=rwc",
    "dev-catalog",
    "/tmp/warehouse",
    store,
).await?;
```

Nota: `sqlite::memory:` não funciona com pool (cada conexão tem DB separado). Use arquivo.

### 12E. Schema criado automaticamente

```sql
CREATE TABLE IF NOT EXISTS iceberg_tables (
    catalog_name      VARCHAR(255) NOT NULL,
    table_namespace   VARCHAR(255) NOT NULL,
    table_name        VARCHAR(255) NOT NULL,
    metadata_location VARCHAR(1000) NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name)
);
```

Cada `commit_snapshot` escreve novo `{uuid}.metadata.json` no Store e faz `UPDATE` do ponteiro no banco. Assumption: single-writer.

---

## 13. GlueCatalog — AWS Glue Data Catalog

Armazena `metadata_location` no Glue. Tabelas ficam visíveis no Athena, EMR, Glue ETL e Redshift Spectrum.

### 13A. Testes unitários (sem AWS)

```bash
cargo test -p ailake-catalog --features catalog-glue
```

Cobre encoding dos parâmetros Glue e formato dos paths.

### 13B. Configuração

```rust
use ailake_catalog::{GlueCatalog, GlueCatalogConfig};
use ailake_store::ObjectStoreBackend;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;

let s3 = AmazonS3Builder::new()
    .with_bucket_name("my-bucket")
    .with_region("us-east-1")
    .build()?;
let store = Arc::new(ObjectStoreBackend::new(Arc::new(s3), "warehouse/"));

// Carrega credenciais do ambiente (AWS_ACCESS_KEY_ID / IAM role / ~/.aws)
let catalog = GlueCatalog::from_env(
    GlueCatalogConfig {
        database: "my_glue_database".into(),
        warehouse: "s3://my-bucket/warehouse".into(),
        region: Some("us-east-1".into()),
    },
    store,
).await;

catalog.create_table(&table, &props).await?;
```

Client explícito (quando você já tem um `aws_sdk_glue::Client`):

```rust
use aws_config::BehaviorVersion;
use aws_sdk_glue::config::Region;

let sdk_config = aws_config::defaults(BehaviorVersion::latest())
    .region(Region::new("us-east-1"))
    .load()
    .await;
let client = aws_sdk_glue::Client::new(&sdk_config);
let catalog = GlueCatalog::from_client(client, config, store);
```

### 13C. Parâmetros criados no Glue

```
table_type        = "ICEBERG"
metadata_location = "s3://bucket/warehouse/ns/table/metadata/{uuid}.metadata.json"
```

Compatível com `SHOW TBLPROPERTIES` no Athena e com o conector Iceberg do AWS Glue ETL.

### 13D. Testar com Localstack (opcional)

```bash
pip install localstack awscli-local
localstack start -d

# criar database no Glue local
awslocal glue create-database --database-input '{"Name": "test_db"}'

# testar
AWS_ENDPOINT_URL=http://localhost:4566 \
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
  cargo test -p tests --test glue_localstack -- --ignored
```

---

## 14. Clippy e formatação

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Ambos devem terminar sem erros ou warnings.

---

## Estrutura dos crates

```
ailake-core/      tipos base: VectorMetric, VectorPrecision, RowId, AilakeError
ailake-parquet/   leitura/escrita Parquet com coluna VECTOR (FIXED_LEN_BYTE_ARRAY)
ailake-vec/       quantização F32→F16, Product Quantization (PQCodebook), BlockCompressor (zstd/lz4), distâncias, centróides
ailake-index/     HNSW via hnsw_rs, serialização bincode, MmapLoader (memmap2)
ailake-file/      arquivo unificado: AILK entre row groups e footer Parquet
ailake-catalog/   catálogo Iceberg: HadoopCatalog (filesystem), RestCatalog (Polaris/Nessie/S3 Tables/Unity Catalog), DatabricksAuth helpers
ailake-store/     abstração de storage: LocalStore + ObjectStoreBackend (S3/GCS/Azure via object_store)
ailake-query/     TableWriter, search() com pruning geométrico, ContextAssembler, CompactionExecutor
ailake-py/        bindings PyO3 (fora do workspace — compilar com maturin)
tests/            testes de integração e compatibilidade
```

---

## Troubleshooting

**`error: linker 'cc' not found`**
```bash
# Ubuntu/Debian
sudo apt install build-essential
```

**`import pyarrow` falha**
```bash
pip install pyarrow
# ou com conda:
conda install pyarrow
```

**`import ailake` falha após `maturin develop`**
```bash
# Verificar que está no diretório ailake-py e no venv correto
cd ailake-py
maturin develop --release
python3 -c "import ailake"
```

**`cargo test` falha em `pyarrow_ignores_ailake_footer`**
Esse teste requer `python3` + `pyarrow`. Execute com `--ignored`:
```bash
cargo test -p tests --test parquet_trailing_bytes -- --ignored
```

**Benchmark falha com `E0601`**
Certifique-se de estar na branch `main` ou `develop` (benches vazios foram corrigidos em `e382e83`).

**`pruning_threshold` remove todos os resultados**
Threshold muito baixo corta arquivos legítimos. Use `f32::INFINITY` para desativar pruning e debugar:
```rust
SearchConfig { top_k: 10, ef_search: 50, pruning_threshold: f32::INFINITY }
```
