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

# Build padrão — CPU paralelo (rayon), sem dependência de CUDA
cargo build --workspace
```

Primeira compilação leva ~2-3 min (baixa dependências Arrow/Parquet).

### Variantes de build

```bash
# Cloud storage
cargo build --workspace --features store-s3      # Amazon S3
cargo build --workspace --features store-gcs     # Google Cloud Storage
cargo build --workspace --features store-azure   # Azure Blob
```

**Regra de ouro**: o build padrão (`cargo build`) compila e roda em qualquer
máquina — CPU-only, NVIDIA ou AMD. Não existe mais flag `ailake-index/gpu`.
Ambas as GPUs são detectadas em runtime via `libloading` sem dependência de
compilação.

---

## 2. Suite de testes completa

```bash
# Testes unitários de todos os crates
cargo test --workspace --lib

# Testes de integração (write + read + search end-to-end)
cargo test -p tests

# Todos de uma vez
cargo test --workspace
```

Deve terminar com `105 passed` (2 ignored — testes que requerem servidor externo).

### Testes por crate

| Crate | O que cobre |
|---|---|
| `ailake-vec` | Quantização F32→F16, PQ (encode/decode/ADC), BlockCompressor (zstd/lz4), centróides, `exact_distance`, SIMD (AVX2/NEON) |
| `ailake-index` | HNSW build/search, IVF-PQ (train/search/serialize), GPU k-means dispatch, serialização bincode, MmapLoader round-trip |
| `ailake-file` | Escrita/leitura do arquivo unificado, layout AILK (HNSW e IVF-PQ), integridade |
| `ailake-query` | ContextAssembler, pruning geométrico, reranking pós-PQ, MemTableWriter, write_batch_ivf_pq |
| `ailake-parquet` | write_batch_multi_vec / read_all_multi_vec (multi-vector columns) |
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

### 8A. SIFT-1M benchmark (ailake-bench)

Benchmark público com o dataset SIFT-1M (1M vetores 128-dim Euclidean).

**Download do dataset** (~164 MB):
```bash
bash ailake-bench/scripts/download_sift1m.sh
# Baixa em data/sift1m/
```

**Rodar:**
```bash
cargo run --release -p ailake-bench -- --dataset-dir data/sift1m
```

Resultado esperado (CPU com AVX2, parallel HNSW build):
```
Write phase
  Throughput    : ~2400 vec/s

Build phase  (10 shards, parallel)
  Build time    : ~155 s  (vs ~220 s single-threaded)

Search phase  (top_k=10, ef=50)  [indexes pre-loaded]
  Recall@10     : ~0.996
  QPS           : ~450
  Latency mean  : ~2.2 ms
  Latency p99   : ~4.5 ms
```

Opções:
```bash
# Limitar base a 10k vetores para smoke-test rápido
cargo run --release -p ailake-bench -- --dataset-dir data/sift1m --limit 10000

# Ajustar ef e top_k
cargo run --release -p ailake-bench -- --dataset-dir data/sift1m --top-k 10 --ef 100
```

### 8B. Benchmark LanceDB (comparação)

```bash
cargo run --release -p ailake-bench --features lancedb-bench -- \
    --dataset-dir data/sift1m --engine lancedb
```

Para comparar AI-Lake vs LanceDB side-by-side:

```bash
cargo run --release -p ailake-bench --features lancedb-bench -- \
    --dataset-dir data/sift1m --engine all
```

### 8C. Benchmark pgvector (comparação)

Requer PostgreSQL com a extensão `pgvector ≥ 0.5.0` (HNSW support).

**Subir PostgreSQL + pgvector via Docker:**
```bash
docker run -d --name pg-ailake \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Rodar benchmark pgvector sozinho:**
```bash
cargo run --release -p ailake-bench --features pgvector-bench -- \
    --dataset-dir data/sift1m \
    --engine pgvector \
    --pgvector-url "host=localhost user=postgres password=postgres dbname=postgres"
```

**Comparação completa AI-Lake + LanceDB + pgvector:**
```bash
cargo run --release -p ailake-bench --features lancedb-bench,pgvector-bench -- \
    --dataset-dir data/sift1m \
    --engine all \
    --pgvector-url "host=localhost user=postgres password=postgres dbname=postgres"
```

Parâmetros pgvector opcionais:
```bash
--pgvector-m 16              # HNSW m (default: 16)
--pgvector-ef-construction 64 # ef_construction (default: 64)
--pgvector-ef-search 50      # ef_search at query time (default: 50)
```

### 8D. Benchmark Deep Lake (Python, opcional)

Deep Lake free tier suporta apenas busca exata (brute-force). O script abaixo mede throughput de escrita e busca exata em um subconjunto:

```bash
pip install deeplake numpy
python3 ailake-bench/scripts/deeplake_bench.py \
    --dataset-dir data/sift1m \
    --limit 10000
```

> **Nota**: ANN aproximado (Deep Memory) requer plano pago da Activeloop. A comparação de recall com AI-Lake/pgvector/LanceDB não é direta.

### 8E. Microbenchmarks criterion

```bash
# HNSW search benchmark (ailake-index)
cargo bench -p ailake-index

# Write benchmark (ailake-file)
cargo bench -p ailake-file
```

---

## 8F. GPU search — NVIDIA CUDA e AMD ROCm

Dois backends GPU independentes. Ambos usam `libloading` dlopen em runtime —
**zero dependência de build** para qualquer GPU. O mesmo binário roda em
CPU-only, NVIDIA e AMD sem recompilação.

| Backend | Detecção | Compute | Requisito de build |
|---------|----------|---------|-------------------|
| **NVIDIA CUDA** | `libcuda.so.1` + `cuDeviceGetCount` | cuBLAS SGEMM via libloading | **nenhum** — runtime only |
| **AMD ROCm** | `libamdhip64.so` + `hipGetDeviceCount` | hipBLAS SGEMM via libloading | **nenhum** — runtime only |
| **CPU** | fallback | rayon paralelo | nenhum |

**Prioridade de detecção**: AMD ROCm → NVIDIA CUDA → CPU SIMD.
AMD é verificado primeiro pois sistemas ROCm frequentemente expõem uma camada de compatibilidade CUDA (`libcuda.so.1`), o que identificaria incorretamente como NVIDIA sem essa prioridade.

---

### 8F-1. NVIDIA CUDA

#### Pré-requisitos de runtime (nenhum de build)

| Requisito | Versão mínima |
|-----------|---------------|
| GPU NVIDIA | Arquitetura Maxwell+ (sm_50) |
| Driver NVIDIA | ≥ 450 |
| `libcudart.so` | CUDA 11.x ou 12.x em `LD_LIBRARY_PATH` |
| `libcublas.so` | CUDA 11.x ou 12.x em `LD_LIBRARY_PATH` |

Não é necessário CUDA Toolkit, `nvcc` ou headers — apenas as bibliotecas de runtime.

#### Build — nenhuma flag adicional necessária

```bash
# Build padrão — NVIDIA detectado e usado em runtime automaticamente
cargo build --release
```

O binário detecta CUDA via `libloading` (dlopen de `libcuda.so.1`,
`libcudart.so`, `libcublas.so`). Sem GPU: cai silenciosamente para AMD ROCm ou CPU.

#### Verificar detecção CUDA

```rust
use ailake_index::{detect_backend, HardwareBackend};
assert_eq!(detect_backend(), HardwareBackend::NvidiaCuda);
```

---

### 8F-2. AMD ROCm

#### Pré-requisitos

| Requisito | Versão mínima |
|-----------|---------------|
| GPU AMD | GCN4+ (Polaris) ou RDNA1+ recomendado |
| ROCm | 5.0+ (`libamdhip64.so` + `libhipblas.so`) |
| Driver amdgpu | incluído no ROCm |

#### Build — nenhuma flag adicional necessária

```bash
# Build padrão — ROCm detectado e usado em runtime automaticamente
cargo build --release

# ROCm + CUDA ao mesmo tempo (ROCm tem prioridade por ser detectado primeiro)
cargo build --release --features ailake-index/gpu
```

O backend ROCm usa `libhipblas.so` via `libloading` — sem dependência de compilação. O binário roda em qualquer máquina e apenas usa ROCm quando `libamdhip64.so` + `libhipblas.so` estiverem instalados.

#### Verificar detecção ROCm

```bash
# Via HardwareProfile::detect()
use ailake_index::{detect_backend, detect_rocm, HardwareBackend};
assert!(detect_rocm());
assert_eq!(detect_backend(), HardwareBackend::AmdRocm);
```

---

### 8F-3. Verificar detecção via benchmark

O engine `ailake-auto` imprime o perfil de hardware antes de rodar:

```bash
cargo run --release -p ailake-bench -- \
    --dataset-dir data/sift1m --engine ailake-auto --limit 50000
```

Saída esperada em máquina NVIDIA:
```
Hardware detection:
  Backend      : NVIDIA CUDA
  CUDA GPU     : true
  ROCm GPU     : false
  CPU cores    : 16
  AVX2         : true
  AVX-512F     : false
  Index chosen : IVF-PQ  (shard_size=100000)
```

Saída esperada em máquina AMD ROCm:
```
Hardware detection:
  Backend      : AMD ROCm
  CUDA GPU     : false
  ROCm GPU     : true
  CPU cores    : 16
  AVX2         : true
  AVX-512F     : false
  Index chosen : IVF-PQ  (shard_size=100000)
```

Saída esperada em CPU-only:
```
Hardware detection:
  Backend      : CPU (no GPU)
  CUDA GPU     : false
  ROCm GPU     : false
  CPU cores    : 8
  AVX2         : true
  AVX-512F     : false
  Index chosen : HNSW  (shard_size=100000)
```

---

### 8F-4. Quando GPU bate CPU

| Arquivo (vetores) | dim | CPU rayon | NVIDIA/AMD GPU |
|-------------------|-----|-----------|----------------|
| 10k | 1536 | ~2 ms | ~3 ms (overhead domina) |
| 100k | 1536 | ~20 ms | ~2 ms |
| 500k | 1536 | ~100 ms | ~4 ms |

GPU vence a partir de ~50k vetores/arquivo para dim=1536. Aplicável para ambos os vendors.

---

### 8F-5. Métricas suportadas em GPU

| Métrica | NVIDIA (cuBLAS SGEMM) | AMD (hipBLAS SGEMM) |
|---------|-----------------------|---------------------|
| **Cosine** | normalizar CPU → SGEMM → `1 + raw` | normalizar CPU → SGEMM → `1 + raw` |
| **Euclidean** | SGEMM (−2·q·d) + norms CPU → sqrt | SGEMM (−2·q·d) + norms CPU → sqrt |
| **DotProduct** | SGEMM com alpha=−1 | SGEMM com alpha=−1 |

Ambos os backends usam formulação SGEMM idêntica (`C[N×Q col-major] = alpha · db^T · queries`); diferem apenas nas constantes de operação (`CUBLAS_OP_T=1` vs `HIPBLAS_OP_T=112`) e nos nomes das bibliotecas.

---

### 8F-6. GPU k-means para IVF-PQ

O treinamento do `IvfPqIndex` (k-means para centróides grosseiros e `PQCodebook`) usa GPU via `kmeans_dispatch` com prioridade NVIDIA → AMD → CPU:

```
NVIDIA (runtime)  →  try_nvidia_kmeans (cuBLAS SGEMM via libloading)
AMD ROCm (runtime)  →  try_rocm_kmeans (hipBLAS SGEMM via libloading)
CPU fallback        →  kmeans_centroids (rayon paralelo)
```

- **Sem GPU em runtime**: `kmeans_dispatch` cai silenciosamente para CPU.
- **NVIDIA sem libcublas instalado**: mesmo que `libcuda.so.1` exista, `try_nvidia_kmeans` retorna `None` se `libcublas.so` não for encontrado → tenta AMD → CPU.
- **AMD sem hipBLAS instalado**: mesmo que `libamdhip64.so` exista, `try_rocm_kmeans` retorna `None` se `libhipblas.so` não for encontrado → CPU fallback.

```bash
# Build padrão — GPU acelerado automaticamente se disponível (NVIDIA ou AMD)
cargo build --release
```

---

## 8G. Benchmark IVF-PQ (`--engine ailake-ivf-pq`)

IVF-PQ usa índice com inverted lists codificadas por Product Quantization, em vez do grafo HNSW. Índice 10-100× menor — boa escolha para S3 onde acesso sequencial é mais barato.

```bash
# Benchmark básico (SIFT-1M, defaults: nlist=256, nprobe=8, pq_m=8)
cargo run --release -p ailake-bench -- --dataset-dir data/sift1m --engine ailake-ivf-pq

# Ajustar parâmetros
cargo run --release -p ailake-bench -- \
    --dataset-dir data/sift1m \
    --engine ailake-ivf-pq \
    --ivf-nlist 512 \
    --ivf-nprobe 16 \
    --ivf-pq-m 16

# Comparação completa AI-Lake HNSW + IVF-PQ + LanceDB + pgvector
cargo run --release -p ailake-bench --features lancedb-bench,pgvector-bench -- \
    --dataset-dir data/sift1m \
    --engine all \
    --pgvector-url "host=localhost user=postgres password=postgres dbname=postgres"
```

Parâmetros IVF-PQ:

| Flag | Default | Descrição |
|---|---|---|
| `--ivf-nlist` | 256 | Células Voronoi (inverted lists). Regra: `sqrt(n)` |
| `--ivf-nprobe` | 8 | Células consultadas por query. Maior = melhor recall |
| `--ivf-pq-m` | 8 | Sub-vetores PQ. Deve dividir 128 para SIFT |

Resultado esperado (SIFT-1M, CPU, nlist=256, nprobe=8):
```
AI-Lake IVF-PQ write phase (nlist=256, nprobe=8, pq_m=8) …
  Throughput    : ~1800 vec/s  (k-means mais lento que HNSW insert)

Search phase  (top_k=10)
  Recall@10     : ~0.94
  QPS           : ~2100
  Index size    : ~5 MB  (vs ~80 MB HNSW para 1M vetores dim=128)
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

## 15. Plugin Trino — VectorScanConnector

O `trino-plugin` expõe tabelas AI-Lake como um conector Trino nativo. Requer a biblioteca nativa `libailake_jni.so` (construída pelo Cargo) e o fat-jar do plugin (construído pelo Gradle).

### 15A. Pré-requisitos adicionais

| Ferramenta | Versão | Instalação |
|---|---|---|
| JDK | 17+ | `sudo apt install openjdk-17-jdk` |
| Gradle | 8+ | `sdk install gradle` (ou usar wrapper) |
| Trino server | 430+ | [trino.io/download](https://trino.io/download.html) |

### 15B. Passo 1 — Compilar a biblioteca nativa

```bash
# Na raiz do projeto
cargo build --release -p ailake-jni

# Linux:
ls -lh target/release/libailake_jni.so

# macOS:
ls -lh target/release/libailake_jni.dylib
```

### 15C. Passo 2 — Compilar o fat-jar do plugin

```bash
cd trino-plugin

# Criar wrapper Gradle (só na primeira vez)
gradle wrapper

# Compilar
./gradlew shadowJar

# Saída:
ls build/libs/trino-plugin-0.1.0-plugin.jar
```

### 15D. Passo 3 — Instalar no Trino

```bash
# Diretório de instalação do Trino (ajuste conforme seu ambiente)
TRINO_HOME=/opt/trino

# Criar diretório do plugin e copiar jar
mkdir -p $TRINO_HOME/plugin/ailake
cp build/libs/trino-plugin-0.1.0-plugin.jar $TRINO_HOME/plugin/ailake/

# Colocar a biblioteca nativa no library path do Trino
# Opção A: copiar para lib/ do Trino
cp ../target/release/libailake_jni.so $TRINO_HOME/lib/

# Opção B: definir no jvm.config do Trino
echo "-Djava.library.path=/path/to/target/release" >> $TRINO_HOME/etc/jvm.config
```

### 15E. Passo 4 — Configurar o catálogo

Criar o arquivo `$TRINO_HOME/etc/catalog/ailake.properties`:

```properties
# Conector AI-Lake
connector.name=ailake

# URI da tabela AI-Lake (filesystem local ou s3://)
ailake.table-uri=/tmp/ailake-demo

# Coluna vetorial e dimensão (devem bater com a tabela)
ailake.vector-column=embedding
ailake.vector-dim=64
```

Para usar com tabela gerada pelo demo (seção 4):

```bash
# Gerar tabela de demo primeiro
cargo run --example demo -p ailake-query 2>&1 | grep "Workspace:"
# Workspace: /tmp/ailakeXXXXXX

# Usar o caminho impresso acima no properties:
# ailake.table-uri=/tmp/ailakeXXXXXX/warehouse/default/demo_table
```

### 15F. Passo 5 — Iniciar o Trino e fazer a primeira busca

```bash
# Iniciar o servidor
$TRINO_HOME/bin/launcher start

# Conectar via CLI
$TRINO_HOME/bin/trino
```

No prompt do Trino:

```sql
-- 1. Verificar que o conector está ativo
SHOW SCHEMAS FROM ailake;
-- default

SHOW TABLES FROM ailake.default;
-- search

DESCRIBE ailake.default.search;
-- row_id    | bigint  | ...
-- distance  | double  | ...
-- file_path | varchar | ...

-- 2. Definir o vetor de consulta (valores separados por vírgula, 64 dims para o demo)
SET SESSION ailake.query_vector = '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4';

SET SESSION ailake.top_k = 5;

-- 3. Executar busca vetorial
SELECT row_id, distance, file_path
FROM ailake.default.search
ORDER BY distance;

-- Resultado esperado:
--  row_id | distance  | file_path
-- --------+-----------+------------------------------
--       0 | 0.000000  | data/part-00000.parquet
--      12 | 0.031456  | data/part-00000.parquet
--     ...

-- 4. Combinar com Iceberg (JOIN com dados tabulares via conector Iceberg padrão)
-- A tabela AI-Lake retorna row_ids; faça JOIN com a tabela Iceberg para obter o texto
SELECT s.row_id, s.distance, i.chunk_text
FROM ailake.default.search s
JOIN iceberg.default.demo_table i ON s.row_id = i.id
ORDER BY s.distance
LIMIT 5;
```

### 15G. Testes do plugin Trino

```bash
cd trino-plugin
./gradlew test

# Saída esperada:
# VectorScanMetadataTest: 7 tests passed
# VectorScanConnectorTest: 7 tests passed
# VectorScanSplitManagerTest: 5 tests passed
# VectorScanRecordSetTest: 9 tests passed
# AilakeNativeTest: 5 tests passed
```

Os testes rodam sem Trino server e sem biblioteca nativa — degradação graciosa garantida.

---

## 16. Plugin Spark — VectorScanStrategy

O `spark-plugin` registra uma `SparkStrategy` customizada no planner Catalyst do Spark 3.5. Converte planos `VectorSearchPlan` em `VectorScanExec` que chama a biblioteca nativa via JNA.

### 16A. Pré-requisitos adicionais

Mesmo JDK 17+ e Gradle do item 15A. Spark 3.5 local (para testes), ou cluster Spark já configurado.

```bash
# Spark local (opcional — para testar)
curl -LO https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar xf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=$(pwd)/spark-3.5.0-bin-hadoop3
```

### 16B. Passo 1 — Biblioteca nativa (mesma da seção 15B)

```bash
cargo build --release -p ailake-jni
# target/release/libailake_jni.so  (Linux)
```

### 16C. Passo 2 — Compilar o fat-jar do plugin

```bash
cd spark-plugin
gradle wrapper   # só na primeira vez
./gradlew shadowJar

ls build/libs/spark-plugin-0.1.0-plugin.jar
```

### 16D. Passo 3 — Gerar tabela de demo

Usar a tabela gerada pelo demo da seção 4:

```bash
# Gerar tabela e anotar o path
cargo run --example demo -p ailake-query 2>&1 | grep "Workspace:"
# Workspace: /tmp/ailakeXXXXXX

export AILAKE_TABLE=/tmp/ailakeXXXXXX/warehouse/default/demo_table
```

### 16E. Passo 4 — spark-shell (Scala interativo)

```bash
$SPARK_HOME/bin/spark-shell \
  --jars $(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$(pwd)/target/release" \
  --conf spark.ui.enabled=false
```

No prompt `scala>`:

```scala
import io.ailake.spark.implicits._

// Tabela gerada pelo demo (dim=64, 1000 linhas)
val tableUri = sys.env("AILAKE_TABLE")

// Vetor de consulta — igual ao primeiro embedding do demo
val query: Array[Float] = Array.fill(64)(0.5f)

// Busca vetorial — retorna DataFrame com (row_id, distance, file_path)
val results = spark.ailakeSearch(
  tableUri    = tableUri,
  queryVector = query,
  topK        = 10,
)

results.show(10, truncate = false)
// +------+-----------+----------------------------+
// |row_id|distance   |file_path                   |
// +------+-----------+----------------------------+
// |0     |0.0        |data/part-00000.parquet     |
// |12    |0.031456   |data/part-00000.parquet     |
// |87    |0.044123   |data/part-00001.parquet     |
// ...

// Ordenar por distância e mostrar top-5
results.orderBy("distance").limit(5).show()

// Schema retornado
results.printSchema()
// root
//  |-- row_id: long (nullable = false)
//  |-- distance: double (nullable = false)
//  |-- file_path: string (nullable = false)
```

### 16F. Passo 4 alternativo — PySpark

```bash
$SPARK_HOME/bin/pyspark \
  --jars $(pwd)/spark-plugin/build/libs/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=$(pwd)/target/release"
```

```python
# PySpark — chamar lógica Scala via py4j
jvm = spark._jvm

# Instanciar o VectorSearchPlan diretamente via py4j
float_array = jvm.Array(jvm.Float.TYPE, 64)
for i in range(64):
    float_array[i] = 0.5

table_uri = "/tmp/ailakeXXXXXX/warehouse/default/demo_table"

# Chamar AilakeNative diretamente (sem passar pelo planner Spark)
native = jvm.io.ailake.spark.AilakeNative
results_java = native.search(table_uri, float_array, 10)

# Converter para Python
results = [
    {"row_id": r.rowId(), "distance": r.distance(), "file_path": r.filePath()}
    for r in results_java
]
for r in results:
    print(f"row_id={r['row_id']}  distance={r['distance']:.6f}  file={r['file_path']}")

# Ou usar ailake-py (recomendado para Python):
import ailake
query = [0.5] * 64
results = ailake.search(path=table_uri, query=query, top_k=10)
```

### 16G. Passo 5 — submeter job Spark (cluster)

```scala
// MyVectorSearchJob.scala
import io.ailake.spark.implicits._
import org.apache.spark.sql.SparkSession

object MyVectorSearchJob {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ailake-vector-search")
      .config("spark.sql.extensions", "io.ailake.spark.AilakeSparkExtensions")
      .getOrCreate()

    val tableUri  = args(0)  // s3://my-lake/docs/
    val topK      = args(1).toInt
    val queryJson = args(2)  // "[0.1, -0.2, 0.3, ...]"

    val query: Array[Float] =
      ujson.read(queryJson).arr.map(_.num.toFloat).toArray

    spark.ailakeSearch(tableUri, query, topK)
      .write.parquet(args(3))  // output path
  }
}
```

```bash
spark-submit \
  --jars spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/opt/ailake/lib" \
  my-vector-search-job.jar \
  "s3://my-lake/docs/" 100 "[0.021, -0.043, ...]" "s3://results/out/"
```

**Importante**: `libailake_jni.so` precisa estar disponível em todos os executores. Adicionar ao `--files` do spark-submit ou instalar via bootstrap script.

### 16H. Testes do plugin Spark

```bash
cd spark-plugin
./gradlew test

# Saída esperada:
# VectorSearchPlanTest: 8 tests passed
# VectorScanStrategyTest: 6 tests passed
# AilakeNativeTest: 4 tests passed
# AilakeSparkExtensionsTest: 5 tests passed  (requer SparkSession local)
```

Os testes de integração (`AilakeSparkExtensionsTest`) iniciam um SparkSession local — levam ~15s na primeira execução.

---

## 17. Plugin Flink — AilakeVectorConnector

O `ailake-flink` é um conector Flink Table API (Kotlin/Gradle) que usa a biblioteca nativa `libailake_jni.so` via JNA para leitura e escrita vetorial em tabelas AI-Lake.

### 17A. Pré-requisitos adicionais

| Ferramenta | Versão | Instalação |
|---|---|---|
| JDK | 17+ | `sudo apt install openjdk-17-jdk` |
| Gradle | 8+ | `sdk install gradle` (ou usar wrapper) |
| Apache Flink | 1.18+ | [flink.apache.org](https://flink.apache.org/downloads/) |

### 17B. Passo 1 — Compilar a biblioteca nativa

```bash
# Na raiz do projeto (mesma lib do plugin Trino/Spark)
cargo build --release -p ailake-jni

# Linux:
ls -lh target/release/libailake_jni.so
```

### 17C. Passo 2 — Compilar o fat-jar

```bash
cd ailake-flink
gradle wrapper   # só na primeira vez
./gradlew shadowJar

ls build/libs/ailake-flink-0.1.0-plugin.jar
```

O shadow jar (`-plugin`) inclui JNA e Jackson. Dependências Flink ficam fora (são `compileOnly`).

### 17D. Registrar o conector no Flink (SQL Client ou DataStream)

```sql
-- Flink SQL Client
ADD JAR '/path/to/ailake-flink-0.1.0-plugin.jar';

-- Tabela source + sink
CREATE TABLE docs (
  id        BIGINT,
  text      STRING,
  embedding BYTES,
  _distance FLOAT   -- preenchido pelo vector search, ignorado em writes
) WITH (
  'connector'        = 'ailake',
  'warehouse'        = 's3://my-lake/',
  'namespace'        = 'default',
  'table-name'       = 'docs',
  'vector.column'    = 'embedding',
  'vector.dim'       = '1536',
  'vector.metric'    = 'cosine',
  'vector.precision' = 'f16',
  'search.top-k'     = '10',
  'search.ef'        = '50'
);

SELECT id, text, _distance FROM docs;
```

O vetor de consulta é passado como parâmetro do job (`ailake.query.vector` — floats separados por vírgula):

```bash
flink run \
  -D "pipeline.global-job-parameters=ailake.query.vector=0.021,-0.043,0.118,..." \
  my-pipeline.jar
```

Para streaming ingestion (sink):

```sql
INSERT INTO docs
SELECT id, chunk_text, embedding FROM kafka_source;
```

O sink (`AilakeSinkFunction`) acumula 10.000 linhas e chama `AilakeNativeLoader.writeBatch()` ao fazer flush.

### 17E. Testes do plugin Flink

```bash
cd ailake-flink
./gradlew test

# Saída esperada:
# AilakeVectorConnectorFactoryTest: tests passed
```

Testes rodam sem servidor Flink nem biblioteca nativa — degradação graciosa via mock JNA.

---

## 18. Multi-vector columns (`write_batch_multi_vec`)

Armazena múltiplos embeddings por linha como `List<FixedSizeBinary>` — útil quando um documento tem vários chunks e você quer evitar explosão de linhas.

### 18A. Escrever com múltiplos vetores por linha

```rust
use ailake_parquet::writer::write_batch_multi_vec;

// 3 documentos, cada um com 2 embeddings de dim=64
let texts = vec!["doc A".to_string(), "doc B".to_string(), "doc C".to_string()];
let multi_embeddings: Vec<Vec<Vec<f32>>> = vec![
    vec![vec![0.1_f32; 64], vec![0.2_f32; 64]],  // doc A: 2 vecs
    vec![vec![0.3_f32; 64]],                       // doc B: 1 vec
    vec![vec![0.4_f32; 64], vec![0.5_f32; 64]],   // doc C: 2 vecs
];

let batch = write_batch_multi_vec(&texts, &multi_embeddings, "embedding", 64)?;
```

### 18B. Ler de volta

```rust
use ailake_parquet::reader::read_all_multi_vec;

let (texts, multi_vecs) = read_all_multi_vec(&parquet_bytes, "embedding", 64)?;
// multi_vecs[i] = Vec<Vec<f32>> para o documento i
```

### 18C. O que leitores Parquet padrão veem

Leitores sem plugin AI-Lake (Spark, Trino, DuckDB) leem a coluna como `List<BINARY>` — bytes opacos, sem erro. A semântica de vetor só é ativada pelo SDK.

---

## 19. MemTableWriter — buffer para streaming ingestion

`MemTableWriter` acumula micro-batches em RAM e faz flush para um único shard Parquet quando os thresholds de tamanho/linhas/tempo são atingidos. Reduz a frequência de builds HNSW em pipelines Flink/Spark Streaming.

### 19A. Uso básico

```rust
use ailake_query::mem_table::{MemTableWriter, MemTableConfig};
use std::time::Duration;

let config = MemTableConfig {
    flush_size_bytes: 128 * 1024 * 1024,  // flush após 128 MiB
    flush_max_rows:   200_000,             // flush após 200k linhas
    flush_interval:   Duration::from_secs(60), // flush após 60s inativo
};

let mut mt = MemTableWriter::new(catalog, store, policy, table, config);

// Loop de ingestão streaming
loop {
    let (batch, embeddings) = receive_micro_batch().await;
    mt.insert(&batch, &embeddings).await?;

    // Flush automático se threshold atingido
    if let Some(snap_id) = mt.flush_if_due().await? {
        println!("Flushed snapshot {snap_id}");
    }
}

// Flush final ao encerrar o job
let snap_id = mt.flush().await?;
```

### 19B. Thresholds padrão

| Threshold | Default | Trigger |
|---|---|---|
| `flush_size_bytes` | 64 MiB | Bytes acumulados de embeddings |
| `flush_max_rows` | 100,000 | Linhas acumuladas |
| `flush_interval` | 30s | Tempo desde último flush |

`flush_if_due()` verifica os três. `flush()` força independente dos thresholds.

### 19C. Inspecionar estado do buffer

```rust
println!("Buffered: {} rows, {} bytes", mt.buffered_rows(), mt.buffered_bytes());
if mt.is_full() { mt.flush().await?; }
```

---

## Estrutura dos crates

```
ailake-core/      tipos base: VectorMetric, VectorPrecision, RowId, AilakeError
ailake-parquet/   leitura/escrita Parquet com coluna VECTOR; write_batch_multi_vec / read_all_multi_vec
ailake-vec/       quantização F32→F16, PQ (PQCodebook), BlockCompressor, SIMD distances (AVX2/NEON), centróides
ailake-index/     HNSW + IVF-PQ (AnyIndex enum); hardware detection (HardwareBackend: NvidiaCuda/AmdRocm/CpuSimd);
                  NVIDIA via cuBLAS libloading (runtime, no build flag); AMD ROCm via hipBLAS libloading (runtime, no build flag);
                  kmeans_dispatch: NVIDIA → ROCm → CPU; bincode, MmapLoader (memmap2)
ailake-file/      arquivo unificado: AILK suporta IndexType::Hnsw e IndexType::IvfPq
ailake-catalog/   catálogo Iceberg: HadoopCatalog, RestCatalog, NessieCatalog, JdbcCatalog, GlueCatalog
ailake-store/     abstração de storage: LocalStore + ObjectStoreBackend (S3/GCS/Azure via object_store)
ailake-query/     TableWriter (write_batch, write_batch_ivf_pq, write_batch_multi), MemTableWriter,
                  search() com pruning geométrico, ContextAssembler, CompactionExecutor
ailake-bench/     benchmark SIFT-1M: HNSW, IVF-PQ (--engine ailake-ivf-pq), LanceDB, pgvector
ailake-py/        bindings PyO3 (fora do workspace — compilar com maturin)
ailake-jni/       bindings uniffi C-ABI para Spark, Trino e Flink
spark-plugin/     Spark 3.5 Catalyst strategy (Kotlin/Gradle)
trino-plugin/     Trino connector (Java/Gradle)
ailake-flink/     Flink Table API connector (Kotlin/Gradle)
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

**Plugin Trino: `UnsatisfiedLinkError: libailake_jni.so`**
A biblioteca nativa não está no `java.library.path` do Trino.
```bash
# Verificar onde o Trino procura
grep "java.library.path" $TRINO_HOME/etc/jvm.config

# Adicionar o caminho
echo "-Djava.library.path=/path/to/target/release" >> $TRINO_HOME/etc/jvm.config

# Reiniciar o Trino
$TRINO_HOME/bin/launcher restart
```

**Plugin Trino: `ailake.table-uri is required`**
A propriedade não foi definida no arquivo de catálogo.
```bash
cat $TRINO_HOME/etc/catalog/ailake.properties
# Verificar que contém: ailake.table-uri=...
```

**Plugin Trino: `SELECT` retorna 0 linhas**
Session property `query_vector` está vazia. Verificar:
```sql
SHOW SESSION LIKE 'ailake%';
SET SESSION ailake.query_vector = '0.1,0.2,0.3,...';
```

**Plugin Spark: `ClassNotFoundException: io.ailake.spark.AilakeSparkExtensions`**
O jar do plugin não foi passado para o Spark.
```bash
# Verificar que --jars inclui o plugin
spark-shell --jars /caminho/spark-plugin-0.1.0-plugin.jar \
  --conf spark.sql.extensions=io.ailake.spark.AilakeSparkExtensions
```

**Plugin Spark: `ailakeSearch` não encontrado**
O import dos implicits está faltando.
```scala
// Adicionar no início do script
import io.ailake.spark.implicits._
```

**Plugin Spark: VectorScanExec retorna DataFrame vazio**
Comportamento esperado em ambiente sem `libailake_jni.so` — degradação graciosa.
Para habilitar busca real, garantir que a lib está no `java.library.path`:
```bash
spark-shell \
  --conf "spark.driver.extraJavaOptions=-Djava.library.path=/path/to/target/release" \
  --conf "spark.executor.extraJavaOptions=-Djava.library.path=/path/to/target/release"
```

**`SearchConfig` falha compilação com `missing field rerank_factor`**
Adicione o campo faltante (introduzido na Phase 4):
```rust
SearchConfig {
    top_k: 10,
    ef_search: 50,
    pruning_threshold: 0.8,
    rerank_factor: None,  // Some(3) para habilitar reranking
}
```

**NVIDIA GPU disponível mas busca usa CPU (`try_nvidia_search_batch` retorna `None`)**
Runtime libraries ausentes ou fora do `LD_LIBRARY_PATH`. Não é necessário CUDA Toolkit — apenas as libs de runtime:
```bash
# Ubuntu — instalar apenas runtime (sem nvcc/headers)
sudo apt install libcuda1 libcublas-12-x

# Ou via CUDA keyring (canonical)
sudo apt install cuda-libraries-12-x

# Verificar presença
ldconfig -p | grep -E "libcuda|libcublas"
ls /usr/local/cuda/lib64/libcudart.so*

# Exportar se não estiver no path padrão
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**GPU build compila mas busca usa CPU (log não mostra "NVIDIA" ou "AMD")**
`detect_backend()` retornou `CpuSimd`. Verificar:
```bash
nvidia-smi                        # GPU NVIDIA visível?
rocm-smi                          # GPU AMD visível?
ls /dev/nvidia*                   # Dispositivos NVIDIA presentes?
ls /dev/kfd                       # Dispositivo AMD KFD presente?
echo $CUDA_VISIBLE_DEVICES        # Não deve ser "NoDevFiles"
```
Para NVIDIA: `libcuda.so.1` deve existir (`ldconfig -p | grep libcuda`).
Para AMD: `libamdhip64.so` e `libhipblas.so` devem existir (`ldconfig -p | grep -E "hip|hsa"`).

**AMD ROCm detectado mas busca usa CPU**
`detect_rocm()` retorna `true` (HIP runtime ok) mas `try_rocm_search_batch` retorna `None`.
Causa mais comum: `libhipblas.so` não instalado (ROCm Math Libraries são opcionais).
```bash
# Verificar presença do hipBLAS
ldconfig -p | grep hipblas

# Instalar hipBLAS (Ubuntu/ROCm PPA)
sudo apt install hipblas
# ou
sudo apt install rocm-libs
```

**`ailake-auto` mostra `Backend: NVIDIA CUDA` em máquina AMD**
ROCm com camada de compatibilidade CUDA instalada — `libcuda.so.1` existe (fornecida pelo pacote `hip-runtime-amd`). Neste caso, se `libamdhip64.so` também existe, o backend correto `AmdRocm` já é escolhido (AMD é verificado primeiro). Se ainda aparece NVIDIA, verificar:
```bash
ldconfig -p | grep -E "libamdhip|libcuda"
# libamdhip64.so deve aparecer para ROCm ser detectado como AmdRocm
```

**Plugin Flink: `ClassNotFoundException: io.ailake.flink.AilakeVectorConnectorFactory`**
O jar do plugin não foi adicionado ao Flink.
```bash
# SQL Client — adicionar antes de CREATE TABLE
ADD JAR '/path/to/ailake-flink-0.1.0-plugin.jar';

# ou via flink-conf.yaml
classloader.parent-first-patterns.additional: io.ailake
```

**Plugin Flink: conector registrado mas sink não persiste**
`libailake_jni.so` não está no `java.library.path` do TaskManager:
```bash
# Adicionar em flink-conf.yaml
env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib
```

**`IvfPqConfig` — `pq_m` deve dividir `dim`**
Build falha com `"pq_m X does not divide dim Y"`. Usar `IvfPqConfig::for_dim(dim)` para derivar valores válidos automaticamente:
```rust
let cfg = IvfPqConfig::for_dim(1536);  // pq_m=96, nlist=256, nprobe=8
```

**`MemTableWriter::flush()` retorna erro após `insert` vazio**
Chamar `flush()` sem nenhum `insert` prévio retorna `Err(EmptyBatch)`. Verificar `buffered_rows() > 0` antes:
```rust
if mt.buffered_rows() > 0 {
    mt.flush().await?;
}
```
