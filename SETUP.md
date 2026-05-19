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

## 9. Clippy e formatação

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
ailake-catalog/   catálogo Iceberg: metadata.json + manifestos Avro, HadoopCatalog
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
