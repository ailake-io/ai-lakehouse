# SETUP.md — Testando o AI-Lake Format localmente

Guia para rodar o formato de arquivo localmente: escrever batches, busca vetorial, inspeção de layout e verificação de compatibilidade Parquet.

---

## Pré-requisitos

| Ferramenta | Versão mínima | Instalação |
|---|---|---|
| Rust + Cargo | 1.75+ (stable) | `curl https://sh.rustup.rs -sSf \| sh` |
| Python 3 | 3.9+ | sistema / conda |
| PyArrow | qualquer | `pip install pyarrow` |

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

---

## 2. Rodar a suite de testes

```bash
# Testes unitários de todos os crates
cargo test --workspace --lib

# Testes de integração (write + read + search end-to-end)
cargo test -p ailake-tests -- --test-threads=1

# Teste de compatibilidade PyArrow (requer pyarrow instalado)
cargo test -p ailake-tests --test parquet_trailing_bytes -- --ignored
```

Todos devem terminar com `test result: ok`.

---

## 3. Demo completo — escrever, buscar, inspecionar

O exemplo `demo` (em `ailake-query/examples/demo.rs`) faz o fluxo completo em filesystem local:

1. Cria uma tabela AI-Lake com 2 arquivos (500 linhas cada)
2. Imprime o layout binário do arquivo (offsets de PAR1, AILK, HNSW)
3. Busca top-5 por similaridade cosine
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

## 4. Verificar compatibilidade PyArrow manualmente

Gerar um arquivo e lê-lo com PyArrow puro (sem SDK AI-Lake):

```bash
# Gerar arquivo temporário via teste
cargo test -p ailake-tests --test parquet_trailing_bytes -- --ignored --nocapture 2>&1 | grep -i "path\|file\|ok\|FAILED"
```

Ou escrever um script Python rápido apontando para um arquivo gerado pelo demo:

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

## 5. Rodar benchmarks

```bash
# HNSW search benchmark (ailake-index)
cargo bench -p ailake-index

# Write benchmark (ailake-file)
cargo bench -p ailake-file
```

---

## 6. Clippy e formatação

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Ambos devem terminar sem erros ou warnings.

---

## Estrutura dos crates

```
ailake-core/      tipos base: VectorMetric, VectorPrecision, RowId, erros
ailake-parquet/   leitura/escrita Parquet com coluna VECTOR
ailake-vec/       quantização F32 → F16
ailake-index/     HNSW (hnsw_rs) + serialização bincode
ailake-file/      arquivo unificado: splica AILK entre row groups e footer
ailake-catalog/   catálogo Iceberg: metadata.json + manifestos Avro
ailake-store/     abstração de storage (local + S3 na Fase 2)
ailake-query/     TableWriter, search(), ContextAssembler
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

**Benchmark falha com `E0601`**
Certifique-se de estar na branch `main` ou `develop` (não em commits antigos — benches vazios foram corrigidos em `e382e83`).
