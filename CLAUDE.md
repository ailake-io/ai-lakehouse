# CLAUDE.md — AI-Lake Format

> **For AI assistants**: This file is the primary architecture context for the AI-Lake project. It documents fixed decisions, the physical file format, storage strategy, catalog backends, and implementation phases. Read this before answering questions about the codebase. The content below is written in Portuguese — use it as authoritative technical context regardless of language.

> **Contexto do Projeto**: Formato de arquivo auto-contido para AI-Lakehouse, escrito em **Rust**, 100% compatível com o ecossistema Apache Iceberg, unificando dados tabulares, embeddings e índice HNSW em um único arquivo físico no S3/Data Lake.

---

## Decisões Arquiteturais Fixadas

| Decisão | Escolha | Justificativa |
|---|---|---|
| **Linguagem core** | **Rust** | Zero-cost abstractions, segurança de memória sem GC, ideal para I/O de vetores em alta frequência |
| **Layout físico** | **Arquivo único auto-contido** | Dados + vetores + índice HNSW no mesmo arquivo Parquet estendido — fonte única da verdade |
| **Compatibilidade** | **Apache Iceberg Spec v2** | Qualquer framework com leitor Iceberg lê tabelas AI-Lake sem modificação |
| **Indexação vetorial** | **`hnsw_rs`** | HNSW puro em Rust, sem deps C++, serialização nativa |
| **Engine de I/O** | **`parquet-rs` + `arrow-rs`** | Base oficial Apache para colunar em Rust |
| **Carga do índice** | **`memmap2`** | mmap do rodapé HNSW sem carregar arquivo inteiro na RAM |
| **Serialização do grafo** | **`bincode`** | Serialização binária ultra-rápida do HNSW |
| **Object storage** | **`object_store`** | Abstração unificada S3 / Azure Blob / GCS em Rust |
| **Async runtime** | **`tokio`** | Leitura concorrente de múltiplos arquivos no S3 |
| **Binding Python** | **PyO3** | Expõe o core Rust como módulo Python nativo (`import ailake`) |
| **Binding JVM** | **JNA + C-ABI** | `ailake-jni` exporta `ailake_search_json` / `ailake_write_batch_json` como C-ABI (`#[no_mangle]`); plugins Trino/Spark/Flink carregam via JNA. Sem geração de código — API única compartilhada por todos os plugins JVM. |

---

## 1. Visão Geral da Arquitetura

### Filosofia Central
O **AI-Lake Format** é um **formato de arquivo único** projetado para ser a camada de persistência de Lakehouses que precisam servir simultaneamente workloads de **Business Intelligence** (consultas SQL analíticas) e **IA Generativa** (busca por similaridade vetorial para RAG, recomendação e fine-tuning de LLMs).

A premissa central é: **dados tabulares, embeddings e índice de busca residem no mesmo arquivo físico**, eliminando a fragmentação entre Data Lake e bancos vetoriais externos (Pinecone, Milvus). Uma única fonte da verdade, governada pelas transações ACID do Iceberg.

### Princípio de Compatibilidade Iceberg

**Um leitor Iceberg padrão deve conseguir ler qualquer tabela AI-Lake sem nenhum plugin.**

Isso significa:
- O `metadata.json` e os manifestos Avro seguem rigorosamente a **Iceberg Spec v2**.
- Os arquivos de dados são **Parquet válidos** — o índice HNSW vive no rodapé do arquivo, em uma seção que leitores Parquet padrão ignoram naturalmente.
- A coluna de vetores é encodada como `FIXED_LEN_BYTE_ARRAY` com metadados de campo custom, que leitores legados leem como bytes opacos sem erro.
- O **AI-Lake Rust SDK** é o único componente capaz de ativar o Vector-Scan. Outros frameworks continuam funcionando via Iceberg padrão, lendo dados colunares normalmente.

---

## 2. Estrutura do Arquivo Físico

O formato é um **arquivo Parquet completamente válido**. A seção AI-Lake (HNSW + centróide) é inserida entre os row groups e o footer Parquet — invisível para leitores padrão porque os offsets dos row groups no footer apontam para antes da seção AILK:

```
┌─────────────────────────────────────────────────────────────────┐
│  PARQUET HEADER (4 bytes: "PAR1")                               │
├─────────────────────────────────────────────────────────────────┤
│  BLOCO DE DADOS COLUNARES                                       │
│  - Colunas tradicionais (id, texto, metadados, timestamps)      │
│  - Coluna VECTOR como FIXED_LEN_BYTE_ARRAY (F16 quantizado)     │
│  - Row groups Parquet padrão                                    │
│  ← offsets do footer Parquet apontam para cá                    │
├─────────────────────────────────────────────────────────────────┤
│  ▼ SEÇÃO AI-LAKE (invisível para leitores Parquet padrão) ▼     │
│                                                                 │
│  AILAKE HEADER (64 bytes: magic "AILK", versão, offsets)        │
│  CENTRÓIDE + RAIO (dim×4 + 4 bytes, F32)                        │
│  GRAFO HNSW SERIALIZADO (bincode)                               │
│  AILAKE TRAILER (24 bytes: footer_offset, footer_len, "AILK")   │
├─────────────────────────────────────────────────────────────────┤
│  PARQUET FOOTER (schema, statistics, key_value_metadata)        │
│  - field metadata: ailake.dim, ailake.metric, ailake.precision  │
│  - file metadata: ailake.footer_offset  ← bootstrap do leitor   │
│  - footer_len (4 bytes, little-endian)                          │
│  - 4 bytes: "PAR1"   ← últimos 4 bytes do arquivo              │
└─────────────────────────────────────────────────────────────────┘
```

### Garantia de Compatibilidade Iceberg/Parquet

Leitores Parquet padrão (PyIceberg, Spark, Trino, DuckDB) lêem o footer do **fim do arquivo**, seguem os offsets dos row groups diretamente e nunca varrem a seção AILK. A extensão AI-Lake é **invisível e inofensiva** — garantia hard do spec, não quality-of-implementation:

| Componente | Spark / Trino / DuckDB (sem plugin) | AI-Lake Rust SDK |
|---|---|---|
| Parquet header + dados | Lê normalmente | Lê normalmente |
| Coluna VECTOR (`FIXED_LEN_BYTE_ARRAY`) | Lê como bytes | Decodifica F16 → F32 |
| Parquet footer | Lê normalmente | Lê + extrai `ailake.hnsw_offset` |
| Rodapé AI-Lake (após PAR1 final) | **Ignora** (fim do Parquet) | Carrega via mmap |

### Layout do diretório de tabela

```
s3://my-lake/my_table/
├── metadata/
│   ├── v3.metadata.json            # Iceberg Spec v2 válido
│   └── snap-001.avro               # Manifesto Iceberg padrão (Avro)
└── data/
    ├── part-00001.parquet          # Parquet estendido (dados + HNSW no rodapé)
    └── part-00002.parquet          # Cada arquivo é auto-contido
```

Não há mais diretórios `vectors/` ou `indexes/` separados. **Cada arquivo `.parquet` carrega seu próprio índice HNSW**, eliminando a complexidade de manter três tipos de arquivo sincronizados.

---

## 3. Particionamento Geométrico e Vector Pruning

### Conceito

A poda de arquivos no AI-Lake usa **Particionamento Geométrico**: em vez de podar por `min/max` de colunas (Iceberg tradicional), poda por **distância geométrica** ao centróide vetorial do arquivo.

### Como funciona

**Na escrita**:
1. O motor calcula o centróide (vetor médio) e o raio máximo (maior distância de qualquer vetor ao centróide) do lote.
2. Essas estatísticas vão para o **key-value metadata do manifesto Iceberg** (`snap-*.avro`), em campos custom prefixados com `ailake.*`.

**Na leitura**:
1. O leitor calcula a distância entre o vetor de consulta e o centróide de cada arquivo (informação já disponível no manifesto, sem abrir arquivos).
2. Se `distance(query, centroid) - radius > search_threshold`, o arquivo é descartado.
3. Apenas arquivos "quentes" (com vetores próximos) são baixados do S3.

### Estatísticas no manifesto Iceberg

As estatísticas vetoriais são armazenadas em `properties` (Iceberg Spec v2 permite key-value metadata custom):

```json
{
  "format-version": 2,
  "table-uuid": "...",
  "properties": {
    "ailake.format-version": "1",
    "ailake.vector-column": "embedding",
    "ailake.vector-dim": "1536",
    "ailake.vector-metric": "cosine",
    "ailake.vector-precision": "f16"
  }
}
```

E por arquivo, em `custom-properties` do snapshot manifest entry:

```
File: data/part-00001.parquet
  ailake.centroid: [0.021, -0.043, 0.118, ...]   (base64-encoded f32 array)
  ailake.radius: 0.342
  ailake.record-count: 50000
  ailake.footer-offset: 12582912                  (byte offset absoluto do AILK header no arquivo)
```

> **Impacto**: Em tabelas com 10.000 arquivos Parquet e embeddings de 1536 dimensões, a poda por centróide pode eliminar 95–99% dos arquivos antes de qualquer I/O real de dados — sem nem abrir o Parquet.

---

## 4. Carga do Índice HNSW via mmap

### Por que mmap

O grafo HNSW serializado tem tipicamente 10-20% do tamanho dos vetores brutos. Para 50k vetores dim=1536 F16, o grafo ocupa ~15 MB. Carregar arquivos inteiros do S3 só para acessar o rodapé seria desperdício.

### Estratégia

1. **No S3**: leitor faz `GET` parcial usando `Range: bytes=N-M`, baixando apenas o footer Parquet primeiro.
2. **Lê do footer Parquet**: extrai `ailake.footer_offset` do key-value metadata — offset absoluto do AILK header no arquivo.
3. **Segundo `GET`**: baixa o AILK header (64 bytes) + seção de centróide para pruning rápido.
4. **Terceiro `GET`** (se não pruned): baixa apenas os bytes do grafo HNSW via offsets do header.
5. **Localmente**: salva os bytes em arquivo temporário e abre via `memmap2::Mmap`.
6. **Deserializa via `bincode`**: o grafo HNSW é carregado preguiçosamente — apenas as páginas tocadas durante a busca são paginadas pelo SO.

```rust
// Pseudocódigo da carga do índice
let footer_bytes = store.get_range(path, parquet_footer_range).await?;
let parquet_meta = parquet::file::footer::parse_metadata(&footer_bytes)?;
let ailk_offset = parquet_meta.kv_metadata("ailake.footer_offset")?.parse::<u64>()?;

// Header + centroid fetch (cheap)
let header_bytes = store.get_range(path, ailk_offset..ailk_offset + HEADER_SIZE as u64).await?;
let header = AilakeHeader::from_bytes(&header_bytes)?;

// HNSW-only fetch
let hnsw_start = ailk_offset + header.hnsw_offset;
let hnsw_bytes = store.get_range(path, hnsw_start..hnsw_start + header.hnsw_len).await?;
let tmp_file = write_to_tmp(hnsw_bytes)?;
let mmap = unsafe { Mmap::map(&tmp_file)? };
let hnsw: HnswIndex = bincode::deserialize(&mmap[..])?;
```

---

## 5. Desafios de Engenharia

### 5A — Imutabilidade e Compaction

**Problema**: Arquivos Parquet/Iceberg são imutáveis. Não dá para inserir um vetor em um arquivo existente — o índice HNSW dentro dele estaria desatualizado.

**Solução**: Como cada arquivo é auto-contido, novos lotes geram **novos arquivos `.parquet` com seu próprio HNSW interno**. Operações:

- **INSERT**: novo arquivo `.parquet` com seu HNSW. Aparece como nova entrada no manifesto Iceberg.
- **DELETE**: usa Position Delete Files do Iceberg, marcando linhas removidas. O HNSW interno fica com "buracos" — o leitor filtra resultados que apontam para linhas deletadas.
- **UPDATE**: implementado como `DELETE + INSERT` (copy-on-write), produzindo novo `.parquet` com HNSW reconstruído para as linhas atualizadas.
- **OPTIMIZE/COMPACT**: job assíncrono periódico que mescla N arquivos pequenos em um arquivo grande, reconstruindo o HNSW único do novo arquivo. Equivalente ao `OPTIMIZE` do Delta Lake.

**Trade-off explícito**: não há índice HNSW global compartilhado entre arquivos. A busca abre múltiplos índices (um por arquivo "quente") e mescla resultados. Isso é compensado pelo Particionamento Geométrico — em vez de buscar em 10.000 índices, busca em 50-100.

### 5B — Alinhamento de Linhas e Consistência ACID

**Invariante**: `parquet_row_groups[row_N].embedding == hnsw_graph.lookup_by_row_id(N)`

Essa garantia é mantida por:

1. **Escrita unificada**: o Parquet e o HNSW são escritos no mesmo arquivo, em uma única transação de I/O. Se a escrita falhar a qualquer momento, o arquivo é descartado antes do commit ao manifesto Iceberg.
2. **Verificação de integridade**: ao abrir, o leitor verifica que `parquet_record_count == hnsw_graph.node_count`.
3. **Deletes lógicos**: registros deletados via Position Delete Files também invalidam o resultado HNSW correspondente no leitor (filtro pós-busca).

### 5C — Tipo Lógico `VECTOR`

Extensão do sistema de tipos do Parquet:

```
VECTOR(dim=1536, distance=cosine)
```

- **Armazenamento físico**: coluna `FIXED_LEN_BYTE_ARRAY(dim*2)` no Parquet (F16 padrão).
- **Metadado**: `field_id`, `dim`, e `distance_metric` são encodados no `key_value_metadata` do campo Parquet.
- **Leitores nativos**: motores com plugin AI-Lake decodificam a coluna e ativam o `Vector-Scan`. Leitores legados a leem como bytes opacos.

---

## 6. Controle de Overhead de Armazenamento

O risco real é que o rodapé HNSW e os vetores aumentem o custo de storage de forma desproporcional. Esta seção define as estratégias para manter o overhead em limites aceitáveis.

### 6A — Quantificação do Overhead Base

| Modelo de Embedding | Dimensões | Precisão | Tamanho por vetor | 1M de registros |
|---|---|---|---|---|
| `text-embedding-3-small` | 1.536 | `float32` | 6 KB | **6 GB** |
| `text-embedding-3-small` | 1.536 | `float16` | 3 KB | **3 GB** |
| `text-embedding-3-small` | 1.536 | `int8` | 1.5 KB | **1.5 GB** |
| `text-embedding-3-large` | 3.072 | `float16` | 6 KB | **6 GB** |
| `nomic-embed-text` | 768 | `float16` | 1.5 KB | **1.5 GB** |

Para 100M de registros com `text-embedding-3-small` em `float16`, os vetores ocupam ~300 GB e o HNSW interno adiciona ~30-60 GB (10-20% extra).

### 6B — Estratégias de Compressão em Camadas

**Nível 1 — Quantização escalar**

```rust
pub enum VectorPrecision {
    F32,          // 4 bytes/dim — precisão máxima
    F16,          // 2 bytes/dim — perda <0.1% em recall@10 (PADRÃO)
    I8Symmetric,  // 1 byte/dim  — perda ~1-3%
    Binary,       // 1 bit/dim   — apenas para modelos binários
}
```

Configurado no `metadata.json`:
```json
"ailake.vector-precision": "f16"
```

**Nível 2 — Product Quantization (PQ)** para casos extremos

Para 1536 dimensões com `M=48` sub-vetores:
- Sem PQ (F32): 6.144 bytes/vetor → 600 GB para 100M
- Com PQ: 48 bytes/vetor → **4.7 GB para 100M** (redução de 99.2%, recall@10 ~93-95%)

**Nível 3 — Compressão Parquet de blocos**

Row groups Parquet aplicam compressão (Snappy/Zstd) sobre a coluna F16 — ganho marginal de ~5% (vetores são pouco compressíveis), mas sem custo adicional pois é padrão Parquet.

### 6C — Estimativa final com política padrão (F16)

Para 100M registros com `text-embedding-3-small` (dim=1536):
- Dados tabulares (Parquet, texto + metadados): ~50 GB
- Coluna VECTOR F16 dentro dos Parquets: ~300 GB
- Rodapé HNSW (10-20% dos vetores): ~30-60 GB
- **Total**: ~380-410 GB vs. ~50 GB sem vetores — overhead de ~7-8×

Com PQ habilitado, o rodapé HNSW reduz para ~5-10 GB, mas os vetores F16 brutos são mantidos na coluna Parquet para reranking preciso.

---

## 7. Preservação de Contexto para Uso com LLMs

Armazenar embeddings sem o contexto que os gerou é um antipadrão: o vetor aponta para o documento, mas o LLM precisa do **texto** — e precisa do texto certo, no formato certo, na quantidade certa.

### 7A — Schema de Contexto Enriquecido (`LlmContextSchema`)

Toda tabela AI-Lake que serve workloads de LLM deve seguir este schema mínimo:

```rust
pub struct LlmContextSchema {
    // Identidade
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub chunk_index: u32,
    pub total_chunks: u32,

    // Conteúdo principal
    pub chunk_text: String,

    // Contexto estrutural (crítico)
    pub document_title: String,
    pub section_path: String,
    pub preceding_context: String,
    pub following_context: String,

    // Contexto semântico
    pub document_summary: String,
    pub chunk_summary: String,

    // Metadados de recuperabilidade
    pub source_uri: String,
    pub page_number: Option<u32>,
    pub created_at: Timestamp,
    pub document_date: Option<Date>,

    // Vetores
    pub embedding: Vector<1536>,
    pub context_embedding: Vector<1536>,
}
```

### 7B — Estratégia de Embeddings Duplos

Cada chunk armazena **dois vetores**:
- **`embedding`**: gerado do `chunk_text` puro. Busca de conteúdo específico.
- **`context_embedding`**: gerado de uma string enriquecida com título + seção + resumo. Captura o contexto posicional/hierárquico do chunk.

Custos: 2× chamadas de embedding na ingestão, 2× storage de vetores. Ganho: recall significativamente superior em perguntas que dependem de contexto.

> Quando dois vetores são usados, **cada coluna gera seu próprio HNSW no rodapé do arquivo**. O leitor escolhe qual usar (ou ambos via RRF) conforme o tipo de query.

### 7C — `ContextAssembler`

O SDK expõe um `ContextAssembler` que monta o contexto para o LLM:

```rust
pub struct ContextAssemblerConfig {
    pub max_tokens: usize,
    pub dedup_threshold: f32,
    pub group_by_document: bool,
    pub max_chunks_per_document: usize,
    pub include_adjacent_context: bool,
}
```

Algoritmo: deduplica chunks similares, agrupa por documento (ordenando por `chunk_index`), aloca dentro do budget de tokens, renderiza em XML estruturado pronto para Claude/GPT-4.

---

## 8. Fluxo Fim-a-Fim da Consulta

```
1. Usuário envia query de texto
2. Query é convertida em vetor de embedding (1536 dim)
3. AI-Lake SDK lê o manifesto Iceberg (metadata.json + snap-*.avro)
4. Para cada arquivo no manifesto:
     a. Lê custom-properties: centroid, radius
     b. Calcula distance(query, centroid)
     c. Se distance - radius > threshold → PRUNE
5. Para arquivos sobreviventes (em paralelo, via Tokio):
     a. GET parcial do footer Parquet → extrai hnsw_offset/len
     b. GET parcial do rodapé HNSW → mmap local
     c. Deserializa HNSW via bincode
     d. Busca top-k local no grafo
6. Merge global dos resultados (top-k de todos os arquivos)
7. Para os k vencedores: lê linhas via Parquet reader (com predicate pushdown)
8. Retorna RecordBatch (todas as colunas + _distance)
```

---

## 9. Comparação de Conceito

| Recurso | Apache Iceberg | Pinecone / Milvus | **AI-Lake Format** |
|---|---|---|---|
| **Foco** | Dados tabulares | Somente vetores | Tabular + Embeddings + Índice |
| **Layout físico** | Múltiplos Parquets | DB proprietário | **Arquivo único auto-contido** |
| **Poda de arquivos** | Min/Max, Partição | N/A (tudo em memória) | **Particionamento Geométrico** |
| **Garantias ACID** | Sim | Não (eventual) | **Sim (via Iceberg)** |
| **SQL nativo** | Sim (Trino/Spark) | Não | **Sim (compatibilidade Iceberg)** |
| **Busca vetorial nativa** | Não | Sim | **Sim (HNSW no rodapé)** |
| **Escala** | Petabytes | Bilhões de vetores | **Petabytes** |
| **Time-travel** | Sim | Não | **Sim (via snapshots Iceberg)** |
| **Caso de uso primário** | BI / Analytics | RAG, Similaridade | **BI + RAG + LLMs** |

---

## 10. Roadmap de Implementação

### Fase 1 — Fundação (MVP local)
- [x] `ailake-core`: tipos base (`VectorColumn`, `VectorMetric`, `LlmContextSchema`, `RowId`)
- [x] `ailake-parquet`: leitor/escritor Parquet com tipo `FIXED_LEN_BYTE_ARRAY` + metadados custom
- [x] `ailake-vec`: quantização escalar F32 → F16 → I8
- [x] `ailake-index`: HNSW via `hnsw_rs`, serialização via `bincode`
- [x] `ailake-file`: writer/reader do arquivo unificado (Parquet + rodapé AI-Lake)
- [x] `ailake-catalog`: escrita de `metadata.json` compatível com Iceberg Spec v2 + custom-properties
- [x] CLI: `ailake create`, `ailake insert`, `ailake search`, `ailake compact`, `ailake info`
- [x] Validação: PyIceberg valida metadata Iceberg Spec v2; PyArrow e DuckDB lêem dados tabulares sem o SDK AI-Lake

### Fase 2 — Distribuição e Object Storage
- [x] `ailake-store`: integração `object_store` (S3, GCS, Azure Blob) — `ObjectStoreBackend`
- [x] `ailake-store`: builders tipados `S3Config`, `GcsConfig`, `AzureConfig` + `store_from_url()`
- [x] `ailake-index`: carga via `memmap2` — tempfile + Mmap, lazy paging do grafo HNSW
- [x] Job de compaction assíncrono (Tokio): `CompactionPlanner` + `CompactionExecutor`
- [x] `ailake-vec`: Product Quantization (PQ) — `PQCodebook` com k-means++ + ADC
- [x] `ailake-vec`: `BlockCompressor` com zstd/lz4
- [x] `ailake-query`: pruning via centróides — `VectorPruner` geométrico
- [x] `ailake-query`: `ContextAssembler` — dedup, agrupamento por doc, budget de tokens, XML
- [x] `ailake-py`: bindings PyO3 — `TableWriter`, `search()`, `assemble_context()`
- [x] Testes de compatibilidade: PyArrow, DuckDB, PyIceberg metadata JSON — validados localmente; Spark + Trino via `compat-heavy.yml` (workflow_dispatch)

### Fase 3 — Integração com Motores de Query
- [x] `ailake-jni`: C-ABI cdylib via JNA para Spark/Trino/Flink
- [x] Plugin Trino: `VectorScanConnector`
- [x] Plugin Spark: `VectorScanStrategy`
- [x] Suporte a múltiplas colunas vetoriais (`embedding` + `context_embedding`)

### Fase 4 — Produção
- [x] Benchmarks públicos vs. LanceDB, Deep Lake, pgvector (`ailake-bench`)
- [x] Avaliação de FFI para GPU (cuVS/NVIDIA) — runtime libloading CUDA + ROCm
- [x] Documentação de spec pública do formato (`docs/specs/FILE_FORMAT.md`)
- [x] Reranking automático após PQ
- [x] IVF-PQ native index + adaptive index selection (HNSW vs IVF-PQ por hardware)
- [x] GPU k-means para treino IVF-PQ; GPU batch search com fallback CPU
- [x] `MemTable` write buffer para ingestão streaming
- [x] AVX-512 + FMA + F16C SIMD para kernels de distância
- [x] Flink connector (`VectorScanSource` + `VectorScanTableFactory`)

### Fase 5 — Próximos Passos

- [ ] **DuckLake catalog backend** — `DuckLakeCatalog` em `ailake-catalog/` implementando `CatalogProvider` sobre catálogo DuckDB (dep: crate `duckdb`); mapeamento de metadados vetoriais (`centroid`, `radius`, `footer-offset`) para tabelas internas DuckLake; modelo de commit via INSERT no catálogo DuckDB. *Aguardar estabilização da spec DuckLake (anunciada mai/2025) antes de implementar.*
- [ ] **dbt integration guide** — documentar fluxo `dbt (transform) → AI-Lake SDK (ingest + HNSW)` para dbt-spark e dbt-trino com plugins AI-Lake carregados

---

## 11. Stack Técnica — Rust

### Crates do Projeto

```
ailake/
├── ailake-core/          # Tipos, traits, schema VECTOR, LlmContextSchema, RowId
├── ailake-parquet/       # Leitor/escritor Parquet com tipo VECTOR
├── ailake-vec/           # Quantização F32/F16/I8/PQ
├── ailake-index/         # HNSW via hnsw_rs, serialização bincode, mmap
├── ailake-file/          # Arquivo unificado: Parquet + rodapé AI-Lake
├── ailake-catalog/       # Catálogo Iceberg: metadata.json, manifestos Avro
├── ailake-store/         # Abstração de object storage via object_store
├── ailake-query/         # Pruning, scan, ContextAssembler
├── ailake-py/            # Bindings Python via PyO3
└── ailake-jni/           # C-ABI cdylib via JNA para Spark/Trino/Flink
```

### Dependências Principais

```toml
[workspace.dependencies]
# I/O de dados
parquet        = "52"                 # parquet-rs — Apache oficial
arrow          = "52"                 # arrow-rs — in-memory columnar
arrow-array    = "52"
object_store   = { version = "0.10", features = ["aws", "gcp", "azure"] }

# Indexação vetorial (Rust puro)
hnsw_rs        = "0.3"               # HNSW puro Rust, serialização nativa

# mmap
memmap2        = "0.9"

# Catálogo Iceberg
iceberg        = "0.3"
apache-avro    = "0.16"

# Serialização
serde          = { version = "1", features = ["derive"] }
serde_json     = "1"
bincode        = "1"                 # serialização ultra-rápida do HNSW

# Compressão
lz4_flex       = "0.11"
zstd           = "0.13"

# Async
tokio          = { version = "1", features = ["full"] }
futures        = "0.3"

# Bindings
pyo3           = { version = "0.22", features = ["extension-module", "abi3-py39"] }
# uniffi removed — JVM bindings use C-ABI + JNA (ailake-jni)

# Half-precision
half           = "2"                 # f16 type

# Observabilidade
tracing        = "0.1"
```

### Por que `hnsw_rs` em vez de Faiss/usearch?

| Critério | Faiss (C++) | usearch | **hnsw_rs (Rust puro)** |
|---|---|---|---|
| Dependência C++ | Sim | Híbrido | **Não** |
| Compilação cruzada | Complexa | Média | **Simples** |
| Segurança de memória | C++ unsafe | Misto | **Rust safe** |
| HNSW + cosine | Sim | Sim | **Sim** |
| Serialização nativa Serde | Não | Não | **Sim (bincode)** |
| Licença | MIT | Apache 2.0 | **MIT/Apache 2.0** |

`hnsw_rs` se integra naturalmente com o ecossistema Rust de serialização, permitindo armazenar o grafo HNSW como bytes via `bincode` sem necessidade de adaptadores FFI.

### Por que SDK direto em vez de DataFusion?

A decisão consciente é **não usar DataFusion como motor de query**. Razões:

- **Foco no formato**: o projeto é um formato de arquivo, não um SQL engine. Adicionar DataFusion expandiria o escopo significativamente.
- **Bindings limpos**: PyO3 (Python) e JNA C-ABI (JVM) expõem operações específicas (`search`, `write_batch`) sem o overhead de um planner SQL completo.
- **Performance previsível**: SDK direto evita custos de planning para queries simples (busca vetorial pura).
- **Integração com Spark/Trino**: esses motores fazem o planning SQL deles; o AI-Lake só precisa fornecer scan eficiente.

Usuários que precisarem de SQL podem usar Spark/Trino/DuckDB sobre AI-Lake via compatibilidade Iceberg, ou construir seu próprio `TableProvider` DataFusion como camada externa.

### Bindings Python (PyO3)

```python
import ailake

# Escrita
writer = ailake.TableWriter("s3://my-lake/docs/")
writer.write_batch(df_arrow, embeddings=np.array(..., dtype=np.float32))
writer.commit()

# Busca vetorial
results = ailake.search(
    table="s3://my-lake/docs/",
    query=my_embedding,
    top_k=100,
    filter="category = 'finance'",
)
# results é um PyArrow RecordBatch — zero-copy para pandas/polars
```

---

## 12. Compatibilidade com o Ecossistema Iceberg

### Como qualquer framework lê uma tabela AI-Lake

```
1. Ler metadata/v{N}.metadata.json    → properties contém hints AI-Lake (ignorados)
2. Ler snap-XXX.avro                  → manifesto Iceberg padrão, aponta para .parquet
3. Ler part-XXXXX.parquet             → leitor Parquet padrão para no PAR1 final
4. Retornar dados ao usuário          → vetores como bytes, demais colunas normais
```

O rodapé AI-Lake após o PAR1 final é **invisível** para leitores Parquet padrão — eles param de ler no fim do footer Parquet conforme a especificação.

### Frameworks validados para leitura de tabelas AI-Lake (modo padrão)

| Framework | Status |
|---|---|
| **PyIceberg** | Dados tabulares ✓, vetores como bytes |
| **Apache Spark** (iceberg-spark) | Dados tabulares ✓, vetores como bytes |
| **Trino** (iceberg connector) | Dados tabulares ✓, vetores como bytes |
| **DuckDB** (iceberg extension) | Dados tabulares ✓, vetores como bytes |
| **Snowflake** (Iceberg tables) | Dados tabulares ✓ |
| **AWS Athena** | Dados tabulares ✓ |

> Para habilitar Vector-Scan, instalar o plugin AI-Lake (Phase 3). Sem ele, a tabela funciona como Iceberg padrão.

---

## 13. Referências Técnicas

- [Apache Iceberg Spec v2](https://iceberg.apache.org/spec/) — base para manifestos, snapshots e custom-properties.
- [iceberg-rust](https://github.com/apache/iceberg-rust) — crate oficial ASF.
- [parquet-rs](https://github.com/apache/arrow-rs/tree/master/parquet) — Parquet em Rust.
- [arrow-rs](https://github.com/apache/arrow-rs) — columnar in-memory em Rust.
- [object_store](https://github.com/apache/arrow-rs/tree/master/object_store) — abstração de storage.
- [hnsw_rs](https://crates.io/crates/hnsw_rs) — HNSW puro Rust.
- [memmap2](https://crates.io/crates/memmap2) — mmap em Rust.
- [bincode](https://crates.io/crates/bincode) — serialização binária Serde.
- [HNSW Paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2018.
- [PyO3](https://pyo3.rs/) — bindings Python.
- [uniffi](https://mozilla.github.io/uniffi-rs/) — bindings JVM/Kotlin/Swift.
