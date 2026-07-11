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

**Compatibilidade de leitura ≠ compatibilidade de escrita** (ADR-018, `docs/contributing/DECISIONS.md`): a garantia de leitor Iceberg padrão (§12) não implica que engines genéricos possam escrever com segurança sobre o índice. Se `OPTIMIZE`/`rewrite_data_files` do Spark/Trino (ou qualquer maintenance job Iceberg-padrão) reescrever um arquivo AI-Lake, o resultado é Parquet válido, mas **sem rodapé AILK e sem `centroid`/`radius` no manifesto** — esse par vive em `key_metadata` (campo reservado do Iceberg pra chave de criptografia), que escritores genéricos nunca populam. Isso não corrompe a tabela nem retorna resultado errado, mas degrada:

- **Busca**: cai pra flat scan O(N) exato nesse arquivo (`scanner.rs`), com `warn!` visível (não mais silencioso) e agregado ao fim de cada busca.
- **Compaction**: `CompactionPlanner::plan()` detecta arquivo sem `centroid_b64` (= nunca escrito pelo SDK AI-Lake) e prioriza reindex, ignorando os limiares de batching (`min_files_to_compact`/tamanho) — um único arquivo "foreign" já dispara reparo.
- **`ailake info`** reporta arquivos "foreign" (sem índice) pra visibilidade proativa, sem depender de uma busca lenta pra descobrir o drift.

A divisão de trabalho documentada continua valendo: SDK AI-Lake (ou plugins Spark/Trino/Flink) escreve; engines genéricos só leem com segurança. Escrita cruzada é suportada (não quebra), mas degrada até o próximo `ailake compact`.

### 5B — Alinhamento de Linhas e Consistência ACID

**Invariante**: `parquet_row_groups[row_N].embedding == hnsw_graph.lookup_by_row_id(N)`

Essa garantia é mantida por:

1. **Escrita unificada**: o Parquet e o HNSW são escritos no mesmo arquivo, em uma única transação de I/O. Se a escrita falhar a qualquer momento, o arquivo é descartado antes do commit ao manifesto Iceberg.
2. **Verificação de integridade**: `AilakeFileReader::verify_integrity()` compara `parquet_record_count == hnsw_graph.node_count == header.record_count`. Roda após todo merge de compaction, antes do commit ao catálogo (`compact()`/`compact_incremental()` em `ailake-query/src/compaction.rs`) — falha o build em vez de deixar um arquivo inconsistente chegar ao manifesto.
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
- [x] `ailake-py`: bindings PyO3 — `TableWriter`, `search()`, `assemble_context()`, `search_with_data()` (full-read via Arrow IPC; `fetch_data=True` em `SearchQuery`)
- [x] Testes de compatibilidade: PyArrow, DuckDB, PyIceberg metadata JSON — validados localmente; Spark + Trino via `compat-heavy.yml` (workflow_dispatch)

### Fase 3 — Integração com Motores de Query
- [x] `ailake-jni`: C-ABI cdylib via JNA para Spark/Trino/Flink
- [x] Plugin Trino: `VectorScanConnector`
- [x] Plugin Spark: `VectorScanStrategy`
- [x] Suporte a múltiplas colunas vetoriais (`embedding` + `context_embedding`)
- [x] `duckdb-ailake`: extensão C++ para DuckDB — `ailake_search()` + `ailake_write_batch()` via `dlopen`/C-ABI; mesmo protocolo JSON-envelope do Spark/Trino; degradação graciosa sem lib

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
- [x] **IVF-PQ shared codebook** — `IvfPqCodebook` treinado uma vez no primeiro shard e reutilizado em todos os shards subsequentes via `Arc<tokio::sync::OnceCell>`; distâncias ADC comparáveis cross-shard sem reranking por codebook incompatível
- [x] **`write_batch_ivf_pq_deferred`** — variante async de IVF-PQ: persiste Parquet imediatamente (~200k vec/s), treina índice IVF-PQ em background (mesmo padrão do HNSW deferred); `IndexStatus::Indexing → Ready`
- [x] **Fix k-means++ O(n×k²) → O(n×k)** — `kmeans_pp_init` usa min-dist incremental; parallelismo via `rayon::par_iter` no assignment loop e no init; speedup 17× em IVF-PQ SIFT-1M
- [x] **Fix `HadoopCatalog::commit_snapshot`** — operações `Replace`/`Overwrite` não herdam manifests anteriores; corrige bug onde `IndexStatus::Ready` nunca convergia com múltiplos background tasks concorrentes
- [x] **`hnsw_m` + `hnsw_ef_construction` per tabela** — `VectorStoragePolicy::hnsw_m` e `hnsw_ef_construction` permitem tunar M e ef por tabela sem mudar o código; armazenados como `ailake.hnsw-m` / `ailake.hnsw-ef-construction` em propriedades Iceberg; sobrepõem os defaults do `HnswConfig` no write. Exposto via CLI (`--hnsw-m`, `--hnsw-ef`) e Python (`hnsw_m=`, `hnsw_ef_construction=`). `None` = usa defaults (backwards-compatible).
- [x] **`VectorMetric::NormalizedCosine` + `pre_normalize`** — `VectorStoragePolicy::pre_normalize = true` normaliza vetores para L2 unitário na escrita e usa `1-dot(a,b)` no hot loop do HNSW em vez de cosine completo (sem sqrt). ~12-20% speedup em search para dim=1536. Query normalizada automaticamente em todos os bindings (Rust, Python, Go, C++). Exposto via `ailake create --pre-normalize` (CLI) e `TableWriter(pre_normalize=True)` (Python).

### Fase 5 — Próximos Passos

- [x] **Busca híbrida BM25+vetor** — `SearchConfig::hybrid: Option<HybridConfig>` adiciona scoring lexical BM25 ao pipeline de busca vetorial, eliminando dependência de infra FTS externa (Tantivy, Elasticsearch) para workloads RAG/híbridos. `BM25Scorer` puro Rust (sem deps C++); `IdfStats` acumulado em write time via `TableWriter::with_bm25("chunk_text")`, persistido em `metadata/ailake_bm25_stats.bin` (bincode+zstd, 50k termos máx). Pipeline: HNSW → pool de candidatos (`10×top_k`) → BM25 score com IDF global → fusão RRF ou linear. `search_text()` = scan bruto BM25 sem HNSW (O(N), documentado). Python: `TableWriter(bm25_text_column="chunk_text")`, `search(..., hybrid_text="query")`, `search_text(path, "query")`. Limitação intencional: sem inverted index; busca lexical pura em escala usa DuckDB/Trino `LIKE` via compat Iceberg.

- [x] **Phase T — Tantivy per-file FTS** — índice invertido Tantivy embedded por arquivo no rodapé AILK (`AILK_FTS` section, magic `AFTS`, zstd). `search_text()` usa Tantivy O(log N) quando disponível; fallback BM25 O(N) para arquivos legados (backward-compat total). Storage: ~3-4 MB/arquivo (50k docs) vs ~15 MB HNSW. Implementado em todo o ecossistema: `ailake-fts` (crate), `ailake-file` (`FLAG_HAS_FTS`, `AilakeHeader.fts_offset`), `ailake-query` (fast path + compaction), CLI (`--fts-columns`, `--text`), Python (`fts_text_columns=`), JNI C-ABI (`fts_columns[]` em `ailake_write_batch_json`; `text_columns[]` em `ailake_search_text_json`), Spark, Trino, Flink (opções DDL `fts.columns`/`fts.tokenizer`), Go (`SearchText()`), C++ (`ailake::search_text()` via CLI), Airflow (`AilakeWriteOperator(fts_columns)` + `AilakeFtsSearchOperator`), Airbyte (`fts_columns` em `spec.json`). Branch: `feature/phase-t-tantivy-fts` (2026-06-20).

- [x] **`write_batch_auto_deferred`** — variante deferred do engine Auto: detecta hardware em runtime e delega para `write_batch_deferred` (HNSW, CPU) ou `write_batch_ivf_pq_deferred` (IVF-PQ, GPU). Eleva throughput do Auto de 6.3k vec/s (inline bloqueante) para ~200k vec/s (Parquet-only imediato, índice async). Exposto via Python `TableWriter.write_batch_auto_deferred()` e CLI.
- [x] **DuckLake catalog backend** — `DuckLakeCatalog` em `ailake-catalog/` (feature `catalog-ducklake`) implementando `CatalogProvider` sobre um catálogo DuckLake real, pilotado via SQL sancionado da extensão `ducklake` real do DuckDB (`ATTACH 'ducklake:...'`, `ducklake_add_data_files`, `ducklake_list_files`) — nunca escreve direto nas tabelas internas do DuckLake (bootstrap/invariantes de versionamento não são 100% públicos). Metadados vetoriais próprios do AI-Lake (`centroid`, `radius`, `hnsw_offset/len`, `index_status`, etc.) vivem numa tabela sidecar (`main.ailake_vector_index`) fora do attachment DuckLake. Suporte completo a `Append`/`Overwrite`/`Replace`/`Delete` (cobre compaction/backfill/memory-decay/migration/patch de índice deferred). Wired no `ailake-cli` via `--catalog hadoop|ducklake` (feature opt-in `catalog-ducklake`, não default — puxa build C++ bundled do DuckDB). Quatro bugs reais achados verificando contra extensão `ducklake` viva e o binário `ailake` real (não só docs/unit tests), documentados em `docs/guides/DUCKLAKE_CATALOG.md`: (1) DuckDB não permite escrever em duas databases attached na mesma transação — commit em duas fases sequenciais; (2) `DELETE FROM lake.tbl WHERE filename=?` esvazia linhas mas não remove o arquivo de `ducklake_list_files()` — autoridade de "ativo" migrada pra flag `active` da sidecar; (3) `DataFileEntry.path` relativo ao warehouse nunca era resolvido pra path absoluto antes de `ducklake_add_data_files`; (4) faltava `allow_missing => true` — arquivo escrito antes de um `evolve_schema` era rejeitado. Achado de quebra (bug 3/4) também expôs bug pré-existente e independente de catálogo: `ParquetVectorReader::read_all()` sempre decodificava como F16 ignorando a precisão real do arquivo (`ailake.precision` no KV metadata nunca lido) — corrompia leitura de qualquer tabela `--precision f32` no compaction/scanner, corrigido. `JdbcCatalog`/`GlueCatalog`/`NessieCatalog` seguem sem wiring (não é lacuna do DuckLake, ninguém tem). Testado com round-trip real (sem mocks) contra catálogo DuckDB/DuckLake ao vivo e binário CLI real.
- [ ] **DuckLake — traduzir equality deletes em `DELETE` nativo** — hoje `delete_where`/`delete_rows` vivem no sidecar e são aplicados só por leitores AI-Lake; um `SELECT` DuckLake-nativo ainda vê as linhas deletadas (limitação documentada em `docs/guides/DUCKLAKE_CATALOG.md`). Quando a coluna do predicado estiver declarada ao DuckLake, `DuckLakeCatalog::commit_snapshot` (op `Delete`) pode adicionalmente emitir `DELETE FROM lake.tbl WHERE col IN (...)` — leitores nativos passam a concordar com os AI-Lake, mesma simetria que a retirada de arquivos já tem.
- [ ] **Idempotência `batch_id` sobrevivendo a compaction** — o tag vive na `DataFileEntry` e o merged da compaction carrega `batch_id: None`; retry que dispara após o lote ter sido compactado re-insere silenciosamente (janela documentada em `write_batch_idempotent`). Fix: merged agrega os batch_ids dos fontes (campo é `Option<String>` — JSON array) e `write_batch_idempotent` passa a checar pertencimento.
- [ ] **Iceberg V3 — `variant` type** — tipo nativo V3 para colunas semi-estruturadas (JSON arbitrário sem schema fixo). Armazenamento físico: `BYTE_ARRAY` Parquet com metadata `iceberg.variant`. Leitores V3 (Spark 4, Trino 450+) decodificam nativamente; AI-Lake lê como bytes opacos sem erro. Implementar `VariantColumn` em `ailake-parquet` quando o uso for relevante — raramente necessário em workloads de embeddings.
- [ ] **Iceberg V3 — Equality Delete com field-id V3-nativo** — Phase H (implementada) usa encoding V2-style para equality delete Avro. V3 nativo requer `equality_delete_files` com `delete-file-format: 2` e `field-id` em campo `equality_ids` do manifesto entry com type annotation V3. Impacto: Spark 4+ e PyIceberg 0.8+ esperam o encoding V3 em tabelas com `format-version: 3`. Atualizar `write_equality_delete_avro` e `write_equality_delete_manifest` em `ailake-query/src/delete.rs` para emitir encoding correto quando `format_version=3`.
- [ ] **Iceberg V3 — Column Statistics estendidas** — V3 expande `ColumnMetrics` nos manifestos com `lower_bounds`/`upper_bounds` obrigatórios para predicado pushdown, mais `nan_value_counts` para tipos float. Atualmente emitidos como `null` no `write_manifest_file`. Implementar coleta de estatísticas via `parquet::file::statistics` durante escrita em `ailake-catalog/src/avro_manifest.rs` e encodar como Avro `map<int, bytes>` (Iceberg single-value serialization) — habilita pruning de row groups em Spark/Trino sem abrir dados.
- [x] **dbt integration guide** — `docs/guides/DBT_INTEGRATION.md`: fluxo `stg_documents → int_chunks → ailake_embeddings`; macro `ailake_write_batch` (Spark/Trino/DuckDB); post-hook incremental; compaction operation; 3 padrões de geração de embedding; recall assertion test; configuração de cluster Spark e Trino.

### Fase 6 — Qualidade de Recall e Developer Experience

- [x] **`VectorMetric::NormalizedCosine` + `pre_normalize`** — `VectorStoragePolicy::pre_normalize = true` normaliza vetores para L2 unitário na escrita e usa `1-dot(a,b)` no hot loop do HNSW em vez de cosine completo (sem sqrt). ~12-20% speedup em search para dim=1536. Query normalizada automaticamente em todos os bindings (Rust, Python, Go, C++). Exposto via `ailake create --pre-normalize` (CLI) e `TableWriter(pre_normalize=True)` (Python).
- [x] **`hnsw_m` + `hnsw_ef_construction` por tabela** — `VectorStoragePolicy::hnsw_m` e `hnsw_ef_construction` permitem tunar M e ef por tabela sem mudar o código; armazenados como `ailake.hnsw-m` / `ailake.hnsw-ef-construction` em propriedades Iceberg. Exposto via CLI (`--hnsw-m`, `--hnsw-ef`) e Python (`hnsw_m=`, `hnsw_ef_construction=`).
- [x] **MRL / dimension truncation documentado** — modelos Matryoshka (OpenAI text-embedding-3-*, Cohere embed-v3, Jina v3, Nomic v1.5) permitem truncar dimensão sem retreinamento; AI-Lake suporta nativamente via `dim` menor na criação da tabela. Documentado em `ailake-py/README.md` e `SETUP.md §8H` com tabela recall×storage (1536→512 = 3× menos storage, ~97% recall@10).

### Fase 7 — Compressão Extrema e Novos Tipos de Índice

> **Nota (v0.0.14)**: RaBitQ e Binary Hamming foram removidos do codebase. Recall ≈ 0 em embeddings float gerais sem alinhamento de treinamento; complexidade não justificada vs. HNSW/IVF-PQ.

### Fase 8 — Multimodal (Imagens, Áudio, Vídeo)

> **Pré-condição**: vetores de imagem/áudio já funcionam hoje via coluna `VECTOR` padrão (CLIP dim=512, ImageBind dim=1024). Esta fase adiciona suporte semântico de primeira classe para dados multimodais.
>
> **Nota (v0.0.19)**: Coluna `MEDIA` (bytes brutos embutidos) descartada — AI-Lake não é blob store. Mídia vive em object storage; apenas URIs e embeddings pertencem ao AI-Lake. Todos os demais itens implementados. **Fase 8 concluída.**

- [x] **`ailake.modality` property** — `VectorModality` enum (`Text`, `Image`, `Audio`, `Video`) em `ailake-core`. `VectorStoragePolicy.modality: Option<VectorModality>` (serde default, backward-compat). Iceberg property: `ailake.modality-<col>`. CLI `ailake create --modality text|image|audio|video`. Permite seleção do HNSW correto por modalidade sem inspecionar dados.

- [x] **Vetores N generalizados** — N colunas `VECTOR` com HNSW próprio no rodapé via `AilakeFileWriter::write_multi` (existente). Python `VectorColSpec(column, dim, metric, modality)` expõe o multi-column write. Cada coluna tem AILK section independente; localização via `ailake.<col>.footer_offset` no KV metadata Parquet.
- [x] **Cross-modal fusion search** — `search_multimodal()` em `ailake-query`: aceita `&[ModalQuery { column, query, weight }]`, roda HNSW por coluna independentemente, funde via Reciprocal Rank Fusion (`score = Σ weight_i / (60 + rank_i)`). Python: `ailake.search_multimodal(path, [(col, query, weight)], top_k)`. Enum `FusionMethod::Rrf` extensível.
- [x] **`MultimodalContextSchema`** — estende `LlmContextSchema` com `media_uri: String`, `media_mime: String`, `media_caption: String`, `image_embedding: Vector<512>`, `audio_transcript: String`. Base64-encode de miniaturas inline para contexto LLM multimodal.
- [x] **Bindings multimodal** — Python `VectorColSpec(column, dim, metric, modality)` + `ailake.search_multimodal(path, [(col, query, weight)], top_k)`. `TableWriter` e `_ailake` module atualizados.

### Fase 9 — Agentes e Memória Episódica

> **Contexto**: AI-Lake já funciona para RAG de agentes (busca semântica de longo prazo via HNSW + `LlmContextSchema`). Esta fase adiciona primitivas específicas para padrões de memória de agentes: episódica, procedural e de trabalho.

- [x] **`ToolCallSchema`** — estende `LlmContextSchema` com campos de agente: `agent_id: Uuid`, `session_id: Uuid`, `step_index: u32`, `tool_name: String`, `tool_input_json: String`, `tool_output_json: String`, `outcome: Enum(Success, Failure, Timeout)`, `latency_ms: u32`. Permite busca vetorial sobre histórico de tool calls ("quando a ferramenta X falhou em contextos similares?").
- [x] **`EpisodicMemorySchema`** — estende `LlmContextSchema` com `recency_weight: f32` (decai com o tempo via `exp(-λ * days_since_access)`), `access_count: u32`, `last_accessed_at: Timestamp`, `importance_score: f32` (definido pelo agente). Scoring híbrido no merge: `final_score = distance * recency_weight * importance_score`.
- [x] **Scoring híbrido no merge de resultados** — `SearchConfig` ganha `score_fn: Option<ScoreFn>` onde `ScoreFn = fn(distance: f32, row: &RecordBatch) -> f32`. Permite o agente injetar recência, importância ou qualquer sinal contextual no ranking final sem re-escrever o índice.
- [x] **Partição por `agent_id`** — `VectorStoragePolicy::partition_by: Option<String>` usa Iceberg hidden partitioning por coluna (`agent_id`, `session_id`). Pruning geométrico aplicado dentro da partição — busca isolada por agente sem filtro pós-scan.
- [x] **`WorkingMemoryBuffer`** — buffer em memória com capacidade limitada (N chunks mais recentes) que drena para AI-Lake via `drain_to_table(&mut TableWriter)`. Interface: `push(text, embedding, importance)`, `search(query, top_k)` (flat cosine scan), `drain_to_table()`, `is_full()`. Em `ailake-query/src/mem_table.rs`. Python: `ailake.WorkingMemoryBuffer(max_rows=1000)`.
- [x] **`MemoryDecayJob`** — job assíncrono que recomputa `recency_weight = exp(-λ × days_since_access)` a partir da coluna `last_accessed_at` (ISO string), reescreve cada arquivo de dados, e commita novo snapshot via `SnapshotOperation::Overwrite`. Em `ailake-query/src/memory_decay.rs`. Python: `ailake.decay_memories(path, decay_lambda=0.1)` → retorna número de arquivos atualizados.
- [x] **Python `Agent` helper** — `ailake.Agent(table_path, embed_fn, agent_id)` com métodos: `remember(text, importance=1.0)`, `recall(query, top_k)` (scoring híbrido automático), `log_tool_call(name, input, output)`, `assemble_context(query, max_tokens)`. Abstração de alto nível sobre `TableWriter` + `search` + `ContextAssembler` para uso em frameworks de agentes (LangChain, CrewAI, AutoGen).

### Fase 10 — Redução de overhead FFI (Arrow IPC no write_batch JNI)

> **ADR**: ADR-017 (docs/contributing/DECISIONS.md) — Arrow Flight rejeitado; melhoria incremental via Arrow IPC bytes no boundary JNI.

- [ ] **Arrow IPC bytes em `ailake_write_batch_json`** — substituir o payload JSON de embeddings (`"embeddings": [[1.0, 2.0, ...]]`) por Arrow IPC serializado passado como `byte[]` via JNI. Elimina a única overhead real da fronteira JNI (~10ms/1k vecs, 12MB JSON → 3MB IPC binário). Protocolo: `ailake_write_batch_ipc(ipc_bytes: *const u8, ipc_len: usize, opts_json: *const c_char) -> *mut c_char` — separação entre dados colunares (IPC) e configuração (JSON pequeno). Expor via JNA em Spark/Trino/Flink; manter `ailake_write_batch_json` como fallback backward-compatible. Implementação: `arrow_ipc::reader::FileReader` no lado Rust; `org.apache.arrow.vector.ipc.ArrowFileReader` no lado JVM (já no classpath Spark/Trino). Esforço estimado: ~1 semana.

### Fase 11 — SQL direto sem JOIN (wire `ailake_scan_json` em Spark/Trino/Flink)

> **Contexto (2026-07-08)**: auditoria fechou o gap de `searchMultimodal()` — Fase 8 estava com `AilakeNative.searchMultimodal`/`AilakeNativeLoader.searchMultimodal` totalmente implementados nos três plugins mas sem superfície SQL/DataFrame em nenhum deles (só chamável via API Scala/Kotlin direta). Fechado: Spark ganhou `implicits.AilakeSession.ailakeSearchMultimodal`, Trino ganhou a tabela `ailake.default.search_multimodal` (+ session property `multimodal_queries`), Flink ganhou o job param `ailake.multimodal.queries`. Auditando o restante das exportações C-ABI (`ailake-jni/src/lib.rs`) contra a superfície SQL de cada plugin, sobra exatamente uma capability nativa sem nenhum wrapper JVM: `ailake_scan_json`.

- [x] **`ailake_scan_json` — search + fetch de linhas completas, wired nos três plugins JVM.** Antes só era consumida pela extensão DuckDB (`duckdb-ailake`) — devolve `{"ok":true,"schema":[...],"num_rows":N,"columns":{"id":[...],"text":[...],"_distance":[...]}}`, ou seja, as colunas reais da tabela (não só `row_id`/`distance`/`file_path`). Sem ela, toda busca via SQL nos três plugins retornava apenas o resultado geométrico e exigia um `JOIN` manual contra uma tabela Iceberg registrada separadamente para recuperar `chunk_text`/`document_title`/etc. (ver `docs/guides/JVM_INTEGRATION.md` §4D). Descoberta que simplificou o plano original: `ailake_scan_json` não filtra colunas — sempre devolve a largura completa da linha — então nenhum plugin precisou de uma property nova tipo `fetch-columns`:
  - **Trino**: nova tabela `ailake.default.search_full` — schema é `ingestColumns()` (já configurado via `ailake.text-columns`, reaproveitado) `+ _distance DOUBLE`. Vetor volta como `VARCHAR` (JSON-encoded, ex. `"[0.1,-0.2]"`) em vez de `ARRAY<DOUBLE>` — deliberado, evita `Block`/`BlockBuilder` manual do `RecordCursor` sem build real do Trino SPI pra validar contra; revisitar quando o `compat-heavy` CI puder confirmar um path `ARRAY<DOUBLE>`. `ScanTableHandle` reaproveita o `VectorScanSplit` existente (mesmos campos bastam); dispatch pro `AilakeNative.scan` é decidido pelo *tipo do table handle* em `VectorScanRecordSetProvider`, não por um split novo.
  - **Spark**: `spark.ailakeSearchWithData(tableUri, queryVector, topK, vectorColumn=...)` — `DataFrame` com schema 100% dinâmico construído a partir do `schema` array da resposta (`int64`→`LongType`, `float32`→`FloatType`, `float64`→`DoubleType`, `bool`→`BooleanType`, `utf8`→`StringType`, `list_float32`→`ArrayType(FloatType)`) — sem precisar de um parâmetro `columns` porque a resposta já vem com a largura completa.
  - **Flink**: nova opção de DDL `search.mode = 'full'` no `CREATE TABLE` já search-shaped — troca o source fixo de 3 colunas (`AilakeVectorTableSource`) por `AilakeScanTableSource`, cujas colunas são as que o usuário declarar na DDL (schema-on-read real, sem limitação de catalog fixo como o Trino) — único requisito: último campo declarado tem que ser `_distance` (FLOAT ou DOUBLE), validado em `AilakeVectorConnectorFactory.validateScanResultSchema`. Suporta `ARRAY<FLOAT>` de verdade via `GenericArrayData` (sem o risco de Block-builder do Trino).
  - Documentação: `docs/guides/JVM_INTEGRATION.md` (§3D/3F/5C/6B) e `docs/specs/JVM_PLUGINS.md` atualizados com os novos entry points.
  - Sem trabalho novo em Rust — wiring puro do lado JVM, mesmo formato do fix de `searchMultimodal`.

### Fase 12 — Paridade completa Spark/Trino/Flink

> **Contexto (2026-07-08)**: auditoria de paridade pós-Fase 11 comparando os 11 capabilities nativos (`ailake-jni`) contra a superfície SQL/DataFrame de cada plugin achou Trino e Flink praticamente empatados (10/11), mas Spark com 5 capabilities implementadas em `AilakeNative.scala` sem nenhum caller DataFrame/SQL — "dead code" real, não decisão documentada (diferente de ALTER TABLE, que antes lançava `UnsupportedOperationException` explícito). Achado simétrico inverso: write multi-column (multimodal) só existia em Spark, zero wrapper em Trino/Flink.

- [x] **Spark — hybrid BM25+vector search** — `ailakeSearch` ganhou `hybridText`/`textColumn`/`bm25Weight` (já existiam em `AilakeNative.search`, nunca repassados pelo único caller DataFrame).
- [x] **Spark — full-text search puro** — `implicits.AilakeSession.ailakeSearchText(tableUri, queryText, ...)`, `DataFrame(row_id, distance, file_path)`.
- [x] **Spark — DELETE** — `AilakeTable` ganhou `SupportsDelete` (equality/IN pushdown, mesma semântica de Trino/Flink — `AilakeNative.deleteWhere` só suporta equality delete file, sem scan-and-delete linha a linha).
- [x] **Spark — ALTER TABLE ADD/RENAME COLUMN** — `AilakeCatalog.alterTable` parava de lançar `UnsupportedOperationException` incondicional; agora chama `AilakeNative.evolveSchema` via `TableChange.AddColumn`/`RenameColumn`, mesma limitação documentada de Trino/Flink (schema em memória é resolvido por chamada a partir das options do catalog, não hardcoded — mas nada aqui rastreia coluna nova adicionada até que o `DataFrame` do próximo `INSERT`/`SELECT` já a inclua).
- [x] **Spark — compact** — `implicits.AilakeSession.ailakeCompact(tableUri, ...)`. Sem `CALL` nativo no Spark SQL fora de uma API de stored procedure completa de catalog, então é um método simples em `SparkSession`, mesmo padrão de `ailakeWrite`.
- [x] **Trino — write multi-column (multimodal)** — nova catalog property `ailake.vector-columns` (JSON `[{"column","dim","metric"?,"precision"?,"modality"?}]`); quando configurada, `ingestColumns()` emite um `ARRAY<DOUBLE>` por entrada em vez do único `vectorColumn`, e `AilakePageSink` chama `AilakeNative.writeBatchMulti` (novo wrapper, `ailake_write_batch_multi_json`) em vez de `writeBatch`. Reaproveita `ailake.text-columns` para as colunas extra, sem propriedade nova ali.
- [x] **Flink — write multi-column (multimodal)** — nova opção de DDL `vector.columns` (mesmo JSON do Trino) no `CREATE TABLE` já ingest-shaped; `AilakeVectorTableSink`/`AilakeSinkFunction` resolvem N colunas `ARRAY<FLOAT>` por nome em vez do único `vecCol`, chamando `AilakeNativeLoader.writeBatchMulti` (novo wrapper) via `ailake_write_batch_multi_json`.
- Testes adicionados nos três plugins; docs (`JVM_INTEGRATION.md`, `JVM_PLUGINS.md`) e `CHANGELOG.md` atualizados.

### Fase 13 — Paridade CLI/DuckDB/Airflow

> **Contexto (2026-07-09)**: mesma auditoria de paridade (padrão Fase 11/12), agora sobre `ailake-cli`, `duckdb-ailake` e `airflow-providers-ailake`. Achado mais sério que os anteriores: não só gaps de cobertura, mas um bug ativo — `ailake insert` não tinha 8 flags que `ailake_write_batch_json` (usado por todo plugin JVM e `ailake-py`) já aceita por escrita (`--partition-by/-value/-fields`, `--format-version`, `--hnsw-m/-ef`, `--pre-normalize`, `--deferred`), então qualquer DAG Airflow usando esses parâmetros no `AilakeWriteOperator` crashava em runtime com erro do clap. `airflow-providers-ailake` tinha sido escrito contra a superfície de capacidade *pretendida*, não a do CLI real (incompleta).

- [x] **`ailake-cli insert` — 8 flags novas** — `--partition-by`, `--partition-value`, `--partition-fields`, `--format-version`, `--hnsw-m`, `--hnsw-ef`, `--pre-normalize`, `--deferred`, espelhando `ailake_write_batch_json`'s `Req`. `--deferred` é mutuamente exclusivo com `--batch-id` (writes deferred ainda não carregam idempotency tag).
- [x] **`ailake decay-memories <table> --lambda <λ>`** — subcommand novo, faltava inteiramente (só existia via `ailake-py`); `AilakeHook.decay_memories()` já chamava esse subcommand assumindo que existia.
- [x] **`duckdb-ailake` — `ailake_write_batch_multi` + `ailake_compact`** — 2 dos 9 capabilities nativos nunca tinham sido wireados numa função SQL. Seguem o padrão arity-ladder já estabelecido (`ailake_write_batch`/`ailake_delete_where`).
- [x] **`airflow-providers-ailake` — 6 capabilities do CLI nunca wrapeadas** — `migrate`/`delete_rows`/`add_vector_column`/`backfill_vector_column`/`estimate` (hook + operator, exceto `estimate` que é hook-only); `AilakeWriteOperator` ganhou `vector_cols` (multimodal); `AilakeCompactOperator` ganhou `max_files_per_pass`.
- [x] **Fix `AilakeHook.compact()`** — nunca pedia `--format json`, então seu parser de texto (`"files_compacted:"`) nunca dava match no output real (`"compacted into <path>"`) — sempre retornava `0` silenciosamente. Corrigido para usar `--format json`.
- Fora de escopo (documentado, não implementado): `search_multimodal`/`ailake_scan_json` não têm superfície nenhuma no `ailake-cli` — precisa de design real de formato de saída, não é wire-up mecânico.
- Testes reais (não só mocks): build real do CLI + round-trip manual; extensão DuckDB compilada e carregada numa sessão DuckDB real (`RTLD_GLOBAL` antes de `import duckdb`); Airflow com venv real (`apache-airflow` + `pytest`), 82 testes (60 existentes + 22 novos), e chamadas diretas contra o binário `ailake` real para todo hook novo. Docs (`README.md` dos três, `CHANGELOG.md`) atualizados.

### Fase 14 — Paridade ailake-go/ailake-cpp

> **Contexto (2026-07-09)**: mesma auditoria de paridade (padrão Fase 12/13), agora sobre os dois SDKs nativos. Ambos misturam reimplementação pura (reads: `Search`/`Scan`/`SearchMultimodal`) com delegação pro CLI (writes/deletes/evolve) — `ailake-go` por ser pure-Go sem cgo, `ailake-cpp` por ser deliberadamente header-only sem FFI Rust. Achado mais sério que gaps de cobertura: ao montar verificação real end-to-end pros novos métodos C++, apareceram 7 bugs de correção pré-existentes na leitura de catálogo/Avro do `ailake-cpp` — nenhum causado pelo código novo, todos reproduzíveis com `search()`/`list_files()` sem modificação — que faziam `search()` retornar zero resultados ou os arquivos errados contra qualquer tabela real com múltiplas linhas ou snapshots.

- [x] **`ailake-cli insert` — também faltava `--metric`/`--precision`/`--embedding-model`** — terceira ocorrência da mesma classe de bug da Fase 13: `ailake-go`'s `WriteBatch` e `ailake-cpp`'s `write_batch` já mandavam essas 3 flags incondicionalmente quando setadas, `Insert` nunca aceitou. Estendido o CLI, mesmo padrão de correção.
- [x] **`ailake-go` — `Compact` + `WriteBatch` multi-coluna** — `Compact(catalog, ns, table, opts)` (pede `--format json` desde o início, evitando o exato bug do `AilakeHook.compact()` da Fase 13); `WriteBatchOptions.VectorCols` habilita `--vector-cols col:dim:metric[:modality],...` (Phase 8 multimodal), capability do CLI que `ailake-go` nunca expunha.
- [x] **`ailake-go` — bug pré-existente de path duplicado** — `resolveAvroPath`/`searchFile`/`searchFileCol` rejuntavam paths já relativos à raiz do warehouse (manifest_list/DataFileEntry.path, que o `HadoopCatalog` Rust sempre grava relativos ao warehouse root, já incluindo namespace/table) contra o diretório da tabela de novo — path quebrado tipo `.../warehouse/default/media/default/media/...`. Corrigido nos 3 call sites, igual ao padrão já correto em `scan.go`'s `FetchRows`.
- [x] **`ailake-cpp` — `write_batch_multi` + `compact`** — espelham as adições do Go e o padrão de shell-out já usado por `write_batch`/`delete_where`/`evolve_schema` em `write.hpp`.
- [x] **`ailake-cpp` — 7 bugs pré-existentes de catálogo/Avro, achados e corrigidos durante a verificação real**:
  - `table_dir()` usava convenção Hive-style `<ns>.db/<tbl>` que não existe em nenhum outro lugar do projeto — layout real (confirmado contra `HadoopCatalog::table_root()` do Rust) é flat `<ns>/<tbl>`.
  - `resolve_path()`/`list_files()` — mesmo bug de double-prefix de path achado no Go.
  - `detail::read_zigzag()` sem checagem de EOF/falha — hang infinito lendo lixo da stack (achado via `strace`); agora lança `std::runtime_error` e ambos os manifest readers tratam EOF limpo em boundary de bloco como fim de loop normal (Rust nunca escreve terminador `count=0` final).
  - Bloco de dados Avro OCF sempre codifica `byte_size` (não só quando `count` é negativo — essa é a convenção de array/map genérico, não de data block top-level); parsing desalinhava por 4 bytes por arquivo.
  - `read_manifest_file()` faltava skip do campo final `first_row_id: union<null,long>` (V3 row lineage, sempre escrito pelo Rust mesmo null em V2) — desalinhava cada record por 1 campo.
  - `parse_key_metadata()` assumia `centroid_b64` empacotando `dim+1` floats (último = radius); na real `radius` é campo JSON top-level separado (`AilakeEntryExt::radius`) — centroid truncado por 1 dimensão + radius lido de bytes errados.
  - `search_file()` tratava `entry.hnsw_offset` do manifesto como posição do header AILK; na real já é o offset absoluto do blob do índice (`ailk_start + header.hnsw_offset`, ver `writer.rs`/`compaction.rs`) — lia o header no offset errado. Corrigido pra back-computar a posição do header, igual ao `searchFileAtOffset` já correto do `ailake-go`.
  - `list_files()`'s parsing de `current-snapshot-id` usava offset hardcoded `+=23` pra uma chave de 22 caracteres, derrubando o primeiro dígito do snapshot-id silenciosamente; o lookup de manifest-list então fazia busca de substring solta por esse ID (já errado) em qualquer lugar do arquivo — só "funcionava" por acidente em tabelas de snapshot único, e sempre resolvia pro primeiro snapshot do array (não o atual) em tabelas com histórico (ex: pós-compaction). Reescrito como walk depth-aware do array `snapshots` comparando o campo `snapshot-id` exato (tolerante a espaço).
- Fora de escopo (documentado): `ailake-cpp`'s `scan` — diferente do Go (que já embute `parquet-go` pro `Scan` nativo), `ailake-cpp` é deliberadamente header-only sem dependência de leitura Parquet; precisaria de nova dependência (contra o design) ou subcommand `scan` no CLI (não existe) — mesmo blocker já sinalizado fora de escopo na Fase 13.
- Testes reais: `ailake-go` — `go build`/`go vet` + round-trip real (`WriteBatch` multi-coluna, `SearchMultimodal` com RRF real, `Compact` mesclando 2 arquivos em 1 com todas as linhas ainda buscáveis) + suite completa de 83+ testes. `ailake-cpp` — `cmake --build` + `ctest` (5 suites verdes) + round-trip end-to-end real contra `target/debug/ailake`: `write_batch_multi` → `search_multimodal` (3 resultados RRF) → 2× `write_batch` → `compact` (1 arquivo) → `search` (12 linhas buscáveis) — cada um dos 7 bugs acima achado e corrigido iterando esse round-trip via debugging nível `strace` até os resultados baterem certo, não só review de código. Docs (`ailake-cpp/README.md`, `docs/guides/CPP_INTEGRATION.md`, `CHANGELOG.md`) atualizados.

### Fase 15 — Paridade ailake-py

> **Contexto (2026-07-09)**: mesma auditoria de paridade (padrão Fase 12/13/14), agora sobre o SDK Python (PyO3). Diferente das fases anteriores, a maioria dos achados aqui não era gap de cobertura — era bug de correção em binding já existente (comportamento silenciosamente errado).

- [x] **`assemble_context()` — dedup permanentemente morto** — binding hardcodava `embedding: None` em todo chunk, então o loop de dedup por distância coseno do `ContextAssembler` era sempre no-op pra qualquer `dedup_threshold`. Corrigido aceitando chave `"embedding"` opcional por chunk. Também passou a expor `group_by_document`/`max_chunks_per_document` (antes hardcoded via `..Default::default()`) e retorna `{"text", "chunk_count", "token_estimate"}` em vez de string pura (breaking change — `Agent.assemble_context()` atualizado pra desempacotar `["text"]`, mantendo seu próprio contrato de retornar str).
- [x] **`search(fetch_data=True)` derrubava `hybrid_text`/`text_column`/`bm25_weight`/`ef_search`/`pruning_threshold`/`rerank_factor`** — `search_with_data` só aceitava `(path,query,top_k,partition_filter)`, diferente do `search()` (modo pointer-only) que já suportava tudo. O mesmo objeto `SearchQuery` dava resultado diferente conforme chamasse `.to_list()` ou `.to_pandas()`. Corrigido dando paridade total de parâmetros; `rerank_factor` também ganhou exposição em `search()`/`search_multimodal()` (antes hardcoded `None` nos três, então tabelas IVF-PQ nunca ganhavam a passada de reranking exato que `SearchConfig::rerank_factor` existe especificamente pra corrigir).
- [x] **`now_ns()` produzia coluna que `decay_memories()` rejeita** — docstring do próprio `now_ns()` manda usar o retorno em `last_accessed_at` via `extra_columns`, mas `build_batch_with_extra` não tinha branch `Timestamp` (só bool/float/int64/string inferidos do primeiro valor Python) — virava `Int64`, tipo que `days_old_vec` explicitamente rejeita, crashando toda chamada seguinte de `decay_memories()`. Nova classe `ailake.TimestampNs(ns: int)` reconhecida por `build_batch_with_extra`, produzindo coluna `Timestamp(Nanosecond, UTC)` real.
- [x] **`Agent` incompatível com o resto do ecossistema e com `decay_memories()`** — `remember()`/`log_tool_call()` empacotavam toda a metadata como blob JSON prefixado na coluna `text` em vez de colunas reais tipadas — tabelas escritas via `Agent` eram opacas pra qualquer outro cliente AI-Lake, e `decay_memories()` não achava `last_accessed_at` real em nenhum arquivo, sempre retornando `0` silenciosamente. `recall()` também reimplementava decay de recência em Python puro com `time.time()` (segundos) em vez de `now_ns()` (nanossegundos) — dois mecanismos de recência paralelos e não-interoperáveis. Reescrito `remember()`/`log_tool_call()`/`recall()` pra escrever/ler colunas reais tipadas (`agent_id`, `session_id`, `step_index`, `mem_type`, `record_id`, `importance`, `created_at`/`last_accessed_at` como `TimestampNs`, `access_count`, `tool_name`, `tool_input_json`, `tool_output_json`, `outcome`, `latency_ms`) batendo com os nomes de campo de `EpisodicMemorySchema`/`ToolCallSchema` — `decay_memories()` agora funciona direto em tabelas escritas por `Agent`. Contrato público de retorno do `recall()` inalterado.
- [x] **`compact()` não usava a binding nativa, chamava binário CLI separado via subprocess** — apesar de `ailake_query::compaction` já estar linkado no `ailake-py`, o `compact()` Python fazia `subprocess.run` contra `$AILAKE_BIN`/`PATH`, retornando `{"ok": true, "files_compacted": 0, "warning": "..."}` silenciosamente quando o binário não existia — lido como sucesso. Substituído por função nativa chamando `CompactionPlanner`/`CompactionExecutor` direto (espelhando o handler `Compact` do próprio `ailake-cli`); corrigido também o default de `target_size_bytes` que divergia do CLI (128 MiB vs 512 MiB do CLI) — agora `536_870_912` nos dois.
- [x] **`add_vector_column`/`backfill_vector_column` compilados mas inacessíveis via `import ailake`** — implementados e registrados no módulo nativo, mas ausentes de `__init__.py`/`__all__`/`.pyi` — `AttributeError`. Exportados.
- [x] **`write_batch_ivf_pq`/`write_batch_ivf_pq_deferred` sem wrapper PyO3** — sem jeito de forçar IVF-PQ do Python (só a heurística de hardware do `write_batch_auto_deferred`). Adicionados como métodos de `TableWriter`.
- [x] **`write_batch_multi`/`_deferred` sem `extra_columns`** — sem jeito de escrever colunas companion de `MultimodalContextSchema` (`media_uri`, `media_caption`, etc) numa escrita multimodal. Adicionado `extra_columns` (reusa `build_batch_with_extra`, ganhando `TimestampNs` de graça também).
- [x] **`VectorColSpec` sem `precision`/`pre_normalize`/`hnsw_m`/`hnsw_ef_construction` por coluna** — toda coluna secundária sempre usava `VectorStoragePolicy::default_f16`. Adicionados os 4 campos.
- [x] **Sem `scan()`** — `ailake_scan_json` (ailake-jni) não tinha equivalente — na real `search_with_data()` já É essa capability (search + fetch de linha completa, sem JOIN); adicionado `ailake.scan` como alias de paridade de nome (igual ao `Scan()` do `ailake-go`), não reimplementação.
- [x] **Sem `estimate()`** — `Estimate` do CLI (matemática pura, sem I/O) não tinha equivalente Python nenhum, nem fallback via subprocess como o `compact()` antigo tinha. Adicionado `estimate()` nativo espelhando a matemática do CLI exatamente.
- [x] **`Table` (API fluente "recomendada") sem `write_batch_idempotent`/`write_batch_multi`/`write_batch_multi_deferred`** — só existiam no `TableWriter` de baixo nível. Adicionados métodos equivalentes em `Table`.
- [x] **`.pyi` stale declarava classe `Agent` fictícia** — com `@property agent_id`/`session_id` e `assemble_context()` retornando `str`, nunca batendo com a classe `Agent` real (Python puro, definida em `__init__.py`, atributos privados) — `Agent`/`Table`/`SearchQuery` não fazem parte da extensão `_ailake` compilada e não pertencem a esse stub. Bloco fictício removido; adicionados stubs de `TimestampNs`, `compact`, `estimate`, `add_vector_column`, `backfill_vector_column`, e atualizadas todas as assinaturas tocadas acima.
- [x] **Zero cobertura de teste** — explica por que os bugs de dedup morto, params derrubados e crash de Timestamp acima nunca foram pegos; só existia um compat script sem infra `pytest`. Adicionado `ailake-py/tests/test_capability_parity.py` (16 testes, build real via `maturin develop`, sem mocks) cobrindo cada fix acima; integrado no job `compat-ailake-py` do CI. `tests/compat/check_ailake_py.py` também atualizado pras novas asserções de `assemble_context` (era assert contra string pura com `dedup_threshold` que provadamente não fazia nada).
- Verificado com build real `maturin develop --release` contra venv real (Python 3.14, pyarrow 24, sem pandas — achou limitação real de `pyarrow.TimestampScalar.as_py()` em resolução nanossegundo, contornada com `.value`), suite pytest nova (16 testes), `tests/compat/check_ailake_py.py` completo (60+ checks, 0 regressões), `mypy` (0 erros), `cargo clippy --release -p ailake-py` (0 warnings). Docs (`ailake-py/README.md`, `_ailake.pyi`, `CHANGELOG.md`) atualizados.

---

## 11. Stack Técnica — Rust

### Crates do Projeto

```
ailake/
├── ailake-core/          # Tipos, traits, schema VECTOR, LlmContextSchema, RowId
├── ailake-parquet/       # Leitor/escritor Parquet com tipo VECTOR
├── ailake-vec/           # Quantização F32/F16/I8/PQ
├── ailake-index/         # HNSW via hnsw_rs, IVF-PQ — serialização bincode, mmap
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
