# AI-Lake Format

[![CI](https://github.com/ThiagoLange/ai-lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/ThiagoLange/ai-lakehouse/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ailake-core.svg)](https://crates.io/crates/ailake-core)
[![PyPI](https://img.shields.io/pypi/v/ailake.svg)](https://pypi.org/p/ailake)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-db61a2.svg)](https://github.com/sponsors/ThiagoLange)

> 🇧🇷 Você está lendo a versão em Português brasileiro. [Read in English →](./README.md)

Formato de Lakehouse nativo para vetores, construído sobre o Apache Iceberg Spec v2/v3, escrito em Rust.

**Arquivo único e auto-contido**: dados tabulares, embeddings e índice HNSW vivem juntos em um único arquivo Parquet estendido na camada S3. Transações ACID via Iceberg. Qualquer framework compatível com Iceberg lê tabelas AI-Lake sem modificação — o índice vetorial no rodapé do arquivo é invisível para leitores Parquet padrão.

---

## Por que AI-Lake?

**Sem segundo sistema.** Stacks tradicionais separam dados tabulares (Parquet/Iceberg) dos vetores (Pinecone, Milvus, Weaviate). Dois sistemas para operar, dois modelos de consistência, duas linhas de cobrança e um join através de uma fronteira de rede no momento da consulta. O AI-Lake colapsa ambos em um único arquivo `.parquet` — uma fonte da verdade, um log de transações, um prefixo S3.

**Vetores com ACID.** O isolamento de snapshot do Iceberg se aplica à busca vetorial da mesma forma que se aplica a consultas SQL. Time-travel, rollback e escritas concorrentes funcionam nativamente. Sem consistência eventual ou janelas de reconstrução de índice.

**Compatível com Iceberg por especificação, não por convenção.** Leitores Parquet padrão (Spark, Trino, DuckDB, Athena, Snowflake) lêem tabelas AI-Lake sem nenhum plugin. O índice HNSW vive no rodapé do arquivo após o marcador final `PAR1` — invisível para leitores que seguem a especificação Parquet. O vector scan é uma capacidade aditiva, não um fork do formato.

**Pruning geométrico reduz custos S3 antes de qualquer I/O.** Cada arquivo registra seu centróide vetorial e raio no manifesto Iceberg. Uma consulta elimina arquivos cujo centróide está geometricamente distante — sem abrir um único arquivo Parquet. Em tabelas com milhares de arquivos, 95–99% dos objetos nunca são buscados.

**Um binário, zero flags de compilação GPU.** NVIDIA cuBLAS e AMD hipBLAS são carregados em tempo de execução via `libloading` (FFI dinâmica — sem dependência em tempo de compilação). O mesmo binário de release seleciona automaticamente GPU em máquinas CUDA/ROCm e cai para AVX-512/AVX2/NEON SIMD em máquinas apenas CPU. Sem recompilação, sem feature flags, sem cabeçalhos de driver. NVIDIA CUDA Toolkit e AMD ROCm são softwares proprietários de seus respectivos fabricantes; o AI-Lake não os embute nem redistribui. Veja [`SETUP.md §8F`](./SETUP.md) para a nota completa de licenciamento.

**Core em Rust, Python e JVM como cidadãos de primeira classe.** O caminho de escrita/busca é Rust puro (sem pausas de GC, sem pressão no heap da JVM). Python recebe resultados `RecordBatch` PyArrow sem cópia. Spark, Trino e Flink recebem uma ponte JNA C-ABI — quatro funções exportadas compartilhadas pelos três plugins JVM.

**Eficiente em armazenamento em escala.** Quantização F16 reduz pela metade o armazenamento de vetores vs. F32. Product Quantization (IVF-PQ) reduz o footprint do índice 10–100× para workloads residentes em S3.

| | Iceberg sozinho | DB vetorial externo | **AI-Lake** |
|---|---|---|---|
| Transações ACID | ✅ | ❌ | ✅ |
| SQL via Spark / Trino | ✅ | ❌ | ✅ |
| Busca vetorial nativa | ❌ | ✅ | ✅ |
| Arquivo único / sistema único | ✅ | ❌ | ✅ |
| Pruning geométrico de arquivos | ❌ | ❌ | ✅ |
| Busca GPU (NVIDIA + AMD) | ❌ | Vendor-específico | ✅ |
| Time-travel em vetores | ❌ | ❌ | ✅ |

→ **[Argumento técnico completo — AI-Lake vs Iceberg vs LanceDB vs DBs vetoriais externos](docs/WHY_AILAKE.md)**

---

## Demo interativa (um único comando)

Sobe um ambiente local com MinIO, Nessie e JupyterLab pré-carregado com 500 documentos sintéticos e um índice HNSW — sem conta cloud, sem credenciais:

```bash
# Da raiz do repositório — constrói o wheel ailake-py na primeira execução (~3-5 min, cacheado depois)
docker compose -f tests/docker/compose-demo.yml up -d
```

Depois abra **http://localhost:8888** e execute os notebooks:

| Notebook | O que demonstra |
|---|---|
| `01_ailake_demo.ipynb` | Escrita, busca, IVF-PQ, PQ residual, escrita diferida, tuning HNSW, API async, estimador de storage, compat Iceberg, montagem de contexto RAG, upload MinIO, escrita multi-coluna, RRF cross-modal, `MultimodalContextSchema`, `delete_where`, `add_column`/`rename_column`, `partition_fields` + Iceberg v3 |
| `02_duckdb.ipynb` | Scan Parquet DuckDB, queries filtradas, estatísticas de storage por arquivo, decodificação F16 |
| `03_spark.ipynb` | PySpark local[*], SQL Iceberg, histórico de snapshots, time-travel `VERSION AS OF`, leitura de tabela particionada v3, visibilidade de delete, leitura de schema evolutivo |
| `04_trino.ipynb` | SQL Trino, propriedades de tabela AI-Lake, tabelas do sistema `$files` / `$manifests`, inspeção de DDL `partition_fields`, visibilidade de equality delete |
| `05_bigquery.ipynb` | Inserções no emulador BigQuery, decodificação BYTES F16, padrão GCS + BigQuery Omni em produção |
| `07_multimodal.ipynb` | `VectorColSpec`, `write_batch_multi`, tags de modalidade, fusão RRF cross-modal, ablação de pesos, constantes `MultimodalContextSchema` |
| `08_agents.ipynb` | `ailake.Agent`, memória episódica, `ToolCallSchema`, `EpisodicMemorySchema`, `WorkingMemoryBuffer`, `decay_memories`, isolamento por agente via partição |
| `09_hybrid_search.ipynb` | Escrita BM25 (`bm25_text_column`), `search_text` lexical puro, RRF híbrido (vetor + BM25), ablação de pesos |
| `10_gpu_demo.ipynb` | `hardware_info()`, `write_batch_auto_deferred`, comparação de tempo HNSW vs diferido, QPS de busca, recall@10, fallback CPU |
| `11_fts.ipynb` | FTS Tantivy por arquivo (`fts_text_columns`), `search_text` fast path O(log N), indexação multi-coluna, sintaxe de query, fallback BM25 para arquivos legados, re-ranking FTS + HNSW híbrido, layout de storage |
| `12_airflow.ipynb` | Apache Airflow 2.9 + provider AI-Lake: `AilakeWriteOperator`, `AilakeSearchOperator`, `AilakeFtsSearchOperator`, trigger de DAG via API REST, inspeção de XCom, padrão PythonOperator direto, configuração de conexão em produção |

Notebooks 03 e 04 requerem o perfil `engines` (adiciona Trino). Notebook 10 requer o perfil `gpu` (NVIDIA Container Toolkit). Notebook 12 requer o perfil `airflow`:

```bash
docker compose -f tests/docker/compose-demo.yml --profile engines up -d   # Trino
docker compose -f tests/docker/compose-demo.yml --profile gpu up -d        # JupyterLab GPU na :8889
docker compose -f tests/docker/compose-demo.yml --profile airflow up -d    # Airflow na :8090
```

Veja [`tests/docker/`](./tests/docker/) para detalhes dos arquivos compose.

> **Atualizando uma stack de demo já existente**: os dados de fixture vivem no
> volume nomeado `demo-data`, que sobrevive a `docker compose build` / `up`
> entre mudanças de código — só `down -v` (ou apagar o volume) remove. O
> `entrypoint.sh` do container detecta isso automaticamente: grava um
> `FIXTURE_VERSION` (vindo de `init_demo.py`) no volume após gerar as
> fixtures, compara a cada start, e limpa + regenera quando a versão da
> imagem em execução não bate com o que está em disco — então `docker
> compose build && docker compose up -d` já é suficiente pra pegar mudanças
> de fixture. Não precisa `down -v` manual, exceto pra debugar o próprio
> entrypoint.

---

## Orientação rápida

| Documento | O que responde |
|---|---|
| [`CLAUDE.md`](./CLAUDE.md) | Decisões arquiteturais, spec do formato, estratégia de storage, design de contexto LLM |
| [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) | Mapa de crates, grafo de dependências, instruções de build |
| [`docs/architecture/DATA_FLOW.md`](./docs/architecture/DATA_FLOW.md) | Caminho de escrita, leitura e compaction fim-a-fim |
| [`docs/architecture/CATALOG_BACKENDS.md`](./docs/architecture/CATALOG_BACKENDS.md) | Trait `CatalogProvider` + backends Hadoop / REST / Glue / Nessie / JDBC |
| [`docs/specs/FILE_FORMAT.md`](./docs/specs/FILE_FORMAT.md) | Especificação binária do arquivo `.parquet` unificado com rodapé AI-Lake |
| [`docs/specs/ICEBERG_COMPAT.md`](./docs/specs/ICEBERG_COMPAT.md) | Como a compatibilidade com leitores Iceberg é mantida |
| [`docs/specs/LLM_CONTEXT.md`](./docs/specs/LLM_CONTEXT.md) | `LlmContextSchema`, embeddings duplos, `ContextAssembler`, `MultimodalContextSchema`, RRF cross-modal |
| [`docs/specs/INTEGRATIONS.md`](./docs/specs/INTEGRATIONS.md) | Spark, Trino, Beam, AWS, GCP, Azure — snippets de config e matriz de compatibilidade |
| [`docs/specs/CLOUD_DEPLOY.md`](./docs/specs/CLOUD_DEPLOY.md) | Deploy passo-a-passo em EMR, Glue, Lambda, Dataproc, Dataflow, Databricks, HDInsight, AzureML |
| [`docs/specs/COMPACTION.md`](./docs/specs/COMPACTION.md) | Design do job de compaction, triggers, estratégia de reconstrução do HNSW |
| [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) | Estratégia de testes, fixtures, matriz CI, harness de testes de compat |
| [`docs/contributing/CODING_STANDARDS.md`](./docs/contributing/CODING_STANDARDS.md) | Convenções Rust, tratamento de erros, política de unsafe, regras de testes |
| [`docs/contributing/DECISIONS.md`](./docs/contributing/DECISIONS.md) | Log de ADRs — por que cada escolha-chave foi feita |
| [`SETUP.md`](./SETUP.md) | Setup de dev local — roda a stack completa (MinIO, Nessie, testes de compat) na sua máquina |
| [`docs/guides/DEMO_NOTEBOOKS.md`](./docs/guides/DEMO_NOTEBOOKS.md) | Guia passo-a-passo da demo — pré-requisitos, 12 notebooks, profiles, troubleshooting |

## Instalação

**Rust** (adicione ao `Cargo.toml`):
```toml
[dependencies]
ailake-core  = "0.0.27"
ailake-query = "0.0.27"   # search(), TableWriter, ContextAssembler, search_multimodal
ailake-store = "0.0.27"   # backends S3 / GCS / Azure / local
```

**Python**:
```bash
pip install ailake
```

```python
import ailake
import numpy as np

# Escrita
table = ailake.open_table("s3://meu-lake/docs/", dim=1536, metric="cosine")
table.insert(texts, np.array(embeddings, dtype=np.float32))
table.commit()

# Busca fluente — encadeável, nativo para DataFrame
df = ailake.search("s3://meu-lake/docs/", query_embedding, top_k=20).to_pandas()

# Leitura completa: todas as colunas Parquet + embedding + _distance
df = ailake.search("s3://meu-lake/docs/", query_embedding, top_k=20, fetch_data=True).to_pandas()

# Async
df = await table.search(query_embedding).limit(10).to_pandas_async()
```

**Apache Airflow**:
```bash
pip install apache-airflow-providers-ailake
```

**JVM (Spark / Trino / Flink)** — baixe os JARs pré-compilados em [GitHub Releases](https://github.com/ThiagoLange/ai-lakehouse/releases):

```bash
VERSION=0.0.27

# Plugin Spark
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/spark-plugin-${VERSION}-plugin.jar

# Plugin Trino
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/trino-plugin-${VERSION}-plugin.jar

# Conector Flink
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/ailake-flink-${VERSION}-plugin.jar

# Biblioteca nativa (necessária pelos três — coloque no java.library.path)
wget https://github.com/ThiagoLange/ai-lakehouse/releases/download/v${VERSION}/libailake_jni.so
```

Veja [`docs/specs/JVM_PLUGINS.md`](./docs/specs/JVM_PLUGINS.md) para instalação e configuração.

## Layout do repositório

```
ailake/
├── CLAUDE.md
├── README.md
├── README.pt-BR.md
├── Cargo.toml                  # raiz do workspace
├── docs/
│   ├── architecture/
│   ├── specs/
│   └── contributing/
├── ailake-core/                # Tipos, traits, schema VECTOR, LlmContextSchema, RowId
├── ailake-parquet/             # Leitor/escritor da seção Parquet
├── ailake-vec/                 # Quantização F32/F16/I8, distância, PQ, compressão
├── ailake-file/                # Arquivo unificado: Parquet + rodapé AI-Lake
├── ailake-catalog/             # Catálogo Iceberg: metadata.json, manifestos Avro
├── ailake-store/               # Abstração de object storage (S3/GCS/Azure/local)
├── ailake-index/               # HNSW via hnsw_rs, IVF-PQ, backends GPU
├── ailake-query/               # Pruning, scan, TableWriter, ContextAssembler
├── ailake-cli/                 # CLI: ailake create / insert / search / compact / info / serve / estimate
├── ailake-py/                  # Bindings Python via PyO3 (wheel abi3-py39)
├── ailake-jni/                 # C-ABI cdylib para Spark/Trino/Flink via JNA
├── duckdb-ailake/              # Extensão DuckDB em C++
├── spark-plugin/               # Plugin Spark em Scala (Gradle)
├── trino-plugin/               # Conector Trino em Kotlin (Gradle)
├── ailake-flink/               # Conector Flink em Kotlin (Gradle)
├── ailake-fts/                 # Índice Tantivy FTS por arquivo (Phase T — Busca full-text)
├── airbyte-destination-ailake/ # Destino Airbyte CDK (Python)
├── ailake-go/                  # SDK Go puro, sem CGo
├── ailake-cpp/                 # SDK C++17 header-only
└── airflow-providers-ailake/   # Provider Apache Airflow 2.x/3.x
tests/
├── Cargo.toml
├── src/lib.rs
├── tests/
│   ├── write_read_roundtrip.rs
│   ├── iceberg_compat.rs
│   ├── parquet_trailing_bytes.rs
│   ├── vector_pruning.rs
│   ├── positional_invariant.rs
│   ├── context_assembler.rs
│   ├── hybrid_search.rs
│   ├── concurrent_writes.rs
│   ├── partition_isolation.rs
│   ├── fts_fast_path.rs
│   └── fixtures/mod.rs
├── fixtures/
│   ├── write_fixture.py
│   └── write_jni_fixture.py
├── compat/
│   ├── check_pyarrow.py
│   ├── check_ailake_py.py
│   ├── check_jni_cabi.py
│   ├── check_pyiceberg.py
│   └── check_duckdb.py
└── docker/
    ├── compose.yml              # MinIO + Nessie + Localstack
    ├── compose-engines.yml      # + Spark + Trino
    ├── compose-demo.yml         # Demo de onboarding; --profile engines/gpu/airflow
    └── demo/
        ├── Dockerfile           # Dois estágios: Rust/maturin → JupyterLab
        ├── Dockerfile.airflow   # Imagem Airflow 2.x com provider ailake instalado
        ├── entrypoint.sh        # Gera fixtures e inicia Jupyter
        ├── airflow-entrypoint.sh # Inicia DB + scheduler + webserver
        ├── init_demo.py         # Gera 11 tabelas fixture (HNSW, PQ-only, Residual-PQ, Deferred, Model-tracked, Multimodal, Agent-memory, Delete-demo, Schema-evo, Partitioned-v3, FTS)
        ├── dags/
        │   ├── dag_ailake_ingest_search.py  # DAG TaskFlow: ingestão + busca vetorial
        │   └── dag_ailake_compaction.py     # DAG de compaction agendado
        ├── trino-catalog/
        │   └── ailake.properties
        └── notebooks/
            ├── 01_ailake_demo.ipynb
            ├── 02_duckdb.ipynb
            ├── 03_spark.ipynb
            ├── 04_trino.ipynb
            ├── 05_bigquery.ipynb
            ├── 06_airbyte_destination.ipynb
            ├── 07_multimodal.ipynb
            ├── 08_agents.ipynb
            ├── 09_hybrid_search.ipynb
            ├── 10_gpu_demo.ipynb
            ├── 11_fts.ipynb
            └── 12_airflow.ipynb
```

## Storage

Estimativas para `text-embedding-3-small` (dim=1536), 100 M vetores.

| Modo | Coluna vetorial | Overhead HNSW/IVF-PQ | Total |
|---|---|---|---|
| F32 (raw) | ~600 GB | ~60–120 GB | ~660–720 GB |
| F16 (padrão) | ~300 GB | ~30–60 GB | ~330–360 GB |
| I8 | ~150 GB | ~15–30 GB | ~165–180 GB |
| IVF-PQ (M=48, K=256) | ~300 GB raw + ~5 GB códigos PQ | ~5 GB | ~310 GB |
| PQ-only (`--pq-only`) | 0 GB (raw omitido) | ~5 GB | **~5 GB** |

Modo PQ-only troca precisão de reranking por 98% de redução de storage. Recall@10 ~93–95%.

**Pruning geométrico** elimina 95–99% dos arquivos antes de qualquer índice ser acessado em tabelas com milhares de shards.

> **NormalizedCosine**: `pre_normalize=True` normaliza vetores para L2 unitário na escrita, substituindo a distância cosseno por `1−dot(a,b)` no hot loop do HNSW (sem `sqrt`). Redução de latência de ~12–20% em dim=1536 (embeddings OpenAI, Cohere). Ative via `ailake create --pre-normalize` ou `TableWriter(pre_normalize=True)`.

> **Tantivy FTS**: quando `fts_columns` está definido, cada arquivo embute um índice invertido por arquivo (seção `AILK_FTS`, comprimida com zstd). Adiciona ~3–4 MB por arquivo (~7 GB para tabela com 2.000 shards a 50 k docs/arquivo) — pequeno em relação ao overhead da coluna vetorial.

---

## Exemplos de código

| Linguagem | Local | Executar |
|---|---|---|
| **Rust** (escrita + busca) | [`ailake-query/examples/demo.rs`](./ailake-query/examples/demo.rs) | `cargo run --example demo -p ailake-query` |
| **Python** (API fluente, async, RAG) | [`ailake-py/README.md`](./ailake-py/README.md) | `python -c "import ailake; ..."` |
| **Go** (busca, scan) | [`ailake-go/examples/search/main.go`](./ailake-go/examples/search/main.go) | `go run . -warehouse /data/warehouse -table default.docs` |
| **C++** (busca, CUDA) | [`ailake-cpp/examples/search.cpp`](./ailake-cpp/examples/search.cpp) | `./build/ailake_search -w /data/warehouse -t default.docs` |
| **Multi-engine** (Spark + Trino + DuckDB) | [`tests/docker/`](./tests/docker/) | `docker compose -f tests/docker/compose-demo.yml up -d` |

## Build

```bash
cargo build --workspace
cargo build --workspace --release
cargo test --workspace
cd ailake-py && maturin develop
cargo check --workspace
```

## Status das fases

| Fase | Status | Escopo |
|---|---|---|
| **Fase 1** | ✅ Completa | MVP local — escrita + busca no filesystem, rodapé HNSW, catálogo Iceberg |
| **Fase 2** | ✅ Completa | Cloud storage (`ObjectStoreBackend`), carregamento HNSW via mmap, compaction, PQ, pruning geométrico, `ContextAssembler`, bindings PyO3 |
| **Fase 3** | ✅ Completa | Backends de catálogo (Nessie/JDBC/Glue), bindings JNA C-ABI, vetores multi-coluna, plugins Spark/Trino/Flink |
| **Fase 4** | ✅ Completa | Reranking pós-PQ, spec pública do formato, busca GPU (NVIDIA cuBLAS + AMD hipBLAS, ambos runtime-only), otimizações HNSW, índice nativo IVF-PQ, k-means GPU, `MemTableWriter`, colunas multi-vetor, seleção adaptativa de índice, conector Kotlin `ailake-flink`; **codebook compartilhado IVF-PQ**; **`write_batch_ivf_pq_deferred`**; **fix k-means++ O(n×k)**; **fix `HadoopCatalog` Replace** |
| **Fase 5** | ✅ Completa | SDKs multi-linguagem (`ailake-go`, `ailake-cpp`), servidor HTTP REST `ailake serve`, provider Apache Airflow, escritas idempotentes, CI Compat Heavy, scanning de segredos TruffleHog, guias de deploy em cloud |
| **Fase 6** | ✅ Completa | Pipeline de distribuição pública — crates.io, PyPI (wheels manylinux abi3), provider Airflow no PyPI, JARs JVM pré-compilados + `libailake_jni.so` no GitHub Releases, versionamento Python dinâmico |
| **Fase 7** | 🚧 Em andamento | Concluído: extensão DuckDB (`duckdb-ailake/`), leitura completa Python (`fetch_data=True`), `write_batch_auto_deferred` + async (~200k vec/s), `pq_only` / `ivf_residual` expostos no SDK Python, guia dbt (`docs/guides/DBT_INTEGRATION.md`), `partition_fields` (spec de partição Iceberg multi-coluna), `format_version=3` (tabelas Iceberg v3), `delete_where` + `evolve_schema` em todos os SDKs (Python, Go, C++, Spark, Trino, Flink, DuckDB, Airflow, Airbyte), binding `hardware_info()` Python, notebook de demo GPU (`10_gpu_demo.ipynb`), demo JupyterLab expandida (10 notebooks), **FTS Tantivy por arquivo** (crate `ailake-fts` — seção `AILK_FTS`, zstd; fast path `search_text()` O(log N); opt-in via `fts_columns` em todos os SDKs e plugins JVM), **busca híbrida BM25+vetor** (`SearchConfig::hybrid`, fusão RRF, fallback BM25 brute-force para arquivos legados). Restante: backend de catálogo DuckLake |
| **Fase 8** | ✅ Completa | Multimodal — enum `VectorModality`, propriedade Iceberg `ailake.modality-<col>`, N colunas vetoriais generalizadas com HNSW independente, `write_batch_multi`, CLI `--vector-cols`, `search_multimodal` (RRF cross-modal), `MultimodalContextSchema` + módulo `multimodal_columns`, Python `VectorColSpec`, notebook e fixture multimodal |
| **Fase 9** | ✅ Completa | Memória de agentes — `ToolCallSchema` (histórico de tool calls pesquisável), `EpisodicMemorySchema` (decaimento de recência, contagem de acesso, pontuação de importância), `ScoreFn` injetável para scoring híbrido (distância × recência × importância), `partition_by`/`partition_value` para isolamento por agente via particionamento Iceberg, `partition_filter` para pruning ao nível de manifesto antes de centroide e HNSW, helper Python `ailake.Agent` (LangChain/CrewAI/AutoGen). Propagado para todos os SDKs e conectores: Spark, Trino, Flink, Go, C++, DuckDB, Airbyte, Airflow. Fix: `TableWriter::create_or_open` inicializa `part_counter` a partir da contagem de arquivos existentes. |

## Apoie o projeto

Se o AI-Lake é útil pra você, considera [apoiar via GitHub Sponsors](https://github.com/sponsors/ThiagoLange) — financia o desenvolvimento e manutenção contínua.

## Sponsors

_Seja o primeiro a patrocinar este projeto! → [github.com/sponsors/ThiagoLange](https://github.com/sponsors/ThiagoLange)_

Veja [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) para o detalhamento completo das fases.
