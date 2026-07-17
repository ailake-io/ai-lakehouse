# Go Integration Guide

The `ailake-go` package is a pure-Go client for AI-Lake tables. Vector search
and catalog reads are implemented entirely in Go (no cgo, no CGO). Write
operations, compaction, FTS search, and schema evolution delegate to the
`ailake` CLI binary, so those require `ailake` on `PATH` or `AILAKE_BIN` set.

---

## 1. Installation

```bash
go get github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go
```

**Runtime dependency for write/delete/FTS operations:**

```bash
# Install ailake CLI (see SETUP.md for build instructions)
cargo install --path ailake-cli
# or export AILAKE_BIN=/path/to/ailake
```

---

## 2. Catalog — connecting to a table

All operations take a `*ailake.HadoopCatalog` pointing to the warehouse root.

```go
import ailake "github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go"

catalog := &ailake.HadoopCatalog{Warehouse: "/data/warehouse"}
// S3 path (read metadata from local mount or use ailake CLI for writes):
// catalog := &ailake.HadoopCatalog{Warehouse: "s3://my-lake/warehouse"}
```

**Inspect table metadata:**

```go
info, err := catalog.LoadTable("default", "docs")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("table:      %s\n", info.Table)
fmt.Printf("dim:        %s\n", info.VectorDim)
fmt.Printf("metric:     %s\n", info.VectorMetric)
fmt.Printf("files:      %d (%d indexed)\n", info.Files, info.IndexedFiles)
fmt.Printf("rows:       %d\n", info.Rows)
fmt.Printf("format:     v%d\n", info.FormatVersion)
```

---

## 3. Writing data

`WriteBatch` wraps `ailake insert`. Pass a local Parquet file that contains
the embedding column alongside the other columns.

```go
err := ailake.WriteBatch(
    catalog,
    "default",         // namespace
    "docs",            // table name
    "/tmp/batch.parquet", // local Parquet file with embedding column
    ailake.WriteBatchOptions{
        VecCol:    "embedding",
        Metric:    "cosine",
        Precision: "f16",
    },
)
```

**All options:**

```go
ailake.WriteBatchOptions{
    VecCol:             "embedding",      // column with float32 vectors (default)
    Metric:             "cosine",         // cosine | euclidean | dot
    Precision:          "f16",            // f32 | f16 | i8 (f16 default)
    EmbeddingModel:     "text-embedding-3-small",
    PartitionBy:        "agent_id",       // optional single partition column
    PartitionValue:     "agent-001",      // value for this batch's partition
    FormatVersion:      2,                // Iceberg format version (2 or 3)
    FtsColumns:         []string{"chunk_text"}, // build Tantivy FTS index
    FtsTokenizer:       "default",
    HnswM:              16,               // HNSW M (0 = table default)
    HnswEfConstruction: 200,              // HNSW ef_construction (0 = default)
    PreNormalize:       false,            // normalize vectors to unit L2 at write
    Deferred:           false,            // commit Parquet now, build index async
    VectorCols:         nil,              // multi-column (multimodal) write — see below
}
```

**Multi-column (Phase 8 multimodal) writes:**

`WriteBatchOptions.VectorCols` writes several vector columns from the same
Parquet file in one call, each getting its own HNSW section in the resulting
AI-Lake file. When `VectorCols` is non-empty, `VecCol`/`Metric`/`Precision`
are ignored — the CLI's `--vector-cols col:dim:metric[:modality],...` spec
carries metric per column, and multi-column mode always writes F16.

```go
err := ailake.WriteBatch(
    catalog,
    "default", "media",
    "/tmp/batch.parquet", // Parquet file with both embedding columns
    ailake.WriteBatchOptions{
        VectorCols: []ailake.VectorColSpec{
            {Column: "embedding", Dim: 1536, Metric: "cosine"},
            {Column: "image_embedding", Dim: 512, Metric: "cosine", Modality: "image"},
        },
    },
)
```

Query it back with `SearchMultimodal` (§7).

A row with a `NaN`/`Infinity` embedding value is rejected by the CLI; the returned
`error` includes the CLI's stderr with the actual reason (`embedding contains
non-finite value (...); NaN/Infinity embeddings are rejected at write time`).

**Compaction:**

`Compact` merges small files into a larger one by delegating to `ailake
compact --format json`. Returns the number of files compacted (0 = nothing
eligible, e.g. fewer than `MinFiles` small files present).

```go
filesCompacted, err := ailake.Compact(
    catalog,
    "default", "docs",
    ailake.CompactOptions{
        TargetSize:      0,     // 0 = CLI default, 512 MiB
        MinFiles:        4,     // 0 = CLI default, 4
        MaxFilesPerPass: 20,    // 0 = CLI default, 20 — bounds peak RAM / HNSW rebuild cost
        Deferred:        false, // true = commit merged Parquet now, rebuild HNSW async
    },
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("compacted %d files\n", filesCompacted)
```

**Preparing a Parquet file in Go:**

The `ailake-go` package does not ship a Parquet writer. Use
[`parquet-go`](https://github.com/parquet-go/parquet-go) to produce the batch:

```go
import (
    "github.com/parquet-go/parquet-go"
    "os"
)

type DocRow struct {
    ChunkID   string    `parquet:"chunk_id"`
    ChunkText string    `parquet:"chunk_text"`
    Embedding []float32 `parquet:"embedding,list"`
}

rows := []DocRow{
    {ChunkID: "a", ChunkText: "hello world", Embedding: myEmbedding},
}

f, _ := os.Create("/tmp/batch.parquet")
defer f.Close()
if err := parquet.Write(f, rows); err != nil {
    log.Fatal(err)
}
```

---

## 4. Vector search

`Search` is implemented in pure Go — no CLI required.

```go
query := []float32{ /* 1536 floats */ }

results, err := ailake.Search(
    catalog,
    "default", "docs",
    query,
    ailake.SearchOptions{
        TopK:             10,
        PruningThreshold: 0.8,  // geometric pruning aggressiveness (0-1)
        EfSearch:         50,   // HNSW ef_search; 0 = TopK*5
    },
)
if err != nil {
    log.Fatal(err)
}
for _, r := range results {
    fmt.Printf("row_id=%d  distance=%.4f  file=%s\n",
        r.RowID, r.Distance, r.FilePath)
}
```

**Partition-filtered search (Phase 9 agent memory):**

```go
results, err := ailake.Search(
    catalog, "default", "agent_memory",
    query,
    ailake.SearchOptions{
        TopK:            20,
        PartitionFilter: "agent-001", // restrict to one agent's files
    },
)
```

**How it works:**

1. `LoadTable` reads `metadata.json` → gets vector metric and snapshot.
2. `ListFiles` reads manifest Avro → gets per-file centroid + radius.
3. Geometric pruning: files where `distance(query, centroid) - radius > threshold` skipped.
4. Surviving files: AILK section read from byte offset, HNSW/IVF-PQ deserialized, searched.
5. Results merged globally, top-K returned.

---

## 5. Full-text search (FTS)

Requires `ailake` CLI. Uses Tantivy O(log N) when FTS index present; falls back
to BM25 brute-force for legacy files.

`SearchText`/`SearchHybrid`'s `top_k` is capped at 100,000 by the underlying
`ailake_query` core (same limit enforced at the JNI C-ABI boundary used by
Spark/Trino/Flink) — a value above that fails cleanly instead of risking an
out-of-memory subprocess. The pure-Go `Search`/`SearchMultimodal`/`Scan` paths
(§4, §7) do not go through the CLI and are not currently subject to this cap.

```go
hits, err := ailake.SearchText(
    catalog,
    "default", "docs",
    "machine learning embeddings",   // query string
    []string{"chunk_text"},          // columns to search
    10,                              // top-K
)
for _, h := range hits {
    fmt.Printf("row_id=%d  score=%.4f  file=%s\n",
        h.RowID, h.Score, h.FilePath)
}
```

---

## 6. Hybrid search (BM25 + vector)

Fuses lexical BM25 and semantic vector results via Reciprocal Rank Fusion.
Requires `ailake` CLI.

```go
hits, err := ailake.SearchHybrid(
    catalog,
    "default", "docs",
    query,                            // float32 embedding vector
    "machine learning embeddings",   // BM25 text query
    10,                              // top-K
    0.5,                             // BM25 weight in RRF (0=pure vector, 1=pure BM25)
    "chunk_text",                    // column for BM25 scoring
)
for _, h := range hits {
    fmt.Printf("row_id=%d  distance=%.4f  file=%s\n",
        h.RowID, h.Distance, h.FilePath)
}
```

---

## 7. Multimodal / multi-column search

Cross-modal RRF fusion over multiple vector columns. Pure Go — no CLI.

```go
textQuery  := []float32{ /* 1536 floats */ }
imageQuery := []float32{ /* 512 floats  */ }

results, err := ailake.SearchMultimodal(
    catalog,
    "default", "media",
    []ailake.ModalQuery{
        {Column: "text_embedding",  Query: textQuery,  Weight: 0.7},
        {Column: "image_embedding", Query: imageQuery, Weight: 0.3},
    },
    ailake.SearchOptions{TopK: 10},
)
for _, r := range results {
    fmt.Printf("row_id=%d  rrf=%.4f  file=%s\n",
        r.RowID, r.RRFScore, r.FilePath)
}
```

---

## 8. Deletes and schema evolution

Both delegate to the `ailake` CLI.

**Logical delete (Iceberg equality delete — no data rewrite):**

```go
err := ailake.DeleteWhere(
    catalog,
    "default", "docs",
    "chunk_id",                              // column to match
    []string{"uuid-aaa", "uuid-bbb"},        // values to delete
)
```

**Schema evolution — add / rename columns:**

```go
newSchemaID, err := ailake.EvolveSchema(
    catalog,
    "default", "docs",
    []ailake.AddColumnReq{
        {Name: "language", Type: "string", InitialDefault: `"en"`},
        {Name: "page_number", Type: "int", InitialDefault: "null"},
    },
    []ailake.RenameColumnReq{
        {From: "chunk_text", To: "text"},
    },
)
fmt.Printf("new schema_id: %d\n", newSchemaID)
```

---

## 9. GPU delegation

The Go client never loads CUDA/ROCm directly (zero cgo). When GPU acceleration
is needed for IVF-PQ batch search, start `ailake serve` (Rust, CUDA-enabled)
and point Go to it:

```bash
# Start GPU search server (Rust binary with CUDA support)
AILAKE_SERVER_URL=http://localhost:7700 ./ailake serve --port 7700
```

```go
// Go automatically uses HTTP delegation when env is set
os.Setenv("AILAKE_SERVER_URL", "http://localhost:7700")

// Search calls below use GPU IVF-PQ automatically
results, _ := ailake.Search(catalog, "default", "docs", query, opts)
```

For HNSW tables, Go always uses the CPU greedy traversal — the graph structure
is sequential by nature and GPU does not help.

---

## 10. Error handling

```go
results, err := ailake.Search(catalog, "default", "docs", query, opts)
if err != nil {
    // Common errors:
    // - "ailake: list files: ..." — metadata unreadable
    // - "ailake: query dim=512 does not match table dim=1536 ..."
    // - "ailake: no CLI binary found (set AILAKE_BIN ...)" — write/FTS ops only
    log.Fatal(err)
}

_, err = ailake.WriteBatch(catalog, "default", "docs", "/tmp/batch.parquet", opts)
if errors.Is(err, ailake.ErrNoBinary) {
    log.Println("ailake CLI not found — set AILAKE_BIN or add to PATH")
}
```

---

## 11. Full example — RAG pipeline

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    ailake "github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go"
    openai "github.com/sashabaranov/go-openai"
)

const (
    warehouse = "/data/warehouse"
    namespace = "default"
    table     = "docs"
)

func main() {
    catalog := &ailake.HadoopCatalog{Warehouse: warehouse}

    // 1. Embed the user query
    oai := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
    resp, err := oai.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
        Input: []string{"What is geometric pruning in AI-Lake?"},
        Model: openai.SmallEmbedding3,
    })
    if err != nil {
        log.Fatal(err)
    }
    query := resp.Data[0].Embedding

    // 2. Hybrid search: BM25 + vector RRF
    hits, err := ailake.SearchHybrid(
        catalog, namespace, table,
        query,
        "geometric pruning vector index",
        10,
        0.4,         // BM25 weight
        "chunk_text",
    )
    if err != nil {
        log.Fatal(err)
    }

    // 3. Print results (load full rows with your Parquet reader of choice)
    for i, h := range hits {
        fmt.Printf("[%d] row_id=%-8d distance=%.4f  %s\n",
            i+1, h.RowID, h.Distance, h.FilePath)
    }
}
```

---

## 12. Package surface reference

| Symbol | Kind | Needs CLI | Description |
|---|---|---|---|
| `HadoopCatalog` | struct | No | Catalog pointing to local/S3 warehouse |
| `catalog.LoadTable(ns, name)` | method | No | Returns `TableInfo` (dim, metric, files, rows, schema) |
| `catalog.ListFiles(ns, name)` | method | No | Returns `[]DataFileEntry` with centroid, HNSW offset |
| `Search(catalog, ns, name, query, opts)` | func | No | Vector search (geometric pruning + HNSW/IVF-PQ) |
| `Scan(catalog, ns, name, query, opts)` | func | No | Search + full-row fetch in one call (no JOIN needed) |
| `SearchMultimodal(catalog, ns, name, queries, opts)` | func | No | Cross-modal RRF fusion |
| `ReadAilakeHeader(path)` | func | No | Introspect AILK section of any `.parquet` file |
| `DecodeF16Vector(raw, dim)` | func | No | Decode F16 Parquet column to `[]float32` |
| `DetectHardware()` | func | No | Reports CPU SIMD / CUDA / ROCm |
| `WriteBatch(catalog, ns, name, parquet, opts)` | func | **Yes** | Ingest Parquet batch + build HNSW (single- or multi-column via `opts.VectorCols`) |
| `Compact(catalog, ns, name, opts)` | func | **Yes** | Merge small files; returns files-compacted count |
| `DeleteWhere(catalog, ns, name, col, vals)` | func | **Yes** | Iceberg equality delete |
| `EvolveSchema(catalog, ns, name, add, rename)` | func | **Yes** | Add/rename columns (metadata-only) |
| `SearchText(catalog, ns, name, query, cols, k)` | func | **Yes** | FTS (Tantivy or BM25 fallback) |
| `SearchHybrid(catalog, ns, name, vec, text, k, w, col)` | func | **Yes** | BM25+vector RRF |
| `ErrNoBinary` | var | — | Returned when CLI not found |

**Schema / agent types:**

| Type | Description |
|---|---|
| `SearchOptions` | `TopK`, `EfSearch`, `PruningThreshold`, `PartitionFilter`, `Hardware` |
| `WriteBatchOptions` | All write parameters incl. `VectorCols []VectorColSpec` for multi-column writes (see §3) |
| `VectorColSpec` | `{Column, Dim, Metric, Modality}` — one column in a multi-column write |
| `CompactOptions` | `{TargetSize, MinFiles, MaxFilesPerPass, Deferred}` (see §3) |
| `ModalQuery` | `{Column, Query, Weight}` for multimodal search |
| `FileSearchResult` | `{RowID, Distance, FilePath}` |
| `ScanRow` | `{RowID, Distance, FilePath, Fields}` — `Fields` holds every Parquet column |
| `RRFResult` | `{RowID, RRFScore, FilePath}` |
| `SearchHybridResult` | `{RowID, Distance, FilePath}` |
| `SearchTextResult` | `{RowID, Score, FilePath}` |
| `TableInfo` | Full table metadata incl. `SchemaFields`, `PartitionFields` |
| `DataFileEntry` | Per-file metadata (centroid, radius, HNSW offset, index status) |
| `ToolCallSchema` | Column layout for agent tool-call history tables |
| `EpisodicMemorySchema` | Column layout for agent episodic memory tables |

---

## Related docs

- [File Format Spec](../specs/FILE_FORMAT.md) — AILK section layout
- [JVM Plugins](../specs/JVM_PLUGINS.md) — Spark / Trino / Flink (C-ABI + JNA)
- [LLM Context](../specs/LLM_CONTEXT.md) — `LlmContextSchema` for RAG
- [GPU FFI Evaluation](../specs/GPU_FFI_EVALUATION.md) — CUDA/ROCm strategy
- [ailake-go source](../../ailake-go/) — package source and tests
