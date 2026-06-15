# ailake-go

Go reader for **AI-Lake Format** — Apache Iceberg-compatible Parquet files extended with HNSW / IVF-PQ vector indexes and geometric pruning statistics.

Zero CGO. Pure Go. Works on any platform where Go runs.

## Install

```bash
go get github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go
```

Requires Go 1.22+.

## Quick start

### Vector search (pointer results)

```go
import ailake "github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go"

catalog := &ailake.HadoopCatalog{Warehouse: "/data/warehouse"}

results, err := ailake.Search(catalog, "default", "docs", query, ailake.SearchOptions{
    TopK:             10,
    PruningThreshold: 0.8, // skip files whose centroid is far from query
})
// results: []FileSearchResult{RowID, Distance, FilePath}
```

### Full-row fetch (RAG path)

```go
rows, err := ailake.Scan(catalog, "default", "docs", query, ailake.SearchOptions{
    TopK: 10,
})
// rows: []ScanRow{RowID, Distance, FilePath, Fields map[string]any}
// Fields contains all Parquet columns; the vector column is decoded to []float32.
```

## API reference

### `HadoopCatalog`

Reads Iceberg metadata from a local filesystem or NFS mount.

```go
type HadoopCatalog struct {
    Warehouse string // root path, e.g. "/data/warehouse" or "s3://..." (via FUSE)
}
```

### `Search`

```go
func Search(
    catalog  *HadoopCatalog,
    namespace, table string,
    query    []float32,
    opts     SearchOptions,
) ([]FileSearchResult, error)
```

Runs geometric file pruning then vector search (HNSW or IVF-PQ) over all surviving shards. Results are merged and sorted by ascending distance.

**Geometric pruning**: each file stores its vector centroid and radius in Iceberg manifest metadata. Files where `distance(query, centroid) - radius > PruningThreshold` are skipped without I/O.

### `Scan`

```go
func Scan(
    catalog  *HadoopCatalog,
    namespace, table string,
    query    []float32,
    opts     SearchOptions,
) ([]ScanRow, error)
```

`Search` + Parquet row fetch in one call. Returns all columns for each top-K hit alongside `Distance`.

### `SearchOptions`

```go
type SearchOptions struct {
    TopK             int            // number of results (default: 10)
    EfSearch         int            // HNSW ef_search (default: TopK*5)
    PruningThreshold float32        // geometric pruning cutoff (default: 0.8)
    Hardware         *HardwareProfile // nil = auto-detect
}
```

### `FileSearchResult`

```go
type FileSearchResult struct {
    RowID    uint64
    Distance float32
    FilePath string
}
```

### `DataFileEntry`

```go
type DataFileEntry struct {
    Path           string
    FileSizeBytes  int64
    RecordCount    int64
    Centroid       []float32
    Radius         float32
    FooterOffset   int64
    EmbeddingModel string // "<name>" or "<name>@<version>"; empty if not set
}
```

`EmbeddingModel` is read from the per-file Avro `key_metadata` JSON (`"embedding_model"` field) written by the AI-Lake SDK at ingest time. Use it to detect mixed-model tables before searching.

### `TableInfo`

```go
type TableInfo struct {
    VectorDim      string
    VectorMetric   string
    VectorPrecision string
    EmbeddingModel string // global model from ailake.embedding-model property
}
```

### Dim validation in `Search()`

`Search()` validates `len(query)` against `TableInfo.VectorDim` before any I/O. If they differ, it returns an error naming the stored model:

```
ailake: query dim=512 does not match table dim=1536 (table model: text-embedding-3-small@v1)
```

### `ScanRow`

```go
type ScanRow struct {
    RowID    uint64
    Distance float32
    FilePath string
    Fields   map[string]any // all Parquet columns; vector column → []float32
}
```

## Index formats

Both HNSW and IVF-PQ indexes are supported transparently — the reader detects the index type from the AILK footer header flags.

### IVF-PQ: residual mode

When the index was written with `ivf_residual=true`, each file's trailing byte signals residual encoding. `DeserializeIvfPq` reads this automatically; `Search` uses per-cluster ADC tables for correct distance computation.

```go
// Low-level: deserialize an IVF-PQ blob directly.
idx, err := ailake.DeserializeIvfPq(blob)
results  := idx.Search(query, topK, nprobe) // []SearchResult{RowID, Distance}
```

## GPU delegation

When `AILAKE_SERVER_URL` is set, Go delegates IVF-PQ search to a running `ailake serve` instance (Rust, CUDA/ROCm). HNSW graph traversal always runs on CPU.

```bash
export AILAKE_SERVER_URL=http://localhost:7700
```

## Run the example

```bash
cd examples/search
go run . -warehouse /data/warehouse -table default.docs -dim 1536 -top-k 10
```

## Test

```bash
go test ./...
# Integration tests (require a fixture table):
AILAKE_FIXTURE=/path/to/fixture go test ./... -run Integration
```

33 unit tests pass without a fixture. 7 integration tests require `AILAKE_FIXTURE` (includes `TestListFilesIntegration` for `EmbeddingModel` and `TestSearchDimMismatchIntegration` for dim validation).

## License

MIT OR Apache-2.0 — same as the rest of the AI-Lake SDK.
