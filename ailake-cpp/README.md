# ailake-cpp

Header-only C++17 reader for **AI-Lake Format** — Apache Iceberg-compatible Parquet files extended with HNSW / IVF-PQ vector indexes and geometric pruning statistics.

- CPU-only by default, no proprietary SDKs required.
- Optional CUDA acceleration (`-DAILAKE_CUDA=ON`).
- Optional AMD ROCm flat-scan via runtime `dlopen` (no ROCm SDK at build time).

## Requirements

| Requirement | Minimum |
|---|---|
| C++ standard | C++17 |
| CMake | 3.20 |
| CUDA Toolkit | 11.0+ (optional, `-DAILAKE_CUDA=ON`) |

## Build

```bash
cmake -B build
cmake --build build
ctest --test-dir build          # footer, hnsw, ivfpq tests
```

With CUDA:

```bash
cmake -B build -DAILAKE_CUDA=ON
cmake --build build
```

Options:

| CMake option | Default | Description |
|---|---|---|
| `AILAKE_CUDA` | `OFF` | Enable NVIDIA CUDA IVF-PQ kernel and flat-scan |
| `AILAKE_EXAMPLES` | `ON` | Build `examples/search.cpp` |
| `AILAKE_TESTS` | `ON` | Build unit tests (footer, hnsw, ivfpq) |

## Use in your project

### CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(ailake
  GIT_REPOSITORY https://github.com/ThiagoLange/iceberg-ai-deltalakehouse
  GIT_TAG        main
  SOURCE_SUBDIR  ailake-cpp
)
FetchContent_MakeAvailable(ailake)

target_link_libraries(my_app PRIVATE ailake::ailake ailake::ailake_catalog)
```

### add_subdirectory

```cmake
add_subdirectory(ailake-cpp)
target_link_libraries(my_app PRIVATE ailake ailake_catalog)
```

### Single header include

```cpp
#include <ailake/ailake.hpp>   // pulls all headers
```

## Quick start

```cpp
#include <ailake/ailake.hpp>
#include <vector>
#include <iostream>

int main() {
    ailake::HadoopCatalog catalog("/data/warehouse");

    std::vector<float> query(1536, 0.0f);
    query[0] = 1.0f; // your embedding here

    ailake::SearchOptions opts;
    opts.top_k             = 10;
    opts.pruning_threshold = 0.8f; // skip files whose centroid is far from query

    auto results = ailake::search(catalog, "default", "docs",
                                  query.data(), query.size(), opts);

    for (size_t i = 0; i < results.size(); ++i)
        printf("%zu  dist=%.4f  row=%llu  file=%s\n",
               i + 1, results[i].distance,
               (unsigned long long)results[i].row_id,
               results[i].file_path.c_str());
}
```

## API reference

### `ailake::search`

```cpp
std::vector<FileSearchResult>
search(HadoopCatalog& catalog,
       const std::string& ns,
       const std::string& table,
       const float* query, size_t dim,
       const SearchOptions& opts = {});
```

Runs geometric pruning across all manifest entries, then per-file HNSW or IVF-PQ search, then top-K merge. `NormalizedCosine` tables auto-normalize the query.

### `SearchOptions`

```cpp
struct SearchOptions {
    int         top_k             = 10;
    int         ef_search         = 0;        // 0 → top_k * 5
    float       pruning_threshold = 0.8f;
    bool        use_flat_fallback = true;     // flat scan when index not yet built
    std::string partition_filter;             // "" = no filter; restrict to matching partition_value (Phase 9)
    const HardwareProfile* hw = nullptr;     // nullptr = auto-detect
};
```

Set `partition_filter` to restrict search to files written with a matching `partition_value`. Pruning happens at the manifest level before any HNSW I/O:

```cpp
ailake::SearchOptions opts;
opts.top_k            = 10;
opts.partition_filter = "agent-42";

auto results = ailake::search(catalog, "default", "agents",
                              query.data(), query.size(), opts);
```

### `FileSearchResult`

```cpp
struct FileSearchResult {
    uint64_t    row_id;
    float       distance;
    std::string file_path;
};
```

### `HadoopCatalog`

```cpp
struct HadoopCatalog {
    explicit HadoopCatalog(std::string warehouse_root);

    TableInfo            load_table(const std::string& ns, const std::string& tbl);
    std::vector<DataFileEntry> list_files(const std::string& ns, const std::string& tbl);
    std::string          resolve_path(const std::string& ns, const std::string& tbl,
                                      const std::string& rel_path);
};
```

### `TableInfo`

```cpp
struct TableInfo {
    std::string vector_dim;
    std::string vector_metric;
    std::string vector_precision;
    std::string embedding_model; // from ailake.embedding-model property; empty if not set
};
```

### `ExtraVectorIndex`

```cpp
struct ExtraVectorIndex {
    std::string column;
    uint32_t    dim          = 0;
    uint64_t    hnsw_offset  = 0; // absolute byte offset of AILK header in file
    uint64_t    hnsw_len     = 0;
    std::string centroid_b64; // base64 F32 centroid (may be empty)
    float       radius       = 0.f;
};
```

### `DataFileEntry`

```cpp
struct DataFileEntry {
    std::string path;
    uint64_t    record_count   = 0;
    uint64_t    file_size_bytes= 0;
    std::vector<float> centroid;
    float       radius         = 0.f;
    std::optional<uint64_t> hnsw_offset;
    std::optional<uint64_t> hnsw_len;
    std::string vector_column;
    uint32_t    vector_dim     = 0;
    std::vector<ExtraVectorIndex> extra_vector_indexes; // secondary columns (Phase 8)
    std::string index_status;   // "ready" | "indexing" | "failed"
    std::string index_error;    // non-empty only when index_status == "failed"
    std::string batch_id;
    std::string embedding_model; // from per-file key_metadata JSON; empty if not set
};
```

`extra_vector_indexes` is populated from the `extra_vector_indexes` JSON array in Avro `key_metadata`; used by `search_multimodal` to locate secondary column HNSW indexes.

### Dim validation in `search()`

`search()` validates `dim` against `TableInfo.vector_dim` before any I/O. On mismatch it throws `std::runtime_error` naming the stored model:

```
ailake: query dim=512 does not match table dim=1536 (table model: text-embedding-3-small@v1)
```

## Cross-modal search (Phase 8)

Tables with N vector columns can be searched simultaneously via Reciprocal Rank Fusion:

```cpp
#include <ailake/ailake.hpp>

ailake::HadoopCatalog catalog("/data/warehouse");

std::vector<ailake::ModalQuery> queries = {
    {"embedding",       text_vec,  0.7f},
    {"image_embedding", image_vec, 0.3f},
};

auto results = ailake::search_multimodal(catalog, "default", "media", queries);
// results: std::vector<MultimodalResult>{row_id, rrf_score, file_path}
// sorted descending by rrf_score = Σ weight_i / (60 + rank_i)
```

### `ModalQuery`

```cpp
struct ModalQuery {
    std::string        column; // vector column name; empty → primary column
    std::vector<float> query;  // query embedding
    float              weight = 1.0f;
};
```

### `MultimodalResult`

```cpp
struct MultimodalResult {
    uint64_t    row_id;
    float       rrf_score; // higher = more relevant
    std::string file_path;
};
```

### `search_multimodal`

```cpp
std::vector<MultimodalResult>
search_multimodal(HadoopCatalog& catalog,
                  const std::string& ns,
                  const std::string& table,
                  const std::vector<ModalQuery>& queries,
                  const SearchOptions& opts = {});
```

Uses geometric pruning on the primary column centroid, dispatches HNSW search per column (using `DataFileEntry::extra_vector_indexes` for secondary columns), then fuses ranked lists with RRF.

## Write operations

The C++ header-only SDK delegates write operations (write_batch, delete_where, evolve_schema) to the `ailake` CLI binary via subprocess. No Rust FFI required at the C++ layer.

### `ailake::delete_where`

```cpp
#include <ailake/write.hpp>

// Commit an Iceberg equality delete (no data files rewritten)
ailake::delete_where(
    "/path/to/warehouse",  // warehouse root
    "default",             // namespace
    "my_table",            // table name
    "id",                  // equality delete column
    {"doc-1", "doc-2"}    // values to delete
);
// throws std::runtime_error on failure
```

### `ailake::evolve_schema`

```cpp
#include <ailake/write.hpp>

// Metadata-only schema evolution (no data files rewritten; field IDs are stable)
ailake::evolve_schema(
    "/path/to/warehouse",
    "default",
    "my_table",
    {{"source_url", "string", false, ""}},  // add_columns: {name, type, required, initial_default}
    {}                                        // rename_columns: {} empty = no renames
);
```

Both functions invoke the `ailake` binary via `resolve_bin()` (respects `AILAKE_BIN` env var) and parse the JSON response. An empty `values` list in `delete_where` is a no-op.

### `ailake::search_text`

```cpp
#include <ailake/ailake.hpp>   // included via ailake.hpp umbrella

// Full-text search (Tantivy O(log N) when FTS index present; BM25 brute-force fallback)
std::vector<ailake::FtsResult> hits = ailake::search_text(
    catalog,                        // HadoopCatalog
    "default",                      // namespace
    "my_table",                     // table
    "rust programming async",       // query text
    {"chunk_text", "document_title"}, // text columns (default: ["chunk_text"])
    20                              // top_k (default: 10)
);
// FtsResult: { int64_t row_id; double score; std::string file_path; }
// score is BM25 (higher = more relevant)
```

Binary resolution same as `delete_where` / `evolve_schema` — throws `std::runtime_error` when no binary is found.

## Low-level index access

### HNSW

```cpp
#include <ailake/hnsw.hpp>

std::vector<uint8_t> blob = /* read AILK section from file */;
auto idx  = ailake::deserialize_hnsw(blob.data(), blob.size());
auto hits = ailake::hnsw_search(idx, query.data(), top_k, ef_search);
// hits: std::vector<SearchResult>{row_id, distance}
```

### IVF-PQ

```cpp
#include <ailake/ivfpq.hpp>

auto idx  = ailake::deserialize_ivfpq(blob.data(), blob.size());
auto hits = ailake::ivfpq_search(idx, query.data(), top_k);
```

`deserialize_ivfpq` reads the optional trailing byte for the residual flag. When `idx.config.residual = true`, `ivfpq_search` uses a per-cluster ADC table automatically — no caller change needed.

## GPU support

### NVIDIA CUDA (`-DAILAKE_CUDA=ON`)

Enables GPU IVF-PQ search (ADC kernels) and GPU flat-scan when `detect_hardware().has_cuda` is true. Requires CUDA Toolkit 11.0+ at **build** time.

```bash
cmake -B build -DAILAKE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build
```

### AMD ROCm (runtime, no SDK needed)

When `detect_hardware().has_rocm` is true, flat-scan delegates to `hipBLAS` SGEMM via runtime `dlopen`. No ROCm SDK is required at build time — graceful CPU fallback when `libhipblas.so` is absent.

> **License note**: CUDA Toolkit and ROCm are third-party proprietary software. They are loaded only when explicitly enabled. Binary distributions of this SDK must not bundle NVIDIA or AMD proprietary libraries.

## Run the example

```bash
cmake -B build && cmake --build build
./build/ailake_search -w /data/warehouse -t default.docs -d 1536 -k 10
```

## License

MIT OR Apache-2.0 — same as the rest of the AI-Lake SDK.
