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
    int   top_k             = 10;
    int   ef_search         = 0;     // 0 → top_k * 5
    float pruning_threshold = 0.8f;
    bool  use_flat_fallback = true;  // flat scan when index not yet built
    const HardwareProfile* hw = nullptr; // nullptr = auto-detect
};
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

### `DataFileEntry`

```cpp
struct DataFileEntry {
    std::string path;
    int64_t     file_size_bytes;
    int64_t     record_count;
    std::vector<float> centroid;
    float       radius;
    int64_t     footer_offset;
    std::string embedding_model; // from per-file key_metadata JSON; empty if not set
};
```

### Dim validation in `search()`

`search()` validates `dim` against `TableInfo.vector_dim` before any I/O. On mismatch it throws `std::runtime_error` naming the stored model:

```
ailake: query dim=512 does not match table dim=1536 (table model: text-embedding-3-small@v1)
```

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
