# C++ Integration Guide

`ailake-cpp` is a C++17 header-only SDK for AI-Lake tables. Vector search and
catalog reads are implemented entirely in C++ (no cgo, no proprietary SDK
required for CPU path). Write operations, FTS search, and schema evolution
delegate to the `ailake` CLI binary via `popen`.

GPU acceleration is opt-in at build time (`-DAILAKE_CUDA=ON` for NVIDIA);
AMD ROCm is probed at runtime via `dlopen` without any build-time SDK.

---

## 1. Requirements

| Requirement | Minimum |
|---|---|
| C++ standard | C++17 |
| CMake | 3.20 |
| Compiler | GCC 9+, Clang 10+, MSVC 2019+ |
| CUDA (optional) | CUDA Toolkit 11.0+ (`-DAILAKE_CUDA=ON`) |
| `ailake` CLI | Required only for write / delete / FTS / schema ops |

---

## 2. Building and linking

### 2A — FetchContent (recommended)

```cmake
include(FetchContent)
FetchContent_Declare(
    ailake_cpp
    GIT_REPOSITORY https://github.com/ThiagoLange/ai-lakehouse.git
    GIT_TAG        main
    SOURCE_SUBDIR  ailake-cpp
)
FetchContent_MakeAvailable(ailake_cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ailake ailake_catalog)
```

### 2B — Local checkout

```bash
cd ailake-cpp
cmake -B build
cmake --build build

# With NVIDIA CUDA support
cmake -B build -DAILAKE_CUDA=ON
cmake --build build
```

Link in your project:

```cmake
find_package(ailake REQUIRED PATHS /path/to/ailake-cpp/build)
target_link_libraries(my_app PRIVATE ailake::ailake ailake::ailake_catalog)
```

### 2C — Header-only (single include, no catalog)

For code that only needs HNSW/IVF-PQ deserialization and distance functions
(e.g. reading a single known file), you can include just the headers:

```cpp
// No CMake target needed — just add ailake-cpp/include to include paths.
#include <ailake/footer.hpp>
#include <ailake/hnsw.hpp>
#include <ailake/distance.hpp>
```

---

## 3. CMake options

| Option | Default | Description |
|---|---|---|
| `AILAKE_CUDA` | `OFF` | Enable NVIDIA CUDA GPU acceleration |
| `AILAKE_EXAMPLES` | `ON` | Build `examples/search.cpp` |
| `AILAKE_TESTS` | `ON` | Build unit tests |

SIMD (`-march=native` / `/arch:AVX2`) is enabled automatically on x86_64.

---

## 4. Catalog — connecting to a table

```cpp
#include <ailake/ailake.hpp>

ailake::HadoopCatalog catalog("/data/warehouse");
// S3-mounted path (read metadata from local mount):
// ailake::HadoopCatalog catalog("/mnt/s3/warehouse");
```

**Inspect table metadata:**

```cpp
auto info = catalog.load_table("default", "docs");
std::cout << "table:   " << info.table         << "\n"
          << "dim:     " << info.vector_dim     << "\n"
          << "metric:  " << info.vector_metric  << "\n"
          << "files:   " << info.files          << " ("
                         << info.indexed_files  << " indexed)\n"
          << "rows:    " << info.rows           << "\n"
          << "format:  v" << info.format_version << "\n";

if (info.snapshot_id)
    std::cout << "snapshot: " << *info.snapshot_id << "\n";
```

---

## 5. Vector search

`search()` is implemented in pure C++ — no CLI required.

```cpp
#include <ailake/ailake.hpp>

std::vector<float> query(1536);  // your embedding

ailake::SearchOptions opts;
opts.top_k             = 10;
opts.pruning_threshold = 0.8f;  // geometric pruning aggressiveness (0-1)
opts.ef_search         = 50;    // HNSW ef_search; 0 → top_k * 5

ailake::HadoopCatalog catalog("/data/warehouse");
auto results = ailake::search(catalog, "default", "docs",
                              query.data(), query.size(), opts);

for (auto& r : results)
    std::printf("row_id=%-8llu  distance=%.6f  %s\n",
                (unsigned long long)r.row_id, r.distance, r.file_path.c_str());
```

**All `SearchOptions` fields:**

```cpp
ailake::SearchOptions opts;
opts.top_k             = 10;
opts.ef_search         = 0;     // 0 → top_k*5 (HNSW only)
opts.pruning_threshold = 0.8f;  // files further than this from query pruned
opts.use_flat_fallback = true;  // flat scan when HNSW graph empty (new file)
opts.partition_filter  = "agent-001"; // restrict to one partition (Phase 9)
opts.hw                = nullptr;     // nullptr → auto-detect hardware
```

**How search works:**

1. `load_table` reads `metadata.json` → finds vector metric and current snapshot.
2. `list_files` reads Avro manifests → gets per-file `centroid` + `radius`.
3. Geometric pruning: skips files where `distance(query, centroid) - radius > threshold`.
4. Surviving files: AILK section read at byte offset, HNSW/IVF-PQ deserialized.
5. GPU dispatch (when enabled) for flat scan and IVF-PQ.
6. Results merged, top-K returned.

**Partition-filtered search (agent memory):**

```cpp
opts.partition_filter = "agent-001";
auto results = ailake::search(catalog, "default", "agent_memory",
                              query.data(), query.size(), opts);
```

---

## 6. Full-text search (FTS)

Requires `ailake` CLI. Uses Tantivy O(log N) when FTS index present; BM25
brute-force fallback for legacy files.

```cpp
auto hits = ailake::search_text(
    catalog,
    "default", "docs",
    "machine learning embeddings",   // query string
    {"chunk_text"},                  // columns to search
    10                               // top-K
);
for (auto& h : hits)
    std::printf("row_id=%-8lld  score=%.4f  %s\n",
                (long long)h.row_id, h.score, h.file_path.c_str());
```

---

## 7. Multimodal / multi-column search

Cross-modal RRF fusion over multiple vector columns — pure C++, no CLI.

```cpp
std::vector<float> text_query (1536);  // text embedding
std::vector<float> image_query(512);   // image embedding (e.g. CLIP)

auto results = ailake::search_multimodal(
    catalog, "default", "media",
    {
        {"text_embedding",  text_query,  0.7f},
        {"image_embedding", image_query, 0.3f},
    },
    opts  // same SearchOptions
);
for (auto& r : results)
    std::printf("row_id=%-8llu  rrf=%.4f  %s\n",
                (unsigned long long)r.row_id, r.rrf_score, r.file_path.c_str());
```

RRF formula: `score_i = weight_i / (60 + rank_i)`,  `final = Σ score_i`.

---

## 8. Writing data

`write_batch` wraps `ailake insert`. Pass a local Parquet file containing the
embedding column.

```cpp
#include <ailake/write.hpp>

ailake::WriteBatchOptions opts;
opts.vec_col   = "embedding";
opts.metric    = "cosine";
opts.precision = "f16";

ailake::write_batch(
    "/data/warehouse",       // warehouse root
    "default.docs",          // "namespace.table"
    "/tmp/batch.parquet",    // local Parquet file with embedding column
    opts
);
```

**All `WriteBatchOptions` fields:**

```cpp
ailake::WriteBatchOptions opts;
opts.vec_col              = "embedding";     // embedding column name
opts.metric               = "cosine";        // cosine | euclidean | dot
opts.precision            = "f16";           // f32 | f16 | i8
opts.embedding_model      = "text-embedding-3-small";
opts.partition_by         = "agent_id";      // single partition column
opts.partition_value      = "agent-001";     // partition value for this batch
opts.format_version       = 2;              // Iceberg format version (2 or 3)
opts.fts_columns          = {"chunk_text"}; // build Tantivy FTS index
opts.fts_tokenizer        = "default";
opts.hnsw_m               = 16;             // HNSW M (0 = table default)
opts.hnsw_ef_construction = 200;            // HNSW ef_construction (0 = default)
opts.pre_normalize        = false;          // normalize to unit L2 at write
opts.deferred             = false;          // build index async (Parquet committed now)
```

**Producing a Parquet file from C++:**

Use [Apache Arrow C++](https://arrow.apache.org/docs/cpp/) or any compliant
Parquet writer. Example with Arrow:

```cpp
#include <arrow/api.h>
#include <parquet/arrow/writer.h>

auto schema = arrow::schema({
    arrow::field("chunk_id",   arrow::utf8()),
    arrow::field("chunk_text", arrow::utf8()),
    arrow::field("embedding",  arrow::list(arrow::float32())),
});

// Build arrays, create RecordBatch, write to /tmp/batch.parquet ...
auto outfile = *arrow::io::FileOutputStream::Open("/tmp/batch.parquet");
PARQUET_THROW_NOT_OK(
    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                               outfile, /*chunk_size=*/65536));
```

Then call `ailake::write_batch("/data/warehouse", "default.docs", "/tmp/batch.parquet", opts)`.

**Multi-column (multimodal) writes:**

`write_batch_multi` wraps `ailake insert --vector-cols` — each column gets its
own independent HNSW index in the AILK footer (Phase 8 multimodal). Requires
at least one `VectorColSpec`.

```cpp
ailake::write_batch_multi(
    "/data/warehouse",
    "default.media",
    "/tmp/batch.parquet",
    {
        {"embedding", 1536, "cosine", ""},          // text column
        {"image_embedding", 512, "cosine", "image"}, // image column
    }
);
```

Multi-column mode hardcodes F16 precision and carries metric per-column via
`VectorColSpec::metric` — `WriteBatchOptions::metric`/`precision` don't apply
in this mode (same contract as `ailake insert --vector-cols`). An optional
trailing `WriteBatchOptions` argument still applies for
`partition_by`/`partition_value`/`format_version`/`fts_columns`/`deferred`.

---

## 9. Deletes and schema evolution

Both delegate to the `ailake` CLI.

**Logical delete (Iceberg equality delete — no data rewrite):**

```cpp
#include <ailake/write.hpp>

ailake::delete_where(
    "/data/warehouse",      // warehouse root
    "default.docs",         // "namespace.table"
    "chunk_id",             // match column
    {"uuid-aaa", "uuid-bbb"} // values to delete
);
```

**Schema evolution — add / rename columns:**

```cpp
int new_schema_id = ailake::evolve_schema(
    "/data/warehouse",
    "default.docs",
    {   // add columns
        {"language",    "string", R"("en")"},
        {"page_number", "int",    "null"},
    },
    {   // rename columns
        {"chunk_text", "text"},
    }
);
std::cout << "new schema_id: " << new_schema_id << "\n";
```

---

## 10. GPU acceleration

### NVIDIA CUDA

Build with `-DAILAKE_CUDA=ON`:

```bash
cmake -B build -DAILAKE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build
```

In code, define `AILAKE_CUDA_ENABLED` before the include (or let CMake do it):

```cpp
#define AILAKE_CUDA_ENABLED
#include <ailake/ailake.hpp>
```

When CUDA is enabled:
- **HNSW flat fallback**: GPU batch cosine scan via `cuda::GpuSearchContext`.
- **IVF-PQ**: pre-built ADC lookup table, GPU distance accumulation via `cuda::GpuIvfPqContext`.
- **HNSW graph traversal**: always CPU (sequential graph nature).

### AMD ROCm

No build-time flag needed. ROCm (hipBLAS) is probed at runtime via `dlopen`:

```cpp
auto hw = ailake::detect_hardware();
if (hw.has_rocm)
    std::cout << "ROCm available — flat scan via hipBLAS SGEMM\n";
```

Falls back to CPU if hipBLAS is unavailable. No ROCm SDK required at compile time.

### Force CPU-only

```cpp
ailake::HardwareProfile cpu_only;
cpu_only.has_cuda = false;
cpu_only.has_rocm = false;

ailake::SearchOptions opts;
opts.hw      = &cpu_only;
opts.top_k   = 10;
auto results = ailake::search(catalog, "default", "docs",
                              query.data(), query.size(), opts);
```

---

## 11. Binary resolution for CLI operations

Write / delete / FTS / schema operations call `popen("ailake ...")`. Resolution
order:

1. `AILAKE_BIN` environment variable — exact path to binary.
2. `ailake` found on `PATH`.

```bash
export AILAKE_BIN=/opt/ailake/bin/ailake
```

All CLI functions throw `std::runtime_error` when the binary is not found or
exits non-zero.

---

## 12. Error handling

```cpp
try {
    auto results = ailake::search(catalog, "default", "docs",
                                  query.data(), query.size(), opts);
} catch (const std::runtime_error& e) {
    // Common messages:
    // "ailake: cannot open /data/warehouse/default/docs/data/part-001.parquet"
    // "ailake: query dim=512 does not match table dim=1536 (table model: text-embedding-3-small)"
    // "ailake CLI failed (exit 1):\nError: table not found"
    std::cerr << "ailake error: " << e.what() << "\n";
}
```

Files still being indexed (`index_status == "indexing"`) are skipped
silently — `search_file` returns an empty result vector when
`hnsw_offset` / `hnsw_len` are not set.

---

## 13. Full example — RAG pipeline

```cpp
// main.cpp — minimal RAG: embed query → hybrid search → print results
// Build: cmake -B build && cmake --build build
// Run:   ./build/ailake_search -w /data/warehouse -t default.docs -d 1536 -k 10

#include <ailake/ailake.hpp>
#include <ailake/write.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Replace with your embedding provider (OpenAI, Cohere, local model, etc.)
std::vector<float> embed(const std::string& /*text*/, int dim) {
    std::vector<float> v(dim);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    const std::string warehouse = "/data/warehouse";
    const std::string ns        = "default";
    const std::string tbl       = "docs";
    const int         dim       = 1536;

    ailake::HadoopCatalog catalog(warehouse);

    // 1. Inspect table
    auto info = catalog.load_table(ns, tbl);
    std::cout << "table: " << info.table
              << "  dim=" << info.vector_dim
              << "  rows=" << info.rows << "\n\n";

    // 2. Embed query
    std::string query_text = "What is geometric pruning in AI-Lake?";
    auto query = embed(query_text, dim);

    // 3. Vector search (pure C++ — no CLI)
    ailake::SearchOptions opts;
    opts.top_k             = 10;
    opts.pruning_threshold = 0.8f;

    auto results = ailake::search(catalog, ns, tbl,
                                  query.data(), query.size(), opts);

    std::printf("%-6s %-12s %s\n", "rank", "distance", "file");
    for (size_t i = 0; i < results.size(); ++i)
        std::printf("%-6zu %-12.6f %s (row_id=%llu)\n",
                    i + 1, results[i].distance, results[i].file_path.c_str(),
                    (unsigned long long)results[i].row_id);

    // 4. FTS search (requires ailake CLI)
    try {
        auto fts = ailake::search_text(catalog, ns, tbl, query_text, {"chunk_text"}, 5);
        std::puts("\nFTS results:");
        for (auto& h : fts)
            std::printf("  row_id=%-8lld  score=%.4f  %s\n",
                        (long long)h.row_id, h.score, h.file_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "FTS skipped: " << e.what() << "\n";
    }

    return 0;
}
```

**CMakeLists.txt for this example:**

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_rag CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    ailake_cpp
    GIT_REPOSITORY https://github.com/ThiagoLange/ai-lakehouse.git
    GIT_TAG        main
    SOURCE_SUBDIR  ailake-cpp
)
FetchContent_MakeAvailable(ailake_cpp)

add_executable(my_rag main.cpp)
target_link_libraries(my_rag PRIVATE ailake ailake_catalog)

# Uncomment for CUDA:
# set_target_properties(ailake PROPERTIES INTERFACE_COMPILE_DEFINITIONS AILAKE_CUDA_ENABLED)
```

---

## 14. API surface reference

| Symbol | Header | Needs CLI | Description |
|---|---|---|---|
| `HadoopCatalog` | `catalog.hpp` | No | Catalog pointing to local/S3 warehouse |
| `catalog.load_table(ns, tbl)` | `catalog.hpp` | No | Returns `TableInfo` (dim, metric, files, rows, schema) |
| `catalog.list_files(ns, tbl)` | `catalog.hpp` | No | Returns `std::vector<DataFileEntry>` |
| `catalog.resolve_path(ns, tbl, rel)` | `catalog.hpp` | No | Resolve relative data file path to absolute |
| `catalog.warehouse()` | `catalog.hpp` | No | Returns warehouse root string |
| `search(catalog, ns, tbl, query, dim, opts)` | `ailake.hpp` | No | Geometric pruning + HNSW/IVF-PQ search |
| `search_multimodal(catalog, ns, tbl, queries, opts)` | `ailake.hpp` | No | Cross-modal RRF fusion |
| `search_text(catalog, ns, tbl, text, cols, k)` | `ailake.hpp` | **Yes** | FTS (Tantivy or BM25 fallback) |
| `write_batch(warehouse, table_id, parquet, opts)` | `write.hpp` | **Yes** | Ingest Parquet batch + build HNSW |
| `write_batch_multi(warehouse, table_id, parquet, cols, opts)` | `write.hpp` | **Yes** | Multi-column (multimodal) ingest |
| `delete_where(warehouse, table_id, col, vals)` | `write.hpp` | **Yes** | Iceberg equality delete |
| `evolve_schema(warehouse, table_id, add, rename)` | `write.hpp` | **Yes** | Add/rename columns (metadata-only) |
| `compact(warehouse, table_id, opts)` | `write.hpp` | **Yes** | Merge small files, returns files compacted |
| `detect_hardware()` | `hardware.hpp` | No | Returns `HardwareProfile` (CUDA/ROCm/SIMD flags) |
| `parse_header(bytes)` | `footer.hpp` | No | Parse AILK header from 64-byte buffer |
| `deserialize_hnsw(data, len)` | `hnsw.hpp` | No | Deserialize HNSW graph from bincode bytes |
| `deserialize_ivfpq(data, len)` | `ivfpq.hpp` | No | Deserialize IVF-PQ index |
| `compute_distance(metric, a, b, dim)` | `distance.hpp` | No | Single vector distance (cosine/euclidean/dot) |

**Key structs:**

| Struct | Description |
|---|---|
| `SearchOptions` | `top_k`, `ef_search`, `pruning_threshold`, `use_flat_fallback`, `partition_filter`, `hw` |
| `WriteBatchOptions` | All write parameters (see §8) |
| `ModalQuery` | `{column, query, weight}` for multimodal search |
| `FileSearchResult` | `{row_id, distance, file_path}` |
| `MultimodalResult` | `{row_id, rrf_score, file_path}` |
| `FtsResult` | `{row_id, score, file_path}` |
| `TableInfo` | Full metadata: dim, metric, files, rows, `schema_fields`, `partition_fields` |
| `DataFileEntry` | Per-file: `centroid`, `radius`, `hnsw_offset`, `hnsw_len`, `index_status` |
| `AddColumnReq` | `{name, type, initial_default}` for schema evolution |
| `RenameColumnReq` | `{from, to}` for schema evolution |
| `VectorColSpec` | `{column, dim, metric, modality}` for `write_batch_multi` |
| `CompactOptions` | `target_size`, `min_files`, `max_files_per_pass`, `deferred` |
| `HardwareProfile` | `has_cuda`, `has_rocm`, `has_avx2`, `has_avx512` |

---

## Related docs

- [File Format Spec](../specs/FILE_FORMAT.md) — AILK section layout
- [GPU FFI Evaluation](../specs/GPU_FFI_EVALUATION.md) — CUDA/ROCm strategy
- [JVM Plugins](../specs/JVM_PLUGINS.md) — Spark / Trino / Flink (C-ABI + JNA)
- [Go Integration](GO_INTEGRATION.md) — pure-Go client (zero cgo)
- [ailake-cpp source](../../ailake-cpp/) — headers, examples, tests
