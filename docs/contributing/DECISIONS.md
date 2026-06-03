# DECISIONS.md — Architecture Decision Records (ADR)

Decisions are numbered and immutable once merged. To change a decision, add a new ADR that supersedes the old one.

---

## ADR-001: Rust as the sole implementation language

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The core operations — reading/writing files, HNSW index construction, quantization, centroid computation — are I/O and CPU-bound at petabyte scale. Language choice directly impacts throughput and operational cost.

**Decision**: Rust for all `ailake-*` crates. Python bindings via PyO3; JVM bindings via JNA + C-ABI (`ailake-jni` exports `#[no_mangle]` functions). No business logic in binding layers.

**Consequences**:
- Zero GC pauses during index traversal (critical for p99 latency).
- `cargo build --target` enables cross-compilation without complex toolchains.
- Binding maintenance cost: PyO3 (Python) and JNA (JVM) are mature. C-ABI surface is minimal — 4 exported functions.
- Contributors need Rust experience. Offset by clear crate boundaries and thorough docs.

**Rejected alternatives**:
- Go: simpler concurrency model, but no zero-cost abstractions and GC pauses at the wrong moments.
- C++: performance equivalent to Rust, but memory safety requires discipline rather than enforcement.
- Python: 10-100× slower for hot paths; GIL limits parallelism.

---

## ADR-002: Apache Iceberg Spec v2 as the catalog layer

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The project needs a catalog that major data platforms (Spark, Trino, Athena, Snowflake) can read without modification. Building a custom catalog would require plugin development for each platform.

**Decision**: Strict conformance to Iceberg Spec v2. Vector extensions stored in `properties` (table-level) and `custom-properties` of DataFile entries (file-level statistics), both of which are spec-defined string maps ignored by unknown readers.

**Consequences**:
- Any Iceberg-compatible framework reads AI-Lake tables without modification or plugin.
- We are bound by Iceberg's immutability and snapshot model.
- Per-file centroid/radius is stored once per file, available without opening Parquet files (in the Avro manifest).

**Rejected alternatives**:
- Delta Lake: fewer framework integrations, especially outside Databricks ecosystem.
- Apache Hudi: COW/MOR complexity, weaker ecosystem coverage.
- Custom catalog: every new framework integration requires bespoke plugin work.

---

## ADR-003: `hnsw_rs` for HNSW indexing

**Date**: 2024-08  
**Status**: Accepted (supersedes earlier proposal of usearch)  

**Context**: HNSW is the standard algorithm for approximate nearest neighbor search. The main Rust-accessible implementations are Faiss (C++ via FFI), usearch (C++ with Rust bindings), and hnsw_rs (pure Rust).

**Decision**: `hnsw_rs` crate — pure Rust implementation, no C++ dependency, native Serde-compatible serialization.

**Consequences**:
- No C++ toolchain required for building or cross-compiling.
- `cargo build --target aarch64-unknown-linux-musl` works without extra setup.
- HNSW graph serializes naturally via `bincode` — no custom FFI serialization adapter needed.
- hnsw_rs has a smaller user base than Faiss. Mitigated by: (a) the unified file architecture bounds individual HNSW size to ~10-20 MB per file, well within hnsw_rs's tested range; (b) the algorithm is the same as Faiss HNSW, so recall characteristics match the published HNSW literature.
- GPU support not available (Faiss has CUDA). Addressed in Phase 4: NVIDIA via candle-core (`gpu` feature), AMD ROCm via hipBLAS libloading — see ADR-012.

**Rejected alternatives**:
- Faiss via C FFI: would require `libfaiss.so` as a runtime dependency, complicating distribution. Cross-compilation becomes a non-trivial build engineering project.
- usearch: hybrid C++/Rust, still requires C++ toolchain in some configurations.
- Building HNSW from scratch: months of work; correctness hard to validate against ANN benchmarks.

---

## ADR-004: F16 as the default vector precision

**Date**: 2024-08  
**Status**: Accepted  

**Context**: Raw F32 vectors for 100M documents with dim=1536 cost 600 GB. Storage cost is a primary adoption blocker for petabyte-scale use cases.

**Decision**: Default precision is F16. F32 is available but not the default. I8 and PQ are opt-in for extreme compression needs.

**Consequences**:
- 50% storage reduction with < 0.1% recall@10 degradation for tested models (text-embedding-3-small, nomic-embed-text).
- F16 requires `half` crate for Rust and is not natively understood by all Parquet readers. The Parquet column stores F16 bytes, which readers treat as opaque.
- hnsw_rs operates on F32 — F16 vectors are expanded to F32 when feeding the HNSW builder and at search time. The HNSW graph itself stores the F32 representation it was built with. This is acceptable: the HNSW graph is 10-20× smaller than raw vectors.

**Rejected alternatives**:
- F32 default: too expensive at scale; makes the format uncompetitive with purpose-built vector DBs.
- I8 default: ~2-3% recall degradation is acceptable for some use cases but not a safe default.
- Binary default: only viable for models trained for binary embeddings; not general-purpose.

---

## ADR-005: Single self-contained file (Parquet + AI-Lake footer)

**Date**: 2024-08  
**Status**: Accepted (supersedes earlier 3-file design)  

**Context**: Earlier design used three separate files per Parquet (`.parquet` + `.vec` + `hnsw-*.bin`). This required:
- Atomic three-way writes
- Manifest references to three file paths per logical "data file"
- Cleanup of orphaned files if any of the three failed to write
- Separate sidecar manifest (`vsnap-*.json`) outside the Iceberg catalog

The single-file design unifies dimensional data, vectors, and the HNSW index into one Parquet-extended file at the storage layer.

**Decision**: One physical file per logical data unit. Layout: `PAR1` + row groups + AI-Lake extension (AILK header + centroid + HNSW graph + trailer) + Parquet footer thrift + `footer_len` + `PAR1`. The AI-Lake extension sits between the row groups and the Parquet footer — invisible to standard readers because row-group offsets in the footer point before the AILK section, so readers jump directly to row groups and never scan the AILK bytes.

**Consequences**:
- **Source-of-truth simplicity**: one file = one snapshot of data + index. No three-way consistency to maintain.
- **Iceberg integration is cleaner**: the Iceberg DataFile entry points to the unified `.parquet` file. Vector statistics (centroid, radius, HNSW byte offsets) go in `custom-properties` of the DataFile entry — a spec-defined extension point.
- **Atomic writes**: one S3 PUT per data file. No partial-state recovery logic.
- **Compatibility is a hard Parquet guarantee**: the AILK section is between row groups and the Parquet footer. Standard readers parse the footer from the end of the file, then seek directly to row-group offsets — they never scan past row groups. No "tolerate trailing data" clause needed.
- **Compaction is conceptually simpler**: merge files = read N files, write 1 file with rebuilt HNSW. No separate index merging logic.
- **Per-file HNSW**: search must open multiple HNSW indexes (one per file) and merge. Mitigated by centroid pruning — typical search opens 50-100 indexes, not 10,000.

**Rejected alternatives** (now):
- Three separate files (`.parquet` + `.vec` + `hnsw-*.bin`): operational complexity outweighed any benefit; no clear performance win.
- Custom file format (no Parquet at all): would lose Iceberg compatibility entirely.

---

## ADR-006: `memmap2` for HNSW loading

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The HNSW graph in the AI-Lake footer is typically 10-20 MB per file. Loading hundreds of files into RAM during a search would exhaust memory. The OS already has efficient demand paging via mmap.

**Decision**: Use `memmap2::Mmap` to load the HNSW graph bytes. The bincode deserialization reads from the mmap'd slice, and HNSW traversal only touches the pages it needs.

**Consequences**:
- RAM usage scales with working set size, not total index size.
- First search on a cold cache is slower (page-in cost); subsequent searches are fast.
- The mmap'd file must outlive the HNSW reference — enforced by tying lifetimes in the `HnswIndex` struct.
- On S3, we first download the footer bytes to a temp file, then mmap. This gives us the OS-level demand paging benefit even when the source is remote.

**Rejected alternatives**:
- Load full HNSW into RAM via `Vec<u8>`: simpler but wasteful for cold indexes.
- Stream HNSW from S3 directly: hnsw_rs requires random access; streaming is not viable.
- Custom paging layer: redundant with what the OS already provides for mmap'd files.

---

## ADR-007: `bincode` for HNSW serialization

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The HNSW graph needs to be serialized into the AI-Lake footer. Options: bincode, MessagePack, FlatBuffers, custom binary format.

**Decision**: `bincode` — Rust-idiomatic, Serde-based, zero-copy where possible, fast.

**Consequences**:
- Serialization speed: ~GB/s on modern hardware.
- Tight binary format with no metadata overhead.
- Deserialization can borrow from the mmap'd byte slice for zero-copy where Serde permits.
- Binding to bincode versioning: we pin the version in `Cargo.toml` because bincode 2.0 has different defaults. Mitigated by version pin and integration test.

**Rejected alternatives**:
- MessagePack: more portable across languages but slower for tight Rust structs.
- FlatBuffers: zero-copy by design but requires schema definition; HNSW internal types make this awkward.
- Custom binary format: months of work to match bincode's performance.

---

## ADR-008: `LlmContextSchema` as canonical RAG table schema

**Date**: 2024-08  
**Status**: Accepted  

**Context**: RAG quality degrades when chunks lack context about where they come from in the source document. This is a known failure mode ("the value increased 23%" without knowing which value).

**Decision**: Define a canonical schema with structural context fields (`document_title`, `section_path`, `preceding_context`, `following_context`, `document_summary`) stored as Parquet columns. The schema is a recommendation, not enforced by the format.

**Consequences**:
- `ContextAssembler` can rely on these fields being present when assembling prompts.
- Tables that don't follow this schema can still use AI-Lake for vector search — they just don't get `ContextAssembler` support.
- Ingest pipelines must generate `document_summary` and `chunk_summary` via an LLM call at write time. This adds cost and latency to ingest. It is the right tradeoff: pay once at ingest, save on every query.

**Rejected alternatives**:
- Store context only in the prompt, not in the table: requires re-fetching source documents at query time; adds latency.
- Store context in a separate table: cross-table join at query time adds complexity; denormalized storage in Parquet compresses well enough to justify it.

---

## ADR-009: No DataFusion dependency

**Date**: 2024-08  
**Status**: Accepted  

**Context**: DataFusion is a popular Rust SQL engine that could provide a `TableProvider` integration for AI-Lake. Adding it would enable SQL queries against AI-Lake tables directly from Rust without going through Spark/Trino.

**Decision**: **Do not** add DataFusion as a dependency in any crate. The AI-Lake project is a file format and SDK, not a SQL engine.

**Consequences**:
- The SDK exposes operations directly: `write_batch`, `search`, `assemble_context`. No SQL planner overhead.
- Users who want SQL access use Spark, Trino, or DuckDB via Iceberg compatibility (these handle SQL planning themselves).
- Users who want a Rust-native SQL engine on top of AI-Lake can build their own `TableProvider` as an external crate — the AI-Lake SDK exposes everything needed.
- Smaller binary size, faster builds, fewer transitive dependencies.

**Rejected alternatives**:
- DataFusion as core: expands project scope significantly; couples format evolution to DataFusion's TableProvider API stability.
- DataFusion as optional feature flag: still pulls the dependency tree in for any user who enables it; better to keep it external.

---

## ADR-010: Dual embeddings (`embedding` + `context_embedding`)

**Date**: 2024-08  
**Status**: Accepted  

**Context**: A single embedding of raw `chunk_text` captures content but not position within the document. Queries that implicitly require context (e.g. "Q3 gross margin") may retrieve the right sentence but from the wrong document or quarter.

**Decision**: Two vector columns per LLM-context table. `embedding` is the raw chunk. `context_embedding` is the chunk prepended with document title, section path, and document summary. Each column gets its own HNSW graph in the AI-Lake footer.

**Consequences**:
- 2× embedding API calls at ingest time. For OpenAI `text-embedding-3-small`, this is approximately $0.02 per 1M tokens — acceptable for document ingest.
- 2× vector storage and 2× HNSW size in the footer. With F16 default, this is ~6 GB per 1M chunks plus ~1 GB of HNSW.
- Recall improvements are significant for position-dependent queries (~15-25% improvement in recall@5 for financial document Q&A benchmarks).
- RRF fusion of both rankings further improves recall at minimal compute cost.

**Rejected alternatives**:
- Single embedding with context prepended at query time: the embedding model would need to be called with the query + all context combinations.
- Single embedding with context prepended at ingest: loses the ability to do precise content-only search.
- Hybrid BM25 + vector: BM25 requires a separate inverted index; deferred to Phase 4.

---

## ADR-011: XML output format for `ContextAssembler`

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The assembled context must be formatted for insertion into an LLM prompt. Format choice affects how well the model can parse and attribute information.

**Decision**: XML with semantic tags (`<context>`, `<document>`, `<chunk>`, `<text>`).

**Consequences**:
- Claude (Anthropic) performs well with XML-structured context per published prompt engineering guidance.
- Other models (GPT-4, Gemini) also handle XML well — widely supported.
- Slightly more tokens than plain text due to tags. Overhead is ~5% for typical chunk sizes — acceptable.
- XML structure makes document attribution unambiguous, reducing hallucination of sources.

**Rejected alternatives**:
- Markdown: headings are less semantically explicit than XML tags.
- JSON: verbose, harder for LLMs to read as natural language context.
- Plain text with separators: loses structural information; model must infer attribution.

---

## ADR-012: Multi-vendor GPU support — NVIDIA CUDA + AMD ROCm (both runtime-only)

**Date**: 2026-05-22
**Status**: Accepted (updated 2026-05-22 — NVIDIA path migrated from candle-core to libloading)

**Context**: Phase 4 originally added NVIDIA GPU acceleration via `candle-core` (compile-time `gpu` feature) and AMD ROCm via `libloading` (runtime-only). Two problems with the asymmetric approach:
1. The build-time `gpu` feature required CUDA Toolkit installed on the build machine — distribution and CI became complex.
2. Binary bloat: `candle-core` + its dependency tree added ~2-3 MB to every build even without CUDA hardware.

**Decision**: Both GPU backends use `libloading` dlopen with no compile-time GPU SDK required:

1. **NVIDIA CUDA** — `libloading` dlopen of `libcudart.so` (tries `.so`, `.so.12`, `.so.11`) + `libcublas.so`. `cublasSgemm_v2` via function pointer. RAII guards (`DevBuf` / `BlasHandle`) identical in structure to ROCm. Returns `None` → CPU fallback if libraries absent. `candle-core` removed from workspace. `gpu` feature flag removed.

2. **AMD ROCm** — `libloading` dlopen of `libamdhip64.so` + `libhipblas.so`. `hipblasSgemm` via function pointer. Same RAII pattern. Returns `None` → CPU fallback if libraries absent.

Both backends share identical SGEMM formulation: `C[N×Q col-major] = alpha · db^T · queries`. Constants differ: `CUBLAS_OP_N=0 / CUBLAS_OP_T=1` vs `HIPBLAS_OP_N=111 / HIPBLAS_OP_T=112`.

Detection priority: AMD ROCm first, then NVIDIA CUDA, then CPU. AMD is checked first because ROCm installations often provide a CUDA compatibility layer (`libcuda.so.1`) that would misidentify the backend without the priority check.

`HardwareBackend` enum (`CpuSimd` / `NvidiaCuda` / `AmdRocm`) — single `OnceLock` caches the result for the process lifetime. `HardwareProfile` struct exposes `has_cuda`, `has_rocm`, `backend`, `cpu_logical_cores`, SIMD flags.

**Consequences**:
- Single binary for all deployments (CPU-only, NVIDIA, AMD) — zero build flag difference.
- No CUDA Toolkit, `nvcc`, or GPU headers required at build time; only runtime libraries (`libcudart.so`, `libcublas.so`) needed on the deployment machine.
- `kmeans_dispatch` in `ivf_pq.rs` follows priority: NVIDIA → ROCm → CPU rayon.
- `SearchSession::search_batch()` follows priority: NVIDIA → AMD → CPU per shard.
- Adaptive index selection (`IndexType::Auto`) uses `has_cuda || has_rocm` as the GPU-capable check; both vendors justify IVF-PQ over HNSW.
- Binary size: 13 MB → 9.3 MB (ailake-bench) after removing candle-core + adding strip/panic=abort.
- Each backend is ~220 lines of `unsafe` libloading code in its own module (`nvidia_impl` / `rocm_impl`) with RAII cleanup — no unsafe surface exposed to callers.

**Rejected alternatives**:
- NVIDIA via `candle-core`: build-time CUDA Toolkit dependency; binary size overhead; `gpu` feature creates two distinct binaries; CI cannot test GPU path without GPU runner. Superseded by libloading approach.
- AMD via `candle-core` ROCm feature: `candle-core/rocm` is not a stable feature; requires build-time dependency.
- AMD via cuVS FFI: cuVS is NVIDIA-only by design; rejected.
- HIP CUDA compatibility layer reliance: AMD ROCm ships `libcuda.so.1` as a compat shim, but relying on it would break vendor identification and could silently use the wrong code path.

---

## ADR-013: IVF-PQ shared codebook across shards

**Date**: 2026-06  
**Status**: Accepted

**Context**: The original multi-shard IVF-PQ design trained an independent PQ codebook per shard (100k vectors each). Each codebook produces ADC distances on a different numerical scale. When search results from 10 shards are merged globally (sorted by distance), the comparison is between values from 10 incomparable scales — systematically biasing the merge toward shards with lower ADC values regardless of true distance. `Recall@10 = 0.32` on SIFT-1M despite correct `nlist/nprobe` parameters.

**Decision**: Train the codebook once on the first shard and reuse it across all shards via `Arc<tokio::sync::OnceCell<IvfPqCodebook>>`. All shards built from the same codebook produce ADC distances that are numerically comparable — cross-shard merge by distance is correct.

New public API:
- `IvfPqCodebook` struct (coarse centroids + PQ)
- `IvfPqIndex::train_codebook(vectors, metric, config) -> IvfPqCodebook`
- `IvfPqIndex::build_with_codebook(row_ids, vectors, codebook) -> IvfPqIndex`
- `AilakeFileWriter::with_shared_ivf_codebook(Arc<IvfPqCodebook>)`

**Consequences**:
- `Recall@10` with `nprobe=nlist/4` and `rerank_factor=3`: 0.32 → 0.91.
- k-means training runs once (first shard) instead of N times. Write speedup: ~4× for inline, ~30× when combined with deferred build.
- Codebook is trained on 100k vectors (first shard). For datasets where shard 0 has an atypical distribution, recall may be slightly lower than a globally-trained codebook. Mitigated by the IVF assignment being robust to moderate distributional drift.

**Rejected alternatives**:
- Global codebook from all shards combined: requires two-pass write (first scan all shards for training, then write with built indexes). Breaks the streaming single-pass write model.
- Per-shard codebook with exact reranking only: still works (reranking corrects the merge), but adds memory overhead for raw vectors and latency for exact distance computation.

---

## ADR-014: Deferred IVF-PQ index build

**Date**: 2026-06  
**Status**: Accepted

**Context**: IVF-PQ k-means training is the bottleneck for write throughput (~7k vec/s inline vs ~200k vec/s for Parquet-only writes). HNSW already had a deferred build path (`write_batch_deferred`) that writes Parquet immediately and builds the HNSW in a background tokio task. The same pattern can apply to IVF-PQ.

**Decision**: `write_batch_ivf_pq_deferred` writes Parquet-only first (same fast path as HNSW deferred), then spawns a background task that:
1. Gets or trains the shared codebook via `Arc<tokio::sync::OnceCell<IvfPqCodebook>>` — first task trains, all others await and reuse.
2. Calls `IvfPqIndex::build_with_codebook` (O(n) assign+encode, no k-means for shards 2-N).
3. Rewrites the file with the AILK section.
4. Transitions `IndexStatus::Indexing → Ready` via the same CAS retry loop used by HNSW deferred.

**Consequences**:
- Write throughput: ~200k vec/s (limited by Parquet writes) vs ~7k vec/s inline.
- Index build time: 42.7s for 1M vectors in background vs 130s inline blocking write.
- Search during the build window: `SearchSession` serves `Indexing` shards via flat scan (exact brute-force). Acceptable for most workloads — HNSW deferred has the same behavior.
- `OnceCell` ensures exactly one k-means training run regardless of how many shards are written concurrently.

**Rejected alternatives**:
- Global k-means before any shard writes: breaks streaming ingestion model; requires knowing all vectors upfront.
- Train on each shard independently (no sharing): works but no write speedup benefit, and cross-shard ADC distances remain incomparable.
