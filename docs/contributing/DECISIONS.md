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

---

## ADR-015: Residual PQ — encode residuals, not raw vectors

**Date**: 2026-06  
**Status**: Accepted

**Context**: Standard IVF-PQ encodes raw vectors. The PQ codebook is trained on the distribution of all vectors across all clusters. For each cluster, the intra-cluster variance is smaller than the global variance — a codebook trained on all vectors wastes representational capacity encoding inter-cluster distances that IVF already captures (the hard assignment to the nearest cluster centroid).

**Decision**: When `ivf_residual = true`, encode `residual = vec - coarse_centroid` instead of `vec`. The PQ codebook is trained on residuals, not raw vectors — it focuses all M sub-codebooks on intra-cluster variance.

Implementation notes:
- On-disk: single trailing byte after the bincode payload (`0x01` = residual, `0x00` / absent = standard). Compatible with existing files (absence defaults to `false`).
- ADC at search time: subtract `cluster_centroid` from query before building the per-cluster LUT. Go, C++, and Rust all have per-cluster LUT logic gated on the residual flag.
- Bincode v1 positional serialization excludes `residual` from the struct (would break existing files); trailing byte is the portable extension point.

**Consequences**:
- ~2-4 pp recall@10 improvement at identical code size (M bytes/vector) and storage cost.
- Encoding/search overhead: one vector subtraction per ADC table build — negligible.
- Bincode backward compat maintained: old files (no trailing byte) still decode correctly.
- All bindings (Rust, Python, Go, C++) automatically use per-cluster LUT when the flag is detected at deserialization time; caller change is not required.

**Rejected alternatives**:
- Retrain codebook on residuals without changing the on-disk format: would produce wrong ADC distances when old files (non-residual) and new files (residual) coexist in the same table — distance scales incomparable.
- Field in the bincode struct: breaks bincode v1 positional deserialization for files without the field.

---

## ADR-016: `write_batch_auto_deferred` as the default high-throughput write path

**Date**: 2026-06  
**Status**: Accepted

**Context**: `write_batch` (HNSW inline) achieves ~6-10k vec/s. `write_batch_deferred` and `write_batch_ivf_pq_deferred` achieve ~200k vec/s each, but require callers to choose the index type explicitly. Most callers want "best index for this hardware" without specifying IVF-PQ vs HNSW.

**Decision**: `write_batch_auto_deferred` combines hardware detection with deferred index build:
1. Write Parquet immediately (same fast path as all deferred variants — ~200k vec/s).
2. Detect hardware: CUDA GPU / AMD ROCm / ≥8 CPU cores + batch ≥5k vectors → IVF-PQ deferred; else → HNSW deferred.
3. Spawn background Tokio task for index build.

Exposed in: Rust (`TableWriter::write_batch_auto_deferred`), Python (`Table.write_batch_auto_deferred()` + `write_batch_auto_deferred_async()`), CLI (`ailake insert --engine auto-deferred`).

**Consequences**:
- Callers get near-optimal index selection (GPU/many-core → IVF-PQ; otherwise HNSW) without specifying hardware.
- Throughput: ~200k vec/s on all hardware — same as HNSW deferred.
- During the index build window, shards are served via flat scan (exact brute-force). Acceptable; identical to HNSW/IVF-PQ deferred behavior.
- Hardware detection result is cached per process (`OnceLock`); no per-call overhead.

**Rejected alternatives**:
- Always IVF-PQ deferred: wrong choice on single-core machines; IVF-PQ quality degrades with <1k training vectors per cluster.
- Always HNSW deferred: misses GPU acceleration available on ML infra nodes.
- Caller-specified engine: puts the burden of hardware knowledge on every application developer.

---

## ADR-017: Reject Arrow Flight as unified interop layer; adopt Arrow IPC bytes in JNI write_batch

**Date**: 2026-06
**Status**: Accepted

**Context**: Proposal to replace per-language FFI/JNI bindings (JNA C-ABI for Spark/Trino/Flink, subprocess for C++, CGo-free direct reads for Go) with Apache Arrow Flight (gRPC + Arrow IPC) as a single transport layer. Rationale: reduce C-binding boilerplate, eliminate CString/pointer lifecycle bugs, standardize data transfer format.

**Decision**: Do **not** adopt Arrow Flight as a unified interop layer. Instead, replace JSON string serialization in `ailake_write_batch_json` with Arrow IPC bytes (binary) as the next incremental improvement to the JNI boundary.

**Reasoning**:

1. **FFI is not the bottleneck.** Write throughput is dominated by S3 PUT (~4000ms). JSON serialization of 1k×1536-dim embeddings adds ~10ms (0.25% overhead). Search JSON is ~6KB (one query vector) — parse time is noise vs HNSW traversal and S3 I/O.

2. **`ailake serve` already provides language-agnostic transport.** The existing Axum HTTP/JSON server covers any language unable to embed native libs. Arrow Flight would be a binary-protocol version of this — better latency by ~0.5ms, but requires a mandatory daemon, port management, health checks, and crash recovery.

3. **PyO3 and Go would regress.** PyO3 is zero-copy native (Arrow IPC in-process). Go reads catalog + AILK directly from S3 with zero FFI. Both would gain a network hop with no upside.

4. **Distributed Spark benefits from distributed search.** Current model: each executor downloads HNSW + searches locally in parallel. Centralising search via a Flight server serialises this parallelism — wrong direction for large clusters.

5. **JVM FFI surface is 10 functions behind JNA — manageable.** The correctness issues fixed this sprint (null-bytes, UTF-8 slices, pointer lifetimes) are addressed by `cstr_json` + clippy, not by a transport change.

6. **Arrow Flight Java client adds 50MB+ of Arrow/gRPC/Netty to the Spark classpath**, with version conflict risk against Spark's own Arrow/Netty stack (Spark 3.5 ships Arrow 12).

**Next incremental step (Fase 10)**: Replace the JSON embedding payload in `ailake_write_batch_json` with Arrow IPC bytes passed via JNI. Eliminates the only meaningful overhead (~10ms/1k vecs JSON encode/decode) without adding a daemon or changing the calling convention. Estimated effort: 1 week.

**Conditions that would reopen Arrow Flight**:
- GPU co-location becomes mandatory (all HNSW/IVF-PQ on dedicated GPU nodes, not co-located with Spark executors).
- JVM plugin count grows beyond 8 and FFI maintenance becomes intractable.
- Multi-tenant shared inference cluster required.
- Streaming search results > 10k rows where HTTP chunked transfer is insufficient.

**Consequences**:
- JNA C-ABI surface (`ailake-jni`) remains the canonical JVM binding path.
- `ailake serve` HTTP/JSON remains the canonical language-agnostic access path.
- PyO3 (Python) and Go SDK remain as embedded zero-FFI paths.
- Arrow IPC bytes in `write_batch` (Fase 10) is the only transport change planned.

**Rejected alternatives**:
- Full Arrow Flight adoption: operational complexity (daemon lifecycle, TLS, service discovery) outweighs benefits at current scale.
- gRPC without Arrow: same daemon overhead, loses zero-copy columnar benefit.

---

## ADR-018: Read compatibility with generic Iceberg engines does not imply write compatibility

**Date**: 2026-07
**Status**: Accepted

**Context**: The AI-Lake file format guarantees that any standard Parquet/Iceberg reader (DuckDB, Spark, Trino, PyIceberg) can read AI-Lake tables without a plugin — the AILK footer sits after the Parquet footer and is invisible to standard readers (§2, §12 of `CLAUDE.md`). This was informally read as "any Iceberg-compliant engine can safely operate on AI-Lake tables." It cannot: if a generic engine's own maintenance operation (Spark/Trino `OPTIMIZE` / `rewrite_data_files`, or any Iceberg-standard compaction) rewrites an AI-Lake data file, the output is valid Parquet with no AILK footer and no `centroid`/`radius` in the manifest entry it produces — because that field piggybacks on Iceberg's `key_metadata` (reserved for encryption metadata) and generic writers never populate it. The AI-Lake SDK does not treat this as corruption; it degrades gracefully. This ADR documents the actual (not assumed) behavior and the fixes needed to make that degradation safe and visible.

**Findings from investigation** (see `ailake-query/src/scanner.rs`, `ailake-query/src/compaction.rs`, `ailake-file/src/reader.rs`):

1. **Query-time**: `VectorPruner::prune` keeps files with no centroid rather than dropping them (`pruner.rs`). `scanner.rs::search` detects a missing AILK footer via `AilakeFileReader::is_ailake_file()` and falls back to an exact O(N) flat scan — correct results, degraded latency. This path was silent (`debug!`-level log only) prior to this ADR.
2. **Compaction-time — the actual risk**: `CompactionExecutor::compact()`/`compact_incremental()` read every input file's data via `read_parquet()`, which decodes the vector column straight from Parquet independent of the AILK footer. Two related bugs meant a foreign-rewritten file's rows could be **silently dropped** during the next compaction pass that swept it up: (a) the per-file read loops treated "no AILK footer" as "skip this file entirely" instead of "no index to reuse, but still read its data"; (b) `CompactionExecutor::run()`/`run_deferred()` committed `SnapshotOperation::Replace` with only the *compacted* subset of files, and `Replace` does not inherit the previous manifest (`HadoopCatalog::commit_snapshot`) — so files outside a partial compaction pass (too large, or beyond `max_files_per_pass`) vanished from the table entirely, whether or not they were AI-Lake-native. Both are fixed as part of this ADR (see Consequences).
3. **No production wiring for the existing invariant check**: `AilakeFileReader::verify_integrity()` (`parquet_count == hnsw_node_count == header.record_count`) existed but was only ever called from tests.

**Decision**: Treat "generic engine rewrote this file" as an expected, recoverable condition, not an error state — but make it loud and self-healing:

1. **Detect**: a `DataFileEntry` with no `centroid_b64` was never produced by the AI-Lake SDK (every write path — inline, deferred, incremental-merge — computes and stores a centroid before the HNSW build itself completes). `CompactionPlanner::plan()` now treats any such file as a priority repair candidate, bypassing both the size filter and `min_files_to_compact` — a single foreign file is worth compacting on its own rather than waiting for a batch.
2. **Warn**: `scanner.rs::search` distinguishes the *expected* transient case (`IndexStatus::Indexing`, our own deferred write still building its index — `debug!`) from the *unexpected* case (no footer, not `Indexing` — `warn!`, with an aggregate summary at the end of each search call). `ailake info` reports foreign-file paths so operators can see drift without waiting for a slow query.
3. **Never lose data on repair**: compaction must read and preserve every row from every input file regardless of index presence, and `Replace`/`Overwrite` commits must always carry the complete resulting file list (untouched files included), never a partial one.
4. **Verify**: `verify_integrity()` now runs after every compaction merge, before the result is committed to the catalog — catching a mismatched merge as a build-time error instead of a query-time surprise.

**Consequences**:
- Vector search over a table touched by a generic engine remains **correct** at all times (flat scan is exact) — never silently wrong.
- Search over such files is **slower** (O(N) instead of O(log N)) until the next compaction pass, which is now guaranteed to happen promptly (bypasses batching thresholds) and to be **visible** (`warn!` logs, `ailake info`).
- `CompactionExecutor::run()`/`run_deferred()` (used by `ailake-jni` — the Spark/Trino/Flink plugins) no longer risk dropping untouched files from a table on a partial compaction pass; `compact()`/`compact_incremental()` no longer risk dropping rows from a file that lost its AI-Lake index. Both were live bugs, not hypothetical — reproduced with regression tests (`ailake-catalog/src/hadoop.rs::replace_does_not_inherit_previous_manifest`, `ailake-query/src/compaction.rs::run_preserves_untouched_files_outside_compaction_pass`, `::compact_preserves_rows_from_footerless_file`).
- At the time this ADR was written, the CLI (`ailake compact`) and `ailake serve`'s `/compact` endpoint already built the correct (full) file list independently of `CompactionExecutor::run()`, so neither was exposed to the Replace bug. The CLI's `Compact` command was later refactored to call `CompactionExecutor::run()`/`run_deferred()` directly instead of duplicating that file-list logic — it now also inherits `run()`'s correct `parent_snapshot_id` (a separate, smaller bug: hardcoded `None` even though a current snapshot always exists at compaction time, breaking Iceberg lineage/`expire_snapshots` for compacted tables), and gained `--deferred`/`--max-files-per-pass`/`--format json` in the process. `ailake serve`'s `/compact` endpoint still builds its own file list independently and was not touched.

**Rejected alternatives**:
- Refuse to open / hard-error on a file with no AILK footer: would turn a recoverable, correctly-served-degraded state into a hard outage for any table ever touched by a standard Iceberg maintenance job — unacceptable given §12's explicit promise that generic engines can operate on AI-Lake tables.
- Encode centroid/HNSW offset in a first-class Iceberg V3 field instead of `key_metadata`: would make foreign writes detectable without the `centroid_b64.is_none()` heuristic, but requires the V3 manifest schema migration tracked separately in `CLAUDE.md` §10 Fase 5 ("Iceberg V3 — Column Statistics estendidas"); the heuristic is correct today because every AI-Lake write path populates centroid unconditionally.
- A file lock / marker to block generic engines from writing to AI-Lake tables at all: not enforceable — Iceberg has no such mechanism, and the whole point of Iceberg compatibility is that generic engines are allowed to operate on the table.
- Protobuf at the JNI boundary: additional schema maintenance without Arrow ecosystem benefits.

---

## ADR-019: DuckLakeCatalog drives the real `ducklake` extension instead of hand-rolling catalog DDL

**Date**: 2026-07
**Status**: Accepted

**Context**: `CLAUDE.md` §10 Fase 5 had a queued item for a DuckDB/DuckLake-backed `CatalogProvider`, blocked on "aguardar estabilização da spec DuckLake" — DuckLake reached v1.0 in April 2026, unblocking the work. The obvious implementation mirrors `JdbcCatalog`/`HadoopCatalog`: read DuckLake's own public table schemas (`ducklake_data_file`, `ducklake_snapshot`, `ducklake_table`, …) and write to them directly. That approach was rejected after reading `duckdb/ducklake`'s own C++ source (`ducklake_metadata_manager.cpp`, `ducklake_initializer.cpp`): DuckLake's *bootstrap* invariants — initial `ducklake_metadata` rows, how `next_catalog_id`/`next_file_id` counters get allocated safely across concurrent transactions, the internal `__ducklake_metadata_<alias>` attachment naming — are implementation details, not public API, and getting them wrong would produce a catalog file that looks fine until a real `ducklake` extension (or a future DuckLake version) tries to read it.

**Decision**: `DuckLakeCatalog` never writes to DuckLake's own catalog tables directly. It only calls sanctioned public SQL surfaces: `CREATE`/`ALTER`/`DROP TABLE` against the attached `lake.<namespace>.<table>`, `CALL ducklake_add_data_files(...)` (the documented external-file-registration function), `ducklake_list_files(...)` (the documented file-listing table function), and plain row-predicate `DELETE`. AI-Lake's own per-file vector metadata (centroid, radius, HNSW offset/len, index status, embedding model, partition value, deletion vector pointer) has no home in DuckLake's fixed `ducklake_data_file` schema, so it lives in a sidecar table (`main.ailake_vector_index`) in the same DuckDB connection but outside the `ducklake:` attachment — a plain table with no DuckLake versioning overhead, entirely owned by AI-Lake.

Four behaviors assumed at design time turned out to be wrong when verified against a **live** `ducklake` extension and, for the last two, the actual built `ailake` CLI binary — not just documentation or unit tests:

1. **Cross-attachment transactions**: DuckDB refuses to write to two attached databases (`lake` and `main`) within a single transaction — a hard, undocumented-until-you-hit-it constraint (`"a single transaction can only write to a single attached database"`). `commit_snapshot`/`evolve_schema` commit in two sequential phases instead of one atomic transaction: `lake` (source of truth for which files/columns exist) first, then `main` (sidecar). A crash between phases only ever degrades gracefully — see `ailake-catalog/src/ducklake.rs`'s module doc comment and `docs/guides/DUCKLAKE_CATALOG.md` for the exact failure modes — never wrong or corrupt data.
2. **File retirement**: the natural design — `DELETE FROM lake.tbl WHERE filename = ?` to retire a file being replaced by compaction/backfill/memory-decay/migration — correctly empties the file's rows for any DuckLake reader, but does **not** remove the file from `ducklake_list_files()`; only a maintenance pass (`ducklake_expire_snapshots`/`ducklake_cleanup_files`) reclaims it. There is no sanctioned "drop this one file, right now" primitive. `list_files()` was redesigned around the sidecar's own `active` boolean as the authority for what AI-Lake considers active, with `ducklake_list_files()` consulted only to detect genuinely foreign (non-AI-Lake-written) files — the same "foreign write" contract ADR-018 established for the Iceberg backends.
3. **Path resolution**: `DataFileEntry::path` is warehouse-relative by convention (`Store::get`/`put` and every other `CatalogProvider` backend agree on this), but `ducklake_add_data_files` resolves relative paths against DuckDB's own process working directory, not the warehouse. Failed the very first real `ailake insert` once this backend was wired into the CLI (`No files found that match the pattern "data/part-00000.parquet"`). Fixed by resolving to an absolute path only at the SQL call sites (`resolve_path`), keeping `DataFileEntry::path` warehouse-relative everywhere else.
4. **`allow_missing`**: `ducklake_add_data_files`'s default `allow_missing => false` rejects any file that predates a column `evolve_schema` has since added — the normal case, since AI-Lake never rewrites old files just because the schema grew. Fixed by always passing `allow_missing => true` alongside `ignore_extra_columns => true`.

Verifying finding 4 against the CLI surfaced a **fifth, unrelated, pre-existing bug** independent of this backend and of DuckLake entirely: `ParquetVectorReader::read_all()` (`ailake-parquet`) always decoded the vector column as F16 regardless of the file's actual stored precision (ignoring the `ailake.precision` KV metadata every AI-Lake writer already embeds), silently corrupting reads of any F32-precision table through `AilakeFileReader::read_parquet()` — the shared path both compaction and the scanner's foreign-file flat scan use. Reproduced identically against the default `HadoopCatalog`, confirming it had nothing to do with DuckLake; fixed in the same pass since it blocked verifying `ailake compact` end-to-end. See the `CHANGELOG.md` "Fixed" entry for the full writeup.

**Consequences**:
- `DuckLakeCatalog` is safe to run against any conformant `ducklake` extension version without depending on undocumented internal schema details staying stable.
- Full `Append`/`Overwrite`/`Replace`/`Delete` support — the operations `compaction.rs`, `backfill.rs`, `memory_decay.rs`, `migration.rs`, and `writer.rs`'s deferred-index-status patch all depend on — works correctly under this design, verified with a real round-trip integration test (create → insert → search-shaped list → Overwrite → evolve_schema → drop) against a live DuckDB/DuckLake catalog file, no mocks, **and** with the actual built `ailake` CLI binary end to end (create → insert → evolve → insert-after-evolve → search → compact → info).
- Physical space reclamation for retired files is not handled by this module — it requires an operator (or a future job) to run DuckLake's own `ducklake_expire_snapshots`/`ducklake_cleanup_files` periodically, the same class of follow-up maintenance Iceberg's `expire_snapshots` needs. Not wired in here: a safe default retention window wasn't verified against a live multi-snapshot scenario while building this backend.
- Wired into `ailake-cli` behind its own opt-in `catalog-ducklake` feature (`--catalog hadoop|ducklake`, defaults to `hadoop`) — local filesystem `--store` only, erroring clearly on object-storage URLs or a build without the feature rather than silently falling back. `JdbcCatalog`/`GlueCatalog`/`NessieCatalog` remain backend-only, unwired into any CLI/Python surface — that's still a distinct, not-yet-started piece of work for each of those, not something this backend's CLI wiring changes; DuckLake's embedded, connection-string-free nature is specifically what made a single flag (no extra connection config) a reasonable CLI surface.
- v1 scopes the DuckLake metadata store to DuckDB-as-catalog only (no multi-writer support — same class of constraint as SQLite). A Postgres-backed DuckLake metadata catalog would lift this but is out of scope.

**Rejected alternatives**:
- Hand-roll DuckLake's own catalog DDL directly (the original queued plan): rejected per Context above — undocumented bootstrap invariants make this fragile in a way the real extension isn't.
- Wrap `lake` + `main` writes in one transaction: impossible — DuckDB's cross-attached-database transaction restriction is a hard constraint, not a configuration option (confirmed via DuckDB's own documentation on `ATTACH`/multi-database support; no bypass flag exists).
- Use DuckLake's own compaction (`ducklake_merge_adjacent_files`) to retire superseded files: would let DuckLake physically rewrite AI-Lake's Parquet files, stripping the AILK footer — the same class of conflict ADR-018 already documents for generic Iceberg engines running `OPTIMIZE`/`rewrite_data_files` on AI-Lake tables.
