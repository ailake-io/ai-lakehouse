# DECISIONS.md — Architecture Decision Records (ADR)

Decisions are numbered and immutable once merged. To change a decision, add a new ADR that supersedes the old one.

---

## ADR-001: Rust as the sole implementation language

**Date**: 2024-08  
**Status**: Accepted  

**Context**: The core operations — reading/writing files, HNSW index construction, quantization, centroid computation — are I/O and CPU-bound at petabyte scale. Language choice directly impacts throughput and operational cost.

**Decision**: Rust for all `ailake-*` crates. Python and JVM bindings are thin wrappers (PyO3, uniffi) that call into Rust. No business logic in binding layers.

**Consequences**:
- Zero GC pauses during index traversal (critical for p99 latency).
- `cargo build --target` enables cross-compilation without complex toolchains.
- Binding maintenance cost: PyO3 and uniffi are mature and well-documented.
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
- GPU support not available (Faiss has CUDA). Deferred to Phase 4 via FFI to cuVS if needed.

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

**Decision**: One physical file per logical data unit. Layout: Parquet section (header + row groups + footer + `PAR1`) followed by AI-Lake footer extension (header + centroid + HNSW graph + trailer). Parquet readers stop at the final `PAR1` and never see the extension.

**Consequences**:
- **Source-of-truth simplicity**: one file = one snapshot of data + index. No three-way consistency to maintain.
- **Iceberg integration is cleaner**: the Iceberg DataFile entry points to the unified `.parquet` file. Vector statistics (centroid, radius, HNSW byte offsets) go in `custom-properties` of the DataFile entry — a spec-defined extension point.
- **Atomic writes**: one S3 PUT per data file. No partial-state recovery logic.
- **Compatibility relies on two specs**: Iceberg Spec v2 AND the Parquet spec's rule that "trailing data after PAR1 must be tolerated." This is implemented by major Parquet readers but is technically a quality-of-implementation guarantee, not a strict spec requirement. We document this dependency and validate via compatibility tests.
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

**Decision**: XML with semantic tags (`<source>`, `<document>`, `<section>`, `<content>`, `<before>`, `<after>`).

**Consequences**:
- Claude (Anthropic) performs well with XML-structured context per published prompt engineering guidance.
- Other models (GPT-4, Gemini) also handle XML well — widely supported.
- Slightly more tokens than plain text due to tags. Overhead is ~5% for typical chunk sizes — acceptable.
- XML structure makes document attribution unambiguous, reducing hallucination of sources.

**Rejected alternatives**:
- Markdown: headings are less semantically explicit than XML tags.
- JSON: verbose, harder for LLMs to read as natural language context.
- Plain text with separators: loses structural information; model must infer attribution.
