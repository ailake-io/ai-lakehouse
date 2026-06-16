# LLM_CONTEXT.md — LLM Context Schema and Retrieval Design

## Problem statement

A vector search returns chunks ranked by embedding similarity. Without additional design, those chunks land in the LLM prompt as isolated fragments — missing which document they came from, which section, what came before and after. The LLM answers with what it has, which is often not enough.

This document specifies how AI-Lake tables that serve RAG workloads store and retrieve context.

---

## `LlmContextSchema` — canonical table schema for RAG

Tables serving LLM workloads SHOULD use this schema. It is not enforced by the format — any schema can have vector columns — but this schema is what `ContextAssembler` expects.

```rust
// ailake-core/src/schema.rs

/// Canonical column names for LLM-context tables.
/// Use these as Parquet column names — ContextAssembler reads them by name.
pub mod llm_columns {
    // Identity
    pub const CHUNK_ID: &str = "chunk_id";
    pub const DOCUMENT_ID: &str = "document_id";
    pub const CHUNK_INDEX: &str = "chunk_index";
    pub const TOTAL_CHUNKS: &str = "total_chunks";

    // Content
    pub const CHUNK_TEXT: &str = "chunk_text";

    // Structural context (critical for disambiguation)
    pub const DOCUMENT_TITLE: &str = "document_title";
    pub const SECTION_PATH: &str = "section_path";
    pub const PRECEDING_CONTEXT: &str = "preceding_context";
    pub const FOLLOWING_CONTEXT: &str = "following_context";

    // Semantic context
    pub const DOCUMENT_SUMMARY: &str = "document_summary";
    pub const CHUNK_SUMMARY: &str = "chunk_summary";

    // Provenance
    pub const SOURCE_URI: &str = "source_uri";
    pub const PAGE_NUMBER: &str = "page_number";
    pub const CREATED_AT: &str = "created_at";
    pub const DOCUMENT_DATE: &str = "document_date";

    // Vectors
    pub const EMBEDDING: &str = "embedding";
    pub const CONTEXT_EMBEDDING: &str = "context_embedding";
}
```

### Parquet schema (Arrow types)

| Column | Arrow type | Nullable | Notes |
|---|---|---|---|
| `chunk_id` | `Utf8` | No | UUID string |
| `document_id` | `Utf8` | No | UUID string |
| `chunk_index` | `UInt32` | No | 0-based position in document |
| `total_chunks` | `UInt32` | No | Total chunks in document |
| `chunk_text` | `LargeUtf8` | No | The text that was embedded |
| `document_title` | `Utf8` | No | Title of the parent document |
| `section_path` | `Utf8` | Yes | e.g. `"Chapter 3 > Section 3.2"` |
| `preceding_context` | `Utf8` | Yes | Last N chars of previous chunk |
| `following_context` | `Utf8` | Yes | First N chars of next chunk |
| `document_summary` | `LargeUtf8` | Yes | ≤256-token summary of full document |
| `chunk_summary` | `Utf8` | Yes | 1-2 sentence summary of chunk |
| `source_uri` | `Utf8` | No | Original document URI |
| `page_number` | `UInt32` | Yes | PDF page or equivalent |
| `created_at` | `Timestamp(Micros, UTC)` | No | Ingest timestamp |
| `document_date` | `Date32` | Yes | Document publication date |
| `embedding` | `FIXED_LEN_BYTE_ARRAY(dim*2)` | No | F16 vector, primary search |
| `context_embedding` | `FIXED_LEN_BYTE_ARRAY(dim*2)` | No | F16 vector, contextual search |

### Multi-column vectors in the unified file

When a table has both `embedding` and `context_embedding`:

- Both columns are stored in the Parquet section as `FIXED_LEN_BYTE_ARRAY`.
- The AI-Lake footer contains **two HNSW graphs**, one per column.
- The AI-Lake header has the `multi-column` flag set (bit 1).
- See [`FILE_FORMAT.md`](./FILE_FORMAT.md) for multi-column footer layout.

The file remains a single self-contained unit. The trade-off: file size grows because each column needs its own HNSW (~10-20% of vector data per column).

### Storage estimates (Parquet with Zstd, 1M chunks, dim=1536, F16)

| Column group | Uncompressed | In Parquet (Zstd) |
|---|---|---|
| IDs and integers | ~60 MB | ~15 MB |
| `chunk_text` (avg 500 tokens ≈ 2 KB) | ~2 GB | ~400 MB |
| Context text fields (200 chars each) | ~400 MB | ~80 MB |
| `document_summary` (avg 256 tokens) | ~1 GB | ~200 MB |
| `chunk_summary` (avg 100 tokens) | ~400 MB | ~80 MB |
| String metadata | ~200 MB | ~50 MB |
| `embedding` F16 | ~3 GB | ~2.9 GB (incompressible) |
| `context_embedding` F16 | ~3 GB | ~2.9 GB |
| **Parquet section total** | **~10 GB** | **~6.6 GB** |
| HNSW for `embedding` (in AI-Lake footer) | — | ~500 MB |
| HNSW for `context_embedding` (in AI-Lake footer) | — | ~500 MB |
| **Grand total (file size)** | | **~7.6 GB** |

---

## Dual embeddings strategy

### Why two embeddings

A single embedding of the raw `chunk_text` captures what the chunk says but not where it lives in the document. A chunk reading "The value increased 23% in Q3" is ambiguous without knowing it refers to gross margin in a specific financial report.

The `context_embedding` embeds a richer string that provides that grounding.

### How `context_embedding` is generated

At ingest time, before calling the embedding model:

```python
def build_context_string(chunk: Chunk) -> str:
    parts = []
    if chunk.document_title:
        parts.append(f"[DOCUMENT: {chunk.document_title}]")
    if chunk.section_path:
        parts.append(f"[SECTION: {chunk.section_path}]")
    if chunk.document_summary:
        parts.append(f"[SUMMARY: {chunk.document_summary}]")
    parts.append("---")
    parts.append(chunk.chunk_text)
    return "\n".join(parts)
```

### When to use each embedding for search

| Query type | Use | Rationale |
|---|---|---|
| Specific fact lookup | `embedding` | Matches exact content |
| "Tell me about X in document Y" | `context_embedding` | Document context matters |
| Multi-hop reasoning | both (RRF fusion) | Higher recall |
| Code search | `embedding` | Code is self-contained |
| Narrative / thematic | `context_embedding` | Topic matters more than words |

### Cross-modal search with `search_multimodal`

For tables with multiple vector columns (text + image, dual embeddings, or any other combination), `search_multimodal` runs an independent HNSW search per column and fuses results via RRF:

```python
# Text + image cross-modal search
results = ailake.search_multimodal(
    "s3://my-lake/media/",
    queries=[
        ("embedding",       text_vec,  0.7),
        ("image_embedding", image_vec, 0.3),
    ],
    top_k=20,
)
# → [{"row_id": int, "rrf_score": float, "file": str}]  — descending rrf_score
```

Columns may have different dimensions (`dim=1536` for text, `dim=512` for images). Per-column dims are auto-detected from `ailake.dim-<col>` Iceberg properties.

### Reciprocal Rank Fusion (RRF) for dual-embedding search

```rust
// ailake-query/src/scanner.rs

pub fn rrf_merge(
    results_a: &[(RowId, f32)],  // from embedding search
    results_b: &[(RowId, f32)],  // from context_embedding search
    k: f32,                       // RRF constant, typically 60.0
    top_n: usize,
) -> Vec<(RowId, f32)> {
    let mut scores: HashMap<RowId, f32> = HashMap::new();

    for (rank, (row_id, _)) in results_a.iter().enumerate() {
        *scores.entry(*row_id).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, (row_id, _)) in results_b.iter().enumerate() {
        *scores.entry(*row_id).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }

    let mut merged: Vec<(RowId, f32)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    merged.truncate(top_n);
    merged
}
```

---

## `ContextAssembler`

### Purpose

Given a list of retrieved chunks, produce a prompt-ready context string that:
- Eliminates near-duplicate chunks
- Restores reading order within each document
- Fits within the LLM's token budget
- Maximizes information diversity

### Configuration

```rust
// ailake-query/src/context_assembler.rs

pub struct ContextAssemblerConfig {
    /// Approximate token budget (4 chars ≈ 1 token). Default: 4096.
    pub max_tokens: usize,

    /// Cosine distance below which two chunks are considered duplicates.
    /// Keep the first chunk (already sorted by relevance). Default: 0.05.
    pub dedup_threshold: f32,

    /// When true, chunks from the same document are grouped and sorted
    /// by chunk_index before inclusion. Default: true.
    pub group_by_document: bool,

    /// Max chunks from a single document_id.
    /// Prevents one document from dominating context. Default: 10.
    pub max_chunks_per_document: usize,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            dedup_threshold: 0.05,
            group_by_document: true,
            max_chunks_per_document: 10,
        }
    }
}
```

### Assembly algorithm

```
Input: Vec<Chunk>, ContextAssemblerConfig

1. DEDUPLICATION
   Build pairwise cosine distance matrix for retrieved embeddings.
   For each pair (a, b) where dist < dedup_threshold:
     Remove the lower-scored chunk.
   O(n²) — acceptable for n ≤ 200 (typical top_k range).

2. GROUPING
   Group remaining chunks by document_id.
   Within each group, sort ascending by chunk_index.
   Cap each group at max_chunks_per_document.

3. ORDERING GROUPS
   Order document groups by the highest score of any chunk in the group.
   (Best-first ordering of documents.)

4. BUDGET ALLOCATION
   token_used = 0
   included = []
   For each chunk (in order from step 3):
     chunk_tokens = estimate_tokens(chunk)
     if token_used + chunk_tokens <= max_tokens:
       included.append(chunk)
       token_used += chunk_tokens
     else:
       break  (greedy — no backtracking)

5. RENDERING
   For each chunk in included:
     render XML block (see below)

Output: AssembledContext { text: String, token_estimate: usize, chunk_count: usize }
```

### Output XML format

```xml
<context>
  <document id="abc-123" title="Annual Report 2023 — ACME Corp" source="s3://lake/annual_report.parquet">
    <chunk index="4" section="Financial Results &gt; Q3 Revenue">
      <text>Gross margin improved to 42.3% in Q3 2023, up from 38.1% in Q3 2022, driven by operational efficiency gains in the APAC manufacturing segment.</text>
    </chunk>
  </document>
  <document id="def-456" title="Q3 2023 Earnings Call Transcript" source="s3://lake/earnings_call.parquet">
    <chunk index="2" section="CFO Remarks">
      <text>We are particularly proud of the margin expansion this quarter. The 420 basis point improvement reflects sustained investment in automation.</text>
    </chunk>
  </document>
</context>
```

Special characters (`&`, `<`, `>`, `"`) are XML-escaped in all attribute and text values.

### Token estimation

```rust
/// Rough token count: char_budget = max_tokens × 4.
/// For precision, replace with tiktoken binding for the target model.
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}
```

---

## Ingest pipeline (reference implementation)

```python
# Reference Python ingest pipeline using ailake-py bindings (Phase 2)
import ailake
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone

def ingest_document(doc, embed_fn, writer: ailake.TableWriter):
    chunks = chunk_document(doc, chunk_size=512, overlap=50)
    doc_summary = embed_fn.summarize(doc.full_text, max_tokens=256)

    rows = []
    raw_embeddings = []
    ctx_embeddings = []

    for i, chunk in enumerate(chunks):
        preceding = chunks[i-1].text[-200:] if i > 0 else ""
        following = chunks[i+1].text[:200] if i < len(chunks)-1 else ""

        # Build context string for context_embedding
        ctx_string = f"[DOCUMENT: {doc.title}]\n"
        if chunk.section:
            ctx_string += f"[SECTION: {chunk.section}]\n"
        ctx_string += f"[SUMMARY: {doc_summary}]\n---\n{chunk.text}"

        rows.append({
            "chunk_id": str(uuid4()),
            "document_id": doc.id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_text": chunk.text,
            "document_title": doc.title,
            "section_path": chunk.section or "",
            "preceding_context": preceding,
            "following_context": following,
            "document_summary": doc_summary,
            "chunk_summary": embed_fn.summarize(chunk.text, max_tokens=64),
            "source_uri": doc.uri,
            "page_number": chunk.page,
            "created_at": datetime.now(timezone.utc),
        })

        raw_embeddings.append(embed_fn.embed(chunk.text))
        ctx_embeddings.append(embed_fn.embed(ctx_string))

    writer.write_batch(
        rows=rows,
        embeddings={
            "embedding": np.array(raw_embeddings, dtype=np.float32),
            "context_embedding": np.array(ctx_embeddings, dtype=np.float32),
        }
    )
    # Each batch creates one unified .parquet file with both HNSW graphs in its footer
```

---

---

## `MultimodalContextSchema` — extending LLM context with media

`MultimodalContextSchema` extends `LlmContextSchema` for tables that also carry media embeddings and references.

### Canonical column names (`multimodal_columns` module)

```rust
// ailake-core/src/schema.rs
pub mod multimodal_columns {
    pub const MEDIA_URI:        &str = "media_uri";        // S3/GCS/HTTPS URI of the raw asset
    pub const MEDIA_MIME:       &str = "media_mime";        // MIME type (image/jpeg, audio/mpeg, …)
    pub const MEDIA_CAPTION:    &str = "media_caption";     // Caption from BLIP-2 / Whisper
    pub const IMAGE_EMBEDDING:  &str = "image_embedding";   // CLIP/SigLIP dim=512, FIXED_LEN_BYTE_ARRAY F16
    pub const AUDIO_TRANSCRIPT: &str = "audio_transcript";  // Whisper transcript
    pub const THUMBNAIL_B64:    &str = "thumbnail_b64";     // Base64 JPEG ≤ 64×64 for inline LLM context
}
```

### Design principle

**AI-Lake is not a blob store.** Media files live in object storage (`s3://`, `gs://`, `az://`). AI-Lake stores only:
- URIs pointing to the media (`media_uri`)
- Embeddings derived from the media (`image_embedding`)
- Derived text (`media_caption`, `audio_transcript`, `thumbnail_b64`)

### Example multimodal table Arrow schema

```
chunk_id:          Utf8
chunk_text:        LargeUtf8
embedding:         FixedSizeBinary(3072)    -- text F16, dim=1536
image_embedding:   FixedSizeBinary(1024)    -- image F16, dim=512  (ailake.modality-image_embedding = "image")
media_uri:         Utf8                     -- s3://bucket/photo.jpg
media_mime:        Utf8                     -- image/jpeg
media_caption:     Utf8                     -- BLIP-2 caption
audio_transcript:  Utf8                     -- Whisper (null for images)
thumbnail_b64:     Utf8                     -- base64 JPEG 64×64
```

Each vector column (`embedding`, `image_embedding`) carries the `ailake.modality-<col>` Iceberg property,
allowing readers to select the correct HNSW by modality tag without inspecting vector data.

See [`07_multimodal.ipynb`](../../tests/docker/demo/notebooks/07_multimodal.ipynb) for a complete demo.

---

## Chunking guidelines

The AI-Lake format is chunking-strategy agnostic. These are recommendations, not enforcement.

| Content type | Recommended chunk size | Overlap | Notes |
|---|---|---|---|
| Prose (articles, reports) | 400–600 tokens | 50–100 tokens | Preserve paragraph boundaries |
| Code | Full function/class | None | Never split mid-function |
| Tables | Full table + caption | None | Tables lose meaning when split |
| Q&A pairs | Full Q+A | None | Keep together |
| Transcripts | ~300 tokens | 50 tokens | Speaker turns as natural breaks |

The `section_path` field is the most important for disambiguation. Always populate it when the source document has structural hierarchy (headers, sections, chapters).
