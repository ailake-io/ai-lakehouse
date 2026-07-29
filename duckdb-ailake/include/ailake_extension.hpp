// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// Shared types and AilakeLib singleton used by search and write functions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ailake {

struct SearchRow {
    int64_t     row_id;
    float       distance;
    std::string file_path;
};

// One query arm for cross-modal RRF search.
struct ModalQueryArg {
    std::string        col;
    std::vector<float> query;
    float              weight = 1.0f;
};

// One result row from ailake_search_multimodal.
struct MultimodalRow {
    int64_t     row_id;
    float       rrf_score;
    std::string file_path;
};

// One vector column in a multi-column (Phase 8 multimodal) write batch — e.g.
// text + image embeddings on the same row, each with its own HNSW index. See
// AilakeLib::write_batch_multi.
struct VectorColSpecArg {
    std::string                     col;
    int32_t                         dim = 0;
    std::string                     metric    = "cosine";
    std::string                     precision = "f16";
    std::string                     modality; // empty = unset
    std::vector<std::vector<float>> embeddings;
};

// ── ailake_scan column types ──────────────────────────────────────────────────

enum class ScanColType { INT64, FLOAT32, FLOAT64, VARCHAR, BOOL, LIST_FLOAT32, UNKNOWN };

// Pre-parsed column data for ailake_scan results (one of the value vectors is active).
struct ScanColumn {
    std::string  name;
    ScanColType  type = ScanColType::UNKNOWN;

    // Active member depends on type:
    std::vector<int64_t>              int_vals;    // INT64
    std::vector<float>                float_vals;  // FLOAT32
    std::vector<double>               double_vals; // FLOAT64
    std::vector<std::string>          str_vals;    // VARCHAR
    std::vector<bool>                 bool_vals;   // BOOL
    std::vector<std::vector<float>>   list_vals;   // LIST_FLOAT32
    std::vector<bool>                 is_null;     // parallel null bitmap (all types)
};

struct ScanResult {
    bool                     ok        = false;
    std::string              error;
    int64_t                  num_rows  = 0;
    std::vector<ScanColumn>  columns;
};

// ── ailake_estimate row shape ───────────────────────────────────────────────

// One storage/precision mode row from ailake_estimate_json's `estimates` array.
struct EstimateRow {
    std::string mode;
    int64_t     vectors_bytes           = 0;
    int64_t     index_bytes             = 0;
    int64_t     total_bytes             = 0;
    double      reduction_vs_f32_hnsw   = 0.0;
    std::string recall;
    std::string note;
};

struct EstimateResult {
    bool                      ok = false;
    std::string               error;
    std::vector<EstimateRow>  rows;
};

// ── AilakeLib singleton ───────────────────────────────────────────────────────

// Singleton holding resolved C-ABI function pointers. `ailake-jni` is linked
// statically into this extension binary (see CMakeLists.txt — corrosion +
// ailake-jni's `staticlib` crate-type), so these always point at real symbols
// resolved at link time — no dlopen, no runtime library search.
// Thread-safe after Load() completes.
class AilakeLib {
public:
    using search_fn_t        = char *(*)(const char *);
    using multimodal_fn_t    = char *(*)(const char *);
    using scan_fn_t          = char *(*)(const char *);
    using write_fn_t         = char *(*)(const char *);
    using search_text_fn_t   = char *(*)(const char *);
    using delete_where_fn_t  = char *(*)(const char *);
    using evolve_schema_fn_t = char *(*)(const char *);
    using write_multi_fn_t   = char *(*)(const char *);
    using create_table_fn_t  = char *(*)(const char *);
    using compact_fn_t       = char *(*)(const char *);
    using delete_rows_fn_t   = char *(*)(const char *);
    using add_vector_column_fn_t     = char *(*)(const char *);
    using backfill_vector_column_fn_t = char *(*)(const char *);
    using decay_memories_fn_t = char *(*)(const char *);
    using migrate_fn_t       = char *(*)(const char *);
    using estimate_fn_t      = char *(*)(const char *);
    using free_fn_t          = void (*)(char *);

    static AilakeLib &get();

    // Wire up the function pointers to the statically-linked ailake-jni
    // symbols. Always succeeds — safe to call multiple times (no-op after
    // the first call).
    bool load();

    // Every AilakeLib method below takes a trailing `catalog_opts_json`
    // (default "" = unset). When non-empty it must be a JSON object literal
    // whose fields are flattened straight into the ailake-jni request body —
    // the same "catalog"/"rest_*" fields the CLI's `--catalog rest --rest-*`
    // flags and ailake-py's `catalog_opts` dict populate (see
    // docs/guides/REST_CATALOG.md). Example:
    //   '{"catalog":"rest","rest_uri":"https://catalog.example.com",
    //     "rest_auth":"bearer","rest_token":"..."}'
    // Absent/"" = default Hadoop-style catalog, unchanged behavior. Malformed
    // JSON or a non-object value throws duckdb::InvalidInputException — never
    // silently falls back to Hadoop.

    bool is_ready()              const { return search_fn_        != nullptr; }
    bool is_multimodal_ready()   const { return multimodal_fn_    != nullptr; }
    bool is_scan_ready()         const { return scan_fn_          != nullptr; }
    bool is_search_text_ready()  const { return search_text_fn_   != nullptr; }
    bool is_delete_ready()       const { return delete_where_fn_  != nullptr; }
    bool is_evolve_ready()       const { return evolve_schema_fn_ != nullptr; }
    bool is_write_multi_ready()  const { return write_multi_fn_   != nullptr; }
    bool is_compact_ready()      const { return compact_fn_       != nullptr; }
    bool is_create_table_ready() const { return create_table_fn_ != nullptr; }
    bool is_delete_rows_ready()  const { return delete_rows_fn_   != nullptr; }
    bool is_add_vector_column_ready() const { return add_vector_column_fn_ != nullptr; }
    bool is_backfill_vector_column_ready() const { return backfill_vector_column_fn_ != nullptr; }
    bool is_decay_memories_ready() const { return decay_memories_fn_ != nullptr; }
    bool is_migrate_ready()      const { return migrate_fn_       != nullptr; }
    bool is_estimate_ready()     const { return estimate_fn_      != nullptr; }

    // Execute ailake_search_json. Returns empty on any error.
    // hybrid_text: when non-empty, enables hybrid BM25+vector RRF fusion.
    // text_column: Parquet column for BM25 scoring (default "chunk_text").
    // bm25_weight: BM25 weight in RRF (0.0 = pure vector, 1.0 = pure BM25).
    std::vector<SearchRow> search(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &vec_col,
        const std::vector<float> &query,
        int                       top_k,
        int                       ef_search        = 50,
        const std::string        &partition_filter = "",
        const std::string        &hybrid_text      = "",
        const std::string        &text_column      = "chunk_text",
        float                     bm25_weight      = 0.5f,
        const std::string        &ns               = "default",
        const std::string        &catalog_opts_json = ""
    ) const;

    // Execute ailake_search_text_json. Tantivy O(log N) fast path when FTS blob
    // present; fallback BM25 O(N) for legacy files. Returns empty on any error.
    // text_columns: Parquet columns to search (sends "text_columns" JSON array).
    std::vector<SearchRow> search_text(
        const std::string              &warehouse,
        const std::string              &table_name,
        const std::string              &query_text,
        int                             top_k,
        const std::vector<std::string> &text_columns    = {"chunk_text"},
        const std::string              &partition_filter = "",
        const std::string              &ns               = "default",
        const std::string              &catalog_opts_json = ""
    ) const;

    // Execute ailake_scan_json. Returns pre-parsed columnar data.
    ScanResult scan(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &vec_col,
        const std::vector<float> &query,
        int                       top_k,
        int                       ef_search = 50,
        const std::string        &ns        = "default",
        const std::string        &catalog_opts_json = ""
    ) const;

    // Execute ailake_search_multimodal_json. Returns empty on any error.
    std::vector<MultimodalRow> search_multimodal(
        const std::string                 &warehouse,
        const std::string                 &table_name,
        const std::vector<ModalQueryArg>  &queries,
        int                                top_k,
        const std::string                 &partition_filter = "",
        const std::string                 &ns               = "default",
        const std::string                 &catalog_opts_json = ""
    ) const;

    // Execute ailake_write_batch_json. Returns snapshot_id or -1 on error.
    // partition_fields_json: JSON array like [{"column":"x","transform":"identity","column_type":"string"}]
    // fts_columns_json: JSON array like ["chunk_text","title"] — empty = no FTS
    // format_version: 2 (default) or 3
    // hnsw_m: HNSW M parameter (-1 = use table default)
    // hnsw_ef_construction: HNSW ef_construction (-1 = use table default)
    // pre_normalize: normalize vectors to unit L2 at write time
    // deferred: build index asynchronously (write_batch_auto_deferred)
    int64_t write_batch(
        const std::string              &warehouse,
        const std::string              &ns,
        const std::string              &table_name,
        const std::string              &vec_col,
        int                             dim,
        const std::string              &metric,
        const std::string              &precision,
        const std::vector<int64_t>     &ids,
        const std::vector<std::vector<float>> &embeddings,
        const std::string              &partition_by          = "",
        const std::string              &partition_value       = "",
        const std::string              &partition_fields_json = "",
        int                             format_version        = 2,
        const std::string              &fts_columns_json      = "",
        const std::string              &fts_tokenizer         = "",
        int                             hnsw_m                = -1,
        int                             hnsw_ef_construction  = -1,
        bool                            pre_normalize         = false,
        bool                            deferred              = false,
        const std::string              &catalog_opts_json     = ""
    ) const;

    // Execute ailake_delete_where_json. Returns true on success.
    bool delete_where(
        const std::string              &warehouse,
        const std::string              &table_name,
        const std::string              &column,
        const std::vector<std::string> &values,
        const std::string              &ns = "default",
        const std::string              &catalog_opts_json = ""
    ) const;

    // Execute ailake_evolve_schema_json. Returns new schema_id or -1 on error.
    // add_columns_json: JSON array of {name, type, initial_default?}
    // rename_columns_json: JSON array of {from, to}
    int32_t evolve_schema(
        const std::string &warehouse,
        const std::string &table_name,
        const std::string &add_columns_json,
        const std::string &rename_columns_json,
        const std::string &ns = "default",
        const std::string &catalog_opts_json = ""
    ) const;

    // Execute ailake_write_batch_multi_json (Phase 8 multimodal write — N
    // independent vector columns, each getting its own HNSW section in the
    // same AI-Lake file). Returns snapshot_id, or -1 on error.
    int64_t write_batch_multi(
        const std::string                   &warehouse,
        const std::string                   &ns,
        const std::string                   &table_name,
        const std::vector<int64_t>          &ids,
        const std::vector<VectorColSpecArg> &vector_columns,
        int                                   format_version = 2,
        bool                                  deferred       = false,
        const std::string                    &catalog_opts_json = ""
    ) const;

    // Execute ailake_create_table_json. Returns true on success.
    // warehouse: table root path.
    // ns: namespace (default "default").
    // table_name: table name (default "table").
    // vector_column: vector column name (default "embedding").
    // dim: vector dimension.
    // metric: distance metric (default "cosine").
    // precision: storage precision (default "f16").
    // format_version: Iceberg format version (2 or 3, default 2).
    // hnsw_m, hnsw_ef_construction: HNSW params (-1 = use default).
    // pre_normalize: normalize vectors to unit L2 at write time.
    // modality: vector modality (empty = unset).
    // partition_by, partition_value, partition_column_type: partition settings.
    // partition_fields_json: JSON array of {column, transform, column_type}.
    // fts_columns: comma-separated FTS text columns (empty = no FTS).
    // fts_tokenizer: FTS tokenizer (default empty).
    // embedding_model: model name (empty = unset).
    bool create_table(
        const std::string &warehouse,
        const std::string &ns,
        const std::string &table_name,
        const std::string &vector_column,
        int                dim,
        const std::string &metric,
        const std::string &precision,
        int                format_version     = 2,
        int                hnsw_m             = -1,
        int                hnsw_ef_construction = -1,
        bool               pre_normalize      = false,
        const std::string &modality           = "",
        const std::string &partition_by       = "",
        const std::string &partition_value    = "",
        const std::string &partition_column_type = "",
        const std::string &partition_fields_json  = "",
        const std::string &fts_columns        = "",
        const std::string &fts_tokenizer      = "",
        const std::string &embedding_model    = "",
        const std::string &catalog_opts_json  = ""
    ) const;

    // Execute ailake_compact_json. Returns files_compacted count, or -1 on
    // error/lib-not-ready. -1 sentinel for min_files/target_size_bytes/
    // max_files_per_pass means "use the native default" (4 / 128MiB / 20).
    int64_t compact(
        const std::string &warehouse,
        const std::string &table_name,
        int64_t             min_files          = -1,
        int64_t             target_size_bytes  = -1,
        int64_t             max_files_per_pass = -1,
        bool                deferred           = false,
        const std::string  &ns                 = "default",
        const std::string  &catalog_opts_json  = ""
    ) const;

    // Execute ailake_delete_rows_json. Position/DV delete by row_id within a
    // single named file — different from delete_where (equality delete by
    // column value across the whole table). Returns true on success.
    bool delete_rows(
        const std::string        &warehouse,
        const std::string        &table_name,
        const std::string        &file,
        const std::vector<uint32_t> &row_ids,
        const std::string        &ns = "default",
        const std::string        &catalog_opts_json = ""
    ) const;

    // Execute ailake_add_vector_column_json. Metadata-only schema evolution
    // adding a new vector column. Returns new_schema_id, or -1 on error.
    int32_t add_vector_column(
        const std::string &warehouse,
        const std::string &ns,
        const std::string &table_name,
        const std::string &column,
        int                 dim,
        const std::string &metric               = "cosine",
        const std::string &precision             = "f16",
        bool                pre_normalize        = false,
        int                 hnsw_m               = -1,
        int                 hnsw_ef_construction = -1,
        const std::string &catalog_opts_json     = ""
    ) const;

    // Execute ailake_backfill_vector_column_json. Populates a vector column
    // added via add_vector_column() by running embed_cmd over text_column.
    // Column must already exist (add_vector_column first). Returns true on
    // success.
    bool backfill_vector_column(
        const std::string &warehouse,
        const std::string &table_name,
        const std::string &column,
        const std::string &text_column,
        const std::string &embed_cmd,
        int64_t             batch_size = 512,
        const std::string &ns = "default",
        const std::string &catalog_opts_json = ""
    ) const;

    // Execute ailake_decay_memories_json. Recomputes recency_weight from
    // last_accessed_at across every data file. Returns files_updated, or -1
    // on error.
    int64_t decay_memories(
        const std::string &warehouse,
        const std::string &table_name,
        float               lambda,
        const std::string &ns = "default",
        const std::string &catalog_opts_json = ""
    ) const;

    // Execute ailake_migrate_json. Re-embeds old_column into new_column via
    // embed_cmd (atomic-replace or dual-write-then-cutover). Returns true on
    // success.
    bool migrate(
        const std::string &warehouse,
        const std::string &table_name,
        const std::string &old_column,
        const std::string &new_column,
        const std::string &embed_cmd,
        const std::string &text_column   = "chunk_text",
        const std::string &strategy      = "atomic-replace",
        int64_t             batch_size    = 10000,
        const std::string &model_name    = "",
        const std::string &model_version = "",
        const std::string &ns            = "default",
        const std::string &catalog_opts_json = ""
    ) const;

    // Execute ailake_estimate_json. Pure storage/index-size math, no I/O, no
    // catalog_opts (nothing to resolve — no warehouse/table involved).
    EstimateResult estimate(
        uint64_t rows,
        int      dim,
        int      hnsw_m = 16,
        int      pq_m   = -1
    ) const;

private:
    AilakeLib() = default;

    search_fn_t        search_fn_        = nullptr;
    multimodal_fn_t    multimodal_fn_    = nullptr;
    scan_fn_t          scan_fn_          = nullptr;
    write_fn_t         write_fn_         = nullptr;
    search_text_fn_t   search_text_fn_   = nullptr;
    delete_where_fn_t  delete_where_fn_  = nullptr;
    evolve_schema_fn_t evolve_schema_fn_ = nullptr;
    write_multi_fn_t   write_multi_fn_   = nullptr;
    create_table_fn_t  create_table_fn_  = nullptr;
    compact_fn_t       compact_fn_       = nullptr;
    delete_rows_fn_t   delete_rows_fn_   = nullptr;
    add_vector_column_fn_t     add_vector_column_fn_     = nullptr;
    backfill_vector_column_fn_t backfill_vector_column_fn_ = nullptr;
    decay_memories_fn_t decay_memories_fn_ = nullptr;
    migrate_fn_t       migrate_fn_       = nullptr;
    estimate_fn_t      estimate_fn_      = nullptr;
    free_fn_t          free_fn_          = nullptr;
};

// Parse `catalog_opts_json` (a JSON object literal, e.g.
// '{"catalog":"rest","rest_uri":"..."}') and return its fields re-serialized
// and ready to splice into a request string being hand-built via
// concatenation — a leading comma plus each "key":value pair. Returns "" for
// an empty input string (no catalog override — defaults to the Hadoop
// catalog, unchanged behavior).
//
// Throws std::invalid_argument on malformed JSON or a non-object value —
// callers (the AilakeLib::* methods, in ailake_extension.cpp) let this
// propagate; DuckDB's function dispatcher surfaces it as a normal SQL error,
// so a typo in catalog_opts_json fails visibly instead of silently falling
// back to Hadoop.
std::string catalog_opts_json_fields(const std::string &catalog_opts_json);

// Escape a string value for embedding in a JSON literal.
inline std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    out += '"';
    return out;
}

} // namespace ailake
