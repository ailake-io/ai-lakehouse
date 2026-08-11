// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// DuckDB extension entry point. Registers ailake_search and ailake_write_batch.

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <nlohmann/json.hpp>

using namespace duckdb;

// Forward declarations from ailake_search.cpp, ailake_search_multimodal.cpp, ailake_write.cpp
void RegisterAilakeSearch(duckdb::ExtensionLoader &loader);
void RegisterAilakeSearchMultimodal(duckdb::ExtensionLoader &loader);
void RegisterAilakeWrite(duckdb::ExtensionLoader &loader);
void RegisterAilakeCreateTable(duckdb::ExtensionLoader &loader);

// ── ailake-jni C-ABI (linked statically — see CMakeLists.txt) ─────────────────
// Same JSON-envelope symbols exported by ailake-jni for Spark/Trino/Flink (JNA);
// here they're resolved at link time instead of via dlopen/dlsym.
extern "C" {
    char *ailake_search_json(const char *);
    char *ailake_search_multimodal_json(const char *);
    char *ailake_scan_json(const char *);
    char *ailake_write_batch_json(const char *);
    char *ailake_search_text_json(const char *);
    char *ailake_delete_where_json(const char *);
    char *ailake_evolve_schema_json(const char *);
    char *ailake_write_batch_multi_json(const char *);
    char *ailake_compact_json(const char *);
    char *ailake_create_table_json(const char *);
    char *ailake_delete_rows_json(const char *);
    char *ailake_add_vector_column_json(const char *);
    char *ailake_backfill_vector_column_json(const char *);
    char *ailake_decay_memories_json(const char *);
    char *ailake_migrate_json(const char *);
    char *ailake_estimate_json(const char *);
    char *ailake_info_json(const char *);
    void  ailake_free_string(char *);
}

// ── AilakeLib implementation ──────────────────────────────────────────────────

namespace ailake {

AilakeLib &AilakeLib::get() {
    static AilakeLib instance;
    return instance;
}

bool AilakeLib::load() {
    if (search_fn_) return true; // already wired up

    search_fn_        = ailake_search_json;
    multimodal_fn_     = ailake_search_multimodal_json;
    scan_fn_           = ailake_scan_json;
    write_fn_          = ailake_write_batch_json;
    search_text_fn_    = ailake_search_text_json;
    delete_where_fn_   = ailake_delete_where_json;
    evolve_schema_fn_  = ailake_evolve_schema_json;
    write_multi_fn_    = ailake_write_batch_multi_json;
    compact_fn_        = ailake_compact_json;
    create_table_fn_   = ailake_create_table_json;
    delete_rows_fn_    = ailake_delete_rows_json;
    add_vector_column_fn_     = ailake_add_vector_column_json;
    backfill_vector_column_fn_ = ailake_backfill_vector_column_json;
    decay_memories_fn_ = ailake_decay_memories_json;
    migrate_fn_        = ailake_migrate_json;
    estimate_fn_       = ailake_estimate_json;
    info_fn_           = ailake_info_json;
    free_fn_           = ailake_free_string;
    return true;
}

std::string catalog_opts_json_fields(const std::string &catalog_opts_json) {
    if (catalog_opts_json.empty()) return "";

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(catalog_opts_json);
    } catch (const std::exception &ex) {
        throw std::invalid_argument(
            std::string("catalog_opts_json is not valid JSON: ") + ex.what()
        );
    }
    if (!j.is_object()) {
        throw std::invalid_argument(
            "catalog_opts_json must be a JSON object, e.g. "
            "'{\"catalog\":\"rest\",\"rest_uri\":\"...\"}'"
        );
    }

    std::string out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        out += "," + nlohmann::json(it.key()).dump() + ":" + it.value().dump();
    }
    return out;
}

std::vector<MultimodalRow> AilakeLib::search_multimodal(
    const std::string                 &warehouse,
    const std::string                 &table_name,
    const std::vector<ModalQueryArg>  &queries,
    int                                top_k,
    const std::string                 &partition_filter,
    const std::string                 &ns,
    const std::string                 &catalog_opts_json
) const {
    if (!multimodal_fn_ || !free_fn_ || queries.empty()) return {};

    std::string queries_json = "[";
    for (size_t i = 0; i < queries.size(); ++i) {
        if (i > 0) queries_json += ',';
        const auto &q = queries[i];
        std::string qvec = "[";
        for (size_t j = 0; j < q.query.size(); ++j) {
            if (j > 0) qvec += ',';
            qvec += std::to_string(q.query[j]);
        }
        qvec += ']';
        queries_json += "{\"col\":" + json_escape(q.col) +
                        ",\"query\":" + qvec +
                        ",\"weight\":" + std::to_string(q.weight) +
                        ",\"dim\":0}";
    }
    queries_json += ']';

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)  +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name) +
        ",\"queries\":"    + queries_json             +
        ",\"top_k\":"      + std::to_string(top_k);
    if (!partition_filter.empty())
        req += ",\"partition_filter\":" + json_escape(partition_filter);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = multimodal_fn_(req.c_str());
    if (!raw) return {};

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return {};
    }
    if (!j.value("ok", false)) {
        // A real backend rejection (e.g. top_k over the cap) must fail visibly, not be
        // folded into the same "no matches" empty result callers can't distinguish.
        throw InvalidInputException(
            "ailake_search_multimodal failed: " + j.value("error", std::string("unknown error"))
        );
    }
    try {
        std::vector<MultimodalRow> rows;
        for (auto &r : j["results"]) {
            rows.push_back({
                r["row_id"].get<int64_t>(),
                r["rrf_score"].get<float>(),
                r["file_path"].get<std::string>()
            });
        }
        return rows;
    } catch (...) {
        return {};
    }
}

std::vector<SearchRow> AilakeLib::search(
    const std::string        &warehouse,
    const std::string        &table_name,
    const std::string        &vec_col,
    const std::vector<float> &query,
    int                       top_k,
    int                       ef_search,
    const std::string        &partition_filter,
    const std::string        &hybrid_text,
    const std::string        &text_column,
    float                     bm25_weight,
    const std::string        &ns,
    const std::string        &catalog_opts_json
) const {
    if (!search_fn_ || !free_fn_ || query.empty()) return {};

    // Build query array JSON
    std::string q_json = "[";
    for (size_t i = 0; i < query.size(); ++i) {
        if (i > 0) q_json += ',';
        q_json += std::to_string(query[i]);
    }
    q_json += ']';

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)   +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"vec_col\":"    + json_escape(vec_col)      +
        ",\"dim\":"        + std::to_string(query.size()) +
        ",\"query\":"      + q_json                   +
        ",\"top_k\":"      + std::to_string(top_k)    +
        ",\"ef_search\":"  + std::to_string(ef_search);
    if (!partition_filter.empty())
        req += ",\"partition_filter\":" + json_escape(partition_filter);
    if (!hybrid_text.empty()) {
        req += ",\"hybrid_text\":"  + json_escape(hybrid_text);
        req += ",\"text_column\":"  + json_escape(text_column);
        req += ",\"bm25_weight\":"  + std::to_string(bm25_weight);
    }
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = search_fn_(req.c_str());
    if (!raw) return {};

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return {};
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_search failed: " + j.value("error", std::string("unknown error"))
        );
    }
    try {
        std::vector<SearchRow> rows;
        for (auto &r : j["results"]) {
            rows.push_back({
                r["row_id"].get<int64_t>(),
                r["distance"].get<float>(),
                r["file_path"].get<std::string>()
            });
        }
        return rows;
    } catch (...) {
        return {};
    }
}

std::vector<SearchRow> AilakeLib::search_text(
    const std::string              &warehouse,
    const std::string              &table_name,
    const std::string              &query_text,
    int                             top_k,
    const std::vector<std::string> &text_columns,
    const std::string              &partition_filter,
    const std::string              &ns,
    const std::string              &catalog_opts_json
) const {
    if (!search_text_fn_ || !free_fn_ || query_text.empty()) return {};

    // Build "text_columns": ["col1","col2",...] JSON array
    std::string cols_json = "[";
    const auto &tc = text_columns.empty()
        ? std::vector<std::string>{"chunk_text"}
        : text_columns;
    for (size_t i = 0; i < tc.size(); ++i) {
        if (i > 0) cols_json += ',';
        cols_json += json_escape(tc[i]);
    }
    cols_json += ']';

    std::string req =
        "{\"warehouse\":"    + json_escape(warehouse)   +
        ",\"namespace\":"    + json_escape(ns)          +
        ",\"table\":"        + json_escape(table_name)  +
        ",\"query_text\":"   + json_escape(query_text)  +
        ",\"top_k\":"        + std::to_string(top_k)    +
        ",\"text_columns\":" + cols_json;
    if (!partition_filter.empty())
        req += ",\"partition_filter\":" + json_escape(partition_filter);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = search_text_fn_(req.c_str());
    if (!raw) return {};

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return {};
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_search_text failed: " + j.value("error", std::string("unknown error"))
        );
    }
    try {
        std::vector<SearchRow> rows;
        for (auto &r : j["results"]) {
            // FTS results carry "score" (higher = more relevant); vector search uses "distance".
            float dist = r.contains("score")
                ? r["score"].get<float>()
                : r.value("distance", 0.0f);
            rows.push_back({
                r["row_id"].get<int64_t>(),
                dist,
                r["file_path"].get<std::string>()
            });
        }
        return rows;
    } catch (...) {
        return {};
    }
}

int64_t AilakeLib::write_batch(
    const std::string              &warehouse,
    const std::string              &ns,
    const std::string              &table_name,
    const std::string              &vec_col,
    int                             dim,
    const std::string              &metric,
    const std::string              &precision,
    const std::vector<int64_t>     &ids,
    const std::vector<std::vector<float>> &embeddings,
    const std::string              &partition_by,
    const std::string              &partition_value,
    const std::string              &partition_fields_json,
    int                             format_version,
    const std::string              &fts_columns_json,
    const std::string              &fts_tokenizer,
    int                             hnsw_m,
    int                             hnsw_ef_construction,
    bool                            pre_normalize,
    bool                            deferred,
    const std::string              &catalog_opts_json
) const {
    if (!write_fn_ || !free_fn_ || ids.empty()) return -1;

    // ids array
    std::string ids_json = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) ids_json += ',';
        ids_json += std::to_string(ids[i]);
    }
    ids_json += ']';

    // embeddings array-of-arrays
    std::string emb_json = "[";
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (i > 0) emb_json += ',';
        emb_json += '[';
        for (size_t j = 0; j < embeddings[i].size(); ++j) {
            if (j > 0) emb_json += ',';
            emb_json += std::to_string(embeddings[i][j]);
        }
        emb_json += ']';
    }
    emb_json += ']';

    std::string req =
        "{\"warehouse\":"      + json_escape(warehouse)  +
        ",\"namespace\":"      + json_escape(ns)          +
        ",\"table\":"          + json_escape(table_name)  +
        ",\"vec_col\":"        + json_escape(vec_col)     +
        ",\"dim\":"            + std::to_string(dim)      +
        ",\"metric\":"         + json_escape(metric)      +
        ",\"precision\":"      + json_escape(precision)   +
        ",\"format_version\":" + std::to_string(format_version) +
        ",\"ids\":"            + ids_json                 +
        ",\"embeddings\":"     + emb_json;
    if (!partition_by.empty())
        req += ",\"partition_by\":"     + json_escape(partition_by);
    if (!partition_value.empty())
        req += ",\"partition_value\":"  + json_escape(partition_value);
    if (!partition_fields_json.empty())
        req += ",\"partition_fields\":" + partition_fields_json;
    if (!fts_columns_json.empty())
        req += ",\"fts_columns\":"      + fts_columns_json;
    if (!fts_tokenizer.empty())
        req += ",\"fts_tokenizer\":"    + json_escape(fts_tokenizer);
    if (hnsw_m > 0)
        req += ",\"hnsw_m\":"           + std::to_string(hnsw_m);
    if (hnsw_ef_construction > 0)
        req += ",\"hnsw_ef_construction\":" + std::to_string(hnsw_ef_construction);
    if (pre_normalize)
        req += ",\"pre_normalize\":true";
    if (deferred)
        req += ",\"deferred\":true";
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = write_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        // A real backend rejection (e.g. NaN/Infinity embeddings) must fail visibly — a
        // batch that used to be silently accepted now gets silently dropped instead
        // (returns -1 with no way for the SQL caller to see why) if left unfixed.
        throw InvalidInputException(
            "ailake_write_batch failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("snapshot_id", int64_t(-1));
}

bool AilakeLib::delete_where(
    const std::string              &warehouse,
    const std::string              &table_name,
    const std::string              &column,
    const std::vector<std::string> &values,
    const std::string              &ns,
    const std::string              &catalog_opts_json
) const {
    if (!delete_where_fn_ || !free_fn_ || values.empty()) return false;

    std::string vals_json = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) vals_json += ',';
        vals_json += json_escape(values[i]);
    }
    vals_json += ']';

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)  +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"column\":"     + json_escape(column)      +
        ",\"values\":"     + vals_json +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = delete_where_fn_(req.c_str());
    if (!raw) return false;

    std::string resp(raw);
    free_fn_(raw);

    try {
        return nlohmann::json::parse(resp).value("ok", false);
    } catch (...) {
        return false;
    }
}

int32_t AilakeLib::evolve_schema(
    const std::string &warehouse,
    const std::string &table_name,
    const std::string &add_columns_json,
    const std::string &rename_columns_json,
    const std::string &ns,
    const std::string &catalog_opts_json
) const {
    if (!evolve_schema_fn_ || !free_fn_) return -1;

    std::string req =
        "{\"warehouse\":"         + json_escape(warehouse)      +
        ",\"namespace\":"         + json_escape(ns)              +
        ",\"table\":"             + json_escape(table_name)     +
        ",\"add_columns\":"       + (add_columns_json.empty()    ? "[]" : add_columns_json) +
        ",\"rename_columns\":"    + (rename_columns_json.empty() ? "[]" : rename_columns_json) +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = evolve_schema_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_evolve_schema failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("new_schema_id", int32_t(-1));
}

int64_t AilakeLib::write_batch_multi(
    const std::string                   &warehouse,
    const std::string                   &ns,
    const std::string                   &table_name,
    const std::vector<int64_t>          &ids,
    const std::vector<VectorColSpecArg> &vector_columns,
    int                                   format_version,
    bool                                  deferred,
    const std::string                    &catalog_opts_json
) const {
    if (!write_multi_fn_ || !free_fn_ || ids.empty() || vector_columns.empty()) return -1;

    std::string ids_json = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) ids_json += ',';
        ids_json += std::to_string(ids[i]);
    }
    ids_json += ']';

    std::string cols_json = "[";
    for (size_t c = 0; c < vector_columns.size(); ++c) {
        if (c > 0) cols_json += ',';
        const auto &vc = vector_columns[c];

        std::string emb_json = "[";
        for (size_t i = 0; i < vc.embeddings.size(); ++i) {
            if (i > 0) emb_json += ',';
            emb_json += '[';
            for (size_t j = 0; j < vc.embeddings[i].size(); ++j) {
                if (j > 0) emb_json += ',';
                emb_json += std::to_string(vc.embeddings[i][j]);
            }
            emb_json += ']';
        }
        emb_json += ']';

        cols_json += "{\"col\":"       + json_escape(vc.col)       +
                     ",\"dim\":"       + std::to_string(vc.dim)    +
                     ",\"metric\":"    + json_escape(vc.metric)    +
                     ",\"precision\":" + json_escape(vc.precision) +
                     ",\"embeddings\":" + emb_json;
        if (!vc.modality.empty())
            cols_json += ",\"modality\":" + json_escape(vc.modality);
        cols_json += '}';
    }
    cols_json += ']';

    std::string req =
        "{\"warehouse\":"       + json_escape(warehouse)  +
        ",\"namespace\":"       + json_escape(ns)          +
        ",\"table\":"           + json_escape(table_name)  +
        ",\"ids\":"             + ids_json                 +
        ",\"vector_columns\":"  + cols_json                +
        ",\"format_version\":"  + std::to_string(format_version);
    if (deferred)
        req += ",\"deferred\":true";
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = write_multi_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_write_batch_multi failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("snapshot_id", int64_t(-1));
}

int64_t AilakeLib::compact(
    const std::string &warehouse,
    const std::string &table_name,
    int64_t             min_files,
    int64_t             target_size_bytes,
    int64_t             max_files_per_pass,
    bool                deferred,
    const std::string  &ns,
    const std::string  &catalog_opts_json
) const {
    if (!compact_fn_ || !free_fn_) return -1;

    std::string req =
        "{\"warehouse\":" + json_escape(warehouse)  +
        ",\"namespace\":" + json_escape(ns)          +
        ",\"table\":"     + json_escape(table_name);
    if (min_files >= 0)
        req += ",\"min_files\":" + std::to_string(min_files);
    if (target_size_bytes >= 0)
        req += ",\"target_size_bytes\":" + std::to_string(target_size_bytes);
    if (max_files_per_pass >= 0)
        req += ",\"max_files_per_pass\":" + std::to_string(max_files_per_pass);
    if (deferred)
        req += ",\"deferred\":true";
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = compact_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_compact failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("files_compacted", int64_t(-1));
}

bool AilakeLib::delete_rows(
    const std::string           &warehouse,
    const std::string           &table_name,
    const std::string           &file,
    const std::vector<uint32_t> &row_ids,
    const std::string           &ns,
    const std::string           &catalog_opts_json
) const {
    if (!delete_rows_fn_ || !free_fn_ || row_ids.empty()) return false;

    std::string ids_json = "[";
    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (i > 0) ids_json += ',';
        ids_json += std::to_string(row_ids[i]);
    }
    ids_json += ']';

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)  +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"file\":"       + json_escape(file)        +
        ",\"row_ids\":"    + ids_json +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = delete_rows_fn_(req.c_str());
    if (!raw) return false;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return false;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_delete_rows failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return true;
}

int32_t AilakeLib::add_vector_column(
    const std::string &warehouse,
    const std::string &ns,
    const std::string &table_name,
    const std::string &column,
    int                 dim,
    const std::string &metric,
    const std::string &precision,
    bool                pre_normalize,
    int                 hnsw_m,
    int                 hnsw_ef_construction,
    const std::string &catalog_opts_json
) const {
    if (!add_vector_column_fn_ || !free_fn_) return -1;

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)  +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"column\":"     + json_escape(column)      +
        ",\"dim\":"        + std::to_string(dim)      +
        ",\"metric\":"     + json_escape(metric)      +
        ",\"precision\":"  + json_escape(precision);
    if (pre_normalize)
        req += ",\"pre_normalize\":true";
    if (hnsw_m >= 0)
        req += ",\"hnsw_m\":" + std::to_string(hnsw_m);
    if (hnsw_ef_construction >= 0)
        req += ",\"hnsw_ef_construction\":" + std::to_string(hnsw_ef_construction);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = add_vector_column_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_add_vector_column failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("new_schema_id", int32_t(-1));
}

bool AilakeLib::backfill_vector_column(
    const std::string &warehouse,
    const std::string &table_name,
    const std::string &column,
    const std::string &text_column,
    const std::string &embed_cmd,
    int64_t             batch_size,
    const std::string &ns,
    const std::string &catalog_opts_json
) const {
    if (!backfill_vector_column_fn_ || !free_fn_) return false;

    std::string req =
        "{\"warehouse\":"    + json_escape(warehouse)    +
        ",\"namespace\":"    + json_escape(ns)             +
        ",\"table\":"        + json_escape(table_name)    +
        ",\"column\":"       + json_escape(column)        +
        ",\"text_column\":"  + json_escape(text_column)   +
        ",\"embed_cmd\":"    + json_escape(embed_cmd);
    if (batch_size > 0)
        req += ",\"batch_size\":" + std::to_string(batch_size);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = backfill_vector_column_fn_(req.c_str());
    if (!raw) return false;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return false;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_backfill_vector_column failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return true;
}

int64_t AilakeLib::decay_memories(
    const std::string &warehouse,
    const std::string &table_name,
    float               lambda,
    const std::string &ns,
    const std::string &catalog_opts_json
) const {
    if (!decay_memories_fn_ || !free_fn_) return -1;

    std::string req =
        "{\"warehouse\":" + json_escape(warehouse)  +
        ",\"namespace\":" + json_escape(ns)          +
        ",\"table\":"     + json_escape(table_name)  +
        ",\"lambda\":"    + std::to_string(lambda) +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = decay_memories_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return -1;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_decay_memories failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return j.value("files_updated", int64_t(-1));
}

bool AilakeLib::migrate(
    const std::string &warehouse,
    const std::string &table_name,
    const std::string &old_column,
    const std::string &new_column,
    const std::string &embed_cmd,
    const std::string &text_column,
    const std::string &strategy,
    int64_t             batch_size,
    const std::string &model_name,
    const std::string &model_version,
    const std::string &ns,
    const std::string &catalog_opts_json
) const {
    if (!migrate_fn_ || !free_fn_) return false;

    std::string req =
        "{\"warehouse\":"    + json_escape(warehouse)    +
        ",\"namespace\":"    + json_escape(ns)             +
        ",\"table\":"        + json_escape(table_name)    +
        ",\"old_column\":"   + json_escape(old_column)    +
        ",\"new_column\":"   + json_escape(new_column)    +
        ",\"text_column\":"  + json_escape(text_column)   +
        ",\"embed_cmd\":"    + json_escape(embed_cmd)     +
        ",\"strategy\":"     + json_escape(strategy);
    if (batch_size > 0)
        req += ",\"batch_size\":" + std::to_string(batch_size);
    if (!model_name.empty())
        req += ",\"model_name\":" + json_escape(model_name);
    if (!model_version.empty())
        req += ",\"model_version\":" + json_escape(model_version);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = migrate_fn_(req.c_str());
    if (!raw) return false;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return false;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_migrate failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return true;
}

EstimateResult AilakeLib::estimate(
    uint64_t rows,
    int      dim,
    int      hnsw_m,
    int      pq_m
) const {
    EstimateResult result;
    if (!estimate_fn_ || !free_fn_) {
        result.error = "ailake_estimate_json not available";
        return result;
    }

    std::string req =
        "{\"rows\":"    + std::to_string(rows)   +
        ",\"dim\":"     + std::to_string(dim)    +
        ",\"hnsw_m\":"  + std::to_string(hnsw_m);
    if (pq_m >= 0)
        req += ",\"pq_m\":" + std::to_string(pq_m);
    req += "}";

    char *raw = estimate_fn_(req.c_str());
    if (!raw) {
        result.error = "ailake_estimate_json returned null";
        return result;
    }

    std::string resp(raw);
    free_fn_(raw);

    try {
        auto j = nlohmann::json::parse(resp);
        if (!j.value("ok", false)) {
            result.error = j.value("error", "unknown error");
            return result;
        }
        for (auto &e : j["estimates"]) {
            EstimateRow row;
            row.mode                  = e.value("mode", "");
            row.vectors_bytes         = e.value("vectors_bytes", int64_t(0));
            row.index_bytes           = e.value("index_bytes", int64_t(0));
            row.total_bytes           = e.value("total_bytes", int64_t(0));
            row.reduction_vs_f32_hnsw = e.value("reduction_vs_f32_hnsw", 0.0);
            row.recall                = e.value("recall", "");
            row.note                  = e.value("note", "");
            result.rows.push_back(std::move(row));
        }
        result.ok = true;
    } catch (const std::exception &ex) {
        result.error = ex.what();
    } catch (...) {
        result.error = "unknown exception in estimate()";
    }
    return result;
}

InfoResult AilakeLib::info(
    const std::string &warehouse,
    const std::string &table_name,
    const std::string &ns,
    const std::string &catalog_opts_json
) const {
    InfoResult result;
    if (!info_fn_ || !free_fn_) {
        result.error = "ailake_info_json not available";
        return result;
    }

    std::string req =
        "{\"warehouse\":" + json_escape(warehouse)  +
        ",\"namespace\":" + json_escape(ns)          +
        ",\"table\":"     + json_escape(table_name)  +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = info_fn_(req.c_str());
    if (!raw) {
        result.error = "ailake_info_json returned null";
        return result;
    }

    std::string resp(raw);
    free_fn_(raw);

    try {
        auto j = nlohmann::json::parse(resp);
        if (!j.value("ok", false)) {
            result.error = j.value("error", "unknown error");
            return result;
        }
        result.table              = j.value("table", "");
        result.location           = j.value("location", "");
        result.vector_column      = j.value("vector_column", "");
        result.vector_dim         = j.value("vector_dim", "");
        result.vector_metric      = j.value("vector_metric", "");
        result.files              = j.value("files", int64_t(0));
        result.indexed_files      = j.value("indexed_files", int64_t(0));
        result.failed_files       = j.value("failed_files", int64_t(0));
        result.foreign_files      = j.value("foreign_files", int64_t(0));
        for (auto &p : j.value("foreign_file_paths", nlohmann::json::array()))
            result.foreign_file_paths.push_back(p.get<std::string>());
        result.rows               = j.value("rows", int64_t(0));
        result.size_bytes         = j.value("size_bytes", int64_t(0));
        if (j.contains("snapshot_id") && !j["snapshot_id"].is_null()) {
            result.has_snapshot_id = true;
            result.snapshot_id     = j["snapshot_id"].get<int64_t>();
        }
        result.ok = true;
    } catch (const std::exception &ex) {
        result.error = ex.what();
    } catch (...) {
        result.error = "unknown exception in info()";
    }
    return result;
}

bool AilakeLib::create_table(
    const std::string &warehouse,
    const std::string &ns,
    const std::string &table_name,
    const std::string &vector_column,
    int                dim,
    const std::string &metric,
    const std::string &precision,
    int                format_version,
    int                hnsw_m,
    int                hnsw_ef_construction,
    bool               pre_normalize,
    const std::string &modality,
    const std::string &partition_by,
    const std::string &partition_value,
    const std::string &partition_column_type,
    const std::string &partition_fields_json,
    const std::string &fts_columns,
    const std::string &fts_tokenizer,
    const std::string &embedding_model,
    const std::string &catalog_opts_json
) const {
    if (!create_table_fn_ || !free_fn_) return false;

    std::string req =
        "{\"warehouse\":"        + json_escape(warehouse)       +
        ",\"namespace\":"        + json_escape(ns)              +
        ",\"table\":"            + json_escape(table_name)      +
        ",\"vector_column\":"    + json_escape(vector_column)   +
        ",\"dim\":"              + std::to_string(dim)          +
        ",\"metric\":"           + json_escape(metric)          +
        ",\"precision\":"        + json_escape(precision);
    if (format_version > 0)
        req += ",\"format_version\":" + std::to_string(format_version);
    if (hnsw_m >= 0)
        req += ",\"hnsw_m\":" + std::to_string(hnsw_m);
    if (hnsw_ef_construction >= 0)
        req += ",\"hnsw_ef_construction\":" + std::to_string(hnsw_ef_construction);
    if (pre_normalize)
        req += ",\"pre_normalize\":true";
    if (!modality.empty())
        req += ",\"modality\":" + json_escape(modality);
    if (!partition_by.empty())
        req += ",\"partition_by\":" + json_escape(partition_by);
    if (!partition_value.empty())
        req += ",\"partition_value\":" + json_escape(partition_value);
    if (!partition_column_type.empty())
        req += ",\"partition_column_type\":" + json_escape(partition_column_type);
    if (!partition_fields_json.empty())
        req += ",\"partition_fields\":" + partition_fields_json;
    if (!fts_columns.empty())
        req += ",\"fts_columns\":" + json_escape(fts_columns);
    if (!fts_tokenizer.empty())
        req += ",\"fts_tokenizer\":" + json_escape(fts_tokenizer);
    if (!embedding_model.empty())
        req += ",\"embedding_model\":" + json_escape(embedding_model);
    req += catalog_opts_json_fields(catalog_opts_json);
    req += "}";

    char *raw = create_table_fn_(req.c_str());
    if (!raw) return false;

    std::string resp(raw);
    free_fn_(raw);

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(resp);
    } catch (...) {
        return false;
    }
    if (!j.value("ok", false)) {
        throw InvalidInputException(
            "ailake_create_table failed: " + j.value("error", std::string("unknown error"))
        );
    }
    return true;
}

ScanResult AilakeLib::scan(
    const std::string        &warehouse,
    const std::string        &table_name,
    const std::string        &vec_col,
    const std::vector<float> &query,
    int                       top_k,
    int                       ef_search,
    const std::string        &ns,
    const std::string        &catalog_opts_json
) const {
    ScanResult result;
    if (!scan_fn_ || !free_fn_ || query.empty()) {
        result.error = "ailake_scan_json not available";
        return result;
    }

    // Build query array JSON
    std::string q_json = "[";
    for (size_t i = 0; i < query.size(); ++i) {
        if (i > 0) q_json += ',';
        q_json += std::to_string(query[i]);
    }
    q_json += ']';

    std::string req =
        "{\"warehouse\":"  + json_escape(warehouse)   +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"vec_col\":"    + json_escape(vec_col)      +
        ",\"dim\":"        + std::to_string(query.size()) +
        ",\"query\":"      + q_json                   +
        ",\"top_k\":"      + std::to_string(top_k)    +
        ",\"ef_search\":"  + std::to_string(ef_search) +
        catalog_opts_json_fields(catalog_opts_json) +
        "}";

    char *raw = scan_fn_(req.c_str());
    if (!raw) {
        result.error = "ailake_scan_json returned null";
        return result;
    }

    std::string resp(raw);
    free_fn_(raw);

    try {
        auto j = nlohmann::json::parse(resp);
        if (!j.value("ok", false)) {
            result.error = j.value("error", "unknown error");
            return result;
        }

        result.num_rows = j.value("num_rows", int64_t(0));
        const auto &cols_json = j["columns"];

        for (auto &schema_entry : j["schema"]) {
            const std::string col_name = schema_entry["name"].get<std::string>();
            const std::string col_type = schema_entry["type"].get<std::string>();

            ScanColumn col;
            col.name = col_name;
            col.is_null.resize(result.num_rows, false);

            const auto &arr = cols_json[col_name];

            if (col_type == "int64") {
                col.type = ScanColType::INT64;
                col.int_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.int_vals.push_back(0);
                    } else {
                        col.int_vals.push_back(arr[i].get<int64_t>());
                    }
                }
            } else if (col_type == "float32") {
                col.type = ScanColType::FLOAT32;
                col.float_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.float_vals.push_back(0.0f);
                    } else {
                        col.float_vals.push_back(arr[i].get<float>());
                    }
                }
            } else if (col_type == "float64") {
                col.type = ScanColType::FLOAT64;
                col.double_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.double_vals.push_back(0.0);
                    } else {
                        col.double_vals.push_back(arr[i].get<double>());
                    }
                }
            } else if (col_type == "utf8") {
                col.type = ScanColType::VARCHAR;
                col.str_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.str_vals.push_back("");
                    } else {
                        col.str_vals.push_back(arr[i].get<std::string>());
                    }
                }
            } else if (col_type == "bool") {
                col.type = ScanColType::BOOL;
                col.bool_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.bool_vals.push_back(false);
                    } else {
                        col.bool_vals.push_back(arr[i].get<bool>());
                    }
                }
            } else if (col_type == "list_float32") {
                col.type = ScanColType::LIST_FLOAT32;
                col.list_vals.reserve(result.num_rows);
                for (size_t i = 0; i < (size_t)result.num_rows; ++i) {
                    if (arr[i].is_null()) {
                        col.is_null[i] = true;
                        col.list_vals.push_back({});
                    } else {
                        std::vector<float> floats;
                        floats.reserve(arr[i].size());
                        for (auto &f : arr[i]) {
                            floats.push_back(f.is_null() ? 0.0f : f.get<float>());
                        }
                        col.list_vals.push_back(std::move(floats));
                    }
                }
            } else {
                // Unknown type — skip column.
                continue;
            }

            result.columns.push_back(std::move(col));
        }

        result.ok = true;
    } catch (const std::exception &ex) {
        result.error = ex.what();
    } catch (...) {
        result.error = "unknown exception in scan()";
    }
    return result;
}

} // namespace ailake

// ── Extension entry points ────────────────────────────────────────────────────

// Forward declarations
void RegisterAilakeScan(duckdb::ExtensionLoader &loader);
void RegisterAilakeSearchText(duckdb::ExtensionLoader &loader);
void RegisterAilakeDeleteWhere(duckdb::ExtensionLoader &loader);
void RegisterAilakeEvolveSchema(duckdb::ExtensionLoader &loader);
void RegisterAilakeWriteBatchMulti(duckdb::ExtensionLoader &loader);
void RegisterAilakeCompact(duckdb::ExtensionLoader &loader);
void RegisterAilakeVersion(duckdb::ExtensionLoader &loader);
void RegisterAilakeDeleteRows(duckdb::ExtensionLoader &loader);
void RegisterAilakeAddVectorColumn(duckdb::ExtensionLoader &loader);
void RegisterAilakeBackfillVectorColumn(duckdb::ExtensionLoader &loader);
void RegisterAilakeDecayMemories(duckdb::ExtensionLoader &loader);
void RegisterAilakeMigrate(duckdb::ExtensionLoader &loader);
void RegisterAilakeEstimate(duckdb::ExtensionLoader &loader);
void RegisterAilakeInfo(duckdb::ExtensionLoader &loader);

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ailake, loader) {
    // ailake-jni is statically linked into this binary — always succeeds.
    ailake::AilakeLib::get().load();

    RegisterAilakeSearch(loader);
    RegisterAilakeSearchMultimodal(loader);
    RegisterAilakeWrite(loader);
    RegisterAilakeCreateTable(loader);
    RegisterAilakeScan(loader);
    RegisterAilakeSearchText(loader);
    RegisterAilakeWriteBatchMulti(loader);
    RegisterAilakeCompact(loader);
    RegisterAilakeDeleteWhere(loader);
    RegisterAilakeEvolveSchema(loader);
    RegisterAilakeDeleteRows(loader);
    RegisterAilakeAddVectorColumn(loader);
    RegisterAilakeBackfillVectorColumn(loader);
    RegisterAilakeDecayMemories(loader);
    RegisterAilakeMigrate(loader);
    RegisterAilakeEstimate(loader);
    RegisterAilakeInfo(loader);
    RegisterAilakeVersion(loader);
}

} // extern "C"
