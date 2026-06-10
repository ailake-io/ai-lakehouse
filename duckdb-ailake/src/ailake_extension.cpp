// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
//
// DuckDB extension entry point. Registers ailake_search and ailake_write_batch.

#include "ailake_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension_util.hpp"

#include <nlohmann/json.hpp>

using namespace duckdb;

// Forward declarations from ailake_search.cpp and ailake_write.cpp
void RegisterAilakeSearch(DatabaseInstance &db);
void RegisterAilakeWrite(DatabaseInstance &db);

// ── AilakeLib implementation ──────────────────────────────────────────────────

namespace ailake {

AilakeLib &AilakeLib::get() {
    static AilakeLib instance;
    return instance;
}

bool AilakeLib::load(const std::string &lib_path) {
    if (search_fn_) return true; // already resolved

    // Try path-based load first
    const char *path = lib_path.empty() ? AILAKE_LIB_NAME : lib_path.c_str();
    void *h = AILAKE_DLOPEN(path);

    // If dlopen failed, symbols may already be in the global table (pre-loaded
    // via ctypes.CDLL(..., RTLD_GLOBAL)). On POSIX, dlsym(RTLD_DEFAULT, ...)
    // searches the global namespace.
#ifndef _WIN32
    void *sym_handle = h ? h : RTLD_DEFAULT;
#else
    if (!h) return false;
    void *sym_handle = h;
#endif

    auto s = reinterpret_cast<search_fn_t>(AILAKE_DLSYM(sym_handle, "ailake_search_json"));
    auto w = reinterpret_cast<write_fn_t> (AILAKE_DLSYM(sym_handle, "ailake_write_batch_json"));
    auto f = reinterpret_cast<free_fn_t>  (AILAKE_DLSYM(sym_handle, "ailake_free_string"));

    if (!s || !w || !f) {
        if (h) AILAKE_DLCLOSE(h);
        return false;
    }

    handle_    = h;   // nullptr when resolved via RTLD_DEFAULT — is_ready() uses search_fn_
    search_fn_ = s;
    write_fn_  = w;
    free_fn_   = f;
    return true;
}

std::vector<SearchRow> AilakeLib::search(
    const std::string        &warehouse,
    const std::string        &table_name,
    const std::string        &vec_col,
    const std::vector<float> &query,
    int                       top_k,
    int                       ef_search
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
        ",\"namespace\":\"default\""                  +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"vec_col\":"    + json_escape(vec_col)      +
        ",\"dim\":"        + std::to_string(query.size()) +
        ",\"query\":"      + q_json                   +
        ",\"top_k\":"      + std::to_string(top_k)    +
        ",\"ef_search\":"  + std::to_string(ef_search) +
        "}";

    char *raw = search_fn_(req.c_str());
    if (!raw) return {};

    std::string resp(raw);
    free_fn_(raw);

    try {
        auto j = nlohmann::json::parse(resp);
        if (!j.value("ok", false)) return {};

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

int64_t AilakeLib::write_batch(
    const std::string              &warehouse,
    const std::string              &ns,
    const std::string              &table_name,
    const std::string              &vec_col,
    int                             dim,
    const std::string              &metric,
    const std::string              &precision,
    const std::vector<int64_t>     &ids,
    const std::vector<std::vector<float>> &embeddings
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
        "{\"warehouse\":"  + json_escape(warehouse)  +
        ",\"namespace\":"  + json_escape(ns)          +
        ",\"table\":"      + json_escape(table_name)  +
        ",\"vec_col\":"    + json_escape(vec_col)      +
        ",\"dim\":"        + std::to_string(dim)      +
        ",\"metric\":"     + json_escape(metric)      +
        ",\"precision\":"  + json_escape(precision)   +
        ",\"ids\":"        + ids_json                 +
        ",\"embeddings\":" + emb_json                 +
        "}";

    char *raw = write_fn_(req.c_str());
    if (!raw) return -1;

    std::string resp(raw);
    free_fn_(raw);

    try {
        auto j = nlohmann::json::parse(resp);
        if (!j.value("ok", false)) return -1;
        return j.value("snapshot_id", int64_t(-1));
    } catch (...) {
        return -1;
    }
}

} // namespace ailake

// ── Extension entry points ────────────────────────────────────────────────────

extern "C" {

DUCKDB_EXTENSION_API void ailake_init(DatabaseInstance &db) {
    // Try to load libailake_jni.so from environment/library path.
    // Non-fatal: functions still register and return clear errors at call time.
    ailake::AilakeLib::get().load();

    RegisterAilakeSearch(db);
    RegisterAilakeWrite(db);
}

DUCKDB_EXTENSION_API const char *ailake_version() {
    return DuckDB::LibraryVersion();
}

} // extern "C"
