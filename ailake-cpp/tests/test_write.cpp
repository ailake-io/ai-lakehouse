// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/write.hpp — Phase O write-op delegation.
//
// Tests cover:
//   - PartitionDef / SchemaField struct construction
//   - AddColumnReq / RenameColumnReq struct construction
//   - shell_quote escaping (embedded single-quotes)
//   - resolve_bin: AILAKE_BIN env override
//   - delete_where / evolve_schema no-op on empty inputs
//   - Integration tests guarded by AILAKE_BIN + AILAKE_FIXTURE env vars
#include <ailake/ailake.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) do { \
    if (!((a) == (b))) { \
        std::fprintf(stderr, "FAIL %s:%d  expected [%s] == [%s]\n", \
                     __FILE__, __LINE__, #a, #b); \
        ++g_fail; \
    } \
} while(0)

// ── PartitionDef / SchemaField ────────────────────────────────────────────────

static void test_partition_def_fields() {
    ailake::PartitionDef pd;
    pd.column      = "agent_id";
    pd.transform   = "identity";
    pd.column_type = "string";
    CHECK_EQ(pd.column,      "agent_id");
    CHECK_EQ(pd.transform,   "identity");
    CHECK_EQ(pd.column_type, "string");
}

static void test_schema_field_defaults() {
    ailake::SchemaField sf;
    CHECK_EQ(sf.id, 0);
    CHECK_EQ(sf.required, false);
    CHECK(sf.name.empty());
    CHECK(sf.type.empty());
}

static void test_schema_field_set() {
    ailake::SchemaField sf;
    sf.id       = 1;
    sf.name     = "doc_id";
    sf.type     = "string";
    sf.required = false;
    CHECK_EQ(sf.id,   1);
    CHECK_EQ(sf.name, "doc_id");
    CHECK_EQ(sf.type, "string");
}

// ── AddColumnReq / RenameColumnReq ───────────────────────────────────────────

static void test_add_column_req() {
    ailake::AddColumnReq r;
    r.name            = "score";
    r.type            = "float";
    r.initial_default = "0.0";
    CHECK_EQ(r.name,            "score");
    CHECK_EQ(r.type,            "float");
    CHECK_EQ(r.initial_default, "0.0");
}

static void test_rename_column_req() {
    ailake::RenameColumnReq r;
    r.from = "old_col";
    r.to   = "new_col";
    CHECK_EQ(r.from, "old_col");
    CHECK_EQ(r.to,   "new_col");
}

// ── shell_quote ───────────────────────────────────────────────────────────────

static void test_shell_quote_simple() {
    CHECK_EQ(ailake::detail::shell_quote("hello"),  "'hello'");
    CHECK_EQ(ailake::detail::shell_quote(""),        "''");
}

static void test_shell_quote_with_single_quote() {
    // "it's" → 'it'\''s'
    CHECK_EQ(ailake::detail::shell_quote("it's"), "'it'\\''s'");
}

static void test_shell_quote_spaces() {
    CHECK_EQ(ailake::detail::shell_quote("a b c"), "'a b c'");
}

// ── resolve_bin ───────────────────────────────────────────────────────────────

static void test_resolve_bin_env() {
    const char* orig = std::getenv("AILAKE_BIN");
    std::string saved = orig ? orig : "";
    setenv("AILAKE_BIN", "/custom/ailake", 1);
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "/custom/ailake");
    if (orig) setenv("AILAKE_BIN", saved.c_str(), 1);
    else      unsetenv("AILAKE_BIN");
}

static void test_resolve_bin_default_when_no_env() {
    // unsetenv() mutates the process environment permanently — save/restore so
    // later tests (including the integration ones) still see a real AILAKE_BIN
    // if the shell provided one.
    const char* orig = std::getenv("AILAKE_BIN");
    std::string saved = orig ? orig : "";
    unsetenv("AILAKE_BIN");
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "ailake");
    if (orig) setenv("AILAKE_BIN", saved.c_str(), 1);
}

// ── No-op paths ───────────────────────────────────────────────────────────────

static void test_delete_where_empty_values_noop() {
    // Empty values vector → no CLI call, no error.
    ailake::delete_where("/tmp/test", "default.table", "doc_id", {});
    // If we get here without exception: pass.
    CHECK(true);
}

static void test_evolve_schema_empty_noop() {
    int id = ailake::evolve_schema("/tmp/test", "default.table", {}, {});
    CHECK_EQ(id, 0);
}

// ── TableInfo new fields ──────────────────────────────────────────────────────

static void test_table_info_format_version_default() {
    ailake::TableInfo info;
    CHECK_EQ(info.format_version, 2);
}

static void test_table_info_partition_fields_empty_default() {
    ailake::TableInfo info;
    CHECK(info.partition_fields.empty());
}

static void test_table_info_schema_fields_empty_default() {
    ailake::TableInfo info;
    CHECK(info.schema_fields.empty());
}

// ── VectorColSpec / CompactOptions ───────────────────────────────────────────

static void test_vector_col_spec_fields() {
    ailake::VectorColSpec spec;
    spec.column   = "image_embedding";
    spec.dim      = 512;
    spec.metric   = "euclidean";
    spec.modality = "image";
    CHECK_EQ(spec.column,   "image_embedding");
    CHECK_EQ(spec.dim,      512);
    CHECK_EQ(spec.metric,   "euclidean");
    CHECK_EQ(spec.modality, "image");
}

static void test_compact_options_fields() {
    ailake::CompactOptions opts;
    opts.target_size        = 1024;
    opts.min_files          = 2;
    opts.max_files_per_pass = 10;
    opts.deferred            = true;
    CHECK_EQ(opts.target_size,        1024);
    CHECK_EQ(opts.min_files,          2);
    CHECK_EQ(opts.max_files_per_pass, 10);
    CHECK_EQ(opts.deferred,           true);
}

static void test_write_batch_multi_empty_cols_throws() {
    bool threw = false;
    try {
        ailake::write_batch_multi("/tmp/test", "default.table", "/tmp/x.parquet", {});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ── Integration: multi-column write + compact (own temp warehouse, requires
// only AILAKE_BIN — writes testdata/multimodal_fixture.parquet, not gated on
// the shared AILAKE_FIXTURE) ─────────────────────────────────────────────────

static std::string make_temp_dir() {
    char tmpl[] = "/tmp/ailake_cpp_test_XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) throw std::runtime_error("mkdtemp failed");
    return std::string(dir);
}

// Resolve testdata/ relative to this source file (not CWD) — ctest runs
// binaries from the build dir, so a bare "testdata/..." relative path
// wouldn't resolve there the way it does for `go test` (which runs from the
// package source dir).
static std::string fixture_path() {
    std::string here = __FILE__; // .../ailake-cpp/tests/test_write.cpp
    auto slash = here.find_last_of('/');
    return here.substr(0, slash) + "/../testdata/multimodal_fixture.parquet";
}

static void test_integration_write_batch_multi() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::write_batch_multi(warehouse, "default.media", fixture_path(), {
            {"embedding", 4, "cosine", ""},
            {"image_embedding", 2, "cosine", "image"},
        });

        ailake::HadoopCatalog cat(warehouse);
        std::vector<ailake::ModalQuery> queries = {
            {"embedding", {0.1f, 0.2f, 0.3f, 0.4f}, 0.7f},
            {"image_embedding", {0.5f, 0.6f}, 0.3f},
        };
        ailake::SearchOptions opts; opts.top_k = 3;
        auto results = ailake::search_multimodal(cat, "default", "media", queries, opts);
        CHECK_EQ(results.size(), (size_t)3);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration write_batch_multi: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_compact() {
    const char* bin = std::getenv("AILAKE_BIN");
    if (!bin) { std::fprintf(stdout, "SKIP: AILAKE_BIN not set\n"); return; }

    try {
        std::string warehouse = make_temp_dir();
        ailake::WriteBatchOptions wopts; wopts.vec_col = "embedding";
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);
        ailake::write_batch(warehouse, "default.docs", fixture_path(), wopts);

        ailake::CompactOptions copts; copts.min_files = 2;
        int n = ailake::compact(warehouse, "default.docs", copts);
        CHECK_EQ(n, 1);

        ailake::HadoopCatalog cat(warehouse);
        float q[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        ailake::SearchOptions sopts; sopts.top_k = 20;
        auto results = ailake::search(cat, "default", "docs", q, 4, sopts);
        CHECK_EQ(results.size(), (size_t)12);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration compact: %s\n", e.what());
        ++g_fail;
    }
}

// ── Integration (AILAKE_BIN + AILAKE_FIXTURE required) ───────────────────────

static void test_integration_load_table() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    if (!fixture) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE not set\n"); return; }

    ailake::HadoopCatalog cat(fixture);
    ailake::TableInfo info = cat.load_table("default", "table");

    CHECK(!info.vector_column.empty());
    CHECK(!info.vector_dim.empty());
    CHECK(info.format_version == 2 || info.format_version == 3);
    // Schema fields should have been populated for tables with a valid schema.
    // (Not strictly required for old fixture tables, so just check non-negative id.)
    for (const auto& sf : info.schema_fields) {
        CHECK(sf.id >= 0);
        CHECK(!sf.name.empty());
    }
}

static void test_integration_delete_where() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    const char* bin     = std::getenv("AILAKE_BIN");
    if (!fixture || !bin) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE/AILAKE_BIN not set\n"); return; }

    setenv("AILAKE_BIN", bin, 1);
    // Delete a non-existent value — zero-row delete is always valid.
    try {
        ailake::delete_where(fixture, "default.table", "document_id", {"__nonexistent_cpp__"});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration delete_where: %s\n", e.what());
        ++g_fail;
    }
}

static void test_integration_evolve_schema() {
    const char* fixture = std::getenv("AILAKE_FIXTURE");
    const char* bin     = std::getenv("AILAKE_BIN");
    if (!fixture || !bin) { std::fprintf(stdout, "SKIP: AILAKE_FIXTURE/AILAKE_BIN not set\n"); return; }

    setenv("AILAKE_BIN", bin, 1);
    try {
        ailake::AddColumnReq ac;
        ac.name            = "_cpp_test_col";
        ac.type            = "string";
        ac.initial_default = "\"\"";
        int id = ailake::evolve_schema(fixture, "default.table", {ac}, {});
        CHECK(id >= -1); // -1 if parse failed, ≥0 on success
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL integration evolve_schema: %s\n", e.what());
        ++g_fail;
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_partition_def_fields();
    test_schema_field_defaults();
    test_schema_field_set();
    test_add_column_req();
    test_rename_column_req();
    test_shell_quote_simple();
    test_shell_quote_with_single_quote();
    test_shell_quote_spaces();
    test_resolve_bin_env();
    test_resolve_bin_default_when_no_env();
    test_delete_where_empty_values_noop();
    test_evolve_schema_empty_noop();
    test_table_info_format_version_default();
    test_table_info_partition_fields_empty_default();
    test_table_info_schema_fields_empty_default();
    test_vector_col_spec_fields();
    test_compact_options_fields();
    test_write_batch_multi_empty_cols_throws();

    test_integration_load_table();
    test_integration_write_batch_multi();
    test_integration_compact();
    test_integration_delete_where();
    test_integration_evolve_schema();

    if (g_fail > 0) {
        std::fprintf(stderr, "%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::fprintf(stdout, "All write tests passed.\n");
    return 0;
}
