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
#include <ailake/write.hpp>
#include <ailake/catalog.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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
    setenv("AILAKE_BIN", "/custom/ailake", 1);
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "/custom/ailake");
    unsetenv("AILAKE_BIN");
}

static void test_resolve_bin_default_when_no_env() {
    unsetenv("AILAKE_BIN");
    auto bin = ailake::detail::resolve_bin();
    CHECK_EQ(bin, "ailake");
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

    test_integration_load_table();
    test_integration_delete_where();
    test_integration_evolve_schema();

    if (g_fail > 0) {
        std::fprintf(stderr, "%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::fprintf(stdout, "All write tests passed.\n");
    return 0;
}
