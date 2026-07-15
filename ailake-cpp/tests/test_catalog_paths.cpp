// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/catalog.hpp — HadoopCatalog::resolve_path() /
// resolve_warehouse_path()'s three path cases (relative, OS-absolute,
// absolute file:// URI). Regression coverage for the file:// double-prefix
// bug: ailake-py's local_catalog_store always writes warehouse_uri as
// file://<absolute path> (Trino Iceberg-connector compatibility), so
// metadata.json's manifest-list can be an absolute file:// URI — the old
// `rel[0] == '/'` check didn't recognize that as absolute (a file:// URI
// starts with 'f'), so it got string-joined onto the warehouse root,
// producing a corrupted double-prefixed path. Confirmed live against a real
// table written by ailake-py before this fix (see ailake-langchain-demo's
// clients/cpp-search).
#include <ailake/ailake.hpp>
#include <cstdio>
#include <string>

static int g_fail = 0;

#define CHECK_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s != %s  (%s vs %s)\n", \
            __FILE__, __LINE__, #a, #b, (a).c_str(), (b).c_str()); \
        ++g_fail; \
    } \
} while(0)

static void test_relative_joins_onto_warehouse() {
    ailake::HadoopCatalog cat("/data");
    CHECK_EQ(cat.resolve_path("ns", "tbl", "file.parquet"), std::string("/data/file.parquet"));
}

static void test_already_os_absolute() {
    ailake::HadoopCatalog cat("/data");
    CHECK_EQ(cat.resolve_path("ns", "tbl", "/other/file.parquet"), std::string("/other/file.parquet"));
}

static void test_absolute_file_uri_scheme_stripped() {
    ailake::HadoopCatalog cat("/data/go_client_test");
    std::string abs_uri =
        "file:///home/thiago/data/go_client_test/default/table/metadata/snap-1.avro";
    std::string want = "/home/thiago/data/go_client_test/default/table/metadata/snap-1.avro";
    CHECK_EQ(cat.resolve_path("default", "table", abs_uri), want);
}

int main() {
    test_relative_joins_onto_warehouse();
    test_already_os_absolute();
    test_absolute_file_uri_scheme_stripped();
    if (g_fail) {
        std::printf("FAILED: %d test(s)\n", g_fail);
        return 1;
    }
    std::printf("ailake test_catalog_paths: all pass\n");
    return 0;
}
