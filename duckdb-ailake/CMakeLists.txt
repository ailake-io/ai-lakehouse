# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
cmake_minimum_required(VERSION 3.20)
project(duckdb_ailake VERSION 0.1.8 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── DuckDB version (must match the duckdb Python package used in tests) ────────
set(DUCKDB_VERSION "v1.5.3" CACHE STRING "DuckDB version to build against")

include(FetchContent)

# ── Build real DuckDB static lib (add_subdirectory, not headers-only) ─────────
# Out-of-tree extensions must link against a real `duckdb_static` built from the
# exact same source/version — resolving duckdb::* symbols from the host process
# at LOAD time instead (the old approach here) works only when the host already
# exposes those symbols globally (true for the Python wheel, false for the
# official duckdb CLI binary — see README "Known limitation", now fixed by this).
# Trim unrelated build targets for a faster configure/build; EXTENSION_STATIC_BUILD
# and BUILD_MAIN_DUCKDB_LIBRARY already default TRUE, which is what produces the
# `duckdb_static` target this extension links against below.
set(BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHELL OFF CACHE BOOL "" FORCE)
set(DISABLE_BUILTIN_EXTENSIONS TRUE CACHE BOOL "" FORCE)

FetchContent_Declare(duckdb
    GIT_REPOSITORY https://github.com/duckdb/duckdb.git
    GIT_TAG        ${DUCKDB_VERSION}
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(duckdb)

# ── nlohmann/json (single header, response parsing) ───────────────────────────
FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Corrosion (link ailake-jni statically — no dlopen, see include/ailake_extension.hpp) ──
FetchContent_Declare(Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG        v0.6.1
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../ailake-jni/Cargo.toml
    CRATE_TYPES   staticlib
)

# ── Extension shared library ──────────────────────────────────────────────────
add_library(duckdb_ailake MODULE
    src/ailake_extension.cpp
    src/ailake_search.cpp
    src/ailake_search_multimodal.cpp
    src/ailake_search_text.cpp
    src/ailake_write.cpp
    src/ailake_scan_table.cpp
    src/ailake_delete.cpp
    src/ailake_evolve_schema.cpp
    src/ailake_write_batch_multi.cpp
    src/ailake_compact.cpp
    src/ailake_create_table.cpp
)

target_include_directories(duckdb_ailake PRIVATE
    include
    ${duckdb_SOURCE_DIR}/src/include
)

# duckdb_static (built by src/CMakeLists.txt) is assembled from that file's own
# local ALL_OBJECT_FILES scope, which never receives the objects added by
# extension/CMakeLists.txt (a separate add_subdirectory, separate variable
# scope). Two consequences, both "undefined symbol" at dlopen time rather than
# at link time (undefined refs are legal in a MODULE/shared lib, resolved
# against the host process later — Python's duckdb, in our case):
#  1. ExtensionHelper::LoadAllExtensions's real (non-dummy) definition lives in
#     duckdb_generated_extension_loader (compiled from
#     extension/generated_extension_loader.cpp.in) — declared and called from
#     duckdb_static's own database.cpp but never *defined* within duckdb_static.a.
#  2. extension/extension_config.cmake unconditionally statically links
#     `core_functions` and `parquet` (duckdb_extension_load(...), independent of
#     DISABLE_BUILTIN_EXTENSIONS above) — the generated loader body references
#     CoreFunctionsExtension/ParquetExtension by type (vtable emitted in their
#     own TUs), so their static targets (core_functions_extension,
#     parquet_extension) must also be linked in or their vtable symbols dangle.
# Official DuckDB builds (CLI/Python wheel) link all of these explicitly
# alongside duckdb_static for the same reason; confirmed against a real local
# build + `LOAD` in Python's duckdb (see CHANGELOG).
target_link_libraries(duckdb_ailake PRIVATE
    nlohmann_json::nlohmann_json
    duckdb_static
    duckdb_generated_extension_loader
    core_functions_extension
    parquet_extension
    ailake_jni
)

# ABI note: match the ABI of the host duckdb Python package.
# Anaconda/pip wheels on modern Linux use C++11 ABI (default, no flag needed).
# Older manylinux wheels use _GLIBCXX_USE_CXX11_ABI=0 — add back if loading fails
# with "undefined symbol: ...B5cxx11..." vs "...ToStringEv" (without B5cxx11).

set_target_properties(duckdb_ailake PROPERTIES
    PREFIX      ""
    OUTPUT_NAME "ailake"
    SUFFIX      ".duckdb_extension"
)

# ── Post-build: append DuckDB 1.0+ extension metadata ────────────────────────
# DuckDB 1.0+ requires a 512-byte metadata block at the end of every extension
# binary (8×32-byte fields + 256-byte signature), even for unsigned extensions.
# Without it DuckDB refuses to load the file regardless of allow_unsigned_extensions.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(TARGET duckdb_ailake POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/append_extension_metadata.py"
            "$<TARGET_FILE:duckdb_ailake>"
            "${DUCKDB_VERSION}"
            "${CMAKE_PROJECT_VERSION}"
    COMMENT "Appending DuckDB extension metadata (512 bytes)"
    VERBATIM
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS duckdb_ailake DESTINATION lib/duckdb_extensions)
