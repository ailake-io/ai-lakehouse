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

target_link_libraries(duckdb_ailake PRIVATE
    nlohmann_json::nlohmann_json
    duckdb_static
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
