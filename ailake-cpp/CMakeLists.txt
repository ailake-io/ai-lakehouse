# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
#
# GPU acceleration note:
#   Default build (AILAKE_CUDA=OFF, AILAKE_ROCM=OFF): CPU-only, no proprietary SDK required.
#   -DAILAKE_CUDA=ON: links against NVIDIA CUDA Toolkit (CUDA::cudart) — proprietary NVIDIA software.
#     CUDA Toolkit license: https://docs.nvidia.com/cuda/eula/
#   -DAILAKE_ROCM=ON (future): links against AMD ROCm — proprietary AMD software.
#     ROCm license: https://rocm.docs.amd.com/en/latest/about/license.html
#   The AI-Lake format itself (this repository) is MIT OR Apache-2.0. The GPU
#   SDKs are third-party proprietary dependencies loaded only when explicitly
#   enabled at build time. Binary distributions of this SDK must not bundle
#   NVIDIA or AMD proprietary libraries.
cmake_minimum_required(VERSION 3.20)
project(ailake-cpp VERSION 0.0.10 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AILAKE_CUDA    "Enable CUDA GPU acceleration"       OFF)
option(AILAKE_EXAMPLES "Build examples"                    ON)
option(AILAKE_TESTS   "Build unit tests"                   ON)

# ── CUDA ─────────────────────────────────────────────────────────────────────
if(AILAKE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  # Default arch list — override with -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
  endif()
endif()

# ── Header-only library ───────────────────────────────────────────────────────
add_library(ailake INTERFACE)
target_include_directories(ailake INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(ailake INTERFACE cxx_std_17)

if(AILAKE_CUDA)
  target_compile_definitions(ailake INTERFACE AILAKE_CUDA_ENABLED)
  target_link_libraries(ailake INTERFACE CUDA::cudart)
endif()

# SIMD: enable AVX2 on x86_64 by default
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(ailake INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
  )
endif()

# ── Compiled catalog + search library (needs file I/O) ───────────────────────
add_library(ailake_catalog STATIC
  src/catalog.cpp
  src/search.cpp
)
target_link_libraries(ailake_catalog PUBLIC ailake)
target_include_directories(ailake_catalog PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(AILAKE_CUDA)
  add_library(ailake_cuda STATIC src/cuda_search.cu)
  target_link_libraries(ailake_cuda PUBLIC ailake CUDA::cudart)
  target_compile_definitions(ailake_cuda PUBLIC AILAKE_CUDA_ENABLED)
endif()

# ── Examples ──────────────────────────────────────────────────────────────────
if(AILAKE_EXAMPLES)
  add_executable(ailake_search examples/search.cpp)
  target_link_libraries(ailake_search PRIVATE ailake_catalog)

  if(AILAKE_CUDA)
    add_executable(ailake_search_gpu examples/search_gpu.cu)
    target_link_libraries(ailake_search_gpu PRIVATE ailake_catalog ailake_cuda CUDA::cudart)
  endif()
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(AILAKE_TESTS)
  enable_testing()

  foreach(_test footer hnsw ivfpq)
    add_executable(ailake_test_${_test} tests/test_${_test}.cpp)
    target_link_libraries(ailake_test_${_test} PRIVATE ailake)
    add_test(NAME test_${_test} COMMAND ailake_test_${_test})
  endforeach()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS ailake ailake_catalog EXPORT ailake-targets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ailake-targets
  FILE ailake-config.cmake
  NAMESPACE ailake::
  DESTINATION lib/cmake/ailake
)
