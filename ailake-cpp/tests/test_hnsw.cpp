// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/hnsw.hpp — flat_search on directly constructed HnswIndex.
#include <ailake/hnsw.hpp>
#include <cstdio>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static void test_flat_search_euclidean() {
    ailake::HnswIndex idx;
    idx.dim    = 2;
    idx.metric = ailake::Metric::Euclidean;

    // 3 vectors at [0,0], [1,0], [2,0]
    idx.flat_vecs = {0.f, 0.f,  1.f, 0.f,  2.f, 0.f};
    idx.row_ids   = {10, 20, 30};

    float q[] = {1.1f, 0.f}; // nearest = [1,0] → row_id 20
    auto hits = ailake::flat_search(idx, q, 2);

    CHECK_EQ(hits.size(), 2u);
    CHECK_EQ(hits[0].row_id, 20u);
    CHECK(hits[0].distance < hits[1].distance);
}

static void test_flat_search_cosine() {
    ailake::HnswIndex idx;
    idx.dim    = 2;
    idx.metric = ailake::Metric::Cosine;

    // 3 unit vectors at angles 0°, 45°, 90°
    idx.flat_vecs = {1.f, 0.f,   0.707f, 0.707f,   0.f, 1.f};
    idx.row_ids   = {1, 2, 3};

    float q[] = {1.f, 0.f}; // exact match with entry 0
    auto hits = ailake::flat_search(idx, q, 1);

    CHECK_EQ(hits.size(), 1u);
    CHECK_EQ(hits[0].row_id, 1u);
    CHECK(hits[0].distance < 1e-5f);
}

static void test_flat_search_top_k_capped() {
    ailake::HnswIndex idx;
    idx.dim    = 1;
    idx.metric = ailake::Metric::Euclidean;
    idx.flat_vecs = {0.f, 1.f, 2.f, 3.f, 4.f};
    idx.row_ids   = {10, 20, 30, 40, 50};

    float q[] = {0.5f};
    auto hits = ailake::flat_search(idx, q, 2);
    CHECK_EQ(hits.size(), 2u);
}

static void test_flat_search_empty() {
    ailake::HnswIndex idx;
    idx.dim    = 4;
    idx.metric = ailake::Metric::Cosine;
    float q[] = {1.f, 0.f, 0.f, 0.f};
    auto hits = ailake::flat_search(idx, q, 5);
    CHECK(hits.empty());
}

int main() {
    test_flat_search_euclidean();
    test_flat_search_cosine();
    test_flat_search_top_k_capped();
    test_flat_search_empty();
    if (g_fail) {
        std::printf("FAILED: %d test(s)\n", g_fail);
        return 1;
    }
    std::printf("ailake test_hnsw: all pass\n");
    return 0;
}
