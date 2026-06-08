// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/ivfpq.hpp — ivfpq_search on a directly constructed IvfPqIndex.
#include <ailake/ivfpq.hpp>
#include <cstdio>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static void test_ivfpq_returns_nearest() {
    // dim=4, m=2, k=2, sub_dim=2
    // 2 coarse cells; query near cell0
    ailake::IvfPqIndex idx;
    idx.dim           = 4;
    idx.metric        = ailake::Metric::Euclidean;
    idx.config.nlist  = 2;
    idx.config.nprobe = 2;
    idx.config.pq_m   = 2;
    idx.config.pq_k   = 2;
    idx.config.max_iter = 10;

    // coarse: cell0=[0,0,0,0], cell1=[10,10,10,10]
    idx.coarse_centroids = {{0.f,0.f,0.f,0.f}, {10.f,10.f,10.f,10.f}};

    // PQ: m=2, k=2; sub-codebook j has code0=[0,0], code1=[5,5]
    idx.pq.m = 2;
    idx.pq.k = 2;
    idx.pq.centroids = {
        {0.f,0.f}, {5.f,5.f},  // sub 0
        {0.f,0.f}, {5.f,5.f}   // sub 1
    };

    // cell0: 1 vector, code=[0,0] (→ approx [0,0,0,0]) row_id=100
    // cell1: 1 vector, code=[1,1] (→ approx [5,5,5,5]) row_id=200
    idx.inv_row_ids = {{100}, {200}};
    idx.inv_codes   = {{0, 0}, {1, 1}};

    float q[] = {0.1f, 0.1f, 0.1f, 0.1f}; // near cell0 / code [0,0]
    auto hits = ailake::ivfpq_search(idx, q, 1);

    CHECK_EQ(hits.size(), 1u);
    CHECK_EQ(hits[0].row_id, 100u);
    CHECK(hits[0].distance < 1.f);
}

static void test_ivfpq_top_k_limited() {
    // 2 cells, 2 vectors, request top_k=2
    ailake::IvfPqIndex idx;
    idx.dim           = 2;
    idx.metric        = ailake::Metric::Euclidean;
    idx.config.nlist  = 2;
    idx.config.nprobe = 2;
    idx.config.pq_m   = 1;
    idx.config.pq_k   = 2;
    idx.config.max_iter = 10;

    idx.coarse_centroids = {{0.f,0.f}, {5.f,5.f}};

    idx.pq.m = 1;
    idx.pq.k = 2;
    idx.pq.centroids = {{0.f,0.f}, {5.f,5.f}};

    idx.inv_row_ids = {{1}, {2}};
    idx.inv_codes   = {{0}, {1}};

    float q[] = {0.f, 0.f};
    auto hits = ailake::ivfpq_search(idx, q, 2);
    CHECK_EQ(hits.size(), 2u);
}

static void test_ivfpq_no_results_when_nprobe_zero() {
    ailake::IvfPqIndex idx;
    idx.dim           = 4;
    idx.metric        = ailake::Metric::Euclidean;
    idx.config.nlist  = 1;
    idx.config.nprobe = 0;  // explicit 0 → no cells scanned
    idx.config.pq_m   = 2;
    idx.config.pq_k   = 2;
    idx.config.max_iter = 10;

    idx.coarse_centroids.push_back({0.f,0.f,0.f,0.f});
    idx.pq.m = 2; idx.pq.k = 2;
    idx.inv_row_ids.push_back({42});
    idx.inv_codes.push_back({0, 0});

    float q[] = {0.f, 0.f, 0.f, 0.f};
    auto hits = ailake::ivfpq_search(idx, q, 5, 0);
    CHECK(hits.empty());
}

int main() {
    test_ivfpq_returns_nearest();
    test_ivfpq_top_k_limited();
    test_ivfpq_no_results_when_nprobe_zero();
    if (g_fail) {
        std::printf("FAILED: %d test(s)\n", g_fail);
        return 1;
    }
    std::printf("ailake test_ivfpq: all pass\n");
    return 0;
}
