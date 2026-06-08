// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/binary.hpp — f32_to_bits, hamming_distance, binary_search.
#include <ailake/binary.hpp>
#include <cstdio>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// ---------------------------------------------------------------------------
// f32_to_bits
// ---------------------------------------------------------------------------

static void test_f32_to_bits_alternating() {
    // dim=8, alternating +/-: bits 7,5,3,1 set → 0xAA (MSB-first)
    float v[] = {1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f};
    auto bits = ailake::f32_to_bits(v, 8);
    CHECK_EQ(bits.size(), 1u);
    CHECK_EQ(bits[0], (uint8_t)0xAA);
}

static void test_f32_to_bits_all_positive() {
    float v[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto bits = ailake::f32_to_bits(v, 8);
    CHECK_EQ(bits[0], (uint8_t)0xFF);
}

static void test_f32_to_bits_all_negative() {
    float v[] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
    auto bits = ailake::f32_to_bits(v, 8);
    CHECK_EQ(bits[0], (uint8_t)0x00);
}

static void test_f32_to_bits_zero_is_positive() {
    // 0.0f >= 0.0f is true in IEEE 754 (signed zero equality)
    float v[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    auto bits = ailake::f32_to_bits(v, 8);
    CHECK_EQ(bits[0], (uint8_t)0xFF);
}

static void test_f32_to_bits_two_bytes() {
    // dim=16: first byte all-positive, second byte all-negative
    float v[16];
    for (int i = 0; i < 8; ++i) v[i] = 1.f;
    for (int i = 8; i < 16; ++i) v[i] = -1.f;
    auto bits = ailake::f32_to_bits(v, 16);
    CHECK_EQ(bits.size(), 2u);
    CHECK_EQ(bits[0], (uint8_t)0xFF);
    CHECK_EQ(bits[1], (uint8_t)0x00);
}

// ---------------------------------------------------------------------------
// hamming_distance
// ---------------------------------------------------------------------------

static void test_hamming_zero() {
    uint8_t a[] = {0xFF, 0xAA};
    uint8_t b[] = {0xFF, 0xAA};
    CHECK_EQ(ailake::hamming_distance(a, b, 2), 0);
}

static void test_hamming_single_byte() {
    uint8_t a[] = {0xFF};
    uint8_t b[] = {0x00};
    CHECK_EQ(ailake::hamming_distance(a, b, 1), 8);

    uint8_t c[] = {0xAA};
    uint8_t d[] = {0x55};
    CHECK_EQ(ailake::hamming_distance(c, d, 1), 8);

    uint8_t e[] = {0x0F};
    uint8_t f[] = {0xFF};
    CHECK_EQ(ailake::hamming_distance(e, f, 1), 4);
}

static void test_hamming_multibyte() {
    uint8_t a[] = {0xFF, 0xFF, 0xFF, 0xFF};
    uint8_t b[] = {0x00, 0x00, 0x00, 0x00};
    CHECK_EQ(ailake::hamming_distance(a, b, 4), 32);
}

static void test_hamming_large_simd() {
    // 32 bytes (AVX2 chunk size) — all bits differ
    uint8_t a[32], b[32];
    for (int i = 0; i < 32; ++i) { a[i] = 0xFF; b[i] = 0x00; }
    CHECK_EQ(ailake::hamming_distance(a, b, 32), 256);
}

// ---------------------------------------------------------------------------
// binary_search
// ---------------------------------------------------------------------------

static void test_binary_search_top1() {
    // 5 entries, dim=8 (bytes_per_vec=1)
    // Codes: 0xFF, 0xFE, 0xFC, 0xF0, 0x00
    // Query all-positive → q_bits=0xFF
    // Hamming: 0, 1, 2, 4, 8 — row_id 10 is nearest
    ailake::BinaryIndex idx;
    idx.dim          = 8;
    idx.bytes_per_vec = 1;
    idx.metric       = ailake::Metric::Cosine;
    idx.row_ids      = {10, 20, 30, 40, 50};
    idx.codes        = {0xFF, 0xFE, 0xFC, 0xF0, 0x00};

    float q[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto hits = ailake::binary_search(idx, q, 2);

    CHECK_EQ(hits.size(), 2u);
    CHECK_EQ(hits[0].row_id, 10u);
    CHECK_EQ(hits[1].row_id, 20u);
    CHECK(hits[0].distance < hits[1].distance);
}

static void test_binary_search_top_k_capped() {
    ailake::BinaryIndex idx;
    idx.dim          = 8;
    idx.bytes_per_vec = 1;
    idx.metric       = ailake::Metric::Euclidean;
    idx.row_ids      = {1, 2, 3};
    idx.codes        = {0xFF, 0xFE, 0xFC};

    float q[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto hits = ailake::binary_search(idx, q, 10); // request more than available
    CHECK_EQ(hits.size(), 3u);
}

static void test_binary_search_with_f16_rerank() {
    // 3 entries, dim=8
    // Hamming order (q_bits=0xFF): entry0(h=0), entry2(h=4), entry1(h=8)
    // After F16 rerank with cosine, entry0 should still win (all-positive ~= all-positive)
    ailake::BinaryIndex idx;
    idx.dim          = 8;
    idx.bytes_per_vec = 1;
    idx.metric       = ailake::Metric::Cosine;
    idx.row_ids      = {1, 2, 3};
    idx.codes        = {0xFF, 0x00, 0xAA};

    // raw_f16: F16 0x3800 = 0.5f, 0x0000 = 0.0f, 0xB800 = -0.5f
    idx.raw_f16.resize(3 * 8);
    // entry0: all 0.5
    for (int i = 0; i < 8; ++i)  idx.raw_f16[0 * 8 + i] = 0x3800;
    // entry1: all 0.0
    for (int i = 0; i < 8; ++i)  idx.raw_f16[1 * 8 + i] = 0x0000;
    // entry2: half +0.5, half -0.5
    for (int i = 0; i < 4; ++i)  idx.raw_f16[2 * 8 + i] = 0x3800;
    for (int i = 4; i < 8; ++i)  idx.raw_f16[2 * 8 + i] = 0xB800;

    float q[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto hits = ailake::binary_search(idx, q, 2, 3);

    CHECK_EQ(hits.size(), 2u);
    CHECK_EQ(hits[0].row_id, 1u); // cosine=0 (perfect match direction)
}

static void test_binary_search_empty() {
    ailake::BinaryIndex idx;
    idx.dim          = 8;
    idx.bytes_per_vec = 1;
    idx.metric       = ailake::Metric::Cosine;

    float q[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto hits = ailake::binary_search(idx, q, 5);
    CHECK(hits.empty());
}

static void test_binary_search_zero_topk() {
    ailake::BinaryIndex idx;
    idx.dim          = 8;
    idx.bytes_per_vec = 1;
    idx.metric       = ailake::Metric::Cosine;
    idx.row_ids      = {1, 2};
    idx.codes        = {0xFF, 0x00};

    float q[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto hits = ailake::binary_search(idx, q, 0);
    CHECK(hits.empty());
}

int main() {
    test_f32_to_bits_alternating();
    test_f32_to_bits_all_positive();
    test_f32_to_bits_all_negative();
    test_f32_to_bits_zero_is_positive();
    test_f32_to_bits_two_bytes();
    test_hamming_zero();
    test_hamming_single_byte();
    test_hamming_multibyte();
    test_hamming_large_simd();
    test_binary_search_top1();
    test_binary_search_top_k_capped();
    test_binary_search_with_f16_rerank();
    test_binary_search_empty();
    test_binary_search_zero_topk();
    if (g_fail) {
        std::printf("FAILED: %d test(s)\n", g_fail);
        return 1;
    }
    std::printf("ailake test_binary: all pass\n");
    return 0;
}
