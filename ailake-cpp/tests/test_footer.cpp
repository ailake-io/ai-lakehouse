// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Unit tests for ailake/footer.hpp — header parsing and flag methods.
#include <ailake/footer.hpp>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// Build a minimal 64-byte AILK header buffer.
static void fill_header(uint8_t* buf, uint16_t version, uint16_t flags,
                        uint32_t dim, uint8_t prec, uint8_t metric) {
    std::memset(buf, 0, 64);
    buf[0] = 'A'; buf[1] = 'I'; buf[2] = 'L'; buf[3] = 'K';
    buf[4] = static_cast<uint8_t>(version & 0xFF);
    buf[5] = static_cast<uint8_t>(version >> 8);
    buf[6] = static_cast<uint8_t>(flags & 0xFF);
    buf[7] = static_cast<uint8_t>(flags >> 8);
    buf[8]  = static_cast<uint8_t>(dim & 0xFF);
    buf[9]  = static_cast<uint8_t>((dim >> 8) & 0xFF);
    buf[10] = static_cast<uint8_t>((dim >> 16) & 0xFF);
    buf[11] = static_cast<uint8_t>(dim >> 24);
    buf[12] = prec;
    buf[13] = metric;
}

static void test_parse_hnsw_header() {
    uint8_t buf[64];
    fill_header(buf, 1, 0, 128, 1 /* F16 */, 0 /* Cosine */);
    // record_count = 1000 at bytes 16-23
    buf[16] = 0xE8; buf[17] = 0x03; // 1000 LE

    auto h = ailake::parse_header(buf);
    CHECK_EQ(h.format_version, 1);
    CHECK_EQ(h.flags, 0);
    CHECK_EQ(h.dim, 128u);
    CHECK(h.precision == ailake::Precision::F16);
    CHECK(h.distance_metric == ailake::Metric::Cosine);
    CHECK_EQ(h.record_count, 1000u);
    CHECK(!h.is_ivf_pq());
}

static void test_flag_ivfpq() {
    uint8_t buf[64];
    fill_header(buf, 1, ailake::kFlagIndexIvfPq, 64, 0, 0);
    auto h = ailake::parse_header(buf);
    CHECK(h.is_ivf_pq());
}

static void test_bad_magic_throws() {
    uint8_t buf[64];
    fill_header(buf, 1, 0, 64, 0, 0);
    buf[0] = 'X'; // corrupt magic
    bool threw = false;
    try { ailake::parse_header(buf); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

static void test_bad_version_throws() {
    uint8_t buf[64];
    fill_header(buf, 99, 0, 64, 0, 0); // unsupported version
    bool threw = false;
    try { ailake::parse_header(buf); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

int main() {
    test_parse_hnsw_header();
    test_flag_ivfpq();
    test_bad_magic_throws();
    test_bad_version_throws();
    if (g_fail) {
        std::printf("FAILED: %d test(s)\n", g_fail);
        return 1;
    }
    std::printf("ailake test_footer: all pass\n");
    return 0;
}
