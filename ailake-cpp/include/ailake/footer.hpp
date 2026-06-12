// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// AILK header/trailer parsing — docs/specs/FILE_FORMAT.md
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ailake {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr std::array<uint8_t, 4> kAilakeMagic = {'A', 'I', 'L', 'K'};
static constexpr uint16_t kFormatVersion  = 1;
static constexpr size_t   kHeaderSize     = 64;
static constexpr size_t   kTrailerSize    = 24;

// index type flags (bit 0 of header.flags)
static constexpr uint16_t kFlagIndexIvfPq  = 0x0001; // bit 0: IVF-PQ. Default (0): HNSW.

// precision values
enum class Precision : uint8_t { F32 = 0, F16 = 1, I8 = 2, Binary = 3 };

// distance metric values
enum class Metric : uint8_t { Cosine = 0, Euclidean = 1, DotProduct = 2, NormalizedCosine = 3 };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline uint16_t read_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t read_le64(const uint8_t* p) {
    return static_cast<uint64_t>(p[0])
         | (static_cast<uint64_t>(p[1]) << 8)
         | (static_cast<uint64_t>(p[2]) << 16)
         | (static_cast<uint64_t>(p[3]) << 24)
         | (static_cast<uint64_t>(p[4]) << 32)
         | (static_cast<uint64_t>(p[5]) << 40)
         | (static_cast<uint64_t>(p[6]) << 48)
         | (static_cast<uint64_t>(p[7]) << 56);
}
inline float read_f32_le(const uint8_t* p) {
    uint32_t bits = read_le32(p);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// ---------------------------------------------------------------------------
// AilakeHeader — 64 bytes at the start of every AILK section
// ---------------------------------------------------------------------------

struct AilakeHeader {
    uint16_t  format_version;
    uint16_t  flags;           // bit 0: 0=HNSW, 1=IVF-PQ
    uint32_t  dim;
    Precision precision;
    Metric    distance_metric;
    uint64_t  record_count;
    uint64_t  centroid_offset; // relative to AILK section start
    uint64_t  centroid_len;
    uint64_t  hnsw_offset;     // relative to AILK section start
    uint64_t  hnsw_len;

    bool is_ivf_pq()  const noexcept { return (flags & kFlagIndexIvfPq)  != 0; }
};

// Parse a 64-byte AILK header from buf.
// Throws std::runtime_error on magic/version mismatch.
inline AilakeHeader parse_header(const uint8_t* buf) {
    if (std::memcmp(buf, kAilakeMagic.data(), 4) != 0)
        throw std::runtime_error("ailake: bad AILK magic");

    AilakeHeader h{};
    h.format_version  = read_le16(buf + 4);
    if (h.format_version != kFormatVersion)
        throw std::runtime_error("ailake: unsupported format version " +
                                 std::to_string(h.format_version));
    h.flags            = read_le16(buf + 6);
    h.dim              = read_le32(buf + 8);
    h.precision        = static_cast<Precision>(buf[12]);
    h.distance_metric  = static_cast<Metric>(buf[13]);
    h.record_count     = read_le64(buf + 16);
    h.centroid_offset  = read_le64(buf + 24);
    h.centroid_len     = read_le64(buf + 32);
    h.hnsw_offset      = read_le64(buf + 40);
    h.hnsw_len         = read_le64(buf + 48);
    return h;
}

// ---------------------------------------------------------------------------
// AilakeTrailer — 24 bytes at the end of every AILK section
// ---------------------------------------------------------------------------

struct AilakeTrailer {
    uint64_t footer_offset; // absolute byte offset of AILK header in file
    uint64_t footer_len;    // total byte length of this AILK section
    uint16_t format_version;
    uint16_t flags;
};

inline AilakeTrailer parse_trailer(const uint8_t* buf) {
    if (std::memcmp(buf + 20, kAilakeMagic.data(), 4) != 0)
        throw std::runtime_error("ailake: bad AILK trailer magic");
    AilakeTrailer t{};
    t.footer_offset  = read_le64(buf);
    t.footer_len     = read_le64(buf + 8);
    t.format_version = read_le16(buf + 16);
    t.flags          = read_le16(buf + 18);
    return t;
}

// ---------------------------------------------------------------------------
// Centroid blob: dim × 4 bytes F32 LE + 4-byte radius F32 LE
// Returns (centroid vector, radius).
// ---------------------------------------------------------------------------

inline std::pair<std::vector<float>, float>
parse_centroid(const uint8_t* buf, uint32_t dim) {
    std::vector<float> vec(dim);
    for (uint32_t i = 0; i < dim; ++i)
        vec[i] = read_f32_le(buf + i * 4);
    float radius = read_f32_le(buf + dim * 4);
    return {std::move(vec), radius};
}

// ---------------------------------------------------------------------------
// F16 → F32 conversion (IEEE 754 half-precision)
// ---------------------------------------------------------------------------

inline float f16_to_f32(uint16_t bits) noexcept {
    uint32_t sign = (uint32_t)(bits >> 15) << 31;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f32bits;
    if (exp == 0) {
        if (mant == 0) {
            f32bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            f32bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f32bits = sign | 0x7F800000 | (mant << 13);
    } else {
        f32bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float v;
    std::memcpy(&v, &f32bits, 4);
    return v;
}

// Decode a FIXED_LEN_BYTE_ARRAY F16 Parquet column value.
inline std::vector<float> decode_f16_vector(const uint8_t* raw, uint32_t dim) {
    std::vector<float> out(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        uint16_t bits = static_cast<uint16_t>(raw[i * 2])
                      | (static_cast<uint16_t>(raw[i * 2 + 1]) << 8);
        out[i] = f16_to_f32(bits);
    }
    return out;
}

} // namespace ailake
