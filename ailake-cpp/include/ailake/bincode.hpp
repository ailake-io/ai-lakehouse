// SPDX-License-Identifier: MIT OR Apache-2.0
// Bincode v1 little-endian decoder.
// usize → u64, Vec<T> → length (u64) + elements, Option<T> → tag u8 + value.
#pragma once

#include "footer.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ailake {

class BincodeReader {
public:
    explicit BincodeReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    size_t pos()       const { return pos_; }
    size_t remaining() const { return size_ - pos_; }

    uint8_t  read_u8()  { return read_raw<uint8_t>(); }
    uint32_t read_u32() { return read_le<uint32_t>(); }
    uint64_t read_u64() { return read_le<uint64_t>(); }
    uint64_t read_usize() { return read_u64(); } // Rust usize = u64 in bincode v1
    float    read_f32()   {
        uint32_t bits = read_u32();
        float v; std::memcpy(&v, &bits, 4); return v;
    }

    std::vector<float> read_f32_vec() {
        auto n = read_usize();
        std::vector<float> out(n);
        for (auto& v : out) v = read_f32();
        return out;
    }

    std::vector<std::vector<float>> read_f32_vec2d() {
        auto n = read_usize();
        std::vector<std::vector<float>> out(n);
        for (auto& v : out) v = read_f32_vec();
        return out;
    }

    std::vector<uint64_t> read_u64_vec() {
        auto n = read_usize();
        std::vector<uint64_t> out(n);
        for (auto& v : out) v = read_u64();
        return out;
    }

    std::vector<std::vector<uint64_t>> read_u64_vec2d() {
        auto n = read_usize();
        std::vector<std::vector<uint64_t>> out(n);
        for (auto& v : out) v = read_u64_vec();
        return out;
    }

    // Vec<Vec<Vec<usize>>> — [node][layer] = neighbor list
    std::vector<std::vector<std::vector<uint64_t>>> read_neighbors() {
        auto node_count = read_usize();
        std::vector<std::vector<std::vector<uint64_t>>> out(node_count);
        for (auto& node : out) {
            auto layer_count = read_usize();
            node.resize(layer_count);
            for (auto& layer : node)
                layer = read_u64_vec();
        }
        return out;
    }

    std::optional<uint64_t> read_option_usize() {
        uint8_t tag = read_u8();
        if (tag == 0) return std::nullopt;
        return read_usize();
    }

    // Vec<Vec<uint8_t>> — flat PQ codes per inverted list
    std::vector<std::vector<uint8_t>> read_u8_vec2d() {
        auto n = read_usize();
        std::vector<std::vector<uint8_t>> out(n);
        for (auto& v : out) {
            auto m = read_usize();
            v.resize(m);
            check(m);
            std::memcpy(v.data(), data_ + pos_, m);
            pos_ += m;
        }
        return out;
    }

private:
    const uint8_t* data_;
    size_t size_, pos_;

    void check(size_t n) {
        if (n > remaining())
            throw std::runtime_error("bincode: unexpected EOF");
    }

    template<typename T>
    T read_raw() {
        check(sizeof(T));
        T v; std::memcpy(&v, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    template<typename T>
    T read_le() {
        uint8_t buf[sizeof(T)];
        check(sizeof(T));
        std::memcpy(buf, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        T v = 0;
        for (size_t i = 0; i < sizeof(T); ++i)
            v |= static_cast<T>(buf[i]) << (8 * i);
        return v;
    }
};

} // namespace ailake
