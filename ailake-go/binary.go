// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Binary Hamming flat index: deserializer and brute-force search.
//
// Wire format: bincode v1 serialization of ailake_index::BinaryIndex:
//
//	codes:         Vec<u8>         (u64 length + flat packed-bit bytes)
//	bytes_per_vec: usize           (u64 LE — ceil(dim/8))
//	row_ids:       Vec<u64>        (u64 length + u64 values)
//	metric:        u32             (enum: 0=cosine, 1=euclidean, 2=dot, 3=normalized-cosine)
//	dim:           u32
//	raw_f16:       Option<Vec<u16>> (tag 0x00=None, 0x01=Some + u64 length + u16 values)
package ailake

import (
	"fmt"
	"math"
	"math/bits"
	"sort"
)

// BinaryIndex is a deserialized AI-Lake Binary Hamming flat index.
type BinaryIndex struct {
	// Packed bit codes, flat: entry i at Codes[i*BytesPerVec : (i+1)*BytesPerVec].
	Codes       []byte
	BytesPerVec int
	RowIDs      []uint64
	Metric      uint8
	Dim         uint32
	// Raw F16 vectors for reranking (nil when keep_raw=false).
	// Flat: entry i at RawF16[i*Dim : (i+1)*Dim].
	RawF16 []uint16
}

// DeserializeBinary deserializes a bincode-encoded BinaryIndex from buf.
func DeserializeBinary(buf []byte) (*BinaryIndex, error) {
	r := newBincodeReader(buf)

	// codes: Vec<u8>
	codesLen, err := r.readUsize()
	if err != nil {
		return nil, fmt.Errorf("binary: read codes length: %w", err)
	}
	codes, err := r.readN(int(codesLen))
	if err != nil {
		return nil, fmt.Errorf("binary: read codes: %w", err)
	}
	codesCopy := make([]byte, len(codes))
	copy(codesCopy, codes)

	// bytes_per_vec: usize
	bpv, err := r.readUsize()
	if err != nil {
		return nil, fmt.Errorf("binary: read bytes_per_vec: %w", err)
	}

	// row_ids: Vec<u64>
	rowIDs, err := r.readU64Slice()
	if err != nil {
		return nil, fmt.Errorf("binary: read row_ids: %w", err)
	}

	// metric: u32
	metricVariant, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("binary: read metric: %w", err)
	}

	// dim: u32
	dim, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("binary: read dim: %w", err)
	}

	// raw_f16: Option<Vec<u16>>
	optTag, err := r.readU8()
	if err != nil {
		return nil, fmt.Errorf("binary: read raw_f16 tag: %w", err)
	}
	var rawF16 []uint16
	if optTag == 1 {
		n, err := r.readUsize()
		if err != nil {
			return nil, fmt.Errorf("binary: read raw_f16 length: %w", err)
		}
		rawF16 = make([]uint16, n)
		for i := range rawF16 {
			v, err := r.readN(2)
			if err != nil {
				return nil, fmt.Errorf("binary: raw_f16[%d]: %w", i, err)
			}
			rawF16[i] = uint16(v[0]) | uint16(v[1])<<8
		}
	}

	return &BinaryIndex{
		Codes:       codesCopy,
		BytesPerVec: int(bpv),
		RowIDs:      rowIDs,
		Metric:      uint8(metricVariant),
		Dim:         dim,
		RawF16:      rawF16,
	}, nil
}

// hammingBinary counts differing bits between two equal-length byte slices.
// Uses uint64 chunks for throughput; bits.OnesCount64 maps to a single
// POPCNT instruction on x86_64 and VCNT+UADDLV on aarch64.
func hammingBinary(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var total int
	chunks := n / 8
	for i := 0; i < chunks; i++ {
		base := i * 8
		var a64, b64 uint64
		for j := 0; j < 8; j++ {
			a64 |= uint64(a[base+j]) << (uint(j) * 8)
			b64 |= uint64(b[base+j]) << (uint(j) * 8)
		}
		total += bits.OnesCount64(a64 ^ b64)
	}
	for i := chunks * 8; i < n; i++ {
		total += bits.OnesCount8(a[i] ^ b[i])
	}
	return total
}

// f32ToBits binarizes a float32 vector: sign(x) ≥ 0 → bit 1, else bit 0.
// Bits packed MSB-first within each byte (bit 7 = dimension 0).
// Output length: ceil(len(v)/8) bytes.
func f32ToBits(v []float32) []byte {
	n := (len(v) + 7) / 8
	out := make([]byte, n)
	for i, val := range v {
		if val >= 0.0 {
			out[i/8] |= 0x80 >> uint(i%8)
		}
	}
	return out
}

// Search returns the top-k (rowID, distance) pairs sorted ascending by Hamming distance.
// rerank: if true and RawF16 is available, the top 3×k Hamming-nearest candidates are
// reranked using exact F16 distances with the configured metric.
func (idx *BinaryIndex) Search(query []float32, topK int, rerank bool) []SearchResult {
	if len(idx.RowIDs) == 0 || topK <= 0 {
		return nil
	}
	n := len(idx.RowIDs)
	bpv := idx.BytesPerVec
	dim := int(idx.Dim)

	qBits := f32ToBits(query)

	type scored struct {
		i    int
		dist float32
	}

	all := make([]scored, n)
	for i := 0; i < n; i++ {
		code := idx.Codes[i*bpv : (i+1)*bpv]
		h := hammingBinary(qBits, code)
		all[i] = scored{i, float32(h)}
	}

	sort.Slice(all, func(a, b int) bool { return all[a].dist < all[b].dist })

	candidates := topK
	if rerank && len(idx.RawF16) > 0 {
		candidates = topK * 3
	}
	if candidates > n {
		candidates = n
	}
	top := all[:candidates]

	if rerank && len(idx.RawF16) > 0 {
		reranked := make([]scored, len(top))
		for j, s := range top {
			dbF16 := idx.RawF16[s.i*dim : (s.i+1)*dim]
			dbF32 := make([]float32, dim)
			for k, bits := range dbF16 {
				dbF32[k] = f16ToF32Go(bits)
			}
			var dist float32
			switch idx.Metric {
			case MetricCosine, MetricNormalizedCosine:
				dist = cosineDistGo(query, dbF32)
			case MetricDotProduct:
				dist = dotDistGo(query, dbF32)
			default:
				dist = euclideanDistGo(query, dbF32)
			}
			reranked[j] = scored{s.i, dist}
		}
		sort.Slice(reranked, func(a, b int) bool { return reranked[a].dist < reranked[b].dist })
		top = reranked
	}

	if topK > len(top) {
		topK = len(top)
	}
	out := make([]SearchResult, topK)
	for i, s := range top[:topK] {
		out[i] = SearchResult{RowID: idx.RowIDs[s.i], Distance: s.dist}
	}
	return out
}

// hammingDistance is the public Hamming distance function (used in tests).
func hammingDistance(a, b []byte) int { return hammingBinary(a, b) }

// ensure math import used
var _ = math.Sqrt
