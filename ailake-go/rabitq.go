// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// RaBitQ flat index: deserializer and brute-force search.
//
// Wire format: bincode v1 serialization of ailake_index::RaBitQIndex:
//   codebook.dim:    usize  (u64 LE)
//   codebook.seed:   u64    (projection matrix is regenerated, not stored)
//   entries:         Vec<RaBitQVec>
//     each entry:
//       code:  Vec<u8>   (packed sign bits, ceil(dim/8) bytes)
//       norm:  f32
//       scale: f32
//   row_ids: Vec<u64>
//   metric:  u32          (enum variant: 0=cosine, 1=euclidean, 2=dot, 3=normalized-cosine)
//   dim:     u32
//   raw_f16: Option<Vec<u16>>  (f16 bits; None=0x00, Some=0x01+slice)
package ailake

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"sort"
)

// RaBitQVec is the binary-quantized representation of one database vector.
type RaBitQVec struct {
	Code  []byte  // packed sign bits: bit i = sign(P·x̂)[i]
	Norm  float32 // original L2 norm
	Scale float32 // sum(|P·x̂|) / sqrt(dim)
}

// RaBitQIndex is a deserialized AI-Lake RaBitQ flat index ready for search.
type RaBitQIndex struct {
	Seed    uint64
	Dim     uint32
	Metric  uint8
	Entries []RaBitQVec
	RowIDs  []uint64
	RawF16  []uint16 // nil when keep_raw=false; length = len(Entries)*Dim
	proj    []float32 // dim×dim, row-major; regenerated from Seed
}

// DeserializeRaBitQ deserializes a bincode-encoded RaBitQIndex from buf.
// The projection matrix is regenerated from the stored seed (not stored on disk).
func DeserializeRaBitQ(buf []byte) (*RaBitQIndex, error) {
	r := newBincodeReader(buf)

	dim, err := r.readUsize()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read codebook.dim: %w", err)
	}
	seed, err := r.readU64()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read codebook.seed: %w", err)
	}

	entryCount, err := r.readUsize()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read entries length: %w", err)
	}
	entries := make([]RaBitQVec, entryCount)
	for i := range entries {
		codeLen, err := r.readUsize()
		if err != nil {
			return nil, fmt.Errorf("rabitq: entry %d code length: %w", i, err)
		}
		code, err := r.readN(int(codeLen))
		if err != nil {
			return nil, fmt.Errorf("rabitq: entry %d code: %w", i, err)
		}
		code2 := make([]byte, len(code))
		copy(code2, code)
		norm, err := r.readF32()
		if err != nil {
			return nil, fmt.Errorf("rabitq: entry %d norm: %w", i, err)
		}
		scale, err := r.readF32()
		if err != nil {
			return nil, fmt.Errorf("rabitq: entry %d scale: %w", i, err)
		}
		entries[i] = RaBitQVec{Code: code2, Norm: norm, Scale: scale}
	}

	rowIDs, err := r.readU64Slice()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read row_ids: %w", err)
	}

	metricVariant, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read metric: %w", err)
	}

	dim32, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read dim: %w", err)
	}

	// Option<Vec<f16>>: tag 0x00=None, 0x01=Some
	optTag, err := r.readU8()
	if err != nil {
		return nil, fmt.Errorf("rabitq: read raw_f16 tag: %w", err)
	}
	var rawF16 []uint16
	if optTag == 1 {
		n, err := r.readUsize()
		if err != nil {
			return nil, fmt.Errorf("rabitq: read raw_f16 length: %w", err)
		}
		rawF16 = make([]uint16, n)
		for i := range rawF16 {
			v, err := r.readN(2)
			if err != nil {
				return nil, fmt.Errorf("rabitq: raw_f16[%d]: %w", i, err)
			}
			rawF16[i] = uint16(v[0]) | uint16(v[1])<<8
		}
	}

	idx := &RaBitQIndex{
		Seed:    seed,
		Dim:     dim32,
		Metric:  uint8(metricVariant),
		Entries: entries,
		RowIDs:  rowIDs,
		RawF16:  rawF16,
	}
	idx.buildProj(int(dim))
	return idx, nil
}

// buildProj regenerates the dim×dim projection matrix from the seed.
// Uses the same algorithm as RaBitQCodebook::rebuild_proj in Rust:
// column-normalized random Gaussian matrix.
func (idx *RaBitQIndex) buildProj(dim int) {
	proj := make([]float32, dim*dim)
	rng := rand.New(rand.NewSource(int64(idx.Seed)))
	for col := 0; col < dim; col++ {
		var normSq float64
		for row := 0; row < dim; row++ {
			v := rng.Float64()*2 - 1
			proj[row*dim+col] = float32(v)
			normSq += v * v
		}
		inv := float32(1.0 / math.Sqrt(normSq+1e-24))
		for row := 0; row < dim; row++ {
			proj[row*dim+col] *= inv
		}
	}
	idx.proj = proj
}

// project applies the rotation matrix P to v, returning P·v.
func (idx *RaBitQIndex) project(v []float32) []float32 {
	dim := int(idx.Dim)
	out := make([]float32, dim)
	for i := 0; i < dim; i++ {
		var s float32
		row := idx.proj[i*dim : (i+1)*dim]
		for j, x := range row {
			s += x * v[j]
		}
		out[i] = s
	}
	return out
}

// bitsFromSigns packs sign bits: bit i = (v[i] > 0).
func bitsFromSigns(v []float32) []byte {
	codeLen := (len(v) + 7) / 8
	code := make([]byte, codeLen)
	for i, val := range v {
		if val > 0 {
			code[i/8] |= 1 << uint(i&7)
		}
	}
	return code
}

// hammingXorPopcount counts XOR bits between two equal-length byte slices.
func hammingXorPopcount(a, b []byte) int {
	var total int
	for i := range a {
		total += bits.OnesCount8(a[i] ^ b[i])
	}
	return total
}

func f16ToF32Go(bits uint16) float32 {
	return f16ToF32(bits)
}

// Search returns top-k (rowID, distance) pairs sorted by distance ascending.
// rerank: if true and RawF16 is set, reranks top 3×k candidates exactly.
func (idx *RaBitQIndex) Search(query []float32, topK int, rerank bool) []SearchResult {
	if len(idx.Entries) == 0 {
		return nil
	}
	dim := int(idx.Dim)

	// Normalize query
	var qNorm float64
	for _, x := range query {
		qNorm += float64(x) * float64(x)
	}
	qNorm = math.Sqrt(qNorm)
	qHat := make([]float32, dim)
	if qNorm > 1e-12 {
		for i, x := range query {
			qHat[i] = float32(float64(x) / qNorm)
		}
	} else {
		copy(qHat, query)
	}

	// Project query
	qProj := idx.project(qHat)
	var qScaleSum float64
	for _, x := range qProj {
		if x < 0 {
			qScaleSum -= float64(x)
		} else {
			qScaleSum += float64(x)
		}
	}
	qScale := float32(qScaleSum / math.Sqrt(float64(dim)))
	qCode := bitsFromSigns(qProj)

	type scored struct {
		idx  int
		dist float32
	}

	all := make([]scored, len(idx.Entries))
	for i, e := range idx.Entries {
		hamming := hammingXorPopcount(qCode, e.Code)
		ip := (1.0 - 2.0*float32(hamming)/float32(dim)) * qScale * e.Scale

		var dist float32
		switch idx.Metric {
		case MetricCosine, MetricNormalizedCosine:
			dist = 1.0 - ip
		case MetricDotProduct:
			dist = -ip * float32(qNorm) * e.Norm
		default: // Euclidean
			normX := e.Norm
			d2 := float32(qNorm*qNorm) + normX*normX - 2*ip*float32(qNorm)*normX
			if d2 < 0 {
				d2 = 0
			}
			dist = float32(math.Sqrt(float64(d2)))
		}
		all[i] = scored{i, dist}
	}

	sort.Slice(all, func(a, b int) bool { return all[a].dist < all[b].dist })

	// Reranking with raw F16 vectors
	candidates := topK
	if rerank && len(idx.RawF16) > 0 {
		candidates = topK * 3
	}
	if candidates > len(all) {
		candidates = len(all)
	}
	top := all[:candidates]

	if rerank && len(idx.RawF16) > 0 {
		reranked := make([]scored, len(top))
		for j, s := range top {
			dbF16 := idx.RawF16[s.idx*dim : (s.idx+1)*dim]
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
			reranked[j] = scored{s.idx, dist}
		}
		sort.Slice(reranked, func(a, b int) bool { return reranked[a].dist < reranked[b].dist })
		top = reranked
	}

	if topK > len(top) {
		topK = len(top)
	}
	out := make([]SearchResult, topK)
	for i, s := range top[:topK] {
		out[i] = SearchResult{RowID: idx.RowIDs[s.idx], Distance: s.dist}
	}
	return out
}

func cosineDistGo(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom < 1e-12 {
		return 1
	}
	return float32(1 - dot/denom)
}

func dotDistGo(a, b []float32) float32 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(-dot)
}

func euclideanDistGo(a, b []float32) float32 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(math.Sqrt(sum))
}
