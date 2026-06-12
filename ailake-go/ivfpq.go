// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// IVF-PQ index deserialization and search.
//
// Wire format: bincode v1 serialization of ailake_index::IvfPqSnapshotCore
// followed by an optional trailing byte for the residual flag:
//
//   config:            IvfPqConfig { nlist, nprobe, pq_m, pq_k, max_iter: usize each }
//   metric:            u8
//   dim:               usize
//   coarse_centroids:  Vec<Vec<f32>>
//   pq:                PQCodebook { m: usize, k: usize, centroids: Vec<Vec<f32>> }
//   inv_row_ids:       Vec<Vec<u64>>
//   inv_codes:         Vec<Vec<u8>>   — flat PQ codes, stride=pq_m per entry
//   [residual: u8]     — optional trailing byte; 0x01 = residual PQ encoding active
package ailake

import (
	"sort"
)

// IvfPqConfig mirrors ailake_index::IvfPqConfig.
type IvfPqConfig struct {
	Nlist   uint64
	Nprobe  uint64
	PqM     uint64 // sub-vector count
	PqK     uint64 // centroids per sub-space (≤ 256)
	MaxIter uint64
	Residual bool // true = codes are per-cluster residuals; requires per-cluster ADC
}

// PQCodebook mirrors ailake_vec::PQCodebook.
type PQCodebook struct {
	M         uint64
	K         uint64
	Centroids [][]float32 // [m*k][sub_dim]
}

// IvfPqIndex is a deserialized AI-Lake IVF-PQ index ready for search.
type IvfPqIndex struct {
	Config          IvfPqConfig
	Metric          uint8
	Dim             uint64
	CoarseCentroids [][]float32 // [nlist][dim]
	PQ              PQCodebook
	InvRowIDs       [][]uint64
	InvCodes        [][]byte // [nlist] flat codes, stride=PqM
}

// DeserializeIvfPq deserializes a bincode-encoded IvfPqSnapshot from buf.
func DeserializeIvfPq(buf []byte) (*IvfPqIndex, error) {
	r := newBincodeReader(buf)

	nlist, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	nprobe, _ := r.readUsize()
	pqM, _ := r.readUsize()
	pqK, _ := r.readUsize()
	maxIter, err := r.readUsize()
	if err != nil {
		return nil, err
	}

	metric, err := r.readU8()
	if err != nil {
		return nil, err
	}
	dim, err := r.readUsize()
	if err != nil {
		return nil, err
	}

	coarse, err := r.readF32Slice2D()
	if err != nil {
		return nil, err
	}

	// PQCodebook
	pqM2, _ := r.readUsize()
	pqK2, _ := r.readUsize()
	pqCentroids, err := r.readF32Slice2D()
	if err != nil {
		return nil, err
	}

	invRowIDs, err := r.readU64Slice2D()
	if err != nil {
		return nil, err
	}
	invCodes, err := r.readU8Slice2D()
	if err != nil {
		return nil, err
	}

	// Optional trailing byte: residual flag (0x01 = residual PQ encoding active).
	// Legacy files have no trailing byte — default to false.
	residual := false
	if r.remaining() > 0 {
		b, rerr := r.readU8()
		if rerr == nil {
			residual = b != 0
		}
	}

	return &IvfPqIndex{
		Config: IvfPqConfig{
			Nlist:    nlist,
			Nprobe:   nprobe,
			PqM:      pqM,
			PqK:      pqK,
			MaxIter:  maxIter,
			Residual: residual,
		},
		Metric:          metric,
		Dim:             dim,
		CoarseCentroids: coarse,
		PQ:              PQCodebook{M: pqM2, K: pqK2, Centroids: pqCentroids},
		InvRowIDs:       invRowIDs,
		InvCodes:        invCodes,
	}, nil
}

// Search runs IVF-PQ search: probe nprobe cells, decode ADC distances.
//
// For residual-encoded indexes (Config.Residual=true), computes a per-cluster ADC
// table for each probed cell. The query residual q - coarse_centroid[cell] is used
// instead of the raw query, matching the encoding done at write time.
func (idx *IvfPqIndex) Search(query []float32, topK int, nprobe int) []SearchResult {
	if nprobe <= 0 {
		nprobe = int(idx.Config.Nprobe)
	}

	// 1. Find nearest coarse centroids
	type cell struct {
		i    int
		dist float32
	}
	cells := make([]cell, len(idx.CoarseCentroids))
	for i, c := range idx.CoarseCentroids {
		cells[i] = cell{i, distanceByMetric(idx.Metric, query, c)}
	}
	sort.Slice(cells, func(a, b int) bool { return cells[a].dist < cells[b].dist })
	if nprobe > len(cells) {
		nprobe = len(cells)
	}

	subDim := int(idx.Dim) / int(idx.PQ.M)

	// 2. For non-residual: precompute one global ADC table.
	//    For residual: ADC table is computed per cluster (see step 3).
	var globalLut [][]float32
	if !idx.Config.Residual {
		globalLut = idx.buildADCTable(query, subDim)
	}

	// 3. Scan probed cells, accumulate candidates
	type hit struct {
		rowID uint64
		dist  float32
	}
	var candidates []hit
	for _, cl := range cells[:nprobe] {
		lut := globalLut
		if idx.Config.Residual {
			// Residual query: subtract coarse centroid for this cluster.
			centroid := idx.CoarseCentroids[cl.i]
			qRes := make([]float32, len(query))
			for d := range query {
				qRes[d] = query[d] - centroid[d]
			}
			lut = idx.buildADCTable(qRes, subDim)
		}

		rowIDs := idx.InvRowIDs[cl.i]
		codes := idx.InvCodes[cl.i]
		for r, rowID := range rowIDs {
			start := r * int(idx.PQ.M)
			if start+int(idx.PQ.M) > len(codes) {
				break
			}
			var d float32
			for j := 0; j < int(idx.PQ.M); j++ {
				code := int(codes[start+j])
				if code < len(lut[j]) {
					d += lut[j][code]
				}
			}
			candidates = append(candidates, hit{rowID, d})
		}
	}

	// 4. Top-K
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}
	out := make([]SearchResult, len(candidates))
	for i, h := range candidates {
		out[i] = SearchResult{RowID: h.rowID, Distance: h.dist}
	}
	return out
}

// buildADCTable computes dist(query_sub_j, codebook[j][c]) for all j, c.
func (idx *IvfPqIndex) buildADCTable(query []float32, subDim int) [][]float32 {
	lut := make([][]float32, idx.PQ.M)
	for j := range lut {
		lut[j] = make([]float32, idx.PQ.K)
		qSub := query[j*subDim : (j+1)*subDim]
		for c := 0; c < int(idx.PQ.K); c++ {
			cIdx := int(j)*int(idx.PQ.K) + c
			if cIdx < len(idx.PQ.Centroids) {
				lut[j][c] = sqEuclidean(qSub, idx.PQ.Centroids[cIdx])
			}
		}
	}
	return lut
}

func sqEuclidean(a, b []float32) float32 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(sum)
}
