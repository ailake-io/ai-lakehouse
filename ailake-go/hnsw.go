// SPDX-License-Identifier: MIT OR Apache-2.0
// HNSW index deserialization and search.
//
// Wire format: bincode v1 serialization of ailake_index::HnswSnapshot:
//   m:                usize  (u64 LE)
//   ef_construction:  usize
//   max_elements:     usize
//   metric:           u8     (0=cosine, 1=euclidean, 2=dotproduct)
//   dim:              u32
//   row_ids:          Vec<u64>
//   flat_vecs:        Vec<f32>  — flat storage, stride=dim
//   neighbors:        Vec<Vec<Vec<usize>>>  — [node][layer] = []neighbor_idx
//   node_levels:      Vec<usize>
//   entry_point:      Option<usize>
//   max_layer:        usize
package ailake

import (
	"container/heap"
	"errors"
	"math"
)

// HnswIndex is a deserialized AI-Lake HNSW graph ready for search.
type HnswIndex struct {
	M              uint64
	EfConstruction uint64
	Metric         uint8
	Dim            uint32
	RowIDs         []uint64
	FlatVecs       []float32 // stride = Dim
	Neighbors      [][][]uint64
	NodeLevels     []uint64
	EntryPoint     uint64
	HasEntry       bool
	MaxLayer       uint64
}

// DeserializeHnsw deserializes a bincode-encoded HnswSnapshot from buf.
func DeserializeHnsw(buf []byte) (*HnswIndex, error) {
	r := newBincodeReader(buf)

	m, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	efConstruction, err := r.readUsize()
	if err != nil {
		return nil, err
	}
	_, err = r.readUsize() // max_elements — not needed for search
	if err != nil {
		return nil, err
	}
	metric, err := r.readU8()
	if err != nil {
		return nil, err
	}
	dim, err := r.readU32()
	if err != nil {
		return nil, err
	}
	rowIDs, err := r.readU64Slice()
	if err != nil {
		return nil, err
	}
	flatVecs, err := r.readF32Slice()
	if err != nil {
		return nil, err
	}
	if uint64(len(flatVecs)) != uint64(len(rowIDs))*uint64(dim) {
		return nil, errors.New("hnsw: flat_vecs length mismatch")
	}
	neighbors, err := r.readNeighbors()
	if err != nil {
		return nil, err
	}
	nodeLevels, err := r.readU64Slice()
	if err != nil {
		return nil, err
	}
	ep, hasEP, err := r.readOptionU64()
	if err != nil {
		return nil, err
	}
	maxLayer, err := r.readUsize()
	if err != nil {
		return nil, err
	}

	return &HnswIndex{
		M:              m,
		EfConstruction: efConstruction,
		Metric:         metric,
		Dim:            dim,
		RowIDs:         rowIDs,
		FlatVecs:       flatVecs,
		Neighbors:      neighbors,
		NodeLevels:     nodeLevels,
		EntryPoint:     ep,
		HasEntry:       hasEP,
		MaxLayer:       maxLayer,
	}, nil
}

// vec returns the vector for node i (read-only slice into FlatVecs).
func (h *HnswIndex) vec(i uint64) []float32 {
	start := i * uint64(h.Dim)
	return h.FlatVecs[start : start+uint64(h.Dim)]
}

// distance computes the distance between query q and node i using the index metric.
func (h *HnswIndex) distance(q []float32, i uint64) float32 {
	v := h.vec(i)
	switch h.Metric {
	case MetricEuclidean:
		return euclideanDistance(q, v)
	case MetricDotProduct:
		return -dotProduct(q, v)
	default:
		return cosineDistance(q, v)
	}
}

// SearchResult is one nearest-neighbour hit.
type SearchResult struct {
	RowID    uint64
	Distance float32
}

// Search runs greedy HNSW search and returns up to topK nearest neighbours.
// efSearch controls recall/speed trade-off (higher = better recall).
func (h *HnswIndex) Search(query []float32, topK int, efSearch int) []SearchResult {
	if !h.HasEntry || len(h.RowIDs) == 0 {
		return nil
	}
	if efSearch < topK {
		efSearch = topK
	}

	ep := h.EntryPoint
	// Traverse from top layer down to layer 1, greedy (ef=1)
	for layer := int(h.MaxLayer); layer > 0; layer-- {
		ep = h.greedyNearest(query, ep, layer)
	}
	// Layer 0: beam search with efSearch candidates
	candidates := h.beamSearch(query, ep, 0, efSearch)

	// Sort ascending by distance, take topK
	sortCandidates(candidates)
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}
	results := make([]SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = SearchResult{RowID: h.RowIDs[c.idx], Distance: c.dist}
	}
	return results
}

// greedyNearest does a greedy 1-nearest-neighbour traversal at the given layer.
func (h *HnswIndex) greedyNearest(query []float32, entry uint64, layer int) uint64 {
	best := entry
	bestDist := h.distance(query, entry)
	for {
		improved := false
		if int(best) >= len(h.Neighbors) || layer >= len(h.Neighbors[best]) {
			break
		}
		for _, nb := range h.Neighbors[best][layer] {
			d := h.distance(query, nb)
			if d < bestDist {
				bestDist = d
				best = nb
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	return best
}

// candidate pairs (node index, distance) used in beam search heaps.
type candidate struct {
	idx  uint64
	dist float32
}

// maxHeap — farthest element on top (for ef-size result set).
type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any           { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

// minHeap — closest element on top (priority queue of candidates to explore).
type minHeap []candidate

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any           { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

func (h *HnswIndex) beamSearch(query []float32, entry uint64, layer, ef int) []candidate {
	visited := make(map[uint64]bool)
	visited[entry] = true

	d0 := h.distance(query, entry)
	candidates := &minHeap{{idx: entry, dist: d0}}
	results := &maxHeap{{idx: entry, dist: d0}}
	heap.Init(candidates)
	heap.Init(results)

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(candidate)
		// If closest candidate is farther than worst result and we have ef results, stop.
		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}
		if int(c.idx) >= len(h.Neighbors) || layer >= len(h.Neighbors[c.idx]) {
			continue
		}
		for _, nb := range h.Neighbors[c.idx][layer] {
			if visited[nb] {
				continue
			}
			visited[nb] = true
			d := h.distance(query, nb)
			if results.Len() < ef || d < (*results)[0].dist {
				heap.Push(candidates, candidate{idx: nb, dist: d})
				heap.Push(results, candidate{idx: nb, dist: d})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}
	return *results
}

func sortCandidates(c []candidate) {
	// Insertion sort — ef is typically small (top_k * 5)
	for i := 1; i < len(c); i++ {
		j := i
		for j > 0 && c[j].dist < c[j-1].dist {
			c[j], c[j-1] = c[j-1], c[j]
			j--
		}
	}
}

// FlatSearch does exact brute-force scan over all vectors.
// Used as fallback when HNSW graph is empty (old-format files).
func (h *HnswIndex) FlatSearch(query []float32, topK int) []SearchResult {
	type hit struct {
		rowID uint64
		dist  float32
	}
	n := uint64(len(h.RowIDs))
	hits := make([]hit, 0, topK+1)

	worst := float32(math.MaxFloat32)
	for i := uint64(0); i < n; i++ {
		d := h.distance(query, i)
		if len(hits) < topK || d < worst {
			hits = append(hits, hit{h.RowIDs[i], d})
			// keep sorted, cap at topK+1, trim
			for j := len(hits) - 1; j > 0 && hits[j].dist < hits[j-1].dist; j-- {
				hits[j], hits[j-1] = hits[j-1], hits[j]
			}
			if len(hits) > topK {
				hits = hits[:topK]
			}
			if len(hits) == topK {
				worst = hits[topK-1].dist
			}
		}
	}
	out := make([]SearchResult, len(hits))
	for i, h := range hits {
		out[i] = SearchResult{RowID: h.rowID, Distance: h.dist}
	}
	return out
}
