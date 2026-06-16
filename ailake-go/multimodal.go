// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// SearchMultimodal — cross-modal RRF fusion search for multi-column AI-Lake tables.
package ailake

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ModalQuery is one arm of a cross-modal RRF search.
type ModalQuery struct {
	Column string
	Query  []float32
	Weight float32
}

// RRFResult is returned by SearchMultimodal.
type RRFResult struct {
	RowID    uint64
	RRFScore float32
	FilePath string
}

// SearchMultimodal runs independent HNSW searches per column, then fuses results
// via Reciprocal Rank Fusion:  final_score = Σ weight_i / (60 + rank_i).
func SearchMultimodal(
	catalog *HadoopCatalog,
	namespace, table string,
	queries []ModalQuery,
	opts SearchOptions,
) ([]RRFResult, error) {
	if len(queries) == 0 {
		return nil, fmt.Errorf("ailake: SearchMultimodal requires at least one ModalQuery")
	}
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	if opts.PruningThreshold == 0 {
		opts.PruningThreshold = 0.8
	}

	info, err := catalog.LoadTable(namespace, table)
	if err != nil {
		return nil, fmt.Errorf("ailake: load table: %w", err)
	}
	entries, err := catalog.ListFiles(namespace, table)
	if err != nil {
		return nil, fmt.Errorf("ailake: list files: %w", err)
	}

	primaryMetric := metricFromString(info.VectorMetric)

	// Geometric pruning using primary column centroid with primary-column query vector.
	pruneQ := queries[0].Query
	for _, mq := range queries {
		if mq.Column == info.VectorColumn {
			pruneQ = mq.Query
			break
		}
	}

	var survivors []DataFileEntry
	for _, e := range entries {
		if len(e.Centroid) == 0 {
			survivors = append(survivors, e)
			continue
		}
		d := distanceByMetric(primaryMetric, pruneQ, e.Centroid)
		if d-e.Radius <= opts.PruningThreshold {
			survivors = append(survivors, e)
		}
	}

	hw := opts.hw()

	// rowKey deduplicates results across files.
	type rowKey struct {
		rowID    uint64
		filePath string
	}
	rrfAccum := make(map[rowKey]float32)

	for _, mq := range queries {
		w := mq.Weight
		if w == 0 {
			w = 1.0
		}

		// Collect hits for this column across all surviving files.
		var colHits []FileSearchResult
		for _, e := range survivors {
			hits, err := searchFileCol(catalog.Warehouse, namespace, table, e, mq, info.VectorColumn, opts, hw)
			if err != nil {
				return nil, fmt.Errorf("ailake: search file %s col %s: %w", e.Path, mq.Column, err)
			}
			colHits = append(colHits, hits...)
		}

		// Sort ascending by distance (lower = better match).
		sort.Slice(colHits, func(i, j int) bool {
			return colHits[i].Distance < colHits[j].Distance
		})

		// Accumulate RRF scores.
		for rank, hit := range colHits {
			k := rowKey{rowID: hit.RowID, filePath: hit.FilePath}
			rrfAccum[k] += w / float32(60+rank+1)
		}
	}

	results := make([]RRFResult, 0, len(rrfAccum))
	for k, score := range rrfAccum {
		results = append(results, RRFResult{
			RowID:    k.rowID,
			RRFScore: score,
			FilePath: k.filePath,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].RRFScore > results[j].RRFScore
	})
	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// searchFileCol searches one file for one column's HNSW index.
// Uses ExtraVectorIndexes for secondary columns.
func searchFileCol(
	warehouse, namespace, table string,
	entry DataFileEntry,
	mq ModalQuery,
	primaryCol string,
	opts SearchOptions,
	hw *HardwareProfile,
) ([]FileSearchResult, error) {
	filePath := entry.Path
	if !filepath.IsAbs(filePath) {
		filePath = filepath.Join(warehouse, namespace, table, filePath)
	}

	var hnswOffset, hnswLen uint64
	var dim uint32

	if mq.Column == "" || mq.Column == primaryCol {
		if entry.HnswOffset == nil || entry.HnswLen == nil {
			return nil, nil
		}
		hnswOffset = *entry.HnswOffset
		hnswLen = *entry.HnswLen
		dim = entry.VectorDim
	} else {
		found := false
		for _, xi := range entry.ExtraVectorIndexes {
			if xi.Column == mq.Column {
				hnswOffset = xi.HnswOffset
				hnswLen = xi.HnswLen
				dim = xi.Dim
				found = true
				break
			}
		}
		if !found || hnswOffset == 0 || hnswLen == 0 {
			return nil, nil
		}
	}

	hits, err := searchFileAtOffset(filePath, hnswOffset, hnswLen, dim, mq.Query, opts, hw)
	if err != nil {
		return nil, err
	}

	out := make([]FileSearchResult, len(hits))
	for i, h := range hits {
		out[i] = FileSearchResult{RowID: h.RowID, Distance: h.Distance, FilePath: entry.Path}
	}
	return out, nil
}

// searchFileAtOffset reads the HNSW/IVF-PQ blob at an absolute file offset and searches it.
// hnswOffset is the absolute byte position of the HNSW blob (NOT the AILK header).
func searchFileAtOffset(
	filePath string,
	hnswOffset, hnswLen uint64,
	dim uint32,
	query []float32,
	opts SearchOptions,
	hw *HardwareProfile,
) ([]SearchResult, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// AILK header precedes the centroid+blob section:
	//   ailk_header_pos = hnswOffset - HeaderSize - (dim+1)*4
	centroidBytes := (uint64(dim) + 1) * 4
	ailkHeaderPos := hnswOffset - uint64(HeaderSize) - centroidBytes

	headerBuf := make([]byte, HeaderSize)
	if _, err := f.ReadAt(headerBuf, int64(ailkHeaderPos)); err != nil {
		return nil, fmt.Errorf("read AILK header: %w", err)
	}
	header, err := ParseHeaderBytes(headerBuf)
	if err != nil {
		return nil, err
	}

	indexBuf := make([]byte, hnswLen)
	if _, err := f.ReadAt(indexBuf, int64(hnswOffset)); err != nil {
		return nil, fmt.Errorf("read index blob: %w", err)
	}

	if header.IsIvfPq() {
		idx, err := DeserializeIvfPq(indexBuf)
		if err != nil {
			return nil, fmt.Errorf("deserialize IVF-PQ: %w", err)
		}
		if hw.HasGPU() {
			if serverURL := gpuServerURL(); serverURL != "" {
				fsr, err := searchViaHTTP(serverURL, filePath, query, opts.TopK)
				if err == nil {
					out := make([]SearchResult, len(fsr))
					for i, r := range fsr {
						out[i] = SearchResult{RowID: r.RowID, Distance: r.Distance}
					}
					return out, nil
				}
			}
		}
		return idx.Search(query, opts.TopK, int(idx.Config.Nprobe)), nil
	}

	idx, err := DeserializeHnsw(indexBuf)
	if err != nil {
		return nil, fmt.Errorf("deserialize HNSW: %w", err)
	}
	if idx.HasEntry && len(idx.Neighbors) > 0 {
		return idx.Search(query, opts.TopK, opts.efSearch()), nil
	}
	return idx.FlatSearch(query, opts.TopK), nil
}
