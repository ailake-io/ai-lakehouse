// SPDX-License-Identifier: MIT OR Apache-2.0
// Package ailake provides a Go reader for AI-Lake Format files.
//
// AI-Lake files are standard Apache Parquet files extended with an AILK
// section containing a vector index (HNSW or IVF-PQ) and geometric
// statistics for file pruning.
//
// Reference spec: docs/specs/FILE_FORMAT.md
//
// Usage:
//
//	catalog := &ailake.HadoopCatalog{Warehouse: "/data/warehouse"}
//	results, err := ailake.Search(catalog, "default", "docs", query, 10, 0.8)
package ailake

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// SearchResult is returned by Search.
type FileSearchResult struct {
	RowID    uint64
	Distance float32
	FilePath string
}

// SearchOptions controls search behaviour.
type SearchOptions struct {
	TopK             int
	EfSearch         int     // HNSW ef_search (default: TopK*5)
	PruningThreshold float32 // geometric pruning (default: 0.8)
}

func (o *SearchOptions) efSearch() int {
	if o.EfSearch > 0 {
		return o.EfSearch
	}
	return o.TopK * 5
}

// Search runs geometric pruning + HNSW/flat search over an AI-Lake table.
//
// query must have the same dimensionality as the table's vector column.
func Search(
	catalog *HadoopCatalog,
	namespace, table string,
	query []float32,
	opts SearchOptions,
) ([]FileSearchResult, error) {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	if opts.PruningThreshold == 0 {
		opts.PruningThreshold = 0.8
	}

	// Load candidate files from catalog
	entries, err := catalog.ListFiles(namespace, table)
	if err != nil {
		return nil, fmt.Errorf("ailake: list files: %w", err)
	}

	// Detect metric from catalog (needed for pruning)
	info, err := catalog.LoadTable(namespace, table)
	if err != nil {
		return nil, fmt.Errorf("ailake: load table: %w", err)
	}
	metric := metricFromString(info.VectorMetric)

	// Geometric pruning
	var survivors []DataFileEntry
	for _, e := range entries {
		if len(e.Centroid) == 0 {
			survivors = append(survivors, e) // no centroid → can't prune
			continue
		}
		d := distanceByMetric(metric, query, e.Centroid)
		if d-e.Radius <= opts.PruningThreshold {
			survivors = append(survivors, e)
		}
	}

	// Per-file HNSW search
	var all []FileSearchResult
	for _, e := range survivors {
		hits, err := searchFile(catalog.Warehouse, namespace, table, e, query, opts)
		if err != nil {
			return nil, fmt.Errorf("ailake: search file %s: %w", e.Path, err)
		}
		all = append(all, hits...)
	}

	// Global top-K merge
	sort.Slice(all, func(i, j int) bool { return all[i].Distance < all[j].Distance })
	if len(all) > opts.TopK {
		all = all[:opts.TopK]
	}
	return all, nil
}

func searchFile(
	warehouse, namespace, table string,
	entry DataFileEntry,
	query []float32,
	opts SearchOptions,
) ([]FileSearchResult, error) {
	// Resolve absolute path
	filePath := entry.Path
	if !filepath.IsAbs(filePath) {
		filePath = filepath.Join(warehouse, namespace+".db", table, filePath)
	}

	if entry.HnswOffset == nil || entry.HnswLen == nil {
		return nil, nil // indexing not complete
	}

	// Read AILK section from file
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read AILK header at hnsw_offset
	headerBuf := make([]byte, HeaderSize)
	if _, err := f.ReadAt(headerBuf, int64(*entry.HnswOffset)); err != nil {
		return nil, fmt.Errorf("read AILK header: %w", err)
	}
	header, err := ParseHeaderBytes(headerBuf)
	if err != nil {
		return nil, err
	}

	// Read index blob
	indexStart := int64(*entry.HnswOffset) + int64(header.HnswOffset)
	indexBuf := make([]byte, header.HnswLen)
	if _, err := f.ReadAt(indexBuf, indexStart); err != nil {
		return nil, fmt.Errorf("read index blob: %w", err)
	}

	var hits []SearchResult
	if header.IsIvfPq() {
		idx, err := DeserializeIvfPq(indexBuf)
		if err != nil {
			return nil, fmt.Errorf("deserialize IVF-PQ: %w", err)
		}
		hits = idx.Search(query, opts.TopK, int(idx.Config.Nprobe))
	} else {
		idx, err := DeserializeHnsw(indexBuf)
		if err != nil {
			return nil, fmt.Errorf("deserialize HNSW: %w", err)
		}
		if idx.HasEntry && len(idx.Neighbors) > 0 {
			hits = idx.Search(query, opts.TopK, opts.efSearch())
		} else {
			hits = idx.FlatSearch(query, opts.TopK)
		}
	}

	out := make([]FileSearchResult, len(hits))
	for i, h := range hits {
		out[i] = FileSearchResult{
			RowID:    h.RowID,
			Distance: h.Distance,
			FilePath: entry.Path,
		}
	}
	return out, nil
}

// ReadAilakeHeader reads and validates the AILK header from an AI-Lake file.
// Useful for introspection and compatibility checks.
func ReadAilakeHeader(path string) (*AilakeHeader, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Scan backwards from the last 8 bytes of the Parquet footer to find the
	// ailake.footer_offset KV entry. For simplicity, scan for AILK magic bytes
	// from the end of file (before the Parquet footer).
	// Full implementation would parse the Parquet footer KV metadata.
	return scanForAilkHeader(data)
}

// scanForAilkHeader finds the first AILK header by scanning for the magic bytes.
// This is a fallback for files where the Parquet footer KV is not available.
func scanForAilkHeader(data []byte) (*AilakeHeader, error) {
	magic := []byte{'A', 'I', 'L', 'K'}
	for i := len(data) - HeaderSize; i >= 0; i-- {
		if data[i] == magic[0] && i+4 <= len(data) {
			if data[i+1] == magic[1] && data[i+2] == magic[2] && data[i+3] == magic[3] {
				h, err := ParseHeaderBytes(data[i : i+HeaderSize])
				if err == nil {
					return h, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("ailake: no AILK header found in file")
}

func metricFromString(s string) uint8 {
	switch s {
	case "euclidean":
		return MetricEuclidean
	case "dotproduct", "dot":
		return MetricDotProduct
	default:
		return MetricCosine
	}
}

// f32FromKVHint parses a KV metadata float value.
func f32FromKVHint(s string) float32 {
	var v float64
	_, _ = fmt.Sscanf(s, "%f", &v)
	return float32(v)
}

// u64FromKVHint parses a KV metadata uint64 value.
func u64FromKVHint(s string) uint64 {
	var v uint64
	_, _ = fmt.Sscanf(s, "%d", &v)
	return v
}

// f16ToF32 converts an IEEE 754 half-precision value to float32.
// Used when decoding vector columns stored as F16.
func f16ToF32(bits uint16) float32 {
	sign := uint32(bits>>15) << 31
	exp := uint32((bits>>10)&0x1F)
	mant := uint32(bits & 0x3FF)

	var f32bits uint32
	if exp == 0 {
		if mant == 0 {
			f32bits = sign
		} else {
			// Denormalized
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f32bits = sign | ((exp + 112) << 23) | (mant << 13)
		}
	} else if exp == 0x1F {
		f32bits = sign | 0x7F800000 | (mant << 13) // inf / nan
	} else {
		f32bits = sign | ((exp + 112) << 23) | (mant << 13)
	}
	return math.Float32frombits(f32bits)
}

// DecodeF16Vector decodes a FIXED_LEN_BYTE_ARRAY F16 Parquet column value to []float32.
func DecodeF16Vector(raw []byte, dim int) []float32 {
	if len(raw) < dim*2 {
		return nil
	}
	out := make([]float32, dim)
	for i := range out {
		bits := binary.LittleEndian.Uint16(raw[i*2:])
		out[i] = f16ToF32(bits)
	}
	return out
}
