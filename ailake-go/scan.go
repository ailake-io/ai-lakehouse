// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Package ailake — Scan: vector search + full row fetch in one call.
//
// Scan() is the high-level API for RAG microservices. It runs Search() to get
// top-K pointers (RowID, Distance, FilePath) and then fetches the full Parquet
// rows for each pointer, returning all columns alongside _distance.
//
// The FIXED_LEN_BYTE_ARRAY vector column is automatically decoded to []float32
// (F16 → F32) when its name matches the table's declared vector column.
// All other columns are returned as native Go types:
//   int32/int64 → int64   |  float/double → float64
//   byte_array → string (if valid UTF-8) or []byte
//   fixed_len_byte_array → []byte
//   boolean → bool

package ailake

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	parquetgo "github.com/parquet-go/parquet-go"
)

// ScanRow is one result from Scan. Fields contains all Parquet columns.
// The vector column is decoded to []float32; other byte columns to string or []byte.
type ScanRow struct {
	RowID    uint64
	Distance float32
	FilePath string
	Fields   map[string]any
}

// fileHit groups a search result with its original position for re-sorting.
type fileHit struct {
	rowID    uint64
	distance float32
	origIdx  int
}

// Scan runs vector search and returns full Parquet rows alongside _distance.
// It is equivalent to Search() followed by FetchRows(), but more convenient.
func Scan(
	catalog *HadoopCatalog,
	namespace, table string,
	query []float32,
	opts SearchOptions,
) ([]ScanRow, error) {
	info, err := catalog.LoadTable(namespace, table)
	if err != nil {
		return nil, fmt.Errorf("ailake scan: load table info: %w", err)
	}

	dim := uint32(0)
	if d, err := strconv.ParseUint(info.VectorDim, 10, 32); err == nil {
		dim = uint32(d)
	}

	results, err := Search(catalog, namespace, table, query, opts)
	if err != nil {
		return nil, err
	}

	return FetchRows(results, catalog.Warehouse, info.VectorColumn, dim)
}

// FetchRows reads full Parquet rows for the given search results.
// vectorCol and dim are used to auto-decode the vector column (F16→F32).
// Pass empty string / 0 to skip vector decoding.
func FetchRows(
	results []FileSearchResult,
	warehouse string,
	vectorCol string,
	dim uint32,
) ([]ScanRow, error) {
	if len(results) == 0 {
		return nil, nil
	}

	// Group by file path — minimize file opens.
	byFile := make(map[string][]fileHit)
	for i, r := range results {
		filePath := resolveWarehousePath(warehouse, r.FilePath)
		byFile[filePath] = append(byFile[filePath], fileHit{r.RowID, r.Distance, i})
	}

	scanRows := make([]ScanRow, 0, len(results))

	for filePath, hits := range byFile {
		rows, err := readParquetRows(filePath, hits, vectorCol, dim)
		if err != nil {
			return nil, fmt.Errorf("ailake fetch rows %s: %w", filePath, err)
		}
		scanRows = append(scanRows, rows...)
	}

	// Restore nearest-first order.
	sort.Slice(scanRows, func(i, j int) bool {
		return scanRows[i].Distance < scanRows[j].Distance
	})

	return scanRows, nil
}

// readParquetRows reads specific rows from a Parquet file by 0-based row ID.
func readParquetRows(
	filePath string,
	hits []fileHit,
	vectorCol string,
	dim uint32,
) ([]ScanRow, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	pf, err := parquetgo.OpenFile(f, stat.Size())
	if err != nil {
		return nil, fmt.Errorf("open parquet: %w", err)
	}

	// Build target rowID → distance map.
	target := make(map[uint64]float32, len(hits))
	for _, h := range hits {
		target[h.rowID] = h.distance
	}

	// schema.Columns() returns ordered leaf column paths as [][]string.
	// For flat AI-Lake schemas each path is a single element; join with "." for nested.
	rawPaths := pf.Schema().Columns()
	colPaths := make([]string, len(rawPaths))
	for i, p := range rawPaths {
		colPaths[i] = strings.Join(p, ".")
	}

	var result []ScanRow
	rowOffset := uint64(0)

	for _, rg := range pf.RowGroups() {
		rgRows := uint64(rg.NumRows())

		// Skip row groups with no targets.
		hasTarget := false
		for rowID := range target {
			if rowID >= rowOffset && rowID < rowOffset+rgRows {
				hasTarget = true
				break
			}
		}
		if !hasTarget {
			rowOffset += rgRows
			continue
		}

		rows := rg.Rows()
		rowBuf := make([]parquetgo.Row, 1)
		localIdx := uint64(0)

		for {
			n, err := rows.ReadRows(rowBuf)
			if n > 0 {
				globalRowID := rowOffset + localIdx
				if dist, ok := target[globalRowID]; ok {
					fields := parquetRowToFields(rowBuf[0], colPaths, vectorCol, dim)
					result = append(result, ScanRow{
						RowID:    globalRowID,
						Distance: dist,
						FilePath: filePath,
						Fields:   fields,
					})
				}
				localIdx++
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				rows.Close()
				return nil, fmt.Errorf("read row %d: %w", rowOffset+localIdx, err)
			}
		}
		rows.Close()

		rowOffset += rgRows
	}

	return result, nil
}

// parquetRowToFields converts a parquet-go Row ([]Value) to map[string]any.
// The vector column (FIXED_LEN_BYTE_ARRAY, F16 encoded) is decoded to []float32.
func parquetRowToFields(
	row parquetgo.Row,
	colPaths []string,
	vectorCol string,
	dim uint32,
) map[string]any {
	fields := make(map[string]any, len(row))

	for _, v := range row {
		if v.IsNull() {
			continue
		}
		colIdx := v.Column()
		if colIdx < 0 || colIdx >= len(colPaths) {
			continue
		}
		name := colPaths[colIdx]

		switch v.Kind() {
		case parquetgo.Boolean:
			fields[name] = v.Boolean()
		case parquetgo.Int32:
			fields[name] = int64(v.Int32())
		case parquetgo.Int64:
			fields[name] = v.Int64()
		case parquetgo.Float:
			fields[name] = float64(v.Float())
		case parquetgo.Double:
			fields[name] = v.Double()
		case parquetgo.ByteArray:
			b := v.ByteArray()
			if utf8.Valid(b) {
				fields[name] = string(b)
			} else {
				cp := make([]byte, len(b))
				copy(cp, b)
				fields[name] = cp
			}
		case parquetgo.FixedLenByteArray:
			b := v.ByteArray()
			// Auto-decode vector column F16 → []float32.
			if name == vectorCol && dim > 0 && len(b) == int(dim)*2 {
				fields[name] = DecodeF16Vector(b, int(dim))
			} else {
				cp := make([]byte, len(b))
				copy(cp, b)
				fields[name] = cp
			}
		}
	}

	return fields
}

