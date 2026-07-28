// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Package ailake — applies Iceberg logical deletes (Phase H) to Search/Scan
// results: Deletion Vectors (position deletes, V3) and equality deletes.
//
// Regression fixed here: this package's native Search/Scan never applied
// either mechanism at all — a row logically deleted via DeleteWhere/DeleteRows
// still came back from Search/Scan, even though the real `ailake` CLI's own
// search always masked it (documented as a known limitation in README.md
// until now). Mirrors ailake-query/src/dv.rs (deletion vectors) and
// ailake-query/src/equality_delete.rs (equality deletes) exactly.
package ailake

import (
	"fmt"
	"os"
	"strconv"

	"github.com/RoaringBitmap/roaring"
	goavro "github.com/linkedin/goavro/v2"
)

// loadDeletionVector fetches and deserializes a Deletion Vector bitmap from a
// Puffin ".dvd" file. Returns the row positions (0-based within the data
// file) that have been deleted. Mirrors ailake_query::dv::load_deletion_vector.
//
// The Roaring Bitmap wire format ("RoaringFormatSpec") is cross-language
// portable — the same bytes Rust's `roaring` crate (serialize_into) and this
// package's github.com/RoaringBitmap/roaring (ReadFrom) both implement.
func loadDeletionVector(warehouse string, dv *DeletionVectorRef) (*roaring.Bitmap, error) {
	path := resolveWarehousePath(warehouse, dv.Path)
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("ailake: open deletion vector file %s: %w", path, err)
	}
	defer f.Close()

	buf := make([]byte, dv.Length)
	if _, err := f.ReadAt(buf, int64(dv.Offset)); err != nil {
		return nil, fmt.Errorf("ailake: read deletion vector blob %s (offset=%d, length=%d): %w",
			path, dv.Offset, dv.Length, err)
	}

	bm := roaring.New()
	if _, err := bm.FromBuffer(buf); err != nil {
		return nil, fmt.Errorf("ailake: deserialize deletion vector bitmap %s: %w", path, err)
	}
	return bm, nil
}

// EqualityDeleteFilter is an in-memory predicate built from one or more
// Iceberg equality delete files (Phase H). Mirrors
// ailake_query::equality_delete::EqualityDeleteFilter.
//
// A row is deleted when, for every column present in both the filter and the
// row, the row's value is a member of that column's delete-value set (AND
// across columns; a row that matches no filter column at all is not
// deleted — a filter column absent from the row is skipped, not treated as a
// non-match).
type EqualityDeleteFilter struct {
	// column name → set of string-normalized values to delete.
	values map[string]map[string]struct{}
}

// IsEmpty reports whether the filter has no predicates (nothing to delete).
func (f *EqualityDeleteFilter) IsEmpty() bool {
	return f == nil || len(f.values) == 0
}

// LoadEqualityDeleteFilter downloads and parses every equality delete file
// referenced by files, merging them into one filter. Pass the result of
// HadoopCatalog.ListEqualityDeletes.
func LoadEqualityDeleteFilter(warehouse string, files []EqualityDeleteFile) (*EqualityDeleteFilter, error) {
	f := &EqualityDeleteFilter{values: make(map[string]map[string]struct{})}
	for _, edf := range files {
		path := resolveWarehousePath(warehouse, edf.Path)
		pairs, err := readEqualityDeleteValues(path)
		if err != nil {
			return nil, fmt.Errorf("ailake: read equality delete file %s: %w", path, err)
		}
		for _, p := range pairs {
			set, ok := f.values[p.column]
			if !ok {
				set = make(map[string]struct{})
				f.values[p.column] = set
			}
			set[p.value] = struct{}{}
		}
	}
	return f, nil
}

type equalityDeleteValue struct {
	column string
	value  string
}

// readEqualityDeleteValues reads (column_name, value) pairs from an equality
// delete Avro file — a standard Avro OCF with one flat record `{col: value}`
// per deleted value (written by ailake_catalog::write_equality_delete_avro).
// Values are stringified the same way Rust's read_equality_delete_values does
// (Display/to_string for numeric types) so the predicate set compares equal
// against row values stringified the same way in shouldDeleteRow below.
func readEqualityDeleteValues(path string) ([]equalityDeleteValue, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ocf, err := goavro.NewOCFReader(f)
	if err != nil {
		return nil, fmt.Errorf("avro: %w", err)
	}

	var pairs []equalityDeleteValue
	for ocf.Scan() {
		raw, err := ocf.Read()
		if err != nil {
			return nil, err
		}
		rec, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		// Schema has exactly one field (the predicate column) per
		// write_equality_delete_avro — iterate the single key/value pair.
		for col, val := range rec {
			pairs = append(pairs, equalityDeleteValue{column: col, value: avroScalarToString(val)})
		}
	}
	return pairs, ocf.Err()
}

// avroScalarToString stringifies a goavro-decoded scalar the same way Rust's
// arrow ToString/Display does for the matching Arrow type, so equality
// comparisons against Parquet-decoded row values (see ScanRow.Fields,
// stringified the same way in shouldDeleteRow) line up exactly.
func avroScalarToString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case int32:
		return strconv.FormatInt(int64(t), 10)
	case int64:
		return strconv.FormatInt(t, 10)
	case float32:
		// 'f' (never scientific notation), shortest round-trip precision —
		// matches Rust's f32::to_string()/Display for normal-range values
		// much more closely than Go's default %v (%g verb, which switches to
		// scientific notation for very large/small magnitudes).
		return strconv.FormatFloat(float64(t), 'f', -1, 32)
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", t)
	}
}

// shouldDeleteRow reports whether fields (a decoded ScanRow.Fields map)
// matches this filter's delete predicate. Mirrors
// EqualityDeleteFilter::should_delete_row exactly: a row is deleted only when
// at least one filter column is present in fields AND every filter column
// present in fields matches its delete-value set (AND semantics; a mismatch
// on any present column short-circuits to "not deleted").
func (f *EqualityDeleteFilter) shouldDeleteRow(fields map[string]any) bool {
	if f.IsEmpty() {
		return false
	}
	anyColumnFound := false
	for col, deleteValues := range f.values {
		v, ok := fields[col]
		if !ok {
			continue // column absent from this row's schema — skip (schema evolution)
		}
		anyColumnFound = true
		if v == nil {
			return false // null never matches — AND tuple fails
		}
		s := avroScalarToString(v)
		if _, matches := deleteValues[s]; !matches {
			return false // column mismatch — AND tuple fails
		}
	}
	return anyColumnFound
}
