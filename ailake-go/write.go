// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Write-side operations for AI-Lake tables (Phase N).
//
// The Go client is a pure-Go reader. Write operations that require Rust
// business logic (equality delete, schema evolution) are delegated to the
// `ailake` CLI binary:
//
//   Priority 1: AILAKE_BIN env var       — path to a specific `ailake` binary
//   Priority 2: `ailake` found in PATH   — system-wide install
//
// Both functions return ErrNoBinary when neither source resolves a binary.
package ailake

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// ErrNoBinary is returned when no `ailake` CLI binary is available.
var ErrNoBinary = errors.New("ailake: no CLI binary found (set AILAKE_BIN or add ailake to PATH)")

// AddColumnReq describes a column addition for EvolveSchema.
type AddColumnReq struct {
	Name           string // Iceberg column name
	Type           string // Iceberg type: "string", "int", "long", "float", "double", "boolean", …
	InitialDefault string // JSON literal (null, 0, 0.0, "unknown"); empty = null
}

// RenameColumnReq describes a column rename for EvolveSchema.
type RenameColumnReq struct {
	From string
	To   string
}

// DeleteWhere logically deletes all rows where `column` equals any value in
// `values`. Writes an Iceberg equality delete file via the `ailake` CLI.
//
// No data files are rewritten; deleted rows are masked at scan time.
func DeleteWhere(
	catalog *HadoopCatalog,
	namespace, table, column string,
	values []string,
) error {
	if len(values) == 0 {
		return nil
	}
	bin, err := resolveBin()
	if err != nil {
		return err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table

	args := []string{
		"--store", warehouse,
		"delete-where", tableID,
		"--col", column,
		"--vals", strings.Join(values, ","),
	}

	cmd := exec.Command(bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ailake delete-where: %w", err)
	}
	return nil
}

// EvolveSchema applies a metadata-only schema evolution to the table.
// Returns the new schema_id on success, or -1 if the CLI did not emit
// new_schema_id (e.g. a no-op evolution where nothing changed).
//
// addCols and renameCols may be empty if only one operation is desired.
func EvolveSchema(
	catalog *HadoopCatalog,
	namespace, table string,
	addCols []AddColumnReq,
	renameCols []RenameColumnReq,
) (int, error) {
	if len(addCols) == 0 && len(renameCols) == 0 {
		return 0, nil
	}
	bin, err := resolveBin()
	if err != nil {
		return 0, err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table

	args := []string{
		"--store", warehouse,
		"evolve", tableID,
	}

	// Build --add and --initial-default args in parallel order.
	for _, ac := range addCols {
		args = append(args, "--add", ac.Name+":"+ac.Type)
		if ac.InitialDefault != "" {
			args = append(args, "--initial-default", ac.InitialDefault)
		}
	}
	for _, rc := range renameCols {
		args = append(args, "--rename", rc.From+":"+rc.To)
	}

	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("ailake evolve: %w\n%s", err, out)
	}

	// Parse "new_schema_id: N" from stdout.
	newSchemaID := -1
	for _, line := range strings.Split(string(out), "\n") {
		var id int
		if _, err := fmt.Sscanf(strings.TrimSpace(line), "new_schema_id: %d", &id); err == nil {
			newSchemaID = id
			break
		}
	}
	return newSchemaID, nil
}

// WriteBatchOptions controls optional parameters for WriteBatch.
type WriteBatchOptions struct {
	// VecCol is the embedding column name (default "embedding").
	VecCol string
	// Metric is the distance metric: cosine | euclidean | dot (default "cosine").
	Metric string
	// Precision is the storage precision: f32 | f16 | i8 (default "f16").
	Precision string
	// EmbeddingModel is an optional label stored in Iceberg metadata.
	EmbeddingModel string
	// PartitionBy is a single partition column for simple partitioning.
	PartitionBy string
	// PartitionValue is the partition value when PartitionBy is set.
	PartitionValue string
	// FormatVersion is the Iceberg format version: 2 (default) or 3.
	FormatVersion int
	// FtsColumns are text columns to embed as Tantivy FTS index.
	FtsColumns []string
	// FtsTokenizer is the Tantivy tokenizer name (default "default").
	FtsTokenizer string
	// HnswM is the HNSW M parameter (0 = use table default).
	HnswM int
	// HnswEfConstruction is the HNSW ef_construction (0 = use table default).
	HnswEfConstruction int
	// PreNormalize normalizes vectors to unit L2 at write time.
	PreNormalize bool
	// Deferred builds the index asynchronously (Parquet committed immediately).
	Deferred bool
	// VectorCols enables multi-column (Phase 8 multimodal) write mode — e.g.
	// text + image embeddings on the same row, each with its own HNSW index.
	// When non-empty, VecCol/Metric/Precision are ignored (the CLI's
	// --vector-cols spec carries per-column metric, and multi-column mode
	// always writes F16).
	VectorCols []VectorColSpec
}

// VectorColSpec describes one vector column in a multi-column (Phase 8
// multimodal) write — e.g. text + image embeddings on the same row, each
// getting its own HNSW section in the same AI-Lake file.
type VectorColSpec struct {
	Column   string
	Dim      int
	Metric   string // default "cosine"
	Modality string // optional: text | image | audio | video
}

// WriteBatch writes a batch of rows and their embeddings to an AI-Lake table
// by delegating to the `ailake insert` CLI binary.
//
// parquetFile must be a local path to a Parquet file containing at least the
// columns named in opts.VecCol (required). The embeddings column in the file
// is used directly; opts.VecCol identifies which column holds the vectors.
func WriteBatch(
	catalog *HadoopCatalog,
	namespace, table, parquetFile string,
	opts WriteBatchOptions,
) error {
	bin, err := resolveBin()
	if err != nil {
		return err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table

	args := []string{
		"--store", warehouse,
		"insert", tableID, parquetFile,
	}
	if len(opts.VectorCols) > 0 {
		// Multi-column (Phase 8 multimodal) mode: --vector-cols carries per-column
		// metric and takes precedence over --embeddings, which the CLI ignores when
		// set. Precision is always F16 in this mode (same as the CLI's own default).
		specs := make([]string, len(opts.VectorCols))
		for i, vc := range opts.VectorCols {
			metric := vc.Metric
			if metric == "" {
				metric = "cosine"
			}
			spec := fmt.Sprintf("%s:%d:%s", vc.Column, vc.Dim, metric)
			if vc.Modality != "" {
				spec += ":" + vc.Modality
			}
			specs[i] = spec
		}
		args = append(args, "--vector-cols", strings.Join(specs, ","))
	} else {
		vecCol := opts.VecCol
		if vecCol == "" {
			vecCol = "embedding"
		}
		args = append(args, "--embeddings", vecCol)
		if opts.Metric != "" {
			args = append(args, "--metric", opts.Metric)
		}
		if opts.Precision != "" {
			args = append(args, "--precision", opts.Precision)
		}
		if opts.EmbeddingModel != "" {
			args = append(args, "--embedding-model", opts.EmbeddingModel)
		}
	}
	if opts.PartitionBy != "" {
		args = append(args, "--partition-by", opts.PartitionBy)
	}
	if opts.PartitionValue != "" {
		args = append(args, "--partition-value", opts.PartitionValue)
	}
	if opts.FormatVersion != 0 && opts.FormatVersion != 2 {
		args = append(args, "--format-version", fmt.Sprintf("%d", opts.FormatVersion))
	}
	if len(opts.FtsColumns) > 0 {
		args = append(args, "--fts-columns", strings.Join(opts.FtsColumns, ","))
		if opts.FtsTokenizer != "" && opts.FtsTokenizer != "default" {
			args = append(args, "--fts-tokenizer", opts.FtsTokenizer)
		}
	}
	if opts.HnswM > 0 {
		args = append(args, "--hnsw-m", fmt.Sprintf("%d", opts.HnswM))
	}
	if opts.HnswEfConstruction > 0 {
		args = append(args, "--hnsw-ef", fmt.Sprintf("%d", opts.HnswEfConstruction))
	}
	if opts.PreNormalize {
		args = append(args, "--pre-normalize")
	}
	if opts.Deferred {
		args = append(args, "--deferred")
	}

	// Capture stderr instead of piping straight to os.Stderr (as this used to do) so a
	// CLI-side rejection — e.g. the new NaN/Infinity embedding validation — reaches the
	// caller's error message, not just the terminal. Matches SearchText/SearchHybrid's
	// existing pattern below.
	if _, err := exec.Command(bin, args...).Output(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && len(exitErr.Stderr) > 0 {
			return fmt.Errorf("ailake insert: %w\nstderr: %s", err, exitErr.Stderr)
		}
		return fmt.Errorf("ailake insert: %w", err)
	}
	return nil
}

// CompactOptions controls optional parameters for Compact.
type CompactOptions struct {
	// TargetSize is the target output file size in bytes (0 = CLI default, 512 MiB).
	TargetSize int64
	// MinFiles is the minimum number of small files required to trigger compaction
	// (0 = CLI default, 4).
	MinFiles int
	// MaxFilesPerPass bounds peak RAM / HNSW rebuild cost (0 = CLI default, 20).
	MaxFilesPerPass int
	// Deferred writes the merged Parquet immediately and builds the HNSW index
	// in the background instead of blocking until it's fully built.
	Deferred bool
}

// compactResponse mirrors the JSON envelope `ailake compact --format json` emits.
type compactResponse struct {
	OK             bool `json:"ok"`
	FilesCompacted int  `json:"files_compacted"`
}

// Compact merges small files in an AI-Lake table into a larger file by
// delegating to the `ailake compact` CLI. Returns the number of files
// compacted (0 = nothing eligible).
func Compact(
	catalog *HadoopCatalog,
	namespace, table string,
	opts CompactOptions,
) (int, error) {
	bin, err := resolveBin()
	if err != nil {
		return 0, err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table

	args := []string{
		"--store", warehouse,
		"compact", tableID,
		"--format", "json",
	}
	if opts.TargetSize > 0 {
		args = append(args, "--target-size", fmt.Sprintf("%d", opts.TargetSize))
	}
	if opts.MinFiles > 0 {
		args = append(args, "--min-files", fmt.Sprintf("%d", opts.MinFiles))
	}
	if opts.MaxFilesPerPass > 0 {
		args = append(args, "--max-files-per-pass", fmt.Sprintf("%d", opts.MaxFilesPerPass))
	}
	if opts.Deferred {
		args = append(args, "--deferred")
	}

	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("ailake compact: %w\n%s", err, out)
	}

	var resp compactResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		return 0, fmt.Errorf("ailake compact: parsing JSON output: %w\n%s", err, out)
	}
	return resp.FilesCompacted, nil
}

// SearchHybridResult is a single hit from SearchHybrid (BM25+vector RRF fusion).
type SearchHybridResult struct {
	RowID    int64
	Distance float64
	FilePath string
}

// SearchHybrid runs a hybrid BM25+vector RRF search on an AI-Lake table.
// query is the f32 embedding vector; text is the BM25 query string.
// bm25Weight controls the BM25 weight in RRF (0.0 = pure vector, 1.0 = pure BM25).
// textColumn is the Parquet column used for BM25 scoring (default "chunk_text").
func SearchHybrid(
	catalog *HadoopCatalog,
	namespace, table string,
	query []float32,
	text string,
	topK int,
	bm25Weight float64,
	textColumn string,
) ([]SearchHybridResult, error) {
	if len(query) == 0 || text == "" {
		return nil, nil
	}
	bin, err := resolveBin()
	if err != nil {
		return nil, err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table
	if topK <= 0 {
		topK = 10
	}
	if textColumn == "" {
		textColumn = "chunk_text"
	}

	floatStrs := make([]string, len(query))
	for i, v := range query {
		floatStrs[i] = fmt.Sprintf("%g", v)
	}

	args := []string{
		"--store", warehouse,
		"search", tableID,
		"--query", strings.Join(floatStrs, ","),
		"--hybrid-text", text,
		"--text-column", textColumn,
		"--bm25-weight", fmt.Sprintf("%g", bm25Weight),
		"--top-k", fmt.Sprintf("%d", topK),
		"--format", "json",
	}

	out, err := exec.Command(bin, args...).Output()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && len(exitErr.Stderr) > 0 {
			return nil, fmt.Errorf("ailake search --hybrid-text: %w\nstderr: %s", err, exitErr.Stderr)
		}
		return nil, fmt.Errorf("ailake search --hybrid-text: %w", err)
	}

	type hit struct {
		RowID    int64   `json:"row_id"`
		Distance float64 `json:"distance"`
		FilePath string  `json:"file_path"`
	}
	type resp struct {
		Results []hit `json:"results"`
	}

	var r resp
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &r); err != nil {
		return nil, fmt.Errorf("ailake search --hybrid-text: parse response: %w", err)
	}

	results := make([]SearchHybridResult, 0, len(r.Results))
	for _, h := range r.Results {
		results = append(results, SearchHybridResult{
			RowID:    h.RowID,
			Distance: h.Distance,
			FilePath: h.FilePath,
		})
	}
	return results, nil
}

// SearchTextResult is a single FTS hit from SearchText.
type SearchTextResult struct {
	RowID    int64
	Score    float64 // BM25 score (higher = more relevant)
	FilePath string
}

// SearchText performs full-text search on an AI-Lake table.
// Uses the Tantivy FTS index when present (O(log N)); falls back to BM25
// brute-force for legacy files.
//
// textColumns is the list of Parquet columns to search; defaults to
// ["chunk_text"] when nil or empty.
func SearchText(
	catalog *HadoopCatalog,
	namespace, table, queryText string,
	textColumns []string,
	topK int,
) ([]SearchTextResult, error) {
	if queryText == "" {
		return nil, nil
	}
	bin, err := resolveBin()
	if err != nil {
		return nil, err
	}

	warehouse := catalog.Warehouse
	if isLocalPath(warehouse) && !filepath.IsAbs(warehouse) {
		if abs, absErr := filepath.Abs(warehouse); absErr == nil {
			warehouse = abs
		}
	}

	tableID := namespace + "." + table
	cols := "chunk_text"
	if len(textColumns) > 0 {
		cols = strings.Join(textColumns, ",")
	}
	if topK <= 0 {
		topK = 10
	}

	args := []string{
		"--store", warehouse,
		"search", tableID,
		"--text", queryText,
		"--text-columns", cols,
		"--top-k", fmt.Sprintf("%d", topK),
		"--format", "json",
	}

	out, err := exec.Command(bin, args...).Output()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && len(exitErr.Stderr) > 0 {
			return nil, fmt.Errorf("ailake search --text: %w\nstderr: %s", err, exitErr.Stderr)
		}
		return nil, fmt.Errorf("ailake search --text: %w", err)
	}

	// Parse JSON output: {"results":[{"rank":N,"row_id":N,"score":F,"file_path":"..."}]}
	type hit struct {
		RowID    int64   `json:"row_id"`
		Score    float64 `json:"score"`
		FilePath string  `json:"file_path"`
	}
	type resp struct {
		Results []hit `json:"results"`
	}

	var r resp
	outStr := strings.TrimSpace(string(out))
	if err := json.Unmarshal([]byte(outStr), &r); err != nil {
		return nil, fmt.Errorf("ailake search --text: parse response: %w", err)
	}

	results := make([]SearchTextResult, 0, len(r.Results))
	for _, h := range r.Results {
		results = append(results, SearchTextResult{
			RowID:    h.RowID,
			Score:    h.Score,
			FilePath: h.FilePath,
		})
	}
	return results, nil
}

// resolveBin returns the path to the `ailake` CLI binary.
// Checks AILAKE_BIN env first, then PATH.
// On non-Windows systems it also verifies the binary is executable.
func resolveBin() (string, error) {
	if bin := os.Getenv("AILAKE_BIN"); bin != "" {
		info, err := os.Stat(bin)
		if err != nil {
			return "", fmt.Errorf("ailake: AILAKE_BIN=%q not found: %w", bin, ErrNoBinary)
		}
		if runtime.GOOS != "windows" && info.Mode()&0111 == 0 {
			return "", fmt.Errorf("ailake: AILAKE_BIN=%q exists but is not executable: %w", bin, ErrNoBinary)
		}
		return bin, nil
	}
	bin, err := exec.LookPath("ailake")
	if err != nil {
		return "", ErrNoBinary
	}
	return bin, nil
}

// isLocalPath reports whether warehouse is a local filesystem path
// (not a URL like s3:// or az://) that needs to be resolved to absolute.
func isLocalPath(warehouse string) bool {
	return !strings.Contains(warehouse, "://") && !strings.HasPrefix(warehouse, `\\`)
}
