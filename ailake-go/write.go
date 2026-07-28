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

// catalogOptFlagOrder maps CatalogOpts map keys to their `ailake` CLI flag
// name (matching ailake-cli's global --catalog/--rest-* flags — see
// docs/guides/REST_CATALOG.md), in a fixed order so the built command line is
// deterministic (map iteration order in Go is randomized).
var catalogOptFlagOrder = []struct{ key, flag string }{
	{"catalog", "--catalog"},
	{"rest-uri", "--rest-uri"},
	{"rest-prefix", "--rest-prefix"},
	{"rest-warehouse", "--rest-warehouse"},
	{"rest-auth", "--rest-auth"},
	{"rest-token", "--rest-token"},
	{"rest-oauth-token-endpoint", "--rest-oauth-token-endpoint"},
	{"rest-oauth-client-id", "--rest-oauth-client-id"},
	{"rest-oauth-client-secret", "--rest-oauth-client-secret"},
	{"rest-oauth-scope", "--rest-oauth-scope"},
}

// appendCatalogArgs appends --catalog/--rest-* CLI flags built from a
// CatalogOpts map (e.g. WriteBatchOptions.CatalogOpts). Nil/empty map = no
// flags added, unchanged default Hadoop-catalog behavior. Unrecognized keys
// are ignored rather than erroring, so a typo degrades silently to "flag not
// passed" instead of crashing a write — same tradeoff CatalogOpts callers on
// the JSON-envelope side (ailake-jni) already accept for unknown fields.
func appendCatalogArgs(args []string, opts map[string]string) []string {
	for _, kf := range catalogOptFlagOrder {
		if v, ok := opts[kf.key]; ok && v != "" {
			args = append(args, kf.flag, v)
		}
	}
	return args
}

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

// CreateTableOptions controls optional parameters for CreateTable.
type CreateTableOptions struct {
	// Metric is the distance metric: cosine | euclidean | dot (default "cosine").
	Metric string
	// Precision is the storage precision: f32 | f16 | i8 (default "f16").
	Precision string
	// Column is the vector column name (default "embedding").
	Column string
	// PreNormalize normalizes vectors to unit L2 at write time (recommended for cosine).
	PreNormalize bool
	// HnswM is the HNSW M parameter (0 = CLI default, 16).
	HnswM int
	// HnswEfConstruction is the HNSW ef_construction (0 = CLI default, 150).
	HnswEfConstruction int
	// PQOnly omits the raw vector column from Parquet files (index BLOB only).
	PQOnly bool
	// IVFResidual encodes (vec - coarse_centroid) per IVF cell instead of raw vec.
	IVFResidual bool
	// Modality tags the primary vector column: "text" | "image" | "audio" | "video".
	// Empty = unset.
	Modality string
	// FormatVersion is the Iceberg format version: 0 | 2 | 3; 0 means omit (CLI default 2).
	FormatVersion int
	// FtsColumns are text columns to embed as a Tantivy FTS index at write time.
	FtsColumns []string
	// FtsTokenizer is the Tantivy tokenizer name (default "default").
	FtsTokenizer string
	// CatalogOpts selects/configures a non-Hadoop catalog backend — see
	// WriteBatchOptions.CatalogOpts for accepted keys and docs/guides/REST_CATALOG.md.
	CatalogOpts map[string]string
}

// CreateTable creates an empty AI-Lake table with the given vector schema and
// policy, by delegating to the `ailake create` CLI. Unlike WriteBatch (which
// auto-creates a table on first insert with default policy), CreateTable is
// the only way to set PQOnly/IVFResidual/Modality or HNSW tuning before any
// data is written.
func CreateTable(
	catalog *HadoopCatalog,
	namespace, table string,
	dim int,
	opts CreateTableOptions,
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
		"create", tableID,
		"--dim", fmt.Sprintf("%d", dim),
	}
	if opts.Metric != "" {
		args = append(args, "--metric", opts.Metric)
	}
	if opts.Precision != "" {
		args = append(args, "--precision", opts.Precision)
	}
	if opts.Column != "" {
		args = append(args, "--column", opts.Column)
	}
	if opts.PreNormalize {
		args = append(args, "--pre-normalize")
	}
	if opts.HnswM > 0 {
		args = append(args, "--hnsw-m", fmt.Sprintf("%d", opts.HnswM))
	}
	if opts.HnswEfConstruction > 0 {
		args = append(args, "--hnsw-ef", fmt.Sprintf("%d", opts.HnswEfConstruction))
	}
	if opts.PQOnly {
		args = append(args, "--pq-only")
	}
	if opts.IVFResidual {
		args = append(args, "--ivf-residual")
	}
	if opts.Modality != "" {
		args = append(args, "--modality", opts.Modality)
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
	args = appendCatalogArgs(args, opts.CatalogOpts)

	if _, err := exec.Command(bin, args...).Output(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && len(exitErr.Stderr) > 0 {
			return fmt.Errorf("ailake create: %w\nstderr: %s", err, exitErr.Stderr)
		}
		return fmt.Errorf("ailake create: %w", err)
	}
	return nil
}

// DeleteWhere logically deletes all rows where `column` equals any value in
// `values`. Writes an Iceberg equality delete file via the `ailake` CLI.
//
// No data files are rewritten; deleted rows are masked at scan time.
//
// catalogOpts is variadic (0 or 1 map) for backward compatibility with
// existing callers — pass a single map[string]string to select/configure a
// non-Hadoop catalog backend (see WriteBatchOptions.CatalogOpts for accepted
// keys and docs/guides/REST_CATALOG.md). Regression: this was never wired to
// appendCatalogArgs when every other write-path function (WriteBatch,
// Compact, DecayMemories, Migrate, DeleteRows, AddVectorColumn,
// BackfillVectorColumn, CreateTable) was — DeleteWhere/EvolveSchema were the
// two remaining Hadoop-only holdouts in the write path.
func DeleteWhere(
	catalog *HadoopCatalog,
	namespace, table, column string,
	values []string,
	catalogOpts ...map[string]string,
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
	if len(catalogOpts) > 0 {
		args = appendCatalogArgs(args, catalogOpts[0])
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
//
// catalogOpts is variadic (0 or 1 map) for backward compatibility — see
// DeleteWhere's doc comment for why this and DeleteWhere were the two
// remaining Hadoop-only holdouts in the write path.
func EvolveSchema(
	catalog *HadoopCatalog,
	namespace, table string,
	addCols []AddColumnReq,
	renameCols []RenameColumnReq,
	catalogOpts ...map[string]string,
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
	if len(catalogOpts) > 0 {
		args = appendCatalogArgs(args, catalogOpts[0])
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
	// CatalogOpts selects/configures a non-Hadoop catalog backend (e.g. REST
	// Catalog — Polaris, Unity Catalog, BigLake, S3 Tables, Nessie, Gravitino).
	// Nil/empty = default Hadoop-style catalog, unchanged behavior. Keys:
	// "catalog", "rest-uri", "rest-prefix", "rest-warehouse", "rest-auth",
	// "rest-token", "rest-oauth-token-endpoint", "rest-oauth-client-id",
	// "rest-oauth-client-secret", "rest-oauth-scope". See docs/guides/REST_CATALOG.md.
	CatalogOpts map[string]string
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
	args = appendCatalogArgs(args, opts.CatalogOpts)

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
	// CatalogOpts selects/configures a non-Hadoop catalog backend — see
	// WriteBatchOptions.CatalogOpts for accepted keys and
	// docs/guides/REST_CATALOG.md. Nil/empty = default Hadoop catalog.
	CatalogOpts map[string]string
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
	args = appendCatalogArgs(args, opts.CatalogOpts)

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

// decayMemoriesResponse mirrors the JSON envelope `ailake decay-memories --format json` emits.
type decayMemoriesResponse struct {
	OK           bool `json:"ok"`
	FilesUpdated int  `json:"files_updated"`
}

// DecayMemories recomputes recency weights (exp(-λ×days_since_access)) across
// all memory files in the table by delegating to the `ailake decay-memories`
// CLI. Returns the number of files updated (0 = nothing changed).
func DecayMemories(
	catalog *HadoopCatalog,
	namespace, table string,
	lambda float32,
	catalogOpts map[string]string,
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
		"decay-memories", tableID,
		"--lambda", fmt.Sprintf("%g", lambda),
		"--format", "json",
	}
	args = appendCatalogArgs(args, catalogOpts)

	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("ailake decay-memories: %w\n%s", err, out)
	}

	var resp decayMemoriesResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		return 0, fmt.Errorf("ailake decay-memories: parsing JSON output: %w\n%s", err, out)
	}
	return resp.FilesUpdated, nil
}

// MigrateOptions controls optional parameters for Migrate.
type MigrateOptions struct {
	// OldColumn is the name of the existing embedding column (default "embedding").
	OldColumn string
	// NewColumn is the name for the migrated column — may equal OldColumn for
	// an in-place upgrade (default "embedding_v2").
	NewColumn string
	// TextColumn is the Parquet column holding raw text to re-embed (default "chunk_text").
	TextColumn string
	// Strategy is "atomic-replace" (lower storage) or "dual-write-then-cutover"
	// (zero downtime, default).
	Strategy string
	// BatchSize is the number of texts per embed-cmd call (default 512).
	BatchSize int
	// ModelName is stored in ailake.embedding-model after migration.
	ModelName string
	// ModelVersion is an optional version tag appended to ModelName (stored as "<name>@<version>").
	ModelVersion string
	// CatalogOpts selects/configures a non-Hadoop catalog backend — see
	// WriteBatchOptions.CatalogOpts for accepted keys and docs/guides/REST_CATALOG.md.
	CatalogOpts map[string]string
}

// Migrate re-embeds a table's vector column via an external embed command, by
// delegating to the `ailake migrate` CLI. embedCmd is a shell command that
// reads a JSON array of strings from stdin and writes a JSON array of float
// arrays to stdout.
func Migrate(
	catalog *HadoopCatalog,
	namespace, table, embedCmd string,
	opts MigrateOptions,
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
		"migrate", tableID,
		"--embed-cmd", embedCmd,
	}
	if opts.OldColumn != "" {
		args = append(args, "--old-column", opts.OldColumn)
	}
	if opts.NewColumn != "" {
		args = append(args, "--new-column", opts.NewColumn)
	}
	if opts.TextColumn != "" {
		args = append(args, "--text-column", opts.TextColumn)
	}
	if opts.Strategy != "" {
		args = append(args, "--strategy", opts.Strategy)
	}
	if opts.BatchSize > 0 {
		args = append(args, "--batch-size", fmt.Sprintf("%d", opts.BatchSize))
	}
	if opts.ModelName != "" {
		args = append(args, "--model-name", opts.ModelName)
	}
	if opts.ModelVersion != "" {
		args = append(args, "--model-version", opts.ModelVersion)
	}
	args = appendCatalogArgs(args, opts.CatalogOpts)

	if out, err := exec.Command(bin, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("ailake migrate: %w\n%s", err, out)
	}
	return nil
}

// DeleteRows marks rows as deleted in a V3 table using Iceberg Deletion
// Vectors, by delegating to the `ailake delete-rows` CLI. file is the Parquet
// data file path as reported by LoadTable (e.g. "data/part-00001.parquet").
// No-op when rowPositions is empty. Requires the table to have been created
// with format_version=3 — the CLI raises a clear error on a V2 table rather
// than corrupting it; use DeleteWhere (equality predicate) for V2 tables.
func DeleteRows(
	catalog *HadoopCatalog,
	namespace, table, file string,
	rowPositions []int,
	catalogOpts map[string]string,
) error {
	if len(rowPositions) == 0 {
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

	rows := make([]string, len(rowPositions))
	for i, r := range rowPositions {
		rows[i] = fmt.Sprintf("%d", r)
	}

	args := []string{
		"--store", warehouse,
		"delete-rows", tableID,
		"--file", file,
		"--rows", strings.Join(rows, ","),
	}
	args = appendCatalogArgs(args, catalogOpts)

	if out, err := exec.Command(bin, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("ailake delete-rows: %w\n%s", err, out)
	}
	return nil
}

// AddVectorColumnOptions controls optional parameters for AddVectorColumn.
type AddVectorColumnOptions struct {
	// Metric is the distance metric for the new column (default "cosine").
	Metric string
	// Precision is the storage precision for the new column (default "f16").
	Precision string
	// PreNormalize normalizes vectors to unit L2 at write time.
	PreNormalize bool
	// HnswM is the HNSW M parameter (0 = CLI default, 16).
	HnswM int
	// HnswEfConstruction is the HNSW ef_construction (0 = CLI default, 150).
	HnswEfConstruction int
	// CatalogOpts selects/configures a non-Hadoop catalog backend — see
	// WriteBatchOptions.CatalogOpts for accepted keys and docs/guides/REST_CATALOG.md.
	CatalogOpts map[string]string
}

// AddVectorColumn adds a new vector column to an existing table schema
// (no data files rewritten) by delegating to the `ailake add-vector-column`
// CLI. Old files return null for the new column until BackfillVectorColumn
// is run.
func AddVectorColumn(
	catalog *HadoopCatalog,
	namespace, table, column string,
	dim int,
	opts AddVectorColumnOptions,
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
		"add-vector-column", tableID,
		"--column", column,
		"--dim", fmt.Sprintf("%d", dim),
	}
	if opts.Metric != "" {
		args = append(args, "--metric", opts.Metric)
	}
	if opts.Precision != "" {
		args = append(args, "--precision", opts.Precision)
	}
	if opts.PreNormalize {
		args = append(args, "--pre-normalize")
	}
	if opts.HnswM > 0 {
		args = append(args, "--hnsw-m", fmt.Sprintf("%d", opts.HnswM))
	}
	if opts.HnswEfConstruction > 0 {
		args = append(args, "--hnsw-ef", fmt.Sprintf("%d", opts.HnswEfConstruction))
	}
	args = appendCatalogArgs(args, opts.CatalogOpts)

	if out, err := exec.Command(bin, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("ailake add-vector-column: %w\n%s", err, out)
	}
	return nil
}

// BackfillVectorColumnOptions controls optional parameters for BackfillVectorColumn.
type BackfillVectorColumnOptions struct {
	// TextColumn is the Parquet column holding raw text to embed (default "chunk_text").
	TextColumn string
	// BatchSize is the number of texts per embed-cmd call (default 512).
	BatchSize int
	// CatalogOpts selects/configures a non-Hadoop catalog backend — see
	// WriteBatchOptions.CatalogOpts for accepted keys and docs/guides/REST_CATALOG.md.
	CatalogOpts map[string]string
}

// BackfillVectorColumn backfills a new vector column in all existing files
// (rewriting each file with new embeddings) by delegating to the `ailake
// backfill-vector-column` CLI. column must already exist via AddVectorColumn.
// Idempotent: files already containing the column are skipped.
func BackfillVectorColumn(
	catalog *HadoopCatalog,
	namespace, table, column, embedCmd string,
	opts BackfillVectorColumnOptions,
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
		"backfill-vector-column", tableID,
		"--column", column,
		"--embed-cmd", embedCmd,
	}
	if opts.TextColumn != "" {
		args = append(args, "--text-column", opts.TextColumn)
	}
	if opts.BatchSize > 0 {
		args = append(args, "--batch-size", fmt.Sprintf("%d", opts.BatchSize))
	}
	args = appendCatalogArgs(args, opts.CatalogOpts)

	if out, err := exec.Command(bin, args...).CombinedOutput(); err != nil {
		return fmt.Errorf("ailake backfill-vector-column: %w\n%s", err, out)
	}
	return nil
}

// EstimateOptions controls optional parameters for Estimate.
type EstimateOptions struct {
	// HnswM is the HNSW M parameter (0 = CLI default, 16).
	HnswM int
	// PqM is the PQ sub-vectors M, used for PQ-only/IVF-PQ estimates
	// (0 = CLI default, dim/32 clamped to [8, dim]).
	PqM int
}

// EstimateResult is the parsed output of `ailake estimate --format json`.
type EstimateResult struct {
	Rows      int64
	Dim       int
	HnswM     int
	PqM       int
	// Estimates holds one entry per storage mode (F32, F16, I8, IVF-PQ
	// variants, PQ-only), each a raw map with keys: mode, vectors_bytes,
	// index_bytes, total_bytes, reduction_factor, recall_at_10, note.
	Estimates []map[string]any
}

// Estimate computes storage-usage estimates before writing (pure math, no
// I/O, no warehouse/catalog involved) by delegating to the `ailake estimate`
// CLI. rows supports K/M/B suffixes (e.g. "1M", "500K").
func Estimate(rows string, dim int, opts EstimateOptions) (*EstimateResult, error) {
	bin, err := resolveBin()
	if err != nil {
		return nil, err
	}

	args := []string{
		"estimate",
		"--rows", rows,
		"--dim", fmt.Sprintf("%d", dim),
		"--format", "json",
	}
	if opts.HnswM > 0 {
		args = append(args, "--hnsw-m", fmt.Sprintf("%d", opts.HnswM))
	}
	if opts.PqM > 0 {
		args = append(args, "--pq-m", fmt.Sprintf("%d", opts.PqM))
	}

	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("ailake estimate: %w\n%s", err, out)
	}

	var raw struct {
		Rows      int64            `json:"rows"`
		Dim       int              `json:"dim"`
		HnswM     int              `json:"hnsw_m"`
		PqM       int              `json:"pq_m"`
		Estimates []map[string]any `json:"estimates"`
	}
	if err := json.Unmarshal(out, &raw); err != nil {
		return nil, fmt.Errorf("ailake estimate: parsing JSON output: %w\n%s", err, out)
	}
	return &EstimateResult{
		Rows:      raw.Rows,
		Dim:       raw.Dim,
		HnswM:     raw.HnswM,
		PqM:       raw.PqM,
		Estimates: raw.Estimates,
	}, nil
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
