// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// Iceberg catalog reader for AI-Lake tables (HadoopCatalog / local / S3).
//
// Reads:
//   metadata/version-hint.text   → current version N
//   metadata/vN.metadata.json    → current-snapshot-id + snapshots array
//   metadata/snap-{id}-1.avro    → manifest list (Avro OCF)
//   metadata/{snap_id}-m0.avro   → manifest file (Avro OCF)
//   key_metadata bytes           → JSON-encoded AilakeEntryExt
package ailake

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	goavro "github.com/linkedin/goavro/v2"
)

// DataFileEntry mirrors ailake_catalog::provider::DataFileEntry.
type DataFileEntry struct {
	Path               string
	RecordCount        uint64
	FileSizeBytes      uint64
	Centroid           []float32 // decoded from centroid_b64
	Radius             float32
	HnswOffset         *uint64
	HnswLen            *uint64
	VectorColumn       string
	VectorDim          uint32
	ExtraVectorIndexes []ExtraVectorIndex // secondary vector columns (Phase 8)
	IndexStatus        string             // "ready" | "indexing" | "failed"
	IndexError         string             // non-empty only when IndexStatus == "failed"
	BatchID            string
	EmbeddingModel     string // "<name>" or "<name>@<version>"; empty if not set
	PartitionValue     string // agent_id or other partition value (Phase 9)
}

// PartitionDef mirrors ailake_core::PartitionDef.
// Transform is "identity" or "truncate[W]" (e.g. "truncate[4]").
type PartitionDef struct {
	Column     string `json:"column"`
	Transform  string `json:"transform"`
	ColumnType string `json:"column_type"` // Iceberg type: "string", "int", "long", …
}

// SchemaField mirrors one field in the Iceberg table schema.
type SchemaField struct {
	ID      int    `json:"id"`
	Name    string `json:"name"`
	Type    string `json:"type"`    // Iceberg primitive type string
	Required bool  `json:"required"`
}

// TableInfo mirrors the JSON output of "ailake info --format json".
type TableInfo struct {
	Table           string         `json:"table"`
	Location        string         `json:"location"`
	VectorColumn    string         `json:"vector_column"`
	VectorDim       string         `json:"vector_dim"`
	VectorMetric    string         `json:"vector_metric"`
	EmbeddingModel  string         `json:"embedding_model,omitempty"`
	Files           int            `json:"files"`
	IndexedFiles    int            `json:"indexed_files"`
	Rows            uint64         `json:"rows"`
	SizeBytes       uint64         `json:"size_bytes"`
	SnapshotID      *int64         `json:"snapshot_id"`
	FormatVersion   int            `json:"format_version"`   // 2 or 3
	PartitionFields []PartitionDef `json:"partition_fields"` // empty for unpartitioned tables
	SchemaFields    []SchemaField  `json:"schema_fields"`    // current schema fields
}

// HadoopCatalog reads an AI-Lake table from a local filesystem path.
// The warehouse root is the directory containing namespace.db directories.
type HadoopCatalog struct {
	Warehouse string // local path, e.g. "/data/warehouse"
}

func (c *HadoopCatalog) tableDir(namespace, name string) string {
	return filepath.Join(c.Warehouse, namespace, name)
}

// LoadTable reads table metadata and returns TableInfo + current snapshot ID.
func (c *HadoopCatalog) LoadTable(namespace, name string) (*TableInfo, error) {
	dir := c.tableDir(namespace, name)
	meta, err := c.readMetadata(dir)
	if err != nil {
		return nil, err
	}

	info := &TableInfo{
		Table:         namespace + "." + name,
		Location:      dir,
		FormatVersion: 2, // default; overwritten below if present
	}
	if props, ok := meta["properties"].(map[string]any); ok {
		info.VectorColumn, _ = props["ailake.vector-column"].(string)
		info.VectorDim, _ = props["ailake.vector-dim"].(string)
		info.VectorMetric, _ = props["ailake.vector-metric"].(string)
		info.EmbeddingModel, _ = props["ailake.embedding-model"].(string)
	}
	if fv, ok := meta["format-version"].(float64); ok {
		info.FormatVersion = int(fv)
	}
	if sid, ok := meta["current-snapshot-id"].(float64); ok {
		id := int64(sid)
		info.SnapshotID = &id
	}

	// Parse current schema fields for schema-evolution context.
	currentSchemaID := -1
	if v, ok := meta["current-schema-id"].(float64); ok {
		currentSchemaID = int(v)
	}
	if schemas, ok := meta["schemas"].([]any); ok {
		for _, s := range schemas {
			sm, ok := s.(map[string]any)
			if !ok {
				continue
			}
			if id, ok := sm["schema-id"].(float64); !ok || int(id) != currentSchemaID {
				continue
			}
			if fields, ok := sm["fields"].([]any); ok {
				for _, f := range fields {
					fm, ok := f.(map[string]any)
					if !ok {
						continue
					}
					sf := SchemaField{
						Name:     str(fm["name"]),
						Required: boolVal(fm["required"]),
					}
					if id, ok := fm["id"].(float64); ok {
						sf.ID = int(id)
					}
					// type can be a string or nested object; stringify either way.
					switch t := fm["type"].(type) {
					case string:
						sf.Type = t
					case map[string]any:
						if raw, err := json.Marshal(t); err == nil {
							sf.Type = string(raw)
						}
					}
					info.SchemaFields = append(info.SchemaFields, sf)
				}
			}
			break
		}
	}

	// Parse partition fields from current partition spec + schema.
	// Build field-id → schema field type map for column_type lookup.
	fieldTypeByID := make(map[int]string, len(info.SchemaFields))
	for _, sf := range info.SchemaFields {
		fieldTypeByID[sf.ID] = sf.Type
	}
	// Also build name → field for name-based fallback.
	fieldTypeByName := make(map[string]string, len(info.SchemaFields))
	for _, sf := range info.SchemaFields {
		fieldTypeByName[sf.Name] = sf.Type
	}

	defaultSpecID := -1
	if v, ok := meta["default-spec-id"].(float64); ok {
		defaultSpecID = int(v)
	}
	if specs, ok := meta["partition-specs"].([]any); ok {
		for _, s := range specs {
			sm, ok := s.(map[string]any)
			if !ok {
				continue
			}
			if id, ok := sm["spec-id"].(float64); !ok || int(id) != defaultSpecID {
				continue
			}
			if fields, ok := sm["fields"].([]any); ok {
				for _, f := range fields {
					fm, ok := f.(map[string]any)
					if !ok {
						continue
					}
					pd := PartitionDef{
						Column:    str(fm["name"]),
						Transform: str(fm["transform"]),
					}
					// source-id links to schema field id for type resolution.
					if srcID, ok := fm["source-id"].(float64); ok {
						pd.ColumnType = fieldTypeByID[int(srcID)]
					}
					if pd.ColumnType == "" {
						pd.ColumnType = fieldTypeByName[pd.Column]
					}
					if pd.ColumnType == "" {
						pd.ColumnType = "string"
					}
					info.PartitionFields = append(info.PartitionFields, pd)
				}
			}
			break
		}
	}

	return info, nil
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func boolVal(v any) bool {
	b, _ := v.(bool)
	return b
}

// ListFiles returns all DataFileEntry for the current snapshot.
func (c *HadoopCatalog) ListFiles(namespace, name string) ([]DataFileEntry, error) {
	dir := c.tableDir(namespace, name)
	meta, err := c.readMetadata(dir)
	if err != nil {
		return nil, err
	}

	// Find current snapshot → manifest-list path
	currentSnapID, _ := meta["current-snapshot-id"].(float64)
	snapshots, _ := meta["snapshots"].([]any)
	manifestList := ""
	for _, s := range snapshots {
		snap, ok := s.(map[string]any)
		if !ok {
			continue
		}
		if sid, _ := snap["snapshot-id"].(float64); sid == currentSnapID {
			manifestList, _ = snap["manifest-list"].(string)
			break
		}
	}
	if manifestList == "" {
		return nil, errors.New("catalog: no manifest-list found for current snapshot")
	}

	// manifest-list path may be absolute or relative to the warehouse root
	// (the Rust catalog writer always stores paths relative to the warehouse
	// root, e.g. "default/docs/metadata/snap-....avro" — NOT relative to the
	// table directory, which would double-prefix namespace/table).
	manifestListPath := c.resolveAvroPath(manifestList)

	// Read manifest list → list of manifest file paths
	manifestPaths, err := readManifestList(manifestListPath)
	if err != nil {
		return nil, fmt.Errorf("catalog: read manifest list: %w", err)
	}

	// Read each manifest file
	var entries []DataFileEntry
	for _, mp := range manifestPaths {
		mp = c.resolveAvroPath(mp)
		fileEntries, err := readManifestFile(mp)
		if err != nil {
			return nil, fmt.Errorf("catalog: read manifest %s: %w", mp, err)
		}
		entries = append(entries, fileEntries...)
	}
	return entries, nil
}

func (c *HadoopCatalog) readMetadata(tableDir string) (map[string]any, error) {
	// Read version-hint.text to get current version number
	hintPath := filepath.Join(tableDir, "metadata", "version-hint.text")
	hintBytes, err := os.ReadFile(hintPath)
	if err != nil {
		return nil, fmt.Errorf("catalog: read version-hint: %w", err)
	}
	version := strings.TrimSpace(string(hintBytes))
	metaPath := filepath.Join(tableDir, "metadata", "v"+version+".metadata.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("catalog: read metadata.json: %w", err)
	}
	var m map[string]any
	return m, json.Unmarshal(data, &m)
}

// resolveAvroPath resolves a manifest-list/manifest-file path emitted by the
// Rust catalog writer. These paths are always relative to the warehouse
// root (e.g. "default/docs/metadata/snap-....avro"), never to the table
// directory — joining against the table directory would double-prefix
// namespace/table (see hadoop.rs's table_root()/manifest_list_path).
func (c *HadoopCatalog) resolveAvroPath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(c.Warehouse, path)
}

// readManifestList reads an Iceberg manifest list (Avro OCF) and returns manifest file paths.
func readManifestList(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ocf, err := goavro.NewOCFReader(f)
	if err != nil {
		return nil, fmt.Errorf("avro: %w", err)
	}
	var paths []string
	for ocf.Scan() {
		raw, err := ocf.Read()
		if err != nil {
			return nil, err
		}
		rec, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if p, ok := rec["manifest_path"].(string); ok {
			paths = append(paths, p)
		}
	}
	return paths, ocf.Err()
}

// ExtraVectorIndex mirrors ailake_catalog::provider::ExtraVectorIndex.
// Populated for secondary vector columns in multi-column tables.
type ExtraVectorIndex struct {
	Column      string   `json:"column"`
	Dim         uint32   `json:"dim"`
	HnswOffset  uint64   `json:"hnsw_offset"`
	HnswLen     uint64   `json:"hnsw_len"`
	CentroidB64 *string  `json:"centroid_b64"`
	Radius      *float32 `json:"radius"`
}

// ailakeEntryExt mirrors the JSON structure stored in key_metadata.
type ailakeEntryExt struct {
	CentroidB64        *string            `json:"centroid_b64"`
	Radius             *float32           `json:"radius"`
	HnswOffset         *uint64            `json:"hnsw_offset"`
	HnswLen            *uint64            `json:"hnsw_len"`
	VectorCol          *string            `json:"vector_column"`
	VectorDim          *uint32            `json:"vector_dim"`
	IndexStatus        string             `json:"index_status"`
	IndexError         *string            `json:"index_error"`
	BatchID            *string            `json:"batch_id"`
	EmbeddingModel     *string            `json:"embedding_model"`
	ExtraVectorIndexes []ExtraVectorIndex `json:"extra_vector_indexes"`
	PartitionValue     *string            `json:"partition_value"`
}

// readManifestFile reads an Iceberg manifest file (Avro OCF) and returns DataFileEntry list.
func readManifestFile(path string) ([]DataFileEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ocf, err := goavro.NewOCFReader(f)
	if err != nil {
		return nil, fmt.Errorf("avro: %w", err)
	}

	var entries []DataFileEntry
	for ocf.Scan() {
		raw, err := ocf.Read()
		if err != nil {
			return nil, err
		}
		rec, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		df, ok := rec["data_file"]
		if !ok {
			continue
		}
		dfRec, ok := df.(map[string]any)
		if !ok {
			continue
		}

		filePath, _ := dfRec["file_path"].(string)
		recordCount := uint64(asInt64(dfRec["record_count"]))
		fileSize := uint64(asInt64(dfRec["file_size_in_bytes"]))

		// Recover AI-Lake extension from key_metadata (bytes JSON).
		// goavro v2 returns Avro union ["null","bytes"] as map[string]interface{}
		// with key "bytes", not as raw []byte.
		var ext ailakeEntryExt
		if km := dfRec["key_metadata"]; km != nil {
			var kmBytes []byte
			switch v := km.(type) {
			case []byte:
				kmBytes = v
			case map[string]interface{}:
				if b, ok := v["bytes"].([]byte); ok {
					kmBytes = b
				}
			}
			if len(kmBytes) > 0 {
				_ = json.Unmarshal(kmBytes, &ext)
			}
		}

		entry := DataFileEntry{
			Path:          filePath,
			RecordCount:   recordCount,
			FileSizeBytes: fileSize,
			IndexStatus:   ext.IndexStatus,
		}
		if ext.IndexError != nil {
			entry.IndexError = *ext.IndexError
		}
		if ext.CentroidB64 != nil {
			if centroid, err := decodeCentroid(*ext.CentroidB64); err == nil {
				entry.Centroid = centroid
			}
		}
		if ext.Radius != nil {
			entry.Radius = *ext.Radius
		}
		entry.HnswOffset = ext.HnswOffset
		entry.HnswLen = ext.HnswLen
		if ext.VectorCol != nil {
			entry.VectorColumn = *ext.VectorCol
		}
		if ext.VectorDim != nil {
			entry.VectorDim = *ext.VectorDim
		}
		if ext.BatchID != nil {
			entry.BatchID = *ext.BatchID
		}
		if ext.EmbeddingModel != nil {
			entry.EmbeddingModel = *ext.EmbeddingModel
		}
		entry.ExtraVectorIndexes = ext.ExtraVectorIndexes
		if ext.PartitionValue != nil {
			entry.PartitionValue = *ext.PartitionValue
		}
		entries = append(entries, entry)
	}
	return entries, ocf.Err()
}

// decodeCentroid decodes centroid_b64 from key_metadata.
// The Rust writer encodes only the vector floats (dim*4 bytes).
// Radius is a separate JSON field and is NOT included in these bytes.
func decodeCentroid(b64 string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 || len(raw)%4 != 0 {
		return nil, errors.New("centroid: unexpected length")
	}
	n := len(raw) / 4
	vec := make([]float32, n)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}

func asInt64(v any) int64 {
	switch t := v.(type) {
	case int64:
		return t
	case int32:
		return int64(t)
	case float64:
		return int64(t)
	case string:
		n, _ := strconv.ParseInt(t, 10, 64)
		return n
	}
	return 0
}
