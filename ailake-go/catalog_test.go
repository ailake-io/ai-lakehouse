// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package ailake

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// ── asInt64 ───────────────────────────────────────────────────────────────────

func TestAsInt64(t *testing.T) {
	cases := []struct {
		input any
		want  int64
	}{
		{int64(42), 42},
		{int32(7), 7},
		{float64(3.9), 3},   // truncates
		{"99", 99},
		{nil, 0},
		{"bad", 0},
	}
	for _, c := range cases {
		if got := asInt64(c.input); got != c.want {
			t.Errorf("asInt64(%v %T): got %d, want %d", c.input, c.input, got, c.want)
		}
	}
}

// ── decodeCentroid ────────────────────────────────────────────────────────────

func TestDecodeCentroid_Valid(t *testing.T) {
	// Rust encodes only the vector floats (dim*4 bytes). Radius is a separate JSON field.
	dim := 2
	buf := make([]byte, dim*4)
	vecs := []float32{1.0, -0.5}
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	b64 := base64.StdEncoding.EncodeToString(buf)

	vec, err := decodeCentroid(b64)
	if err != nil {
		t.Fatalf("decodeCentroid: %v", err)
	}
	if len(vec) != dim {
		t.Fatalf("vec len: got %d, want %d", len(vec), dim)
	}
	if math.Abs(float64(vec[0]-1.0)) > 1e-6 {
		t.Errorf("vec[0]: got %v, want 1.0", vec[0])
	}
	if math.Abs(float64(vec[1]+0.5)) > 1e-6 {
		t.Errorf("vec[1]: got %v, want -0.5", vec[1])
	}
}

func TestDecodeCentroid_BadBase64(t *testing.T) {
	if _, err := decodeCentroid("!!!not-base64!!!"); err == nil {
		t.Error("decodeCentroid bad base64: expected error, got nil")
	}
}

func TestDecodeCentroid_TooShort(t *testing.T) {
	// 3 bytes — not a multiple of 4.
	b64 := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
	if _, err := decodeCentroid(b64); err == nil {
		t.Error("decodeCentroid not multiple of 4: expected error, got nil")
	}
}

// ── HadoopCatalog.tableDir ────────────────────────────────────────────────────

func TestTableDir(t *testing.T) {
	c := &HadoopCatalog{Warehouse: "/data/warehouse"}
	got := c.tableDir("default", "docs")
	want := "/data/warehouse/default/docs"
	if got != want {
		t.Errorf("tableDir: got %q, want %q", got, want)
	}
}

// ── ExtraVectorIndex — ailakeEntryExt JSON unmarshal ─────────────────────────

func unmarshalExt(t *testing.T, jsonStr string) ailakeEntryExt {
	t.Helper()
	var ext ailakeEntryExt
	if err := json.Unmarshal([]byte(jsonStr), &ext); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	return ext
}

func TestExtraVectorIndexes_Valid(t *testing.T) {
	dim := 2
	buf := make([]byte, dim*4)
	vecs := []float32{0.1, -0.9}
	for i, v := range vecs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	centroidB64 := base64.StdEncoding.EncodeToString(buf)

	ext := unmarshalExt(t, `{
		"extra_vector_indexes": [
			{
				"column": "context_embedding",
				"dim": 2,
				"hnsw_offset": 131072,
				"hnsw_len": 65536,
				"centroid_b64": "`+centroidB64+`",
				"radius": 0.42
			}
		]
	}`)

	if len(ext.ExtraVectorIndexes) != 1 {
		t.Fatalf("len: got %d, want 1", len(ext.ExtraVectorIndexes))
	}
	xi := ext.ExtraVectorIndexes[0]
	if xi.Column != "context_embedding" {
		t.Errorf("Column: got %q", xi.Column)
	}
	if xi.Dim != 2 {
		t.Errorf("Dim: got %d, want 2", xi.Dim)
	}
	if xi.HnswOffset != 131072 {
		t.Errorf("HnswOffset: got %d, want 131072", xi.HnswOffset)
	}
	if xi.HnswLen != 65536 {
		t.Errorf("HnswLen: got %d, want 65536", xi.HnswLen)
	}
	if xi.Radius == nil || math.Abs(float64(*xi.Radius-0.42)) > 1e-5 {
		t.Errorf("Radius: got %v, want ~0.42", xi.Radius)
	}
	if xi.CentroidB64 == nil || *xi.CentroidB64 != centroidB64 {
		t.Error("CentroidB64 mismatch")
	}
}

func TestExtraVectorIndexes_Empty(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096}`)
	if len(ext.ExtraVectorIndexes) != 0 {
		t.Errorf("got %d entries, want 0", len(ext.ExtraVectorIndexes))
	}
}

func TestExtraVectorIndexes_Multiple(t *testing.T) {
	ext := unmarshalExt(t, `{
		"extra_vector_indexes": [
			{"column": "col_a", "dim": 4, "hnsw_offset": 1000, "hnsw_len": 500, "centroid_b64": "", "radius": 0.1},
			{"column": "col_b", "dim": 8, "hnsw_offset": 2000, "hnsw_len": 1000, "centroid_b64": "", "radius": 0.2}
		]
	}`)
	if len(ext.ExtraVectorIndexes) != 2 {
		t.Fatalf("len: got %d, want 2", len(ext.ExtraVectorIndexes))
	}
	if ext.ExtraVectorIndexes[0].Column != "col_a" {
		t.Errorf("entry[0].Column: got %q", ext.ExtraVectorIndexes[0].Column)
	}
	if ext.ExtraVectorIndexes[1].HnswOffset != 2000 {
		t.Errorf("entry[1].HnswOffset: got %d, want 2000", ext.ExtraVectorIndexes[1].HnswOffset)
	}
}

// ── PartitionValue in ailakeEntryExt (Phase 9) ────────────────────────────────

func TestAilakeEntryExt_PartitionValue_Parsed(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096, "partition_value": "agent-A"}`)
	if ext.PartitionValue == nil {
		t.Fatal("PartitionValue: expected non-nil, got nil")
	}
	if *ext.PartitionValue != "agent-A" {
		t.Errorf("PartitionValue: got %q, want %q", *ext.PartitionValue, "agent-A")
	}
}

func TestAilakeEntryExt_PartitionValue_Missing_IsNil(t *testing.T) {
	ext := unmarshalExt(t, `{"hnsw_offset": 4096}`)
	if ext.PartitionValue != nil {
		t.Errorf("PartitionValue: expected nil when absent, got %q", *ext.PartitionValue)
	}
}

func TestAilakeEntryExt_PartitionValue_EmptyString_IsNonNil(t *testing.T) {
	ext := unmarshalExt(t, `{"partition_value": ""}`)
	if ext.PartitionValue == nil {
		t.Fatal("PartitionValue: expected non-nil pointer for empty string field")
	}
	if *ext.PartitionValue != "" {
		t.Errorf("PartitionValue: got %q, want empty string", *ext.PartitionValue)
	}
}

// ── str / boolVal helpers ─────────────────────────────────────────────────────

func TestStr_String(t *testing.T) {
	if got := str("hello"); got != "hello" {
		t.Errorf("str: got %q", got)
	}
}

func TestStr_Nil(t *testing.T) {
	if got := str(nil); got != "" {
		t.Errorf("str(nil): got %q, want empty", got)
	}
}

func TestStr_WrongType(t *testing.T) {
	if got := str(42); got != "" {
		t.Errorf("str(int): got %q, want empty", got)
	}
}

func TestBoolVal_True(t *testing.T) {
	if !boolVal(true) {
		t.Error("boolVal(true): got false")
	}
}

func TestBoolVal_FalseNil(t *testing.T) {
	if boolVal(nil) {
		t.Error("boolVal(nil): got true")
	}
}

// ── PartitionDef / SchemaField structs ────────────────────────────────────────

func TestPartitionDef_Fields(t *testing.T) {
	pd := PartitionDef{Column: "agent_id", Transform: "identity", ColumnType: "string"}
	if pd.Column != "agent_id" || pd.Transform != "identity" || pd.ColumnType != "string" {
		t.Errorf("unexpected: %+v", pd)
	}
}

func TestSchemaField_Fields(t *testing.T) {
	sf := SchemaField{ID: 1, Name: "agent_id", Type: "string", Required: false}
	if sf.ID != 1 || sf.Name != "agent_id" || sf.Type != "string" || sf.Required {
		t.Errorf("unexpected: %+v", sf)
	}
}

// ── LoadTable metadata parsing (unit, no FS) — via readMetadata mock ─────────

// buildMetaJSON constructs a minimal metadata.json map for testing.
func buildTestMetaMap(fv int, schemaID int, specs bool) map[string]any {
	meta := map[string]any{
		"format-version":    float64(fv),
		"current-schema-id": float64(schemaID),
		"default-spec-id":   float64(0),
		"properties": map[string]any{
			"ailake.vector-column": "embedding",
			"ailake.vector-dim":    "128",
			"ailake.vector-metric": "cosine",
		},
		"schemas": []any{
			map[string]any{
				"schema-id": float64(schemaID),
				"type":      "struct",
				"fields": []any{
					map[string]any{"id": float64(1), "name": "agent_id", "type": "string", "required": false},
					map[string]any{"id": float64(2), "name": "ts",       "type": "long",   "required": false},
				},
			},
		},
	}
	if specs {
		meta["partition-specs"] = []any{
			map[string]any{
				"spec-id": float64(0),
				"fields": []any{
					map[string]any{
						"field-id":  float64(1000),
						"source-id": float64(1), // → agent_id: string
						"name":      "agent_id",
						"transform": "identity",
					},
					map[string]any{
						"field-id":  float64(1001),
						"source-id": float64(2), // → ts: long
						"name":      "ts_trunc",
						"transform": "truncate[4]",
					},
				},
			},
		}
	}
	return meta
}

func parseTableInfoFromMeta(t *testing.T, meta map[string]any) *TableInfo {
	t.Helper()
	// Replicate the LoadTable logic on a pre-built map without FS access.
	info := &TableInfo{
		Table:         "default.table",
		Location:      "/tmp/test",
		FormatVersion: 2,
	}
	if props, ok := meta["properties"].(map[string]any); ok {
		info.VectorColumn, _ = props["ailake.vector-column"].(string)
		info.VectorDim, _ = props["ailake.vector-dim"].(string)
		info.VectorMetric, _ = props["ailake.vector-metric"].(string)
	}
	if fv, ok := meta["format-version"].(float64); ok {
		info.FormatVersion = int(fv)
	}
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
					sf := SchemaField{Name: str(fm["name"]), Required: boolVal(fm["required"])}
					if id, ok := fm["id"].(float64); ok {
						sf.ID = int(id)
					}
					if tp, ok := fm["type"].(string); ok {
						sf.Type = tp
					}
					info.SchemaFields = append(info.SchemaFields, sf)
				}
			}
			break
		}
	}
	fieldTypeByID := make(map[int]string)
	for _, sf := range info.SchemaFields {
		fieldTypeByID[sf.ID] = sf.Type
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
					pd := PartitionDef{Column: str(fm["name"]), Transform: str(fm["transform"])}
					if srcID, ok := fm["source-id"].(float64); ok {
						pd.ColumnType = fieldTypeByID[int(srcID)]
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
	return info
}

func TestTableInfo_FormatVersion3(t *testing.T) {
	info := parseTableInfoFromMeta(t, buildTestMetaMap(3, 0, false))
	if info.FormatVersion != 3 {
		t.Errorf("FormatVersion: got %d, want 3", info.FormatVersion)
	}
}

func TestTableInfo_FormatVersion2_Default(t *testing.T) {
	meta := buildTestMetaMap(2, 0, false)
	delete(meta, "format-version")
	info := parseTableInfoFromMeta(t, meta)
	if info.FormatVersion != 2 {
		t.Errorf("FormatVersion: got %d, want 2 (default)", info.FormatVersion)
	}
}

func TestTableInfo_SchemaFields_Parsed(t *testing.T) {
	info := parseTableInfoFromMeta(t, buildTestMetaMap(2, 0, false))
	if len(info.SchemaFields) != 2 {
		t.Fatalf("SchemaFields len: got %d, want 2", len(info.SchemaFields))
	}
	if info.SchemaFields[0].Name != "agent_id" || info.SchemaFields[0].Type != "string" {
		t.Errorf("SchemaFields[0]: %+v", info.SchemaFields[0])
	}
	if info.SchemaFields[1].Name != "ts" || info.SchemaFields[1].Type != "long" {
		t.Errorf("SchemaFields[1]: %+v", info.SchemaFields[1])
	}
}

func TestTableInfo_PartitionFields_MultiColumn(t *testing.T) {
	info := parseTableInfoFromMeta(t, buildTestMetaMap(2, 0, true))
	if len(info.PartitionFields) != 2 {
		t.Fatalf("PartitionFields len: got %d, want 2", len(info.PartitionFields))
	}
	p0 := info.PartitionFields[0]
	if p0.Column != "agent_id" || p0.Transform != "identity" || p0.ColumnType != "string" {
		t.Errorf("PartitionFields[0]: %+v", p0)
	}
	p1 := info.PartitionFields[1]
	if p1.Column != "ts_trunc" || p1.Transform != "truncate[4]" || p1.ColumnType != "long" {
		t.Errorf("PartitionFields[1]: %+v", p1)
	}
}

func TestTableInfo_NoPartitionSpec_EmptyFields(t *testing.T) {
	info := parseTableInfoFromMeta(t, buildTestMetaMap(2, 0, false))
	if len(info.PartitionFields) != 0 {
		t.Errorf("PartitionFields: expected empty, got %d entries", len(info.PartitionFields))
	}
}
