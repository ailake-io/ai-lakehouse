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
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	if !filepath.IsAbs(warehouse) && !strings.Contains(warehouse, "://") {
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
// Returns the new schema_id on success.
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
	if !filepath.IsAbs(warehouse) && !strings.Contains(warehouse, "://") {
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

// resolveBin returns the path to the `ailake` CLI binary.
// Checks AILAKE_BIN env first, then PATH.
func resolveBin() (string, error) {
	if bin := os.Getenv("AILAKE_BIN"); bin != "" {
		if _, err := os.Stat(bin); err == nil {
			return bin, nil
		}
		return "", fmt.Errorf("ailake: AILAKE_BIN=%q not found: %w", bin, ErrNoBinary)
	}
	bin, err := exec.LookPath("ailake")
	if err != nil {
		return "", ErrNoBinary
	}
	return bin, nil
}
