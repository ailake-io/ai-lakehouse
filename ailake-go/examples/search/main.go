// SPDX-License-Identifier: MIT OR Apache-2.0
// Example: vector search over a local AI-Lake table.
//
// Usage:
//   go run . -warehouse /data/warehouse -table default.docs -dim 4
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"

	ailake "github.com/ThiagoLange/iceberg-ai-deltalakehouse/ailake-go"
)

func main() {
	warehouse := flag.String("warehouse", ".", "Warehouse root path")
	tableFlag := flag.String("table", "default.table", "namespace.table")
	dim := flag.Int("dim", 64, "Query vector dimensionality")
	topK := flag.Int("top-k", 10, "Number of results")
	flag.Parse()

	parts := strings.SplitN(*tableFlag, ".", 2)
	if len(parts) != 2 {
		log.Fatalf("table must be namespace.name, got %q", *tableFlag)
	}
	namespace, name := parts[0], parts[1]

	// Generate a random query vector
	query := make([]float32, *dim)
	for i := range query {
		query[i] = rand.Float32()
	}

	catalog := &ailake.HadoopCatalog{Warehouse: *warehouse}

	// Print table info
	info, err := catalog.LoadTable(namespace, name)
	if err != nil {
		log.Fatalf("load table: %v", err)
	}
	infoJSON, _ := json.MarshalIndent(info, "", "  ")
	fmt.Printf("Table info:\n%s\n\n", infoJSON)

	// Run search
	results, err := ailake.Search(catalog, namespace, name, query, ailake.SearchOptions{
		TopK:             *topK,
		PruningThreshold: 0.8,
	})
	if err != nil {
		log.Fatalf("search: %v", err)
	}

	fmt.Printf("%-6s %-12s %s\n", "rank", "distance", "file_path")
	for i, r := range results {
		fmt.Printf("%-6d %-12.6f %s (row_id=%d)\n", i+1, r.Distance, r.FilePath, r.RowID)
	}
}
