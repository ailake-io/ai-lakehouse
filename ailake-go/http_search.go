// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// searchViaHTTP delegates vector search to a running `ailake serve` instance.
//
// Used when GPU is detected but Go cannot call CUDA kernels without cgo.
// Set AILAKE_SERVER_URL=http://localhost:7700 to enable GPU delegation.
//
// The `ailake serve` process runs the Rust IVF-PQ search with CUDA/ROCm
// acceleration. Go reads the JSON results and returns FileSearchResult.
package ailake

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 30 * time.Second}

type httpSearchRequest struct {
	Query  []float32 `json:"query"`
	TopK   int       `json:"top_k"`
}

type httpSearchResponse struct {
	Results []struct {
		Rank     int     `json:"rank"`
		RowID    uint64  `json:"row_id"`
		Distance float32 `json:"distance"`
		FilePath string  `json:"file_path"`
	} `json:"results"`
}

// searchViaHTTP POSTs to serverURL/search and returns results.
// serverURL is the base URL of a running `ailake serve` (e.g. http://localhost:7700).
func searchViaHTTP(serverURL, _ string, query []float32, topK int) ([]FileSearchResult, error) {
	url := strings.TrimRight(serverURL, "/") + "/search"

	body, err := json.Marshal(httpSearchRequest{Query: query, TopK: topK})
	if err != nil {
		return nil, err
	}

	resp, err := httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ailake HTTP search: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ailake HTTP search: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ailake HTTP search: status %d: %s", resp.StatusCode, raw)
	}

	var parsed httpSearchResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("ailake HTTP search: parse response: %w", err)
	}

	out := make([]FileSearchResult, len(parsed.Results))
	for i, r := range parsed.Results {
		out[i] = FileSearchResult{RowID: r.RowID, Distance: r.Distance, FilePath: r.FilePath}
	}
	return out, nil
}
