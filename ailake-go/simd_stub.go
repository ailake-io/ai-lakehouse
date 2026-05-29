// SPDX-License-Identifier: MIT OR Apache-2.0
//go:build !amd64

package ailake

// cpuidLeaf7EBX returns 0 on non-amd64 — no AVX2/AVX-512.
func cpuidLeaf7EBX() uint32 { return 0 }
