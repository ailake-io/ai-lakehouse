// SPDX-License-Identifier: MIT OR Apache-2.0
// CPUID stub for amd64 — returns EBX from leaf 7, sub-leaf 0.
// Bit 5 = AVX2, bit 16 = AVX-512F.
#include "textflag.h"

TEXT ·cpuidLeaf7EBX(SB),NOSPLIT,$0-4
    MOVL $7, AX
    MOVL $0, CX
    CPUID
    MOVL BX, ret+0(FP)
    RET
