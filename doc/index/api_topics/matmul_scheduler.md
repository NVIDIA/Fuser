# MatmulScheduler

## Synopsis
Schedules matmul-like graphs using architecture-specific strategies (Ampere− / Hopper+), shared-memory staging, and MMA tiling.

## Sources
- Class: [`MatmulScheduler`](../../../csrc/scheduler/matmul.h#L26)
- Heuristics: `../../../csrc/scheduler/matmul_heuristic.h`
- Utilities: `../../../csrc/scheduler/mma_utils.h`, `../../../csrc/scheduler/matmul_utils.h`

## Overview
Detects matmul patterns, classifies operand/tile roles, sets up caches (`cacheBefore/After`), and applies MMA tiling with appropriate `LoadStoreOpType`/`CacheOp` for efficient operand movement. Architecture variants encapsulate differences in epilogue, staging, and vector sizes.

Interfaces:
- `canScheduleCompileTime/RunTime`
- `computeHeuristics(...)` → `MatmulParams`
- `schedule(fusion, params)`

## Related
- `LoadStoreOp`, `Swizzle2D`, persistent buffers, vectorization helpers
