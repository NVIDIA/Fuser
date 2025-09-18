# ReductionScheduler

## Synopsis
Schedules reductions with heuristics for persistent buffers, warp reductions, and grid/block strategies.

## Sources
- Class: [`ReductionScheduler`](../../../csrc/scheduler/reduction.h#L22)
- Heuristics: `../../../csrc/scheduler/reduction_heuristic.h`
- Utils: `../../../csrc/scheduler/reduction_utils.h`

## Overview
Determines reduction strategy (e.g., persistent kernels, multi-stage reductions), chooses axes mappings, and applies tiling, rFactor, and vectorization when profitable. Uses runtime info for sizes/strides and compile-time info for TV sets and persistence.

Interfaces:
- `canScheduleCompileTime(fusion)`
- `canScheduleRunTime(fusion, runtime_info, cache)`
- `computeHeuristics(...)` → `ReductionParams`
- `schedule(fusion, params)`

## Related
- `WelfordOp`, `ReductionOp`
- Persistent buffer info, vectorization breakpoints
