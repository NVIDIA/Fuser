# PointWiseScheduler

## Synopsis
Schedules pointwise/broadcast graphs, often using a 2D tiling across two logical axes to maximize broadcast reuse.

## Sources
- Class: [`PointWiseScheduler`](../../../csrc/scheduler/pointwise.h#L161)
- Heuristics: `../../../csrc/scheduler/pointwise_heuristic.h`
- Utils: `../../../csrc/scheduler/pointwise_utils.h`, `../../../csrc/scheduler/vectorize_helper.h`

## Overview
The pointwise scheduler chooses a breakpoint to form a 2D tile `[BIDy,TIDy | BIDx,TIDx]`, promoting reuse along both axes. It accounts for broadcast multiples and view-induced coherence, and relies on helper utilities to determine vectorization opportunities and inner-dim groupings.

Interfaces:
- `canScheduleCompileTime(fusion)` — structural checks
- `canScheduleRunTime(fusion, runtime_info, cache)` — runtime checks (simplified)
- `computeHeuristics(...)` → `PointwiseParams`
- `schedule(fusion, params)` — applies splits, reorders, bindings, caching

## Example (conceptual)
```cpp
// Given reference tvR with domains [i0, i1, i2]
// choose break at 1: [i0 | i1, i2] -> map to [BIDy,TIDy | BIDx,TIDx]
```

## Related
- `BroadcastMultiples` entry in compile-time info
- `computeAt`, `LogicalDomainMap`, `view` handling
