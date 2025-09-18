# TransposeScheduler

## Synopsis
Schedules transpose-like graphs, aligning inner-most dims and choosing tile shapes suitable for memory coalescing and vectorization.

## Sources
- Class: [`TransposeScheduler`](../../../csrc/scheduler/transpose.h#L93)
- Heuristics: `../../../csrc/scheduler/transpose_heuristic.h`

## Overview
Analyzes logical/physical reorders, decides whether to use the transpose scheduler, and applies scheduling that favors coalesced access on both input and output sides when feasible. Uses compile-time `InnerMostDimInfo` and domain maps.

Interfaces:
- `canScheduleCompileTime(fusion)` / `canScheduleRunTime(...)`
- `computeHeuristics(...)` → `TransposeParams`
- `schedule(fusion, params)`

## Related
- `LogicalDomainMap`, `LogicalReorderMap` entries
- Vectorization and alignment in `vectorize_helper.*`
