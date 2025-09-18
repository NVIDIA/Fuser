# ResizeScheduler

## Synopsis
Schedules graphs dominated by `Resize` (halo/padding) operations to maintain coalesced access and vectorization when expansions occur.

## Sources
- Class: [`ResizeScheduler`](../../../csrc/scheduler/resize.h#L19)
- Heuristics: `../../../csrc/scheduler/resize_heuristic.h`

## Overview
Accounts for left/right expansions when planning tiling and vectorization, ensures halo regions are handled efficiently, and leverages compile-time `ResizeVectorizationFactors` to choose legal vector sizes.

Interfaces:
- `canScheduleCompileTime(fusion)` / `canScheduleRunTime(...)`
- `computeHeuristics(...)` → `ResizeParams`
- `schedule(fusion, params)`

## Related
- `Resize` IR op; `scheduler/tools/resize_utils.*`
