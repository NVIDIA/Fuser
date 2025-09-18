# Persistent Kernel Schedulers

## Synopsis
Schedulers for cases requiring persistent buffers/register blocking across inner/outer dimensions.

## Sources
- Inner: [`InnerPersistentKernelScheduler`](../../../csrc/scheduler/normalization_inner.h#L28)
- Outer: [`OuterPersistentKernelScheduler`](../../../csrc/scheduler/normalization_outer.h#L28)
- InnerOuter: [`InnerOuterPersistentKernelScheduler`](../../../csrc/scheduler/normalization_inner_outer.h#L27)

## Overview
Persistent schedulers target normalization-like graphs where reuse and occupancy are balanced via buffers that persist across iterations. Variants differ by where persistence is applied (inner, outer, or both), and expose heuristic params to manage block sizes, smem usage, and vectorization.

Interfaces (each):
- `canScheduleCompileTime/RunTime`
- `computeHeuristics(...)`
- `schedule(fusion, params)`

## Related
- `scheduler/normalization_utils.*`
- Reduction heuristics; persistent buffer info entries
