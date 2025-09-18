# NoOpScheduler

## Synopsis
Pass-through scheduler that applies no scheduling and forwards the fusion to codegen (handles degenerate cases).

## Source
- Class: [`NoOpScheduler`](../../../csrc/scheduler/no_op.h#L26)

## Overview
Used when all tensors are trivial (e.g., size-1/size-0) or no scheduling is beneficial. Still participates in the registry and selection pipeline.

Interfaces:
- `canScheduleCompileTime/RunTime`
- `computeHeuristics(...)`
- `schedule(...)`
