# Scheduler Types

## Synopsis
Enumerates supported scheduler kinds and documents the expected interfaces and priority order used for heuristic selection.

## Source
- Enum: [`SchedulerType`](../../../csrc/scheduler/scheduler_types.h#L49)
- Priority table: [`all_heuristics_in_priority_order`](../../../csrc/scheduler/scheduler_types.h#L64)

## Overview
Each `SchedulerType` corresponds to a concrete scheduler class implementing:
1) `canScheduleCompileTime(Fusion*)`
2) `canScheduleRunTime(Fusion*, SchedulerRuntimeInfo&, HeuristicDataCache*)`
3) `schedule(Fusion*, const HeuristicParams*)`

Types include:
- None, NoOp, PointWise, Matmul, Reduction, InnerPersistent, InnerOuterPersistent, OuterPersistent, Transpose, ExprEval, Resize, Communication

Priority order guides selection during segmentation and compilation.

## Related
- Registry: `../../../csrc/scheduler/registry.h`
- Per-scheduler headers in `../../../csrc/scheduler/`
