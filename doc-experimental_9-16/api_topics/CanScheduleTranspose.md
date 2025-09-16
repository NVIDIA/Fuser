# CanScheduleTranspose

Source: [CanScheduleTranspose](../../csrc/scheduler/compile_time_info.h#L195)

## Synopsis
- **Kind**: class (entry type definition for cached compile-time info)
- **File**: `csrc/scheduler/compile_time_info.h`
- **Role**: Helper wrapper class identifying the compile-time cache entry `CAN_SCHEDULE_TRANSPOSE`

## Purpose
- Encapsulates a boolean flag (`using DataType = bool`) stored in the compile-time [HeuristicDataCache](../../csrc/scheduler/compile_time_info.h#L249) under `CompileTimeEntryType::CAN_SCHEDULE_TRANSPOSE` indicating whether the transpose scheduler can handle the current Fusion.
- Used by scheduling heuristics and segmentation logic to quickly check/record the transpose scheduler viability without recomputation.
