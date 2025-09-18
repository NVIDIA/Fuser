# SchedulerRuntimeInfo

## Synopsis
Runtime metadata for schedulers: input pointers, sizes/strides, alignment, and expression evaluation context.

## Source
- Class: [`SchedulerRuntimeInfo`](../../../csrc/scheduler/runtime_info.h#L33)

## Overview
Constructed per Fusion and inputs, this class provides schedulers with:
- Input tensor allocation sizes and strides (elements) in allocation domain order
- Input pointers and alignment in bits
- An `ExpressionEvaluator` bound to fusion inputs for symbolic evaluation
- Optional forced index type control

Key APIs:
- `getInputAllocationSizes(TensorView*)`
- `getInputAllocationStrides(TensorView*)`
- `ptrOf(TensorView*)`, `getAlignmentSizeBit(TensorView*)`, `computeAlignmentSizeBit(...)`
- `expressionEvaluator()`

## Notes
- Only valid for complete Fusion inputs whose allocation domain is a permutation of root domain.
- Used by segmentation and kernel cache.

## Related
- `CompileTimeInfo` and `HeuristicDataCache`
