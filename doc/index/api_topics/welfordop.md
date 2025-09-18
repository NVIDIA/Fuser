# WelfordOp

## Synopsis
`WelfordOp` represents numerically-stable mean/variance/count (N) reductions (Welford’s algorithm) in nvFuser. It produces a triplet of outputs.

## Source
- Class: [`WelfordOp`](../../../csrc/ir/internal_nodes.h#L1202)

## Overview
Unlike simple reductions that output a single tensor, Welford maintains a triplet `(avg, var, N)` and updates them in a stable manner across elements and tiles. The IR provides structured access to each component and utility APIs to query initialization and fused behavior.

Key APIs:
- Outputs: `outAvg()`, `outVar()`, `outN()`
- Inputs: `inAvg()`, `inVar()`, `inN()`; initial values: `initAvg()`, `initVar()`, `initN()`
- Helpers: `outputTriplet()`, `inputTriplet()`, `initTriplet()`, `getInitVals()`, `getInitValOfOutput(...)`
- Properties: `singleValue()`, `hasInit()`, `isAllreduce()`

## Example
```cpp
// Example sketch (actual helper may differ by frontend convenience):
TensorView* X = ...; // [N, D]
// Compute mean/var along D
auto w = welford(X, /*axes=*/{1}); // returns triplet-like outputs
TensorView* mean = w.avg;
TensorView* var  = w.var;
TensorView* count= w.N;
```

## Additional Guidance
- Scheduling: Welford reductions typically require careful parallelization and potential rFactor to distribute work across threads/blocks.
- Fused behavior: `isAllreduce()` indicates fused multi-stage reductions (API evolves with backend support).
- Grouped variants (`GroupedWelfordOp`) exist for multi-output/grouped use-cases.

## Where to Look in the Codebase
- Definition and triplet utilities: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L1081) (WelfordTriplet), [`WelfordOp`](../../../csrc/ir/internal_nodes.h#L1202)
- Example scheduling patterns: tests under `tests/cpp` with softmax/layernorm examples using Welford

## See Also
- `ReductionOp`: single-output reductions
- `ComputeAtLogicalDomainMap`: mapping constraints for reductions and consumers
