# ReductionOp

## Synopsis
`ReductionOp` represents a reduction expression (e.g., sum, max) over one or more reduction axes.

## Source
- Class: [`ReductionOp`](../../../csrc/ir/internal_nodes.h#L955)

## Overview
A `ReductionOp` initializes an output with an `init` value and combines input values along reduction axes using a `BinaryOpType` (e.g., Add, Max). It supports grid/block reductions and scheduling options.

Key APIs:
- Accessors: `in()`, `out()`, `init()`
- Kind: `getReductionOpType()`
- Grid scheduling hint: `requestSerialGridReduction(bool)` / `serialGridReductionRequested()`

## Example
```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>

using namespace nvfuser;

int main() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* A = TensorViewBuilder().ndims(2).shape({64,32}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);

  // Sum over axis 1
auto* S = sum(A, {1});
  fusion.addOutput(S);

  auto* def = S->definition()->as<ReductionOp>();
  auto kind = def->getReductionOpType();         // BinaryOpType::Add
  (void)kind;
  return 0;
}
```

## Additional Guidance
- Reductions constrain computeAt: consumers of reduction outputs cannot be computed inside the open reduction loop.
- RFactor and multi-output grouped reductions exist; see `rFactor` and `GroupedReductionOp` for advanced transformations.
- Parallelization strategy (thread/block/grid) is driven by scheduling on `TensorDomain` and codegen passes.

## Where to Look in the Codebase
- Definition and helpers: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L955)
- User-facing helpers: `sum`, `max`, etc. in [`ops/arith.h`](../../../csrc/ops/arith.h)
- Grouped reductions: [`GroupedReductionOp`](../../../csrc/ir/internal_nodes.h#L1020)

## See Also
- `WelfordOp`: mean/variance reductions
- `ComputeAtLogicalDomainMap`: mapping constraints with reductions
