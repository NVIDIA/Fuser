# TernaryOp

## Synopsis
`TernaryOp` represents a three-input, one-output expression (e.g., select/where, clamp-like patterns) in the nvFuser IR.

## Source
- Class: [`TernaryOp`](../../../csrc/ir/internal_nodes.h#L468)

## Overview
A `TernaryOp` connects three input `Val`s to a single output `Val`. The operation kind is stored as a `TernaryOpType` attribute. User-facing helpers in `ops/arith.h` (and other op headers) create these nodes.

Key APIs:
- Accessors: `in1()`, `in2()`, `in3()`, `out()`
- Kind: `getTernaryOpType()`
- Rendering/eval: `toString`, `toInlineString`, `evaluate(...)`

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

  TensorView* A = TensorViewBuilder().ndims(1).shape({64}).dtype(DataType::Float).contiguity(true).build();
  TensorView* B = TensorViewBuilder().ndims(1).shape({64}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);
  fusion.addInput(B);

  // Example: where(A > 0, A, B)
  TensorView* mask = gt(A, IrBuilder::create<Val>(0.0, DataType::Float));
  TensorView* C = where(mask, A, B);  // Typically represented as a TernaryOp
  fusion.addOutput(C);

  auto* def = C->definition()->as< TernaryOp >();
  auto kind = def->getTernaryOpType();
  (void)kind;
  return 0;
}
```

## Where to Look in the Codebase
- Definition and accessors: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L468)
- Helper APIs for creation: [`ops/arith.h`](../../../csrc/ops/arith.h)

## See Also
- `UnaryOp`, `BinaryOp`: other expression arities
- `Expr`: base expression class (`expr.md`)
