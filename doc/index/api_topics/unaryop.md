# UnaryOp

## Synopsis
`UnaryOp` represents a one-input, one-output expression (e.g., cast, negation, unary math, certain reductions that are represented as unary forms) in the nvFuser IR.

## Source
- Class: [`UnaryOp`](../../../csrc/ir/internal_nodes.h#L385)

## Overview
A `UnaryOp` connects a single input `Val` to a single output `Val`. The operation kind is stored in a `UnaryOpType` attribute. Typical user-facing helpers in `ops/arith.h` create and wire these nodes.

Key APIs:
- Accessors: `in()`, `out()`
- Kind: `getUnaryOpType()`
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
  fusion.addInput(A);

  TensorView* B = neg(A);                 // UnaryOp: Neg
  TensorView* C = castOp(DataType::Double, B); // UnaryOp: Cast
  fusion.addOutput(C);

  auto* def = C->definition()->as<UnaryOp>();
  auto kind = def->getUnaryOpType();
  (void)kind;
  return 0;
}
```

## Where to Look in the Codebase
- Definition and accessors: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L379)
- Helper APIs for creation: [`ops/arith.h`](../../../csrc/ops/arith.h)

## See Also
- `BinaryOp`, `TernaryOp`: other expression arities
- `Expr`: base expression class (`expr.md`)
