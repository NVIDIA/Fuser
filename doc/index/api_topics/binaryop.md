# BinaryOp

## Synopsis
`BinaryOp` represents a two-input, one-output expression (e.g., add/mul/sub/div/compare) in the nvFuser IR.

## Source
- Class: [`BinaryOp`](../../../csrc/ir/internal_nodes.h#L425)

## Overview
A `BinaryOp` connects two input `Val`s to a single output `Val`. Common uses include arithmetic and comparison on tensors/scalars. The operation kind is stored as a `BinaryOpType` attribute.

Key APIs:
- Accessors: `lhs()`, `rhs()`, `out()`
- Kind: `getBinaryOpType()` returns the `BinaryOpType`
- Rendering/eval: `toString`, `toInlineString`, `evaluate(...)`

Typical creation is via user-facing helper functions in `ops/arith.h` (e.g., `add`, `mul`, `sub`, `div`, logical ops), which build the appropriate `BinaryOp` and wire inputs/outputs.

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

  TensorView* A = TensorViewBuilder().ndims(2).shape({64,64}).dtype(DataType::Float).contiguity(true).build();
  TensorView* B = TensorViewBuilder().ndims(2).shape({64,64}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);
  fusion.addInput(B);

  TensorView* C = add(A, B);           // BinaryOp under the hood
  fusion.addOutput(C);

  auto* def = C->definition()->as<BinaryOp>();
  auto kind = def->getBinaryOpType();  // e.g., BinaryOpType::Add
  (void)kind;
  return 0;
}
```

## Additional Guidance
- Broadcasting: standard tensor broadcasting rules are modeled using `BroadcastOp` and domain alignment; `BinaryOp` relies on compatible shapes/domains.
- DTypes: `dtype` of the output follows type promotion rules enforced elsewhere in the IR builder/ops layer.
- Scheduling: `BinaryOp` does not directly encode scheduling; loop/order is determined by `TensorDomain` transforms on its `TensorView` inputs/outputs.

## Where to Look in the Codebase
- Definition and accessors: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L421)
- Helper APIs for creation: [`ops/arith.h`](../../../csrc/ops/arith.h)

## See Also
- `UnaryOp`, `TernaryOp`: other expression arity classes
- `Expr`: base expression class (`expr.md`)
