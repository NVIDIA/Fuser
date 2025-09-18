# Val

## Synopsis
The base IR node for values in nvFuser. `Val` represents data (tensors, scalars, domains, etc.) that are inputs and outputs of `Expr` computations in the fusion graph.

## Source
- Class: [`Val`](../../../csrc/ir/base_nodes.h#L224)

## Overview
`Val` is the foundational value node in the nvFuser IR. Every non-input `Val` is defined exactly once by the output of some `Expr` (SSA discipline). `Val` carries a `ValType`, a `DataType`, and, optionally, a concrete `PolymorphicValue` for constants. It supports introspection of its defining expression and its uses.

What a `Val` can represent:
- Tensor-like values (e.g., `TensorView`)
- Scalars (constant or symbolic)
- Iteration and domain constructs (e.g., `IterDomain`, handled in other headers)
- Kernel-time named values (e.g., `NamedScalar` for `threadIdx.x`, `blockDim.y`)

Key responsibilities:
- Type info: `vtype()`, `dtype()`; constant storage via `value()` when applicable
- Graph wiring: `definition()` returns the producing `Expr`; `uses()` lists consumer `Expr`s
- Constant/utility checks: `isConst()`, `isConstScalar()`, `isConstInt()`, `isIntegralScalar()`, `isFloatingPointScalar()`, `isZero/One/True/False`

## Common Derived/Related Types
- `TensorView` (tensor values and domain/view transforms): [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h)
- `NamedScalar` (parallel dims/indices like `threadIdx.x`): [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L1983)
- Domain/value constructs (see `ir/internal_base_nodes.h` and `ir/internal_nodes.h`) used in scheduling

## Working with Val in Practice
- Creating inputs: `fusion.addInput(tv_or_val)` marks `Val` as fusion input (`isFusionInput() == true`)
- Outputs: `fusion.addOutput(val)` marks fusion outputs (`isFusionOutput() == true`)
- Definition and uses:
  - `val->definition()` returns the single `Expr*` that defines it (or `nullptr` for inputs)
  - `val->uses()` lists all expressions that consume it
- Data type and value:
  - `dtype()` returns `DataType` (throws if absent via `getDataType()`)
  - `value()` provides constant storage when `Val` is a compile-time constant

## Heavily Commented Usage Example
Creating a scalar constant and a tensor, inspecting definition and uses:

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>

using namespace nvfuser;

int main() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic tensor input
  TensorView* A = TensorViewBuilder().ndims(2).shape({64, 64}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);

  // Compile-time scalar constant (Val) via IrBuilder utilities under arith ops
  Val* three = IrBuilder::create<Val>(3.0, DataType::Double); // constant scalar Val

  // B = A + 3
  TensorView* B = add(A, three);  // B is a Val (TensorView) defined by a BinaryOp Expr
  fusion.addOutput(B);

  // Introspection
  Expr* defB = B->definition();         // BinaryOp
  auto consumersOfA = A->uses();        // includes BinaryOp for B
  bool aIsInput = A->isFusionInput();   // true
  bool bIsOutput = B->isFusionOutput(); // true after addOutput

  (void)defB; (void)consumersOfA; (void)aIsInput; (void)bIsOutput;
  return 0;
}
```

## Additional Guidance and Gotchas
- SSA values: a `Val` is defined once; altered computations yield new `Val`s
- DataType constraints: packed types must be unpacked at definition time (see checks in constructor)
- Constants vs symbolic: `isConst()` is true when `value()` is present and there is no `definition()`; otherwise the `Val` is symbolic until defined by an `Expr`
- Evaluation: `evaluate()` can produce a `PolymorphicValue` when all dependencies are constants

## Where to Look in the Codebase
- Base definitions and comments: [`base_nodes.h`](../../../csrc/ir/base_nodes.h#L193)
- Tensor values and transforms: [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h)
- Named scalars, domain nodes, and other IR constructs: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h)
- IR architecture graphic: [`csrc/docs/images/ir_architecture.png`](../../../csrc/docs/images/ir_architecture.png)

## See Also
- `Expr` (computations): [`base_nodes.h`](../../../csrc/ir/base_nodes.h#L466) and API topic `expr.md`
- `TensorView` (tensor `Val`): API topic `tensorview.md`
