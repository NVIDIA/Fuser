# Statement

## Synopsis
The root base class for all IR nodes in nvFuser. Both `Val` (values) and `Expr` (operations) derive from `Statement`.

## Source
- Class: [`Statement`](../../../csrc/ir/base_nodes.h#L96)

## Overview
`Statement` is the common base providing identity, ownership, and dynamic dispatch over the IR hierarchy. It allows traversing unknown concrete node types via a centralized dispatch mechanism (see `dispatch.h`/`dispatch.cpp`). Each `Statement` belongs to an IR container (`Fusion` or kernel IR) and supports cloning and string rendering.

Key features:
- Container ownership: `container()`, `fusion()`, `kernel()`
- Type queries: `isVal()`, `isExpr()`, `getValType()`, `getDataType()` (Vals only)
- Conversions: `asVal()`, `asExpr()`
- Dispatch: `Statement::dispatch`, `constDispatch`, `mutatorDispatch` (runtime downcasting)
- Identity: `name()`, `setName(...)`, `sameType`, `sameAs`, ordering via `lessThan`
- Diagnostics: `toString`, `toInlineString`
- Cloning: `clone(IrCloner*)`

## Role in the IR
All IR nodes are `Statement`s. Concrete types layer on top:
- `Val` (values; tensors, scalars, domains): see `val.md`
- `Expr` (computations; unary/binary/ternary ops, reductions, etc.): see `expr.md`

This base enables generic algorithms (visitors/mutators/printers/lowering) to operate without knowing concrete types at compile time.

## Minimal Example
Below, `toString` renders different `Statement` subtypes, while dynamic dispatch handles actual concrete cases.

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>

using namespace nvfuser;

int main() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* A = TensorViewBuilder().ndims(1).shape({16}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);
  TensorView* B = add(A, A);    // creates a BinaryOp Expr
  fusion.addOutput(B);

  Statement* sA = A;            // A is a Val -> Statement
  Statement* sDefB = B->definition(); // BinaryOp -> Expr -> Statement

  // Runtime identification
  bool aIsVal = sA->isVal();
  bool defIsExpr = sDefB->isExpr();

  // Printing
  std::cout << sA->toString() << "\n";
  std::cout << sDefB->toString() << "\n";

  (void)aIsVal; (void)defIsExpr;
  return 0;
}
```

## Where to Look in the Codebase
- Base definition and comments: [`base_nodes.h`](../../../csrc/ir/base_nodes.h#L87)
- Dynamic dispatch implementations: `csrc/dispatch.h`, `csrc/dispatch.cpp`
- IR container APIs: `csrc/fusion.h`

## See Also
- `Val`: `val.md`
- `Expr`: `expr.md`
