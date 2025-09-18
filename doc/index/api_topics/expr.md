# Expr

## Synopsis
The IR node for computations in nvFuser. `Expr` connects input `Val` nodes to output `Val` nodes, representing an operation in the fusion graph.

## Source
- Class: [`Expr`](../../../csrc/ir/base_nodes.h#L505)

## Overview
`Expr` is the base class of all IR operations. Each `Expr` has zero or more inputs and produces one or more outputs, all of which are `Val` instances. The IR is SSA: a `Val` is defined exactly once by the output of an `Expr` (unless it is a fusion input). `Expr` is immutable; new expressions are created, not modified in-place. Subclasses capture concrete operations and their attributes.

Derived operation families include (selection):
- Tensor/view/dataflow ops: [`LoadStoreOp`](../../../csrc/ir/internal_nodes.h#L1665), [`ViewOp`](../../../csrc/ir/internal_nodes.h#L1631), [`BroadcastOp`](../../../csrc/ir/internal_nodes.h#L848), [`SqueezeOp`](../../../csrc/ir/internal_nodes.h#L899), [`ExpandOp`](../../../csrc/ir/internal_nodes.h#L1526), [`RepeatOp`](../../../csrc/ir/internal_nodes.h#L1567)
- Arithmetic ops: [`UnaryOp`](../../../csrc/ir/internal_nodes.h#L385), [`BinaryOp`](../../../csrc/ir/internal_nodes.h#L425), [`TernaryOp`](../../../csrc/ir/internal_nodes.h#L468)
- Reductions and statistics: [`ReductionOp`](../../../csrc/ir/internal_nodes.h#L955), [`GroupedReductionOp`](../../../csrc/ir/internal_nodes.h#L1020), [`WelfordOp`](../../../csrc/ir/internal_nodes.h#L1201), [`GroupedWelfordOp`](../../../csrc/ir/internal_nodes.h#L1314)
- Indexing/shape ops: [`SelectOp`](../../../csrc/ir/internal_nodes.h#L64), [`IndexSelectOp`](../../../csrc/ir/internal_nodes.h#L97), [`GatherOp`](../../../csrc/ir/internal_nodes.h#L188), [`SliceOp`](../../../csrc/ir/internal_nodes.h#L2135), [`CatOp`](../../../csrc/ir/internal_nodes.h#L2185), [`ArgsortOp`](../../../csrc/ir/internal_nodes.h#L2844), [`TopKOp`](../../../csrc/ir/internal_nodes.h#L3283)
- Matmul and fused math: [`MmaOp`](../../../csrc/ir/internal_nodes.h#L1436), [`GroupedMmaOp`](../../../csrc/ir/internal_nodes.h#L2966), [`ScaledMmaOp`](../../../csrc/ir/internal_nodes.h#L3153), [`MatmulOp`](../../../csrc/ir/internal_nodes.h#L2234), [`LinearOp`](../../../csrc/ir/internal_nodes.h#L2266)
- Domain/schedule ops: [`Split`](../../../csrc/ir/internal_nodes.h#L1720), [`Merge`](../../../csrc/ir/internal_nodes.h#L1764), [`Swizzle`](../../../csrc/ir/internal_nodes.h#L1794), [`Swizzle2D`](../../../csrc/ir/internal_nodes.h#L1841), [`Resize`](../../../csrc/ir/internal_nodes.h#L1936)
- RNG and factories: [`RNGOp`](../../../csrc/ir/internal_nodes.h#L747), [`IotaOp`](../../../csrc/ir/internal_nodes.h#L303), [`EyeOp`](../../../csrc/ir/internal_nodes.h#L356), [`FullOp`](../../../csrc/ir/internal_nodes.h#L41)
- Struct/array utilities: [`ArrayConstruct`](../../../csrc/ir/internal_nodes.h#L523), [`ReverseArray`](../../../csrc/ir/internal_nodes.h#L547), [`GetItem`](../../../csrc/ir/internal_nodes.h#L576), [`StructConstruct`](../../../csrc/ir/internal_nodes.h#L609), [`GetAttr`](../../../csrc/ir/internal_nodes.h#L641), [`TensorConstruct`](../../../csrc/ir/internal_nodes.h#L716), [`GetMetaData`](../../../csrc/ir/internal_nodes.h#L674)

Key `Expr` capabilities:
- Inputs/outputs/attributes accessors: `inputs()`, `outputs()`, `attributes()`, `getOpString()` in [`base_nodes.h`](../../../csrc/ir/base_nodes.h)
- Evaluation hooks used by host-side evaluators: `evaluate(...)`
- Predication (kernel IR): `predicate()`, `writePredicate()`, `withPredicate(...)`

## How `Expr` Relates to `Val` and the IR
- `Val` represents values (tensors, scalars, domains). Every non-input `Val` is defined by exactly one `Expr` (`Val::definition()`).
- `Expr` edges connect `Val` producers to consumers. `Val::uses()` returns all consumer `Expr`s.
- `TensorView` is a `Val` subtype; many ops produce/consume `TensorView*`.

## Common Patterns and Subclasses
- Pointwise math: `UnaryOp`, `BinaryOp`, `TernaryOp`
- Reductions: `ReductionOp`, `WelfordOp` families
- Shape/view: `BroadcastOp`, `SqueezeOp`, `ExpandOp`, `ViewOp`, `RepeatOp`
- Scheduling/domain: `Split`, `Merge`, `Swizzle`, `Resize` (operate on `IterDomain`/`TensorDomain`)
- Data movement: `LoadStoreOp` (explicit state-space moves, e.g., Set, CpAsync)
- Advanced fused ops: `MmaOp`, `SdpaFwdOp`/`SdpaBwdOp`, `GroupedMmaOp`, `LinearOp`

## Heavily Commented Usage Example
Minimal sketch showing how `Expr` instances appear when constructing a fusion. The user-level API calls (e.g., `add`, `transpose`) create specific `Expr` subclasses under the hood and wire inputs/outputs.

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>
#include <ops/alias.h>

using namespace nvfuser;

int main() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Build two TVs and a simple expr graph: C = A + transpose(A)
  TensorView* A = TensorViewBuilder().ndims(2).shape({64, 64}).dtype(DataType::Float).contiguity(true).build();
  fusion.addInput(A);
  TensorView* B = transpose(set(A));             // creates Set (LoadStoreOp) and a transpose expr
  TensorView* C = add(A, B);                     // creates a BinaryOp expr under the hood
  fusion.addOutput(C);

  // Introspect via Val API
  Expr* defC = C->definition();                  // BinaryOp
  auto in0 = defC->input(0);                     // A
  auto in1 = defC->input(1);                     // B
  const char* op = defC->getOpString();          // "BinaryOp"

  return 0;
}
```

## Additional Guidance and Gotchas
- SSA discipline: a `Val` is defined once; creating alternate computations yields new `Val`s and `Expr`s.
- Attributes: many ops store parameters as `attributes()` rather than additional `inputs()` (see accessors in each subclass).
- Kernel predicates: in lowered (kernel) IR, `predicate` and `writePredicate` control conditional emission.
- Domain ops vs tensor ops: `Split/Merge/Swizzle/Resize` operate on `IterDomain`/`TensorDomain` (not on full tensors) and are part of scheduling.

## Where to Look in the Codebase
- Base definitions and APIs: [`base_nodes.h`](../../../csrc/ir/base_nodes.h#L466)
- Concrete ops and rich comments: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h)
- IR architecture image: [`csrc/docs/images/ir_architecture.png`](../../../csrc/docs/images/ir_architecture.png)
- Developer docs (helpful context):
  - `/doc/dev/visibility.md`, `/doc/dev/tmem.md`, `/doc/dev/tma.md`

## See Also
- `Val` and `TensorView`: [`base_nodes.h`](../../../csrc/ir/base_nodes.h), [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h)
- Scheduling utilities: compute-at and replay headers under `csrc/`
