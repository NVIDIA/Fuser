# Building IR with FusionGuard and Free Functions

## Key types
- [`Fusion`](../csrc/fusion.h)
- [`FusionGuard`](../csrc/fusion_guard.h)
- [`TensorView`](../csrc/ir/interface_nodes.h)

## Key pattern
- Create a `Fusion` and scope it with `FusionGuard`
- Use free functions (`add`, `transpose`, `reshape`, `set`, etc.) to emit IR nodes into the active Fusion via `IrBuilder`
- Register inputs/outputs; inspect transforms and IR prints

References:
- Notes: `../doc-bot/experimenting/9-3-2025/key_insights.md`
- Sample: `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp`
- Ops implementation hints: `../csrc/ops/alias.h` (free ops create IR nodes in the current Fusion)

---

## Heavily commented template

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ir/iostream.h>

using namespace nvfuser;

int main(){
  // 1) Create a Fusion (graph container) and make it current with FusionGuard.
  Fusion fusion;
  FusionGuard guard(&fusion); // sets thread-local current fusion

  // 2) Free ops below add IR nodes to the current fusion via IrBuilder.
  //    No data moves at IR-building time; this is purely symbolic.

  // Define an input TV (symbolic)
  TensorView* A = TensorViewBuilder().ndims(2).shape({16, 16})
                                     .contiguity({true,true})
                                     .dtype(DataType::Float).build();
  fusion.addInput(A);

  // Permute A (no copy needed). For 2D, transpose(A) swaps dims 0 and 1.
  TensorView* B = transpose(A);

  // Compute C = A + transpose(A)
  TensorView* C = add(A, B);
  fusion.addOutput(C);

  // 3) Inspect per-TV transforms and full IR
  A->printTransforms();
  B->printTransforms();
  C->printTransforms();

  std::cout << "\n=== Fusion IR (with transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/true);
  std::cout << "\n=== Fusion IR (without transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/false);
}
```

---

## Why RAII + free ops?
- Concise DSL for IR building (no need to thread `Fusion*` through every call)
- Guard defines the active container; builders verify container consistency
- Prevents cross-fusion mistakes by checking `container()` on IR nodes

Notes and tips:
- `transpose(x, dim0, dim1)` also supports explicit dims (negative dims allowed and wrapped). Using `transpose(x)` for 2D is shorthand for `transpose(x, 0, 1)`.
- `set(tv)` inserts an explicit copy in the IR. It's not required before `transpose` and is usually unnecessary.
- `TensorViewBuilder().shape({-1, 32})` can be used for symbolic extents (use -1) alongside concrete ones.
- To print only math ops without tensor transforms, you can also call `fusion.printMath(/*from_outputs_only=*/true);`

Common pitfalls:
- Forgetting to scope with `FusionGuard` results in IR nodes not being attached to any active `Fusion`.
- Mixing IR from different `Fusion` instances is rejected; free ops use `FusionGuard::getCurFusion()` and will error on container mismatch.
- Not registering inputs/outputs with `fusion.addInput`/`fusion.addOutput` will lead to empty or incomplete IR when printing or lowering.

See also:
- `./instantiate-bind-execute.md`
- `./how-to-read-fusion-ir-dumps.md`

---

## Needs improvement
- Add a short subsection contrasting alias/view ops (`set`, `permute`, `reshape`) with materializing ops, and when each appears in lowered code.
- Provide an example with symbolic shapes (using `-1` in `TensorViewBuilder::shape`) and show how they print in IR.
- Show a variant that uses tensor factory ops (e.g., `iota`, `full`, `zeros`) to construct inputs without `TensorViewBuilder`.
- Briefly mention negative-dimension handling in `transpose(x, -2, -1)` and dim-wrapping rules.
- Link to headers for additional ops (`ops/indexing.h`, `ops/composite.h`) and a note on include paths when building outside the repo (see `build-and-run-standalone-samples.md`).
