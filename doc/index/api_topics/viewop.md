# ViewOp

## Synopsis
`ViewOp` represents view-like shape re-interpretation between two `TensorView`s, capturing shape changes that preserve data without materializing copies.

## Source
- Class: [`ViewOp`](../../../csrc/ir/internal_nodes.h#L1640)

## Overview
A view transforms one `TensorView` into another with different logical domains while preserving data order. It is used by helper APIs like `view(TensorView*, ...)` and analyzed by scheduling utilities to align domains for fusion and vectorization.

Key APIs:
- Accessors: `in()`, `out()` (both `TensorView*`)
- Rendering/eval: `toString`, `toInlineString`, `evaluate(...)`

Notes:
- Works closely with `TensorDomain::view(...)` analysis and the scheduler’s view utilities.
- Commonly appears in pointwise pipelines where rank/shape adjustments are needed.

## Example
```cpp
using namespace nvfuser;
Fusion fusion;
FusionGuard fg(&fusion);

TensorView* tv0 = makeConcreteTensor({2, 3, 5});
fusion.addInput(tv0);

// Collapse [3,5] into a single dim -> [2, 15]
TensorView* tv1 = view(tv0, {2, 15});

// Use tv1 in further ops
TensorView* tv2 = neg(tv1);
fusion.addOutput(tv2);
```

## Related
- API helper: `../../../csrc/ops/alias.h` (`view`)
- Schedulers: `../../../csrc/scheduler/pointwise.h`, `.../vectorize_helper.h`
- Domain: `TensorDomain`, `IterDomain`

## References
- Impl/decl: `../../../csrc/ir/internal_nodes.h#L1640`
- Scheduler mentions: `scheduler/pointwise.h`, `scheduler/vectorize_helper.h`, `scheduler/utils.h`
