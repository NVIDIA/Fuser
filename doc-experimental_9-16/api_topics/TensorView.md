# TensorView

Source: [TensorView](../../csrc/ir/interface_nodes.h#L383)

## Synopsis
- **Kind**: class (inherits from `Val`)
- **File**: `csrc/ir/interface_nodes.h`
- **What it represents**: The user-facing tensor node in nvFuser IR. Wraps and exposes a `TensorDomain` to define and schedule tensor axes.
- **Core state/relationships**:
  - Owns a `TensorDomain` (`tv->domain()`), which holds `IterDomain*` axes
  - Connects to producing/consuming `Expr` nodes that define how values are computed

## Purpose
- Entry point for tensor definition and scheduling APIs. Most axis-level transforms are performed through `TensorView` convenience methods that update the underlying `TensorDomain` and related IR.
- Used throughout scheduling heuristics and codegen to reason about layout, parallelization, and indexing.

## How to use it
```cpp
// Assume FusionGuard active and builder utilities available
TensorView* tv0 = makeConcreteTensor({N, M});
TensorView* tv1 = unaryOp(UnaryOpType::Neg, tv0);

// Schedule: split/reorder/merge via TensorView APIs
tv1->split(0, 32);
std::unordered_map<int64_t,int64_t> old2new{{0,1},{1,0}};
tv1->reorder(old2new);

// Parallelize inner axis
tv1->axis(-1)->parallelize(ParallelType::TIDx);
```

## See also
- [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
- [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)
- [ParallelType](../../csrc/type.h#L671), [IterType](../../csrc/type.h#L723)
