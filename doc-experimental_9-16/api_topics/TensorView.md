# TensorView

Source: [TensorView](../../csrc/ir/interface_nodes.h#L383)

## Synopsis
- **Kind**: class (inherits from `Val`)
- **File**: `csrc/ir/interface_nodes.h`
- **What it represents**: The primitive tensor node used in code generation.
  - Represents the computation view of a tensor, while its `TensorDomain` holds how it is iterated/scheduled
  - Dimensionality changes as transforms (split/merge/reorder/etc.) are applied; history drives codegen over physical memory

## Purpose
- Serves as the primary, user-facing handle for defining tensor computations and driving scheduling in nvFuser.
- Separates “what” is computed (ops on `TensorView`s) from “how” it is computed (`TensorDomain` axis transforms), enabling clear orchestration of loop nests, layout, and parallel mapping.

## Core relationships
- Owns a [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415) (`domain()`), which aggregates [IterDomain](../../csrc/ir/internal_base_nodes.h#L83) axes across root/logical/allocation/loop views
- Producing/consuming [Expr](../../csrc/ir/base_nodes.h#L505) connect `TensorView`s to form the IR graph inside a [Fusion](../../csrc/fusion.h#L134)

## Key capabilities (selected APIs)
- Domain accessors and properties:
  - `domain()`, `nDims()`, `getRootDomain()`, `getLogicalDomain()`, `getAllocationDomain()`, `getLoopDomain()`
  - Reductions/broadcasts: `hasReduction()`, `hasBlockReduction()`, `hasGridReduction()`, `hasBroadcast()`, `getReductionAxis()`
  - Contiguity modeling: `setContiguity(...)`, `getContiguity()`, `getMaybeAllocationDomain()`
- Axis transformations (update underlying `TensorDomain`):
  - `split`, `inner_split`, `outer_split`, `merge`, `flatten`, `reorder`
  - 2D/3D swizzles: `swizzle(...)` with [SwizzleType](../../csrc/type.h#L832)/[Swizzle2DType](../../csrc/type.h#L833)/[SwizzleMode](../../csrc/type.h#L836)
  - `resize(axis, left, right, iter_type)` for slicing/expansion semantics
  - Broadcast materialization: `broadcast(axis, extent)`
- Compute placement and inlining:
  - `computeAt(consumer, position, mode)`, `inlineAt(pos)`, `computeWith(pos)`, `resolveComputeWith(...)`, queries like `getComputeAtPosition()`, `getComputePosition(consumer)`
- rFactor and multi-output rFactor for reduction restructuring: `rFactor(axes)` / `rFactor(axes, tvs)`
- Caching/memory utilities:
  - `cacheBefore(...)`, `cacheAfter(...)`, `cacheFork()`, `getMemoryType()`, `setMemoryType(...)`
  - Circular buffering: `circularBuffer(stages, prefetch, type)`, `isCircularBuffered()`, `circularBufferOptions()`
- MMA/TMA-specific helpers:
  - `applyMmaSwizzle(...)`, `applyMmaSwizzleForTMALoad(...)`, `swizzleTMABox(...)`
- Lifecycle helpers:
  - `clearReductionIterDomains()`, `commitLeafToLogical()`, `promoteReuse(true)`

## How to use it (illustrative)
```cpp
// Assume FusionGuard active and builder utilities available
TensorView* tv0 = makeConcreteTensor({N, M});
TensorView* tv1 = unaryOp(UnaryOpType::Neg, tv0);

// Axis transformations
tv1->split(0, 32);
std::unordered_map<int64_t,int64_t> old2new{{0,1},{1,0}};
tv1->reorder(old2new);

// Map inner axis to threads (after scheduling decisions)
tv1->axis(-1)->parallelize(ParallelType::TIDx);

// Reduction restructuring (example)
// auto tv2 = sum(tv1, {1});
// auto tv_rf = tv2->rFactor({1});
```

## Notes and guidance
- Code comments emphasize the separation of concerns: `TensorView` captures the mathematical operation, while `TensorDomain` records the axis history for codegen.
- Compute placement APIs (`computeAt`, `computeWith`, `inlineAt`) control loop nest sharing and inlining; use `ComputeAtMode::BestEffort`/`MostInlined` for adaptive placement.
- Use caching helpers to route through shared/register memory and improve locality; circular buffering can reduce latency by staging data.
- MMA/TMA helpers align thread/tile swizzles and box scheduling with specialized hardware paths.

## See also
- [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
- [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)
- [ParallelType](../../csrc/type.h#L671), [IterType](../../csrc/type.h#L723), [SwizzleType](../../csrc/type.h#L832), [Swizzle2DType](../../csrc/type.h#L833), [SwizzleMode](../../csrc/type.h#L836)
