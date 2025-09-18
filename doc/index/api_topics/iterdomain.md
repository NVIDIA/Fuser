# IterDomain

## Synopsis
Axis-level domain descriptor in nvFuser. `IterDomain` represents a 1D iterable with annotations (extent/start/stop, iter type, parallel type, padding), used to build `TensorDomain`.

## Source
- Class: [`IterDomain`](../../../csrc/ir/internal_base_nodes.h#L83)

## Overview
An `IterDomain` captures a single axis’ iteration semantics:
- Range: `start()`, `extent()`, optional `expandedExtent()`, `stop()`, `stopOffset()`
- IterType: `Iteration`, `Reduction`, `Broadcast`, `Symbolic`, `GatherScatter`, `Stride`, `VectorComponent`
- ParallelType: `Serial`, or mapped to CUDA dimensions (`TIDx/y/z`, `BIDx/y/z`), and special tags (`Mma`, `Bulk`)
- RFactor and padding flags for scheduling and warp-aligned transforms

It exposes static helpers for axis transforms (merge/split/resize/swizzle) and queries (e.g., `isReduction`, `isBroadcast`, `isParallelized`).

## Key APIs
- Construction via `IterDomainBuilder` (start/extent/etc.)
- Static transforms: `merge`, `split`, `resize`, `swizzle`
- Parallelization: `parallelize(ParallelType)`, plus queries `isThreadDim()`, `isBlockDim()`
- Utilities: `stridedSplit`, `isImplicitBroadcast`, `maybePartial`

## Example
```cpp
// Build an axis with static extent 128
Val* start = IrBuilder::create<Val>(0, DataType::Index);
Val* extent = IrBuilder::create<Val>(128, DataType::Index);
IterDomain* id = IterDomainBuilder(start, extent).iter_type(IterType::Iteration).build();

// Split into outer/inner by factor 32 (inner is fast-changing)
auto [outer, inner] = IterDomain::split(id, IrBuilder::create<Val>(32, DataType::Index), /*inner_split=*/true);

// Mark inner as vector component or parallel thread-dim
inner->parallelize(ParallelType::TIDx);
```

## Additional Guidance
- Symbolic vs concrete: iter types are resolved as scheduling and concretization proceed; broadcast size-1 logical dims are implicit broadcasts.
- Padding: `padToMultipleOfWarp()` can mark padded warp-aligned dimensions (restricted to `TIDx`).
- Instruction loops: special tags (`Mma`, `Bulk`) indicate instruction-implemented loops that should not be realized as software loops.

## Where to Look in the Codebase
- Definition and detailed comments: [`internal_base_nodes.h`](../../../csrc/ir/internal_base_nodes.h#L79)
- Scheduling ops involving axes: `Split`, `Merge`, `Swizzle`, `Resize` in [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h)

## See Also
- `TensorDomain` (collection of `IterDomain`s): `tensor_domain.md`
- `TensorView` (tensor `Val`): `tensorview.md`
