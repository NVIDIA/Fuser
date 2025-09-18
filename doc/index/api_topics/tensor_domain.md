# TensorDomain

## Synopsis
The multi-dimensional domain descriptor for tensors in nvFuser. `TensorDomain` is composed of `IterDomain` axes and tracks root/logical/allocation/loop domains across scheduling transforms.

## Source
- Class: [`TensorDomain`](../../../csrc/ir/internal_base_nodes.h#L411)

## Overview
`TensorDomain` describes how a tensor is iterated, stored, and looped over during scheduling and codegen. It holds multiple domain views:
- Root domain: input logical axes as originally defined
- Logical domain: current logical axes after view-like transforms
- Allocation domain: memory-layout order (outer→inner)
- Loop domain: final loop order after scheduling (split/merge/reorder)
- Optional alternate/initial loop domains for special cases

It also tracks contiguity (per logical dim), broadcast/reduction presence, and provides utilities to traverse domain history and statements.

Key APIs:
- Domain accessors: `root()`, `logical()`, `allocation()`, `loop()`, `maybeRoot()`, `maybeAllocation()`, `initialLoop()`, `alternateLoop()`
- Queries: `hasReduction()`, `hasBroadcast()`, `hasAllocation()`, `hasRoot()`, `hasVectorize()`, `hasSymbolicAxis()`
- Axis ops (in-place on this domain): `broadcast`, `split`, `merge`, `reorder`, `swizzle`, `resize`, `flatten`, `view`
- Global utilities: `noReductions`, `noBroadcasts`, `orderedAs`, `getContiguityFilledWith`

## Relationship to IterDomain
Each axis is an `IterDomain` (1D annotated iterable). See `iterdomain.md` for axis-level properties (extent, parallel type, iter type, padding, etc.).

## Example: Basic Scheduling Flow
```cpp
// Given a 2D TV with root/logical [I0, I1]
TensorDomain* td = tv->domain();
// Split axis 1 into outer/inner by factor 4
// (loop domain gains two axes; allocation/logical may diverge as scheduling proceeds)
td->split(1, IrBuilder::create<Val>(4, DataType::Index), /*inner_split=*/true);
// Reorder and mark broadcast as needed
td->reorder({{0,1}, {1,0}});
```

## Additional Guidance
- Contiguity vector semantics: non-broadcast dims must be true/false; broadcast dims are `nullopt`. See comments around `getContiguityFilledWith` and `contiguity()`.
- Root vs logical: consumers’ root often matches producers’ logical (ignoring reductions), enabling alignment across views.
- Allocation vs loop: allocation controls memory layout; loop controls generated loop nests and parallelization.

## Where to Look in the Codebase
- Definition and rich comments: [`internal_base_nodes.h`](../../../csrc/ir/internal_base_nodes.h#L397)
- Axis-level ops and types: `IterDomain` (same header)
- Scheduling ops also appear as `Expr`s in [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h) (`Split`, `Merge`, `Swizzle`, `Resize`)

## See Also
- `IterDomain`: axis descriptor (`iterdomain.md`)
- `TensorView`: tensor value (`tensorview.md`)
