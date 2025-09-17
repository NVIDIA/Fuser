# TensorDomain

Source: [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)

## Synopsis
- **Kind**: class (inherits from [Val](../../csrc/ir/base_nodes.h#L224))
- **File**: `csrc/ir/internal_base_nodes.h`
- **What it represents**: An ordered set of `IterDomain*` axes describing a tensor’s iteration spaces at different stages (root, logical, allocation, loop)
- **Key domains tracked**:
  - `root_domain`: initial logical axes at definition (often equal to `logical_domain` for producers)
  - `logical_domain`: semantic axes after definition ops (e.g., broadcast/expand)
  - `allocation_domain`: how data is laid out in memory (outer-to-inner order)
  - `loop_domain`: scheduled axes used to build loop nests
  - `alternate_loop_domain`, `initial_loop_domain`, and `additional_ids` for advanced scheduling
- **Key queries/ops**:
  - Comparisons: `sameAs`, `operator==`
  - Introspection: `nDims`, `hasReduction`, `hasBroadcast`, `hasVectorize`, `hasSymbolicAxis`
  - Axis access: `axis(i)`, `posOf(id)`, `rootPosOf(id)`
  - Domain transforms: `split`, `merge`, `reorder`, `broadcast`, `resize`, `view`, `flatten`, `swizzle`
  - Domain subsets: `noReductions`, `noBroadcasts`, `noDevices`
  - rFactor: `rFactor(axes)` to restructure reductions
  - Contiguity modeling: `contiguity()`, `setAllocationDomain(..., contiguity)`

## Purpose
- **Role**: Binds together multiple [IterDomain](../../csrc/ir/internal_base_nodes.h#L83) views of a tensor, capturing how axes evolve from definition to scheduled loop structure and how they map to memory layout.
- **Scheduling**: Central orchestration of axis transformations for code generation—split/merge/reorder/swizzle/resize drive the loop nest and parallelization mappings.
- **Interoperability**: Provides utilities to query and align domains between producers and consumers during scheduling and lowering.

## Practical usage
Typically manipulated via [TensorView](../../csrc/ir/interface_nodes.h#L383) high-level APIs, which internally update the underlying [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415):

```cpp
// tv: TensorView*
// Split axis 0 by factor 32; map inner to threads later
tv->split(0, 32);
// Reorder for desired memory/loop order
std::unordered_map<int64_t,int64_t> old2new{{0,1},{1,0}};
tv->reorder(old2new);
// Merge axes
tv->merge(0, 1);
```

## Notes
- `contiguity` is an `optional<bool>` per logical axis; broadcast axes carry `nullopt`, others indicate density with the next non-broadcasting axis.
- `maybeRoot`, `maybeAllocation` provide convenience to operate when specific domains are absent.
- `allIDs`/`allExprs`/`allStatements` traverse along defining/using Exprs, useful for analysis.

## See also
- [IterDomain](../../csrc/ir/internal_base_nodes.h#L83): 1D axis descriptor used to build the domains above
- [TensorView](../../csrc/ir/interface_nodes.h#L383): user-facing tensor IR node that owns a [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
