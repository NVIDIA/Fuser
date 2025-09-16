# IterDomain

Source: [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)

## Synopsis
- **Kind**: class (inherits from [Val](../../csrc/ir/base_nodes.h#L224))
- **File**: `csrc/ir/internal_base_nodes.h`
- **Approx. size**: ~320 lines for the class definition body
- **What it represents**: A single annotated 1D iteration domain (range) with metadata used by nvFuser’s IR and scheduler
- **Key state**:
  - `start`, `extent`, optional `expanded_extent`, `stop_offset`
  - `parallel_type` ([ParallelType](../../csrc/type.h#L671)): mapping to block/thread/grid, vectorization, MMA, etc.
  - `iter_type` ([IterType](../../csrc/type.h#L723)): Iteration, Reduction, Broadcast, Symbolic, Stride, VectorComponent, GatherScatter
  - `is_rfactor_domain`, padding flags (`is_padded_dimension`, `padded_to_size`)
- **Key operations**:
  - Static transforms: `merge`, `split`, `resize`, `swizzle`
  - Queries: `isReduction`, `isBroadcast`, `isIteration`, `isParallelized`, `isThreadDim`, `isBlockDim`, `isMma`, `isBulk`, etc.
  - Mutation of scheduling metadata: `parallelize(ParallelType)`
  - Construction helpers: [IterDomainBuilder](../../csrc/ir/internal_base_nodes.h#L36) (sets start, extent, iter/parallel kinds, rfactor, padding)

## Purpose
- **Role in the IR**: The fundamental 1D axis descriptor used to build higher-dimensional tensor iteration spaces. A [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415) is essentially a vector of `IterDomain*`, capturing root/logical/allocation/loop views of a tensor’s axes.
- **What it enables**:
  - Expressing transformations that shape loop structure: `split`/`merge`/`resize`/`swizzle`
  - Attaching execution mapping: [ParallelType](../../csrc/type.h#L671) (e.g., `BIDx`, `TIDx`, `Vectorize`, `Mma`, `Bulk`)
  - Describing semantic role of an axis: [IterType](../../csrc/type.h#L723) (e.g., `Reduction`, `Broadcast`, `Iteration`, `Symbolic`)
  - Marking rfactor axes used to restructure reductions (`is_rfactor_domain`)
  - Padding semantics for warp-aligned dimensions and compile-time sizing
- **Used by**:
  - [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415) (groups IDs and tracks relationships among root/logical/allocation/loop domains)
  - Scheduling and lowering passes (e.g., mapping to CUDA grid/block/thread loops, vectorization, MMA tiling)
  - Index/Predicate compute utilities and swizzle utilities that operate on axis pairs

## How to use it (illustrative)
The examples assume you’re inside a valid Fusion/IR context (guards elided) and using the nvFuser IR builder facilities.

```cpp
#include <ir/internal_base_nodes.h>
#include <ir/interface_nodes.h>
#include <fusion.h>

using namespace nvfuser;

void build_simple_axis_example(Fusion* fusion, Val* N /* extent */) {
  FusionGuard fg(fusion);

  // 1) Construct an IterDomain with start=0, extent=N via the builder
  IterDomainBuilder id_builder(/*start=*/fusion->oneVal(), /*extent=*/N);
  id_builder.iter_type(IterType::Iteration)
            .parallel_type(ParallelType::Serial);
  IterDomain* id = id_builder.build();

  // 2) Attach scheduling metadata (e.g., map to a thread dimension later)
  id->parallelize(ParallelType::Serial); // can be updated during scheduling

  // 3) Compose higher-dimensional domains in a TensorDomain
  std::vector<IterDomain*> logical = {id};
  TensorDomain* td = IrBuilder::create<TensorDomain>(logical);

  // 4) Transform the axis: split then map inner to threads
  Val* factor = IrBuilder::create<Int>(4);
  auto [outer, inner] = IterDomain::split(id, factor, /*inner_split=*/true);
  inner->parallelize(ParallelType::TIDx);

  // 5) Merge back if desired
  IterDomain* merged = IterDomain::merge(outer, inner);
  (void)merged; // continue building the schedule
}
```

Another common pattern is to operate through [TensorView](../../csrc/ir/interface_nodes.h#L383) helpers (e.g., `tv->split(...)`, `tv->merge(...)`), which internally manipulate the underlying [IterDomain](../../csrc/ir/internal_base_nodes.h#L83) objects associated with that tensor’s [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415).

## Practical notes
- `expanded_extent` is tracked separately for broadcasted dimensions that are later “materialized” (e.g., via expand) to a concrete physical size.
- `resize` can represent slicing/expansion; unless known at definition, the output [IterType](../../csrc/type.h#L723) is often `Symbolic` and resolved during concretization.
- `rfactor` domains allow reductions to be restructured for scheduling (e.g., split reduction across axes).
- Parallelization queries (`isThreadDim`, `isBlockDim`, `isDeviceDim`) reflect current mapping decisions and are central during lowering/codegen.

## See also
- [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415): groups [IterDomain](../../csrc/ir/internal_base_nodes.h#L83) instances into root/logical/allocation/loop domains and provides end-to-end axis transformation utilities
- [ParallelType](../../csrc/type.h#L671), [IterType](../../csrc/type.h#L723), [SwizzleType](../../csrc/type.h#L832), [Swizzle2DType](../../csrc/type.h#L833), [SwizzleMode](../../csrc/type.h#L836) (in `csrc/type.h`) for the enumerations referenced by [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)
