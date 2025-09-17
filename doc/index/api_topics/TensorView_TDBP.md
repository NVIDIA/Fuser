# TensorView – Natural Language Draft (Stage 8)

## Overview
TensorView is the primary tensor node in nvFuser’s IR. It represents what is being computed, while its associated [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)/[IterDomain](../../csrc/ir/internal_base_nodes.h#L83) objects represent how it is computed through axis transformations and scheduling decisions. That separation enables flexible loop construction, parallelization, and memory layout without changing the math.

- Definition: [TensorView](../../csrc/ir/interface_nodes.h#L383)
- IR context: [Val](../../csrc/ir/base_nodes.h#L224), [Expr](../../csrc/ir/base_nodes.h#L505), [Fusion](../../csrc/fusion.h#L134)

## Axis Transformations
TensorView exposes transformation APIs that mutate its [TensorDomain](../../../csrc/ir/internal_base_nodes.h#L411), changing loop structure and iteration semantics:

- split / inner_split / outer_split: factor an axis into outer×inner pieces (interface_nodes.h ~ L585-L609)
- merge: combine two consecutive axes (L611-L617)
- flatten: fuse a range of axes `[from..to]` (L619-L626)
- reorder: permute axes by mapping old→new positions (L623-L633)
- resize: expand/shrink by left/right extents; often symbolic until concretized (L643-L651)
- swizzle: 2D/3D tile swizzles controlled by [SwizzleType](../../csrc/type.h#L832)/[Swizzle2DType](../../csrc/type.h#L833)/[SwizzleMode](../../csrc/type.h#L836) (L634-L641)
- broadcast: materialize a size-1 axis in the loop domain (L581-L584)

## Compute Placement
Placement determines how much of a producer’s loop nest is shared with a consumer:

- computeAt(consumer, position, mode): standard/best-effort/most-inlined search (L576-L579)
- inlineAt(pos): force inlining to a position, optionally best-effort (L776-L779)
- computeWith(pos) and resolveComputeWith(...): compute “with” first consumer in topological order (L781-L809)
- Queries: `getComputeAtPosition`, `getComputeWithPosition`, `getMaxComputePosition`, `getComputePosition(consumer)`

## Memory & Caching
Route data through different memory spaces or stage for reuse:

- cacheBefore / cacheAfter / cacheFork (L679-L708): interpose staged tensors around an op; select [LoadStoreOpType](../../../csrc/type.h#L772), [CacheOp](../../../csrc/type.h#L759), and subset of uses
- setMemoryType / getMemoryType (L709-L714): choose [MemoryType](../../csrc/type.h#L720)
- circularBuffer(stages, prefetch, type) (L715-L729): overlap production/consumption via multi-stage buffering
- promoteReuse (L847-L859): shared-memory reuse request with required synchronization

## Hardware-Specific Helpers
Use hardware-aware helpers when targeting tensor cores or TMA:

- applyMmaSwizzle(MmaOperand/MmaInputSmemSwizzle), applyMmaSwizzleForTMALoad, swizzleTMABox (L731-L750, L739-L745)

## Mapping & Replay Infrastructure
Keep producer/consumer consistent with mapping and replay:

- [ReplayTransformations](../../csrc/transform_iter.h#L49): re-apply target-domain transforms onto mapped IDs
- [BestEffortReplay](../../csrc/transform_iter.h#L318): discover best-effort mappings (with forwarding through broadcasts/resizes when enabled)
- [ComputeAtMap](../../csrc/compute_at_map.h#L182): equivalence classes and concrete IDs under modes (EXACT, AlmostExact, PERMISSIVE, PERMISSIVE_RESIZE, INNERMOST, LOOP)

## Safety Notes
`commitLeafToLogical()` (L838-L846) commits loop-domain changes into rFactor/logical domains, changing semantics; callers must ensure consumer consistency. Placement legality checks guard against data hazards and axis mismatches; LOOP-sets must not carry conflicting non-serial ParallelTypes.

## References
- TensorView: [interface_nodes.h#L383](../../csrc/ir/interface_nodes.h#L383)
- TensorDomain: [internal_base_nodes.h#L415](../../csrc/ir/internal_base_nodes.h#L415)
- IterDomain: [internal_base_nodes.h#L83](../../csrc/ir/internal_base_nodes.h#L83)
- Enums: [type.h](../../csrc/type.h#L671)
- Replay/BestEffort: [transform_iter.h#L49](../../csrc/transform_iter.h#L49), [transform_iter.h#L318](../../csrc/transform_iter.h#L318)
- ComputeAtMap: [compute_at_map.h#L182](../../csrc/compute_at_map.h#L182)
