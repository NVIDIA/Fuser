# computeAt (scheduling)

## Synopsis
`computeAt` schedules a producer `TensorView` to be computed at a loop position relative to a consumer, controlling fusion and inlining depth.

## Source
- API: [`TensorView::computeAt`](../../../csrc/ir/interface_nodes.h#L576)
- Modes: `ComputeAtMode::{Standard, BestEffort, MostInlined}` (in `interface_nodes.h`)
- Mapping: `ComputeAtLogicalDomainMap` (see `logical_domain_map.md`)

## Overview
`computeAt(producer, consumer, position)` requests the producer to be realized within the consumer’s loop nest up to `position` axes. This aligns loop domains to enable fusion, locality, and reduced memory traffic.

- `position = -1`: inline as deep as possible (often fully inlined)
- `position = 0`: no shared loops (compute producer outside)
- Positive positions share that many outer-most loops
- Modes:
  - Standard: error if requested position is illegal
  - BestEffort: lower the position until it becomes valid
  - MostInlined: find deepest valid position automatically

`computeWith` offers similar inlining but defers the concrete consumer until lowering.

## Example
```cpp
TensorView* A = ...; // input
TensorView* B = ...; // input
TensorView* C = add(A, B);
TensorView* D = sum(C, {1}); // reduction

// Inline C into D’s loop nest at the deepest valid position
auto* inlined = C->computeAt(D, -1, ComputeAtMode::MostInlined);
(void)inlined;
```

## Additional Guidance
- Validity depends on logical-domain mappings; use `ComputeAtLogicalDomainMap` to reason about mappable dims
- Reductions constrain mapping (consumers of reduction outputs cannot be computed inside the reduction loop)
- Combine with domain transforms (`split`, `merge`, `reorder`) and memory ops (`cacheBefore/After`) for performance

## Where to Look in the Codebase
- TV scheduling APIs: [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h)
- Tests demonstrating patterns: `tests/cpp/test_*` (many `computeAt` cases)
- Mapping internals: [`csrc/logical_domain_map.h`](../../../csrc/logical_domain_map.h)

## See Also
- `logical_domain_map.md` for mapping mechanics
- `split.md`, `merge.md` for axis transforms
- `expr.md`, `tensorview.md` for IR building blocks
