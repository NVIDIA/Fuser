## Domains Overview: IterDomain, TensorDomain, TensorView

This article explains domain concepts and their roles in scheduling and indexing: root/logical, allocation, and loop domains, and how they relate across producers/consumers.

### Key definitions

- `IterDomain`: one axis’ semantic unit; carries `IterType` (Iteration/Broadcast/Reduction/etc.)
- `TensorDomain`: ordered set of `IterDomain` axes plus derived views (root/logical, allocation, loop)
- `TensorView`: symbolic tensor with a `TensorDomain` and transforms recorded

### The three domain views

- **Root/Logical**: problem-space axes as built by IR (pre-schedule). Often referred to as “root” or “logical.”
- **Allocation**: memory layout used for indexing/contiguity checks
- **Loop**: execution iteration structure (consumer-derived)

APIs: on a `TensorView` see `getLogicalDomain()`, `getMaybeAllocationDomain()`, `getLoopDomain()`.

### Who determines the loop domain?

- The consumer typically determines the loop domain for a producer’s values. Indexing and compute-at use consumer loop structure to align producers.

### Practical consequences

- Indivisible splits introduce “holes” in loop/alloc domains; we mask with predicates for correctness and tailor schedules for performance (see vectorization notes).
- Domain views are kept distinct to reason independently about iteration vs storage.

### Producer↔Consumer mapping

- `LogicalDomainMap` and replay helpers align axes across TVs so indexing and transforms stay consistent.

```970:1025:/opt/pytorch/nvfuser/csrc/transform_iter.cpp
BestEffortReplay::replayCasP(...)
BestEffortReplay::replayPasC(...)
// Align loop domains between producer and consumer based on logical mappings.
```

### References

- Q&A consolidation: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topics 2, 12–14)
- Iter transforms and replay: `../csrc/transform_iter.cpp`
- Indexing: `../csrc/index_compute.cpp`


