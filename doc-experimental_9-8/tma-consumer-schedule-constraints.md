# TMA Consumer Schedule Constraints

## Key headers
- Scheduler registry/types: [`csrc/scheduler/registry.h`](../csrc/scheduler/registry.h), [`csrc/scheduler/scheduler_types.h`](../csrc/scheduler/scheduler_types.h)
- Transpose and normalization helpers: [`csrc/scheduler/transpose.h`](../csrc/scheduler/transpose.h), [`csrc/scheduler/normalization_utils.h`](../csrc/scheduler/normalization_utils.h)

## Core constraints (from Q&A)
- Separate branches: transform tile IterDomains and non‑tile IterDomains separately; do not merge/swizzle a tile axis with a non‑tile axis
- Contiguity and whole‑tile allocation: shared‑memory tile must be contiguous in the allocation domain, and allocation size must be an integer multiple of whole tiles
- Swizzle constraints: inner extent must match swizzle size (e.g., 32/64/128B); shared‑memory view must match the prescribed shape
- Indexing replay limitation: producer and consumer must be replayable for indexing; some valid schedules can be unindexable otherwise

Reference:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 6)

---

## Why these matter
- Mixing tile/non‑tile axes breaks the assumptions TMA uses to compute box and tile indexing
- Non‑contiguous or partial‑tile allocations violate whole‑tile transfer requirements
- Swizzle requires exact layout contracts; mismatches lead to invalid shared‑memory views

---

## Practical guidance
- Keep tile axes grouped and isolate their transforms
- Verify allocation domain contiguity for tiles; size to multiples of full tiles
- Choose swizzle shapes that match inner extents; validate via dumps

See also:
- `./tma-fttc-and-weak-correctness.md`
