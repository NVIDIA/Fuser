# TMA FTTC: When Strong Correctness Is Impossible (and What to Do)

## Key headers
- Scheduler registry/types: [`csrc/scheduler/registry.h`](../csrc/scheduler/registry.h), [`csrc/scheduler/scheduler_types.h`](../csrc/scheduler/scheduler_types.h)
- Transpose/normalization (TMA-related paths): [`csrc/scheduler/transpose.h`](../csrc/scheduler/transpose.h), [`csrc/scheduler/normalization_inner_outer_tma_ws.h`](../csrc/scheduler/normalization_inner_outer_tma_ws.h)

## FTTC condition (impossibility of strong correctness)
Strong correctness is impossible iff BOTH:
1) The element stride does not divide the box size and is smaller than the box size (`e < B` and `e ∤ B`)
2) The box size is smaller than the tensor size on that dimension (`B < S`)

Intuition:
- A strided tile partially overlaps valid and invalid regions within a single tile; hardware zero-fill can’t apply to only part of a tile.

Reference:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topics 4–6)

---

## Practical responses
- Accept weak correctness (unpredicated TMA), and avoid feeding reductions that require strong correctness
- Predicate only non‑TMA‑protected IterDomains to limit traffic but keep overall behavior weak-correct
- Keep tile vs non-tile transforms separate in the consumer (see constraints article)

---

## Notes on swizzle/contiguity
- Swizzle sizes (e.g., 32/64/128B) require matching inner extents and specific shared-memory views
- Shared-memory tile must be contiguous in allocation and sized as an integer multiple of whole tiles

See also:
- `./tma-consumer-schedule-constraints.md`
- `./weak-vs-strong-correctness.md`
