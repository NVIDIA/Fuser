# LdMatrix/StMatrix Inner-Loop Parallelization

## Key headers
- Scheduler utils (fragments/tiles): [`csrc/scheduler/mma_utils.h`](../csrc/scheduler/mma_utils.h), [`csrc/scheduler/utils.h`](../csrc/scheduler/utils.h)

## What they are
- Warp-level shared↔register matrix transfer instructions (feed/consume MMA)
- Not the MMA op itself, but tightly coupled to tensor core usage

Reference:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 7)

---

## Common parallelization patterns
- `ParallelType::TIDx` on the inner tile axis
  - Launch multiple `ldmatrix.x4` / `stmatrix.x4` across warp lanes
  - Effect: lanes cooperatively move sub-tiles in parallel
- `ParallelType::Vectorize` on the innermost element axis
  - Matches hardware fragment width; removes a serial per-element loop

---

## Rationale
- Map warp lanes efficiently to sub-tiles to exploit instruction-level parallelism
- Keep memory accesses coalesced/contiguous as required by the fragment layout
- Combine with tile-contiguity and whole-tile allocation in shared memory

---

## Tips
- Validate fragment shapes and swizzle against the kernel’s tile config
- Use IR dumps and kernel code prints to confirm expected parallelization
- Keep vectorization widths aligned with fragment transfers to avoid scalar tails

See also:
- `./tma-consumer-schedule-constraints.md`
- `./vectorization-alignment-and-tails.md`
