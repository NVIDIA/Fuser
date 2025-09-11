## LdMatrix/StMatrix Inner-Loop Parallelization (TIDx, Vectorize)

Common choices:
- `ParallelType::TIDx` maps warp lanes across sub-tiles, enabling concurrent ld/stmatrix ops
- `ParallelType::Vectorize` on the innermost element axis matches instruction granularity and removes a serial inner loop

Rationale: match warp-level transfer patterns and per-thread fragment widths.

Refs: Q&A Topic 7 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.

