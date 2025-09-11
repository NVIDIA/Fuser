## Consumer Schedule Constraints for TMA

Rules of thumb:
- Separate tile vs non‑tile transforms in the consumer; do not merge/swizzle tile axes with non‑tile axes
- Ensure the shared‑memory tile is contiguous in the allocation domain
- Allocate whole tiles (allocation is an integer multiple of the tile shape)
- Respect swizzle constraints when applicable (e.g., inner size matches swizzle unit)

Refs: Q&A Topic 6 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.

See also:
- FTTC: `./tma_fundamentals_fttc.md`
- LdMatrix/StMatrix: `./ldmatrix_stmatrix_inner_loop_parallelization.md`


