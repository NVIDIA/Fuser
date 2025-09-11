# Weak vs Strong Correctness

## Key headers
- Scheduler context: [`csrc/scheduler/registry.h`](../csrc/scheduler/registry.h), [`csrc/scheduler/scheduler_types.h`](../csrc/scheduler/scheduler_types.h)

## Definitions
- Weak correctness: hole elements may contain arbitrary data; consumers must guard/mask appropriately
- Strong correctness: hole elements contain the correct neutral value for the operation (e.g., 0 for sum, 1 for product, ±inf for min/max)

Reference:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 5)

---

## Why it matters
- Unpredicated reductions over regions with holes require strong correctness; otherwise results are corrupted
- Vectorization/TMA choices may favor weak correctness for performance when safe (e.g., not feeding reductions)

---

## Examples

```cpp
// Sum reduction over a tail with holes
// Strong correctness: holes are zero → unpredicated sum is valid
// Weak correctness: holes arbitrary → must predicate (mask) or split off tail
```

```cpp
// Neutral values per op
// sum: 0, product: 1, min: +inf, max: -inf
// Choose hole fill accordingly when aiming for strong correctness
```

---

## Strategies
- Predicate only where needed; keep main path fast
- Zero-fill (or neutral-fill) allocation domains for strong correctness where reductions require it
- Separate tile paths for TMA where strong correctness is impossible (see FTTC) and accept weak correctness if safe

See also:
- `./tma-fttc-and-weak-correctness.md`
- `./split-divisibility-and-predication.md`
