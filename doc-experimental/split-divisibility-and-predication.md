# Split Divisibility and Predication

## Key headers
- Vectorization helpers: [`csrc/scheduler/vectorize_helper.h`](../csrc/scheduler/vectorize_helper.h)
- Scheduler registry/types: [`csrc/scheduler/registry.h`](../csrc/scheduler/registry.h), [`csrc/scheduler/scheduler_types.h`](../csrc/scheduler/scheduler_types.h)

## Concept
- Split transforms an axis `I` into `(Io, Ii)` with factor `f` so `i = Io*f + Ii`
- Divisible if `extent % f == 0`; otherwise indivisible → "holes" in iteration
- Holes require predication (guarding) to preserve correctness

References:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 3)
- Guide: `../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md`

---

## Heavily commented example

```cpp
// Given logical extent N and factor f
int64_t N = 13; // not divisible by 2
int64_t f = 2;  // split factor

// Conceptual loop structure after split
auto ceilDiv = [](int64_t a, int64_t b){ return (a + b - 1)/b; };
for (int64_t Io = 0; Io < ceilDiv(N, f); ++Io) {
  for (int64_t Ii = 0; Ii < f; ++Ii) {
    int64_t i = Io * f + Ii;
    if (i < N) {         // predicate masks the hole when indivisible
      use(i);
    }
  }
}
```

- Allocation domain vs loop domain:
  - If allocation uses `(Io, Ii)`, size becomes `ceilDiv(N,f)*f` (may exceed `N`)
  - Strong correctness may require zero-fill semantics for holes

---

## Performance implications and strategies
- Predication cost is often small if the tail is short
- To preserve vectorization/TMA:
  - Pick factors that divide the innermost contiguous extent
  - Pad extents to multiples and mask tails
  - Isolate scalar tail paths (prologue/epilogue) and keep main loop divisible

See also:
- `./vectorization-alignment-and-tails.md`
- `./weak-vs-strong-correctness.md`
