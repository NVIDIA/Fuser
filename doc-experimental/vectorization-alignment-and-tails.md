# Vectorization Alignment and Tail Handling

## Key headers
- Vectorization helpers: [`csrc/scheduler/vectorize_helper.h`](../csrc/scheduler/vectorize_helper.h)
- Scheduler utils/context: [`csrc/scheduler/utils.h`](../csrc/scheduler/utils.h), [`csrc/scheduler/registry.h`](../csrc/scheduler/registry.h)

## Problem
Vectorization requires aligned, contiguous inner loops. When slices or sizes aren’t multiples of the vector width, handle the misaligned head/tail without reading out-of-bounds or corrupting results.

Reference:
- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 16)

---

## Prologue / Main / Epilogue pattern

```cpp
int64_t start = 15;  // e.g., slice offset
int64_t len   = 31;  // slice length
int64_t V     = 4;   // vector width

int64_t head = (V - (start % V)) % V; // elements until alignment
int64_t main_len = (len - head) / V;  // number of full vectors
int64_t tail = (len - head) % V;      // remaining scalars

// Prologue: scalar until aligned
for (int64_t i = 0; i < head; ++i) {
  scalar_op(start + i);
}

// Main: vectorized loop over full chunks
for (int64_t k = 0; k < main_len; ++k) {
  vec_op(start + head + k * V);
}

// Epilogue: scalar tail
for (int64_t i = 0; i < tail; ++i) {
  scalar_op(start + head + main_len * V + i);
}
```

- No need to read before the slice start; process only the in-range elements
- Predication can replace explicit tail loops in generated code; the idea is identical

---

## Tips
- Choose the vectorization axis where contiguity and divisibility are most favorable
- When combined with splits, ensure the innermost contiguous extent is divisible by the vector width (or isolate tails)

See also:
- `./split-divisibility-and-predication.md`
