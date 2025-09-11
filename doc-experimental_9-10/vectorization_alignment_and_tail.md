## Vectorization Alignment and Tail Handling (Prologue/Main/Epilogue)

When vectorizing with width V, align the head to V, process the main body vectorized, and finish a scalar epilogue for the remainder.

Pattern:
1) Prologue: scalar until index % V == 0
2) Main: vectorized blocks of V
3) Epilogue: scalar for remaining < V elements

Refs: Q&A Topic 16 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.

See also:
- Slicing offsets: `./slicing_offsets_and_indexing.md`
- Split semantics (for divisible factors): `./split_merge_reorder_basics.md`


