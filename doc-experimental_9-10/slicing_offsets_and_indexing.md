## Slicing, Offsets, and Indexing

Consumer slicing is realized via index offsets on producers rather than materializing sub-tensors.

Example (conceptual):
```cpp
// C[0:32] = (A + B)[16:48]
for (int i = 0; i < 32; ++i) {
  C[i] = A[i+16] + B[i+16];
}
```

IndexCompute derives the `i+16` offsets; compute-at ensures only the consumed range is computed. No extract tensors are required.

Refs: Q&A Topics 15–16 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.

See also:
- Domain mapping and replay: `./domain_mapping_and_indexing.md`
- Vectorization tails: `./vectorization_alignment_and_tail.md`

