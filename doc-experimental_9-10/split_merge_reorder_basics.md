## Split/Merge/Reorder Basics; Divisible vs Indivisible Split; Predication

This article introduces IterDomain transforms and the implications of indivisible splits.

### Concepts

- Split: rewrite axis I → (Io, Ii) with factor f; index i = Io*f + Ii
- Merge: inverse of split; linearize two axes
- Reorder: permute axis order

### Divisible vs Indivisible split

- Divisible: `extent % f == 0` (no holes)
- Indivisible: introduces “holes” that must be predicated; allocation may overshoot to `ceilDiv(extent,f)*f`

Predication pattern:
```cpp
for (Io in 0..ceilDiv(N,f))
  for (Ii in 0..f-1) {
    int i = Io*f + Ii;
    if (i < N) use(i); // mask holes
  }
```

Refs: Q&A Topic 3 in `../doc-bot/experimenting/8-28-2025/q_and_a.md`.

See also:
- Vectorization alignment: `./vectorization_alignment_and_tail.md`
- Correctness/holes: `./weak_vs_strong_correctness.md`


