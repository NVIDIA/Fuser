## Transpose and Shape Compatibility for Elementwise Ops

This note explains how transpose affects shape compatibility and when an extra permute is required to align shapes for elementwise operations.

### Example context

- From `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp`: `B = transpose(copy(A))`, then `C = A + B`
- For non-square shapes (M≠N), `A` and `transpose(A)` do not share the same shape; you must permute back or choose a different operation

```33:42:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md
// A +(transpose A) requires either broadcast/reshape or an additional transpose back
```

### Takeaway

- Elementwise ops require shape compatibility; transpose swaps axes. If shapes differ, apply another permute or reshape/broadcast to match before addition.


