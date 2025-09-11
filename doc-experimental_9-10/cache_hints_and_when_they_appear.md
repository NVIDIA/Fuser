## Cache Hints (`cache_op`) and When They Appear

nvFuser prints cache hints on load/store expressions to communicate desired cache behavior.

### Enum values

Defined in `../csrc/type.h` (enum `CacheOp`):
- `Unspecified`: default behavior
- `AllLevels`: cache at all levels
- `Streaming`: prefer streaming / less polluting
- `Global`: direct to global paths

Printer examples show hints on `Set`/`Set.Permute`:

```33:41:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md
T1_l_float[...] = Set(T0_g_float[...], cache_op=Streaming)
T2_l_float[...] = Set.Permute(T1_l_float[...], cache_op=Streaming)
```

### Practical guidance

- Hints depend on lowering choices; not all loads/stores carry explicit hints
- Consider `Streaming` for one-pass reads to reduce cache pollution; measure

### References

- Enum and printer operators: `../csrc/type.h`
- IR examples: `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`


