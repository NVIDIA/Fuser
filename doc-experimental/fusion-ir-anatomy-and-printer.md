# Fusion IR Anatomy and Printer Semantics (Heavily Commented)

## Key headers
- IR interfaces: [`interface_nodes.h`](../csrc/ir/interface_nodes.h)
- IR printing: [`iostream.h`](../csrc/ir/iostream.h)

## What you’ll learn
- How to read Fusion IR prints (with vs without transforms)
- Meaning of symbolic extent labels (e.g., `iS0{16}`) and memory space letters (`g`, `l`)
- Difference between `Set` and `Set.Permute` and where `cache_op` hints appear

Primary reference: `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`

---

## Example IR (excerpt)

```text
=== Fusion IR (without transforms) ===
Inputs:
  T0_g_float[iS0{16}, iS1{32}]
Outputs:
  T4_g_float[iS8{16}, iS9{32}]

%kernel {
T1_l_float[iS2{16}, iS3{32}]
   = Set( T0_g_float[iS0{16}, iS1{32}], cache_op=Streaming )
T2_l_float[iS5{32}, iS4{16}]
   = Set.Permute( T1_l_float[iS2{16}, iS3{32}], cache_op=Streaming )
T3_l_float[iS7{16}, iS6{32}]
   = Set.Permute( T2_l_float[iS5{32}, iS4{16}], cache_op=Streaming )
T4_g_float[iS8{16}, iS9{32}]
   = T0_g_float[iS0{16}, iS1{32}]
   + T3_l_float[iS7{16}, iS6{32}];
} // %kernel
```

### Reading the pieces
- `T#_{mem}_{dtype}[...]`:
  - `T0`, `T1`, ... are TensorView identifiers in the printed graph
  - `{mem}`: `g` = global/global-like, `s` = shared memory, `l` = local/register-like
  - `{dtype}`: element type, e.g., `float` for float32
- Bracketed extents `[iS0{16}, iS1{32}]`:
  - `iS#` is a symbolic identifier for an IterDomain extent (axis size) in this IR snapshot
  - `{16}` shows the concrete size for that axis when known
  - Labels are per-node identities, not "same size implies same label" across TVs
- `Set` vs `Set.Permute`:
  - `Set(...)` models moving/aliasing values between TVs (a copy in IR terms)
  - `Set.Permute(...)` indicates the output TV’s logical domain is a permutation of the input (e.g., transpose)
- `cache_op=Streaming`:
  - Cache hint carried on the load/store; options include `Unspecified`, `AllLevels`, `Streaming`, `Global`

### With transforms vs without transforms
- With transforms: prints per-TV transform sections (root/logical/loop domain views)
- Without transforms: prints the operator graph only (cleaner for high-level structure)

---

## Mini walk-through: transpose back before add (rectangular case)

When `A` is `M×N` and `transpose(A)` is `N×M`, adding requires matching shapes. Two options:
- Transpose back (`transpose(transpose(A))`) to restore `[M, N]`
- Or choose a different op/broadcast strategy that aligns shapes

In our sample, we insert a second `Set.Permute` to return to `[M, N]` before the `add`.

---

## Tips
- Use `tv->printTransforms()` to see per-TV domain history and mappings
- Use "with transforms" print to study view/schedule transforms (`split/merge/reorder`, `broadcast`, `permute`, `view`)
- Use "without transforms" print for a concise operator graph

See also:
- Getting started: `./getting-started-tv-add-2d.md`
- Reading IR dumps: `./how-to-read-fusion-ir-dumps.md` (next article)
