# How to Read Fusion IR Dumps (With Annotations)

## Goals
- Decode `T#_{mem}_{dtype}[iS# {...}]` names and what they imply
- Understand Set vs Set.Permute and where transforms show up
- Learn when to use “with transforms” vs “without transforms” prints

---

## Quick legend
- `T#` — TensorView id in this print
- `{mem}` — `g` = global memory, `l` = local/reg-like temporaries
- `{dtype}` — element type (e.g., `float`)
- `[iS0{16}, iS1{32}]` — symbolic axis ids with concrete extents when known
- `Set(...)` — value move/alias in IR; `Set.Permute(...)` — same but with permuted logical domain (e.g., transpose)
- `cache_op=` — cache hint on load/store (`Unspecified`, `AllLevels`, `Streaming`, `Global`)

---

## Example and guided notes

```text
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
}
```

- The `T0` input is 16×32 in global memory; intermediate `T1`..`T3` are locals
- `T2` permutes `(16,32) → (32,16)` (transpose); `T3` permutes back to `(16,32)`
- Final add is shape-compatible (`16×32 + 16×32 → 16×32`)

---

## With vs without transforms
- With transforms: shows per-TV domain history (root/logical/loop) and mappings
- Without transforms: concise operator graph for quick shape sanity

When debugging schedules or view sequences, prefer “with transforms.” When explaining a compute graph structure, “without transforms” is easier to read.

---

## Practical tips
- If two TVs both show `{16}` for a dim, do not assume it is the same axis: `iS#` identifiers are per-node labels in a print
- To trace where a dim came from, check the TransformPrinter section (in the "with transforms" print)
- For rectangular `M×N`, inserting a second permute before add may be required to restore shape compatibility

See also:
- `./fusion-ir-anatomy-and-printer.md`
- `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`
