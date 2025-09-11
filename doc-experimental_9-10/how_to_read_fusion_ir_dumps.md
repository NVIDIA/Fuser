## How to Read Fusion IR Dumps

This guide explains the notation used by nvFuser’s Fusion IR printer: tensor naming, memory spaces, symbolic axis labels, and how to switch between “with” and “without” transforms.

### Legend

- `T#_{mem}_{dtype}[axes...]` — TensorView with memory space and dtype
  - `{mem}`: `g` global, `l` local, `s` shared, `t` tensor
  - `{dtype}`: e.g., `float`, `half`
- Axis labels: `iS# {extent}`
  - `iS#` is a symbolic IterDomain label; `{extent}` shows concrete size
  - Labels are per-IterDomain node inside one printed IR snapshot; equal numbers across TVs do not imply identity unless it’s the same node

### Two IR views

- “With transforms”: includes per-TensorView transform history (split/merge/reorder/view) and TransformPrinter section
- “Without transforms”: concise ops graph; no per-TV transform annotations

```36:46:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp
  std::cout << "\n=== Fusion IR (with transforms) ===\n";
  fusion->print(std::cout, /*include_tensor_transforms=*/true);
  std::cout << "\n=== Fusion IR (without transforms) ===\n";
  fusion->print(std::cout, /*include_tensor_transforms=*/false);
```

### Example (annotated)

```28:43:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md
Inputs:
  T0_g_float[iS0{16}, iS1{32}]           // global float 16x32
Outputs:
  T4_g_float[iS8{16}, iS9{32}]           // global float 16x32

%kernel {
T1_l_float[iS2{16}, iS3{32}]             // local temp
   = Set( T0_g_float[iS0{16}, iS1{32}], cache_op=Streaming )
T2_l_float[iS5{32}, iS4{16}]             // permute to (32,16)
   = Set.Permute( T1_l_float[iS2{16}, iS3{32}], cache_op=Streaming )
T3_l_float[iS7{16}, iS6{32}]             // permute back to (16,32)
   = Set.Permute( T2_l_float[iS5{32}, iS4{16}], cache_op=Streaming )
T4_g_float[iS8{16}, iS9{32}]
   = T0_g_float[iS0{16}, iS1{32}] + T3_l_float[iS7{16}, iS6{32}];
}
```

### `Set` vs `Set.Permute` and cache hints

- `Set`: move/alias between TVs with same logical axis order
- `Set.Permute`: consumer’s logical axes are a permutation of the producer (e.g., transpose)
- `cache_op=...` hints appear on loads/stores; see `../csrc/type.h` (`CacheOp` enum) for values: `Unspecified`, `AllLevels`, `Streaming`, `Global`

### Useful sources

- Example and annotations: `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`, `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp`
- IR node and printer code: `../csrc/ir/nodes.cpp`, `../csrc/tensor_view.cpp`
- Alias/permute ops: `../csrc/ops/alias.cpp`
- Cache hints: `../csrc/type.h` (`CacheOp`)

See also:
- IR anatomy: `./fusion_ir_anatomy_and_printer_semantics.md`
- Domains and mapping: `./domains_iter_tensor_views.md`, `./domain_mapping_and_indexing.md`


