## Fusion IR Anatomy and Printer Semantics

This article explains how to read nvFuser Fusion IR dumps: symbol labels, memory space indicators, and the difference between `Set` and `Set.Permute`. It includes an annotated example derived from `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp` and notes from `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`.

### Quick legend

- `T#_{mem}_{dtype}[...]` — TensorView name, memory space, and dtype
  - `{mem}`: `g` global, `l` local, `s` shared (when present), `t` tensor memory
  - `{dtype}`: e.g., `float`
- `iS# {extent}` — Symbolic IterDomain label with concrete extent in braces
  - Labels are per-IterDomain node within the printed IR snapshot
  - Same numeric value across TVs does not imply axis identity unless it refers to the same IterDomain node

### Annotated example dump (M=16, N=32)

```28:43:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md
Inputs:
  T0_g_float[iS0{16}, iS1{32}]           // T0: global tensor (16x32)
Outputs:
  T4_g_float[iS8{16}, iS9{32}]           // Output: global (16x32)

%kernel {
T1_l_float[iS2{16}, iS3{32}]             // l: local temp
   = Set( T0_g_float[iS0{16}, iS1{32}], cache_op=Streaming )
T2_l_float[iS5{32}, iS4{16}]             // logical permute to (32,16)
   = Set.Permute( T1_l_float[iS2{16}, iS3{32}], cache_op=Streaming )
T3_l_float[iS7{16}, iS6{32}]             // permute back to (16,32)
   = Set.Permute( T2_l_float[iS5{32}, iS4{16}], cache_op=Streaming )
T4_g_float[iS8{16}, iS9{32}]
   = T0_g_float[iS0{16}, iS1{32}]
   + T3_l_float[iS7{16}, iS6{32}];       // shapes now compatible for add
} // %kernel
```

Key points:
- The printer shows both a “with transforms” and “without transforms” view. With transforms includes per-TV transform history (split/merge/reorder/view).
- Cache hints (`cache_op`) appear on `Set` ops as guidance for memory behavior (e.g., `Streaming`). See `../csrc/type.h` for enums.
- `Set` vs `Set.Permute`:
  - `Set`: copy/alias between TVs with the same logical order.
  - `Set.Permute`: `Set` where the consumer’s logical domain is a permutation of the producer (e.g., transpose). Domains are aligned by permuting axes.

See also:
- IR dumps tutorial: `./how_to_read_fusion_ir_dumps.md`
- Cache hints reference: `./cache_hints_and_when_they_appear.md`

### Building the example

See `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp` for a runnable program that prints both IR variants and executes on CUDA.

```36:46:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp
  std::cout << "\n=== Fusion IR (with transforms) ===\n";
  fusion->print(std::cout, /*include_tensor_transforms=*/true);
  std::cout << "\n=== Fusion IR (without transforms) ===\n";
  fusion->print(std::cout, /*include_tensor_transforms=*/false);
```

### Where these constructs are defined

- Printer and IR nodes:
  - `../csrc/ir/nodes.cpp` — IR node types and toString/toInlineString
  - `../csrc/tensor_view.cpp` — `TensorView::toString`, `printTransforms`
- Ops producing these forms:
  - `../csrc/ops/alias.cpp` — `set`, `transpose` (`Set`/`Set.Permute` paths)
- Cache hints:
  - `../csrc/type.h` — `CacheOp` enums (Unspecified, AllLevels, Streaming, Global)

### Reading `iS#` labels safely

- Treat labels as identifiers of specific IterDomain nodes; equality of braces `{extent}` does not imply shared identity.
- Use the TransformPrinter output to see how producer axes map to consumer axes across `Set` and `Set.Permute` operations.

### Common pitfalls

- Assuming `T0` and `T1` share the same axis because both show `{16}` — they are distinct unless printer reuses the same IterDomain node (rare across new TVs created by Set/Permute).
- Confusing memory space (`g`, `l`, `s`) with layout — memory space is storage class; layout/transforms are shown via transforms and permutation.


