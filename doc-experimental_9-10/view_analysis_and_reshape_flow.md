## View Analysis and Reshape Flow: analyzeView → applyViewTransforms → ViewOp

This guide explains how nvFuser performs reshape/view without data movement, using a compact plan and a single IR node.

### High-level flow

1) `analyzeView(x, original_sizes, new_sizes)` builds an `AnalyzeViewResult` with:
   - `squeeze_axes`: which size-1 dims are removed
   - `transforms`: ordered Split/Merge to reconcile sizes
   - `broadcast_axes`: which size-1 dims are inserted
2) `transform_view::reshape(inp_tv, view_analysis)` applies:
   - Squeeze → Apply view transforms → Broadcast
3) Emit `ViewOp(out, in)` in the IR linking the original and new `TensorView`

```41:57:/opt/pytorch/nvfuser/csrc/transform_view.h
struct AnalyzeViewResult { ... };
AnalyzeViewResult analyzeView(...);
NVF_API TensorView* reshape(TensorView* inp_tv, const AnalyzeViewResult&);
```

```367:381:/opt/pytorch/nvfuser/csrc/transform_view.cpp
// reshape(inp_tv, view_analysis): squeeze → applyViewTransforms → broadcast
```

```445:456:/opt/pytorch/nvfuser/csrc/ir/nodes.cpp
// ViewOp::toString shows:  out = view(in)
```

### Handling `-1` and empty dimensions

- Single `-1` is inferred; constraints validated by `inferViewShapes`
- If original sizes contain a `0`, the “empty reshape” path constructs a zero-filled tensor via `full(...)` with shape constraints; see `../csrc/ops/alias.cpp`

### Practical notes

- `ViewOp` is the IR node; Split/Merge are used to construct the new logical domain and are not separate IR exprs
- Broadcasting and squeezing are applied around the Split/Merge plan as needed

### Related APIs

- Squeeze/Broadcast/Permute: `../csrc/ops/alias.cpp`
- View plumbing and constraints: `../csrc/transform_view.{h,cpp}`

See also:
- Empty reshape semantics: `./empty_dimensions_in_reshape.md`
- IR anatomy and dumps: `./fusion_ir_anatomy_and_printer_semantics.md`, `./how_to_read_fusion_ir_dumps.md`


