## "Empty" Dimensions in Reshape: Semantics and Implications

This article clarifies how nvFuser treats reshapes involving zero-sized dimensions and how it differs from the standard view-only path.

### Empty reshape path

When any original size is 0, reshape follows a special path that constructs a zero-filled tensor of the target shape with constraints:

```70:90:/opt/pytorch/nvfuser/csrc/ops/alias.cpp
// If any original dim is 0, build new_shape; require at most one -1; require an output 0
return full(new_shape, x->fusion()->zeroVal(x->dtype()), x->dtype());
```

Implications:
- Produces a tensor filled with the neutral zero value (not a view of input)
- Enforces: ≤1 inferred `-1`; output shape must contain at least one 0 if input does

### Normal (non-empty) path

Uses `analyzeView(...)` to compute a view plan, then emits a `ViewOp` that reinterprets shape without data movement:
- See `../csrc/transform_view.{h,cpp}` and `../csrc/ops/alias.cpp`

### Practical guidance

- For zero-sized inputs, do not expect a view-only reshape; you’ll get a zero-filled output tensor
- For dynamic shapes, prefer `reshape(TV, new_sizes)` with `-1` at one position when inference is needed


