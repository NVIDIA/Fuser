# Empty Dimensions in Reshape: Semantics, Inference, and Behavior

## Key headers
- Frontend reshape in [`alias.h`](../csrc/ops/alias.h) / `alias.cpp`
- Analysis and application in [`transform_view.h`](../csrc/transform_view.h) / `transform_view.cpp`

## What counts as “empty” here?
- An original logical size of 0 on any axis when calling the alias front-end reshape path
- This triggers a special branch: instead of creating a view, the API returns a zero-filled tensor of the requested output shape (subject to inference rules)

Pointers:
- Frontend path: `../csrc/ops/alias.cpp` (reshape overload)
- Inference details: `../csrc/transform_view.cpp` (non-empty path)
- Notes: `../doc-bot/experimenting/9-3-2025/key_insights.md`

---

## Rules in practice
- Single `-1` rule: at most one `-1` (inferred size) in the target shape
- Empty branch requirements:
  - Replace `-1` with `0` in the constructed new shape
  - The output must contain at least one `0`
  - Return `full(new_shape, 0, dtype)` (zero-filled) rather than a view
- Non-empty branch:
  - Enforce single `-1` inference; check divisibility; construct `AnalyzeViewResult`

---

## Heavily commented example scenarios

```cpp
// Scenario A: Empty dimension in input → zero-filled output
// Suppose runtime input has shape [0, 5]
auto tv = TensorViewBuilder().ndims(2).dtype(DataType::Float)
                             .contiguity({true,true})
                             .shape({-1, -1}) // symbolic, actual may be [0,5]
                             .build();
fusion->addInput(tv);

// Reshape request (original_sizes known, new_sizes allows -1 once)
// Empty branch fires because an original dimension is 0:
auto tv_out = reshape(tv,
  std::vector<int64_t>{0, 5},      // original_sizes (runtime)
  std::vector<int64_t>{-1, 10});   // new_sizes
// Behavior: constructs new_shape with -1→0, enforces ≥1 zero, returns zeros
// No data is read; semantics are defined via zero fill.
```

```cpp
// Scenario B: Non-empty path with inference and divisibility
// Input [4, 6], reshape → [2, 12]
auto tv2 = TensorViewBuilder().ndims(2).dtype(DataType::Float)
                              .contiguity({true,true})
                              .shape({4, 6})
                              .build();
fusion->addInput(tv2);

// No zero dims → normal path: compute AnalyzeViewResult + apply
auto tv2_out = reshape(tv2,
  std::vector<int64_t>{4, 6},
  std::vector<int64_t>{2, 12});
// analyzeView decides: maybe keep 2, merge (2×6)=12, no squeeze/broadcast
// applyViewTransforms builds the new logical domain and emits ViewOp
```

---

## Tips
- Keep `-1` inference unique; multiple `-1`s are rejected
- For documentation clarity, mention when a sample uses the empty branch vs normal path
- Test both branches to ensure consistent behavior and messaging

See also:
- `./what-is-view-analysis.md`
- `./squeeze-and-broadcast.md`
