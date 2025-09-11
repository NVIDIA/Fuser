# What Is a ViewAnalysis, and How Is It Used?

## Key headers
- [`AnalyzeViewResult`, `analyzeView(...)`](../csrc/transform_view.h), and implementation in [transform_view.cpp](../csrc/transform_view.cpp)
- Frontend reshape overload in [`alias.h`](../csrc/ops/alias.h)

## TL;DR
A ViewAnalysis is nvFuser’s compact plan for reshaping without moving data. It specifies which axes to squeeze, which to broadcast, and which split/merge steps to apply so an input logical domain becomes a target logical domain.

Primary sources:
- Q&A/log notes: `../doc-bot/experimenting/9-3-2025/chat_log_3.md` (search: "view analysis", `analyzeView`)
- Code: `../csrc/transform_view.h` (struct `AnalyzeViewResult`), `../csrc/transform_view.cpp` (function `analyzeView(...)` and `reshape(inp, view_analysis)`)
- Frontend callsite: `../csrc/ops/alias.cpp` (reshape overload that calls `analyzeView` then applies it)

---

## Where it lives (pointers)
- `struct AnalyzeViewResult` — holds:
  - `broadcast_axes`, `squeeze_axes` (boolean vectors)
  - `transforms` — ordered `SplitTransform`/`MergeTransform` decisions
- `AnalyzeViewResult analyzeView(...)` — computes the plan from `original_sizes` → `new_sizes`
- `TensorView* reshape(inp_tv, const AnalyzeViewResult& view_analysis)` — applies the plan and emits a `ViewOp`

---

## The flow (frontend → analysis → apply)

```cpp
// Frontend entry (simplified)
TensorView* reshape(TensorView* x,
                    const std::vector<int64_t>& original_sizes,
                    const std::vector<int64_t>& new_sizes) {
  // 1) Compute the plan
  auto view_analysis = analyzeView(x, original_sizes, new_sizes);
  // 2) Apply: squeeze → applyViewTransforms (split/merge) → broadcast
  return reshape(x, view_analysis);
}
```

- Empty dimension guard: the alias frontend has an "empty reshape" branch if any original size is 0 (producing a full of zeros). Otherwise, it computes the plan and applies it.

---

## How the plan is computed (mental model)
- Walk original and target shapes with a running product `current_size`
- Emit `SplitTransform` when `current_size % new_dim == 0`
- Emit `MergeTransform` to accumulate original dims until divisibility holds
- Record `squeeze_axes` (remove size-1 dims) and `broadcast_axes` (insert size-1 dims)

---

## Applying the plan

```cpp
// Pseudocode for applying AnalyzeViewResult
TensorView* reshape(TensorView* inp, const AnalyzeViewResult& r) {
  TensorView* x = inp;
  if (any(r.squeeze_axes)) {
    x = squeeze(x, r.squeeze_axes); // remove size-1 dims first
  }
  x = r.transforms.empty() ? x : applyViewTransforms(inp, x, r); // split/merge
  if (any(r.broadcast_axes)) {
    x = broadcast(x, r.broadcast_axes); // insert size-1 dims
  }
  // Emit ViewOp to link new TV in the IR
  return x;
}
```

---

## Heavily commented example

Goal: `[2, 3, 4] → [2, 2, 2, 3]` (reshape then transpose dim-1 and dim-2)

```cpp
// 1) Build a 3-D input with symbolic sizes
auto tv0 = TensorViewBuilder().ndims(3).dtype(DataType::Float)
                              .contiguity({true,true,true})
                              .shape({-1,-1,-1}) // symbolic
                              .build();
// Register as fusion input
fusion->addInput(tv0);

// 2) Reshape via view analysis
// analyzeView computes: split(4->2,2), merge as needed, etc.
auto tv1 = reshape(tv0, std::vector<int64_t>{2,3,4}, std::vector<int64_t>{2,2,2,3});

// 3) Transpose logical dims 1 and 2 (no data move; IR permute)
auto tv2 = transpose(tv1, 1, 2);

// 4) Inspect transforms; tv2 has view + permute in its history
tv2->printTransforms();
```

Notes:
- All steps are IR-level; no data is moved during IR building
- At execution, indexing/scheduling respect the logical/loop domains produced by these transforms

---

## “Empty” dimensions, `-1` inference, and validation (pointers)
- Empty reshape branch in `alias.cpp` handles zero-sized dims (returns a zero-filled tensor)
- Single `-1` inference rule is enforced (location depends on branch)
- Divisibility checks ensure legal split/merge decisions

See also:
- `./reshape-empty-dimensions.md`
- `./squeeze-and-broadcast.md`
