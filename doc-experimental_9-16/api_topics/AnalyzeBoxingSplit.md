# AnalyzeBoxingSplit

Source: [AnalyzeBoxingSplit](../../csrc/device_lower/analysis/tma.cpp#L237)

## Synopsis
- **Kind**: class (inherits from `Pass`)
- **File**: `csrc/device_lower/analysis/tma.cpp`
- **What it analyzes**: Detects a “boxing split” along the traversal of the expression graph when inferring TMA (Tensor Memory Accelerator) tile/box/stride dimensions.

## Purpose
- During TMA analysis, we infer a set of dimensions (`TMADim`) that describe how data is tiled and traversed: partitioned (coordinate), box (tile region), and stride (step between tiles). A “boxing split” is a split/merge point where a previously partitioned path branches into an outer coordinate group and an inner box group.
- This pass identifies such split/merge expressions and updates the inferred dimensions so that the partitioned (coordinate) group is correctly propagated to the parent node of the box when valid.

## Concept: Boxing split
- A boxing split is a split or merge that splits a partitioned group into:
  - an outer coordinate group (must be a non-bulk group), and
  - an inner box group (either a bulk group or previously identified box group).
- Based on traversal direction, the expression can appear as either a `Split` or a `Merge`.

## Pattern recognized (from code comments)
Initial detection (non-bulk + partitioned → to):
```
               to
              /  \
from0[non-bulk]  from1[partitioned]
                   \
                    ...
                      \
                      box
```

If extent(box) divides extent(from1), the partitioned flag can be lifted to `to`:
```
               to[partitioned]
              /  \
from0[non-bulk]  from1
                   \
                    ...
                      \
                      box
```

Rationale: Before this pass runs, `box` is always an inner child of `partitioned`, so the first application encounters the pattern with `from1[partitioned]`. By validating divisibility and group membership (non-bulk/bulk), the pass can move the `partitioned` mark up the traversal to reflect how the tile box is produced.

## Conditions (simplified)
- Expression node is `Split` or `Merge`.
- Traversal frame has `from.size() == 2` and `to.size() == 1`.
- `from[0]` is in non-bulk groups (coordinate candidate).
- `from[1]` is in bulk groups or is already recognized as a `box` in `inferred_dims`.

## Actions
- If the box group was already recognized in prior passes, update its `partitioned` field to `to[0]` (no new dim entry needed).
- Otherwise, create a new `TMADim`:
  - `partitioned = to[0]`
  - `box = from[1]`
  - `tile = from[1]`
  - `stride = nullptr` (to be inferred elsewhere)

## Interaction with AnalyzeStridingSplit
- Earlier, `AnalyzeStridingSplit` may set up `box`, `tile`, and `stride` assuming a certain traversal (bulk/non-bulk). `AnalyzeBoxingSplit` may subsequently adjust the `partitioned` group to the appropriate ancestor node if the boxing pattern is found, ensuring consistency across tiling and striding inferences.

## See also
- [AnalyzeStridingSplit](../../csrc/device_lower/analysis/tma.cpp#L221)
- `Split`, `Merge` (IR transformations that form the split/merge expressions)
- `TMADim` (the structure holding `box`, `tile`, `stride`, `partitioned` used by TMA analysis)
