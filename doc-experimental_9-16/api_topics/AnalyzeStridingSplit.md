# AnalyzeStridingSplit

Source: [AnalyzeStridingSplit](../../../csrc/device_lower/analysis/tma.cpp#L191)

## Synopsis
- **Kind**: class (inherits from [Pass](../../csrc/device_lower/analysis/tma.cpp#L76))
- **File**: `csrc/device_lower/analysis/tma.cpp`
- **Approx. size**: ~38 lines

## Context (from code comments)
A striding split is a split or merge expression that splits a box group into
a tile group (outer) and a stride group (inner). Depending on the direction
of traversal, this expression can be either a split or merge. The stride
group must be a non-bulk group, and the tile group must be a bulk group. The
information extracted from a striding split is stored in inferred_dims_.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
