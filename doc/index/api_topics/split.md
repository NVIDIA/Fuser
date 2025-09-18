# Split (axis transform)

## Synopsis
`Split` is a scheduling expression that splits an `IterDomain` into two axes (outer and inner) by a given factor.

## Source
- Class: [`Split`](../../../csrc/ir/internal_nodes.h#L1720)

## Overview
`Split` transforms a single axis into two axes. With factor F:
- inner_split = true: produces [ceilDiv(extent, F), F]
- inner_split = false: produces [F, ceilDiv(extent, F)]

This operation is fundamental to shaping loop nests and preparing vectorization/tiling.

Key APIs (on `Split` Expr and related):
- Constructor stores `in`, `factor`, and `inner_split` flag
- Outputs: `outer()`, `inner()`, Inputs: `in()`, Attributes: `factor()`, `innerSplit()`
- Axis-level convenience exists via `TensorDomain::split(axis, factor, inner_split)`

## Example
```cpp
// Given TensorDomain td with axis 1 of extent N
td->split(1, IrBuilder::create<Val>(4, DataType::Index), /*inner_split=*/true);
// Loop domain now has an extra axis; downstream scheduling can reorder/parallelize
```

## Additional Guidance
- Split interacts with parallelization and vectorization; the inner axis is often mapped to thread or vector lanes.
- For symbolic or runtime extents, split uses symbolic `ceilDiv` semantics.

## Where to Look in the Codebase
- Definition: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L1720)
- Axis-level API: [`TensorDomain::split`](../../../csrc/ir/internal_base_nodes.h#L693)
- Related transforms: `Merge`, `Swizzle`, `Resize`

## See Also
- `Merge`: combine two axes into one (`merge.md`)
- `IterDomain` / `TensorDomain`: axis and tensor domains (`iterdomain.md`, `tensor_domain.md`)
