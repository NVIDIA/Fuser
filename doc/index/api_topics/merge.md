# Merge (axis transform)

## Synopsis
`Merge` is a scheduling expression that combines two `IterDomain` axes (outer and inner) into a single axis. The inner axis is the fast-changing dimension.

## Source
- Class: [`Merge`](../../../csrc/ir/internal_nodes.h#L1764)

## Overview
`Merge` is the inverse of `Split`: it fuses two adjacent axes into one, restoring a single iteration space. It’s commonly used after scheduling to simplify loop structure or to prepare for views.

Key APIs:
- Accessors: `out()`, `outer()`, `inner()`
- Axis-level API: `TensorDomain::merge(axis_o, axis_i)`

## Example
```cpp
// Suppose td has axes [I0, I1] and you previously split I1.
// You can merge them back:
td->merge(/*axis_o=*/0, /*axis_i=*/1);
// The loop domain reduces by one axis accordingly.
```

## Additional Guidance
- Merge is order-sensitive: `axis_i` is the fast-changing dimension.
- Ensure the axes are compatible (iter/reduction/broadcast type) before merging.

## Where to Look in the Codebase
- Definition: [`internal_nodes.h`](../../../csrc/ir/internal_nodes.h#L1764)
- Axis-level API: [`TensorDomain::merge`](../../../csrc/ir/internal_base_nodes.h#L695)
- Related transforms: `Split`, `Swizzle`, `Resize`

## See Also
- `Split`: split one axis into two (`split.md`)
- `IterDomain` / `TensorDomain`: axis and tensor domains (`iterdomain.md`, `tensor_domain.md`)
