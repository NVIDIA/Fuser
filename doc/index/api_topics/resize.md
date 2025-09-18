# Resize (axis transform)

## Synopsis
`Resize` expands an `IterDomain` by adding elements on the left/right sides, effectively adjusting start/stop while preserving interior mapping.

## Source
- Class: [`Resize`](../../../csrc/ir/internal_nodes.h#L1936)

## Overview
A `Resize` produces an output `IterDomain` whose iteration range is expanded by `left_expand` and `right_expand`. This is useful for halo regions, padding, or aligning tiles for cooperative loads.

Key APIs:
- Accessors: `out()`, `in()`
- Attributes: `leftExpand()`, `rightExpand()` (as `Val*`)
- Printing: `toString`, `toInlineString`

## Example
```cpp
// Expand an axis by 1 element on both sides (halo)
// IR: Resize(out, in, /*left_expand=*/1, /*right_expand=*/1)
```

## Related
- Scheduling tools: `scheduler/tools/resize_utils.*`
- Axis transforms: `Split`, `Merge`, `Swizzle`

## References
- Decl: `../../../csrc/ir/internal_nodes.h#L1936`
