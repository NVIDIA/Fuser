# Swizzle / Swizzle2D

## Synopsis
Axis remapping expressions that permute iteration/data patterns over 1D/2D tiles to improve locality, avoid bank conflicts, or implement non-affine thread/data mappings.

## Sources
- `Swizzle`: [`../../../csrc/ir/internal_nodes.h#L1794`]
- `Swizzle2D`: [`../../../csrc/ir/internal_nodes.h#L1841`]

## Overview
Swizzling changes how indices map to data or loops. Two orthogonal attributes control behavior:
- Type: `SwizzleType` or `Swizzle2DType` (e.g., NoSwizzle, XOR, Z-mort, etc.)
- Mode: `SwizzleMode::{Data, Loop}`
  - Data mode permutes data layout (affects producers/consumers unless materialized)
  - Loop mode permutes iteration order without changing stored layout

Key APIs (Swizzle / Swizzle2D):
- Inputs: `inX()`, `inY()`; Outputs: `outX()`, `outY()`
- Attributes: `swizzleType()`; for 2D: `swizzleMode()` as well
- Printing: `toString`, `toInlineString`

## Example
```cpp
// Conceptual example: apply a 2D swizzle on a tile (ix, iy)
// IR: Swizzle2D(out_x, out_y, in_x, in_y, Swizzle2DType::SomePattern, SwizzleMode::Data)
```

## Related
- Tiling/vectorization utilities; shared memory layout planning
- `Merge`/`Split` to form tiles pre-swizzle

## References
- Decl: `../../../csrc/ir/internal_nodes.h#L1794`, `#L1841`
