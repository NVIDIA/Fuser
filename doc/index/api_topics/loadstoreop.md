# LoadStoreOp

## Synopsis
Explicit data-movement IR between GPU memory spaces (global, shared, register), enabling generation of specialized instructions (e.g., `cp.async`, `ldmatrix`).

## Source
- Class: [`LoadStoreOp`](../../../csrc/ir/internal_nodes.h#L1665)

## Overview
`LoadStoreOp` models movement of values across memory spaces with optional cache hints. It is used by lowerings that generate hardware-accelerated transfers and cache operations.

Key APIs:
- Accessors: `in()`, `out()`
- Kind: `opType()` → `LoadStoreOpType` (e.g., Set, Load, Store, CpAsync)
- Cache hints: `cacheOp()` / `setCacheOp(...)`
- Rendering/eval: `toString`, `toInlineString`, `evaluate(...)`

Behavior notes:
- Setting `opType` to non-Set and non-CpAsync resets cache op to `Unspecified`.
- Used heavily in tiled/mma pipelines and async shared-memory staging.

## Example
```cpp
// Pseudocode: stage from global to shared via cp.async-like op
TensorView* gmem = ...; // global
TensorView* smem = ...; // shared-allocated
// Lowering will emit appropriate memory instructions based on op type.
// IR: LoadStoreOp(CpAsync, smem, gmem)
```

## Related
- MMA/TMA utilities in `scheduler/mma_utils.*`
- Caching helpers and circular buffering patterns

## References
- Decl: `../../../csrc/ir/internal_nodes.h#L1665`
