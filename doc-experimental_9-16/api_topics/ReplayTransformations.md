# ReplayTransformations

Source: [ReplayTransformations](../../csrc/transform_iter.h#L49)

## Synopsis
- **Kind**: class (inherits from `IterVisitor`)
- **File**: `csrc/transform_iter.h`
- **What it does**: Replays the history of a target tensor’s `IterDomain` transformations (split/merge/swizzle/resize) onto other IDs using a provided mapping.

## Purpose
- Given a target `TensorDomain`’s axis history and an `id_map` from its `IterDomain*` to those of another tensor, reapply equivalent transforms to align loop structures.
- Supports selective replay (`setReplaySwizzle`, `setReplayResize`, `setReplayRFactor`) and best-effort operation when exact reproduction isn’t possible.

## Algorithm (high level)
- Traverse the target’s defining `Expr`s in dependency order.
- For each encountered transform (`Split`, `Merge`, `Swizzle`, `Resize`):
  - Verify mapped inputs exist per `id_map` (forwarding/broadcast handling is performed elsewhere)
  - Apply the corresponding transform to the mapped IDs and track the new outputs within `id_map`
- Track leaf/loop IDs (`getLeafIDs`, `getUnorderedLeafIDs`) to reason about terminal loop axes.

## See also
- [BestEffortReplay](../../csrc/transform_iter.h#L318)
- [IterVisitor](../../csrc/iter_visitor.h#L39), [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)
