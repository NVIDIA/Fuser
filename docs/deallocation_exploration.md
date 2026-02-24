# Deallocation Exploration: Excluding Views from Deallocation

## Problem

The current `insertDeallocations` in `allocate_and_deallocate.cpp` deallocates all TVs except fusion inputs and outputs. However, some TVs are **views** that don't own memory:

1. **ShardByStream outputs** - Slices of the source TV (via `in_tensor.chunk().at(index)`)
2. **HirAliasSelect outputs** - Alias "slice" of the input (via `input.select(axis, index)`)
3. **SymmetricContiguousView outputs** - View of cached symmetric memory
4. **Fusion-level aliases** - Outputs with `AllocationType::ReuseBuffer` or `AllocationType::Evaluate`

Deallocating these would be incorrect—they reuse buffers owned by other TVs.

## MemoryType Analysis

**Conclusion: MemoryType does NOT distinguish views from allocations.**

`MemoryType` enum (`Local`, `Shared`, `Global`, `Tensor`, `Symmetric`) is used for **GPU memory placement**:
- Where the buffer lives (local/thread, shared/block, global/device)
- Scheduler and codegen decisions (e.g., `prepareForMemoryTypePromotion`)

View ops like `ShardByStream` use `ops::newValLike(source)` which creates a new TensorView via `newOutputTV`. The output inherits domain from source but gets default MemoryType (not explicitly copied from source in `newOutputTV`). In Host IR context after lowering, most tensors are `MemoryType::Global`. So both allocated buffers and view outputs typically have the same MemoryType.

**MemoryType cannot be used** to skip view TVs during deallocation.

## Host IR View-Producing Ops

| Op | Output semantics | Allocates? |
|----|------------------|------------|
| `ShardByStream` | Aliasing slice via `in.chunk().at(index)` | No |
| `HirAliasSelect` | Slice via `input.select(axis, index)` | No |
| `SymmetricContiguousView` | View of `SymMemForContiguousView::tensor()` | No |
| `kir::Allocate` | Explicit allocation | Yes (buffer owner) |

## Expr-Eval Outputs Without Explicit Allocate

**Important nuance**: Outputs of expr-eval ops (add, BinaryOp, etc.) may **not** have an explicit `kir::Allocate`:

1. **`insertAllocations`** only preallocates for `MatmulOp` and `LinearOp` (`needsOutputPreallocation`). Add, BinaryOp, etc. are skipped.
2. **`lowerSegment` (ExprEval, no loops)**: When `loop_nest.empty()`, it just pushes exprs—no allocations. Outputs like `c = add(a, b)` never get Allocate.
3. **`lowerSegment` (ExprEval, with loop)**: When the output's allocation is already stream-sharded, no Allocate is inserted; the expr uses the output directly.
4. **`HostIrLower::lower`**: For standalone ops (e.g., single add), it pushes the expr with no Allocate.

In these cases, the evaluator allocates at runtime via `expr->evaluate()`. If we only deallocate TVs with explicit Allocate, we would **leak** these buffers.

## Recommended Approach: Deallocate TVs That Own Memory

A TV **owns memory** if:
1. It is the buffer of a `kir::Allocate`, **or**
2. It is the output of an expr that allocates (i.e., not a view-producing op).

**View ops** (output reuses buffer; do NOT deallocate):
- `ShardByStream`
- `HirAliasSelect`
- `SymmetricContiguousView`

### Current Bug

`outermost_scope` is populated from:
1. `alloc->buffer()` for each `kir::Allocate`
2. **Every TV used as input** to any expression

So ShardByStream outputs get incorrectly considered for deallocation (they're views), while expr-eval outputs without Allocate are not considered (we'd leak them).

### Solution: Ownership-Based Deallocation

```cpp
bool isViewProducingOp(Expr* e) {
  return e->isA<ShardByStream>() || e->isA<HirAliasSelect>() ||
         e->isA<SymmetricContiguousView>();
}

bool tvOwnsMemory(Fusion* fusion, TensorView* tv) {
  if (tv->isFusionInput() || tv->isFusionOutput()) return false;
  if (fusion->getOutputAlias(tv).type != AllocationType::New) return false;
  if (/* tv is buffer of some kir::Allocate */) return true;
  auto* def = tv->definition();
  if (!def) return false;  // No definition, not our allocation
  return !isViewProducingOp(def);  // Expr allocates unless it's a view op
}
```

Deallocate only when `tvOwnsMemory(hic, tv)` is true. This covers:
- Explicit Allocate buffers
- Expr-eval outputs (add, MatmulOp, etc.) that allocate at runtime
- Excludes: fusion io, ReuseBuffer/Evaluate aliases, view outputs (ShardByStream, HirAliasSelect, SymmetricContiguousView)

## Fusion-Level Aliases (AllocationType)

The fusion has `io_alias_` with `AllocationType::New`, `ReuseBuffer`, `Evaluate`. For Host IR:

- **ReuseBuffer / Evaluate**: Outputs reuse another buffer; we must NOT deallocate them. Use `fusion->getOutputAlias(tv)` and skip when `type != AllocationType::New`.
- **HostIrContainer** inherits from Fusion, so `getOutputAlias` is available when the pass runs on the container.

## Summary

| Approach | Use? |
|----------|------|
| MemoryType | No—doesn't distinguish views |
| Only deallocate Allocate buffers | **No—too restrictive**; misses expr-eval outputs (add, etc.) that allocate at runtime |
| Ownership-based (exclude view ops) | **Yes—recommended** |

Implement **ownership-based deallocation**: deallocate TVs that own memory—either (1) buffer of `kir::Allocate`, or (2) output of an expr that is not a view op (ShardByStream, HirAliasSelect, SymmetricContiguousView).
