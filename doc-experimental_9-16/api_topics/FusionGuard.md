# FusionGuard

Source: [FusionGuard](../../csrc/fusion_guard.h#L21)

## Synopsis
- **Kind**: class (RAII scope helper)
- **File**: `csrc/fusion_guard.h`
- **What it represents**: Guard that sets/restores the current active `Fusion` for IR-building APIs.

## Purpose
- Ensures that IR construction utilities (`IrBuilder`, Tensor/Expr creation) operate on the intended [Fusion](../../csrc/fusion.h#L134) within a lexical scope.
- Prevents accidental cross-container mutations when multiple fusions are being built/testing concurrently.

## How to use it
```cpp
Fusion fusion;
{
  FusionGuard fg(&fusion);
  // Build IR targeting `fusion`
}
```
