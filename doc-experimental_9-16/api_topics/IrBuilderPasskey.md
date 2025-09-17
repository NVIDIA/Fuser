# IrBuilderPasskey

Source: [IrBuilderPasskey](../../csrc/ir/builder_passkey.h#L16)

## Synopsis
- **Kind**: class (passkey idiom)
- **File**: `csrc/ir/builder_passkey.h`
- **What it represents**: A capability token required to invoke IR node constructors, enforcing construction through controlled builders.

## Purpose
- Restricts creation of IR nodes (`IterDomain`, `TensorDomain`, etc.) to authenticated contexts (e.g., `IrBuilder`), preserving invariants across the IR.
- Encourages use of dedicated construction helpers rather than ad-hoc direct instantiation.

## See also
- [Statement](../../csrc/ir/base_nodes.h#L96), [Val](../../csrc/ir/base_nodes.h#L224), [Expr](../../csrc/ir/base_nodes.h#L505)
- [Fusion](../../csrc/fusion.h#L134), [FusionGuard](../../csrc/fusion_guard.h#L21)
