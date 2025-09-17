# Statement

Source: [Statement](../../csrc/ir/base_nodes.h#L96)

## Synopsis
- **Kind**: class (inherits from `NonCopyable`, `PolymorphicBase`)
- **File**: `csrc/ir/base_nodes.h`
- **What it represents**: Base of all IR nodes; `Val` and `Expr` both derive from `Statement`.

## Purpose
- Provides identity, cloning, dispatch, and RTTI-style helpers shared by values and expressions in the nvFuser IR.
- Friends with IR container utilities and visitors; participates in ownership by [Fusion](../../csrc/fusion.h#L134).

## See also
- [Val](../../csrc/ir/base_nodes.h#L224), [Expr](../../csrc/ir/base_nodes.h#L505)
- [IrContainer](../../csrc/ir/container.h#L35), [Fusion](../../csrc/fusion.h#L134)
