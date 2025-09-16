# Val

Source: [Val](../../csrc/ir/base_nodes.h#L224)

## Synopsis
- **Kind**: class (inherits from `Statement`)
- **File**: `csrc/ir/base_nodes.h`
- **What it represents**: Base class for IR values (scalars, tensors, ids) in nvFuser. Many IR node types derive from `Val`.

## Purpose
- Provides common interfaces for values (type info, printing, cloning) and participates in the `Statement`/`Expr` graph maintained by `Fusion`.

## See also
- [Expr](../../csrc/ir/base_nodes.h#L505), [Fusion](../../csrc/fusion.h#L134)
- [TensorView](../../csrc/ir/interface_nodes.h#L383), [IterDomain](../../csrc/ir/internal_base_nodes.h#L83), [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
