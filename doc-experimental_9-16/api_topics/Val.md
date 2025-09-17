# Val

Source: [Val](../../csrc/ir/base_nodes.h#L224)

## Synopsis
- **Kind**: class (inherits from [Statement](../../csrc/ir/base_nodes.h#L96))
- **File**: `csrc/ir/base_nodes.h`
- **What it represents**: Base class for IR values (scalars, tensors, ids) in nvFuser. Many IR node types derive from [Val](../../csrc/ir/base_nodes.h#L224).

## Purpose
- Provides common interfaces for values (type info, printing, cloning) and participates in the [Statement](../../csrc/ir/base_nodes.h#L96)/[Expr](../../csrc/ir/base_nodes.h#L505) graph maintained by [Fusion](../../csrc/fusion.h#L134).

## See also
- [Expr](../../csrc/ir/base_nodes.h#L505), [Fusion](../../csrc/fusion.h#L134)
- [TensorView](../../csrc/ir/interface_nodes.h#L383), [IterDomain](../../csrc/ir/internal_base_nodes.h#L83), [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
