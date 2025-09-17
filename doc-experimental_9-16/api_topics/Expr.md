# Expr

Source: [Expr](../../csrc/ir/base_nodes.h#L505)

## Synopsis
- **Kind**: class (inherits from [Statement](../../csrc/ir/base_nodes.h#L96))
- **File**: `csrc/ir/base_nodes.h`
- **What it represents**: Base class for IR expressions (operations) connecting input/output [Val](../../csrc/ir/base_nodes.h#L224) nodes.

## Purpose
- Forms the computational graph edges in [Fusion](../../csrc/fusion.h#L134). Concrete ops (e.g., unary/binary/ternary ops, reductions, broadcasts) derive from [Expr](../../csrc/ir/base_nodes.h#L505).

## See also
- [Val](../../csrc/ir/base_nodes.h#L224), [Fusion](../../csrc/fusion.h#L134)
