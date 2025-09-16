# Expr

Source: [Expr](../../csrc/ir/base_nodes.h#L505)

## Synopsis
- **Kind**: class (inherits from `Statement`)
- **File**: `csrc/ir/base_nodes.h`
- **What it represents**: Base class for IR expressions (operations) connecting input/output `Val` nodes.

## Purpose
- Forms the computational graph edges in `Fusion`. Concrete ops (e.g., unary/binary/ternary ops, reductions, broadcasts) derive from `Expr`.

## See also
- [Val](../../csrc/ir/base_nodes.h#L224), [Fusion](../../csrc/fusion.h#L134)
