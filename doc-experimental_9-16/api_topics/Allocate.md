# Allocate

Source: [Allocate](../../../csrc/kernel_ir.h#L286)

## Synopsis
- **Kind**: class (inherits from [Expr](../../csrc/ir/base_nodes.h#L505))
- **File**: `csrc/kernel_ir.h`
- **Approx. size**: ~194 lines

## Context (from code comments)
! Allocate is a lower level Node that describes a buffer of memory that
! is required as an intermediate within a kernel. The extent is the expression
! of the size of the buffer that is generated from the TensorView that
! describes the output of an operation.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
