# AllocationDomainPass

Source: [AllocationDomainPass](../../../csrc/preseg_passes/allocation_order_inference.h#L20)

## Synopsis
- **Kind**: class (inherits from `OptimizationPass<AllocationDomainPass>`)
- **File**: `csrc/preseg_passes/allocation_order_inference.h`
- **Approx. size**: ~8 lines

## Context (from code comments)
Realize allocation order propagation on fusion inputs to optimize allocation
domain of output tensor. This optimization pass currently only applies to
fusion outputs, but not intermediate tensors.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
