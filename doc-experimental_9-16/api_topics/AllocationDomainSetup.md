# AllocationDomainSetup

Source: [AllocationDomainSetup](../../../csrc/device_lower/pass/allocation.cpp#L90)

## Synopsis
- **Kind**: class (inherits from `kir::IrVisitor`)
- **File**: `csrc/device_lower/pass/allocation.cpp`
- **Approx. size**: ~842 lines

## Context (from code comments)
Preparing allocation info for indexing. Because of broadcasting,
just looking at the loop groups of a tensor may not be enough to
determine the allocation of the tensor. For example, this happens
when a tensor is broadcast and inlined, where the original
pre-broadcast tensor may not have corresponding domains. If that
missing domain is annotated with ParallelType::Unroll, which
affects all inner loops, just looking at the inlined tensor itself
would miss the unrolling. Since unrolling changes allocation
shapes, missing unroll can result in incorrect allocations.
TODO: Refactor this and the allocation lowering pass

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
