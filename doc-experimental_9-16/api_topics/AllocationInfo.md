# AllocationInfo

Source: [AllocationInfo](../../../csrc/device_lower/pass/alias_memory.cpp#L558)

## Synopsis
- **Kind**: struct
- **File**: `csrc/device_lower/pass/alias_memory.cpp`
- **Approx. size**: ~32 lines

## Context (from code comments)
! Utility class to record the read and write of each
! allocated buffer.
!
! Note:
!  this simplified interval analysis only works on pointwise ops and
!  reductions and broadcast. With no non-trivial IfThenElse and no
!  non-trivial re-computation.
!
!  Will probably at some point need dataflow and index analysis to precisely
!  handle loop carried dependency.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
