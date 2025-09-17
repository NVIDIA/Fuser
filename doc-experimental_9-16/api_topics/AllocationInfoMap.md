# AllocationInfoMap

Source: [AllocationInfoMap](../../../csrc/device_lower/pass/alias_memory.cpp#L624)

## Synopsis
- **Kind**: class (inherits from `kir::IrVisitor`)
- **File**: `csrc/device_lower/pass/alias_memory.cpp`
- **Approx. size**: ~430 lines

## Context (from code comments)
! Analysis pass to collect the liveness info of local and shared buffers:
! The liveness info is illustrated as follows:
!
! For Idx0 ...
!   Alloc(T1, register)
!   Alloc(T2, register)
!   Alloc(T3, register)
!
!   For Idx1 ...     <---------- Outer Live Interval of T1 begin
!     For Idx2 ...
!       T1 = ...            <--  Inner Live Interval of T1 begin
!       T2 = ...
!       T3 = T1 + ...    <-- Inner Live Interval of T1 end
!       T5 = T3 + ...
!     EndFor Idx2 ...
!   EndFor Idx1 ... <-------  Outer Live Interval of T1 end
!
!   Alloc(T4, register)
!   For Idx3 ...
!     T4 = ...
!
!  Each buffer is associated with an `inner_live_interval` and an
!  `outer_live_interval`. Inner interval marks the exprs that are the first
!  write and last read of the buffer. Outer interval marks the beginning of
!  the loop of first write and end of the loop of last read, at the same loop
!  level as the buffer allocation. Note that the end of a ForLoop is marked by
!  the last expression within it. In the case of an outer live interval, if
!  the end point is the end of a for loop, it is given a position at which
!  that expression would reside, but no actual [Expr](../../csrc/ir/base_nodes.h#L505) is associated with that
!  position.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
