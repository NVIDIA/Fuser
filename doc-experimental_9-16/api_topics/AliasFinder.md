# AliasFinder

Source: [AliasFinder](../../../csrc/alias_analysis.cpp#L32)

## Synopsis
- **Kind**: class (inherits from [OptOutConstDispatch](../../csrc/dispatch.h#L204))
- **File**: `csrc/alias_analysis.cpp`
- **Approx. size**: ~24 lines

## Context (from code comments)
Finds aliases between `expr`'s inputs and outputs and stores the findings in
`analysis`.
The current implementation does the bare minimum to detect some aliasing
that the codegen can use to generate a kernel skipping unnecessary
computation.

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
