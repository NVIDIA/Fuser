# AliasAnalysisResult

Source: [AliasAnalysisResult](../../../csrc/alias_analysis.h#L31)

## Synopsis
- **Kind**: class
- **File**: `csrc/alias_analysis.h`
- **Approx. size**: ~37 lines

## Context (from code comments)
Holds aliases found in a fusion. The expected user flow is
```
AliasAnalysisResult analysis;
analysis.add(...);
...
analysis.add(...);
analysis.finalize(...);
// The user can now call const methods to retrieve information.
```

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
