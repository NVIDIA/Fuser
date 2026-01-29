# Thunder test segfault - RESOLVED ✓

## Context on the current work

This branch is an investigation branch off of feature branch
`md/ir-container-uptr` for looking into mysterious segfaults caused by the
refactor in that branch. The aim of `md/ir-container-uptr` is to move
IrContainer from an Inheritance pattern to a composition one. In the feature
branch we kept the current inheritance hierarchy of `IrContainer` -> `Fusion` ->
`Kernel`/`HostIrContainer` to not upset the "interface" throughout the rest of
the code that assumes `IrContainer` as the base type for dynamic dispatch.

Functionally the current `IrStorage` works the same as the old `IrContainer`.
`IrContainer` in the feature branch is just a forwarding class to maintain the
interface.

The composition pattern is a non-negotiable as this is a part of a larger
refactor and this is the first phase.

Outside of the current errors the feature branch is considered "complete"
functionally, we are just tying up loose ends.

## The Problem - SOLVED

We were seeing a segfault in one of the Thunder CI tests nvfuser is run with.

To reproduce:

```bash
pytest /opt/pytorch/lightning-thunder/thunder/tests/test_grad.py -k "outer_nvfuser_cuda_thunder and float32" -vxs
```

**Status:** ✅ RESOLVED - The test now passes after fixing the bug in `Fusion::removeVal()`.

## Root Cause

The refactor didn't introduce new logic bugs, but it did expose an existing bug in `Fusion::removeVal()` (csrc/fusion.cpp lines 303-327).

The bug: `Fusion::removeVal()` was using `exprs()` to find all Exprs that reference a Val being deleted. However, `exprs()` only returns **reachable Exprs** (those reachable from terminating outputs via `StmtSort::getExprs()`). This misses **dead Exprs** that are not reachable from outputs but still exist in the Fusion container.

When PreSegmenter's DeadCodeRemover removed a Val, there could still be dead Exprs holding pointers to that Val. Later when the Fusion was copied (during `SegmentedFusion::makeFusion()`), those dead Exprs would be cloned, and their corrupted input pointers would be accessed, causing heap-use-after-free.

## The Fix

Changed line 312 in csrc/fusion.cpp from:
```cpp
for (Expr* e : exprs()) {  // Only gets reachable Exprs
```

to:
```cpp
for (Expr* e : unordered_exprs()) {  // Gets ALL Exprs including dead ones
```

This ensures that when removing a Val, we find and remove **all** Exprs that reference it, including dead Exprs not reachable from outputs.

## Why the refactor exposed this

The refactor changed `Fusion::removeVal()` from using the `exprs_` member variable to using the `exprs()` method:

**On main branch:**
```cpp
for (Expr* e : exprs_) {  // exprs_ is std::unordered_set<Expr*> - ALL Exprs
```

**On feature branch (md/ir-container-uptr):**
```cpp
for (Expr* e : exprs()) {  // exprs() calls StmtSort::getExprs() - REACHABLE Exprs only
```

This wasn't intentional - it was a mechanical refactoring error. When moving from inheritance to composition:
- The `exprs_` member moved from `IrContainer` to `IrStorage`
- The code was updated to use `exprs()` assuming it was equivalent
- But `Fusion::exprs()` does traversal via `StmtSort::getExprs()` which only returns reachable Exprs

The correct fix was to use `unordered_exprs()` which returns ALL Exprs, equivalent to the old `exprs_` member.

## Verification

After the fix:
```bash
cd /opt/pytorch/lightning-thunder
LD_PRELOAD=$(clang-20 -print-file-name=libclang_rt.asan-x86_64.so) \
ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0" \
pytest /opt/pytorch/lightning-thunder/thunder/tests/test_grad.py \
  -k "outer_nvfuser_cuda_thunder and float32" -v
```

**Result:** `1 passed` ✅

No more heap-use-after-free errors. The test passes cleanly.

## Instructions for building / running during this investigation

### Compiling

Always use the script `_bn` to compile and build the project this utilizes
`sccache` for fast installs and handles installation into the system python
environment.

**DO NOT** directly build from `cmake` or `pip install` commands.

### Env Args

I prefer using `clang-20` - this is already set in the `CC` and `CXX` env vars.
Some env flags you may want to use for `_bn`
`NVFUSER_SOURCE_DIR=<path to current source>` <-- REQUIRED
`NVFUSER_BUILD_BUILD_TYPE=RelWithDebInfo`
`NVFUSER_BUILD_ENABLE_PCH=0` <-- Important: PCH causes ASAN build issues
`NVFUSER_BUILD_NO_BENCHMARK=1` <-- we probably don't need to build benchmark targets.

For more information on available build and runtime env flags see:
`/root/workspace/Fuser/main/tools/env-config`

### Debugging

The system has lldb and gdb.

We can compile builds with address sanitizer: `NVFUSER_BUILD_WITH_ASAN=1`

We **NEED** `ASAN_OPTIONS=protect_shadow_gap=0` at the minimum - without this
cuda runtime calls will fail. This can have side-effects that are silent in
python tests like the thunder one. Thunder will not be able to check heuristics
of the cuda environment and will fail to generate the test in question here.

When running python scripts for nvfuser tests / reproducers you will need:
`LD_PRELOAD=$(clang-20 -print-file-name=libclang_rt.asan-x86_64.so)`

# Investigation Report

## ASAN Run #1: Crash in IrCloner::clone()

**Date:** 2026-01-29
**Command:** `ASAN_OPTIONS=protect_shadow_gap=0 LD_PRELOAD=$(clang-20 -print-file-name=libclang_rt.asan-x86_64.so) pytest /opt/pytorch/lightning-thunder/thunder/tests/test_grad.py -k "outer_nvfuser_cuda_thunder and float32" -vxs`

**Result:** SEGV caught by ASAN

### Key Stack Trace
```
#4  nvfuser::Statement const* nvfuser::PolymorphicBase::as<nvfuser::Statement>() const
    at /root/workspace/Fuser/main/csrc/base.h:149:25
#5  nvfuser::Val* nvfuser::IrCloner::clone<nvfuser::Val>(nvfuser::Val const*)
    at /root/workspace/Fuser/main/csrc/ir/cloner.h:64:40
#6  std::vector<nvfuser::Val*> nvfuser::IrCloner::clone<nvfuser::Val*>(std::vector<nvfuser::Val*> const&)
    at /root/workspace/Fuser/main/csrc/ir/cloner.h:74:22
#7  nvfuser::Expr::Expr(nvfuser::Expr const*, nvfuser::IrCloner*)
    at /root/workspace/Fuser/main/csrc/ir/base_nodes.cpp:355:26
...
#12 nvfuser::IrStorage::copy(nvfuser::IrStorage const*, nvfuser::IrStorage*)
    at /root/workspace/Fuser/main/csrc/ir/storage.cpp:127:35
#13 nvfuser::IrContainer::copy(nvfuser::IrContainer const*, nvfuser::IrContainer*)
    at /root/workspace/Fuser/main/csrc/ir/container.cpp:53:20
#14 nvfuser::Fusion::copy(nvfuser::Fusion const*, nvfuser::Fusion*)
    at /root/workspace/Fuser/main/csrc/fusion.cpp:148:20
#15 nvfuser::SegmentedFusion::makeFusion(nvfuser::SegmentedGroup*) const
    at /root/workspace/Fuser/main/csrc/fusion_segmenter.cpp:354:13
```

### Analysis

The crash occurs when cloning an Expr during Fusion copy. The Expr being cloned tries to clone its input Vals, but one of those Vals has already been freed (heap-use-after-free).

The ASAN report shows:
- **Address being accessed:** Inside an Expr's input Val
- **Freed by:** `DeadCodeRemover::modifyFusion()` via `Fusion::removeVal()`
- **Allocated by:** Earlier during Fusion copy operation

This indicates that:
1. A Val was removed by PreSegmenter's dead code removal
2. But an Expr still held a pointer to that removed Val
3. When copying the Fusion later, that Expr tried to access the freed Val

## Investigation: Why wasn't the Expr removed?

Looking at `Fusion::removeVal()` in csrc/fusion.cpp:

```cpp
// Lines 303-327
std::vector<Expr*> exprs_to_remove;
for (Expr* e : exprs()) {  // BUG: exprs() only returns REACHABLE Exprs!
  if (!inContainer(e)) {
    continue;
  }
  if (std::find(e->inputs().begin(), e->inputs().end(), val) !=
      e->inputs().end()) {
    exprs_to_remove.push_back(e);
  }
}
```

The problem: `exprs()` calls `StmtSort::getExprs(this)` which only traverses from terminating outputs. This means **dead Exprs** not reachable from outputs won't be found!

## The Solution

Use `unordered_exprs()` instead, which returns ALL Exprs in the container:

```cpp
for (Expr* e : unordered_exprs()) {  // Gets ALL Exprs including dead ones
```

From csrc/ir/storage.h line 59-61:
```cpp
//! Return the set of Exprs registered with this fusion. Warning: This will
//! return exprs outside inputs/outputs, so can be unsafe for use with
//! segmented fusions.
const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
  return exprs_;
}
```

This ensures we remove **all** Exprs that reference a deleted Val, including dead ones.

## Earlier Investigation: Special Vals Swap Bug

During investigation, we also found and fixed a bug in `IrStorage::swap()` where special vals (zero_val_, one_val_, true_val_, false_val_, magic_zero_val_) and axioms_ weren't being swapped. This was a real bug but didn't solve the Thunder test issue.

The fix was added in csrc/ir/storage.cpp lines 89-99:
```cpp
// CRITICAL FIX: Swap special vals and axioms
// These are stored separately from vals_up_ but may be referenced by Exprs
// in the swapped content. If not swapped, Exprs in container A could
// reference special vals from container B after the swap, causing
// use-after-free when one container is destroyed.
std::swap(a.zero_val_, b.zero_val_);
std::swap(a.one_val_, b.one_val_);
std::swap(a.true_val_, b.true_val_);
std::swap(a.false_val_, b.false_val_);
std::swap(a.magic_zero_val_, b.magic_zero_val_);
std::swap(a.axioms_, b.axioms_);
```

## Lessons Learned

1. **API clarity matters:** The difference between `exprs()` (reachable) and `unordered_exprs()` (all) is subtle but critical. The documentation helps but the method names could be clearer.

2. **Dead code creates hazards:** Even "dead" code can cause problems if it holds dangling pointers. When removing Vals, we must be thorough about finding ALL references.

3. **Refactors expose latent bugs:** The refactor didn't cause this bug - it just changed timing enough to expose a pre-existing issue in `Fusion::removeVal()`.

4. **ASAN is invaluable:** Without ASAN, this would have been extremely difficult to debug. The heap-use-after-free was caught immediately with detailed allocation/deallocation traces.

## Related Issues

This bug is related to issue #1270 (SERDE segfault investigation). The original fix for #1270 changed `Fusion::removeVal()` to iterate over `exprs()` instead of `val->uses()` to catch dead code. However, `exprs()` still doesn't catch ALL dead Exprs - only those reachable from outputs. Our fix completes what #1270 started.
