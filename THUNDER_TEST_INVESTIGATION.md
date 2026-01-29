# Thunder test segfualt

## Context on the current work

This branch is an investigation branch off of feature branch
`md/ir-container-uptr` for looking into mysterious segfualts caused by the
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

Outside of the current errors the eature branch is considered "complete"
functionally, we are just tying up loose ends.

## The Problem

We are seeing a segfault in one of the Thunder CI test nvfuser is run with, this
could be similar to the error in the SERDE_SEGFAULT_INVESTIGATION. That issue
has been resolved as that specific failing test has been removed for unrelated
reasons to this IrContainer work (note we did not find the error in this branch
related to that failure).

To reproduce:

```bash
pytest /opt/pytorch/lightning-thunder/thunder/tests/test_grad.py -k "outer_nvfuser_cuda_thunder and float32" -vxs
```

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
`NVFUSER_BUILD_BUILD_TYPE=RelWithDebInfo` `NVFUSER_BUILD_ENABLE_PCH=1` <-- for
faster build w/ clang clean builds e2e ~4-5 minutes.
`NVFUSER_BUILD_NO_BENCHMARK=1` <-- we probably dont need to build benchmark
targets.

For more information on available build and runtime env flags see:
`/root/workspace/Fuser/main/tools/env-config`

### Debugging

The system has lldb and gdb.

We can compile builds with address sanitizer: `NVFUSER_BUILD_WITH_ASAN=1`

We **NEED** `ASAN_OPTIONS=protect_shadow_gap=0` at the minimum - without this
cuda runtime calls will fail. This can have side-effects that are silent in
python tests like the thunder one. Thunder will not be able to check heurisitics
of the cuda environment and will fail to generate the test in question here.

Whan running python scripts for nvfuser tests / reproducers you will need:
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
    at /root/workspace/Fuser/main/csrc/fusion_segmenter.cpp:1858:7
#16 nvfuser::SegmentedGroup::makeClonedFusion()
    at /root/workspace/Fuser/main/csrc/fusion_segmenter.cpp:150:59
```

### ASAN Error Details
```
==334587==ERROR: AddressSanitizer: SEGV on unknown address 0x000000051afb (pc 0x7ca925ab1b2c ...)
==334587==The signal is caused by a READ memory access.
```

Address `0x51afb` is a very low address, suggesting a null or near-null pointer dereference.

### Analysis

1. **Crash Location**: The crash occurs in `PolymorphicBase::as<Statement>()` at the `dynamic_cast` operation (base.h:149).

2. **Specific Line**:
   ```cpp
   auto downcast_ptr = dynamic_cast<const T*>(this);  // Line 149
   NVF_ERROR(downcast_ptr != nullptr);                // Line 150
   ```

3. **Root Cause**: The `this` pointer passed to `dynamic_cast` is corrupted. The pointer value is `0x51afb`, which is clearly invalid.

4. **Context**: This happens during `Fusion::copy()` → `IrStorage::copy()` → cloning vals from `deterministic_vals()`.

5. **Code Flow in IrStorage::copy()** (csrc/ir/storage.cpp:118-120):
   ```cpp
   for (auto val : from->deterministic_vals()) {
     if (from->vals().count(val) > 0) {
       to->vals_.insert(ir_cloner.clone(val));  // Crash happens here
     }
   }
   ```

6. **deterministic_vals() Implementation** (csrc/ir/storage.cpp:19-27):
   ```cpp
   const std::deque<Val*> IrStorage::deterministic_vals() const noexcept {
     std::deque<Val*> vals_deque;
     std::transform(
         vals_up_.begin(),
         vals_up_.end(),
         std::back_inserter(vals_deque),
         [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
     return vals_deque;
   }
   ```

### Key Observations

1. **Corrupted Val Pointer**: The `val` pointer in the loop is already pointing to invalid memory (0x51afb) BEFORE we try to clone it.

2. **Heap Corruption**: This low address suggests either:
   - A use-after-free where the Val was deleted but the pointer remains in `vals_up_`
   - Heap corruption from elsewhere in the code that corrupted the `vals_up_` deque
   - A dangling pointer stored in `vals_up_`

3. **No Swap Involved**: Unlike the SERDE issue, this crash happens during a straightforward copy operation, NOT during a swap. The swap code is not in the call stack.

4. **Fusion Cache Context**: The crash happens in `SegmentedGroup::makeClonedFusion()`, which is creating a copy of a cached Fusion object. The cached Fusion may have been corrupted over time.

### Hypotheses

1. **Hypothesis A: Statement Pointer Corruption**
   - During earlier operations, a Statement's pointer was corrupted
   - The corrupted pointer ended up in `vals_up_` deque
   - When `deterministic_vals()` extracts pointers via `.get()`, we get the corrupted pointer
   - This would be a "delayed effect" bug where corruption happens much earlier

2. **Hypothesis B: Deque Corruption**
   - The `vals_up_` deque itself was corrupted (heap overflow, buffer overrun)
   - When iterating, we read corrupted memory that looks like a pointer
   - ASAN should have caught this earlier if it were a simple buffer overflow

3. **Hypothesis C: Use-After-Free**
   - A Val object was deleted (unique_ptr released)
   - But the unique_ptr still remains in `vals_up_` with a dangling pointer
   - This could happen if there's manual memory management mixing with unique_ptr

4. **Hypothesis D: Cache Lifetime Issue**
   - The Fusion being copied was stored in `FusionExecutorCache`
   - Over multiple Thunder operations, the cached Fusion's internal state became corrupted
   - The corruption accumulated until the copy operation exposed it

### Next Steps

1. **Add Validation Before Cloning**: Check pointer validity before attempting to clone
2. **Inspect FusionExecutorCache**: Look at how Fusion objects are stored and retrieved
3. **Add Memory Guards**: Use ASAN poison/unpoison to detect use-after-free
4. **Trace Val Creation/Destruction**: Add logging to track when Val objects are created and destroyed

### Code Locations to Investigate

- `csrc/ir/storage.cpp:118-122` - Where the crash occurs (Val cloning)
- `csrc/ir/storage.cpp:175` - Where crash occurs with validation (Expr cloning)
- `csrc/runtime/fusion_executor_cache.cpp` - Fusion caching mechanism
- `csrc/fusion_segmenter.cpp:150` - Where makeClonedFusion is called
- `csrc/ir/base_nodes.cpp` - Val/Statement construction and destruction

## ASAN Run #2: With Validation Code

**Date:** 2026-01-29
**Changes:** Added pointer validation before cloning Vals and Exprs

**Result:** Still crashed, but now at line 175 (Expr cloning) instead of Val cloning

### Key Observations

1. **Crash Location Changed**: With validation added, the crash moved from Val cloning to Expr cloning (storage.cpp:175).

2. **Validation Didn't Catch It**: The pointer validation checks for NULL and low addresses (`< 0x10000`) didn't trigger, meaning the corrupted pointer looks valid enough to pass those checks.

3. **Address Changed**: New crash address is `0x000000655e0` vs previous `0x51afb`. Both are low addresses but different.

4. **Crash in Expr's Inputs**: Looking at the stack trace, the crash is during `Expr::Expr(const Expr*, IrCloner*)` at line 355 of base_nodes.cpp, which clones the input Vals of the Expr. The Expr itself may be valid, but one of its input Val pointers is corrupted.

### Updated Hypothesis

The corruption is not in the `vals_up_` deque directly, but in the **inputs stored in Expr objects**. When we clone an Expr, we clone its inputs (which are Val pointers). One of these input Val pointers is corrupted.

This suggests:
- The Expr object itself is valid
- But Expr's `inputs_` vector contains a corrupted Val pointer
- This could happen if a Val was deleted but an Expr still references it
- Or if an Expr's inputs_ vector was corrupted by a buffer overflow elsewhere

### Next Investigation Step

Need to add validation in `Expr::Expr(const Expr*, IrCloner*)` copy constructor to check each input Val pointer before attempting to clone it.

## ASAN Run #3: **ROOT CAUSE FOUND** - Heap-Use-After-Free

**Date:** 2026-01-29
**Changes:** Added validation in Expr copy constructor to check input pointers

**Result:** ASAN caught heap-use-after-free!

### ASAN Output
```
==417453==ERROR: AddressSanitizer: heap-use-after-free on address 0x72ea55c29250 at pc 0x71a729f1b4e2
READ of size 8 at 0x72ea55c29250 thread T0
    #0 nvfuser::Statement::fusion() const /root/workspace/Fuser/main/csrc/ir/base_nodes.cpp:78:3
    #1 nvfuser::Expr::Expr(nvfuser::Expr const*, nvfuser::IrCloner*) /root/workspace/Fuser/main/csrc/ir/base_nodes.cpp:381:31
```

### Where Object Was Freed
```
freed by thread T0 here:
    #1 std::default_delete<nvfuser::Val>::operator()(nvfuser::Val*) const
    #6 std::unique_ptr<nvfuser::Val, std::default_delete<nvfuser::Val>>* std::__copy_move_backward<...>
    #10 std::_Deque_iterator<...> std::__copy_move_backward_dit<...>
```

The Val was deleted during a **deque reordering operation** (copy_move_backward).

### Where Object Was Allocated
```
previously allocated by thread T0 here:
    #1 nvfuser::TensorView* nvfuser::IrBuilder::clone<nvfuser::TensorView>(...)
    #4 nvfuser::IrStorage::copy(nvfuser::IrStorage const*, nvfuser::IrStorage*) at storage.cpp:144
    #7 nvfuser::Fusion::Fusion(nvfuser::Fusion const&) at fusion.cpp:212 [Copy Constructor]
    #9 nvfuser::FusionExecutorCache::getKernelRuntimeFor(...) at fusion_executor_cache.cpp:664
```

The Val (TensorView) was allocated during a **previous Fusion copy operation**.

### Analysis - The Root Cause

1. **First Fusion Copy**: A Fusion is copied in `FusionExecutorCache::getKernelRuntimeFor()` at line 664. During this copy, TensorViews are cloned and added to `vals_up_` deque.

2. **Deque Reallocation**: At some point, the `vals_up_` deque needs to grow or reorganize. During `std::copy_move_backward`, elements are moved within the deque. **This causes Val objects to be deleted and reallocated**.

3. **Dangling Pointers in Expr**: Meanwhile, Expr objects store **raw pointers** to Val objects in their `inputs_` vector. When a Val is moved/deleted during deque reorganization, these raw pointers become **dangling**.

4. **Second Fusion Copy**: When we try to copy the Fusion again (in `SegmentedGroup::makeClonedFusion()`), we iterate over Exprs and try to access their input Vals. The Val pointer is now dangling → **use-after-free**.

### The Fundamental Bug

**Mixing ownership models:**
- `IrStorage::vals_up_` uses `unique_ptr` for **ownership**
- `Expr::inputs_` uses **raw pointers**
- When `vals_up_` reorganizes (via deque operations), objects move in memory
- But `Expr::inputs_` still points to old addresses

### Why This Happens With IrContainer Refactor

The refactor changed how IrStorage is managed:
- **Before**: IrContainer owned vals/exprs directly via inheritance
- **After**: IrContainer owns `unique_ptr<IrStorage>` which owns vals/exprs

This likely changed the **timing** of when deque reallocations happen, making the bug visible.

### Solution Direction

The issue is NOT with `ir_container_` pointers or swap operations. It's with **deque reallocations** invalidating raw pointers.

**Possible fixes:**
1. Use `std::deque<Val*>` with manual memory management (not ideal)
2. Use stable containers like `std::list<unique_ptr<Val>>` (no reallocations)
3. Use indirection: `std::deque<unique_ptr<unique_ptr<Val>>>` (double indirection for stability)
4. Store indices instead of pointers in Expr::inputs_
5. Redesign to avoid storing pointers that can be invalidated

**Most likely solution**: Change `vals_up_` and `exprs_up_` from `std::deque` to `std::list` to guarantee pointer stability.

## ASAN Run #4: **ACTUAL ROOT CAUSE FOUND** - Cross-Fusion References

**Date:** 2026-01-29
**Changes:** Changed `std::deque` to `std::list` for pointer stability

**Result:** Still heap-use-after-free, but NOW we see the REAL problem!

### The REAL Root Cause

The issue is NOT about deque reallocation. It's about **cross-Fusion references**!

### Timeline of Events

1. **Fusion A is copied** (FusionExecutorCache line 664):
   - Creates Fusion B as a copy
   - Fusion B's Exprs contain pointers to Vals in Fusion B

2. **Fusion A runs dead code removal** (FusionKernelRuntime line 81):
   - `DeadCodeRemover::modifyFusion()` removes unused Vals
   - Calls `IrStorage::removeVal()` which **deletes** the Val object
   - The Val's unique_ptr in `vals_up_` list is erased

3. **Later, Fusion B is copied again** (SegmentedGroup::makeClonedFusion):
   - Tries to clone Fusion B's Exprs
   - Exprs have input pointers to Vals
   - **BUT**: Some of these Vals were actually from Fusion A and were deleted!
   - Result: use-after-free

### ASAN Evidence

**Freed by**:
```
#7 nvfuser::IrStorage::removeVal(nvfuser::Val*) at storage.cpp:242
#10 nvfuser::DeadCodeRemover::modifyFusion() at iter_visitor.cpp:1240
#15 nvfuser::FusionKernelRuntime::FusionKernelRuntime() at fusion_kernel_runtime.cpp:81
```

**Previously allocated**:
```
#4 nvfuser::IrStorage::copy() at storage.cpp:146
#6 nvfuser::Fusion::copy() at fusion.cpp:148
#7 nvfuser::Fusion::Fusion(const Fusion&) at fusion.cpp:212 [Copy Constructor]
#9 nvfuser::FusionExecutorCache::getKernelRuntimeFor() at fusion_executor_cache.cpp:664
```

### The Bug in IrCloner

When cloning a Fusion, `IrCloner` is supposed to create a complete, independent copy. However, there must be a case where **Val pointers from the source Fusion leak into the cloned Fusion's Exprs**.

This could happen if:
1. The `IrCloner::clones_map_` doesn't properly map ALL Vals
2. Some Vals are shared between Fusions (like special vals: zero_val_, one_val_, etc.)
3. The cloning process reuses some Vals from the source instead of cloning them

### Next Steps

1. Check if special vals (zero_val, one_val, etc.) are being properly cloned or reused
2. Investigate `IrCloner::clone()` to ensure all Vals are mapped
3. Check if there's any Val sharing between Fusions that shouldn't happen
4. Add validation in Fusion::copy() to ensure no cross-Fusion references exist after copying
