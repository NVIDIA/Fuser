# tcgen05 Synchronization Guide

This document provides a detailed overview of tcgen05 instruction synchronization dependencies and fence requirements, based on [PTX ISA Section 9.7.16.6.2: tcgen05 Memory Consistency Model - Pipelined Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions).

## Table of Contents

- [1. Overview](#1-overview)
- [2. Completion Mechanisms](#2-completion-mechanisms)
  - [2.1. MBarrier Completion (tcgen05.commit)](#21-mbarrier-completion-tcgen05commit)
  - [2.2. Direct Wait Completion (tcgen05.wait)](#22-direct-wait-completion-tcgen05wait)
  - [2.3. tcgen05 Instruction Completion Reference](#23-tcgen05-instruction-completion-reference)
    - [2.3.1. Key Observations](#231-key-observations)
  - [2.4. Completion Pattern Examples](#24-completion-pattern-examples)
- [3. tcgen05 Instruction Categories](#3-tcgen05-instruction-categories)
  - [3.1. Control Instructions (Generic Proxy)](#31-control-instructions-generic-proxy)
  - [3.2. Data Movement Instructions (Generic Proxy)](#32-data-movement-instructions-generic-proxy)
  - [3.3. Compute Instructions (Async Proxy)](#33-compute-instructions-async-proxy)
  - [3.4. Commit Instructions (Generic Proxy)](#34-commit-instructions-generic-proxy)
  - [3.5. Fence Instructions (Generic Proxy)](#35-fence-instructions-generic-proxy)
- [4. Synchronization Dependencies](#4-synchronization-dependencies)
  - [4.1. Dependency Matrix](#41-dependency-matrix)
  - [4.2. Automatic Pipelining](#42-automatic-pipelining)
  - [4.3. Required Fences](#43-required-fences)
- [5. Code Motion Fences](#5-code-motion-fences)
  - [5.1. `tcgen05.fence::before_thread_sync`](#51-tcgen05fencebefore_thread_sync)
  - [5.2. `tcgen05.fence::after_thread_sync`](#52-tcgen05fenceafter_thread_sync)
  - [5.3. Why Code Motion Fences Are Necessary](#53-why-code-motion-fences-are-necessary)
- [6. Proxy Execution Context](#6-proxy-execution-context)
  - [6.1. Generic Proxy Operations](#61-generic-proxy-operations)
  - [6.2. Async Proxy Operations](#62-async-proxy-operations)
  - [6.3. Memory Ordering Requirements](#63-memory-ordering-requirements)
- [7. Practical Implementation](#7-practical-implementation)
- [8. Examples](#8-examples)
  - [8.1. Example 1: Simple tcgen05 Pipeline (No Fences Needed)](#81-example-1-simple-tcgen05-pipeline-no-fences-needed)
  - [8.2. Example 2: tcgen05 with Thread Synchronization (Fences Required)](#82-example-2-tcgen05-with-thread-synchronization-fences-required)
  - [8.3. Example 3: tcgen05 with Proxy Context Transitions](#83-example-3-tcgen05-with-proxy-context-transitions)
  - [8.4. Example 4: Complete tcgen05 Pattern with MBarrier](#84-example-4-complete-tcgen05-pattern-with-mbarrier)
- [9. Performance Considerations](#9-performance-considerations)
  - [9.1. Minimizing Fence Overhead](#91-minimizing-fence-overhead)
  - [9.2. Synchronization Overhead](#92-synchronization-overhead)
  - [9.3. Best Practices](#93-best-practices)
- [10. Questions and Ambiguous Behaviors](#10-questions-and-ambiguous-behaviors)
  - [10.1. tcgen05.wait Instruction Variants](#101-tcgen05wait-instruction-variants)
  - [10.2. Proxy Context for tcgen05.cp](#102-proxy-context-for-tcgen05cp)
  - [10.3. Automatic Pipelining Rules](#103-automatic-pipelining-rules)
  - [10.4. Memory Ordering Between Contexts](#104-memory-ordering-between-contexts)
  - [10.5. Code Motion Fence Placement](#105-code-motion-fence-placement)
- [11. References](#11-references)
  - [11.1. Primary Sources](#111-primary-sources)
  - [11.2. Additional Resources](#112-additional-resources)
  - [11.3. Performance Analysis Tools](#113-performance-analysis-tools)

## 1. Overview

The tcgen05 instruction family provides tensor memory operations for modern GPU architectures (Hopper+). Unlike other asynchronous operations, tcgen05 instructions have specific synchronization dependencies that determine when fences are required vs. when operations can be automatically pipelined.

For complete instruction details, see the [PTX ISA tcgen05 Instructions Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05).

**Key Concepts:**
- **Generic Proxy**: Traditional CUDA execution context for most operations
- **Async Proxy**: Specialized execution context for high-throughput async operations
- **Code Motion Fence**: Prevents instruction reordering (not memory ordering)
- **Memory Fence**: Ensures memory visibility between execution contexts

## 2. Completion Mechanisms

tcgen05 instructions use two distinct completion mechanisms, with each instruction having a fixed completion mechanism determined by its design. Understanding which instructions use which mechanism is crucial for proper synchronization.

### 2.1. MBarrier Completion (tcgen05.commit)

The [`tcgen05.commit`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-commit) instruction provides **batch-oriented completion** by signaling an mbarrier when all pending tcgen05 operations complete.

**Key Characteristics:**
- **Bulk synchronization** - Signal completion of multiple operations at once
- **MBarrier integration** - Works with existing mbarrier-based synchronization
- **Batch completion** - Single commit operation can complete multiple outstanding operations

**Usage Pattern:**
```cuda
// Start multiple tcgen05 operations
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [addr], size;
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_a], [smem_a];
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_b], [smem_b];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_a], [tmem_b], idesc, enable;
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_out], [tmem];

// Signal completion of all operations to mbarrier
tcgen05.commit.cta_group::1 [mbar];

// Later: wait for all operations to complete
mbarrier.arrive_wait_expect_tx.shared::cta.b64 new_token, [mbar], old_token, count;
```

**Required For:**
- `tcgen05.mma` operations (no other completion mechanism available)
- Multi-stage tensor pipelines requiring batch completion

### 2.2. Direct Wait Completion (tcgen05.wait)

The [`tcgen05.wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-wait) instruction provides **operation-specific completion** by directly waiting for individual tcgen05 operations.

**Key Characteristics:**
- **Fine-grained control** - Wait for specific operations individually
- **Direct completion** - No mbarrier setup required
- **Operation-specific** - Wait for individual `ld`, `st`, or `cp` operations

**Usage Pattern:**
```cuda
// Start specific operations
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_a], [smem_a];
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_b], [smem_b];

// Wait for specific loads to complete before MMA
tcgen05.wait::ld.cta_group::1 [tmem_a];
tcgen05.wait::ld.cta_group::1 [tmem_b];

// Now safe to use loaded data
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_a], [tmem_b], idesc, enable;

// Note: tcgen05.mma can only be completed via tcgen05.commit + mbarrier
// Direct wait on mma operations is not supported
tcgen05.commit.cta_group::1 [mbar];
mbarrier.wait.shared.b64 [mbar], expected_state;
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_out], [acc];
```

**Available For:**
- `tcgen05.ld`, `tcgen05.st` operations only (via `tcgen05.wait::ld`, `tcgen05.wait::st`)
- Fine-grained synchronization when individual operation completion is needed

### 2.3. tcgen05 Instruction Completion Reference

The following table shows which completion mechanism and execution context each tcgen05 instruction uses:

| Instruction | Execution Context | Completion Mechanism | Notes |
|-------------|------------------|---------------------|-------|
| [`tcgen05.alloc`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-alloc) | Generic Proxy | None (synchronous) | Allocation completes immediately |
| [`tcgen05.dealloc`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-dealloc) | Generic Proxy | None (synchronous) | Deallocation completes immediately |
| [`tcgen05.ld`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-ld) | Generic Proxy | `tcgen05.wait::ld` or `tcgen05.commit` | Load data from memory to tensor memory |
| [`tcgen05.st`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-st) | Generic Proxy | `tcgen05.wait::st` or `tcgen05.commit` | Store data from tensor memory to memory |
| [`tcgen05.cp`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-cp) | Async Proxy | `tcgen05.commit` only | Copy operations between memory spaces |
| [`tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-mma) | Async Proxy | `tcgen05.commit` only | Matrix multiply-accumulate operations |
| [`tcgen05.wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-wait) | Generic Proxy | N/A (completion instruction) | Waits for specific operation completion |
| [`tcgen05.commit`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-commit) | Generic Proxy | N/A (completion instruction) | Signals mbarrier when operations complete |
| [`tcgen05.fence`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-fence) | Generic Proxy | None (code motion only) | Prevents instruction reordering |

#### 2.3.1. Key Observations

- **Load/store operations** (`ld`, `st`) support **both** completion mechanisms (`tcgen05.wait::ld/st` or `tcgen05.commit`)
- **Async operations** (`cp`, `mma`) **only** support `tcgen05.commit` + mbarrier completion
- **Control operations** (`alloc`, `dealloc`, `fence`) complete synchronously or provide code motion barriers
- **All tcgen05 operations** execute in either Generic Proxy or Async Proxy context
- **Completion instructions** (`wait`, `commit`) always execute in Generic Proxy

### 2.4. Completion Pattern Examples

**MBarrier Completion Pattern (Required for `tcgen05.mma`):**
```cuda
// Batch completion - required for mma operations
for (int tile = 0; tile < num_tiles; ++tile) {
    tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem + tile_offset];
    tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, enable;
}
tcgen05.commit.cta_group::1 [mbar];  // Single bulk completion
mbarrier.arrive_wait_expect_tx.shared::cta.b64 token, [mbar], old_token, count;
```

**Direct Wait Completion Pattern (Available for load/store only):**
```cuda
// Individual waits - only available for ld/st operations
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem];
tcgen05.wait::ld.sync.aligned;  // Wait for this specific load

tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_out], [tmem];
tcgen05.wait::st.sync.aligned;  // Wait for this specific store
```

## 3. tcgen05 Instruction Categories

Based on [PTX ISA Section 9.7.16.6.2](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions), tcgen05 instructions fall into several categories with different synchronization requirements:

### 3.1. Control Instructions (Generic Proxy)
- [`tcgen05.alloc`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-alloc) - Allocate tensor memory
- [`tcgen05.dealloc`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-dealloc) - Deallocate tensor memory
- [`tcgen05.relinquish_alloc_permit`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-relinquish-alloc-permit) - Release allocation permit

### 3.2. Data Movement Instructions (Generic Proxy)
- [`tcgen05.ld`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-ld) - Load data to tensor memory
- [`tcgen05.st`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-st) - Store data from tensor memory
- [`tcgen05.wait::ld`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-wait) - Wait for load completion
- [`tcgen05.wait::st`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-wait) - Wait for store completion

### 3.3. Compute Instructions (Async Proxy)
- [`tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-mma) - Matrix multiply-accumulate operation
- [`tcgen05.cp`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-cp) - Copy operation between memory spaces

**Note**: Both `tcgen05.mma` and `tcgen05.cp` are confirmed to execute in the Async Proxy context per [PTX ISA Section 9.7.16.6.5](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-smem-access).

### 3.4. Commit Instructions (Generic Proxy)
- [`tcgen05.commit`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-commit) - Commit operations with mbarrier synchronization

### 3.5. Fence Instructions (Generic Proxy)
- [`tcgen05.fence::before_thread_sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-fence) - Code motion fence before synchronization
- [`tcgen05.fence::after_thread_sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-fence) - Code motion fence after synchronization

## 4. Synchronization Dependencies

### 4.1. Dependency Matrix

> **⚠️ INCOMPLETE**: This matrix contains assumptions that need verification based on PTX ISA documentation. See [question 10.3](#103-automatic-pipelining-rules) for details.

**📚 Verification Source**: [PTX ISA Section 9.7.16.6.2: tcgen05 Memory Consistency Model - Pipelined Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions)
Specifically refer to the **"Implicitly pipelined tcgen05 instructions"** subsection for authoritative pipelining rules.

The following matrix shows pipelining relationships between tcgen05 instructions. Each cell indicates whether the Producer → Consumer relationship is automatically pipelined or requires explicit synchronization.

**Legend:**
- ✅ **Pipelined**: Operations are automatically pipelined (no fence needed)
- ❌ **Manual Sync**: Requires explicit synchronization (fence or wait)
- ⚠️ **Conditional**: Pipelining depends on specific conditions
- ❓ **Unknown**: Needs verification from PTX ISA documentation

|            | **alloc** | **ld** | **st** | **cp** | **mma** | **wait::ld** | **wait::st** | **commit** | **dealloc** | **fence** |
|------------|-----------|--------|--------|--------|---------|--------------|--------------|------------|-------------|-----------|
| **alloc**  | ❓        | ✅     | ✅     | ✅     | ✅      | ❓           | ❓           | ❓         | ❓          | ❓        |
| **ld**     | ❌        | ❓     | ❌     | ✅     | ✅      | ✅           | ❌           | ⚠️         | ❌          | ❓        |
| **st**     | ❌        | ❌     | ❓     | ❌     | ❌      | ❌           | ✅           | ⚠️         | ✅          | ❓        |
| **cp**     | ❌        | ❌     | ✅     | ❓     | ✅      | ❌           | ❌           | ⚠️         | ❌          | ❓        |
| **mma**    | ❌        | ❌     | ✅     | ✅     | ❓      | ❌           | ❌           | ⚠️         | ❌          | ❓        |
| **wait::ld** | ❌      | ❓     | ❌     | ❌     | ❌      | ❓           | ❌           | ❌         | ❌          | ❓        |
| **wait::st** | ❌      | ❌     | ❓     | ❌     | ❌      | ❌           | ❓           | ❌         | ❌          | ❓        |
| **commit** | ❌        | ❌     | ❌     | ❌     | ❌      | ❌           | ❌           | ❓         | ❌          | ❓        |
| **dealloc** | ❌       | ❌     | ❌     | ❌     | ❌      | ❌           | ❌           | ❌         | ❓          | ❓        |
| **fence**  | ❓        | ❓     | ❓     | ❓     | ❓      | ❓           | ❓           | ❓         | ❓          | ❓        |

**How to Read the Matrix:**
- **Row** = Producer instruction (executes first)
- **Column** = Consumer instruction (executes second)
- **Cell Value** = Synchronization requirement for Producer → Consumer ordering

**Matrix Analysis:**

**✅ Confident Pipelined (likely correct):**
- `alloc` → `{ld, st, cp, mma}`: Allocation makes tensor memory available to operations
- `ld` → `{cp, mma}`: Load completion feeds directly into async proxy operations
- `cp` → `{st, mma}`: Copy completion feeds into store or MMA operations
- `mma` → `{st, cp}`: MMA results available for store or copy operations
- `{ld, st}` → `wait::{ld, st}`: Wait instructions designed to track specific operation completion
- `st` → `dealloc`: Store completion required before memory deallocation

**⚠️ Conditional Dependencies:**
- `{ld, st, cp, mma}` → `commit`: Depends on which operations use mbarrier completion mechanism

**❌ Manual Synchronization Required (likely correct):**
- Any operation → `alloc`: Cannot pipeline into memory allocation
- Operations across different memory resources
- Cross-proxy transitions without documented pipelining

**❓ Unknown/Verification Needed:**
- All `fence` relationships: Code motion fences have different semantics
- Self-dependencies (e.g., `ld` → `ld`): May depend on resource overlap
- `wait` instruction interactions: Unclear how wait operations interact with each other

**Critical Gaps:**
- Missing verification from [PTX ISA "Implicitly pipelined tcgen05 instructions"](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions)
- Resource-specific rules (same vs. different tensor memory)
- Proxy context transitions (Generic ↔ Async)

### 4.2. Automatic Pipelining

> **⚠️ INCOMPLETE**: The examples below may contain incorrect assumptions. Only the specific combinations listed in the [PTX ISA "Implicitly pipelined tcgen05 instructions" section](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions) are guaranteed to be pipelined.

**Important**: According to the PTX ISA documentation, "The asynchronous tcgen05 operations may execute and complete in a different order than they were issued." Only explicitly documented pipelined combinations should be assumed to work without fences.

The following instruction sequences are **claimed to be automatically pipelined** but **need verification against PTX ISA**:

```cuda
// ✅ No fence needed - automatic pipelining
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [addr], size;
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem];  // Auto-pipelined

// ✅ No fence needed - automatic pipelining
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem];
tcgen05.cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dest], [src], size, [mbar];  // Auto-pipelined

// ✅ No fence needed - automatic pipelining
tcgen05.cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dest], [src], size, [mbar];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_a], [tmem_b], idesc, enable;  // Auto-pipelined

// ✅ No fence needed - automatic pipelining
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_a], [tmem_b], idesc, enable;
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem], [tmem];  // Auto-pipelined
```

### 4.3. Required Fences

Fences are **required** when tcgen05 operations interact with thread synchronization:

```cuda
// ⚠️ Code motion fence needed before thread synchronization
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem];
tcgen05.fence::before_thread_sync;  // Required
__syncthreads();

// ⚠️ Code motion fence needed after thread synchronization
__syncthreads();
tcgen05.fence::after_thread_sync;  // Required
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem], [tmem];
```

## 5. Code Motion Fences

tcgen05 fence instructions are **code motion fences**, not memory fences. They prevent compiler optimizations from reordering tcgen05 instructions across synchronization boundaries, which is essential for preserving intended synchronization semantics during instruction-level parallelism (ILP) optimizations.

### 5.1. `tcgen05.fence::before_thread_sync`
- **Purpose**: Prevents reordering of tcgen05 instructions before thread synchronization
- **Usage**: Place before `__syncthreads()`, barriers, or other sync primitives
- **Effect**: Code motion only - no memory ordering guarantees

**Example of Invalid Reordering Without Fence:**
```cuda
// Original PTX code (intended behavior)
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_input];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;
__syncthreads();  // All threads must complete their MMA before proceeding

// ❌ INVALID: Compiler might reorder without fence (breaks synchronization)
__syncthreads();  // Moved up - threads sync before MMA completes!
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_input];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;

// ✅ CORRECT: With fence, reordering is prevented
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_input];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;
tcgen05.fence::before_thread_sync;  // Prevents tcgen05 from moving past sync
__syncthreads();
```

### 5.2. `tcgen05.fence::after_thread_sync`
- **Purpose**: Prevents reordering of tcgen05 instructions after thread synchronization
- **Usage**: Place after `__syncthreads()`, barriers, or other sync primitives
- **Effect**: Code motion only - no memory ordering guarantees

**Example of Invalid Reordering Without Fence:**
```cuda
// Original PTX code (intended behavior)
__syncthreads();  // Ensure all threads reach this point
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_shared];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;

// ❌ INVALID: Compiler might reorder without fence (breaks synchronization)
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_shared];  // Moved up!
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;
__syncthreads();  // Some threads might access uninitialized shared memory

// ✅ CORRECT: With fence, reordering is prevented
__syncthreads();
tcgen05.fence::after_thread_sync;  // Prevents tcgen05 from moving before sync
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_shared];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem_b], idesc, 1;
```

### 5.3. Why Code Motion Fences Are Necessary

The compilation pipeline includes several optimization phases:
1. **PTX to SASS compilation** transforms high-level PTX to machine code
2. **Instruction scheduling** reorders instructions for better ILP and latency hiding
3. **Register allocation** may further affect instruction ordering

Without `tcgen05.fence` instructions, the optimizer might:
- Move `tcgen05` operations across `__syncthreads()` boundaries
- Violate intended synchronization semantics
- Create race conditions or data hazards
- Break the assumption that all threads complete operations before synchronization

**Important**: tcgen05 fences do **NOT** provide memory ordering between generic and async execution contexts. Use [`fence.proxy.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence-proxy) for memory ordering.

## 6. Proxy Execution Context

tcgen05 operations span both execution contexts:

### 6.1. Generic Proxy Operations
- `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`
- `tcgen05.ld`, `tcgen05.st`, `tcgen05.wait::*`
- `tcgen05.commit`
- `tcgen05.fence::*`

### 6.2. Async Proxy Operations
- `tcgen05.mma` (matrix multiply-accumulate operations)
- `tcgen05.cp` (asynchronous copy operations)

Both operations are confirmed to execute in the Async Proxy context per [PTX ISA Section 9.7.16.6.5](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-smem-access).

### 6.3. Memory Ordering Requirements

**Memory visibility** between proxy contexts requires explicit fences when shared memory is accessed by both Generic and Async proxy operations. This is distinct from execution ordering - even if instructions execute in the correct order, memory visibility between proxy contexts is not guaranteed without explicit fences.

**tcgen05-Specific Requirements:**
- **No implicit proxy fences**: Unlike WgMMA operations, tcgen05 instructions do NOT include implicit `fence.proxy.async`
- **User responsibility**: Explicit `fence.proxy.async` must be inserted when transitioning between proxy contexts
- **Memory visibility**: Required when Generic Proxy writes shared memory consumed by tcgen05 async operations, or vice versa

```cuda
// Generic Proxy → Async Proxy (tcgen05)
st.shared.f32 [smem], value;                    // Generic proxy writes
fence.proxy.async.shared::cta;                  // REQUIRED: Memory visibility fence
tcgen05.mma.cta_group::1.kind::f16 [acc], ...;  // Async proxy reads

// Async Proxy (tcgen05) → Generic Proxy
tcgen05.mma.cta_group::1.kind::f16 [acc], ...;  // Async proxy writes
fence.proxy.async.shared::cta;                  // REQUIRED: Memory visibility fence
ld.shared.f32 result, [smem];                   // Generic proxy reads
```

**Contrast with WgMMA:**
```cuda
// WgMMA includes implicit proxy fence upon completion
wgmma.mma_async.sync.aligned.m16n8k32.f16.f16.f16.f16 [acc], [smem_a], [smem_b];
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;  // Implicit fence.proxy.async upon completion
ld.shared.f32 result, [smem];     // NO explicit fence needed
```

For complete details, see [SyncGuide.md](./SyncGuide.md#proxy-execution-contexts) and [PTX ISA Proxy Execution Context](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence-proxy).

## 7. Practical Implementation

nvFuser provides comprehensive support for tcgen05 operations with automatic synchronization management and multi-role warp specialization. For detailed implementation information, see:

- **[nvFuser Synchronization Guide](./SyncGuide.md#13-tcgen05-nodes)**: Complete coverage of tcgen05 synchronization patterns, code locations, and implementation details
- **[Multi-Role Warp Specialization Plan](./MultiRoleWarpSpecializationPlan.md)**: Advanced circular buffering strategies for Blackwell MMA operations with tcgen05

**Key nvFuser Features:**
- **Automatic sync insertion** for tcgen05 operations
- **MBarrier-based synchronization** between async and compute warps
- **Multi-role warp specialization** for complex pipeline patterns
- **Circular buffer management** with proper fence placement

## 8. Examples

### 8.1. Example 1: Simple tcgen05 Pipeline (No Fences Needed)

```cuda
// All operations automatically pipelined - no explicit fences required
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem_addr], 1024;
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_addr], [smem_a];
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_addr+512], [smem_b];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_addr], [tmem_addr+512], idesc, 1;
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_c], [acc];
tcgen05.dealloc.cta_group::1.sync.aligned.shared::cta [tmem_addr];
```

### 8.2. Example 2: tcgen05 with Thread Synchronization (Fences Required)

```cuda
// Phase 1: Load data
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem], 1024;
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_input];

// Code motion fence before thread sync
tcgen05.fence::before_thread_sync;
__syncthreads();  // All threads wait

// Code motion fence after thread sync
tcgen05.fence::after_thread_sync;

// Phase 2: Compute and store
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem], idesc, 1;
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_output], [acc];
tcgen05.dealloc.cta_group::1.sync.aligned.shared::cta [tmem];
```

### 8.3. Example 3: tcgen05 with Proxy Context Transitions

```cuda
// Generic proxy stores data that will be used by tcgen05.mma
st.shared.f32 [operand_buffer], computed_value;
fence.proxy.async.shared::cta;  // Memory ordering fence

// tcgen05 pipeline
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem], 1024;
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [operand_buffer];
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem], idesc, 1;
tcgen05.st.sync.aligned.32x32b.x1.b32 [result_buffer], [acc];

// tcgen05.mma results used by generic proxy
fence.proxy.async.shared::cta;  // REQUIRED: No implicit fence in tcgen05
ld.shared.f32 final_result, [result_buffer];
```

### 8.4. Example 4: Complete tcgen05 Pattern with MBarrier

```cuda
// 1. Allocate tensor memory
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem], 2048;

// 2. Load data to tensor memory
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem], [smem_a];
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem+1024], [smem_b];

// 3. Perform matrix multiply-accumulate
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem], [tmem+1024], idesc, 1;

// 4. Commit with mbarrier synchronization
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];

// 5. Wait for completion
mbarrier.wait.shared.b64 [mbar], expected_state;

// 6. Memory ordering fence for proxy transition (tcgen05 requires explicit fence)
fence.proxy.async.shared::cta;

// 7. Use results in generic proxy
add.f32 result, acc, bias_value;

// 8. Store final results
tcgen05.st.sync.aligned.32x32b.x1.b32 [smem_output], [result];

// 9. Clean up
tcgen05.dealloc.cta_group::1.sync.aligned.shared::cta [tmem];
```

## 9. Performance Considerations

### 9.1. Minimizing Fence Overhead

1. **Leverage Automatic Pipelining**: Most tcgen05 → tcgen05 operations don't need fences
2. **Minimize Thread Synchronization**: Reduce `__syncthreads()` calls to avoid code motion fences
3. **Batch Proxy Transitions**: Group operations to minimize `fence.proxy.async` usage
4. **Use MBarrier Efficiently**: Prefer mbarrier-based sync over traditional barriers

### 9.2. Synchronization Overhead

> **Note**: Always profile your specific use case for actual performance characteristics.

| Synchronization Type | Use Case | Reference |
|---------------------|----------|----------|
| Automatic pipelining | tcgen05 → tcgen05 operations | [PTX ISA 9.7.16.6.2](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions) |
| `tcgen05.fence::*` | Code motion around thread sync | [PTX ISA tcgen05.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05-fence) |
| `fence.proxy.async` | Proxy context transitions | [PTX ISA fence.proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence-proxy) |
| `__syncthreads()` | Full thread block synchronization | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) |

### 9.3. Best Practices

1. **Design for Auto-Pipelining**: Structure code to maximize automatic pipelining
2. **Minimize Sync Points**: Reduce thread synchronization requirements
3. **Use Appropriate Fence Scopes**: Choose minimal required fence scope
4. **Batch Operations**: Group related tcgen05 operations together
5. **Profile Critical Paths**: Measure actual fence overhead in your kernels

## 10. Questions and Ambiguous Behaviors

The following behaviors need clarification or verification against PTX ISA documentation:

### 10.1. tcgen05.wait Instruction Variants
**Question**: Does `tcgen05.wait::mma` actually exist, or can MMA operations only be completed via `tcgen05.commit`?
- **Current assumption**: Only `tcgen05.wait::ld`, `tcgen05.wait::st`, `tcgen05.wait::cp` are available
- **Needs verification**: PTX ISA documentation for complete list of wait variants

**Answer**: According to the [PTX ISA documentation for tcgen05.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait), the syntax specification shows:
```
tcgen05.wait_operation.sync.aligned;

.wait_operation = { .wait::ld, .wait::st }
```
This means **only** `tcgen05.wait::ld.sync.aligned` and `tcgen05.wait::st.sync.aligned` are supported. There is **no** `tcgen05.wait::mma` or `tcgen05.wait::cp` instruction variant.

### 10.2. Proxy Context for tcgen05.cp
**Question**: The document claims `tcgen05.cp` executes in Async Proxy, but this needs verification.
- **Current assumption**: Based on it being a copy operation similar to async bulk operations
- **Needs verification**: PTX ISA documentation for execution context

**Answer**: According to [Section 9.7.16.6.5 of the PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-smem-access), both `tcgen05.mma` and `tcgen05.cp` use shared memory accesses in the async proxy. This confirms that **`tcgen05.cp` executes in the Async Proxy context**, as documented in our instruction completion reference table.

### 10.3. Automatic Pipelining Rules
**Question**: Are the automatic pipelining rules in the dependency matrix complete and accurate?
- **Current assumption**: Based on PTX ISA section 9.7.16.6.2 description
- **Needs verification**: Complete testing or official documentation of all pipelining scenarios

**Answer**: The current dependency matrix is **incomplete and contains incorrect assumptions**. According to the [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions):

> "The asynchronous tcgen05 operations may execute and complete in a different order than they were issued."

**Key Correction**: **NOT all tcgen05 → tcgen05 operations are automatically pipelined**. Only the specific combinations listed in the PTX ISA documentation under "Implicitly pipelined tcgen05 instructions" are guaranteed to be pipelined. Other pairings may execute in a different order than issued and should not be assumed to be pipelined.

The dependency matrix in section 4.1 needs to be revised to only include the explicitly documented pipelined combinations from the PTX ISA, rather than assuming all tcgen05 operations are automatically pipelined.

### 10.4. Memory Ordering Between Contexts
**Question**: When exactly is `fence.proxy.async` required vs. optional for tcgen05 operations?
- **Current assumption**: Required when transitioning between Generic and Async proxy contexts
- **Needs verification**: Specific rules for tcgen05 operations vs. other async operations

**Answer**: Although execution ordering of instructions may be correct, **memory visibility** between proxy contexts requires explicit fences when one instruction accesses shared memory in the async proxy and another in the generic proxy. A `fence.proxy.async[.shared::cta]` is needed unless there's an implicit fence from a previously completed instruction.

**Key Distinctions:**
- **WgMMA operations**: Include implicit proxy fences upon completion (e.g., after `wgmma.commit` + `wgmma.wait`, the async and generic proxies are already synced) - see [SyncGuide.md](./SyncGuide.md#wgmma-operations)
- **tcgen05 operations**: Do **NOT** include implicit proxy fences, so `fence.proxy.async` use is up to the user when transitioning between proxy contexts

For complete details on proxy fence semantics, see [SyncGuide.md](./SyncGuide.md#proxy-execution-contexts) and [PTX ISA fence.proxy documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence-proxy).

### 10.5. Code Motion Fence Placement
**Question**: Are there specific rules about where `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` must be placed?
- **Current assumption**: Immediately before/after thread synchronization primitives
- **Needs verification**: Exact placement rules and whether multiple fences are needed

**Answer**: The `tcgen05.fence` instructions are designed to prevent instruction reordering during compilation optimizations. The compilation pipeline transforms PTX to SASS machine code and includes instruction-level parallelism (ILP) optimizations that can reorder instructions to hide latency. The `tcgen05.fence` instructions specifically limit instruction movement with respect to synchronization commands to preserve intended synchronization semantics. They must be placed to prevent `tcgen05` instructions from being reordered **across** synchronization boundaries.

## 11. References

### 11.1. Primary Sources
- [PTX ISA Section 9.7.16.6.2: tcgen05 Memory Consistency Model - Pipelined Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions)
- [PTX ISA: tcgen05 Instructions Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- [PTX ISA: Proxy Execution Context](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence-proxy)
- [PTX ISA: Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)

### 11.2. Additional Resources
- [CUDA Programming Guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
- [CUDA Programming Guide: Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
- [nvFuser SyncGuide.md](./SyncGuide.md#13-tcgen05-nodes)

### 11.3. Performance Analysis Tools
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) - For detailed kernel performance analysis
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - For system-wide performance profiling
