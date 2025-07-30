# nvFuser Synchronization Guide

This document provides a comprehensive overview of synchronization operations in nvFuser, including their high-level placement strategy and detailed code locations.

## Table of Contents

- [High-Level Summary](#high-level-summary)
- [Detailed Sync Types and Code Locations](#detailed-sync-types-and-code-locations)
  - [Basic Synchronization](#basic-synchronization)
    - [1. Block Syncs](#1-block-syncs)
    - [2. Grid Syncs](#2-grid-syncs)
    - [3. Warp Syncs](#3-warp-syncs)
    - [4. __threadfence Operations](#4-__threadfence-operations)
    - [5. __threadfence_block Operations](#5-__threadfence_block-operations)
    - [6. __nanosleep Operations](#6-__nanosleep-operations)
  - [Proxy Execution Contexts](#proxy-execution-contexts)
    - [7. Generic and Async Proxies](#7-generic-and-async-proxies)
    - [8. Async Proxy Fences (FenceAsyncProxy)](#8-async-proxy-fences-fenceasyncproxy)
  - [Reductions and Implicit Syncs](#reductions-and-implicit-syncs)
    - [9. Implicit Syncs from Reductions](#9-implicit-syncs-from-reductions)
  - [WgMMA Operations](#wgmma-operations)
    - [10. WgMMA Operations](#10-wgmma-operations)
    - [11. WgMMA Commits and Waits](#11-wgmma-commits-and-waits)
    - [12. WgMMA Fences](#12-wgmma-fences)
  - [Tensor Memory (TMem) Operations](#tensor-memory-tmem-operations)
    - [13. tcgen05 Nodes](#13-tcgen05-nodes)
  - [MBarrier Operations](#mbarrier-operations)
    - [14. MBarrier Operations](#14-mbarrier-operations)
  - [Advanced Synchronization](#advanced-synchronization)
    - [15. Block Serialization Syncs](#15-block-serialization-syncs)
    - [16. Semaphore Operations](#16-semaphore-operations)
    - [17. Acquire/Release Memory Operations](#17-acquirerelease-memory-operations)
  - [Host and Stream Synchronization](#host-and-stream-synchronization)
    - [18. Host-Level Synchronization](#18-host-level-synchronization)
    - [19. Stream-Based Synchronization](#19-stream-based-synchronization)
- [Sync Insertion Strategy](#sync-insertion-strategy)
- [Runtime Sync Implementations](#runtime-sync-implementations)
- [Implicit Synchronization Summary](#implicit-synchronization-summary)
- [Environment Variables](#environment-variables)

## High-Level Summary

nvFuser places synchronization operations strategically to ensure correct execution of parallel kernels. The primary sync placement strategy involves:

1. **Block-level synchronization** - Ensures all threads in a block reach the same point before proceeding
2. **Grid-level synchronization** - Coordinates across multiple thread blocks for reductions and broadcasts
3. **Warp-level synchronization** - Manages warp-specific operations and async operations
4. **Memory fence operations** - Ensures memory operations are visible across different execution contexts
5. **Async operation synchronization** - Manages asynchronous operations like TMA, WgMMA, and cp.async

The sync insertion is primarily handled in `csrc/device_lower/pass/insert_syncs.cpp`, with runtime implementations in the `runtime/` directory.

## Detailed Sync Types and Code Locations

### Basic Synchronization

#### 1. Block Syncs

Block synchronization ensures all threads in a thread block reach the same execution point.

**Code Locations:**
- **IR Node Definition**: [`csrc/kernel_ir.h:509`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L509) - `BlockSync` class
- **Code Generation**: [`csrc/codegen.cpp:3975`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L3975) - `handle(const kir::BlockSync* sync)`
- **Runtime Implementation**: 
  - [`runtime/block_sync_default.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_default.cu) - Default implementation using `__syncthreads()`
  - [`runtime/block_sync_atomic.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_atomic.cu) - Atomic-based implementation for debugging
- **Sync Insertion**: [`csrc/device_lower/pass/insert_syncs.cpp:229`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L229) - `handle(kir::BlockSync*)`

**Key Features:**
- Uses `__syncthreads()` for aligned cases - [CUDA Programming Guide: __syncthreads()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
- Uses `__barrier_sync(0)` for unaligned cases - [PTX ISA: bar.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
- **Implicit Memory Fence**: `__syncthreads()` includes an implicit `__threadfence_block()` - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- Supports custom atomic-based sync for debugging via `USE_BLOCK_SYNC_ATOMIC` environment variable
- Handles warp-specialized kernels with custom block dimensions

#### 2. Grid Syncs

Grid synchronization coordinates across multiple thread blocks using semaphore-based mechanisms.

**Code Locations:**
- **IR Node Definition**: [`csrc/kernel_ir.h:549`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L549) - `GridSync` class
- **Code Generation**: [`csrc/codegen.cpp:4000`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L4000) - `handle(const kir::GridSync* sync)`
- **Runtime Implementation**: [`runtime/grid_sync.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu) - Complete grid sync implementation
- **Sync Insertion**: [`csrc/device_lower/pass/insert_syncs.cpp:233`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L233) - `handle(kir::GridSync*)`

**Key Features:**
- Uses semaphore-based synchronization across blocks
- Supports persistent and non-persistent modes
- Handles multi-dimensional grid synchronization (X/Y/Z_BLOCK parameters)
- Includes `__threadfence()` for global memory visibility - [CUDA Programming Guide: __threadfence()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- **Implicit Block Sync**: Grid sync operations include implicit `block_sync::sync()` calls
- Provides block serialization utilities (`blockSerializeWait`, `blockSerializeRelease`)

#### 3. Warp Syncs

nvFuser does **not explicitly implement warp-level synchronization** using `__syncwarp()`. Instead, warp synchronization is handled implicitly through:

**Implicit Warp Syncs:**
- **Warp Reductions**: `runtime/warp.cu` - Uses warp shuffle operations (`shfl_xor`) - [CUDA Programming Guide: Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- **Warp-level Operations**: Handled through block syncs that naturally synchronize warps
- **Async Operations**: WgMMA and other async operations have their own sync mechanisms

**Code Locations:**
- **Warp Reduction**: [`runtime/warp.cu:40`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/warp.cu#L40) - `warpReduce` function
- **Warp Shuffle**: Uses CUDA's built-in `shfl_xor` operations - [PTX ISA: shfl](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-shfl)

### Proxy Execution Contexts

#### 7. Generic and Async Proxies

Modern GPU architectures (Hopper+) support two distinct execution contexts: **Generic Proxy** and **Async Proxy**. These contexts have separate memory spaces and execution pipelines, requiring explicit synchronization between them.

**Generic Proxy**: The traditional CUDA execution context that handles most compute operations, memory accesses, and synchronization primitives. - [CUDA Programming Guide: Memory Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-model)

**Async Proxy**: A specialized execution context designed for high-throughput asynchronous operations like Tensor Memory Access (TMA), Warp Group Matrix Multiply Accumulate (WgMMA), and tensor memory operations. - [CUDA Programming Guide: Asynchronous Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-operations)

**Proxy Operations Table:**

| Operation Type | Execution Context | Implicit Fence | Documentation |
|---|---|---|---|
| **TMA Load** | Async Proxy | None | [CUDA Programming Guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator) |
| **TMA Store** | Async Proxy | None | [CUDA Programming Guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator) |
| **WgMMA mma_async** | Async Proxy | **[Generic → Async upon completion](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)** | [PTX ISA: wgmma.mma_async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma) |
| **tcgen05 utcmma** | Async Proxy | **None** - uses mbarrier-based sync, no implicit proxy fence | [PTX ISA: tcgen05.utcmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05) |
| **cp.async** | Async Proxy | None | [PTX ISA: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async) |
| **Regular Compute** | Generic Proxy | None | [CUDA Programming Guide: Memory Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-model) |
| **Block/Grid Syncs** | Generic Proxy | None | [CUDA Programming Guide: Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) |

**Important**: Most async operations do not have implicit proxy fences. **WgMMA mma_async operations include implicit generic-async proxy fences upon completion** as documented in the PTX ISA. Other async operations may have similar behavior but require verification from the PTX documentation. Explicit `fence.proxy.async` instructions must be inserted manually for operations without documented implicit fences to ensure visibility between generic and async execution contexts.

**Fence Requirements Examples:**

**Example 1: WgMMA on TMA-loaded data**
```cuda
// 1. TMA load data to shared memory (Async Proxy)
cp.async.bulk.global.shared::cluster.mbarrier::complete_tx::bytes [smem_addr], [gmem_addr], mbarrier;

// 2. Wait for TMA load completion (Async Proxy)
mbarrier.wait.shared.b64 [mbarrier], state;  // No implicit memory fence

// 3. WgMMA fence required by PTX assembler (Generic → Async)
wgmma.fence.sync.aligned;  // Required before wgmma.commit_group, even if not logically needed

// 4. WgMMA operation (Async Proxy)
wgmma.mma_async.sync.aligned.m16n8k32.f16.f16.f16.f16 [acc], [smem_a], [smem_b];

// 5. Commit WgMMA operations (Async Proxy)
wgmma.commit_group.sync.aligned;  // No implicit memory fence

// 6. Wait for WgMMA completion (Async Proxy)
wgmma.wait_group.sync.aligned 0;  // Implicit generic-async proxy fence upon completion

// 7. Use WgMMA results in generic proxy (Generic Proxy)
// Now safe to use accumulator registers for further computation
add.f32 result, acc, other_value;
```

**Example 2: Shared memory write followed by TMA store**
```cuda
// 1. Write to shared memory (Generic Proxy)
st.shared.f32 [smem_addr], reg_value;

// 2. Explicit fence required: Generic → Async
fence.proxy.async.shared::cluster;  // Ensures shared memory write visible to TMA store

// 3. TMA store to global memory (Async Proxy)
cp.async.bulk.shared.global::cluster.mbarrier::complete_tx::bytes [gmem_addr], [smem_addr], mbarrier;
```

**Memory Fence Semantics:**

**MBarrier Operations:**
- **MBarrier Wait**: `mbarrier.wait.shared.b64` - **No implicit memory fence** - [PTX ISA: mbarrier.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrier Arrive**: `mbarrier.arrive.shared.b64` - **No implicit memory fence** - [PTX ISA: mbarrier.arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrier Test Wait**: `mbarrier.test_wait.shared.b64` - **No implicit memory fence** - [PTX ISA: mbarrier.test_wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

**WgMMA Operations:**
- **WgMMA Commit**: `wgmma.commit_group.sync.aligned` - **No implicit memory fence** - [PTX ISA: wgmma.commit_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- **WgMMA Wait**: `wgmma.wait_group.sync.aligned` - **Includes implicit generic-async proxy fence** upon completion - [PTX ISA: wgmma.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- **WgMMA Fence**: `wgmma.fence.sync.aligned` - **Includes implicit memory ordering** for register and tensor memory within async proxy - [PTX ISA: wgmma.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)

**Proxy Fence Operations:**
- **Fence Proxy Async**: `fence.proxy.async` - **Includes implicit memory ordering** between generic and async execution contexts - [PTX ISA: fence.proxy.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence)

**Key Difference**: `wgmma.fence.sync.aligned` ensures memory ordering **within the async proxy**, while `fence.proxy.async` ensures memory ordering **between generic and async execution contexts**.

**When `fence.proxy.async` is needed:**
- **Before TMA stores**: When generic proxy writes to shared memory that will be stored via TMA
- **Before WgMMA operations**: When generic proxy writes to operands that will be consumed by WgMMA (but NOT when operands come from TMA loads)

**When `fence.proxy.async` is NOT needed:**
- **Between TMA load and WgMMA**: When WgMMA operands come from TMA-loaded shared memory (both are in async proxy)
- **Between async proxy operations**: When both producer and consumer are in the same execution context
- **After WgMMA completion**: When WgMMA results need to be used by generic proxy operations (implicit fence provided by `wgmma.wait_group`)

**PTX Assembler Requirements:**
- **`wgmma.fence` is required** before any `wgmma.commit_group` operation, even when not logically needed for memory ordering
- The PTX assembler will complain if `wgmma.commit_group` is used without a preceding `wgmma.fence`

**Documentation References:**
- [CUDA Programming Guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator) - TMA operations and requirements
- [CUDA Programming Guide: Warp Group Matrix Multiply Accumulate](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-group-matrix-multiply-accumulate) - WgMMA operations and requirements
- [PTX ISA: wgmma.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma) - Implicit generic-async proxy fence upon completion
- [PTX ISA: fence.proxy.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence) - Proxy fence instruction

#### 8. Async Proxy Fences (FenceAsyncProxy)

Async proxy fences ensure visibility between generic and async execution contexts.

**Code Locations:**
- **IR Node Definition**: [`csrc/kernel_ir.h:577`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L577) - `FenceAsyncProxy` class
- **Sync Insertion**: [`csrc/device_lower/pass/insert_syncs.cpp:60`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L60) - `getAsyncFence()`
- **Code Generation**: [`csrc/codegen.cpp:453`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L453) - FenceAsyncProxy generation

**Features:**
- `fence.proxy.async` instruction - [PTX ISA: fence.proxy.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence)
- Ensures writes in generic proxy are visible to async proxy
- **Required before**: TMA stores, WgMMA operations, and other async operations that consume data written by generic proxy operations
- **Optional Modifiers**: The PTX instruction supports optional `.shared::cta`, `.shared::cluster`, and `.global` modifiers to restrict the fence to specific memory spaces. nvFuser uses `.shared::cluster` for TMA operations and `.shared::cta` for tcgen05 operations - [PTX ISA: fence.proxy.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-fence)
- **Implicit Memory Fence**: Proxy fence includes implicit memory ordering between generic and async execution contexts - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

### Reductions and Implicit Syncs

#### 9. Implicit Syncs from Reductions

Reductions automatically provide synchronization guarantees.

**Block Reductions:**
- **Runtime**: [`runtime/block_reduction.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_reduction.cu) - `blockReduce` function
- **Features**: Uses shared memory and block syncs internally
- **Sync Points**: Automatically includes `block_sync::sync()` calls
- **Implicit Memory Fence**: Shared memory operations in reductions include implicit memory ordering - [CUDA Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)

**Grid Reductions:**
- **Runtime**: [`runtime/grid_reduction.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_reduction.cu) - `gridReduce` function
- **Features**: Combines block reductions with grid synchronization
- **Sync Points**: Uses both block syncs and grid syncs
- **Implicit Global Memory Fence**: Grid reductions include implicit global memory ordering

**Welford Reductions:**
- **Runtime**: [`runtime/welford.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/welford.cu) - Welford reduction implementation
- **Features**: Maintains running statistics with proper synchronization
- **Implicit Sync**: Welford operations include implicit block synchronization for shared memory access

### MBarrier Operations

#### 14. MBarrier Operations

MBarriers provide asynchronous synchronization for modern GPU architectures (Hopper+). **MBarrier operations execute in the Generic Proxy context.**

**Code Locations:**
- **IR Node Definitions**: [`csrc/kernel_ir.h:670-802`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L670-L802) - All MBarrier classes
- **Runtime Implementation**: [`runtime/mbarrier.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/mbarrier.cu) - Complete mbarrier implementation
- **Code Generation**: [`csrc/codegen.cpp:4056-4112`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L4056-L4112) - All mbarrier handlers
- **Index Lowering**: [`csrc/device_lower/pass/index.cpp:1432-1519`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/index.cpp#L1432-L1519) - MBarrier lowering

**MBarrier Types:**
- **MBarrierInit**: [`csrc/kernel_ir.h:670`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L670) - Initialize mbarrier with thread count - [PTX ISA: mbarrier.init](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrierInvalidate**: [`csrc/kernel_ir.h:696`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L696) - Invalidate mbarrier state - [PTX ISA: mbarrier.inval](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrierArrive**: [`csrc/kernel_ir.h:715`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L715) - Arrive at mbarrier (**No implicit memory fence**) - [PTX ISA: mbarrier.arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrierArriveExpectTx**: [`csrc/kernel_ir.h:745`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L745) - Arrive with expected transaction count (**No implicit memory fence**) - [PTX ISA: mbarrier.arrive.expect_tx](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrierWait**: [`csrc/kernel_ir.h:779`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L779) - Wait for mbarrier completion (**No implicit memory fence**) - [PTX ISA: mbarrier.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- **MBarrierWaitParity**: [`csrc/kernel_ir.h:802`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L802) - Wait with parity check (**No implicit memory fence**) - [PTX ISA: mbarrier.wait.parity](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

**Memory Fence Semantics**: MBarrier operations provide synchronization but **do not include implicit memory fences**. Explicit memory fences must be used when memory ordering is required between different execution contexts - [CUDA Programming Guide: Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier)

### WgMMA Operations

#### 10. WgMMA Operations

WgMMA (Warp Group Matrix Multiply Accumulate) operations have their own sync mechanisms.

**Code Locations:**
- **IR Node Definition**: [`csrc/kernel_ir.h:594`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L594) - `WgMmaFence` class
- **Sync Insertion**: [`csrc/device_lower/pass/insert_syncs.cpp:460-490`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L460-L490) - WgMMA sync handling
- **Code Generation**: [`csrc/codegen.cpp:438-476`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L438-L476) - WgMMA fence generation

**WgMMA Sync Types:**
- **WgMmaFence**: [`csrc/kernel_ir.h:594`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L594) - Fence before WgMMA operations - [PTX ISA: wgmma.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- **Async Commit**: [`csrc/kernel_ir.h:928`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L928) - `AsyncCommit` for WgMMA commit operations - [PTX ISA: wgmma.commit_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- **Async Wait**: [`csrc/kernel_ir.h:893`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L893) - `AsyncWait` for WgMMA wait operations - [PTX ISA: wgmma.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)

**Implicit Memory Fence**: WgMMA operations include implicit memory ordering for tensor memory access - [CUDA Programming Guide: Warp Group Matrix Multiply Accumulate](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-group-matrix-multiply-accumulate)

#### 11. WgMMA Commits and Waits

WgMMA operations use commit/wait patterns for async execution.

**Code Locations:**
- **Async Commit**: [`csrc/device_lower/pass/insert_syncs.cpp:70`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L70) - `getAsyncCommit()`
- **Async Wait**: [`csrc/device_lower/pass/insert_syncs.cpp:77`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L77) - `getAsyncWait()`
- **Code Generation**: [`csrc/codegen.cpp:993-1010`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L993-L1010) - Async commit/wait generation

**Features:**
- `wgmma.commit_group.sync.aligned` for committing operations - [PTX ISA: wgmma.commit_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- `wgmma.wait_group.sync.aligned` for waiting on completion - [PTX ISA: wgmma.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- Supports keeping multiple stages active (`keep_stages` parameter)
- **Implicit Sync**: The `.sync.aligned` qualifier ensures warp-level synchronization - [PTX ISA: Synchronization Qualifiers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#synchronization-qualifiers)

#### 12. WgMMA Fences

WgMMA fences ensure proper ordering of operations.

**Code Locations:**
- **IR Node**: [`csrc/kernel_ir.h:594`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L594) - `WgMmaFence` class
- **Sync Insertion**: [`csrc/device_lower/pass/insert_syncs.cpp:60`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L60) - `getAsyncFence()`
- **Code Generation**: [`csrc/codegen.cpp:448`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L448) - WgMMA fence generation

**Features:**
- `wgmma.fence.sync.aligned` instruction - [PTX ISA: wgmma.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-wgmma)
- Ensures writes to operands are visible to async proxy
- Placed before first WgMMA operation in warp group
- **Implicit Memory Fence**: WgMMA fence includes implicit memory ordering for register and tensor memory - [CUDA Programming Guide: Warp Group Matrix Multiply Accumulate](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-group-matrix-multiply-accumulate)



### Tensor Memory (TMem) Operations

#### 13. tcgen05 Nodes

tcgen05 nodes handle tensor memory operations with built-in synchronization.

**Code Locations:**
- **IR Node**: [`csrc/kernel_ir.cpp:438-548`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.cpp#L438-L548) - `Asm` class with tcgen05 utilities
- **Inline PTX**: [`csrc/device_lower/pass/inline_ptx.cpp:128-177`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/inline_ptx.cpp#L128-L177) - tcgen05 instruction generation
- **Allocation**: [`csrc/device_lower/pass/allocation.cpp:1638-1731`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/allocation.cpp#L1638-L1731) - tcgen05 allocation

**tcgen05 Sync Types:**
- **tcgen05.wait::ld.sync.aligned**: Wait for load completion (Generic Proxy) - [PTX ISA: tcgen05.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.wait::st.sync.aligned**: Wait for store completion (Generic Proxy) - [PTX ISA: tcgen05.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.alloc.cta_group::1.sync.aligned**: Allocate tensor memory (Generic Proxy) - [PTX ISA: tcgen05.alloc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned**: Release allocation permit (Generic Proxy) - [PTX ISA: tcgen05.relinquish_alloc_permit](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.dealloc.cta_group::1.sync.aligned**: Deallocate tensor memory (Generic Proxy) - [PTX ISA: tcgen05.dealloc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64**: Commit with mbarrier (Generic Proxy) - [PTX ISA: tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)

**tcgen05 Fence Instructions:**
- **tcgen05.fence::before_thread_sync**: **Code motion fence** - prevents reordering of tcgen05 instructions before thread synchronization - [PTX ISA: tcgen05.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)
- **tcgen05.fence::after_thread_sync**: **Code motion fence** - prevents reordering of tcgen05 instructions after thread synchronization - [PTX ISA: tcgen05.fence](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tcgen05)

**Important**: tcgen05 fence instructions are **code motion fences**, not memory fences. They prevent the PTX assembler and hardware from reordering tcgen05 instructions but do not provide memory ordering guarantees between generic and async execution contexts.

**tcgen05 Synchronization Pattern:**
```cuda
// 1. Allocate tensor memory (Generic Proxy)
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [address], num_columns;

// 2. Load data to tensor memory (Generic Proxy)
tcgen05.ld.sync.aligned.32x32b.x1.b32 [tmem_addr], [smem_addr];

// 3. Code motion fence (if needed for instruction ordering)
tcgen05.fence::before_thread_sync;  // Prevents reordering of tcgen05 instructions

// 4. tcgen05 utcmma operation (Async Proxy)
tcgen05.mma.cta_group::1.kind::f16 [acc], [tmem_a], [tmem_b], idesc, enable_input_d;

// 5. Commit with mbarrier (Generic Proxy)
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbarrier];

// 6. Wait for completion using mbarrier (Generic Proxy)
mbarrier.wait.shared.b64 [mbarrier], state;  // No implicit memory fence

// 7. Use results in generic proxy (Generic Proxy)
// Explicit fence.proxy.async required since tcgen05.mma (Async Proxy) produced the data
fence.proxy.async.shared::cta;  // Ensures tcgen05.mma results visible to generic proxy
add.f32 result, acc, other_value;
```

**Key Differences from WgMMA:**
- **tcgen05 utcmma**: Uses mbarrier-based synchronization, **no implicit proxy fence** upon completion
- **WgMMA**: Uses `wgmma.wait_group` which **includes implicit generic-async proxy fence** upon completion
- **tcgen05 fence**: **Code motion fence only** - prevents instruction reordering, no memory ordering
- **WgMMA fence**: **Memory fence** - ensures operand visibility within async proxy

**When `fence.proxy.async` is needed for tcgen05:**
- **Before tcgen05 utcmma**: When generic proxy writes to operands that will be consumed by tcgen05 utcmma (Async Proxy)
- **After tcgen05 utcmma**: When tcgen05 utcmma results need to be used by generic proxy operations (requires explicit fence since no implicit proxy fence)

**Implicit Sync**: The `.sync.aligned` qualifier ensures proper synchronization for tensor memory operations - [PTX ISA: Synchronization Qualifiers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#synchronization-qualifiers)

#### 15. __threadfence Operations

Thread fence operations ensure memory visibility across different execution contexts.

**Code Locations:**
- **Grid Sync**: [`runtime/grid_sync.cu:44`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L44) - `__threadfence()` before grid synchronization
- **Block Serialize Release**: [`runtime/grid_sync.cu:264`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L264) - `__threadfence()` before semaphore release
- **Grid Broadcast**: [`runtime/grid_broadcast.cu:70`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_broadcast.cu#L70) - `__threadfence()` after work buffer write
- **Cycle Counter**: [`runtime/helpers.cu:520`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/helpers.cu#L520) - `__threadfence()` before reading cycle counter

**Features:**
- Ensures all global memory transactions complete before proceeding - [CUDA Programming Guide: __threadfence()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- Used before cross-block synchronization
- Critical for maintaining memory consistency in distributed operations
- **Implicit Memory Ordering**: `__threadfence()` ensures all memory operations are visible to all threads and devices - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

### Advanced Synchronization

#### 16. Block Serialization Syncs

Block serialization provides ordered execution of blocks within reduction segments.

**Code Locations:**
- **IR Node Definitions**: [`csrc/kernel_ir.h:859-893`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/kernel_ir.h#L859-L893) - `BlockSerializeWait` and `BlockSerializeRelease` classes
- **Code Generation**: [`csrc/codegen.cpp:4112-4160`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/codegen.cpp#L4112-L4160) - Block serialization handlers
- **Runtime Implementation**: [`runtime/grid_sync.cu:218-276`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L218-L276) - `blockSerializeWait` and `blockSerializeRelease` functions
- **Sync Insertion**: [`csrc/device_lower/pass/grid_serialization.cpp:91-142`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/grid_serialization.cpp#L91-L142) - Grid serialization pass

**Features:**
- Serializes blocks within reduction segments using semaphore-based waiting
- `BlockSerializeWait`: Waits for turn to proceed, then block sync
- `BlockSerializeRelease`: Block sync, then signals next block to proceed
- Includes `__threadfence()` and `__syncthreads()` for proper memory visibility
- Used for serial grid reductions
- **Implicit Memory Fence**: Block serialization includes implicit memory ordering for global memory - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

#### 17. Semaphore Operations

Low-level semaphore operations for cross-block coordination.

**Code Locations:**
- **Runtime**: [`runtime/grid_sync.cu:175-218`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L175-L218) - `semaphoreFetch`, `semaphoreRelease`, `semaphoreWait`
- **Features**: Uses CUDA's acquire/release operations for memory ordering

**Semaphore Types:**
- **semaphoreFetch**: [`runtime/grid_sync.cu:175`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L175) - Non-blocking semaphore read with `ld.global.acquire.gpu.b64` - [PTX ISA: ld.global.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-instructions-ld)
- **semaphoreRelease**: [`runtime/grid_sync.cu:192`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L192) - Non-blocking semaphore write with `st.global.release.gpu.b64` - [PTX ISA: st.global.release](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-instructions-st)
- **semaphoreWait**: [`runtime/grid_sync.cu:200`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L200) - Busy-wait until semaphore matches trigger value

**Implicit Memory Ordering**: Acquire/release operations provide implicit memory ordering - [CUDA Programming Guide: Memory Consistency Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-consistency-model)

#### 18. __threadfence_block Operations

Block-level memory fences ensure memory visibility within blocks.

**Code Locations:**
- **Atomic Block Sync**: [`runtime/block_sync_atomic.cu:32`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_atomic.cu#L32) - `__threadfence_block()` before atomic sync
- **Block Serialize Release**: [`runtime/grid_sync.cu:264`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L264) - `__threadfence_block()` implied by `__syncthreads()`

**Features:**
- Ensures all block-level memory operations are visible to all threads in the block - [CUDA Programming Guide: __threadfence_block()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- Used before block synchronization operations
- **Implicit in __syncthreads()**: `__syncthreads()` includes an implicit `__threadfence_block()` - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

#### 19. __nanosleep Operations

Sleep-based backoff used in busy-wait loops to reduce contention.

**Code Locations:**
- **Grid Sync**: [`runtime/grid_sync.cu:67, 154`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L67) - `__nanosleep(ns)` in busy-wait loops with exponential backoff
- **Atomic Block Sync**: [`runtime/block_sync_atomic.cu:47`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_atomic.cu#L47) - `__nanosleep(backoff)` in atomic sync with backoff
- **MBarrier Wait**: [`runtime/mbarrier.cu:75, 95`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/mbarrier.cu#L75) - `nanosleep.u32 20` in mbarrier wait loops

**Features:**
- Reduces contention in busy-wait loops - [PTX ISA: nanosleep](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep)
- Uses exponential backoff (8ns to 256ns) to avoid overwhelming the system
- Only available on compute capability 7.0 or higher
- **Not a sync operation**: `__nanosleep` is not a synchronization primitive but a performance optimization

### Host and Stream Synchronization

#### 20. Host-Level Synchronization

Host-side synchronization operations for multi-device and communication.

**Code Locations:**
- **Host IR**: [`csrc/host_ir/host_ir.h:242-309`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/host_ir/host_ir.h#L242-L309) - `Wait` and `Synchronize` classes
- **Multi-device**: [`csrc/multidevice/communicator.cpp:406`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/communicator.cpp#L406) - `barrier()` function
- **P2P Communication**: [`csrc/multidevice/cuda_p2p.cpp:40-70`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/cuda_p2p.cpp#L40-L70) - Stream-based synchronization

**Host Sync Types:**
- **Wait**: [`csrc/host_ir/host_ir.h:242`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/host_ir/host_ir.h#L242) - Makes current stream wait on another stream - [CUDA Programming Guide: Stream Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-management)
- **Synchronize**: [`csrc/host_ir/host_ir.h:275`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/host_ir/host_ir.h#L275) - Non-blocking host synchronization - [CUDA Programming Guide: Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization)
- **Barrier**: [`csrc/multidevice/communicator.cpp:406`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/communicator.cpp#L406) - Process group barrier for multi-device - [CUDA Programming Guide: Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)

#### 21. Stream-Based Synchronization

CUDA stream synchronization for async operations and communication.

**Code Locations:**
- **P2P Communication**: [`csrc/multidevice/cuda_p2p.cpp:40-70`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/cuda_p2p.cpp#L40-L70) - `cuStreamWriteValue32`, `cuStreamWaitValue32`
- **Features**: Uses CUDA stream synchronization primitives

**Stream Sync Types:**
- **Stream Write Value**: [`csrc/multidevice/cuda_p2p.cpp:42-50`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/cuda_p2p.cpp#L42-L50) - Write value to stream with memory barrier - [CUDA Programming Guide: Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-memory-operations)
- **Stream Wait Value**: [`csrc/multidevice/cuda_p2p.cpp:58-62`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/multidevice/cuda_p2p.cpp#L58-L62) - Wait for stream value to match condition - [CUDA Programming Guide: Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-memory-operations)

**Implicit Memory Fence**: Stream memory operations include implicit memory ordering - [CUDA Programming Guide: Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-memory-operations)

#### 22. Acquire/Release Memory Operations

Memory ordering operations for proper memory visibility across execution contexts.

**Code Locations:**
- **Semaphore Fetch**: [`runtime/grid_sync.cu:175`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L175) - `ld.global.acquire.gpu.b64` for acquire semantics - [PTX ISA: ld.global.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-instructions-ld)
- **Semaphore Release**: [`runtime/grid_sync.cu:192`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu#L192) - `st.global.release.gpu.b64` for release semantics - [PTX ISA: st.global.release](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-instructions-st)

**Features:**
- Ensures proper memory ordering for cross-block communication - [CUDA Programming Guide: Memory Consistency Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-consistency-model)
- Requires compute capability 7.0 or higher
- Critical for maintaining consistency in distributed operations
- **Implicit Memory Ordering**: Acquire/release operations provide implicit memory ordering - [CUDA Programming Guide: Release and Acquire Patterns](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#release-and-acquire-patterns)

## Sync Insertion Strategy

The sync insertion process follows these phases:

1. **RAW (Read After Write) Syncs**: [`csrc/device_lower/pass/insert_syncs.cpp:440`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L440) - `ReadAfterWriteSyncs` class
   - Inserts syncs between writes and subsequent reads
   - Handles async operations (WgMMA, cp.async, TMA)
   - Manages mbarrier operations

2. **WAR (Write After Read) Syncs**: [`csrc/device_lower/pass/insert_syncs.cpp:194`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L194) - `WarSyncInserter` class
   - Inserts syncs to prevent write-after-read hazards
   - Handles shared memory and tensor memory
   - Manages loop-based sync insertion

3. **Async Wait Syncs**: [`csrc/device_lower/pass/insert_syncs.cpp:989`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/csrc/device_lower/pass/insert_syncs.cpp#L989) - `WarAsyncWaitInserter` class
   - Inserts async wait operations
   - Handles circular buffer stages
   - Manages warp-specialized kernels

## Runtime Sync Implementations

The runtime directory contains optimized implementations:

- **Block Sync**: [`runtime/block_sync_default.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_default.cu) and [`runtime/block_sync_atomic.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_sync_atomic.cu)
- **Grid Sync**: [`runtime/grid_sync.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_sync.cu) - Complete grid synchronization
- **Reductions**: [`runtime/block_reduction.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/block_reduction.cu) and [`runtime/grid_reduction.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_reduction.cu)
- **Warp Operations**: [`runtime/warp.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/warp.cu) - Warp-level operations
- **MBarriers**: [`runtime/mbarrier.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/mbarrier.cu) - Modern async barriers
- **Broadcasts**: [`runtime/grid_broadcast.cu`](https://github.com/NVIDIA/Fuser/blob/0b43ff27f42c9fbca001f656f055aeef885ebf31/runtime/grid_broadcast.cu) - Grid-level broadcasts

## Implicit Synchronization Summary

Several operations in nvFuser include implicit synchronization or memory fences:

### Implicit Memory Fences
- **`__syncthreads()`**: Includes implicit `__threadfence_block()` - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- **Reductions**: Shared memory operations include implicit memory ordering - [CUDA Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- **MBarriers**: Include implicit memory ordering for shared memory - [CUDA Programming Guide: Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier)
- **WgMMA**: Includes implicit memory ordering for tensor memory - [CUDA Programming Guide: Warp Group Matrix Multiply Accumulate](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-group-matrix-multiply-accumulate)
- **Proxy Fences**: Include implicit memory ordering between execution contexts - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
- **Stream Memory Operations**: Include implicit memory ordering - [CUDA Programming Guide: Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-memory-operations)

### Implicit Synchronization
- **Grid Syncs**: Include implicit `block_sync::sync()` calls
- **Reductions**: Include implicit block synchronization for shared memory access
- **WgMMA**: The `.sync.aligned` qualifier ensures warp-level synchronization - [PTX ISA: Synchronization Qualifiers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#synchronization-qualifiers)
- **tcgen05**: The `.sync.aligned` qualifier ensures proper synchronization - [PTX ISA: Synchronization Qualifiers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#synchronization-qualifiers)

### Memory Ordering
- **Acquire/Release Operations**: Provide implicit memory ordering - [CUDA Programming Guide: Release and Acquire Patterns](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#release-and-acquire-patterns)
- **`__threadfence()`**: Ensures all memory operations are visible to all threads and devices - [CUDA Programming Guide: Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

## Environment Variables

- `USE_BLOCK_SYNC_ATOMIC`: Enables atomic-based block synchronization for debugging
- Various other environment variables control sync behavior and debugging

This comprehensive sync system ensures correct execution across all nvFuser kernels while maintaining high performance through optimized implementations and strategic placement. 