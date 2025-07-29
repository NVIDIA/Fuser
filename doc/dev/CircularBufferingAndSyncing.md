# Circular Buffering and Syncing in nvFuser

This document describes how nvFuser implements circular buffering for memory optimization and the various synchronization mechanisms required to ensure correctness.

## Overview

Circular buffering is a memory optimization technique that allows overlapping computation and memory access by using multiple buffers in a rotating fashion. nvFuser implements this through several key components:

1. **Loop Cloning**: Circular buffered loops are cloned into Prologue, Main, and Epilogue phases
2. **Memory Aliasing**: Shared memory buffers are aliased to reuse memory space
3. **Predication**: Predicates are specified and converted to If-Then-Else (ITE) structures
4. **Sync Insertion**: Various types of synchronization are automatically inserted

## Circular Buffer Loop Structure

### Basic Structure

A circular buffered loop structure follows this pattern:

```cpp
// Original loop
for i in 0..N:
  for j in ...
    x[j] = y[i, j]  // Load
  for j in ...
    ... = x[j]       // Use

// After circular buffering (stage depth D)
allocate X[S*D]  // D times original size

// Prologue: Load initial D-1 stages
for i in 0..D-1:
  for j in ...
    if pred:
      x[i*S+j] = y[i, j]

// Main loop: Overlap load and compute
for i in 0..N:
  for j in ...
    if pred:
      x[((i+D-1)%D)*S+j] = y[i+D-1, j]  // Load next stage
  for j in ...
    ... = x[(i%D)*S+j]                    // Use current stage

// Epilogue: Use remaining stages
for i in N..N+D-1:
  for j in ...
    ... = x[(i%D)*S+j]
```

### Loop Cloning Process

The circular buffer pass (`CircularBufferPass`) performs the following transformations:

1. **Identifies circular buffer axes**: Finds the innermost non-parallelized, non-unrolled axis
2. **Clones loops**: Creates Prologue, Main, and Epilogue versions of the original loop
3. **Adjusts indexing**: Modifies tensor indexing to use modulo arithmetic for stage selection
4. **Handles multiple tensors**: When multiple tensors share the same circular buffer loop, all are processed together

### Stage Depth and Prefetch Distance

- **Stage Depth (D)**: Number of buffer stages used (e.g., D=2 for double buffering)
- **Prefetch Distance**: How many iterations ahead to load data
- **Keep Stages**: Number of incomplete transactions to maintain in flight

## Memory Aliasing and Allocation

### Shared Memory Buffer Aliasing

The `alias_memory` pass implements memory reuse through aliasing:

#### Alias Criteria
- Input and output allocations have the same size
- Thread bindings match
- Input is not used after the operation
- Memory types are compatible

#### Two Types of Aliasing

1. **Inner Aliasing**: Reuse within the same loop iteration
   - Requires exact index mapping
   - No synchronization needed
   - Limited to pointwise and reduction operations

2. **Outer Aliasing**: Reuse across different loop iterations
   - Requires block synchronization
   - More aggressive memory reuse
   - Handles complex access patterns

#### Stack-Based Allocation

The `StackBasedSharedMemAllocator` implements a stack-based approach for shared memory allocation:

```cpp
// Stack tracks active allocations
std::vector<AllocationInfo*> alloc_stack_;

// When encountering a sync point
void reclaimMemory() {
  while (!alloc_stack_.empty()) {
    auto last_read = lastAliasedRead(alloc_stack_.back());
    if (last_read <= position_) {
      alloc_stack_.pop_back();  // Reclaim memory
    } else {
      break;
    }
  }
}
```

### Memory Reuse and WAR Synchronization

Memory reuse creates a critical connection to WAR (Write-After-Read) synchronization. When memory is reused (not aliased), a WAR hazard occurs because the new allocation might overwrite memory that is still being read by other threads.

#### The Problem
```cpp
// Time 1: Thread A reads from shared memory
T0[i, j] = smem_buffer[k]  // Read operation

// Time 2: Memory is reused for new allocation
smem_buffer = new_allocation  // Same memory location

// Time 3: Thread B writes to the reused memory
smem_buffer[k] = T1[i, j]  // Write operation - WAR HAZARD!
```

#### The Solution: Promote Reuse Syncs
The `promoteReuseSyncs` function identifies intervals where memory reuse occurs and ensures proper synchronization:

```cpp
// Find intervals between last read and first write
for (const auto& alloc_info : allocation_info_map.allAllocationInfos()) {
  if (tv->shouldPromoteReuse()) {
    auto last_read = alloc_info->getAliasedOuterLastRead();
    auto first_write = findNextFirstWrite(last_read);
    
    // Create sync interval (last_read, first_write)
    sync_intervals_.emplace(last_read, first_write);
  }
}

// Insert syncs in intervals where reuse occurs
for (auto [last_read, first_write] : sync_intervals_) {
  if (!hasBlockSyncBetween(last_read, first_write)) {
    insertBlockSyncBefore(first_write);
  }
}
```

#### Stack-Based Memory Reclamation
The `StackBasedSharedMemAllocator` uses synchronization points to safely reclaim memory:

```cpp
// When encountering a sync point
void reclaimMemory() {
  while (!alloc_stack_.empty()) {
    auto last_read = lastAliasedRead(alloc_stack_.back());
    if (last_read <= position_) {
      alloc_stack_.pop_back();  // Safe to reclaim - all reads complete
    } else {
      break;  // Still being read - cannot reclaim yet
    }
  }
}
```

#### Memory Reuse vs Aliasing
- **Aliasing**: No synchronization needed - same memory location used for different purposes
- **Memory Reuse**: Requires WAR synchronization - different allocations using same memory location at different times

## Predication and If-Then-Else Conversion

### Predicate Types

The unroll pass (`UnrollPass`) handles several types of predicates:

1. **Thread Predicates**: Ensure only active threads execute operations
2. **Inline Predicates**: Simple if-then-else wrapping
3. **Vectorize Predicates**: For vectorized operations
4. **Reduction Write Predicates**: Special handling for reduction outputs
5. **ElectSync Predicates**: For TMA and Blackwell MMA operations

### Predicate Conversion Process

```cpp
// Original expression
T0[i, j] = T1[i, j] + T2[i, j]

// After predication
if (thread_pred) {
  T0[i, j] = T1[i, j] + T2[i, j]
}
```

### Warp Specialization and Predicate Hierarchy

Warp specialization introduces a hierarchical predicate system where different types of predicates are applied at different levels:

#### 1. Top-Level Warp Specialization Predicates
The highest level predicates separate warp groups:

```cpp
// Top-level ITE for warp specialization
if (thread_axis >= block_dim_axis - padded_value) {
  // AsyncWarp operations
  if (ElectSync()) {
    mbarrier::init(...)
  }
} else {
  // ComputeWarp operations
  if (thread_pred) {
    compute_operation(...)
  }
}
```

#### 2. Register Management Predicates
Within each warp group, register management predicates control resource allocation:

```cpp
// In AsyncWarp branch
setMaxNReg(decrease_num_registers);  // Fewer registers for async ops

// In ComputeWarp branch  
setMaxNReg(increase_num_registers);   // More registers for compute ops
```

#### 3. Thread-Level Predicates
Within each warp group, standard thread predicates ensure correct execution:

```cpp
// Standard thread predicates within warp groups
if (thread_pred) {
  // Individual thread operations
  T0[i, j] = T1[i, j] + T2[i, j]
}
```

#### 4. Specialized Predicates for Async Operations
Async operations within warp-specialized code use special predicates:

```cpp
// TMA operations in AsyncWarp
if (PredicateType::ElectSync) {
  tma_load(...)
}

// MBarrier operations
if (PredicateType::OneDimTmaLoadExpectArrive) {
  mbarrier::arriveExpectTx(...)
}
```

### Special Cases

#### Circular Buffer Predicates
For circular buffered TMA loads, special predicates are used:

```cpp
// 1D TMA loads with circular buffering
if (PredicateType::OneDimTmaLoadExpectArrive) {
  tma_load(...)
}

// MBarrier wait with parity
if (PredicateType::OneDimTmaWaitParity) {
  mbarrier::wait_parity(...)
}
```

#### Warp Specialization Predicates and Top-Level ITEs

Warp specialization creates a fundamental transformation in the IR structure by introducing top-level If-Then-Else (ITE) statements that separate different warp groups. This is a key optimization for Hopper GPUs that allows overlapping computation and memory access.

##### Warp Specialization Structure

Warp specialization pads the CTA (Cooperative Thread Array) with 128 threads to support register sharing and creates specialized warp groups:

```cpp
// Original loop structure
for i in 0..N:
  for j in ...
    load_data(...)
  for j in ...
    compute(...)

// After warp specialization
if (thread_axis >= block_dim_axis - padded_value) {
  // AsyncWarp: Handle memory operations
  setMaxNReg(decrease_num_registers);
  for i in 0..N:
    for j in ...
      tma_load(...)  // TMA operations
  return;  // Terminate async warp immediately
} else {
  // ComputeWarp: Handle computation
  setMaxNReg(increase_num_registers);
  for i in 0..N:
    for j in ...
      compute(...)  // Computation operations
}
```

##### Predicate Creation for Warp Specialization

The warp specialization predicate is created based on the padded thread dimension:

```cpp
kir::Predicate* getAsyncWarpPredicate(const CircularBufferOptions& options) {
  ParallelType warp_specialize_on = std::get<WarpSpecialized>(options.type).on;
  int64_t warp_specialization_pad = 
      GpuLower::current()->parallelDimensionMap()
          .getWarpSpecializationPaddedVal(warp_specialize_on);
  Val* raw = GpuLower::current()->parallelDimensionMap().get(warp_specialize_on);
  Val* raw_minus_pad = SimplifyingIrBuilder::subExpr(
      raw, IrBuilder::create<Val>(warp_specialization_pad, DataType::Index));
  
  // Predicate: thread_axis >= block_dim_axis - padded_value
  return IrBuilder::create<kir::Predicate>(IrBuilder::geExpr(
      NamedScalar::getParallelIndex(warp_specialize_on), raw_minus_pad));
}
```

##### Top-Level ITE Creation

The `WarpSpecializedCircularBufferInserter` creates a top-level ITE that replaces the original loop:

```cpp
void insertTmaWarpSpecialized(ForLoop* circular_buffer_loop, 
                              const std::vector<Expr*>& loads,
                              int64_t insertion_position) {
  // Create the top-level ITE with warp specialization predicate
  kir::IfThenElse* warp_dispatch_ite = 
      IrBuilder::create<kir::IfThenElse>(getAsyncWarpPredicate(options));
  
  // AsyncWarp branch (then body)
  if (enable_register_sharing) {
    // Decrease registers for async operations
    kir::SetMaxNReg* dec_reg_async_warp = 
        IrBuilder::create<kir::SetMaxNReg>(decrease_num_registers, false);
    warp_dispatch_ite->thenBody().push_back(dec_reg_async_warp);
  }
  
  // Load loop for async operations
  ForLoop* load_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
      circular_buffer_loop, loads, CircularBufferLoopStage::AsyncWarp, 
      insertion_position);
  warp_dispatch_ite->thenBody().push_back(load_loop);
  
  // Terminate async warp immediately after loading
  kir::Return* ret = IrBuilder::create<kir::Return>();
  warp_dispatch_ite->thenBody().push_back(ret);
  
  // ComputeWarp branch (else body)
  if (enable_register_sharing) {
    // Increase registers for compute operations
    kir::SetMaxNReg* inc_reg_async_warp = 
        IrBuilder::create<kir::SetMaxNReg>(increase_num_registers, true);
    warp_dispatch_ite->elseBody().push_back(inc_reg_async_warp);
  }
  
  // Compute loop for computation operations
  ForLoop* compute_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
      circular_buffer_loop, loads, CircularBufferLoopStage::ComputeWarp, 
      insertion_position);
  warp_dispatch_ite->elseBody().push_back(compute_loop);
  
  // Replace original loop with the ITE
  registerReplace(circular_buffer_loop, warp_dispatch_ite);
}
```

##### Continue Insertion for Register Sharing

When certain patterns match (particularly for register sharing scenarios), a `kir::Continue` is inserted to optimize control flow. This is especially important in warp-specialized kernels where register pressure is high.

###### Pattern Matching for Continue Insertion

The system identifies specific patterns where Continue insertion is beneficial:

```cpp
// Pattern 1: Persistent loops with register sharing
if (for_loop_stack_.size() == 1 && insertion_position_ != 1 && 
    !for_loop_stack_.front()->isTrivial()) {
  kir::IfThenElse* ite = createPersistentShortCircuit(for_loop_stack_.front(), cloned_loop);
  if (ite != nullptr) {
    for_loop_stack_.back()->body().push_back(ite);
  }
}

// Pattern 2: Async warp termination
// Async warps terminate immediately after loading to free up resources
kir::Return* ret = IrBuilder::create<kir::Return>();
warp_dispatch_ite->thenBody().push_back(ret);
```

###### Continue Insertion Implementation

```cpp
// Create persistent short-circuit to minimize wave quantization
kir::IfThenElse* createPersistentShortCircuit(ForLoop* outer_fl, ForLoop* cloned_loop) {
  // Create predicate for early termination
  Val* predicate_val = SimplifyingIrBuilder::geExpr(lhs, presplit_extent);
  kir::Predicate* predicate = IrBuilder::create<kir::Predicate>(predicate_val);
  kir::IfThenElse* ite = IrBuilder::create<kir::IfThenElse>(predicate);
  
  // Insert continue to skip remaining iterations
  kir::Continue* cont = IrBuilder::create<kir::Continue>();
  ite->thenBody().push_back(cont);
  
  return ite;
}
```

###### Benefits of Continue Insertion

1. **Register Pressure Reduction**: Early termination frees up registers for other operations
2. **Wave Quantization**: Minimizes wave quantization effects in persistent kernels
3. **Resource Management**: Allows better resource allocation between warp groups
4. **Control Flow Optimization**: Reduces unnecessary iterations in specialized scenarios

###### Integration with Warp Specialization

Continue insertion works in conjunction with warp specialization:

```cpp
// In warp-specialized circular buffering
if (thread_axis >= block_dim_axis - padded_value) {
  // AsyncWarp: Load data and terminate
  tma_load(...);
  return;  // Immediate termination
} else {
  // ComputeWarp: Process data with continue optimization
  if (persistent_condition) {
    continue;  // Skip remaining iterations
  }
  compute(...);
}
```

##### Warp Specialization Effects on Predication

Warp specialization affects predication in several ways:

1. **Register Management**: Different register limits for async and compute warps
2. **Control Flow**: Async warps terminate immediately after loading
3. **Synchronization**: MBarrier operations coordinate between warp groups
4. **Memory Access**: TMA operations are restricted to async warps
5. **Computation**: Regular computation is restricted to compute warps

This separation allows for better resource utilization and overlapping of memory and compute operations.

##### Warp Specialization Summary

Warp specialization is a sophisticated optimization that transforms the IR structure through several key mechanisms:

1. **Top-Level ITE Creation**: Replaces loops with If-Then-Else structures that separate warp groups
2. **Predicate Hierarchy**: Implements a multi-level predicate system for different optimization goals
3. **Register Management**: Uses different register limits for async and compute operations
4. **Continue Insertion**: Optimizes control flow for register sharing and persistent kernels
5. **Immediate Termination**: Async warps terminate immediately after loading to free resources

The combination of these techniques enables efficient resource utilization and overlapping of memory and compute operations on modern GPU architectures.

## Synchronization Types and Insertion

### RAW (Read-After-Write) Synchronization

The `ReadAfterWriteSyncs` class handles RAW hazards by inserting synchronization between writes and subsequent reads.

#### Block Synchronization
```cpp
// Insert block sync before reading shared memory
BlockSync();
// Read from shared memory
```

#### Grid Synchronization
```cpp
// For global memory operations
GridSync(bitmap);
```

#### Async Operation Synchronization
```cpp
// For async operations like wgmma
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

### WAR (Write-After-Read) Synchronization

The `WarSyncInserter` class handles WAR hazards by inserting synchronization at the end of loops. These WAR syncs are particularly important for memory reuse scenarios.

#### Memory Reuse WAR Hazards
When memory is reused (not aliased), WAR hazards occur because:
1. **Memory Location Reuse**: The same memory location is used for different allocations at different times
2. **Read-Write Overlap**: A new allocation might start writing before the previous allocation has finished being read
3. **Thread Synchronization**: Different threads might be reading and writing to the same memory location

#### Block Sync for WAR
```cpp
// At end of loop to prevent overwriting before reading
BlockSync(/*war_sync=*/true);
```

#### Memory Reuse WAR Syncs
For memory reuse scenarios, additional WAR syncs are inserted:

```cpp
// When memory is reused, ensure all reads complete before reuse
if (memory_reuse_scenario) {
  BlockSync();  // Ensure all threads finish reading old allocation
  // Memory can now be safely reused for new allocation
}
```

#### Async Wait for WAR
```cpp
// For async operations like wgmma
wgmma.wait_group.sync.aligned keep_stages;
```

#### Stack-Based WAR Prevention
The stack-based allocator prevents WAR hazards by only reclaiming memory at synchronization points:

```cpp
// Only reclaim memory when we know all reads are complete
void reclaimMemory() {
  while (!alloc_stack_.empty()) {
    auto last_read = lastAliasedRead(alloc_stack_.back());
    if (last_read <= position_) {
      // Safe to reclaim - all reads are complete
      alloc_stack_.pop_back();
    } else {
      // Cannot reclaim yet - still being read
      break;
    }
  }
}
```

### MBarrier Synchronization

For Hopper ping-pong warp specialization:

```cpp
// Arrive to release resources to next warp group
mbarrier::arrive(ping_pong_mbarriers[index]);

// Wait for resources to be available
mbarrier::wait(ping_pong_mbarriers[index], parity);
```

## Circular Buffer Synchronization

### Stage Management

Circular buffering requires careful management of async operations:

```cpp
// Prologue: Load initial stages
for i in 0..D-1:
  tma_load(...)
  cp.async.commit_group.sync.aligned

// Main loop: Overlap load and compute
for i in 0..N:
  cp.async.wait_group.sync.aligned (D-2);  // Wait for D-2 stages
  tma_load(...)                            // Load next stage
  compute(...)                             // Use current stage
  cp.async.commit_group.sync.aligned       // Commit current load
  __syncthreads();                         // Ensure all threads sync
```

### Warp Specialization Synchronization

For warp-specialized circular buffering:

```cpp
// AsyncWarp: Load data
if (AsyncWarp):
  mbarrier::wait(empty_operands)
  mbarrier::arriveExpectTx(full_operands)
  tma_load(...)

// ComputeWarp: Use data
if (ComputeWarp):
  mbarrier::wait(full_operands)
  wgmma(...)
  mbarrier::arrive(full_tmem_output)
```

## Memory Allocation and Reuse

### Shared Memory Allocation

The `assignSharedMemoryAllocations` function assigns addresses to shared memory allocations:

```cpp
void assignNextAddress(AllocationInfo* alloc_info) {
  if (alloc_stack_.empty()) {
    alloc->setAddress(FusionGuard::getCurFusion()->zeroVal());
  } else {
    auto top_alloc = alloc_stack_.back()->alloc_expr;
    auto top_size = allocSizeBytes(top_alloc);
    auto unaligned_address = 
        SimplifyingIrBuilder::addExpr(top_alloc->address(), top_size);
    auto aligned_address = 
        alignExpr(unaligned_address, alloc_info->alignment);
    alloc->setAddress(aligned_address);
  }
}
```

### Memory Reuse Promotion and WAR Syncs

The `promoteReuseSyncs` function is specifically designed to handle WAR hazards that arise from memory reuse. It works in conjunction with the stack-based allocator:

```cpp
// Entry point for all memory reuse
std::vector<Expr*> reuseMemoryAllocations(const std::vector<Expr*>& exprs) {
  AllocationInfoMap allocation_info_map(exprs, debug_print);
  
  // Step 1: Find aliasing opportunities (no sync needed)
  const auto aliased_exprs = aliasMemoryAllocations(exprs, allocation_info_map);
  
  // Step 2: Insert WAR syncs for memory reuse (not aliasing)
  const auto [synced_exprs, inserted_syncs] = 
      promoteReuseSyncs(aliased_exprs, allocation_info_map);
  
  // Step 3: Assign addresses using stack-based allocation
  assignSharedMemoryAllocations(synced_exprs, allocation_info_map);
  
  return synced_exprs;
}
```

#### WAR Sync Interval Detection
The system identifies intervals where WAR hazards can occur:

```cpp
// For each allocation marked for reuse promotion
for (const auto& alloc_info : allocation_info_map.allAllocationInfos()) {
  if (tv->shouldPromoteReuse()) {
    auto last_read = alloc_info->getAliasedOuterLastRead();
    
    // Find the next allocation that will reuse this memory
    for (const auto& other : allocation_info_map.allAllocationInfos()) {
      auto first_write = other->outer_live_interval->firstWrite();
      if (first_write > last_read) {
        // Create sync interval: (last_read, first_write)
        sync_intervals_.emplace(last_read, first_write);
        break;
      }
    }
  }
}
```

#### Automatic WAR Sync Insertion
When traversing expressions, the system automatically inserts syncs:

```cpp
void dispatch(Expr* expr) final {
  auto position = allocation_info_map_.getScopeMap().getExprPos(expr);
  
  // Process last reads (end of allocation lifetimes)
  processLastReads(position);
  
  // Process first writes (beginning of new allocations)
  bool inserted_sync = processFirstWrites(expr, position);
  
  // If we have upcoming first writes that haven't been cleared by existing syncs
  if (!inserted_sync && upcoming_first_writes_.count(position)) {
    // Insert WAR sync before this expression
    auto new_sync = IrBuilder::create<kir::BlockSync>();
    registerInsertBefore(expr, new_sync);
    upcoming_first_writes_.clear();
  }
}
```

## Implementation Details

### Loop Stage Tracking

Circular buffer loops are tracked with different stages:

```cpp
enum class CircularBufferLoopStage {
  Prologue,      // Initial loading phase
  Main,          // Overlapped computation phase  
  Epilogue,      // Final consumption phase
  AsyncWarp,     // Warp-specialized async phase
  ComputeWarp    // Warp-specialized compute phase
};
```

### Memory Information Tracking

The `WarMemoryInfo` struct tracks memory usage:

```cpp
struct WarMemoryInfo {
  bool sync_after_read = false;    // Sync after last read
  bool sync_before_write = false;  // Sync before first write
  bool read_hit = false;           // Has been read
  bool write_hit = false;          // Has been written
  ForLoop* ca_loop = nullptr;      // Compute-at loop
};
```

### Async Operation Tracking

Async operations are tracked for proper synchronization:

```cpp
// Track async MMA pipeline
bool fill_async_mma_pipeline_ = false;
bool flush_async_mma_pipeline_ = false;

// Insert appropriate waits
if (async_type == AsyncOpType::WgMma) {
  insertSyncExpr(ops, expr, getAsyncWait(async_type, keep_stages), nullptr);
}
```

## Summary

Circular buffering in nvFuser is a sophisticated optimization that combines:

1. **Loop Transformation**: Cloning loops into Prologue/Main/Epilogue phases
2. **Memory Optimization**: Aliasing and reuse of shared memory buffers  
3. **Predication**: Converting predicates to If-Then-Else structures
4. **Synchronization**: Inserting appropriate syncs for RAW/WAR hazards

### Memory Reuse and WAR Synchronization

A key insight is the critical connection between memory reuse and WAR synchronization:

- **Memory Aliasing**: When the same memory location is used for different purposes simultaneously, no additional synchronization is needed
- **Memory Reuse**: When the same memory location is used for different allocations at different times, WAR synchronization is essential to prevent race conditions

The system uses a sophisticated approach to handle memory reuse:
1. **Interval Detection**: Identifies intervals between last read and first write of reused memory
2. **Sync Insertion**: Automatically inserts WAR syncs in intervals where reuse occurs
3. **Stack-Based Allocation**: Uses synchronization points to safely reclaim memory
4. **Promote Reuse Syncs**: Ensures proper synchronization for memory reuse scenarios

The system ensures correctness while maximizing memory reuse and overlapping computation with memory access. The various synchronization mechanisms (block syncs, grid syncs, mbarriers, async waits) work together to prevent race conditions while enabling aggressive optimizations. 
