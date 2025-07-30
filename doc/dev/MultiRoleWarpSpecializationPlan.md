# Multi-Role Warp Specialization Plan for Blackwell MMA

## Table of Contents
- [Overview](#overview)
- [Current Architecture](#current-architecture)
- [Planned Architecture](#planned-architecture)
- [Key Design Changes](#key-design-changes)
  - [1. Scheduling Changes](#1-scheduling-changes)
  - [2. Circular Buffer Analysis Changes](#2-circular-buffer-analysis-changes)
  - [3. Circular Buffer Pass Changes](#3-circular-buffer-pass-changes)
  - [4. MBarrier Management Changes](#4-mbarrier-management-changes)
  - [5. Indexing and Synchronization Changes](#5-indexing-and-synchronization-changes)
- [Implementation Plan](#implementation-plan)
  - [Phase 1: Analysis and Detection](#phase-1-analysis-and-detection)
  - [Phase 1.5: Dependency DAG Analysis](#phase-15-dependency-dag-analysis)
  - [Phase 2: Code Generation](#phase-2-code-generation)
  - [Phase 2.5: Backward Compatibility](#phase-25-backward-compatibility)
  - [Phase 3: Synchronization](#phase-3-synchronization)
  - [Phase 4: Testing and Validation](#phase-4-testing-and-validation)
- [Key Technical Considerations](#key-technical-considerations)
- [Code Locations to Modify](#code-locations-to-modify)
  - [Primary Files](#primary-files)
  - [Supporting Files](#supporting-files)
- [Testing Strategy](#testing-strategy)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
- [Success Criteria](#success-criteria)
- [Conclusion](#conclusion)

## Overview

This document outlines the plan to extend nvFuser's circular buffering support to enable multi-role warp specialization for Blackwell MMA operations. The current system supports single-role warp specialization where one async warp handles TMA loads and compute warpgroups handle WgMMA and epilogue operations. For Blackwell, we need to support a second async warp (the "mma warp") that consumes the operand circular buffer populated by the first async warp (the "load warp").

## Current Architecture

### Hopper Architecture (Current)
- **AsyncWarp**: Handles TMA loads for operands
- **ComputeWarpGroups**: Handle WgMMA operations and epilogue computations
- **Circular Buffer Tensors**: Only operand tensors (A and B) are circular buffered in shared memory
- **Synchronization**: Uses mbarriers for each circular buffer slot:
  - **"Slot Empty" mbarrier**: Indicates slot is available for loading
  - **"Slot Full" mbarrier**: Indicates slot contains valid data for consumption
  - **AsyncWarp**: Waits for "slot empty" → issues TMA loads → arrives at "slot full"
  - **ComputeWarpGroups**: Wait for "slot full" → consume data → arrive at "slot empty"

## Planned Architecture

### Blackwell Architecture (Target)
- **LoadWarp**: Handles TMA loads for operands (first async warp)
- **MmaWarp**: Handles tcgen05 utcmma operations (second async warp)
- **EpilogueWarpGroups**: Handle epilogue computations only
- **Circular Buffer Tensors**:
  - Operand circular buffer tensors (A and B) - populated by LoadWarp, consumed by MmaWarp
  - MMA result circular buffer tensor - populated by MmaWarp, consumed by EpilogueWarpGroups
- **Synchronization**: Extended mbarriers for three-way synchronization:
  - **"Operand Slot Empty" mbarrier**: Indicates operand slot is available for loading
  - **"Operand Slot Full" mbarrier**: Indicates operand slot contains valid operands for MMA
  - **"Result Slot Empty" mbarrier**: Indicates result slot is available for MMA results
  - **"Result Slot Full" mbarrier**: Indicates result slot contains valid MMA results for epilogue
  - **LoadWarp**: Waits for "operand slot empty" → issues TMA loads → arrives at "operand slot full"
  - **MmaWarp**: Waits for "operand slot full" → issues tcgen05 utcmma → arrives at "result slot full"
  - **EpilogueWarpGroups**: Wait for "result slot full" → consume results → arrive at "result slot empty"

**Note**: The above describes a simple case with a single pipeline of chained circular buffers. In general, we might have multiple pipelines of chained circular buffers. For example, the epilogue warp group might also consume a circular buffer of epilogue inputs which is filled by another async warp. One of the key challenges to the analysis is:
1. **Grouping async operations and circular buffered tensors** into parts that can be computed by a single warp
2. **Determining the producer/consumer relationships** between the async warps and the epilogue compute warps
3. **Managing multiple circular buffer chains** that may have different synchronization requirements
4. **Building a dependency DAG** that represents all producer/consumer relationships and deriving mbarrier semantics from the graph structure

## Key Design Changes

### 1. Scheduling Changes

#### Current Behavior
- During scheduling for Hopper, circular buffering is applied only to operand tensors in shared memory (`acw_smems_` and `bcw_smems_`)
- The circular buffer type is determined by `getCircularBufferType()` which returns `WarpSpecialized(ParallelType::TIDy)` for warp-specialized strategy
- Circular buffer analysis detects TMA load patterns and creates single AsyncWarp

#### Required Changes
- **File**: `csrc/scheduler/matmul_hopper+.cpp`
- **Function**: `Blackwell::setUpCircularBuffering()`
- **Changes**:
  - Mark `mma_results_` for circular buffering (in addition to operands)
  - Handle potential assertion issues with circular buffered tensor definitions
  - Note: This is a key difference from Hopper, which only circular buffers operand tensors (`acw_smems_` and `bcw_smems_`) in shared memory

### 2. Circular Buffer Analysis Changes

#### Current Behavior
- `createAsyncWarps()` only detects `cpAsyncBulkLoad` operations
- Creates single AsyncWarp with all async operations
- TODO comment indicates need for multi-role support

#### Required Changes
- **File**: `csrc/device_lower/analysis/circular_buffer.cpp`
- **Function**: `createAsyncWarps()`
- **Changes**:
  - Find all definitions of circular buffered tensors (instead of explicitly looking for specific operation types)
  - Analyze and separate async operations into different roles:
    - `LoadAsyncWarp`: Contains TMA load operations
    - `MmaAsyncWarp`: Contains tcgen05 utcmma operations
  - Validate that operations in each warp have compatible `stage_slice_position`
  - Update circular buffered tensor validation to accept async operations (not just LoadStoreOp)

### 3. Circular Buffer Pass Changes

#### Current Behavior
- `WarpSpecializedCircularBufferInserter` creates single ITE branch for async operations
- Single mbarrier synchronization between async and compute warpgroups

#### Required Changes
- **File**: `csrc/device_lower/pass/circular_buffer.cpp`
- **Class**: `WarpSpecializedCircularBufferInserter`
- **Changes**:
  - Modify `insertTmaWarpSpecialized()` to handle multiple async warps
  - Add new method `insertMultiRoleWarpSpecialized()` for Blackwell pattern
  - Create separate ITE branches for load warp and mma warp
  - Add intermediate mbarrier between mma warp and epilogue warpgroups

### 4. MBarrier Management Changes

#### Current Behavior
- Uses `HopperPingPongMbarriers` for async ↔ compute synchronization
- Two mbarriers per circular buffer slot: "slot empty" and "slot full"
- AsyncWarp waits for "slot empty", arrives at "slot full"
- ComputeWarpGroups wait for "slot full", arrive at "slot empty"

#### Required Changes
- **File**: `csrc/device_lower/pass/circular_buffer.h`
- **Class**: `HopperPingPongMbarriers`
- **Changes**:
  - Create general DAG-based mbarrier management system
  - Represent async warps and compute warp groups as nodes in a dependency DAG
  - Derive mbarrier semantics from the graph structure
  - Support arbitrary pipeline configurations (not just 2-way or 4-way)
  - Generate appropriate mbarriers based on producer/consumer relationships

### 5. Indexing and Synchronization Changes

#### Current Behavior
- `insert_syncs.cpp` handles WgMMA synchronization with `wgmma.commit_group` and `wgmma.wait_group`
- Single async fence pattern

#### Required Changes
- **File**: `csrc/device_lower/pass/insert_syncs.cpp`
- **Changes**:
  - Add detection for Blackwell MMA operations
  - Insert `tcgen05.commit` and `tcgen05.wait` operations
  - Add `fence.proxy.async` for mma warp synchronization
  - Ensure proper mbarrier indexing for multi-role pattern

## Implementation Plan

### Phase 1: Analysis and Detection
1. **Extend AsyncWarp Detection**
   - Modify `createAsyncWarps()` to find all definitions of circular buffered tensors
   - Analyze and separate async operations into different roles (load vs mma)
   - Validate stage_slice_position compatibility
   - Create utility function to identify all async operations

2. **Update Circular Buffer Analysis**
   - Extend `CircularBufferInfo` to track multi-role patterns
   - Add detection for producer-consumer relationships between circular buffered tensors
   - Validate that TMA-loaded operands feed into Blackwell MMA operations
   - Update circular buffered tensor validation to accept async operations (not just LoadStoreOp)

### Phase 1.5: Dependency DAG Analysis
1. **Group Async Operations into Warps**
   - **Stage Slice Position Analysis**: Group operations with compatible `stage_slice_position`
   - **Operation Type Analysis**: Separate TMA loads from MMA operations
   - **Resource Sharing Analysis**: Ensure operations in same warp can share resources
   - **Validation**: Verify each warp group can be executed by a single warp

2. **Detect Dependencies Between Warps**
   - **Data Flow Analysis**: Track which circular buffered tensors are consumed by which operations
   - **Producer/Consumer Mapping**: Build dependency edges between warps based on tensor usage
   - **Circular Buffer Chain Detection**: Identify chains of circular buffered tensors
   - **Dependency Graph Construction**: Create DAG representing all warp dependencies

3. **DAG-Based MBarrier Generation**
   - **Graph Traversal**: Analyze DAG to determine synchronization requirements
   - **MBarrier Allocation**: Generate appropriate mbarriers for each producer/consumer edge
   - **Wait/Arrive Pattern Generation**: Derive synchronization patterns from graph structure
   - **Validation**: Ensure no deadlocks and proper synchronization order

### Phase 2: Code Generation
1. **Extend Warp Specialization Pass**
   - Modify `WarpSpecializedCircularBufferInserter` to handle multiple async warps
   - Create separate ITE branches for load warp and mma warp
   - Add register sharing management for both async warps

2. **Update MBarrier Management**
   - Create general DAG-based mbarrier management system
   - Build dependency graph from async warps and compute warp groups
   - Generate mbarriers dynamically based on producer/consumer relationships
   - Support arbitrary pipeline configurations

### Phase 2.5: Backward Compatibility
1. **Feature Flag Implementation**
   - Add feature flag to enable/disable multi-role warp specialization
   - Default to current Hopper behavior for existing kernels
   - Allow opt-in for Blackwell kernels

2. **Gradual Rollout Strategy**
   - **Phase 1**: Implement new analysis with feature flag disabled (current behavior)
   - **Phase 2**: Enable for Blackwell kernels only
   - **Phase 3**: Enable for all kernels after validation
   - **Phase 4**: Remove old code paths

3. **Testing and Validation**
   - **Regression Testing**: Ensure all existing Hopper tests pass
   - **Performance Testing**: Validate no performance regression for Hopper kernels
   - **Compatibility Testing**: Test mixed Hopper/Blackwell kernels

### Phase 3: Synchronization
1. **Update Sync Insertion**
   - Extend `insert_syncs.cpp` for Blackwell MMA synchronization
   - Add proper `tcgen05` commit/wait pattern
   - Ensure correct `fence.proxy.async` placement

2. **Update Indexing**
   - Modify `index.cpp` for tcgen05 operations
   - Ensure proper mbarrier indexing for multi-role pattern
   - Add support for tcgen05 tensor memory operations

### Phase 4: Testing and Validation
1. **Unit Tests**
   - Add tests for multi-role warp specialization detection
   - Test circular buffer analysis with Blackwell MMA patterns
   - Validate mbarrier synchronization correctness

2. **Integration Tests**
   - Test end-to-end Blackwell matmul with circular buffering
   - Validate performance improvements
   - Ensure backward compatibility with Hopper patterns

## Key Technical Considerations

### 1. Async Operation Detection
- **Current**: Only detects `cpAsyncBulkLoad` operations (TMA loads for operands)
- **Required**: Find all definitions of circular buffered tensors to detect any async operation
- **Implementation**: Modify `createAsyncWarps()` to iterate over circular buffered tensors and find their definitions

### 2. Stage Slice Position Compatibility
- **Current**: All operations in single AsyncWarp must have same `stage_slice_position`
- **Required**: Load and mma operations can have different positions but must be compatible
- **Implementation**: Add validation for producer-consumer relationships

### 3. MBarrier Indexing
- **Current**: Two mbarriers per circular buffer slot ("slot empty", "slot full")
- **Required**: Dynamic mbarrier allocation based on DAG dependencies
- **Implementation**: Create general DAG-based system that generates mbarriers based on producer/consumer relationships

### 4. Register Sharing
- **Current**: Single register sharing configuration for async warp
- **Required**: Separate register sharing for load warp and mma warp
- **Implementation**: Extend `WarpSpecialized` options to support multiple async warps

### 5. Backward Compatibility
- **Requirement**: Existing Hopper patterns must continue to work
- **Implementation**: Detect multi-role pattern and fall back to single-role for Hopper
- **Testing**: Ensure all existing tests pass

## Code Locations to Modify

### Primary Files
1. **`csrc/device_lower/analysis/circular_buffer.cpp`**
   - `createAsyncWarps()`: Find all definitions of circular buffered tensors and analyze roles
   - `CircularBufferInfo`: Extend for multi-role patterns
   - Update circular buffered tensor validation

2. **`csrc/device_lower/pass/circular_buffer.cpp`**
   - `WarpSpecializedCircularBufferInserter`: Add multi-role support
   - `HopperPingPongMbarriers`: Extend for three-way sync

3. **`csrc/device_lower/pass/insert_syncs.cpp`**
   - Add Blackwell MMA synchronization
   - Extend async fence patterns

4. **`csrc/scheduler/matmul_hopper+.cpp`**
   - `Blackwell::setUpCircularBuffering()`: Mark mma_results_ for circular buffering

5. **`csrc/device_lower/utils.cpp`**
   - Create utility function to identify all async operations
   - Update circular buffered tensor validation logic

### Supporting Files
1. **`csrc/device_lower/pass/index.cpp`**
   - Add tcgen05 indexing support

2. **`csrc/device_lower/pass/allocation.cpp`**
   - Extend mbarrier allocation for multi-role pattern

## Testing Strategy

### Unit Tests
1. **AsyncWarp Detection and Role Analysis**
   - Test detection of all circular buffered tensor definitions
   - Test role analysis and separation of async operations
   - Test validation of stage_slice_position compatibility
   - Test utility function for async operation detection

2. **Dependency DAG Analysis**
   - Test grouping of async operations into warps
   - Test dependency detection between warps
   - Test DAG construction and validation
   - Test mbarrier generation from dependency graph

3. **Circular Buffer Analysis**
   - Test producer-consumer relationship detection
   - Test multi-role pattern validation
   - Test circular buffered tensor validation with async operations

4. **MBarrier Management**
   - Test DAG-based mbarrier generation
   - Test proper synchronization for arbitrary pipeline configurations
   - Test mbarrier wait/arrive patterns derived from dependency graph

5. **Backward Compatibility**
   - Test feature flag functionality
   - Test current Hopper behavior with new code
   - Test gradual rollout strategy

### Integration Tests
1. **End-to-End Blackwell Matmul**
   - Test complete circular buffering pipeline
   - Validate performance improvements

2. **Backward Compatibility**
   - Ensure Hopper patterns still work correctly
   - Test mixed Hopper/Blackwell kernels

## Success Criteria

1. **Functional Correctness**
   - Blackwell matmul with circular buffering produces correct results
   - All existing Hopper tests continue to pass
   - No performance regression for existing patterns

2. **Performance Improvement**
   - Blackwell matmul achieves better performance with multi-role specialization
   - Proper overlap between load, mma, and epilogue operations

3. **Code Quality**
   - Clean separation between single-role and multi-role patterns
   - Maintainable and extensible design
   - Comprehensive test coverage





## Conclusion

This plan provides a comprehensive approach to extending nvFuser's circular buffering support for multi-role warp specialization in Blackwell MMA operations. The implementation will maintain backward compatibility while enabling the performance benefits of the new architecture. The phased approach allows for incremental development and testing, reducing risk and ensuring quality
