# Edge Handling Refactoring Impact Analysis

## Overview
The first step of the planned edge handling optimizations involves implementing centralized edge manipulation functions in SegmentedFusion. The proposed helper functions are:

```cpp
void connectGroups(SegmentedGroup* from, SegmentedGroup* to, Val* val);
void disconnectGroups(SegmentedGroup* group1, SegmentedGroup* group2);
void removeEdge(SegmentedEdge* edge);
std::vector<SegmentedEdge*> getEdgesBetween(SegmentedGroup* g1, SegmentedGroup* g2);
```

## Implementation Progress

### Completed: removeEdge

The `removeEdge` function has been implemented and added to the codebase. Key aspects of the implementation:

1. **Interface**
   ```cpp
   //! Remove an edge from the segmented fusion graph and update all affected groups
   //! The edge object will be deleted and should not be used after this call
   void removeEdge(SegmentedEdge* edge);
   ```

2. **Implementation Details**
   - Handles removal from all three locations where edges are referenced:
     - Producer group's consumer_edges
     - Consumer group's producer_edges
     - Global edges_ list
   - Includes comprehensive error checking
   - Leverages RAII for edge object cleanup through Impl class

3. **Error Handling**
   - Validates non-null edge pointer
   - Ensures edge exists in all expected locations
   - Uses NVF_ERROR macro for consistency with codebase

4. **Memory Management**
   - Edge object cleanup handled by SegmentedFusion::Impl
   - Relies on unique_ptr ownership in Impl class
   - No manual memory management required

### Remaining Functions To Implement

1. `connectGroups`
   - Will use newEdge internally
   - Needs to handle edge list updates
   - Should validate group parameters

2. `disconnectGroups`
   - Will use removeEdge for each edge between groups
   - Needs to handle multiple edges between same groups
   - Should validate group parameters

3. `getEdgesBetween`
   - Query-only function, no graph modifications
   - Will help simplify edge lookup operations
   - Important for other functions' implementations

## Impacted Locations

### 1. SegmentedFusion Class Definition (fusion_segmenter.h)
- Add the new helper functions to the public interface of SegmentedFusion class around line 300
- These functions would be placed with other edge management functions like `newEdge()`

### 2. Edge Creation/Modification Sites

#### In SegmentCandidateFinder::buildInitialSegments() (fusion_segmenter.cpp ~line 1900)
```cpp
auto new_edge = segmented_fusion_->newEdge(aux_group, expr_group, inp);
expr_group->producer_edges.push_back(new_edge);
aux_group->consumer_edges.push_back(new_edge);
```
Would be replaced with:
```cpp
segmented_fusion_->connectGroups(aux_group, expr_group, inp);
```

#### In SegmentCandidateFinder::disconnectGroup() (fusion_segmenter.cpp ~line 1800)
Current manual edge removal logic would be replaced with calls to:
```cpp
disconnectGroups()
removeEdge()
```

#### In SegmentCandidateFinder::mergeNodes() (fusion_segmenter.cpp ~line 2000-2100)
Multiple locations where edges are manually created and connected:
```cpp
auto new_edge = segmented_fusion_->newEdge(from, joined_group, val);
joined_group->producer_edges.push_back(new_edge);
from->consumer_edges.push_back(new_edge);
```
Would be replaced with:
```cpp
segmented_fusion_->connectGroups(from, joined_group, val);
```

#### In SegmentCandidateFinder::mergeAllGivenGroups() (fusion_segmenter.cpp ~line 2200)
Similar edge creation patterns that would use the new helper functions.

### 3. Edge Query Sites

#### In SegmentCandidateFinder::finalMerge() (fusion_segmenter.cpp ~line 3700)
Current manual edge traversal could use `getEdgesBetween()`:
```cpp
std::unordered_map<SegmentedGroup*, SegmentedEdge*> consumer_edge_map;
```

#### In various utility functions:
- getMergedProducerEdges()
- getMergedConsumerEdges() 
Could potentially be simplified using `getEdgesBetween()`

### 4. Edge Cleanup Sites

#### In SegmentCandidateFinder::eraseGroups() (fusion_segmenter.cpp)
Edge cleanup logic could be simplified using `removeEdge()`

#### In SegmentCandidateFinder::cleanupForwardedInputs() (fusion_segmenter.cpp)
Edge cleanup during input forwarding cleanup

## Benefits of Refactoring

1. **Centralized Edge Management**
   - All edge operations would go through a single interface
   - Easier to maintain consistency and add validation
   - Reduces duplicate code

2. **Improved Error Handling**
   - Central location for edge-related error checking
   - Consistent validation of edge operations

3. **Better Encapsulation**
   - Groups and edges managed through SegmentedFusion interface
   - Reduced direct manipulation of internal data structures

4. **Simplified Client Code**
   - Cleaner, more readable edge operations
   - Less error-prone edge management
   - Reduced boilerplate code

## Implementation Strategy

1. Implement the new helper functions in SegmentedFusion
2. Update test cases to use new interface
3. Gradually refactor existing code to use new interface
4. Add validation and error checking
5. Remove redundant/manual edge management code

## Next Steps

1. Implement remaining helper functions in this order:
   - getEdgesBetween (needed by other functions)
   - connectGroups
   - disconnectGroups

2. Update existing code to use new interfaces:
   - Start with simple cases using removeEdge
   - Gradually replace edge management code
   - Update tests to use new interfaces

3. Add test coverage:
   - Unit tests for each new function
   - Edge cases and error conditions
   - Integration tests with existing functionality

## Note on Backward Compatibility
During the transition, the new functions could initially be implemented in terms of the existing lower-level operations, allowing for gradual migration of the codebase while maintaining functionality. 

## Current failing tests:
[  FAILED  ] NVFuserTest.FusionSegmenterCombineReductionsCycleRepro_CUDA
[  FAILED  ] AliasTest.Bookend_Issue2375
[  FAILED  ] CombinedSchedulerTest.LayerNormBackward/dtype___half_batch_216_hidden_65536, where GetParam() = (__half, 216, 65536)
[  FAILED  ] CombinedSchedulerTest.LayerNormBackward/dtype___bfloat_batch_216_hidden_65536, where GetParam() = (__bfloat, 216, 65536)
[  FAILED  ] MovePadTest.PadReplayOnMultipleUsesCase1
[  FAILED  ] PresegTest.FusionTestCastOptimizationMetaOp3
[  FAILED  ] ResizeTest.AvoidCachingSliceInput
[  FAILED  ] ResizeSchedulerTest.PropagateMultipleSlicesToInputs2
[  FAILED  ] ResizeSchedulerTest.PropagateMultipleSlicesToInputs6
[  FAILED  ] SegmentationTest.InputForwardingUntilBinary
[  FAILED  ] SegmentationTest.InputForwardingUntilOutput
[  FAILED  ] SegmentationTest.ForwardedExprsAreReplicated
[  FAILED  ] SegmentationTest.codeGenSupportedMergeIssue1970