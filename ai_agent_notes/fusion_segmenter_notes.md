# NVFuser Fusion Segmenter Development Notes

## Overview of Fusion Segmentation

The fusion segmentation process partitions a Fusion DAG into SegmentedGroups, where each group represents a sub-graph that can potentially be scheduled and executed independently. The process involves:

1. **Initial Segmentation**
   - Traverse the fusion DAG
   - Group expressions based on heuristics/properties
   - Create edges between groups for data dependencies

2. **Group Merging**
   - Identify merge candidates
   - Validate merge feasibility
   - Combine groups while preserving dependencies

3. **Finalization**
   - Process inputs/outputs
   - Handle precision requirements
   - Generate final segmented fusion

## Edge System Design

### Core Concepts
1. **Purpose**
   - Tracks producer-consumer relationships between groups
   - Handles multi-consumer scenarios
   - Preserves data dependencies during merges
   - Enables efficient graph traversal

2. **Key Components**
   - SegmentedEdge: Represents data flow between groups
   - Edge Collections: Track producer/consumer relationships
   - Helper Functions: Centralize edge operations

### Implementation Details
1. **Edge Operations**
   ```cpp
   void connectGroups(SegmentedGroup* from, SegmentedGroup* to, Val* val);
   void disconnectGroups(SegmentedGroup* group1, SegmentedGroup* group2);
   void removeEdge(SegmentedEdge* edge);
   std::vector<SegmentedEdge*> getEdgesBetween(SegmentedGroup* g1, SegmentedGroup* g2);
   ```

2. **Data Structures**
   - VectorOfUniqueEntries for input/output storage
   - Centralized edge management in SegmentedFusion
   - Clear ownership semantics

3. **Critical Patterns**
   - Create new edges before removing old ones
   - Validate edge existence before operations
   - Handle multi-consumer cases explicitly
   - Preserve necessary edges during merges

## Implementation Status

### Completed Phase 1: Helper Functions
1. **Edge Handling**
   - Centralized helper functions implemented and tested:
     ```cpp
     void connectGroups(SegmentedGroup* from, SegmentedGroup* to, Val* val);
     void disconnectGroups(SegmentedGroup* group1, SegmentedGroup* group2);
     void removeEdge(SegmentedEdge* edge);
     std::vector<SegmentedEdge*> getEdgesBetween(SegmentedGroup* g1, SegmentedGroup* g2);
     ```
   - Proper edge operation ordering
   - Improved edge cleanup during merges
   - Better error handling and validation

2. **Data Structure Improvements**
   - Migrated to VectorOfUniqueEntries
   - Removed fusion input group specific functionality
   - Improved producer/consumer relationship tracking

3. **Testing**
   - All tests passing, including:
     - LayerNorm backward
     - Input forwarding
     - Edge deduplication
     - Precision handling

### Planned Phases

#### Phase 2: Shared Pointer Edge Ownership
- Convert edge storage to use std::shared_ptr
- Benefits:
  - Explicit ownership semantics
  - Safer memory management
  - Clearer object lifetime management
- Implementation notes:
  - Update both SegmentedFusion and SegmentedGroup edge storage
  - Ensure proper cleanup in destructors
  - Consider impact on serialization

#### Phase 3: Value-Based Edge Management
- Long-term architectural enhancement
- Maintain dual representation:
  - Traditional edge-based connectivity
  - Value-based producer/consumer relationships
- Key considerations:
  - Synchronization between representations
  - Impact on existing algorithms
  - Migration strategy

### Current State
- Phase 1 helper functions are complete and working well
- Edge system is robust and maintainable
- All edge operations go through centralized functions
- Proper ordering prevents dependency issues
- Multi-consumer scenarios handled correctly
- Edge cleanup during merges is reliable

## Next Steps

1. **Monitoring**
   - Continue monitoring edge handling in complex cases
   - Watch for any new edge-related issues
   - Maintain test coverage

2. **Documentation**
   - Keep edge handling documentation up to date
   - Document any new edge patterns
   - Update comments as needed

3. **Optimization**
   - Look for opportunities to further improve edge operations
   - Consider performance optimizations
   - Evaluate potential simplifications

## Current failing tests:
(previous failing tests section preserved as is) 