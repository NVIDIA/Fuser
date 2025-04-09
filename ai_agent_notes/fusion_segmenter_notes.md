# NVFuser Fusion Segmenter Development Notes

## Current Changes (from main)

### Edge Handling Improvements
1. Added helper functions in SegmentedFusion:
   - `connectGroups(from, to, val)` - Creates and connects new edge
   - `disconnectGroups(group1, group2)` - Removes all edges between groups
   - `removeEdge(edge)` - Removes single edge and updates all references
   - `getEdgesBetween(from, to)` - Gets all edges between two groups

2. Refactored edge manipulation code:
   - Moved edge creation/deletion to SegmentedFusion class
   - Using helper functions instead of direct manipulation
   - Fixed edge invalidation issues in merge operations
   - Improved edge cleanup during group merging

3. Edge handling safety improvements:
   - Better documentation of edge lifetime/validity
   - Clear comments about when edges become invalid
   - Proper ordering of edge operations during merges
   - Collection of edges before removal when needed

### Data Structure Changes
1. Changed input/output storage in SegmentedGroup from std::vector to VectorOfUniqueEntries:
   - `input_vals` -> `input_vals_`
   - `output_vals` -> `output_vals_`

2. Removed fusion input group specific functionality:
   - Removed `is_fusion_input_` flag
   - Removed `isFusionInputGroup()` method
   - Removed `isConnected()` method
   - Removed special constructor for fusion input groups

### Debug Output
1. Added more debug output for FusionSegmentGuard:
   - Added entry/exit logging
   - Added fusion math printing
   - Added merge operation logging

### Algorithmic Changes
1. Changed how groups are merged:
   - Now uses VectorOfUniqueEntries for deduplication
   - Improved producer/consumer relationship detection
   - Added ProducerConsumerRelationship struct for better relationship tracking

2. Modified input/output handling:
   - Changed how forwarded inputs are processed
   - Simplified input group creation
   - Improved scalar input handling

### Code Cleanup
1. Improved error handling and validation:
   - Added more error checks for group operations
   - Improved validation of group merging
   - Better handling of edge cases

2. Better code organization:
   - Centralized edge manipulation in SegmentedFusion
   - Clearer separation of responsibilities
   - More consistent API usage

## Revised Understanding of SegmentedEdge

The initial plan to remove SegmentedEdge was based on the assumption that group dependencies could be tracked purely through input/output relationships. However, this assumption has proven incorrect due to several key insights:

1. **Multi-Consumer Challenge**: A key realization is that outputs from one SegmentedGroup can be consumed by multiple other groups. Simply tracking inputs and outputs is insufficient because:
   - We can't remove a value from a group's outputs just because it's consumed by one merged group
   - The same output may need to remain as an output for other consumer groups
   - The current edge-based system correctly handles these many-to-many relationships

2. **Dependency Tracking Complexity**: The current approach using getAllInputs/getAllOutputs for merging groups isn't sufficient because:
   - Simple intersection and union operations on input/output lists don't capture the full complexity of group dependencies
   - The edge system provides explicit producer-consumer relationships that are harder to infer from just input/output lists
   - Edges maintain important information about data flow that isn't captured by just knowing inputs and outputs

## Fusion Segmentation Process & Role of SegmentedEdge

The core idea is to partition the original `Fusion` DAG into `SegmentedGroup`s, where each group represents a sub-graph that can potentially be scheduled and executed independently. `SegmentedEdge` objects historically played a crucial role in representing the data dependencies (`Val`s) between these groups.

**Construction:**

1.  **Initial Segmentation (`SegmentCandidateFinder::buildInitialSegments`):** The fusion DAG is initially traversed, and expressions are grouped based on heuristics or properties (like scheduler compatibility). When an expression's input comes from a `Val` produced by an expression in a *different* initial group, a `SegmentedEdge` is created via `SegmentedFusion::newEdge` connecting the producer group to the consumer group via that `Val`.
2.  **Precision Casting (`SegmentedFusion::castInputOutputToLowerPrecision`):** During segmentation testing or finalization, edges might be temporarily modified or new edges created if Cast operations are inserted between groups to handle precision differences (e.g., FP32 to FP16).

**Modification:**

*   The `SegmentedEdge` objects themselves (their `from`, `to`, `val` fields) are generally *not* modified after creation, except potentially during the precision casting process mentioned above.
*   However, the *collections* of edges are heavily modified:
    *   `SegmentedGroup` stores `producer_edges` and `consumer_edges`.
    *   `SegmentedFusion` stores a master list `edges_`.
    *   **Merging (`SegmentCandidateFinder::mergeNodes`):** When two `SegmentedGroup`s are merged, the edges connecting them are effectively removed. Edges connecting the *original* groups to *other* groups are re-parented to the new, merged group. The `disconnectGroup` helper facilitates removing edges from the global list and neighbor lists.

**Usage (Historically and Currently):**

1.  **Graph Topology:** Edges explicitly define the DAG structure of the segmented fusion, showing which groups produce data (`Val`s) consumed by other groups.
2.  **Inter-Group I/O:** They were the primary mechanism to identify the specific `Val`s that serve as inputs and outputs *between* groups. This information was critical in `SegmentedGroup::finalize` to populate the `input_vals_` and `output_vals_` lists for inter-group communication (distinct from the overall fusion's inputs/outputs).
3.  **Merge Candidate Selection:** Identifying potential groups to merge relies on traversing the graph using edges (`getNeighborGroups`, `getMergeCandidates`). The specific edge (`merge_through_`) connecting merge candidates is stored.
4.  **Scheduling & Code Generation:** The boundaries defined by edges (implicitly through group inputs/outputs derived from them) determine the scope of sub-fusions tested for schedulability (`FusionSegmentGuard`, `tryMerge`).
5.  **Analysis:** Dependency analysis (`GroupDependencyAnalysis`) uses edge information extensively to understand the producer-consumer relationships between groups.
6.  **Serialization:** The complete structure, including edges, is serialized to save/load segmented fusions.

**Edge Handling During Group Merging:**

When two or more `SegmentedGroup`s are merged into a new `joined_group` (e.g., in `SegmentCandidateFinder::mergeNodes` or `SegmentCandidateFinder::mergeAllGivenGroups`):

1.  **Identify External Edges:** Helper functions like `getMergedProducerEdges` and `getMergedConsumerEdges` (or iterating through individual group edges in `mergeAllGivenGroups`) gather all edges connected to the groups being merged.
2.  **Filter Internal Edges:** Edges *between* the groups being merged are identified and discarded. They represent dependencies that are now internal to the `joined_group`.
3.  **Create New Edges:** For each remaining external edge (connecting one of the original groups to a group *outside* the merge set):
    *   A **new** `SegmentedEdge` is created using `SegmentedFusion::newEdge`.
    *   This new edge connects the original external group (`from` or `to`) to the new `joined_group`.
    *   The `Val` remains the same as the original edge.
4.  **Update Neighbor Connections:** The `producer_edges` and `consumer_edges` lists of the `joined_group` and the external neighboring groups are updated with these new edges.
5.  **Disconnect Original Groups:** The original groups being merged are completely disconnected from the graph using `SegmentCandidateFinder::disconnectGroup`. This involves removing their edges from the global `edges_` list in `SegmentedFusion` and from the edge lists of their former neighbors.
6.  **Cleanup:** The original groups and their associated internal/external edges are marked for cleanup and eventually removed from the `SegmentedFusion`'s `groups_` and `edges_` lists.

Essentially, the merging process preserves the external connectivity by creating new edges linked to the merged group, while internalizing the dependencies previously represented by edges between the merged groups. 

## Planned Edge Handling Optimizations

After evaluating the current edge system and its challenges, we have identified several optimization approaches to be implemented in order of priority:

1. **Helper Functions for Edge Management**
   - Implement centralized edge manipulation functions in SegmentedFusion
   - Key functions to add:
     ```cpp
     void connectGroups(SegmentedGroup* from, SegmentedGroup* to, Val* val);
     void disconnectGroups(SegmentedGroup* group1, SegmentedGroup* group2);
     void removeEdge(SegmentedEdge* edge);
     std::vector<SegmentedEdge*> getEdgesBetween(SegmentedGroup* g1, SegmentedGroup* g2);
     ```
   - Benefits:
     - Reduces error-prone manual edge management
     - Centralizes edge manipulation logic
     - Improves code maintainability
     - Minimal refactoring required

2. **EdgeSet Abstraction**
   - Create EdgeSet class to encapsulate edge operations and queries
   - Key features:
     - Maintains deterministic edge ordering
     - Provides efficient group-based queries
     - Supports standard iteration
   - Implementation considerations:
     - Keep vector for deterministic iteration
     - Add optional indexing for faster lookups if needed
     - Replace direct edge vectors in SegmentedGroup

3. **Shared Pointer Edge Ownership**
   - Convert edge storage to use std::shared_ptr
   - Benefits:
     - Explicit ownership semantics
     - Safer memory management
     - Clearer object lifetime management
   - Implementation notes:
     - Update both SegmentedFusion and SegmentedGroup edge storage
     - Ensure proper cleanup in destructors
     - Consider impact on serialization

4. **Value-Based Edge Management (Future)**
   - Long-term architectural enhancement
   - Maintain dual representation:
     - Traditional edge-based connectivity
     - Value-based producer/consumer relationships
   - Key considerations:
     - Synchronization between representations
     - Impact on existing algorithms
     - Migration strategy

**Implementation Strategy:**

1. **Phase 1: Helper Functions**
   - Add helper functions to SegmentedFusion
   - Update existing code to use new helpers
   - Add tests for edge manipulation
   - Document common edge operation patterns

2. **Phase 2: EdgeSet Introduction**
   - Implement EdgeSet class
   - Add unit tests for EdgeSet
   - Gradually migrate SegmentedGroup to use EdgeSet
   - Update documentation

3. **Phase 3: Shared Pointer Migration**
   - Convert edge storage to shared_ptr
   - Update edge creation/deletion code
   - Verify proper cleanup
   - Add ownership documentation

4. **Phase 4: Value-Based Enhancement**
   - Design value-based representation
   - Implement dual representation
   - Add synchronization logic
   - Migrate algorithms as needed

**Key Areas to Monitor:**
- Performance impact of each change
- Memory usage patterns
- Code complexity metrics
- Impact on existing algorithms
- Test coverage

**Success Metrics:**
- Reduced code complexity in edge handling
- Fewer edge-related bugs
- Improved code readability
- Maintained or improved performance
- Better test coverage of edge operations

The implementation will proceed incrementally, with each phase being fully tested before moving to the next. Regular reviews will ensure the changes maintain the system's correctness while improving its maintainability. 

## Next Steps

### Edge Handling
1. Consider creating EdgeSet abstraction:
   - Maintain deterministic ordering
   - Provide efficient queries
   - Support standard iteration
   - Replace direct edge vectors in SegmentedGroup

2. Evaluate shared pointer usage for edges:
   - Consider explicit ownership semantics
   - Improve memory management
   - Make object lifetime clearer

3. Remaining direct edge manipulations to review:
   - Edge value modifications in cast operations
   - Scalar edge removal process
   - Edge handling during group merging

### Testing
1. Add more edge manipulation tests:
   - Edge creation/deletion
   - Group connection/disconnection
   - Edge invalidation cases
   - Edge cleanup verification

## Current failing tests:
(previous failing tests section preserved as is) 