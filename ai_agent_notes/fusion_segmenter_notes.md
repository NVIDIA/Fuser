# NVFuser Fusion Segmenter Development Notes

## Current Changes (from main)

### Data Structure Changes
1. Changed input/output storage in SegmentedGroup from std::vector to VectorOfUniqueEntries:
   - `input_vals` -> `input_vals_`
   - `output_vals` -> `output_vals_`

2. Removed fusion input group specific functionality:
   - Removed `is_fusion_input_` flag
   - Removed `isFusionInputGroup()` method
   - Removed `isConnected()` method
   - Removed special constructor for fusion input groups

### Edge Handling
1. Improved edge management and documentation:
   - Added comments clarifying edge usage
   - Simplified edge container organization
   - Added better type safety for edge handling

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
1. Removed redundant functions:
   - Removed uniqueValConcat in favor of VectorOfUniqueEntries
   - Simplified getAllInputs/getAllOutputs implementations
   - Removed duplicate code in merging operations

2. Improved error handling and validation:
   - Added more error checks for group operations
   - Improved validation of group merging
   - Better handling of edge cases

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

## Next Steps

1. **Reevaluation of Edge Removal**:
   - Reconsider the motivation for removing SegmentedEdge
   - Identify what specific problems we were trying to solve
   - Evaluate if there are alternative solutions that don't require edge removal

2. **Alternative Approaches**:
   - Consider enhancing rather than removing the edge system
   - Look for ways to simplify edge handling while maintaining correctness
   - Explore hybrid approaches that might combine benefits of both systems

3. **Investigation Areas**:
   - Study how other systems handle similar graph dependency tracking
   - Look for opportunities to optimize the current edge-based implementation
   - Consider if partial refactoring of the edge system could achieve our goals

4. **Documentation**:
   - Better document the current edge system's role and importance
   - Clarify edge handling in complex scenarios (multiple consumers, merging)
   - Update comments and documentation to reflect latest understanding

The focus should shift from removing SegmentedEdge to finding the right balance between simplicity and correctness in dependency tracking.

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

**Goal: Removing SegmentedEdge:**

The current development aims to **eliminate the `SegmentedEdge` class**. The "Current Changes (from main)" section reflects this:

*   `SegmentedGroup::input_vals_` and `output_vals_` (now `VectorOfUniqueEntries`) are being repurposed to store *all* inputs/outputs for a group, including those that connect to other groups (previously only represented by edges).
*   `SegmentedGroup::producer_groups_` and `SegmentedGroup::consumer_groups_` (also `VectorOfUniqueEntries`) have been added to directly store references to neighboring groups.
*   **(Correction):** As of the last update, `producer_groups_` and `consumer_groups_` have *not* yet been added. They represent a planned mechanism to facilitate `SegmentedEdge` removal.

The intended future state is to represent inter-group dependencies implicitly. The specific `Val`s connecting a `producer` group to a `consumer` group can be derived by computing the intersection of their respective outputs and inputs: `producer->output_vals_.computeIntersect(consumer->input_vals_)`. This removes the need for explicit edge objects to store this information.

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

## Identified Failures

### Instance Normalization Merge Failure (Issue ##TBD##)

- **Test Case:** `NVFuserTest.FusionMagicSchedulerInstanceNormalization_CUDA`
- **Symptom:** Test fails with an `INTERNAL ASSERT FAILED` in `logical_domain_map.cpp` during segment merging.
- **Analysis:** The failure occurs when attempting to merge two segments:
    1. A reduction segment derived from a `Welford` operation (calculating mean `T5` and variance `T6` from input `T0`, then using `T5` to compute `T8`).
    2. A pointwise segment that uses both the original input `T0` and a broadcast of the Welford mean `T5` (specifically, `T22 = T0 - broadcast(T5)`).
- **Root Cause:** The `MERGE_DEBUG` logs preceding the crash indicate that the segmenter incorrectly identifies the Welford mean (`T5_g_float`) as an **input** to the hypothetically merged segment. However, `T5` is produced *internally* within this combined segment by the `Welford` operation acting on the true input `T0`. This incorrect input identification leads to the failure when `PairwiseLogicalDomainMap` attempts to map the domains between the operations, as it's working with an incorrect understanding of the segment's dependencies.
- **Resolution:** The logic for determining the inputs of a merged segment needs to be corrected to properly distinguish between true external inputs and values produced and consumed internally within the potential merged segment. 