# Shared Pointer Edge Ownership Changes

## Current State
- Changed EdgePtr typedef in SegmentedFusion::Impl to use shared_ptr
- Changed SegmentedGroup's producer_edges and consumer_edges to use shared_ptr
- Updated NeighborGroup to use shared_ptr for edge
- Updated makeEdge() to create shared_ptr edges

## Current Approach
We're taking a two-layer approach to maintain compatibility:

1. Internal Layer (SegmentedFusion::Impl)
   - Uses shared_ptr for edge ownership
   - Creates and manages edges using shared_ptr
   - Provides getRawEdges() to expose raw pointers when needed

2. Public Interface Layer (SegmentedFusion)
   - Maintains raw pointer interface for backward compatibility
   - Uses cached_raw_edges_ to store raw pointers
   - Updates cache when edges() or cedges() called

This approach allows us to:
- Keep shared_ptr ownership in the implementation
- Maintain existing API for external code
- Avoid breaking changes in the public interface

## Next Steps

1. Update Edge Management
   - Update removeEdge() to handle shared_ptr properly
   - Update cleanUnused() to work with shared_ptr
   - Update edge comparison/search functions

2. Update Edge Collection
   - Ensure proper edge lifetime management
   - Handle edge caching correctly
   - Update edge iteration code

3. Update Serialization
   - Update serialization to handle shared_ptr
   - Update deserialization to create proper shared_ptr edges

## Benefits of Current Approach
1. Minimizes impact on existing code
2. Provides clear ownership semantics internally
3. Allows gradual migration to shared_ptr
4. Maintains performance by caching raw pointers

## Potential Risks
1. Need to keep cached_raw_edges_ in sync
2. Must ensure proper cleanup of shared resources
3. Need to handle edge lifetime carefully
4. Must maintain consistency between layers

## Previous Approach Issues
The previous approach of changing everything to shared_ptr would have:
1. Required extensive changes to public API
2. Broken existing code that uses raw pointers
3. Made integration more difficult
4. Required more extensive testing

## Implementation Status
- [x] Initial shared_ptr conversion in Impl
- [x] SegmentedGroup edge storage update
- [x] Basic edge creation changes
- [ ] Edge management updates
- [ ] Collection/caching implementation
- [ ] Serialization updates

## Goal
Convert the edge ownership model in the fusion segmenter to use shared pointers and simplify edge management by removing redundant storage. This change will:

1. Make edge ownership semantics more explicit and safer
2. Improve memory management through shared ownership
3. Eliminate redundant edge storage
4. Simplify the edge management code

## Current Implementation
Currently edges are:
- Stored as unique_ptr in SegmentedFusion::Impl
- Stored as raw pointers in SegmentedGroup producer/consumer edges
- Redundantly stored in SegmentedFusion edges_ member
- Managed through manual cleanup in various places

## Implementation Plan

### 1. Remove Redundant Edge Storage
- Remove `edges_` vector from SegmentedFusion class
- Update `edges()` and `cedges()` methods to collect edges from groups instead
- Keep edge storage only in SegmentedFusion::Impl for creation/management
- Update any code that directly accessed SegmentedFusion::edges_

### 2. Update Edge Storage Types
- Change `EdgePtr` typedef in `SegmentedFusion::Impl` from `unique_ptr` to `shared_ptr`
- Update any related type declarations that depend on edge pointer types

### 3. Update Edge Access Methods
- Modify `edges()` to collect unique edges from all groups' producer/consumer edges
- Update any code that relied on the global edges_ vector
- Ensure edge uniqueness when collecting from groups

### 4. Update Edge Creation and Management
- Modify `SegmentedFusion::Impl::makeEdge()` to use `std::make_shared`
- Update edge creation in `SegmentedFusion::newEdge()`
- Update edge handling in `connectGroups()` and `disconnectGroups()`
- Ensure edges are properly added to groups' producer/consumer edges

### 5. Update Edge Cleanup Logic
- Modify `SegmentedFusion::removeEdge()` to handle shared ownership
- Update `SegmentedFusion::Impl::cleanUnused()` to properly handle shared pointers
- Ensure proper cleanup in SegmentedGroup destructor
- Remove edges from both groups they connect

### 6. Update Edge Serialization
- Review and update edge serialization code to handle shared pointers
- Update edge deserialization to properly reconstruct shared ownership

### 7. Testing and Validation
- Add tests to verify proper edge ownership and cleanup
- Verify no memory leaks with shared ownership
- Test edge sharing between groups works correctly
- Validate edge lifetime management in complex scenarios

### 8. Documentation Updates
- Update comments to reflect new ownership model
- Document any changes in behavior or requirements
- Update the design notes with implementation details

## Benefits
1. Clearer ownership semantics through shared_ptr
2. Reduced code complexity by eliminating redundant storage
3. More robust memory management
4. Better encapsulation of edge management within groups

## Potential Risks
1. Need to carefully manage circular references
2. Must ensure proper cleanup of shared resources
3. Performance impact of shared pointer overhead (likely minimal)
4. Migration complexity - need to update all edge-related code 