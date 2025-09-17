# ComputeAtMap

Source: [ComputeAtMap](../../csrc/compute_at_map.h#L182)

## Synopsis
- **Kind**: class
- **File**: `csrc/compute_at_map.h`
- **What it represents**: A global mapping structure over `IterDomain*` that captures equivalence classes of axes under various mapping modes (EXACT, PERMISSIVE, LOOP, PERMISSIVE_RESIZE, INNERMOST, AlmostExact) to support scheduling, parallel propagation, and indexing.

## Purpose
- Provides consistent axis mapping across producer/consumer tensors for scheduling and lowering, especially for compute-at/compute-with reasoning.
- Propagates parallelization constraints, pre-allocates loop index variables, and offers queries to fetch concrete mapped IDs.

## Mapping modes (from code)
- `EXACT`: strict, no broadcast forwarding; parameters must match
- `AlmostExact`: forward through broadcasts but not to non-broadcast; allow split(…,1)
- `PERMISSIVE`: forward broadcasts and map all IDs; include root mappings
- `PERMISSIVE_RESIZE`: like PERMISSIVE but includes resize/gather/scatter adjacency for propagation
- `INNERMOST`: builds on PERMISSIVE_RESIZE, maps through to inner domains (e.g., transpose scheduler)
- `LOOP`: leaf axes to left of compute-at; preserves parallel strategies; excludes axes outside producer’s compute-at

## Key APIs
- `areMapped(id0, id1, mode)`: query axis equivalence
- `getConcreteMappedID(id, mode)`: representative axis for loop opening/indexing
- `validateAndPropagatePType()`: ensure consistent parallelization in LOOP map
- `allocateIndexVariables()`: pre-allocate loop indices per LOOP-set (and circular buffer variants)
- Sets and queries: `getIdSets(mode)`, `idExistsInMap`, `uniqueExactDefinitions(id)`, `uniqueExactUses(id)`

## Notes
- Internally builds an `IterDomainGraph` of disjoint sets for each mapping mode and consumers/producers relationships derived from the most permissive map.
- Works hand-in-glove with [BestEffortReplay](../../csrc/transform_iter.h#L318) and logical domain mapping to reconcile producer/consumer schedules.

## See also
- [IterDomainGraph](../../csrc/compute_at_map.h#L79), [IterDomain](../../csrc/ir/internal_base_nodes.h#L83)
- [TensorView](../../csrc/ir/interface_nodes.h#L383), [TensorDomain](../../csrc/ir/internal_base_nodes.h#L415)
