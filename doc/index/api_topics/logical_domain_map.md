# LogicalDomainMap and Subclasses

## Synopsis
Mapping utilities for relating logical iteration domains (`IterDomain`) across tensors. These maps drive producer→consumer alignment and computeAt analysis.

## Sources
- Interface: [`LogicalDomainMap`](../../../csrc/logical_domain_map.h#L20)
- Pairwise mapping: [`PairwiseLogicalDomainMap`](../../../csrc/logical_domain_map.h#L83)
- ComputeAt mapping: [`ComputeAtLogicalDomainMap`](../../../csrc/logical_domain_map.h#L270)
- Exact mapping: [`ExactLogicalDomainMap`](../../../csrc/logical_domain_map.h#L558)

## Overview
Logical-domain mapping answers: “Which axis of one tensor corresponds to which axis of another?” This is central to scheduling, broadcasting, and compute-at. The framework provides:
- Generic mapping interface (`LogicalDomainMap`) with producer→consumer and consumer→producer mapping helpers
- Pairwise mapping for a specific producer-consumer TV pair with configurable behavior (e.g., broadcast handling, symbolic extents, gather/indexed domains)
- ComputeAt mapping for building global equivalence classes that enable computeAt feasibility checks and mappable axis discovery
- Exact mapping building disjoint sets of axes that are exactly equivalent across a fusion (excludes broadcast↔non-broadcast)

## Key Classes
- `LogicalDomainMap`
  - `mapProducerToConsumer(producer_td, consumer_td, dims_to_map?)`
  - `mapConsumerToProducer(consumer_td, producer_td, dims_to_map?)`
- `PairwiseLogicalDomainMap(producer_tv, consumer_tv)`
  - Options: `mapBroadcast(bool)`, `mapSymbolic(bool)`, `mapDifferentExtents(bool)`, `mapIndexedDomains(bool)`
  - Convenience: `mapProducerToConsumer(...)`, `mapConsumerToProducer(...)`
- `ComputeAtLogicalDomainMap`
  - `build(map_through_reduction=false)`, `canMap(...)`, `setAlias(...)`, `mapBestEffort(...)`, `getMappableDims(...)`, `isConcretized(...)`
  - Uses `DomainKey` and a disjoint-set of `<TensorDomain, IterDomain>` pairs to record equivalences
- `ExactLogicalDomainMap`
  - Constructs disjoint sets of exactly-mapped `IterDomain*` across the whole fusion

## Example: Pairwise Mapping
```cpp
// Given producer and consumer TVs in a fusion
auto pwm = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
              .mapBroadcast(true)
              .mapSymbolic(false);
// Map producer logical domains to consumer logical domains
auto p2c = pwm.mapProducerToConsumer();
for (auto& [p_id, c_id] : p2c) {
  (void)p_id; (void)c_id; // axis correspondences
}
```

## Example: ComputeAt Mapping
```cpp
ComputeAtLogicalDomainMap ca_map;
ca_map.build();
// Check if two axes across TVs can be mapped for computeAt
bool ok = ca_map.canMap(tvA->domain(), tvA->axis(0), tvB->domain(), tvB->axis(0));
// Find all mappable dims between producer and consumer domains
auto mappable = ca_map.getMappableDims(tvP->domain(), tvC->domain());
```

## Additional Guidance
- Reductions: consumers of reduction outputs cannot be mapped inside the reduction loop; see `UnmappableReductionDomains` helper.
- Broadcasts: concrete/broadcast relationships are tracked so that mapping respects broadcast materialization/removal.
- Views: a view can map multiple domains to a single domain; best-effort mapping may not be one-to-one and can throw on ambiguities.

## Where to Look in the Codebase
- APIs and rich comments: [`csrc/logical_domain_map.h`](../../../csrc/logical_domain_map.h)
- Iteration traversal helpers: `iter_visitor.h`, back/forward visitors used by builders
- Scheduling/dataflow ops affecting mappings: `internal_nodes.h` (`BroadcastOp`, `SqueezeOp`, `ExpandOp`, `ViewOp`, `RepeatOp`, `SliceOp`, `PadOp`, `CatOp`, reductions)

## See Also
- `TensorDomain` / `IterDomain`: domain structures (`tensor_domain.md`, `iterdomain.md`)
- `Expr`: operations that produce/consume domains (`expr.md`)
