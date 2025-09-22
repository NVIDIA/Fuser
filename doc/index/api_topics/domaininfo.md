# DomainInfo

# this document produced by Claude @ 9-22-2025

## Synopsis

Information tracking struct that enables nvFuser's spanning tree algorithm to preserve maximum semantic information about tensor domain relationships during transform propagation across complex DAGs.

## Overview

DomainInfo serves as the core information payload for [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167), implementing Prim's algorithm with semantic information preservation rather than traditional cost minimization. It tracks domain mapping completeness across tensor transforms, bridges root and logical domain representations, and provides quantified metrics for algorithmic traversal decisions.

This enables transform-safe coordination across complex tensor DAGs where maintaining domain correspondence is critical for scheduling correctness. Unlike traditional graph algorithms that optimize for distance or cost, DomainInfo guides traversal toward paths that preserve maximum semantic information about tensor domain relationships.

**Key Capabilities:**
- Information-preserving graph traversal for optimal transform propagation paths
- Domain mapping completeness tracking to validate transform feasibility
- Root-logical domain semantic bridging for accurate axis correspondence
- Transform-safe DAG coordination ensuring scheduling consistency
- Semantic information quantification enabling algorithmic decision-making

## Source Location

**Primary Definition**: [`../../../csrc/scheduler/tools/maxinfo_propagator.h#L212`](../../../csrc/scheduler/tools/maxinfo_propagator.h#L212)

**Implementation**: [`../../../csrc/scheduler/tools/maxinfo_propagator.cpp#L170`](../../../csrc/scheduler/tools/maxinfo_propagator.cpp#L170)

**Context**: DomainInfo is a specialized Information struct within [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167), part of nvFuser's transform propagation infrastructure used throughout the scheduler system.

## Core Concepts

### Information-Preserving Graph Traversal

Transform propagation success depends on maintaining domain correspondence, not minimizing traversal cost. DomainInfo quantifies semantic relationships to guide Prim's algorithm toward information-rich paths rather than distance-optimal paths.

The `operator<()` comparison enables the spanning tree algorithm to select edges that preserve maximum domain mapping information:

```cpp
bool DomainInfo::operator<(const Information& r) const {
  auto rr = dynamic_cast<const DomainInfo&>(r);

  // More IDInfo entries = more preserved domains
  if (info.size() != rr.info.size()) {
    return info.size() < rr.info.size();
  }

  // More complete mappings = higher quality information
  size_t l_complete = std::count_if(info.begin(), info.end(),
                                   [](const IDInfo& i) { return i.is_complete; });
  size_t r_complete = std::count_if(rr.info.begin(), rr.info.end(),
                                   [](const IDInfo& i) { return i.is_complete; });
  return l_complete < r_complete;
}
```

### Domain Mapping Completeness Tracking

Tensor transforms can lose information - reductions eliminate dimensions, views may not preserve all relationships. DomainInfo tracks whether sufficient information exists to reconstruct reference tensor domain characteristics through transformation sequences.

Each IDInfo maintains completeness state and tracks which domains preserve reference information:

```cpp
struct IDInfo {
  std::unordered_set<[IterDomain](../../../csrc/ir/internal_base_nodes.h#L83)*> mapped_ids;  // Domain IDs preserving reference info
  bool is_complete;   // Can reference ID be fully reconstructed?
  bool is_logical;    // Are mapped_ids from root or logical domain?
};
```

During propagation, completeness can degrade when domain mappings fail:

```cpp
for (auto producer_id : producer_mapped_logical_ids) {
  auto it = p2c_map.find(producer_id);
  if (it != p2c_map.end()) {
    consumer_info.mapped_ids.insert(it->second);
  } else {
    consumer_info.is_complete = false;  // Mapping failure degrades completeness
  }
}
```

### Root-Logical Domain Semantic Bridge

Transform propagation requires mapping between domain types - root domains contain "raw" untransformed structure while logical domains contain semantically aligned, transformation-ready structure. DomainInfo mediates between these representations for accurate correspondence.

The `is_logical` flag distinguishes domain types, with P2C/C2P propagation handling transitions appropriately:

```cpp
// Domain type handling in P2C propagation
std::unordered_set<IterDomain*> producer_mapped_logical_ids;
if (producer->hasRoot() && !info.is_logical) {
  // Convert root domain info to logical domain for mapping
  producer_mapped_logical_ids = mapRootToLogical(
      producer, info.mapped_ids, propagate_through_resize_);
} else {
  producer_mapped_logical_ids = info.mapped_ids;
}

// Result stored in consumer's root domain (raw destination)
consumer_info.is_logical = false;
```

### Transform-Safe DAG Coordination

Tensor DAGs have complex producer-consumer-sibling relationships. Transforms must propagate in coordination-safe order to maintain mathematical correctness and avoid conflicts.

The standard coordination pattern ensures systematic propagation:

```cpp
// Apply transforms to reference tensor
tv3->split(1, 128);          // Split dimension: [M, N] -> [M, N/128, 128]
tv3->axis(0)->parallelize(ParallelType::BIDx);  // Block parallelism
tv3->axis(-1)->parallelize(ParallelType::TIDx); // Thread parallelism

// Build spanning tree for safe traversal order
[MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(tv3);

// Create propagator to apply transforms
[TransformPropagator](../../../csrc/transform_replay.h#L296) tp(tv3);

// Execute coordinated propagation across DAG
tree.traverse(&tp);  // Visits each tensor once in safe order

// Apply consistent parallelization
scheduler_utils::parallelizeAllLike(tv3);
```

### Semantic Information Quantification

Compilers need objective criteria for optimization decisions. DomainInfo provides concrete metrics comparing semantic value of different tensor domain mapping relationships, enabling automatic path selection in complex scenarios.

Information content is quantified by relationship count and quality metrics:

```cpp
// Algorithm selects path preserving more domain information
DomainInfo path_a, path_b;
// path_a.info.size() = 3, with 2 complete mappings
// path_b.info.size() = 2, with 2 complete mappings

if (path_a > path_b) {  // Uses operator< for comparison
  // Take path preserving more domain information
}
```

## Technical Implementation

### Core Data Structure

```cpp
struct DomainInfo : public Information {
  std::vector<IDInfo> info;  // One IDInfo per reference tensor root domain

  operator bool() const override {
    return !info.empty();  // Has domain information
  }

  bool operator<(const Information& r) const override;
};
```

**IDInfo Components:**
- `mapped_ids`: Set of [IterDomain](../../../csrc/ir/internal_base_nodes.h#L83)* preserving reference tensor information
- `is_complete`: Whether reference domain fully reconstructable from mapped_ids
- `is_logical`: Whether mapped_ids reference root (false) or logical (true) domains

**Memory Management**: Uses shared_ptr<Information> to avoid copying during algorithm execution; unordered_set provides O(1) ID lookup.

### Comparison Operations

The boolean conversion indicates whether domain information exists:

```cpp
DomainInfo::operator bool() const {
  return !info.empty();
}
```

Information comparison determines which path preserves more semantic information:

```cpp
bool DomainInfo::operator<(const Information& r) const {
  auto rr = dynamic_cast<const DomainInfo&>(r);

  // Primary criterion: More IDInfo entries = more preserved domains
  if (info.size() != rr.info.size()) {
    return info.size() < rr.info.size();
  }

  // Secondary criterion: More complete mappings = higher quality
  size_t l_complete = std::count_if(info.begin(), info.end(),
                                   [](const IDInfo& i) { return i.is_complete; });
  size_t r_complete = std::count_if(rr.info.begin(), rr.info.end(),
                                   [](const IDInfo& i) { return i.is_complete; });
  return l_complete < r_complete;
}
```

### Domain Mapping Utilities

Root to logical domain mapping handles transformation paths:

```cpp
std::unordered_set<[IterDomain](../../../csrc/ir/internal_base_nodes.h#L83)*> mapRootToLogical(
    [TensorView](../../../csrc/ir/interface_nodes.h#L383)* tv,
    const std::unordered_set<[IterDomain](../../../csrc/ir/internal_base_nodes.h#L83)*>& root_ids,
    bool propagate_through_resize) {

  std::unordered_set<[IterDomain](../../../csrc/ir/internal_base_nodes.h#L83)*> mapped_logical_ids;
  const auto& logical_dom = tv->getLogicalDomain();

  for (auto id : logical_dom) {
    // Direct mapping if ID exists in both domains
    if (root_ids.count(id) > 0) {
      mapped_logical_ids.emplace(id);
      continue;
    }

    // Check transformation path from root to logical
    for (auto root_id : root_ids) {
      auto exprs = DependencyCheck::getAllExprsBetween({root_id}, {id});
      if (!exprs.empty() &&
          (propagate_through_resize ||
           std::none_of(exprs.begin(), exprs.end(),
                       [](Expr* expr) { return expr->isA<Resize>(); }))) {
        mapped_logical_ids.emplace(id);
        break;
      }
    }
  }
  return mapped_logical_ids;
}
```

## Usage Patterns

### Standard Transform Propagation

The fundamental pattern for coordinated transform application:

```cpp
void applyTransformPropagation([TensorView](../../../csrc/ir/interface_nodes.h#L383)* reference_tv) {
  // Apply desired transforms to reference tensor
  reference_tv->split(1, 128);  // Create tiling structure
  reference_tv->axis(0)->parallelize(ParallelType::BIDx);
  reference_tv->axis(-1)->parallelize(ParallelType::TIDx);

  // Create spanning tree for information-preserving traversal
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(reference_tv);

  // Create transform propagator
  [TransformPropagator](../../../csrc/transform_replay.h#L296) propagator(reference_tv);

  // Execute coordinated propagation
  tree.traverse(&propagator);

  // Apply parallel patterns consistently
  scheduler_utils::parallelizeAllLike(reference_tv);
}
```

### Scheduler Integration

DomainInfo enables consistent transform propagation across all scheduler types:

```cpp
// Usage in PointWise Scheduler
void PointWiseScheduler::schedule(Fusion* fusion) {
  auto reference_tv = selectReferenceTV(fusion);

  // Apply pointwise-specific transforms
  reference_tv->split(1, 128);
  reference_tv->split(0, 4);

  // Use DomainInfo-based propagation for consistency
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(reference_tv);
  [TransformPropagator](../../../csrc/transform_replay.h#L296) tp(reference_tv);
  tree.traverse(&tp);

  applyPointwiseOptimizations(fusion);
}
```

### Python Frontend Access

Manual scheduling through Python bindings supports both full and selective propagation:

```cpp
if (selected_tensors.empty()) {
  // Propagate to entire fusion
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167)(reference_tv).traverse(&propagator);
} else {
  // Propagate to selected subset
  SetSelector selector({selected_tv_set.begin(), selected_tv_set.end()});
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167)(reference_tv, &selector)
      .traverse(&propagator);
}
```

## Code Examples

### Basic Usage Example

```cpp
#include <scheduler/tools/maxinfo_propagator.h>
#include <transform_propagator.h>

void basicTransformPropagation() {
  // Create simple DAG: tv0 -> tv1 -> tv2 -> tv3
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = TensorViewBuilder().ndims(2).build();
  auto tv1 = add(tv0, tv0);
  auto tv2 = relu(tv1);
  auto tv3 = sum(tv2, {1});

  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  // Apply transforms to reference tensor
  tv3->split(0, 128);  // Split outer dimension

  // Create spanning tree for information-preserving traversal
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(tv3);

  // Create propagator to apply transforms
  [TransformPropagator](../../../csrc/transform_replay.h#L296) tp(tv3);

  // Execute propagation - applies split(0,128) to tv0,tv1,tv2
  // where domains correspond to tv3's reference domain
  tree.traverse(&tp);
}
```

### Manual Scheduling Example

Complete manual scheduling workflow from the experimenting folder:

```cpp
void manualSchedulingExample() {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Build tensor computation DAG
  auto tv0 = TensorViewBuilder()
                 .ndims(2)
                 .dtype(DataType::Float)
                 .build();
  fusion->addInput(tv0);

  auto tv1 = add(tv0, tv0);        // Element-wise addition
  auto tv2 = sin(tv1);             // Element-wise sine
  auto tv3 = add(tv2, tv2);        // Final element-wise addition

  fusion->addOutput(tv3);

  // Manual scheduling on reference tensor tv3
  {
    // Apply tiling transform: [M, N] -> [M, N/128, 128]
    tv3->split(1, 128);

    // Apply parallelization strategy
    tv3->axis(0)->parallelize(ParallelType::BIDx);   // Block dimension
    tv3->axis(-1)->parallelize(ParallelType::TIDx);  // Thread dimension

    // Create spanning tree for transform propagation
    [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(tv3);

    // Create propagator applying tv3's transforms to other tensors
    [TransformPropagator](../../../csrc/transform_replay.h#L296) tp(tv3);

    // Execute propagation - visits tv0, tv1, tv2 with corresponding transforms
    tree.traverse(&tp);

    // Apply parallelization patterns consistently across DAG
    scheduler_utils::parallelizeAllLike(tv3);
  }

  // Compilation and execution
  FusionExecutorCache fec(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({1024, 2048}, options);
  auto outputs = fec.runFusionWithInputs({input});
}
```

### Debug and Analysis Example

Custom propagator for inspecting traversal decisions:

```cpp
void debugDomainInfo() {
  [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(reference_tv);

  // Custom debug propagator to inspect traversal
  struct DebugPropagator : public [MaxInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L44)::Propagator {
    void propagateC2P([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
      std::cout << "C2P: " << from->toString() << " -> " << to->toString() << "\\n";

      // Inspect domain info at each step
      auto info = [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167)::getReferenceIDInfo(to);
      std::cout << "  Domain info entries: " << info->info.size() << "\\n";

      for (const auto& id_info : info->info) {
        std::cout << "    Mapped IDs: " << id_info.mapped_ids.size()
                  << ", Complete: " << id_info.is_complete
                  << ", Logical: " << id_info.is_logical << "\\n";
      }
    }

    void propagateP2C([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
      std::cout << "P2C: " << from->toString() << " -> " << to->toString() << "\\n";
    }

    void propagateSibling([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
      std::cout << "Sibling: " << from->toString() << " -> " << to->toString() << "\\n";
    }
  };

  DebugPropagator debug_prop;
  tree.traverse(&debug_prop);
}
```

## Advanced Topics

### Performance Characteristics

- **Algorithm Complexity**: O(E log V) for spanning tree construction where E=DAG edges, V=tensor vertices
- **Memory Usage**: O(D×R) where D=number of domains, R=number of reference tensor root domains
- **Scalability**: Efficient for typical fusion DAGs (10-100 tensors); large DAGs (1000+ tensors) may need optimization

### Error Handling

Completeness validation and resize operation handling:

```cpp
void validateDomainCompleteness(const DomainInfo& info) {
  for (const auto& id_info : info.info) {
    if (!id_info.is_complete) {
      NVF_WARNING("Incomplete domain mapping detected - some transforms may fail");
    }

    if (id_info.mapped_ids.empty()) {
      NVF_ERROR("Empty domain mapping - transform propagation will fail");
    }
  }
}

// Resize operation handling
[MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(
    reference_tv,
    nullptr,  // default selector
    false     // don't propagate through resize for inlining
);
```

### Custom Selector Implementation

Extending traversal behavior for specialized use cases:

```cpp
class TypeSelector : public [MaxInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L44)::Selector {
  DataType target_type_;
public:
  TypeSelector(DataType type) : target_type_(type) {}

  bool allowC2P([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
    return to->dtype() == target_type_;
  }

  bool allowP2C([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
    return to->dtype() == target_type_;
  }

  bool allowSibling([TensorView](../../../csrc/ir/interface_nodes.h#L383)* from, [TensorView](../../../csrc/ir/interface_nodes.h#L383)* to) override {
    return true;  // Always allow sibling propagation
  }
};

// Usage
TypeSelector float_only(DataType::Float);
[MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167) tree(reference_tv, &float_only);
```

## Related Components

### [MaxLogicalDomainInfoSpanningTree](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167)
**Relationship**: DomainInfo serves as the Information payload for this spanning tree implementation
**Location**: [`../../../csrc/scheduler/tools/maxinfo_propagator.h#L167`](../../../csrc/scheduler/tools/maxinfo_propagator.h#L167)
**Algorithm**: Implements Prim's algorithm using DomainInfo comparison for maximum information preservation

### [TransformPropagator](../../../csrc/transform_replay.h#L296) Integration
**Relationship**: Visitor pattern collaboration - spanning tree provides traversal order, propagator applies transforms
**Usage**: `tree.traverse(&propagator)` executes coordinated transform application

### Scheduler System Architecture
**Integration Points**:
- **PointWise Scheduler**: [`../../../csrc/scheduler/pointwise.cpp`](../../../csrc/scheduler/pointwise.cpp)
- **Reduction Scheduler**: [`../../../csrc/scheduler/reduction_utils.cpp`](../../../csrc/scheduler/reduction_utils.cpp)
- **Transpose Scheduler**: [`../../../csrc/scheduler/transpose.cpp`](../../../csrc/scheduler/transpose.cpp)
- **Python Frontend**: [`../../../python/python_frontend/schedule_bindings.cpp`](../../../python/python_frontend/schedule_bindings.cpp)

**Role**: DomainInfo enables consistent transform propagation across all scheduler types, maintaining mathematical correctness in complex DAG scenarios.
