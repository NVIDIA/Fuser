<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Design Document: Component IterDomain Tracking for RaggedIterDomain

## Problem Statement

When calling `RaggedIterDomain::combine(component, ragged)`, we need to validate that the `component` IterDomain is the correct one that was originally paired with `ragged` during the Partition operation that created it.

### The Challenge

The naive approach of checking `ragged->definition()` for a Partition expression fails because:

1. **Tensor-level operations break the definition chain**: Operations like `set()` create new TensorViews with new IterDomains
2. **IterDomains are propagated without definitions**: The new IterDomains are clones/descendants but don't have the original Partition as their definition
3. **The pairing information is lost**: After propagation, there's no explicit link between the ragged IterDomain and its paired component

### Concrete Example

```cpp
// tv0: [i0] - regular tensor
auto result = asNested(tv0, 0, extents);  // Creates Partition expression
// result.ragged: RaggedIterDomain with Partition definition
// result.component: Component IterDomain

auto tv1 = set(tv0);  // Propagates IterDomains
// tv1 has a RaggedIterDomain, but it's a clone without Partition definition

combine(result.component, tv1->getRaggedDomain());  // How do we validate?
```

## Design Alternatives

### Option 1: Store Component Pointer in RaggedIterDomain

**Approach**: Add a `component_` member variable to `RaggedIterDomain` that points to the paired component IterDomain.

**Implementation**:
```cpp
class RaggedIterDomain : public IterDomain {
 private:
  TensorView* extents_ = nullptr;
  IterDomain* component_ = nullptr;  // NEW: paired component

 public:
  IterDomain* component() const { return component_; }
  void setComponent(IterDomain* component) { component_ = component; }
};
```

**How It Works**:
1. When `Partition` creates a RaggedIterDomain, it sets the component pointer
2. When IterDomains are cloned (e.g., during `set()`), the component pointer is cloned/mapped too
3. In `combine()`, validate that the provided component matches `ragged->component()`

**Pros**:
- ✅ Simple and direct solution
- ✅ Component pointer is automatically preserved during cloning
- ✅ Fast O(1) lookup - no graph traversal needed
- ✅ Follows existing pattern (similar to how `extents_` is stored)
- ✅ Self-documenting - makes the pairing explicit in the data structure

**Cons**:
- ❌ **CRITICAL: Dependency ordering not guaranteed** - Since `component_` is not an input to RaggedIterDomain's definition, IR graph traversal (during lowering, cloning, replay) has no guarantee that the component IterDomain will be visited/cloned before the ragged IterDomain. This can lead to:
  - Dangling pointers during cloning (trying to remap component before it's cloned)
  - Incorrect mappings in IrCloner when component hasn't been processed yet
  - Failures in topological traversal algorithms that expect dependencies to be explicit
- ❌ **Fragile during replacements** - When IterDomains are replaced (e.g., via `replaceAllUsesWith()`), the component pointer in ragged doesn't get updated automatically. Would require special-case handling throughout the codebase to maintain this hidden dependency.
- ❌ **Strong implicit coupling** - Creates a dependency that's not reflected in the IR graph structure, making the IR harder to reason about and maintain. Optimization passes and transformations that don't know about this hidden link could break the invariant.
- ❌ Component pointer could become stale if component IterDomain is replaced/transformed

**Why This Is Problematic**:

The fundamental issue is that this approach tries to store a relationship that *should* be part of the IR graph structure as an *out-of-band* pointer. nvFuser's IR infrastructure is designed around explicit dependency edges (inputs/outputs of expressions). Adding a pointer that doesn't follow these edges creates a parallel tracking mechanism that must be manually maintained across all IR operations:

1. **IrCloner** would need to special-case the component pointer remapping, but it can't guarantee ordering
2. **replaceAllUsesWith()** and similar operations wouldn't know to update the component pointer
3. **Replay** operations that transform IterDomains wouldn't propagate the component link correctly
4. **Serialization/deserialization** would need special handling for this out-of-band pointer

---

### Option 2: Traverse IR Graph to Find Original Partition

**Approach**: Walk backward through the IterDomain definition chain to find the original Partition expression.

**Implementation**:
```cpp
IterDomain* findOriginalComponent(RaggedIterDomain* ragged) {
  // Traverse backward through set operations, clones, etc.
  auto* current = ragged;
  while (current != nullptr) {
    if (current->definition() && current->definition()->isA<Partition>()) {
      return current->definition()->as<Partition>()->component();
    }
    // Follow the chain backward (e.g., through set operations)
    current = getSourceIterDomain(current);
  }
  return nullptr;  // No Partition found
}
```

**How It Works**:
1. Start from the given RaggedIterDomain
2. Traverse backward through the IR graph following definition chains
3. Find the original Partition expression
4. Extract the component from that Partition

**Pros**:
- ✅ No additional memory overhead
- ✅ No new state to maintain
- ✅ Always finds the "true" original component by traversing the IR
- ✅ Component pointer can't become stale (computed on demand)

**Cons**:
- ❌ **CRITICAL: Fusion segmentation breaks traversal** - When a fusion is segmented (split into multiple kernels), each segment contains only a subset of the full IR graph. A segment may contain a RaggedIterDomain that needs to be combined, but the original Partition expression that created it may be in a different segment. Traversal cannot cross segment boundaries, making it impossible to find the original component.
- ❌ **CRITICAL: External ragged tensors have no Partition** - When RaggedIterDomain support is extended in the future to accept ragged tensors from PyTorch as fusion inputs, these would arrive as RaggedIterDomains without any Partition expression in the nvFuser IR. There would be nothing to traverse back to, yet we still need to know the component for validation.
- ❌ **Unreliable chain traversal** - Even when a Partition exists in the same segment, the definition chain can be broken or complex:
  - Operations like `set()` intentionally break the definition chain
  - Multiple paths back through different transformations
  - Split/merge operations on the path complicate tracking
- ❌ Requires IR graph traversal - O(n) where n is chain depth
- ❌ Complex implementation - need to handle all propagation patterns (set, replay, clone, etc.)
- ❌ Performance cost on every combine() call

**Why This Is Problematic**:

This approach assumes the Partition expression is always reachable, but there are fundamental scenarios where it isn't:

1. **Segmented Fusions**: nvFuser segments complex fusions into multiple kernels. Each segment is scheduled and lowered independently. A RaggedIterDomain in segment N may have been created by a Partition in segment M, but segment boundaries are opaque - you can't traverse across them.

2. **Future External Inputs**: When RaggedIterDomain support is extended to accept ragged tensors from PyTorch as fusion inputs, these RaggedIterDomains will have no corresponding nvFuser Partition expression. They represent already-partitioned data from outside nvFuser.

3. **Definition Chain Breaks**: Even within a segment, operations like `set()` intentionally create new IterDomains without definitions, breaking the chain.

The fundamental flaw is assuming component information can be recovered from the IR graph structure, when in reality the information may not exist in the graph at all.

---

### Option 3: Track Component in Partition Expression Only

**Approach**: Only validate when a direct Partition definition exists, otherwise trust the user.

**Implementation**:
```cpp
void combine(IterDomain* component, RaggedIterDomain* ragged) {
  // Only validate if we can find a Partition
  if (ragged->definition() && ragged->definition()->isA<Partition>()) {
    auto* partition = ragged->definition()->as<Partition>();
    NVF_ERROR(component == partition->component(),
              "Component doesn't match partition");
  }
  // Otherwise, no validation - trust the user

  // Proceed with combine...
}
```

**How It Works**:
1. Check if ragged has a Partition definition
2. If yes, validate the component
3. If no, skip validation and trust the user provided the correct component

**Pros**:
- ✅ Minimal implementation - no new infrastructure
- ✅ No memory overhead
- ✅ Simple to understand
- ✅ Validation when possible, permissive when not

**Cons**:
- ⚠️ Validation is incomplete - only validates when Partition definition is directly available
- ⚠️ After propagation operations (set, segmentation), relies on user correctness

**Why This Is Actually Reasonable**:

This approach aligns with how nvFuser handles other operations:
- **Arithmetic operations** (add, mul, etc.) assume inputs have matching shapes - they don't validate
- **User responsibility**: If users call `combine(component, ragged)`, we trust they're providing the correct component
- **Validation where possible**: When we CAN validate (Partition definition exists), we do
- **Fail-fast when detectable**: Catches obvious errors early in the fusion definition
- **Pragmatic**: Acknowledges that complete validation isn't feasible given segmentation and external inputs

The key insight is that `combine()` is a user-facing API. Users are expected to know which component pairs with which ragged domain, just as they're expected to know when tensor shapes are compatible for arithmetic operations.

---

### Option 4: Store Component Pairing in TensorDomain

**Approach**: Store component-ragged pairings in TensorDomain rather than in RaggedIterDomain itself.

**Implementation**:
```cpp
// In TensorDomain
class TensorDomain {
 private:
  std::vector<IterDomain*> logical_domain_;
  // Other domain vectors...

  // NEW: Track ragged-component pairings for IterDomains in this TensorDomain
  struct RaggedComponentPair {
    RaggedIterDomain* ragged;
    IterDomain* component;
  };
  std::vector<RaggedComponentPair> ragged_component_pairs_;

 public:
  // Get the component for a ragged IterDomain in this TensorDomain
  IterDomain* getComponentFor(RaggedIterDomain* ragged) const;

  // Register a ragged-component pairing (called when creating from Partition)
  void registerRaggedComponentPair(RaggedIterDomain* ragged, IterDomain* component);
};
```

**How It Works**:
1. When Partition creates a TensorView with ragged and component IterDomains, register the pairing in the TensorDomain
2. The pairing is stored alongside the IterDomains themselves, ensuring both ragged and component are in `allIds()`
3. When tensor operations (like `set()`) propagate TensorDomains, they also propagate the pairing information
4. In `combine()`, look up the component from the TensorView's TensorDomain

**Pros**:
- ✅ **Looser coupling**: The relationship is stored in TensorDomain, not in RaggedIterDomain itself
- ✅ **Follows containment**: TensorDomain already owns and manages its IterDomains, so it's natural to manage their relationships
- ✅ **Explicit in domain operations**: Operations that propagate TensorDomain can explicitly propagate pairings
- ✅ **Validates across propagation**: Works even after `set()` if the pairing is propagated correctly
- ✅ **Both IDs guaranteed present**: Since both must be in `allIds()`, dependency ordering is less problematic

**Cons**:
- ❌ **Propagation must be explicit**: Every operation that creates/clones TensorDomain must handle pairing propagation
- ❌ **More complex than Option 3**: Requires changes to TensorDomain and all operations that manipulate it
- ❌ **Still has propagation challenges**: Operations like replay, resize, or transformations need to update pairings
- ❌ **Segmentation issues remain**: After fusion segmentation, TensorDomain in one segment may not have the original pairing information

**Key Challenge**:

The main implementation challenge is ensuring pairing propagation through all tensor operations:
- `set()`: Must copy pairings from input TensorDomain to output
- `view/reshape`: Must map pairings through transformations
- Replay operations: Must track how ragged and component are transformed
- Cloning: Must clone pairings along with IterDomains

**Why This Is Better Than Option 1**:

Unlike storing the pointer in RaggedIterDomain:
- TensorDomain already manages relationships between IterDomains (root→logical→allocation mappings)
- Both ragged and component are explicitly part of the domain, reducing implicit dependencies
- The coupling is at the TensorDomain level, not at the individual IterDomain level

**Why This May Not Be Worth It**:

While architecturally cleaner than Option 1, it's still significantly more complex than Option 3:
- Requires modifying TensorDomain and many tensor operations
- Still doesn't solve segmentation (segments may not preserve original TensorDomain)
- Adds complexity for validation that may not be critical (users can track pairings)

If Option 3's "trust the user" approach is sufficient, Option 4's additional complexity may not be justified.

---

## Analysis Summary

### Why Options 1 & 2 Are Not Viable

**Option 1 (Stored Pointer)**: Fundamentally flawed due to dependency ordering. The component pointer would be an out-of-band dependency not reflected in the IR graph. IR traversal algorithms follow explicit input/output edges, with no guarantee that component will be processed before ragged during cloning/lowering/replay. Violates nvFuser's design principle of explicit dependency edges.

**Option 2 (IR Traversal)**: Fails in two critical scenarios:
1. **Fusion Segmentation**: Partition expression may be in a different segment, unreachable via traversal
2. **Future External Inputs**: When RaggedIterDomain support is extended to accept ragged tensors from PyTorch as fusion inputs, these will have no nvFuser Partition expression to traverse to

These aren't edge cases - they're fundamental use cases that must be supported.

### Why Option 3 Is The Pragmatic Choice

**Option 3** aligns with nvFuser's design philosophy: like arithmetic operations that assume shape compatibility, `combine()` trusts users to provide correct inputs. It validates when Partition definition exists but otherwise relies on user correctness. Simple to implement, handles all use cases (propagation, segmentation, external inputs), and acknowledges that complete validation is architecturally infeasible.

**Option 4 (TensorDomain Pairing)** is architecturally cleaner than Option 1 (looser coupling) but requires extensive changes to TensorDomain operations and still has segmentation issues. Could be a future enhancement if user errors become problematic, but Option 3's simplicity is preferred for now.

## Recommendation

### Proposed Solution: **Option 3 - Validate When Partition Definition Exists**

**This is the current design choice.** We will reconsider Option 4 (TensorDomain Pairing) if it proves more appropriate based on practical experience or future requirements.

**Rationale**:

Option 3 is the most reasonable approach because it:

1. **Aligns with nvFuser's design philosophy**: Like arithmetic operations that assume shape compatibility, `combine()` trusts users to provide correct inputs
2. **Provides validation where feasible**: When a Partition definition is directly accessible, we validate the component
3. **Simple and maintainable**: No complex infrastructure, no global state, no dependency ordering issues
4. **Handles all use cases**: Works for direct Partition usage, propagated domains, segmented fusions, and future external inputs
5. **Pragmatic**: Acknowledges that complete validation is architecturally infeasible

**Implementation**:

```cpp
void combine(IterDomain* component, RaggedIterDomain* ragged) {
  // Basic validation (null checks, type checks, etc.)
  NVF_ERROR(component != nullptr && ragged != nullptr, "Null inputs");
  NVF_ERROR(!component->isRaggedDomain(), "Component must be regular IterDomain");

  // Validate against Partition definition if available
  if (ragged->definition() && ragged->definition()->isA<Partition>()) {
    auto* partition = ragged->definition()->as<Partition>();
    NVF_ERROR(
        component == partition->component(),
        "Component mismatch: provided ", component->toString(),
        " but Partition expects ", partition->component()->toString());
  }

  // If no Partition definition (after set, in segmented fusion, or external input),
  // trust the user and proceed

  // Create combined IterDomain...
}
```

**What This Means**:

- ✅ Early error detection when Partition definition is available
- ✅ No architectural violations or fragile infrastructure
- ✅ Users are responsible for correct usage (like other operations)
- ✅ Works across all scenarios (propagation, segmentation, external inputs)
- ⚠️ After propagation/segmentation, incorrect usage won't be caught by validation
- ⚠️ Users must track component-ragged pairings themselves

**Comparison to Other Operations**:

This is consistent with how nvFuser handles other operations:
- `add(tv1, tv2)` doesn't validate that shapes match - user responsibility
- `set(tv)` doesn't validate all properties - user responsibility
- `combine(component, ragged)` doesn't always validate pairing - user responsibility

## Implementation Notes

1. **Testing Strategy**:
   - Test validation when Partition definition exists (should catch errors)
   - Test that validation is skipped after `set()` operations (should succeed with correct usage)
   - Document user responsibility in API documentation

2. **Future Considerations**:
   - Option 4 (TensorDomain Pairing) remains a viable alternative if the current approach proves insufficient
   - We will reconsider Option 4 based on practical experience, user feedback, or new requirements
   - If incorrect `combine()` usage becomes a common source of bugs, we can implement Option 4's more comprehensive validation
   - For now, follow the principle of trusting user-facing APIs
   - The `extents_` pointer handling may also need similar considerations in the future

3. **Documentation**:
   - Clearly document that users must provide the correct component that was paired with the ragged domain
   - Note that validation is best-effort and may not catch all errors
   - Provide examples of correct usage patterns
