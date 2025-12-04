<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# RaggedIterDomain for Nested Tensors

---

## 1. Overview

This document proposes adding `RaggedIterDomain`, a new `IterDomain` subclass to nvFuser that represents dimensions with variable extents across batch components (ragged/jagged dimensions). This enables efficient compilation of PyTorch nested tensors and other variable-length data structures without padding overhead.

**Scope Note**: This proposal represents a **minimalistic initial version** containing only the capabilities that are absolutely necessary for expert parallelism. The exact requirements for expert parallelism are still being clarified with Jingyue. As those requirements become clear, additional capabilities (such as merge operations, specific IdModel integrations, or lowering support) may be added to this design. The features described here should be considered the bare minimum starting point.

---

## 2. Motivation

NvFuser currently lacks support for compiling operations on nested tensors. This design adds native ragged dimension support to nvFuser's IR.

---

## 3. Background

### PyTorch Nested Tensor Semantics

Please review the PyTorch [semantics](https://docs.pytorch.org/docs/stable/nested.html) of nested tensors.
PyTorch nested tensors represent collections of tensors with varying shapes along one dimension. For example, a batch of 3 sequences with lengths [3, 5, 2] is stored contiguously with offset-based indexing rather than padding to length 5.

```python
# Batch of 3 sequences with different lengths
t1 = torch.randn(3, 4)  # length 3
t2 = torch.randn(5, 4)  # length 5
t3 = torch.randn(2, 4)  # length 2

nt = torch.nested.nested_tensor([t1, t2, t3])
# Shape notation: [batch=3, ragged_dim=[3,5,2], feature=4]
```

The ragged dimension uses contiguous storage with offsets [0, 3, 8, 10] to locate each component.

**PyTorch Restriction**: PyTorch nested tensors currently support only one ragged dimension per tensor. nvFuser extends beyond this limitation to support:

1. **Multi-level nesting** (ragged within ragged): For example, tokens organized by expert, where each expert's tokens are further organized by source rank.
2. **Multiple independent ragged dimensions**: For example, a tensor with both a ragged expert dimension (different GPUs handle different numbers of experts) and a ragged tokens-per-expert dimension (each expert processes different numbers of tokens).

These extensions are necessary for expert parallelism with load balancing, where both the distribution of experts across devices and the distribution of tokens per expert are non-uniform.

### nvFuser IterDomain System

IterDomain is the fundamental abstraction in nvFuser representing a single axis/dimension of a tensor. Key properties include:
- **extent**: Size of the dimension
- **IterType**: Iteration, Reduction, Broadcast, etc.
- **ParallelType**: Mapping to GPU parallelism (threadIdx, blockIdx, etc.)

TensorView contains a TensorDomain which groups multiple IterDomains representing all axes. Transformations like split, merge, and reorder manipulate IterDomains to optimize execution.

### Terminology

- **Ragged (or jagged) dimension**: A dimension with variable extent across batch elements
- **Nested domains**: The individual IterDomains contained within a RaggedIterDomain, one per batch component
- **Component**: One element of the batch (e.g., one sequence in a batch of sequences)
- **Offset**: Starting position of each component in contiguous storage
- **Extent**: Sum of all nested domain extents (total extent)

### Expert Parallelism Use Case

Expert parallelism in Mixture-of-Experts (MoE) models requires complex multi-level nested structures. The key operations involve:

1. **Token Dispatch**: Route input tokens to different experts based on routing decisions
2. **Expert Distribution**: Distribute experts across multiple GPUs for parallel processing
3. **Load Balancing**: Handle non-uniform token and expert distributions

#### Single-Level Nesting Example

After routing, tokens are grouped by expert with variable counts:

```python
# GPU 0 processes experts 0, 1, 2
# Tokens per expert: [127, 0, 198]
tokens_by_expert = nested_tensor([
    tokens_for_expert_0,  # 127 tokens
    tokens_for_expert_1,  # 0 tokens (expert not used)
    tokens_for_expert_2   # 198 tokens
])
# Shape: [experts=3, tokens=[127,0,198], hidden=512]
```

This requires a single ragged dimension for tokens-per-expert.

#### Multi-Level Nesting Example

In distributed expert parallelism, tokens from each rank are routed to different experts on different GPUs. After `all_to_all` communication, GPU 0 receives tokens for its assigned experts (experts 0, 1, 2) from all ranks:

```python
# BEFORE all_to_all: Each rank has tokens for all experts (rank-first layout)
# Rank 0: [tokens_for_expert_0, tokens_for_expert_1, ..., tokens_for_expert_N]
# Rank 1: [tokens_for_expert_0, tokens_for_expert_1, ..., tokens_for_expert_N]

# AFTER all_to_all: GPU 0 receives tokens for experts 0-2 from all ranks (expert-first layout)
# Data arrives in rank-first order:
tokens_received = [
    tokens_expert_0_rank_0 || tokens_expert_1_rank_0 || tokens_expert_2_rank_0,  # From rank 0
    tokens_expert_0_rank_1 || tokens_expert_1_rank_1 || tokens_expert_2_rank_1   # From rank 1
]
# Token counts: Expert 0: [127, 98], Expert 1: [0, 45], Expert 2: [198, 156]

# The all_to_all shuffle transforms the data layout from rank-first to expert-first
# This is where we need the partition operation to reorganize:
tokens_by_expert = nested_tensor([
    nested_tensor([tokens_from_rank_0, tokens_from_rank_1]),  # Expert 0: [127, 98]
    nested_tensor([tokens_from_rank_0, tokens_from_rank_1]),  # Expert 1: [0, 45]
    nested_tensor([tokens_from_rank_0, tokens_from_rank_1])   # Expert 2: [198, 156]
])
# Shape: [experts=3, ranks=2, tokens=[[127,98],[0,45],[198,156]], hidden=512]
```

This requires **two-level nesting**: outer ragged dimension for experts, inner ragged dimension for tokens-per-rank-per-expert. The key insight is that `all_to_all` delivers data in rank-first order, but we need expert-first order for processing, requiring a layout transformation.

#### Multiple Independent Ragged Dimensions Example

For load balancing, different GPUs may handle different numbers of experts:

```python
# GPU 0 handles 5 experts (higher capacity)
# GPU 1 handles 3 experts (lower capacity)

# Shape: [experts=[5,3], tokens_per_expert=[127,0,198,64,89,103,45,201], hidden=512]
```

This requires **two independent ragged dimensions**:
1. Expert distribution across GPUs: `[5, 3]`
2. Token distribution per expert: `[127, 0, 198, 64, 89, 103, 45, 201]`

Both dimensions are ragged but independent (not nested within each other).

---

## 4. Goals and Non-Goals

### Goals

- Support ragged dimensions in fusion IR for expert parallelism
- Support multi-level nesting (ragged within ragged) for expert parallelism use cases
- Support multiple independent ragged dimensions per tensor for load balancing scenarios

### Non-Goals

- Inlining
- Extension of single-GPU schedulers for ragged dimensions

Inlining would be eventually necessary for lowering to CUDA code but should not be necessary for distributed device (multi-GPU) parallelization. Similarly, the schedulers will be left as is for now.

---

## 5. Requirements

### Functional Requirements

- **IR Representation**: Represent ragged dimensions with variable extents per batch component
- **Offset Computation**: Compute and provide access to offsets for indexing into contiguous storage
- **Operations Support**: Support creation and basic transformations on ragged dimensions (merge operation TBD based on expert parallelism requirements)
- **Code Generation**: Generate correct CUDA code with offset-based indexing for ragged iteration
- **Uniform Properties**: Enforce uniform execution properties (ParallelType, IterType) across components
- **Integration**: Work within existing TensorView/TensorDomain infrastructure

### Non-Functional Requirements

- **Backward Compatibility**: No breaking changes to existing IterDomain API or behavior
- **Maintainability**: Clear separation of ragged-specific logic from regular IterDomain code

---

## 6. Proposed Design

### 6.1 High-Level Architecture

`RaggedIterDomain` is a direct subclass of `IterDomain`, allowing it to be used anywhere an IterDomain is expected (in TensorDomain, transformations, etc.) while providing ragged-specific functionality.

```
Val
 └── IterDomain
      ├── (regular IterDomain instances)
      └── RaggedIterDomain (new subclass)
```

#### The Partition Operation: Core Primitive for Nested Tensors

The `partition` operation is the fundamental IterDomain-level transformation that creates ragged dimensions by splitting a regular IterDomain into non-uniform segments based on offsets.

**Operation Signature:**
```cpp
// Static method in IterDomain class
static std::pair<IterDomain*, RaggedIterDomain*> partition(
    IterDomain* in,           // Input IterDomain to partition
    TensorView* offsets       // Offset tensor defining partition boundaries
);
```

**Semantics:**

**Case 1: Input is regular IterDomain**
- Input: Regular IterDomain with total extent N
- Offsets: 1D TensorView with K+1 elements `[0, offset_1, ..., offset_K=N]`
- Output:
  - Batch IterDomain with extent K (number of partitions)
  - RaggedIterDomain with K nested domains (extents = differences between consecutive offsets)

**Case 2: Input is RaggedIterDomain**
- Input: RaggedIterDomain with M components (each with potentially different extents)
- Offsets: 2D TensorView with shape `[M, K+1]` where:
  - Outer dimension M corresponds to each component of the input RaggedIterDomain
  - Inner dimension K+1 defines partition boundaries for that component
  - Each row i contains offsets `[0, offset_1, ..., offset_K=extent[i]]`
- Output:
  - Batch IterDomain with extent K (number of partitions - uniform across all components)
  - RaggedIterDomain nested within RaggedIterDomain (2-level nesting)

**Key constraint for Case 2**: All components must be partitioned into the **same number K of partitions**, but the offset values can differ per component since each component has different extent

#### The Merge Operation: Inverse of Partition

The `merge` operation is the inverse of `partition`, collapsing a batch IterDomain and RaggedIterDomain pair back into a single regular IterDomain by concatenating all components.

**Operation Signature:**
```cpp
// Static method in IterDomain class
static IterDomain* merge(
    IterDomain* batch_id,         // Batch dimension (number of components)
    RaggedIterDomain* ragged_id   // Ragged dimension to merge
);
```

**Semantics:**
- Input: Batch IterDomain (extent K) and RaggedIterDomain with K nested domains
- Output: Regular IterDomain with extent = sum of all nested domain extents
- This concatenates all ragged components into a single contiguous dimension

**Example:**
```cpp
// Input: batch_id with extent 3, ragged_id with extents [127, 0, 198]
auto merged = IterDomain::merge(batch_id, ragged_id);
// Output: Regular IterDomain with extent 325 (= 127 + 0 + 198)
```

#### Key Usage Patterns

The `partition` and `merge` operations are inverse primitives for creating and flattening ragged structures:

**Example - Complete workflow:**
```cpp
// Start: Flattened tokens [325 tokens, 512 hidden_dim]
auto token_dim = IrBuilder::create<IterDomain>(0, 325);

// Forward: Partition into 3 experts with extents [127, 0, 198]
// offsets = [0, 127, 127, 325]
auto [expert_dim, tokens_per_expert] = IterDomain::partition(token_dim, offsets);
// expert_dim: IterDomain with extent 3
// tokens_per_expert: RaggedIterDomain with nested extents [127, 0, 198]

// ... process with ragged structure ...

// Reverse: Merge back to flattened form
auto merged_dim = IterDomain::merge(expert_dim, tokens_per_expert);
// merged_dim: Regular IterDomain with extent 325
```

**Usage patterns:**

1. **Representing inherent data structure**: Use `asNested` (which internally uses `partition`) to represent ragged dimensions that are intrinsic properties of the user data (e.g., variable tokens per expert)

2. **Scheduling decisions**: Use `partition` directly for multi-GPU distribution strategies where work is split non-uniformly across devices

3. **Merging back**: Use `merge` to convert ragged dimensions back to regular dimensions when needed (e.g., after expert processing, concatenate results back to merged form)

The `partition` and `merge` operations enable both single-level and multi-level nested structures, as shown in the examples below.

#### Tensor Domain Structure

A `TensorDomain` can contain a mix of regular `IterDomain` and `RaggedIterDomain` instances, representing tensors with both uniform and ragged dimensions. nvFuser supports:

1. **Multiple RaggedIterDomains per TensorDomain**: For load balancing scenarios
2. **Multi-level nesting**: RaggedIterDomain can contain other RaggedIterDomains as nested domains

**Example 1: Single ragged dimension**
```cpp
// Batch of sequences with variable lengths
auto batch = IrBuilder::create<IterDomain>(0, 3);
auto ragged_seq = IrBuilder::create<RaggedIterDomain>({seq0, seq1, seq2});
auto feature = IrBuilder::create<IterDomain>(0, 4);

auto tensor_domain = IrBuilder::create<TensorDomain>({batch, ragged_seq, feature});
```

**Example 2: Multiple independent ragged dimensions**
```cpp
// Expert parallelism with load balancing
// Different GPUs handle different numbers of experts (ragged)
// Each expert processes different numbers of tokens (ragged)

// Partition creates (batch_id, ragged_id) pairs
auto [gpu_id, ragged_experts] = IterDomain::partition(expert_dim, gpu_offsets);
// gpu_id: IterDomain with extent = number of GPUs
// ragged_experts: RaggedIterDomain with variable experts per GPU

auto [expert_id, ragged_tokens] = IterDomain::partition(token_dim, expert_offsets);
// expert_id: IterDomain with extent = number of experts
// ragged_tokens: RaggedIterDomain with variable tokens per expert

auto hidden = IrBuilder::create<IterDomain>(0, 512);

// TensorDomain must include BOTH the batch IterDomains AND the RaggedIterDomains
auto tensor_domain = IrBuilder::create<TensorDomain>(
    {gpu_id, ragged_experts, expert_id, ragged_tokens, hidden}
);
```

**Example 3: Multi-level nesting (ragged within ragged)**
```cpp
// Tokens organized by expert, each expert's tokens organized by source rank
auto token_dim = IrBuilder::create<IterDomain>(0, 325);  // Total flattened tokens

// First partition: by expert (1D offsets)
// expert_offsets: [0, 127, 127, 325] - 3 experts with [127, 0, 198] tokens
auto [expert_id, tokens_per_expert] = IterDomain::partition(token_dim, expert_offsets);
// expert_id: IterDomain with extent = 3 experts
// tokens_per_expert: RaggedIterDomain with variable tokens per expert [127, 0, 198]

// Second partition: each expert's tokens by rank (2D offsets)
// rank_offsets: 2D tensor with shape [3, 3] (3 experts, 2 ranks + 1)
// rank_offsets[0] = [0, 50, 127]   - Expert 0: 127 tokens split as [50, 77] across 2 ranks
// rank_offsets[1] = [0, 0, 0]      - Expert 1: 0 tokens split as [0, 0] across 2 ranks
// rank_offsets[2] = [0, 100, 198]  - Expert 2: 198 tokens split as [100, 98] across 2 ranks
auto [rank_id, tokens_per_expert_per_rank] = IterDomain::partition(tokens_per_expert, rank_offsets);
// rank_id: IterDomain with extent = 2 ranks
// tokens_per_expert_per_rank: RaggedIterDomain nested within RaggedIterDomain
// Structure: [[50, 77], [0, 0], [100, 98]]

auto hidden = IrBuilder::create<IterDomain>(0, 512);

// TensorDomain includes both batch IterDomains (expert_id, rank_id) and the nested ragged structure
auto tensor_domain = IrBuilder::create<TensorDomain>(
    {expert_id, rank_id, tokens_per_expert_per_rank, hidden}
);
```

The compilation pipeline detects ragged dimensions and routes to appropriate indexing and code generation paths.

### 6.2 Core Abstractions

#### RaggedIterDomain Class

```cpp
class RaggedIterDomain : public IterDomain {
 public:
  // Constructor
  RaggedIterDomain(IrBuilderPasskey, std::vector<IterDomain*> nested_domains);

  // Accessors
  const std::vector<IterDomain*>& nestedDomains() const;
  Val* extentForComponent(int64_t idx) const;
  Val* offsetForComponent(int64_t idx) const;

  // This overrides IterDomain::extent and returns the total extent
  Val* extent() const override;

  // This overrides IterDomain::parallelize and calls nested_domains[i]->parallelize(pt) for all nested domains
  void parallelize(ParallelType pt);

 private:
  std::vector<IterDomain*> nested_domains_;  // One per batch component
  std::vector<Val*> offsets_;                // Cumulative sum
};
```

#### Nested Domains

The `nested_domains_` vector contains one `IterDomain` per batch component. For a nested tensor with batch size N, there are N nested IterDomains, each with potentially different extent. This represents the ragged dimension structure: component i has extent `nested_domains_[i]->extent()`.

**Multi-level nesting**: The nested domains can themselves be `RaggedIterDomain` instances, enabling ragged-within-ragged structures. For example, tokens organized by expert (outer ragged dimension), where each expert's tokens are organized by source rank (inner ragged dimension).

#### Offset Computation

Offsets are computed as the cumulative sum of extents: `offsets_[i] = sum(extents[0..i-1])`. This enables efficient indexing into contiguous storage:
- Component 0 starts at offset 0
- Component 1 starts at offset = extent[0]
- Component 2 starts at offset = extent[0] + extent[1]
- And so on...

For a ragged dimension with extents [3, 5, 2], the offsets are [0, 3, 8, 10].

#### Extent Semantics

**extent()**: Overrides `IterDomain::extent()` to return the sum of all nested domain extents (total extent). This represents the total storage size needed for the ragged dimension when data is stored contiguously.

### 6.3 Property Uniformity

Certain IterDomain properties are allowed to be non-uniform across all nested domains, such as extents. In the initial version, properties that have no reason to be non-uniform are set to be uniform to simplify the overall design and implementation.

| Property | Uniformity | Rationale |
|----------|------------|-----------|
| **extent** | **VARIABLE** | Core ragged characteristic - each component has different length |
| **ParallelType** | **UNIFORM** | GPU execution model requires consistent thread mapping |
| **IterType** | **UNIFORM** | All components perform same operation (iteration/reduction) |
| **start** | **UNIFORM (=0)** | Simplifies offset computation; all components start at 0 |
| **is_rfactor_domain** | **UNIFORM** | Reduction transformation applies uniformly |

The following properties are out of scope of this initial buildout:
- `is_padded_dimension`
- `is_clustered_dimension`
- `padded_to_size`

The constructor validates that:
1. Uniform properties (ParallelType, IterType, start, is_rfactor_domain) are consistent across all nested domains
2. Out-of-scope properties are not set (must be false/nullptr for all nested domains)

If any validation fails, an error is thrown.

### 6.4 Key Operations

#### Creation and Nesting

Create a RaggedIterDomain using the static factory method:
```cpp
auto id0 = IrBuilder::create<IterDomain>(0, 3);  // extent 3
auto id1 = IrBuilder::create<IterDomain>(0, 5);  // extent 5
auto id2 = IrBuilder::create<IterDomain>(0, 2);  // extent 2

auto ragged = IrBuilder::create<RaggedIterDomain>({id0, id1, id2});
// Creates ragged dimension with extents [3, 5, 2]
```

Create a TensorView with a ragged dimension:
```cpp
auto batch = IrBuilder::create<IterDomain>(0, 3);  // batch dimension
auto nested_tensor_domain = IrBuilder::create<TensorDomain>({batch, ragged});
```

**Creating Nested Tensors from Data and Offsets: asNested**

To create a nested tensor from separate data and offset tensors (similar to PyTorch's `torch.nested.nested_tensor_from_jagged`), use the `asNested` operation:

```cpp
// In csrc/ops/alias.h
NVF_API TensorView* asNested(
    TensorView* data,      // Data tensor with contiguous ragged storage
    TensorView* offsets,   // Offset tensor [num_components + 1]
    int64_t ragged_dim     // Which dimension of data is ragged
);
```

**Usage Example**:
```cpp
// Input tensors
auto data_tv = ...; // [325, 512] - flattened ragged dimension
auto offsets_tv = ...; // [4] - offsets [0, 127, 127, 325]

// Create nested tensor
auto nested_tv = asNested(data_tv, offsets_tv, /*ragged_dim=*/0);
// Result: TensorView with shape [batch=3, ragged=[127,0,198], hidden=512]
```

**Semantics**:
- `asNested` is a tensor-level operation (like `reshape`) that creates an output tensor with a ragged dimension
- Internally, it uses `partition` as a transform operation between the root and logical domains
- Similar to how `reshape` uses splits/merges between root and logical domains
- The offset tensor provides the boundaries for each component (see Section 6.7 for offset tensor format)
- This is not a view operation; it involves actual data transformation

**Merging Nested Tensors: asFlattened**

To convert a nested tensor back to a regular flattened tensor, use the `asFlattened` operation:

```cpp
// In csrc/ops/alias.h
NVF_API TensorView* asFlattened(
    TensorView* nested_tensor,    // Nested tensor to flatten
    int64_t batch_dim,            // Batch dimension index
    int64_t ragged_dim            // Ragged dimension index to flatten
);
```

**Usage Example**:
```cpp
// Input: Nested tensor [batch=3, ragged=[127,0,198], hidden=512]
auto nested_tv = ...;

// Flatten back to regular tensor
auto flattened_tv = asFlattened(nested_tv, /*batch_dim=*/0, /*ragged_dim=*/1);
// Result: TensorView with shape [token=325, hidden=512]
```

**Semantics**:
- `asFlattened` is a tensor-level operation that flattens a ragged dimension back to regular
- Internally, it uses `merge` as a transform operation between the root and logical domains
- The batch and ragged dimensions are merged into a single regular dimension
- This is the inverse of `asNested`

**Typical Usage Pattern**:
```cpp
// Expert parallelism workflow:
// 1. Start with flattened tokens
auto tokens = ...; // [325, 512]

// 2. Create nested structure for expert processing
auto nested = asNested(tokens, expert_offsets, /*ragged_dim=*/0);
// [expert=3, tokens_per_expert=[127,0,198], hidden=512]

// 3. Process with experts (some operations on nested tensor)
auto processed = expert_processing(nested);

// 4. Flatten back to regular tensor
auto result = asFlattened(processed, /*batch_dim=*/0, /*ragged_dim=*/1);
// [325, 512]
```

#### Transformations

**Split**: Split a regular IterDomain and merge with a RaggedIterDomain to create a new ragged structure.

```cpp
auto split_result = IterDomain::split(ragged, 2);
auto outer = split_result.first;   // extents = [2, 3, 1], ragged dimension
auto inner = split_result.second;  // extent = 2, regular dimension
```

**Merge**: Merge a RaggedIterDomain with a regular IterDomain.

```cpp
auto inner = IrBuilder::create<IterDomain>(0, 4);       // extent 4

// Merge: ragged dimension becomes outer, inner becomes feature
auto merged = IterDomain::merge(ragged, inner);
// Result: RaggedIterDomain with nested extents [3*4, 5*4, 2*4] = [12, 20, 8]
```

**Implementation notes**:
- Merge with non-ragged dimension: multiply each nested extent by non-ragged extent
- Split on ragged: split each nested IterDomain individually, creating new RaggedIterDomain with split components
- All transformations preserve uniform property requirements

#### Parallelization

The `parallelize()` method applies uniformly to all nested domains:
```cpp
ragged->parallelize(ParallelType::TIDx);
// All nested domains now have ParallelType::TIDx
```

#### Select Operation

The `select` operation extracts a specific component from a ragged dimension, converting it to a regular IterDomain. This requires two steps:

1. Select on the batch dimension to choose which component
2. The ragged dimension automatically becomes a regular IterDomain with that component's extent

```cpp
// Starting tensor: [batch=3, ragged=[3,5,2], feature=4]
auto tv = makeContigTensor(3);  // Assume ragged dimension is at position 1

// Select batch component 1
auto selected = tv->select(/*batch_dim=*/0, /*index=*/1);
// Result: [ragged_extent=5, feature=4]
// The ragged dimension collapsed to a regular IterDomain with extent 5
```

**Implementation notes:**
- Select on batch dimension causes the RaggedIterDomain to resolve to the corresponding nested IterDomain
- The nested IterDomain at the selected index replaces the RaggedIterDomain in the output TensorDomain
- This enables direct access to individual components, similar to PyTorch's `nested_tensor[i]` indexing

#### Partition Operation

The `partition` operation splits a regular IterDomain into a batch IterDomain and a RaggedIterDomain, based on variable-length segments defined by an offset tensor. This is the inverse of `merge` and is essential for expert parallelism workflows where tokens need to be grouped by expert or rank.

**API Design (following split/merge pattern):**

```cpp
// Static method in IterDomain class (csrc/ir/interface_nodes.h)
class IterDomain : public Val {
  // ...
  static std::pair<IterDomain*, RaggedIterDomain*> partition(
      IterDomain* in,           // Input IterDomain to partition
      Val* offsets              // Offset values defining partition boundaries
  );
  // ...
};

// TensorView API (csrc/ir/interface_nodes.h)
class TensorView : public Val {
  // ...
  // Partition dimension 'dim' using the provided offsets
  // Returns new TensorView with partitioned dimension replaced by (batch_id, ragged_id)
  TensorView* partition(int dim, TensorView* offsets);
  // ...
};
```

**Semantics:**
- **Input**: Regular IterDomain with total extent N
- **Offsets**: TensorView (1D tensor) with K+1 elements `[0, offset_1, ..., offset_K=N]`
- **Output**:
  - Batch IterDomain with extent K (number of partitions/components)
  - RaggedIterDomain with K nested domains, extent of component i is `offsets[i+1] - offsets[i]`

**Example 1: Simple partitioning**
```cpp
// Starting with flattened tokens: [token=325, hidden=512]
auto tv = makeContigTensor(2);

// Partition into 3 experts with token counts [127, 0, 198]
// offsets = [0, 127, 127, 325]
auto partitioned = tv->partition(/*dim=*/0, offsets);
// Result: [expert=3, tokens_per_expert=[127,0,198], hidden=512]
// Dimension 0 is replaced by (expert_id, ragged_tokens_id)
```

**Example 2: Multi-level partitioning (expert parallelism)**
```cpp
// Input: Distributed tokens across D=2 GPUs, each GPU has S/D tokens
// Shape: [D=2, S/D=100, hidden=512]
// Total S=200 tokens evenly distributed: 100 tokens per GPU
auto tokens = makeContigTensor(3);  // [2, 100, 512]

// Step 1: Partition S/D dimension by expert (inherent data property)
// Tokens from each GPU are routed to E=4 experts with different counts per expert per GPU.
// expert_offsets: 2D tensor [D=2, E+1=5] with per-GPU offsets for E=4 experts:
// expert_offsets[0] = [0, 30, 30, 70, 100]    - GPU 0: [30, 0, 40, 30] tokens per expert
// expert_offsets[1] = [0, 25, 60, 85, 100]    - GPU 1: [25, 35, 25, 15] tokens per expert
auto by_expert = tokens->partition(/*dim=*/1, expert_offsets);
// Result: [gpu=2, expert=4, tokens_per_expert=[[30,0,40,30],[25,35,25,15]], hidden=512]
// Now we have nested ragged: outer gpu dimension, inner ragged tokens per expert

// Step 2: Shuffle to expert-first layout and distribute across GPUs
// The merge operation represents the shuffling that reorganizes from
// [gpu, expert, ragged] to expert-first layout. The actual implementation
// performs the communication to change the data layout.
auto merged = IterDomain::merge(gpu_dim, tokens_per_expert);
// This creates: [expert=4, merged_ragged=[55,35,65,45]]
// Where merged tokens per expert = sum across source GPUs:
//   Expert 0: 30 (from GPU 0) + 25 (from GPU 1) = 55 tokens
//   Expert 1: 0 (from GPU 0) + 35 (from GPU 1) = 35 tokens
//   Expert 2: 40 (from GPU 0) + 25 (from GPU 1) = 65 tokens
//   Expert 3: 30 (from GPU 0) + 15 (from GPU 1) = 45 tokens

// Then split experts across GPUs for parallel processing (2 experts per GPU)
auto [gpu_out, expert_per_gpu] = IterDomain::split(expert_dim, /*factor=*/2);
// Result: [gpu=2, expert_per_gpu=2, merged_ragged=[[55,35],[65,45]], hidden=512]
//   GPU 0 processes experts 0-1 with [55, 35] tokens respectively
//   GPU 1 processes experts 2-3 with [65, 45] tokens respectively

// Summary:
// - Input: [D=2, S/D=100] uniform tokens per GPU
// - After partition: [D=2, E=4, ragged] non-uniform tokens per (GPU, expert)
// - After merge+split: [D=2, E/D=2, ragged] expert-first, distributed for processing
```

**Relationship to other operations:**
- **vs. split**: `split` divides uniformly; `partition` divides non-uniformly based on offsets
- **Inverse of merge**: `merge` concatenates ragged components; `partition` splits regular dimension into ragged
- **Used by asNested**: The `asNested` tensor-level operation uses `partition` internally as a transform between root and logical domains

**Implementation notes:**
- Partition is a view operation (no data movement at runtime)
- The offsets TensorView must be computable (either fusion input or computed by previous operations)
- For multi-level partitioning, nested RaggedIterDomains track their own offset computations
- Critical for rank-to-expert and expert-to-rank reshuffling in distributed expert parallelism

### 6.5 Indexing and Code Generation

#### Offset-Based Indexing

For ragged iteration, global indices are computed as:
```
global_index = offset[component_idx] + local_index
```

Where `component_idx` is the batch index and `local_index` iterates from 0 to `extent[component_idx]`. The offset array provides the starting position of each component in contiguous storage.

#### Loop Structure

Generated CUDA code follows this pattern:
```cuda
for (int batch = 0; batch < num_components; batch++) {
  int offset = offsets[batch];
  int extent = extents[batch];

  // Uniform parallelization (e.g., threadIdx.x)
  for (int tid = threadIdx.x; tid < extent; tid += blockDim.x) {
    int global_idx = offset + tid;
    // Process element at global_idx
  }
}
```

#### Indexer Strategy

RaggedIterDomain will integrate with the IdModel-based indexing system. This requires extending IdModel to handle ragged dimensions, including new expression types for ragged transformations and modifications to ValGraph handling.

### 6.6 Memory Layout

Ragged data is stored contiguously in memory with components placed sequentially:

```
Ragged dimension with extents [3, 5, 2]:

Storage: [c0_0, c0_1, c0_2, c1_0, c1_1, c1_2, c1_3, c1_4, c2_0, c2_1]
         └─ Component 0 ─┘ └─────── Component 1 ───────┘ └─ Comp 2 ─┘

Offsets: [0, 3, 8, 10]
  - Component 0: indices [0, 1, 2]
  - Component 1: indices [3, 4, 5, 6, 7]
  - Component 2: indices [8, 9]

Properties:
  - extent = 10
```

This layout enables efficient sequential access and avoids padding overhead.

### 6.7 Extent and Offset Tensor Management

In generated CUDA code, non-nested tensors are represented using the Tensor struct, which has extents as a property. For nested tensors, we cannot have extents of nested domains as they may be dynamically computed.

Consider the mixture of experts (MoE) use case where a kernel dynamically creates a nested tensor output:

```python
# Input: tokens [num_tokens, hidden_dim], routing decisions per token
# Output: nested tensor where each component corresponds to tokens for one expert

# At kernel launch time:
# - Total number of tokens: KNOWN (e.g., 1024)
# - Number of experts: KNOWN (e.g., 8)
# - Tokens per expert: UNKNOWN (depends on routing computation inside kernel)

# Inside the kernel:
# 1. Compute routing: which tokens go to which expert
# 2. Count tokens per expert: [127, 0, 198, 64, 412, 89, 103, 31]
# 3. Compute offsets: [0, 127, 127, 325, 389, 801, 890, 993, 1024]
# 4. Reorder token data: group tokens by expert assignment
# 5. Write nested tensor output with ragged dimension

# Result: nested tensor [num_experts=8, ragged_tokens=[127,0,198,...], hidden_dim]
```

**Key Observation**: The nested domain extents are computed **inside the kernel** and are not known at kernel launch time.

**Implication**: We cannot bundle extent/offset information with the nested tensor itself.

This problem can be addressed by managing the offsets as a separate tensor that can be computed dynamically on GPU and passed between kernels. That effectively means a logical nested tensor consists of two Vals: one tensor for the nested tensor itself and another tensor for the offsets. More concretely, here's a fusion that creates a nested tensor with `asNested` as an output:

```cpp
// User-defined Fusion
Fusion fusion;
FusionGuard fg(&fusion);

// User provides data and offsets as separate inputs
auto tv_data = TensorViewBuilder()
    .ndims(2)
    .shape({-1, 512})  // [total_tokens, hidden]
    .dtype(DataType::Float)
    .build();
fusion.addInput(tv_data);

auto tv_offsets = TensorViewBuilder()
    .ndims(1)
    .shape({9})  // [num_experts + 1]
    .dtype(DataType::Int)
    .build();
fusion.addInput(tv_offsets);

// User explicitly creates nested tensor
auto tv_nested = asNested(tv_data, tv_offsets, /*ragged_dim=*/0);
// tv_nested has shape [batch=8, ragged_tokens, hidden=512]

// Operations on the nested tensor
auto tv_result = some_operation(tv_nested);

fusion.addOutput(tv_result);
```

The output tensor, `tv_result`, is a nested tensor. The extents of the nested domains are given as a fusion input, but in general, they are not known until the fusion is executed. Thus, if the nested tensor struct were defined like:

```cpp
template <typename DT, int rank>
struct NestedTensor {
	DT* ptr;
	int64_t extents[rank];
	int64_t nested_domain_extents[ragged_dimension_rank];
};
```

The value of `nested_domain_extents` is not available until the completion of the kernel, which would block the launch of the subsequent kernel.

Instead, we would like the fusion to be defined as follows:

```cpp
fusion.addInput(tv_data);      // Original data input (unchanged)
fusion.addInput(tv_offsets);   // Original offset input (unchanged)

auto tv_nested = asNested(tv_data, tv_offsets, /*ragged_dim=*/0);
auto tv_result = some_operation(tv_nested);

auto tv_result_offsets = /* extract/compute offset part of tv_result */;

fusion.addOutput(tv_result);      // Data tensor output
fusion.addOutput(tv_result_offsets);   // Offset tensor output (injected)
```

Here, for `tv_result` we would use the same `Tensor` struct as the normal tensor. The offset tensor would be a 1D tensor with the `ptr` val referring to the vector holding the offsets on the device memory. In this case, there's nothing to block the launch of the subsequent kernel as the offset vector would remain on the device memory.

Since it is an implementation detail, the offset tensor should be hidden behind the nested tensor in the user-facing Fusion definition. When a user uses `asNested` to create a nested tensor, it should still create a single nested tensor Val, as illustrated in the first case above. The translation to the second pattern should be done automatically, e.g., by a new preseg pass.

---

## 7. System Integration

### IR Layer

**Type System**:
- Add `RaggedIterDomain` to `ValType` enum in `csrc/type.h`
- Update `DISPATCH_FOR_ALL_VALS` macro in `csrc/dispatch.h`

**Class Implementation**:
- Class declaration in `csrc/ir/internal_base_nodes.h` after IterDomain
- Implementation in `csrc/ir/nodes.cpp` with validation logic

**Dispatch and Visitors**:
- Add dispatch handlers in `csrc/dispatch.cpp`
- Update visitor patterns to traverse nested domains

**Cloning and Printing**:
- Implement cloning constructor with `NVFUSER_DEFINE_CLONE`
- Add printing support in `csrc/ir/iostream.cpp` showing nested structure

### Indexing Layer

**IdModel Extensions**:
- Extend IdModel to handle RaggedIterDomain in ValGraph
- Modify `TensorIndexer` to compute offset-based indices
- Update loop promotion logic for ragged dimensions

**Detection**:
- Detect RaggedIterDomain during graph building
- Route to ragged-aware indexing logic

### Lowering and CodeGen

**Device Lowering**:
- Handle RaggedIterDomain in allocation passes
- Generate offset array computations
- Create predicates for ragged bounds checking

**Code Generation**:
- Emit nested loop structure with offset-based indexing
- Generate uniform parallelization across components
- Handle extent and offset lookups in generated CUDA code

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure
- Type system updates (ValType enum, dispatch macros)
- RaggedIterDomain class declaration and implementation
- Basic validation, accessors, printing
- **Goal**: Can create and inspect RaggedIterDomain instances

### Phase 2: IdModel Integration
- Extend IdModel ValGraph to handle RaggedIterDomain
- Modify TensorIndexer for offset-based indexing
- Add new expression types for ragged operations
- Predicate generation for ragged bounds
- **Goal**: Can compile and execute simple ragged operations

### Phase 3: Transformations
- Implement split operations on ragged dimensions
- Implement merge operations (ragged with regular IterDomain)
- Add parallelize override
- Additional operations (flatten, nest/unnest) as determined by expert parallelism requirements
- **Goal**: Can create and transform ragged dimensions

### Phase 4: Full Integration
- TensorView integration (allow RaggedIterDomain in TensorDomain)
- Device lowering passes for ragged
- CUDA code generation with offset-based indexing
- Comprehensive end-to-end tests
- **Goal**: Production-ready ragged tensor support

---

## 9. Future Work

- **Python Frontend**: Expose RaggedIterDomain to Python API for direct construction
- **Ragged-Aware Schedulers**: Specialized pointwise, reduction, and matmul schedulers for ragged patterns
- **Broadcast Operations**: Support broadcasting to/from ragged dimensions

---

## 10. References

- [PyTorch Nested Tensor Documentation](https://pytorch.org/docs/stable/nested.html)

---
