# RaggedIterDomain for Nested Tensors

---

## 1. Overview

This document proposes adding `RaggedIterDomain`, a new `IterDomain` subclass to nvFuser that represents dimensions with variable extents across batch components (ragged/jagged dimensions). This enables efficient compilation of PyTorch nested tensors and other variable-length data structures without padding overhead.

**Scope Note**: This proposal represents a **minimalistic initial version** containing only the capabilities that are absolutely necessary for expert parallelism. The exact requirements for expert parallelism are still being clarified with Jingyue. As those requirements become clear, additional capabilities (such as flatten operations, specific IdModel integrations, or lowering support) may be added to this design. The features described here should be considered the bare minimum starting point.

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

**PyTorch Restriction**: PyTorch nested tensors currently support only one ragged dimension per tensor. For the initial implementation, nvFuser will have the same restriction to simplify the design. Support for multiple ragged dimensions per tensor is deferred to future work.

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

---

## 4. Goals and Non-Goals

### Goals

- Support ragged dimensions in fusion IR for expert parallelism

### Non-Goals

- Multi-level nesting (ragged within ragged)
- Multiple ragged dimensions per tensor (deferred to future work)
- Inlining
- Extension of single-GPU schedulers for ragged dimensions

Inlining would be eventually necessary for lowering to CUDA code but should not be necessary for distributed device (multi-GPU) parallelization. Similarly, the schedulers will be left as is for now.

---

## 5. Requirements

### Functional Requirements

- **IR Representation**: Represent ragged dimensions with variable extents per batch component
- **Offset Computation**: Compute and provide access to offsets for indexing into contiguous storage
- **Operations Support**: Support creation and basic transformations on ragged dimensions (flatten operation TBD based on expert parallelism requirements)
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

A `TensorDomain` can contain a mix of regular `IterDomain` and `RaggedIterDomain` instances, representing tensors with both uniform and ragged dimensions. For the initial implementation, **only one RaggedIterDomain per TensorDomain is supported** (matching PyTorch's restriction). Example:

```cpp
// Example: Batch of sequences with variable lengths
auto batch = IrBuilder::create<IterDomain>(0, 3);
auto ragged_seq = IrBuilder::create<RaggedIterDomain>({seq0, seq1, seq2});
auto feature = IrBuilder::create<IterDomain>(0, 4);

// TensorDomain: {batch, ragged_seq, feature}
// Only one ragged dimension allowed
auto tensor_domain = IrBuilder::create<TensorDomain>({batch, ragged_seq, feature});
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

**Creating Nested Tensors from Data and Offsets: viewAsNested**

nvFuser provides the `viewAsNested` operation, mirroring PyTorch's `torch.nested.nested_tensor_from_jagged(values, offsets)`, to create a nested tensor from separate data and offset tensors:

```cpp
// In csrc/ops/alias.h (or new csrc/ops/nested.h)
NVF_API TensorView* viewAsNested(
    TensorView* data,      // Data tensor with contiguous ragged storage
    TensorView* offsets,   // Offset tensor [num_components + 1]
    int64_t ragged_dim     // Which dimension of data is ragged
);
```

**Semantics**:
- Takes a regular TensorView `data` and reinterprets one dimension as ragged
- `offsets` tensor provides the boundaries for each component (see Section 6.7 for offset tensor format)
- Returns a TensorView with a RaggedIterDomain at position `ragged_dim`
- This is a view operation (no data copy), similar to `reshape` or `view`

**Example Usage**:
```cpp
// Data: [325, 512] - flattened ragged dimension
// Offsets: [0, 127, 127, 325] - 3 components with lengths [127, 0, 198]
auto nested_tv = viewAsNested(data_tv, offsets_tv, /*ragged_dim=*/0);
// Result: TensorView with shape [batch=3, ragged=[127,0,198], hidden=512]
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

This problem can be addressed by managing the offsets as a separate tensor that can be computed dynamically on GPU and passed between kernels. That effectively means a logical nested tensor consists of two Vals: one for the nested tensor itself and another tensor for the offsets.

Since it is an implementation detail, the offset tensor should not be visible in the user-facing Fusion definition. When a user uses `viewAsNested` to create a nested tensor, it should still create a single nested tensor Val. At the time of the preseg phase, we automatically inject the offset tensor as a TensorView to the fusion. For example:

**Original Fusion (user-defined)**:
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

// User explicitly creates nested tensor view
auto tv_nested = viewAsNested(tv_data, tv_offsets, /*ragged_dim=*/0);
// tv_nested has shape [batch=8, ragged_tokens, hidden=512]

// Operations on the nested tensor
auto tv_result = some_operation(tv_nested);

fusion.addOutput(tv_result);
```

**Transformed Fusion (after preseg pass)**:
```cpp
// After offset injection preseg pass
// The pass ensures offset tensors are threaded through to all nested tensor outputs

fusion.addInput(tv_data);      // Original data input (unchanged)
fusion.addInput(tv_offsets);   // Original offset input (unchanged)

auto tv_nested = viewAsNested(tv_data, tv_offsets, /*ragged_dim=*/0);
auto tv_result = some_operation(tv_nested);

// INJECTED: Extract data and offset tensors from nested result
auto tv_result_offsets = /* extract/compute offset part of tv_result */;

fusion.addOutput(tv_result);      // Data tensor output
fusion.addOutput(tv_result_offsets);   // Offset tensor output (injected)
```

This design should also allow eliminating CPU and GPU synchronization for offset transfers.

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
