# TensorView

TensorView is nvFuser's symbolic tensor metadata container that enables sophisticated GPU kernel compilation through domain-driven transformations.

## Synopsis

TensorView represents tensor properties (shape, type, memory layout) without storing actual data, serving as the foundation for nvFuser's compile-time optimization and scheduling system. It provides the primary interface for tensor operations in fusion construction workflows and acts as a bridge between high-level tensor operations and low-level GPU kernel generation.

## Overview

TensorView embodies five core design principles that make nvFuser's JIT compilation system both powerful and flexible:

**Symbolic Foundation**: TensorView represents tensor properties without storing data, enabling powerful compile-time analysis and optimization strategies. This separation allows complex transformations and analysis to occur before any kernel execution, supporting dynamic shapes and efficient compilation caching.

**Domain Architecture**: Uses multiple domain types (root, logical, allocation, loop) to systematically track transformations from initial specification through final kernel generation. Each domain serves specific compilation phases, with transforms recorded and available for replay during analysis and optimization.

**Fusion Integration**: Lives within Fusion containers that manage computation graphs, enabling cross-tensor optimizations and coordinated compilation. The Fusion provides context for sophisticated dependency analysis, memory planning, and optimization opportunities across the entire computation graph.

**Runtime Separation**: IR construction happens independently of data binding, allowing dynamic shapes and efficient kernel reuse across different tensor inputs. Physical ATen tensors are bound to TensorViews only during execution, with shape and type validation occurring at runtime.

**Scheduling Control**: Provides sophisticated scheduling system through ComputeAt positioning and transform propagation for memory and performance optimization. This hierarchical approach enables both automatic scheduling for common cases and manual fine-tuning for expert optimization.

## Source

Primary implementation: [`../../../csrc/ir/interface_nodes.h#L383`](../../../csrc/ir/interface_nodes.h#L383)

Supporting implementation: [`../../../csrc/tensor_view.cpp`](../../../csrc/tensor_view.cpp)

## Core Concepts

### Symbolic Tensor Representation

TensorView contains shape, dtype, memory type, and domain information but never actual tensor values. This metadata-only design enables complex analysis, optimization, and transform validation before any kernel execution.

```cpp
// TensorView represents metadata, not data
TensorView* tv = TensorViewBuilder()
    .ndims(2)
    .shape({N, M})
    .dtype(DataType::Float)
    .build();
// No actual tensor data is allocated or stored
```

The symbolic approach provides several key benefits:
- **Compilation Efficiency**: Complex analysis and optimization without data overhead
- **Dynamic Shape Support**: Runtime data binding supports varying tensor sizes
- **Memory Efficiency**: Lightweight representation enables complex fusion graphs
- **Analysis Foundation**: Metadata enables dependency analysis and optimization opportunities

### Domain-Driven Transform System

TensorView organizes tensor dimensions through multiple domain types that track transformations throughout the compilation pipeline:

- **Root Domain**: Original tensor dimensions as initially specified
- **Logical Domain**: User-visible dimensions after view/reshape operations
- **Allocation Domain**: Physical memory layout dimensions for optimization
- **Loop Domain**: Final kernel generation structure for execution

```cpp
// Access different domain types
auto root_dims = tv->getRootDomain();      // Original structure
auto logical_dims = tv->getLogicalDomain(); // After view operations
auto alloc_dims = tv->getAllocationDomain(); // Memory layout
auto loop_dims = tv->getLoopDomain();       // Kernel generation
```

All transform operations (split, merge, reorder, broadcast) are recorded and can be replayed for analysis:

```cpp
tv->split(0, 128);        // Split dimension for parallelization
tv->merge(1, 2);          // Merge dimensions for coalescing
tv->reorder({2, 1, 0});   // Optimize memory access patterns
tv->printTransforms();    // Show complete transformation history
```

### Fusion-Centric Execution Model

TensorViews exist within Fusion containers that manage the complete computation graph:

```cpp
auto fusion_ptr = std::make_unique<Fusion>();
FusionGuard fg(fusion_ptr.get());  // Establish active context

// Create TensorViews within fusion context
TensorView* input = TensorViewBuilder().ndims(2).build();
fusion->addInput(input);   // Register as runtime boundary

TensorView* result = add(input, input);
fusion->addOutput(result); // Ensure result is returned
```

Key aspects of the fusion model:
- **Context Management**: FusionGuard establishes active context for IR construction
- **Boundary Control**: Input/output registration prevents DCE and establishes execution boundaries
- **Cross-Tensor Analysis**: Enables analysis across multiple TensorViews for optimization
- **Compilation Unit**: Complete Fusion becomes the target for kernel generation

### Runtime Data Binding Architecture

Physical ATen tensors are bound to TensorViews only during execution, not during IR construction:

```cpp
// IR construction (no data involved)
TensorView* tv = TensorViewBuilder().ndims(2).shape({N, N}).build();
fusion->addInput(tv);

// Runtime binding (data provided separately)
at::Tensor data = at::randn({N, N}, at::device(at::kCUDA));
KernelArgumentHolder args;
args.push(data);  // Bind physical tensor to TensorView

FusionExecutorCache fec(std::move(fusion_ptr));
auto outputs = fec.runFusionWithInputs(args);
```

This separation provides:
- **Dynamic Validation**: Shape and type checking at runtime
- **Memory Management**: nvFuser handles device allocation independently
- **Execution Efficiency**: Single compiled kernel reuses across different tensors
- **ATen Integration**: Seamless PyTorch ecosystem compatibility

### Hierarchical Scheduling and Positioning

TensorViews participate in a scheduling system that controls when and where intermediate values are computed:

```cpp
// Control computation placement
producer->computeAt(consumer, position);

// Scheduling modes
producer->computeAt(consumer, pos, ComputeAtMode::Standard);    // Strict
producer->computeAt(consumer, pos, ComputeAtMode::BestEffort);  // Adaptive
producer->computeAt(consumer, pos, ComputeAtMode::MostInlined); // Maximum sharing
```

Scheduling decisions directly impact:
- **Memory Usage**: Positioning affects register pressure and memory footprint
- **Access Patterns**: Controls memory coalescing and bandwidth utilization
- **Parallelization**: Integrates with CUDA execution model for optimal threading
- **Performance**: Fine-grained control enables expert optimization

## Creation and Construction

### TensorViewBuilder Pattern

The recommended approach for creating TensorViews uses the builder pattern:

```cpp
// Basic construction
TensorView* tv = TensorViewBuilder()
    .ndims(2)
    .shape({height, width})
    .dtype(DataType::Float)
    .contiguity(true)
    .build();

// Scalar tensor (0-dimensional)
TensorView* scalar = TensorViewBuilder()
    .dtype(DataType::Float)
    .build();

// Symbolic shapes
Val* dim1 = IrBuilder::create<Val>(DataType::Index);
Val* dim2 = IrBuilder::create<Val>(DataType::Index);
TensorView* symbolic = TensorViewBuilder()
    .shape({dim1, dim2})
    .dtype(DataType::Float)
    .build();
```

Builder advantages:
- **Validation**: Construction-time checking of parameter consistency
- **Readability**: Fluent interface makes intent clear
- **Safety**: Prevents common configuration mistakes
- **Defaults**: Sensible defaults minimize required specification

### IR Operation Creation

Most TensorViews are created automatically by nvFuser operations:

```cpp
TensorView* a = /* ... */;
TensorView* b = /* ... */;

// Operations create new TensorViews
TensorView* sum = add(a, b);           // Elementwise addition
TensorView* product = mul(a, b);       // Elementwise multiplication
TensorView* transposed = transpose(a); // View operation
TensorView* reshaped = reshape(a, {-1}); // Reshape operation
```

Operation-created TensorViews:
- **Automatic Metadata**: Inherit appropriate shape, type, and memory properties
- **Transform Integration**: Domain structure reflects the applied operation
- **Dependency Tracking**: Maintain producer-consumer relationships in the graph
- **Type Inference**: Result types follow nvFuser's type promotion rules

## Essential Operations

### Domain Access and Manipulation

```cpp
// Query domain information
const auto& root = tv->getRootDomain();      // Original dimensions
const auto& logical = tv->getLogicalDomain(); // After view operations
const auto& alloc = tv->getAllocationDomain(); // Memory layout
const auto& loop = tv->getLoopDomain();       // Kernel generation

// Domain properties
bool hasReductions = tv->hasReduction();
bool hasBroadcasts = tv->hasBroadcast();
int64_t numDims = tv->nDims();

// Explicit domain control (advanced)
tv->setLoopDomain(new_loop_domain);
tv->setAllocationDomain(new_alloc_domain, contiguity);
```

### Transform Operations

#### Splitting and Merging
```cpp
// Split dimension for parallelization
tv->split(0, 128);                    // Split axis 0, factor 128
tv->split(1, blockDim.x, true);       // Inner split (factor inside)
tv->split(2, blockDim.y, false);      // Outer split (factor outside)

// Merge adjacent dimensions
tv->merge(0, 1);                      // Merge axes 0 and 1
tv->merge(2);                         // Merge axis 2 with axis 3

// Flatten multiple dimensions
tv->flatten(1, 3);                    // Flatten axes 1 through 3
```

#### Reordering and Broadcasting
```cpp
// Reorder dimensions
tv->reorder({{0, 2}, {1, 0}, {2, 1}}); // Swap dimensions
tv->reorder({2, 1, 0});                // Reverse dimension order

// Add broadcast dimensions
tv->broadcast(0);                      // Add broadcast at position 0
tv->broadcast(2, extent_val);          // Add broadcast with specific extent
```

#### Advanced Transforms
```cpp
// Memory optimization transforms
tv->swizzle(SwizzleType::Transpose, 0, 1); // Bank conflict avoidance
tv->resize(1, left_pad, right_pad);        // Add padding

// Convenience methods
tv->inner_split(0, 32);               // Always inner split
tv->outer_split(1, 64);               // Always outer split
```

### Scheduling and Positioning

```cpp
// Basic ComputeAt usage
producer->computeAt(consumer, 2);     // Share first 2 loop levels

// Scheduling modes for different strategies
producer->computeAt(consumer, -1, ComputeAtMode::MostInlined);
producer->computeAt(consumer, pos, ComputeAtMode::BestEffort);

// Query scheduling state
int64_t compute_pos = tv->getComputeAtPosition();
int64_t producer_pos = tv->getMaxProducerPosition();
bool has_compute_at = tv->hasComputeAt();

// Parallelization integration
tv->axis(0)->parallelize(ParallelType::BIDx);  // Block dimension
tv->axis(1)->parallelize(ParallelType::TIDx);  // Thread dimension
```

## Code Examples

### Complete Basic Workflow

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>
#include <runtime/fusion_executor_cache.h>

using namespace nvfuser;

int main() {
    constexpr int64_t N = 1024;

    // 1. Create Fusion and establish context
    auto fusion_ptr = std::make_unique<Fusion>();
    FusionGuard fg(fusion_ptr.get());

    // 2. Create symbolic TensorViews (no data yet)
    TensorView* input = TensorViewBuilder()
        .ndims(2)
        .shape({N, N})
        .dtype(DataType::Float)
        .contiguity(true)
        .build();
    fusion->addInput(input);

    // 3. Build computation graph
    TensorView* transposed = transpose(input);
    TensorView* result = add(input, transposed);
    fusion->addOutput(result);

    // 4. Apply scheduling (optional)
    result->split(0, 128);
    result->split(1, 32);
    result->reorder({{0, 1}, {1, 2}, {2, 0}});

    // 5. Runtime execution with data binding
    at::TensorOptions opts = at::TensorOptions()
        .device(at::kCUDA)
        .dtype(at::kFloat);
    at::Tensor data = at::randn({N, N}, opts);

    KernelArgumentHolder args;
    args.push(data);

    FusionExecutorCache fec(std::move(fusion_ptr));
    auto outputs = fec.runFusionWithInputs(args);

    // 6. Extract results
    at::Tensor output = outputs[0].as<at::Tensor>();

    return 0;
}
```

### Manual Scheduling Example

```cpp
// Advanced scheduling for performance optimization
void apply_manual_scheduling(TensorView* tv) {
    // Split for thread-level parallelism
    tv->split(0, 128);        // Split outer dimension
    tv->split(1, 32);         // Split inner dimension

    // Reorder for memory coalescing
    tv->reorder({{0, 1}, {1, 2}, {2, 0}});

    // Bind to CUDA execution model
    tv->axis(0)->parallelize(ParallelType::BIDx);  // Block X
    tv->axis(1)->parallelize(ParallelType::BIDy);  // Block Y
    tv->axis(2)->parallelize(ParallelType::TIDx);  // Thread X

    // Apply scheduling to related tensors
    MaxLogicalDomainInfoSpanningTree tree(tv);
    TransformPropagator propagator(tv);
    tree.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(tv);
}
```

### Domain Analysis Example

```cpp
// Analyze and debug TensorView domain structure
void analyze_tensorview(TensorView* tv) {
    std::cout << "=== TensorView Analysis ===\n";
    std::cout << "String representation: " << tv->toString() << "\n";

    // Domain information
    std::cout << "Root domain size: " << tv->getRootDomain().size() << "\n";
    std::cout << "Logical domain size: " << tv->getLogicalDomain().size() << "\n";
    std::cout << "Loop domain size: " << tv->getLoopDomain().size() << "\n";

    // Properties
    std::cout << "Has reductions: " << tv->hasReduction() << "\n";
    std::cout << "Has broadcasts: " << tv->hasBroadcast() << "\n";
    std::cout << "Number of dimensions: " << tv->nDims() << "\n";

    // Scheduling information
    std::cout << "ComputeAt position: " << tv->getComputeAtPosition() << "\n";
    std::cout << "Max producer position: " << tv->getMaxProducerPosition() << "\n";

    // Complete transformation history
    std::cout << "\n=== Transform History ===\n";
    tv->printTransforms();
}
```

## Integration Patterns

### Error Handling and Validation

```cpp
// Runtime validation example
try {
    KernelArgumentHolder args;
    args.push(input_tensor);

    auto outputs = executor.runFusionWithInputs(args);

} catch (const std::exception& e) {
    // Handle shape mismatches, type errors, device placement issues
    std::cerr << "Execution error: " << e.what() << std::endl;

    // Common fixes:
    // 1. Verify tensor shapes match TensorView specifications
    // 2. Check device placement (CPU vs CUDA)
    // 3. Validate data types match DataType specifications
    // 4. Ensure contiguity requirements are met
}
```

### Memory Type Optimization

```cpp
// Memory type selection for performance
TensorView* create_optimized_tensor(const std::vector<int64_t>& shape) {
    auto tv = TensorViewBuilder()
        .shape(shape)
        .dtype(DataType::Float)
        .build();

    // Memory type affects allocation strategy
    // - MemoryType::Local: Register/local memory (default)
    // - MemoryType::Global: Global memory for I/O
    // - MemoryType::Shared: Shared memory for communication

    return tv;
}
```

## Troubleshooting and Best Practices

### Common Patterns

1. **Always use TensorViewBuilder** for construction rather than direct constructors
2. **Register inputs and outputs** with the Fusion to prevent DCE elimination
3. **Start with automatic scheduling** before attempting manual optimization
4. **Profile before and after** scheduling changes to verify improvements
5. **Use printTransforms()** for debugging domain and scheduling issues

### Common Pitfalls

**Unregistered Outputs**
```cpp
// WRONG: Result may be eliminated by DCE
TensorView* result = add(a, b);
// Missing: fusion->addOutput(result);

// CORRECT: Always register intended outputs
TensorView* result = add(a, b);
fusion->addOutput(result);
```

**Shape Mismatches**
```cpp
// WRONG: Runtime shape doesn't match TensorView
TensorView* tv = TensorViewBuilder().shape({64, 64}).build();
at::Tensor data = at::randn({32, 32}); // Different shape!

// CORRECT: Ensure shapes match
TensorView* tv = TensorViewBuilder().shape({64, 64}).build();
at::Tensor data = at::randn({64, 64});
```

**Context Management**
```cpp
// WRONG: No FusionGuard context
auto fusion = std::make_unique<Fusion>();
TensorView* tv = TensorViewBuilder().build(); // May fail!

// CORRECT: Establish proper context
auto fusion = std::make_unique<Fusion>();
FusionGuard fg(fusion.get());
TensorView* tv = TensorViewBuilder().build();
```

### Performance Considerations

- **ComputeAt positioning** directly affects register pressure and memory bandwidth
- **Transform sequences** have different efficiency characteristics - profile to validate
- **Memory coalescing** depends on contiguity and allocation domain decisions
- **Manual scheduling** can outperform automatic for specialized workloads
- **Use profiling tools** and memory analysis for optimization guidance

### Debug Workflows

1. **IR Analysis**: Use `fusion->print()` to examine the complete computation graph
2. **Transform History**: Use `tv->printTransforms()` to understand scheduling decisions
3. **Domain Structure**: Examine different domain types to understand compilation phases
4. **Memory Analysis**: Use nvFuser's memory analysis tools for allocation optimization
5. **Performance Profiling**: Integrate with CUDA profiling tools for bottleneck identification

For additional examples and advanced usage patterns, see the [nvFuser examples directory](../../../examples/) and [experimental code samples](../../../doc-bot/experimenting/).