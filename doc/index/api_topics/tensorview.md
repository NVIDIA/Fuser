# TensorView

## Synopsis
The IR handle for tensors in nvFuser’s fusion graph. `TensorView` represents a tensor’s values and its scheduling/view transformations on the path to generated CUDA code.

## Source
- Class: [`TensorView`](../../../csrc/ir/interface_nodes.h#L383)

## Overview
`TensorView` is the primary, user-facing IR node for tensors during code generation. Conceptually, it represents the values of a tensor and the sequence of view/scheduling transforms applied to derive its loop and allocation shapes from its logical shape. While a `TensorDomain` captures the domains (root/logical/allocation/loop), `TensorView` holds the tensor identity and APIs to transform and schedule it.

Why this split matters:
- `TensorDomain` answers “over which axes do we iterate and how are they laid out?”
- `TensorView` answers “what tensor are we talking about and how do we transform/schedule it for codegen?”

Typical lifecycle in a fusion:
1) Create or obtain a `TensorView` (inputs via `fusion.addInput`, or ops like `add`, `transpose`, `reshape`, etc.)
2) Apply view/schedule transforms (e.g., `split`, `merge`, `reorder`, `broadcast`, `flatten`), optionally compute/schedule relations (`computeAt`, `computeWith`), caching (`cacheBefore`, `cacheAfter`), or pipeline/circular-buffering
3) Mark graph boundaries (e.g., `fusion.addOutput(tv)`) so compiled kernels produce the result

`TensorView` is used pervasively by IR-building ops in `csrc/ops/*.h` and by scheduling/lowering utilities. It is the object most developers interact with when shaping loops for performant kernels.

## Key Concepts and APIs
- Domains and shapes
  - Root/Logical/Allocation/Loop domains: see accessors like `getRootDomain`, `getLogicalDomain`, `getAllocationDomain`, `getLoopDomain` in [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h)
  - Contiguity metadata: `setContiguity`, `getContiguity`

- Scheduling/view transforms (shape/axis transforms)
  - `split` / `merge` / `flatten` / `reorder`
  - `broadcast`, `resize`, `swizzle`, `applyMmaSwizzle` (for MMA/TMA swizzles)
  - Selected references in class definition: [`split` et al.](../../../csrc/ir/interface_nodes.h#L585), [`merge`](../../../csrc/ir/interface_nodes.h#L611), [`flatten`](../../../csrc/ir/interface_nodes.h#L619), [`reorder`](../../../csrc/ir/interface_nodes.h#L623)

- Compute placement and inlining
  - `computeAt` to align a producer with a consumer’s loop nest (controls where a tensor is produced)
  - `computeWith` to inline compute with a consumer (resolved during lowering)
  - See: [`computeAt`](../../../csrc/ir/interface_nodes.h#L576), [`computeWith`](../../../csrc/ir/interface_nodes.h#L781)

- Caching and memory
  - `cacheBefore`/`cacheAfter` to insert explicit cache TVs (e.g., registers/shared memory) and memory ops between producer/consumer
  - `getMemoryType`/`setMemoryType` (e.g., Local, Shared)
  - Circular buffering for pipelining: [`circularBuffer`](../../../csrc/ir/interface_nodes.h#L715)

- Printing and inspection
  - `toString`, `toInlineString`, `printTransforms` for IR/schedule visualization

## How `TensorView` Relates to Other Types
- `TensorDomain` (in `ir/internal_base_nodes.h`): per-axis domain structure (root/logical/allocation/loop) and contiguity
- `IterDomain`: single axis descriptor composing `TensorDomain`
- `Val`/`Expr`: base IR node types; `TensorView` derives from `Val`

## Common Operations that Produce/Consume TensorView
- Pointwise and other ops (subset):
  - [`add(TensorView*, TensorView*)`](../../../csrc/ops/arith.h#L441)
  - [`sum_to`](../../../csrc/ops/arith.h#L685)
  - [`fusedMultiplySum`](../../../csrc/ops/arith.h#L714)
  - [`tensor(val)`](../../../csrc/ops/arith.h#L720)
- Aliasing/shape/view ops:
  - [`transpose`](../../../csrc/ops/alias.h#L107), [`transpose(x)`](../../../csrc/ops/alias.h#L110)
  - `set` (copy/alias) in [`alias.h`](../../../csrc/ops/alias.h)

## Heavily Commented Usage Example
Below is a self-contained sketch (adapted from `doc-bot/experimenting`) showing how to create an input `TensorView`, apply transforms, build a simple graph, print transforms, and execute with ATen tensors.

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>      // TensorView, TensorViewBuilder, ops signatures
#include <ir/iostream.h>
#include <ops/arith.h>               // add, iota, reshape
#include <ops/alias.h>               // set, transpose
#include <runtime/fusion_executor_cache.h>
#include <runtime/executor_kernel_arg.h>
#include <ATen/ATen.h>

using namespace nvfuser;

int main() {
  constexpr int64_t N = 31;

  // Build IR in a Fusion. Fusion owns all IR nodes, including TensorViews.
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion); // sets active Fusion for IR-building functions

  // Define an input TensorView [N, N]. TensorView (IR) carries shape/dtype
  // metadata but not data. addInput marks it as runtime input.
  TensorView* A = TensorViewBuilder()
                      .ndims(2)
                      .shape({N, N})
                      .contiguity({true, true})
                      .dtype(DataType::Float)
                      .build();
  fusion->addInput(A);

  // B = transpose(copy(A)). Ops consume/produce TensorViews.
  TensorView* B_copy = set(A);
  TensorView* B = transpose(B_copy);

  // Pointwise: C = A + B
  TensorView* C = add(A, B);
  fusion->addOutput(C); // mark graph boundary so executor returns C

  // Inspect schedule/view transforms associated with TVs (no data printed).
  A->printTransforms();
  B->printTransforms();
  C->printTransforms();

  // Execute: provide at::Tensor inputs via KernelArgumentHolder.
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  at::Tensor Ain = at::arange(1, (int64_t)N * N + 1, opts).view({N, N});

  KernelArgumentHolder args;
  args.push(Ain);

  FusionExecutorCache fec(std::move(fusion_ptr));
  KernelArgumentHolder outs = fec.runFusionWithInputs(args);
  at::Tensor Cout = outs[0].as<at::Tensor>();

  // Cout is A + A^T on device. Move to CPU for printing in this demo.
  std::cout << Cout.to(at::kCPU) << "\n";
}
```

## Additional Guidance and Gotchas
- `TensorView` vs data: A `TensorView*` is an IR node, not storage. Real data lives in runtime tensors (`at::Tensor`) passed to/returned from executors.
- Domains evolve with transforms: `split/merge/reorder/flatten/broadcast/view` change the loop/allocation domains. Use `printTransforms` to debug.
- Compute placement matters: `computeAt`/`computeWith` control where a producer is realized relative to consumers, impacting memory locality and parallelization.
- Caching and memory reuse: `cacheBefore/After` introduce read/write caches (e.g., shared memory) between producers/consumers. `promoteReuse` can enable shared-memory reuse with proper synchronization.
- Circular buffering: `circularBuffer(stages, prefetch, type)` enables overlapped load/compute pipelines; see design comments around the API in the class for constraints and semantics.

## Where to Look in the Codebase
- Definition & comments: [`interface_nodes.h`](../../../csrc/ir/interface_nodes.h#L359)
- Common ops that produce/consume TVs:
  - Pointwise and tensor constructors in [`ops/arith.h`](../../../csrc/ops/arith.h)
  - Aliases/views in [`ops/alias.h`](../../../csrc/ops/alias.h)
- Experimenting samples (learn-by-example):
  - [`tv_add_2d_SAMPLE_1.cpp`](../../experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp)
  - [`tv_add_2d_SAMPLE_2.cpp`](../../experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp)
  - [`tv_deeper_dive.cpp`](../../experimenting/8-26-2025/tv_deeper_dive.cpp)

## See Also
- `TensorDomain` and `IterDomain` (domain structure): [`ir/internal_base_nodes.h`](../../../csrc/ir/internal_base_nodes.h)
- Scheduling utilities and analysis: e.g., [`compute_at.h`](../../../csrc/compute_at.h), [`transform_replay.h`](../../../csrc/transform_replay.h)

