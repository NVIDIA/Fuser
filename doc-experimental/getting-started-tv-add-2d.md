# Getting Started: A Minimal Standalone nvFuser Program (tv_add_2d)

## What you’ll learn
- Build a tiny Fusion IR in C++
- Create `TensorView`s, apply a transpose, and add tensors
- Print unscheduled vs scheduled IR and per-TensorView transforms
- (Optional) Execute the fusion with real tensors using `FusionExecutorCache`

## Prerequisites
- A C++17 compiler and CUDA toolkit installed (for execution)
- PyTorch/ATen libraries available (e.g., from a pip install) and discoverable at runtime
- This repo checked out at `/opt/pytorch/nvfuser` (paths assume that location)

## Key types
- [`TensorView`](../csrc/ir/interface_nodes.h)
- [`Fusion`](../csrc/fusion.h), [`FusionGuard`](../csrc/fusion_guard.h)
- [`FusionExecutorCache`](../csrc/runtime/fusion_executor_cache.h)
- [`KernelArgumentHolder`](../csrc/runtime/executor_kernel_arg.h)

## API references
- View/alias ops: [`set`, `reshape`, `transpose`, `broadcast`, `squeeze` in alias.h](../csrc/ops/alias.h)
- Arithmetic/constructors: [`add`, `iota` in arith.h](../csrc/ops/arith.h)
- IR printing: [`iostream.h`](../csrc/ir/iostream.h)

## Minimal IR-only example (prints IR, no execution)

This variant constructs the IR and prints it. It does not depend on ATen at runtime.

```cpp
#include <iostream>
#include <fusion.h>                 // nvFuser Fusion container (IR graph)
#include <fusion_guard.h>           // RAII guard that sets the active Fusion
#include <ir/interface_nodes.h>     // Val, TensorView, Expr node types
#include <ir/iostream.h>            // pretty printers for IR
#include <ops/arith.h>              // add(...), iota(...), etc.
#include <ops/alias.h>              // set(...), reshape(...), transpose(...)

using namespace nvfuser;

int main() {
  constexpr int64_t N = 31; // intentionally not divisible by common vector widths

  // Create a Fusion and establish it as the active build context.
  // While the guard is alive, free functions like add/transpose/reshape
  // record IR nodes into this Fusion via IrBuilder.
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Build a 1D iota of length N*N (float), starting at 1.0 with step 1.0.
  // iota(...) returns a TensorView* that symbolically represents the sequence.
  Val* n    = IrBuilder::create<Val>(N, DataType::Index);
  Val* len  = IrBuilder::create<Val>(N * N, DataType::Index);
  Val* start= IrBuilder::create<Val>(1.0, DataType::Float);
  Val* step = IrBuilder::create<Val>(1.0, DataType::Float);
  TensorView* A1d = iota(len, start, step, DataType::Float); // shape: [N*N]

  // Reshape the 1D iota into a 2D tensor [N, N]. This is a view op in IR.
  // Note: reshape returns a new TensorView and emits a ViewOp in the graph.
  TensorView* A = reshape(A1d, std::vector<Val*>{n, n});

  // Copy A (IR-level aliasing) and then transpose it to get B = A^T logically.
  // set(...) emits a Set; transpose(...) emits a Set.Permute linking B to A.
  TensorView* B_copy = set(A);
  TensorView* B      = transpose(B_copy);

  // Define C = A + B (pointwise add). Register as an output so it appears
  // at the graph boundary (and would be returned by an executor if we ran).
  TensorView* C = add(A, B);
  fusion.addOutput(C);

  // Inspect per-TensorView transform history (root/logical/loop domains).
  std::cout << "=== TensorView A ===\n" << A->toString() << "\n";
  A->printTransforms();
  std::cout << "\n=== TensorView B (transposed copy of A) ===\n" << B->toString() << "\n";
  B->printTransforms();
  std::cout << "\n=== TensorView C ===\n" << C->toString() << "\n";
  C->printTransforms();

  // Print the full Fusion IR with and without per-TV transform sections.
  std::cout << "\n=== Fusion IR (with transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/true);
  std::cout << "\n=== Fusion IR (without transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/false);

  return 0; // No execution in this variant
}
```

Reference source: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp`

## Executable variant (runs on GPU via ATen + FusionExecutorCache)

This version defines an input `TensorView` placeholder, binds a real `at::Tensor`, JITs, and runs the kernel.

```cpp
#include <iostream>
#include <fusion.h>                       // Fusion IR container
#include <fusion_guard.h>                 // Sets active Fusion (RAII)
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ops/arith.h>                    // add(...)
#include <ops/alias.h>                    // set(...), transpose(...)
#include <runtime/fusion_executor_cache.h>// JIT cache: compile + run
#include <runtime/executor_kernel_arg.h>  // KernelArgumentHolder
#include <ATen/ATen.h>                    // PyTorch tensor library (ATen)

using namespace nvfuser;

int main() {
  constexpr int64_t N = 31;

  // Build the IR inside a Fusion we can move into the executor cache.
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Define a symbolic input TensorView A with shape [N, N].
  // TensorViewBuilder creates a TV (no data) and we register it as input.
  TensorView* A = TensorViewBuilder()
                      .ndims(2)
                      .shape({N, N})
                      .contiguity({true, true})
                      .dtype(DataType::Float)
                      .build();
  fusion->addInput(A); // marks A as a runtime input (order matters)

  // Produce B = transpose(copy(A)) and C = A + B.
  TensorView* B_copy = set(A);      // IR alias/copy node
  TensorView* B      = transpose(B_copy);
  TensorView* C      = add(A, B);
  fusion->addOutput(C);             // register C as runtime output

  // Prepare a real input tensor on CUDA and bind it via KernelArgumentHolder.
  // Note: TVs are symbolic; at::Tensor provides actual storage at execution.
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  at::Tensor Ain = at::arange(1, (int64_t)N * N + 1, opts).view({N, N});

  KernelArgumentHolder args;
  args.push(Ain); // push inputs in the same order as fusion->inputs()

  // JIT compile and run. The cache handles segmentation and kernel builds.
  FusionExecutorCache fec(std::move(fusion_ptr));
  KernelArgumentHolder outs = fec.runFusionWithInputs(args);

  // Retrieve the first output as an at::Tensor and print it (CPU).
  at::Tensor Cout = outs[0].as<at::Tensor>();
  std::cout << "=== Execution Output (C) ===\n" << Cout.to(at::kCPU) << "\n";

  return 0;
}
```

Reference source: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp`

## Notes and tips
- `TensorView` is symbolic: it does not hold data. Real tensors are bound at execution.
- `FusionGuard` scopes the active Fusion so free functions (e.g., `add`, `transpose`) record nodes into it.
- Use `printTransforms()` on any `TensorView` to see view/schedule history.
- Print IR with and without tensor transforms to study graph vs per-TV transform details.

## See also
- Build/run helpers: `../doc-bot/how_to_build`, `../doc-bot/how_to_run`
- Deeper IR walk-through: `../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md`
