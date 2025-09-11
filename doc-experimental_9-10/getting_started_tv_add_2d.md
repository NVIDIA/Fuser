## Getting Started: Your First Standalone nvFuser Program (tv_add_2d)

This article walks you through building the smallest end-to-end nvFuser program that adds two 2D tensors on GPU. We introduce the Fusion IR concepts involved, show a fully commented C++ example, and link to relevant nvFuser source for deeper reading.

Why this matters:
- Understand how to author a minimal Fusion IR graph using `Fusion` and `TensorView`.
- See how inputs/outputs are registered and executed via `FusionExecutorCache`.
- Learn where to inspect generated IR and CUDA code for debugging.

References to code (relative to this document):
- nvFuser core: `../csrc/fusion.h`, `../csrc/fusion_guard.h`, `../csrc/ops/arith.h`, `../csrc/tensor_view.cpp`
- Runtime executor: `../csrc/runtime/fusion_executor_cache.h`, `../csrc/runtime/executor_kernel_arg.h`
- Sample programs: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp`, `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp`

### Minimal, heavily commented C++ sample

The following uses nvFuser C++ APIs directly to build a Fusion that computes `C = A + B` for two 2D tensors. It demonstrates Fusion lifecycle, input/output registration, IR dumping, and execution through `FusionExecutorCache`.

```cpp
#include <ATEN/ATen.h>                     // For at::Tensor
#include <c10/cuda/CUDACachingAllocator.h>

#include <fusion.h>                         // nvfuser::Fusion
#include <fusion_guard.h>                   // nvfuser::FusionGuard
#include <ir/builder.h>                     // nvfuser::IrBuilder helpers
#include <ops/arith.h>                      // nvfuser::add, etc.
#include <runtime/executor_kernel_arg.h>    // nvfuser::KernelArgumentHolder
#include <runtime/fusion_executor_cache.h>  // nvfuser::FusionExecutorCache

using namespace nvfuser;

int main() {
  // 1) Create a Fusion container and set it active via FusionGuard.
  //    The active Fusion collects all IR nodes we define below.
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // 2) Define symbolic 2D input tensors (dtype: float). We use TensorViewBuilder
  //    to declare rank and dtype. Concrete extents will come from runtime inputs.
  auto tv0 = TensorViewBuilder()
                 .ndims(2)                 // rank-2 tensor
                 .dtype(DataType::Float)   // float32
                 .build();
  auto tv1 = TensorViewBuilder().ndims(2).dtype(DataType::Float).build();

  // 3) Register them as Fusion inputs so the executor knows to bind data.
  fusion_ptr->addInput(tv0);
  fusion_ptr->addInput(tv1);

  // 4) Define computation: out = tv0 + tv1 (elementwise add).
  //    Broadcasting rules are handled by IR if shapes require it.
  auto out = add(tv0, tv1);

  // 5) Mark the output tensor.
  fusion_ptr->addOutput(out);

  // (Optional) Print IR to understand what was built.
  fusion_ptr->print();           // math + transforms
  fusion_ptr->printTransforms(); // scheduling transforms (empty initially)

  // 6) Prepare runtime inputs using ATen. These must live on CUDA device.
  const int H = 4, W = 8;
  at::Tensor A = at::rand({H, W}, at::device(at::kCUDA).dtype(at::kFloat));
  at::Tensor B = at::rand({H, W}, at::device(at::kCUDA).dtype(at::kFloat));

  // 7) Pack inputs into KernelArgumentHolder.
  KernelArgumentHolder args;
  args.push(A);
  args.push(B);

  // 8) Build an executor cache from the Fusion and run. The cache handles
  //    concretization, segmentation, scheduling, compilation, and reuse.
  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(args);

  // 9) Retrieve the single output tensor as ATen tensor.
  //    outputs is a KernelArgumentHolder; the first element is our result.
  at::Tensor C = outputs[0].toTensor();

  // 10) Optionally fetch generated CUDA or scheduled IR for inspection.
  // std::string cuda_src = fec.getMostRecentCode();
  // std::string sched_ir = fec.getMostRecentScheduledIr(/*tensor_transforms=*/true);

  // 11) Basic correctness check against PyTorch reference.
  at::Tensor ref = A + B;
  TORCH_CHECK(at::allclose(C, ref, 1e-5, 1e-5));
  return 0;
}
```

Key API notes:
- `Fusion` and `FusionGuard` set and manage the active IR container. See `../csrc/fusion.h`, `../csrc/fusion_guard.h`.
- `TensorViewBuilder` creates symbolic tensors. See `TensorView` implementation in `../csrc/tensor_view.cpp`.
- Elementwise add is defined via `add(tv0, tv1)`. See `../csrc/ops/arith.h`.
- `FusionExecutorCache` compiles, caches, and runs the fusion. See `../csrc/runtime/fusion_executor_cache.h`.
- Runtime argument packing uses `KernelArgumentHolder`. See `../csrc/runtime/executor_kernel_arg.h`.

### Build and run (standalone)

Adapt the following to your environment. Ensure you link against ATen and nvFuser libraries and compile with CUDA enabled.

```bash
# Example (adjust include/library paths as needed)
c++ -std=c++17 -O2 -I$NVFUSER_ROOT/csrc -I$PYTORCH_INCLUDE \
    -L$PYTORCH_LIB -lc10 -ltorch -ltorch_cpu -lcuda \
    tv_add_2d.cpp -o tv_add_2d

./tv_add_2d
```

If you have project-specific helper scripts, prefer those:
- `../doc-bot/how_to_build`
- `../doc-bot/how_to_run`
- `../doc-bot/setup_libs_to_run`

### Common questions and pitfalls

- How are shapes specified? The IR is symbolic. Actual extents come from runtime `at::Tensor` inputs.
- Can I see IR and CUDA? Yes: `Fusion::print()`, `Fusion::printTransforms()`, and `FusionExecutorCache::getMostRecentCode()` / `getMostRecentScheduledIr()`.
- Do I need to schedule? For simple examples, auto-scheduler suffices. For performance work, introduce manual schedule transforms later.
- CPU tensors? Inputs should be CUDA tensors; nvFuser generates CUDA kernels.

### Next steps

- Extend to broadcasting shapes (e.g., (H, W) + (1, W)).
- Print scheduled IR and reason about transforms.
- Explore manual scheduling and vectorization once comfortable with basics.

Canonical samples are available in `../doc-bot/experimenting/8-22-2025`.

### Canonical sample references

Sample 1 (IR build, reshape/transpose/add, IR printing):

```35:47:../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp
  std::cout << "\n=== Fusion IR (with transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/true);

  std::cout << "\n=== Fusion IR (without transforms) ===\n";
  fusion.print(std::cout, /*include_tensor_transforms=*/false);
```

Sample 2 (executor usage and printing output tensor):

```103:112:../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp
  FusionExecutorCache fec(std::move(fusion_ptr));
  KernelArgumentHolder outs = fec.runFusionWithInputs(args);

  std::cout << "\n=== Execution Output (C) ===\n";
  at::Tensor Cout = outs[0].as<at::Tensor>();
  std::cout << Cout.to(at::kCPU) << "\n";
```


