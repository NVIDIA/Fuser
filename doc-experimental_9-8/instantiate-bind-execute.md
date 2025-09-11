# Instantiate TensorViews, Bind Tensors, Execute via FusionExecutorCache

## Key types
- [`TensorView`](../csrc/ir/interface_nodes.h)
- [`Fusion`](../csrc/fusion.h), [`FusionGuard`](../csrc/fusion_guard.h)
- [`FusionExecutorCache`](../csrc/runtime/fusion_executor_cache.h)
- [`KernelArgumentHolder`](../csrc/runtime/executor_kernel_arg.h)

## Overview
- Instantiate TVs with `TensorViewBuilder`
- Register inputs/outputs on the Fusion
- Bind real `at::Tensor` arguments in order
- Execute via `FusionExecutorCache` (compiles and launches kernels)

References:
- Sample: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp`
- Notes: `../doc-bot/experimenting/9-3-2025/key_insights.md`

---

## Heavily commented example

```cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ops/arith.h>
#include <ops/alias.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/executor_kernel_arg.h>
#include <ATen/ATen.h>

using namespace nvfuser;

int main(){
  constexpr int64_t N = 31;

  // 1) Build IR inside a movable Fusion
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion); // sets active Fusion context

  // 2) Instantiate a symbolic input TV A: rank-2, float, contiguous
  TensorView* A = TensorViewBuilder().ndims(2).shape({N, N})
                                     .contiguity({true,true})
                                     .dtype(DataType::Float).build();
  fusion->addInput(A); // register as runtime input (ordering matters)

  // 3) Build IR: B = transpose(A); C = A + B
  TensorView* B = transpose(A);
  TensorView* C = add(A, B);
  fusion->addOutput(C); // register as runtime output

  // 4) Prepare real input on CUDA and bind via KernelArgumentHolder
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  at::Tensor Ain = at::arange(1, (int64_t)N * N + 1, opts).view({N, N});

  KernelArgumentHolder args;
  args.push(Ain); // push inputs in the same order as fusion->inputs()

  // 5) Execute: JIT compile + run
  FusionExecutorCache fec(std::move(fusion_ptr));
  KernelArgumentHolder outs = fec.runFusionWithInputs(args);

  // 6) Retrieve outputs
  at::Tensor Cout = outs[0].as<at::Tensor>();
  std::cout << "=== C ===\n" << Cout.to(at::kCPU) << "\n";
}
```

---

## Gotchas and tips
- Always push inputs in the exact order they were added via `fusion->addInput(...)`
- TVs are symbolic; `at::Tensor` provides real storage only at execution
- Use `printTransforms()` to understand how view/schedule transforms affected a TV

See also:
- `./tensorview-mental-model.md`
- `./build-and-run-standalone-samples.md`
