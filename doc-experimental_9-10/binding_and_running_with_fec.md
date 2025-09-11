## Binding Physical Tensors and Executing via FusionExecutorCache

Use `KernelArgumentHolder` to pass `at::Tensor` inputs to the compiled fusion; retrieve outputs from the return value.

```104:112:/opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp
FusionExecutorCache fec(std::move(fusion_ptr));
KernelArgumentHolder outs = fec.runFusionWithInputs(args);
at::Tensor Cout = outs[0].as<at::Tensor>();
```

Refs: `../csrc/runtime/fusion_executor_cache.h`, `../csrc/runtime/executor_kernel_arg.h`.


