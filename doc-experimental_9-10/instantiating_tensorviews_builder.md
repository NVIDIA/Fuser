## Instantiating TensorViews (TensorViewBuilder); Cannot Import from at::Tensor

Create symbolic inputs with `TensorViewBuilder`; you cannot “wrap” an `at::Tensor` as a `TensorView` directly.

Example:
```cpp
auto tv = TensorViewBuilder().ndims(2).dtype(DataType::Float).build();
fusion->addInput(tv);
```

At runtime:
```cpp
KernelArgumentHolder args; args.push(Ain);
FusionExecutorCache fec(std::move(fusion_ptr));
auto outs = fec.runFusionWithInputs(args);
```

Refs: `../csrc/ops/alias.cpp`, `../csrc/tensor_view.cpp`.


