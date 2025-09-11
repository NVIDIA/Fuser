## Anatomy of Building an IR (FusionGuard, Free Functions)

Pattern:
```cpp
auto fusion_ptr = std::make_unique<Fusion>();
FusionGuard fg(fusion_ptr.get());
auto A = TensorViewBuilder().ndims(2).dtype(DataType::Float).build();
fusion_ptr->addInput(A);
auto B = transpose(set(A));
auto C = add(A, B);
fusion_ptr->addOutput(C);
```

Refs: Example `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp`.


