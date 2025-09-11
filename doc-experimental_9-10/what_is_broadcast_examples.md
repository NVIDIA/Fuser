## What is "Broadcast"? With Examples

Broadcast inserts size‑1 dimensions so tensors can align for elementwise ops.

### API and constraints

```991:1047:/opt/pytorch/nvfuser/csrc/ops/alias.cpp
TensorView* broadcast(TensorView* inp, const std::vector<bool>& is_broadcast_dim);
// The number of false entries must equal the input rank.
```

### Example

```cpp
// Input [4,5]; Broadcast mask {true,false,false} → Output [1,4,5]
auto tv_in  = TensorViewBuilder().ndims(2).shape({4,5}).contiguity({true,true}).dtype(DataType::Float).build();
auto tv_out = broadcast(tv_in, std::vector<bool>{true,false,false});
```

Tip: Combine `squeeze` and `broadcast` to align shapes for elementwise ops.


