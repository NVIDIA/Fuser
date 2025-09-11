## What is "Squeeze"? With Examples

Squeeze removes size‑1 broadcast dimensions from a `TensorView`’s logical domain.

### APIs

```252:288:/opt/pytorch/nvfuser/csrc/ops/alias.cpp
TensorView* squeeze(TensorView* x, const std::vector<int64_t>& dims, bool squeeze_expanded);
TensorView* squeeze(TensorView* x, const std::vector<bool>& to_squeeze, bool squeeze_expanded);
```

### Example

```cpp
// Input: [4, 1, 5] → Squeeze dim {1} → Output: [4, 5]
auto tv_in  = TensorViewBuilder().ndims(3).shape({4,1,5}).contiguity({true,true,true}).dtype(DataType::Float).build();
auto tv_out = squeeze(tv_in, std::vector<int64_t>{1});
```

Notes:
- `squeeze_expanded=false` refuses to remove expanded dims unless explicitly allowed
- Non-broadcast dims (extent ≠ 1 or non-broadcast) are rejected


