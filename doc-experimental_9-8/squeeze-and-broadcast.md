# Squeeze (and Broadcast) in nvFuser: Semantics, APIs, Examples

## Key headers
- Squeeze/Broadcast declarations in [`alias.h`](../csrc/ops/alias.h)

## What is "squeeze"?
- Removes dimensions (typically size-1) from a `TensorView`’s logical domain
- It’s a view operation (IR-level) that rewires logical axes; no data move

APIs (from `../csrc/ops/alias.cpp`):
- `TensorView* squeeze(TensorView* x, const std::vector<int64_t>& dims, bool squeeze_expanded=false)`
- `TensorView* squeeze(TensorView* x, const std::vector<bool>& to_squeeze, bool squeeze_expanded=false)`

## What is "broadcast"?
- Inserts broadcast dimensions (logical size 1) so shapes align for elementwise operations
- Another view op (IR-level), implemented in `alias.cpp`

API:
- `TensorView* broadcast(TensorView* inp, const std::vector<bool>& is_broadcast_dim)`

---

## Heavily commented examples

```cpp
// Example 1: Remove a trivial dimension via squeeze
// Input: [4, 1, 5]  →  Output: [4, 5]
auto tv_in  = TensorViewBuilder().ndims(3).shape({4, 1, 5})
                                 .contiguity({true,true,true})
                                 .dtype(DataType::Float).build();
// Remove dimension index 1 (the middle size-1)
auto tv_out = squeeze(tv_in, std::vector<int64_t>{1});
// tv_out logical domain is now [4, 5].
```

```cpp
// Example 2: Align ranks for elementwise via broadcast
// tv_x: [4, 5]  and  tv_y: [5]
auto tv_x = TensorViewBuilder().ndims(2).shape({4, 5})
                                .contiguity({true,true})
                                .dtype(DataType::Float).build();
auto tv_y = TensorViewBuilder().ndims(1).shape({5})
                                .contiguity({true})
                                .dtype(DataType::Float).build();

// Broadcast tv_y to [1, 5] so we can add with tv_x ([4,5])
// broadcast mask length equals output rank; number of false entries equals input rank
auto tv_y2 = broadcast(tv_y, std::vector<bool>{true, false}); // [1,5]

// Elementwise add now aligns: result shape [4, 5]
auto tv_out = add(tv_x, tv_y2);
```

```cpp
// Example 3: Squeeze then broadcast to align differing ranks
// tv_x: [4, 1, 5]  → squeeze dim 1 → [4, 5]
auto tv_x2 = squeeze(tv_x, std::vector<int64_t>{1});
// tv_y: [5] → broadcast to [1, 5]
auto tv_y2 = broadcast(tv_y, std::vector<bool>{true, false});
// Add: [4, 5] + [1, 5] → [4, 5]
auto tv_z  = add(tv_x2, tv_y2);
```

---

## Tips and validation
- The broadcast mask length equals the output rank; ensure the count of `false` entries equals the input rank
- `squeeze_expanded` controls behavior on expanded axes; leave default unless a specific broadcast/expand case needs it
- Inspect `tv->printTransforms()` to confirm view ops in the logical domain history

See also:
- `./what-is-view-analysis.md`
- `./reshape-empty-dimensions.md`
