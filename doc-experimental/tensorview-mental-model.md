# TensorView Mental Model: Symbolic View, Not Data

## Key headers
- [`TensorView`](../csrc/ir/interface_nodes.h)

## Core idea
A `TensorView` (TV) is a symbolic description of a tensor’s domains (rank, dtype, contiguity, broadcast/expanded semantics) and the logical/schedule transforms applied to them. TVs do not own or point to data; data only appears when you execute the fusion with real `at::Tensor` inputs.

References:
- Notes: `../doc-bot/experimenting/9-3-2025/key_insights.md`
- Samples: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp`, `../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp`

---

## Why this matters
- IR building is fast and pure (no allocations for tensor contents)
- You can reshape, transpose, squeeze, broadcast at the IR level without moving data
- At execution, nvFuser binds `at::Tensor` arguments to inputs, selects a scheduler, lowers to CUDA/Host IR, and runs

---

## Evidence in code
- `TensorViewBuilder` constructs TVs with dtype/contiguity/rank; no data pointer
- `fusion->addInput(tv)` registers an expected runtime input
- `FusionExecutorCache::runFusionWithInputs(args)` consumes real tensors and produces real outputs

Snippet (heavily commented):

```cpp
// Define a symbolic input [N, N]
TensorView* A = TensorViewBuilder().ndims(2).shape({N, N})
                                   .contiguity({true,true})
                                   .dtype(DataType::Float).build();
fusion->addInput(A); // marks A as runtime input; A has no storage yet

// Build IR: B = transpose(A); C = A + B
// Note: set(A) would insert an explicit copy in IR; not required here.
TensorView* B = transpose(A);
TensorView* C = add(A, B);
fusion->addOutput(C); // mark C as runtime output

// Bind real data at execution time
KernelArgumentHolder args;
args.push(Ain_at_tensor); // order must match fusion->inputs()
FusionExecutorCache fec(std::move(fusion_ptr));
auto outs = fec.runFusionWithInputs(args);
```

---

## Inspecting symbolic transforms
Use `tv->printTransforms()` to see the view/schedule history on any TV (root/logical/loop domains, split/merge/reorder, permute, broadcast, view).

See also:
- `./what-is-view-analysis.md`
- `./squeeze-and-broadcast.md`
- `./fusion-ir-anatomy-and-printer.md`
