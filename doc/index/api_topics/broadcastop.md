# BroadcastOp

## Synopsis
`BroadcastOp` inserts broadcast dimensions to make an input match an output shape. Semantics align with PyTorch-style unsqueeze/broadcast to a target shape.

## Source
- Class: [`BroadcastOp`](../../../csrc/ir/internal_nodes.h#L848)

## Overview
Broadcasting creates new broadcast domains on the output where needed so that a tensor can participate in elementwise or other ops with larger-rank shapes. The `BroadcastOp` keeps a flag per output dimension indicating whether that dimension is a newly introduced broadcast domain or inherited from the input.

Key points:
- Flags are relative to the output’s `IterDomain` list; each flag is true if the output dim is a new broadcast domain.
- Inputs/outputs are `Val` (usually `TensorView`).
- Works in tandem with `SqueezeOp` for the inverse operation.

Key APIs:
- Construction: takes `out`, `in`, and `is_broadcast_dims` vector
- Accessors: `out()`, `in()`
- Flags: `getBroadcastDimFlags()`, `isBroadcastDim(dim)`
- Rendering/eval: `toString`, `toInlineString`, `evaluate(...)`

## Example
```cpp
using namespace nvfuser;
Fusion fusion;
FusionGuard fg(&fusion);

// tv0: [I0, I1]
TensorView* tv0 = makeConcreteTensor({-1, -1});
fusion.addInput(tv0);

// Broadcast tv0 to [1, I0, I1, 1]
auto tv1 = broadcast(tv0, {true, false, false, true});
// Now tv1 has two new broadcast dims at positions 0 and 3.

auto tv2 = add(tv1, tv1);
fusion.addOutput(tv2);
```

## Related
- `SqueezeOp` (inverse)
- `UnaryOp`/`BinaryOp` elementwise ops that consume broadcasted tensors
- `TensorDomain`/`IterDomain` for domain semantics

## References
- Impl: `../../../csrc/ir/internal_nodes.h#L848`
- Usage examples in tests and pointwise schedulers.
