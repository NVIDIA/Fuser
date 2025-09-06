<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Multi-GPU Support in nvFuser

## Introduction

A key strength of nvFuser has been its ability to automatically optimize CUDA
kernels by parallelizing them across threads and blocks. This is achieved by
cleanly separating the *definition* of what to compute from the *schedule* that
determines how to execute it efficiently.

We believe this principle extends naturally from single-GPU to multi-GPU
workloads. For instance, online softmax has been the core idea behind
FlashAttention for single-GPU and context parallelism for multi-GPU. Programmers
also strive to overlap communication and GEMMs, much like overlaping TMA
loads/stores with tensor-core operations.

Therefore, since 2024, we have been generalizing these representations and
algorithms -- originally built for single-GPU execution -- to enable efficient
parallelization of deep learning workloads across multiple GPUs.

## User API

The following example demonstrates how to run a distributed GPT-3 style MLP
block using nvFuser's multi-GPU API.

```python
def define_fusion(fd: FusionDefinition):
    inp = fd.define_tensor([-1, -1, h])
    up_w = fd.define_tensor([h * 4, h])
    out = fd.ops.linear(inp, up_w)

    # `gelu` runs a series of pointwise operations. For simplicity, I omit the
    # details and treat it as one pointwise operation.
    out = gelu(out)

    down_w = fd.define_tensor([h, h * 4])
    out = fd.ops.linear(out, down_w)

    fd.add_output(out)

inp_dtensors: list[DTensor]
fdw = FusionDefinitionWrapper(define_fusion)
out_dtensors: list[DTensor] = fdw(inp_dtensors)
```

A user initializes a
[`FusionDefinitionWrapper`](https://github.com/NVIDIA/Fuser/blob/84c46fed9256b94c4eb9c7aa7f5757056dc88783/tests/python/multidevice/fusion_definition_wrapper.py#L21)
with a **single-GPU** fusion definition. Then, they can invoke it with a list of
three `DTensor` objects -- corresponding to the input activations, the
up-projection weights, and the down-projection weights. The result is a list
containing a single `DTensor` that represents the output activations of the MLP
block.

Under the hood, nvFuser derives a **multi-GPU** schedule from the definition and
the input DTensors, then executes that schedule across multiple GPUs. This
automatically handles sharding and communication and therefore removes the need
for users to explicitly orchestrate communications such as
`torch.distributed.all_reduce`.

By default, nvFuser strives to generate an efficient schedule automatically.
However, for performance-critical workloads, we plan to let users guide
scheduling by providing partial schedules for selected intermediate and/or
output `TensorView`s. This can be done through the scheduling Python API, using
primitives such as `TensorView.split` and `IterDomain.parallelize`.

## Parallelisms

This section goes through several common parallelisms and see how
they can be represented and implemented in fusion IR.

### Tensor Parallelism

To apply [tensor
parallelism](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#tensor-parallelism), the user calls the `FusionDefinitionWrapper` with the following DTensors:
* `inp` with placement `Replicated()`
* `up_w` with placement `Shard(0)`
* `down_w` with placement `Shard(1)`

Therefore, nvFuser starts with the following fusion IR:
```
 inp: [t, h]                          up_w: [4h,  h]
                                             /\
                                            d
                        |
                        | linear
                        |
                     [t, 4h, r{h}]
                        |
                        | gelu
                        |
                     [t, 4h]                down_w: [h, 4h]
                                                        /\
                                                       d
                                       |
                                       | linear
                                       |
                                  [t, h, r{4h}]
```

Then, nvFuser propagates sharding from inputs to outputs:
```
 in: [t, h]                           up_w: [4h,  h]
                                             /\
                                            d
                        |
                        | linear
                        |
                     [t, 4h, r{h}]
                         /\.
                        d
                        |
                        | gelu
                        |
                     [t, 4h]                down_w: [h, 4h]
                         /\                             /\
                        d                              d
                                       |
                                       | linear
                                       |
                                  [t, h, r{4h}]
                                           /\
                                        r{d}
```

Then, nvFuser decomposes every Expr that does both computation and communication
so every Expr can be lowered to either host IR or kernel IR. In the above fusion
IR, the second `linear` does both intra-GPU reduction and inter-GPU reduction.
Therefore, after decomposition, the fusion IR becomes:
```
 in: [t, h]                           up_w: [4h,  h]
                                             /\
                                            d
                        |
                        | linear
                        |
                     [t, 4h, r{h}]
                         /\.
                        d
                        |
                        | gelu
                        |
                     [t, 4h]                down_w: [h, 4h]
                         /\                             /\
                        d                              d
                                       |
                                       | linear
                                       |
                               [t, h, d, r{4h/d}]
                                       |
                                       | sum
                                       |
                                  [t, h, r{d}]
```

This fusion IR then goes through segmentation, (intra-GPU) scheduling, device
lowering, and host IR lowering. Eventually, the `linears` and the `gelu` become
CUDA kernels and the `sum` becomes a call to `ncclAllReduce`.
