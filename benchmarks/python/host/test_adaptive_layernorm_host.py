# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from ..core import run_benchmark
import torch


def adaptive_layernorm_fwd_fusion(fd: FusionDefinition, eps: float = 1e-6) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1, -1],
        contiguity=[True, True, True],
        dtype=DataType.Half,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Half,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Half,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T3 = fd.ops.cast(T0, dtype=DataType.Float)
    T4, T5 = fd.ops.var_mean(T3, dims=[2], correction=0, keepdim=False)
    T10 = fd.ops.broadcast_in_dim(
        T4, shape=[T0.size(0), T0.size(1), 1], broadcast_dims=[0, 1]
    )
    T15 = fd.ops.broadcast_in_dim(
        T5, shape=[T0.size(0), T0.size(1), 1], broadcast_dims=[0, 1]
    )
    S16 = fd.define_scalar(eps, dtype=DataType.Double)
    T17 = fd.ops.add(T10, S16)
    T22 = fd.ops.broadcast_in_dim(T15, shape=T0.shape(), broadcast_dims=[0, 1, 2])
    T23 = fd.ops.rsqrt(T17)
    T24 = fd.ops.sub(T3, T22)
    T29 = fd.ops.broadcast_in_dim(T23, shape=T0.shape(), broadcast_dims=[0, 1, 2])
    T30 = fd.ops.mul(T24, T29)
    T35 = fd.ops.reshape(T1, new_shape=[T1.size(0), 1, T1.size(1)])
    T36 = fd.ops.cast(T35, dtype=DataType.Float)
    S37 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T38 = fd.ops.add(S37, T36)
    T39 = fd.ops.cast(T38, dtype=DataType.Half)
    T44 = fd.ops.broadcast_in_dim(T39, shape=T0.shape(), broadcast_dims=[0, 1, 2])
    T45 = fd.ops.cast(T44, dtype=DataType.Float)
    T46 = fd.ops.mul(T30, T45)
    T51 = fd.ops.reshape(T2, new_shape=[T2.size(0), 1, T2.size(1)])
    T56 = fd.ops.broadcast_in_dim(T51, shape=T0.shape(), broadcast_dims=[0, 1, 2])
    T57 = fd.ops.cast(T56, dtype=DataType.Float)
    T58 = fd.ops.add(T46, T57)
    T59 = fd.ops.cast(T58, dtype=DataType.Half)
    fd.add_output(T5)
    fd.add_output(T23)
    fd.add_output(T59)


# This benchmark is to particularly track nvFuser host overhead for shape
# change (dynamic shape support) in the adapative layernorm case. Running a
# new shape on this fusion without recompiling a new kernel can have significant overhead.
@pytest.mark.parametrize("host_bench_mode", ["compile", "steady", "dynamic"])
def test_adaptive_layernorm_fwd_benchmark(
    benchmark,
    host_bench_mode: str,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    B = 1
    T = 30 * 1024
    D = 1024
    inputs = [
        torch.randn(B, T, D, device="cuda", dtype=torch.float16, requires_grad=True),
        torch.randn(B, D, device="cuda", dtype=torch.float16, requires_grad=True),
        torch.randn(B, D, device="cuda", dtype=torch.float16, requires_grad=True),
    ]

    # Generate multiple inputs to measure dynamic shape overhead.
    if host_bench_mode == "dynamic":
        inputs = []
        for B in range(1, 3, 1):
            for T in range(30 * 1024, 30 * 1024 + 5 * 128, 128):
                inputs.append(
                    [
                        torch.randn(
                            B,
                            T,
                            D,
                            device="cuda",
                            dtype=torch.float16,
                            requires_grad=True,
                        ),
                        torch.randn(
                            B, D, device="cuda", dtype=torch.float16, requires_grad=True
                        ),
                        torch.randn(
                            B, D, device="cuda", dtype=torch.float16, requires_grad=True
                        ),
                    ]
                )

    with FusionDefinition() as fd:
        adaptive_layernorm_fwd_fusion(fd)

    def validate(input):
        eps = 1e-6
        in_tensor, scale, shift = input
        norm_state = torch.nn.LayerNorm(D, elementwise_affine=False, eps=eps)
        norm_out = norm_state(in_tensor)
        mean = in_tensor.to(torch.float).mean(dim=-1)
        variance = in_tensor.to(torch.float).var(dim=-1, unbiased=False)
        invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(-1)
        eager_output = norm_out * (1 + scale.view(-1, 1, D)) + shift.view(-1, 1, D)
        fd.validate(input, [mean, invstd, eager_output])

    if not disable_validation:
        if host_bench_mode == "dynamic":
            # Run validate for all input sizes.
            for input in inputs:
                validate(input)
        else:
            validate(inputs)

    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            None,
            inputs,
            device=f"host:{host_bench_mode}",
            fusion_fn=adaptive_layernorm_fwd_fusion,
        )
