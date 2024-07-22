# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_cuda_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np

def groupnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    n_groups: int = 32,
    eps: float = 1e-5,
) -> None:
    # inputs, T0-x, T1-weight, T2-bias
    T0 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    # initial input shape: [N, C, H, W]
    # reshape input to [N, n_groups, C//n_groups, H, W] and do normalization with scale and bias
    # reshape back to [N, C, H, W]
    V0 = T0.shape()
    G = fd.define_scalar(n_groups, dtype=DataType.Int)
    CdivG = fd.ops.div(T0.size(1), G)
    V1 = fd.define_vector([T0.size(0), n_groups, CdivG, T0.size(2), T0.size(3)], dtype=DataType.Int)
    T0 = fd.ops.reshape(T0, new_shape=V1)
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    T3, T4 = fd.ops.var_mean(T0, dims=[2, 3, 4], correction=0, keepdim=False)

    V2 = fd.define_vector([T0.size(0), n_groups, 1, 1, 1], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T3, shape=V2, broadcast_dims=[0, 1])
    T11 = fd.ops.broadcast_in_dim(T4, shape=V2, broadcast_dims=[0, 1])

    S12 = fd.define_scalar(eps, dtype=DataType.Double)
    T13 = fd.ops.add(T7, S12)
    T14 = fd.ops.rsqrt(T13)
    # N, G, C//G, H, W
    V3 = T0.shape()
    T18 = fd.ops.broadcast_in_dim(T11, shape=V3, broadcast_dims=[0, 1, 2, 3, 4])
    T19 = fd.ops.sub(T0, T18)
    T23 = fd.ops.broadcast_in_dim(T14, shape=V3, broadcast_dims=[0, 1, 2, 3, 4])
    T24 = fd.ops.mul(T19, T23)

    # reshape weights and bias to [1, n_groups, C//n_groups, 1, 1]
    V4 = fd.define_vector([1, n_groups, CdivG, 1, 1], dtype=DataType.Int)
    T1 = fd.ops.reshape(T1, new_shape=V4)
    T2 = fd.ops.reshape(T2, new_shape=V4)

    # broadcast weights and bias to [N, n_groups, C//n_groups, H, W]
    T25 = fd.ops.broadcast_in_dim(T1, shape=V3, broadcast_dims=[0, 1, 2, 3, 4])
    T26 = fd.ops.mul(T24, T25)
    T27 = fd.ops.broadcast_in_dim(T2, shape=V3, broadcast_dims=[0, 1, 2, 3, 4])
    T28 = fd.ops.add(T26, T27)

    # convert back to original dtype and shape
    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)
    T28 = fd.ops.reshape(T28, new_shape=V0)
    fd.add_output(T28)


def groupnorm_fwd(inputs: list):  # [in_tensor, weights, bias, n_groups]
    # Check if inputs is None
    if inputs is None:
        raise ValueError("The inputs parameter is None")
    
    # Check if inputs is a list and has the expected number of elements
    if not isinstance(inputs, list) or len(inputs) < 4:
        raise ValueError("The inputs parameter must be a list with at least 4 elements")
    
    # Check if each expected element in inputs is not None
    in_tensor, weights, bias, n_groups = inputs
    if in_tensor is None or weights is None or bias is None or n_groups is None:
        raise ValueError("One or more elements in the inputs list are None")    
    return torch.nn.functional.group_norm(
        inputs[0],
        num_groups = inputs[3],
        weight=inputs[1],
        bias=inputs[2],
        eps=1e-05
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_groupnorm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    clear_cuda_cache()

    N, C, H, W = size
    x = torch.randn(size, device="cuda", dtype=dtype)
    weight = torch.randn(C, device="cuda", dtype=dtype)
    bias = torch.randn(C, device="cuda", dtype=dtype)
    # start num_groups from 1 and increase to 32 at max
    num_groups = 1
    while num_groups*2 <= 32 and C % (num_groups*2) == 0:
        num_groups *= 2

    with FusionDefinition() as fd:
        groupnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), num_groups)


    if not disable_validation:
        eager_output = groupnorm_fwd([x, weight, bias, num_groups])
        fd.validate([x, weight, bias], [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [x, weight, bias])


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_groupnorm_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    compile: bool,
):
    clear_cuda_cache()
    N, C, H, W = size
    
    inputs = [
        torch.randn(size, device="cuda", dtype=dtype),
        torch.randn(C, device="cuda", dtype=dtype),
        torch.randn(C, device="cuda", dtype=dtype),
    ]

    # start num_groups from 1 and increase to 32 at max
    num_groups = 1
    while num_groups*2 <= 32 and C % (num_groups*2) == 0:
        num_groups *= 2

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        torch.compile(groupnorm_fwd) if compile else groupnorm_fwd,
        inputs.append(num_groups),
    )
