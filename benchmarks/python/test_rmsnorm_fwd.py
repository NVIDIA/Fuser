# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np
from .torch_ops import rmsnorm


def rmsnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
):
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
    S3 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T4 = fd.ops.pow(T0, S3)
    T5 = fd.ops.sum(T4, dims=[1], keepdim=False, dtype=DataType.Null)
    V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T9 = fd.ops.broadcast_in_dim(T5, shape=V8, broadcast_dims=[0])
    S11 = fd.ops.reciprocal(T0.size(1))
    T12 = fd.ops.mul(T9, S11)
    S13 = fd.define_scalar(eps, dtype=DataType.Double)
    T14 = fd.ops.add(T12, S13)
    T15 = fd.ops.sqrt(T14)

    T20 = fd.ops.broadcast_in_dim(T15, shape=T0.shape(), broadcast_dims=[0, 1])
    T22 = fd.ops.reciprocal(T20)
    T23 = fd.ops.mul(T0, T22)
    T27 = fd.ops.broadcast_in_dim(T1, shape=T0.shape(), broadcast_dims=[1])
    T29 = fd.ops.mul(T27, T23)
    if dtype in PROMOTE_DTYPES:
        T29 = fd.ops.cast(T29, dtype=dtype)

    fd.add_output(T29)
    fd.add_output(T15)


def rmsnorm_fwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOBytes computation required since nvFuser outputs (out, rms) differs from baselines (out)
    # Total IO bytes = in_tensor (size, dtype) + weights (size[1], dtype) +
    #       rms_eps (size[0], float) + outputs (size, dtype)
    return int(
        dtype.itemsize * (2 * np.prod(size) + size[1]) + torch.float.itemsize * size[0]
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        rmsnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        squared_mean = (inputs.to(torch.float) ** 2).mean(1, keepdim=True)
        rms_eps = torch.sqrt(squared_mean + eps)
        eager_output = weights * (inputs / rms_eps)
        fd.validate([inputs, weights], [eager_output.to(dtype), rms_eps])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weights])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    benchmark_fn = with_executor(executor, rmsnorm)
    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, weights],
        iobytes=rmsnorm_fwd_iobytes(size, dtype),
    )
