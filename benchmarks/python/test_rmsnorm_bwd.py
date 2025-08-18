# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import (
    run_benchmark,
    clear_dynamo_cache,
    unary_bwd_torch,
    with_executor,
    DEFAULT_EXECUTORS,
)
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np
from .torch_ops import rmsnorm


def rmsnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T4 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T5 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )
    T6 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T7 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    S0 = fd.define_scalar(2.0, dtype=DataType.Double)

    if dtype in PROMOTE_DTYPES:
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
        T6 = fd.ops.cast(T6, dtype=DataType.Float)
        T7 = fd.ops.cast(T7, dtype=DataType.Float)

    T14 = fd.ops.broadcast_in_dim(T5, shape=T4.shape(), broadcast_dims=[0, 1])
    T15 = fd.ops.reciprocal(T14)
    T16 = fd.ops.mul(T4, T15)
    T20 = fd.ops.broadcast_in_dim(T7, shape=T4.shape(), broadcast_dims=[1])

    T23 = fd.ops.mul(T6, T16)
    T24 = fd.ops.mul(T6, T20)
    T25 = fd.ops.sum(T23, dims=[0], keepdim=False, dtype=DataType.Null)

    T28 = fd.ops.mul(T24, T15)
    T29 = fd.ops.neg(T24)
    T30 = fd.ops.mul(T29, T4)
    T32 = fd.ops.pow(T14, S0)
    T33 = fd.ops.reciprocal(T32)
    T34 = fd.ops.mul(T30, T33)
    T35 = fd.ops.sum(T34, dims=[1], keepdim=False, dtype=DataType.Null)
    V39 = fd.define_vector([T4.size(0), 1], dtype=DataType.Int)
    T41 = fd.ops.broadcast_in_dim(T35, shape=V39, broadcast_dims=[0])
    T43 = fd.ops.mul(S0, T5)
    T44 = fd.ops.reciprocal(T43)
    T45 = fd.ops.mul(T41, T44)
    S48 = fd.ops.reciprocal(T4.size(1))
    T49 = fd.ops.mul(T45, S48)
    T50 = fd.ops.sum(T49, dims=[1], keepdim=False, dtype=DataType.Null)
    T54 = fd.ops.broadcast_in_dim(T50, shape=V39, broadcast_dims=[0])
    T58 = fd.ops.broadcast_in_dim(T54, shape=T4.shape(), broadcast_dims=[0, 1])
    T59 = fd.ops.mul(T58, S0)
    T62 = fd.ops.mul(T59, T4)
    T63 = fd.ops.add(T28, T62)

    if dtype in PROMOTE_DTYPES:
        T63 = fd.ops.cast(T63, dtype=dtype)
        T25 = fd.ops.cast(T25, dtype=dtype)

    fd.add_output(T63)
    fd.add_output(T25)


def rmsnorm_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOBytes computation since nvfuser input/outputs (in_tensor, grad_out, rms, weigts, grad_in, grad_weights) differ from baselines (out, grad_out)
    # Total IO bytes = in_tensor (size, dtype) + grad_out (size, dtype) + rms_eps(size[0], float) + weights (size[1], dtype) +
    #       grad_in (size, dtype) + grad_weights (size[1], dtype)
    return int(
        dtype.itemsize * (3 * np.prod(size) + 2 * size[1])
        + torch.float.itemsize * size[0]
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_rmsnorm_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    squared_mean = (inputs.to(torch.float) ** 2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + eps)

    with FusionDefinition() as fd:
        rmsnorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = weights.to(torch.double) * (
            inputs.to(torch.double) / rms_eps.to(torch.double)
        )
        eager_output.backward(grads.to(torch.double))
        fd.validate([inputs, rms_eps, grads, weights], [inputs.grad, weights.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, rms_eps, grads, weights])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
@pytest.mark.inner_persistent
@pytest.mark.pointwise
@pytest.mark.reduction
@pytest.mark.transpose
def test_rmsnorm_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, rmsnorm)
    fwd_inputs = [inputs, weights]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=rmsnorm_bwd_iobytes(size, dtype),
    )
