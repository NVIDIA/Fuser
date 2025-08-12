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
from .torch_ops import scale_bias_relu


def sbr_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.Bool,
        is_cpu=False,
    )
    T2 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    V7 = T2.shape()
    T8 = fd.ops.broadcast_in_dim(T0, shape=V7, broadcast_dims=[1])

    S10 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T11 = fd.ops.where(T1, T2, S10)
    T15 = fd.ops.mul(T11, T8)

    if dtype in PROMOTE_DTYPES:
        T15 = fd.ops.cast(T15, dtype=dtype)
    fd.add_output(T15)


def sbr_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Total IO bytes = grad_out (dtype, size)+ scale (dtype, size[-1])+ bool_mask (bool, size) + grad_in (dtype, size)
    return int(
        dtype.itemsize * (2 * np.prod(size) + size[-1])
        + torch.bool.itemsize * np.prod(size)
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sbr_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    scale = torch.ones(size[-1], device="cuda", dtype=dtype)
    bias = torch.ones(size[-1], device="cuda", dtype=dtype)
    bool_mask = torch.gt(inputs * scale + bias, 0.0)

    with FusionDefinition() as fd:
        sbr_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.relu(inputs * scale + bias)
        eager_output.backward(grads)
        fd.validate([scale, bool_mask, grads], [inputs.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [scale, bool_mask, grads])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sbr_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    scale = torch.ones(size[-1], device="cuda", dtype=dtype)
    bias = torch.ones(size[-1], device="cuda", dtype=dtype)

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, scale_bias_relu)
    fwd_inputs = [inputs, scale, bias]
    outputs = fwd_fn(fwd_inputs)

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=sbr_bwd_iobytes(size, dtype),
    )
