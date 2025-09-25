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
from .global_params import generate_input_sizes, FLOAT_DTYPES
import numpy as np
from .torch_ops import softmax


def softmax_bwd_fusion(
    fd: FusionDefinition, dtype: DataType, reduction_axis: int
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )

    if dtype is not DataType.Float:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T4 = fd.ops.mul(T0, T1)
    T5 = fd.ops.sum(T4, dims=[reduction_axis], keepdim=False, dtype=DataType.Null)

    if reduction_axis:
        V9 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    else:
        V9 = fd.define_vector([1, T0.size(1)], dtype=DataType.Int)
    bcast_dim = 1 - reduction_axis

    T10 = fd.ops.broadcast_in_dim(T5, shape=V9, broadcast_dims=[bcast_dim])

    if dtype is not DataType.Float:
        T10 = fd.ops.cast(T10, dtype=dtype)

    V15 = fd.define_vector([T0.size(0), T0.size(1)], dtype=DataType.Int)
    T16 = fd.ops.broadcast_in_dim(T10, shape=V15, broadcast_dims=[0, 1])

    if dtype is not DataType.Float:
        T16 = fd.ops.cast(T16, dtype=DataType.Float)

    T18 = fd.ops.sub(T1, T16)
    T19 = fd.ops.mul(T0, T18)

    if dtype is not DataType.Float:
        T19 = fd.ops.cast(T19, dtype=dtype)
    fd.add_output(T19)


def softmax_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Total IO bytes = output + grad_out + grad_input
    return int(np.prod(size) * dtype.itemsize * 3)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "reduction_axis",
    [
        pytest.param(0, marks=pytest.mark.outer_persistent),
        pytest.param(1, marks=pytest.mark.inner_persistent),
    ],
)
def test_softmax_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [
        torch.randn(size, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(size, device="cuda", dtype=dtype),
    ]

    with FusionDefinition() as fd:
        softmax_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = torch.nn.functional.softmax(inputs[0], dim=reduction_axis)
        eager_output.backward(inputs[1])
        fd.validate([eager_output, inputs[1]], [inputs[0].grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "reduction_axis",
    [
        pytest.param(0, marks=pytest.mark.outer_persistent),
        pytest.param(1, marks=pytest.mark.inner_persistent),
    ],
)
def test_softmax_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size, device="cuda", dtype=dtype)

    fwd_fn = with_executor(executor, softmax)
    fwd_inputs = [inputs, reduction_axis]
    outputs = fwd_fn(fwd_inputs)

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=softmax_bwd_iobytes(size, dtype),
    )
