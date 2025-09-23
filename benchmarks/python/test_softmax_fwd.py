# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser_direct import FusionDefinition, DataType
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np
from .torch_ops import softmax


def softmax_fwd_fusion(
    fd: FusionDefinition, dtype: DataType, reduction_axis: int
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.max(T0, dims=[reduction_axis], keepdim=False, dtype=DataType.Null)

    if reduction_axis:
        V6 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    else:
        V6 = fd.define_vector([1, T0.size(1)], dtype=DataType.Int)
    bcast_dim = 1 - reduction_axis

    T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[bcast_dim])

    V11 = T0.shape()
    T12 = fd.ops.broadcast_in_dim(T7, shape=V11, broadcast_dims=[0, 1])
    T13 = fd.ops.sub(T0, T12)
    T14 = fd.ops.exp(T13)
    T15 = fd.ops.sum(T14, dims=[reduction_axis], keepdim=False, dtype=DataType.Null)

    T20 = fd.ops.broadcast_in_dim(T15, shape=V6, broadcast_dims=[bcast_dim])
    T25 = fd.ops.broadcast_in_dim(T20, shape=V11, broadcast_dims=[0, 1])

    T26 = fd.ops.reciprocal(T25)
    T27 = fd.ops.mul(T14, T26)

    if dtype in PROMOTE_DTYPES:
        T27 = fd.ops.cast(T27, dtype=dtype)
    fd.add_output(T27)


def softmax_fwd_iobytes(size: tuple, dtype: torch.dtype):
    # Total IO bytes = input + output
    return int(np.prod(size) * dtype.itemsize * 2)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "reduction_axis",
    [
        pytest.param(0, marks=pytest.mark.outer_persistent),
        pytest.param(1, marks=pytest.mark.inner_persistent),
    ],
)
def test_softmax_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        softmax_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis)

    if not disable_validation:
        eager_output = softmax([inputs[0], reduction_axis])
        fd.validate(inputs, [eager_output])

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
def test_softmax_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    reduction_axis: int,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)

    benchmark_fn = with_executor(executor, softmax)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, reduction_axis],
        iobytes=softmax_fwd_iobytes(size, dtype),
    )
