# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def bcast_add_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    bcast_axis: int,
    contiguous: bool,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False, stride_order=[0]
    )
    stride_order = [0, 1] if not contiguous else [1, 0]
    T1 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
        stride_order=stride_order,
    )

    bcast_shape = [T1.size(0), T1.size(1)]
    bcast_shape[bcast_axis] = 1

    # For outer broadcast, if the fn is x + bias, then Thunder only produces one bcast op.
    # The use of unsqueeze generates 2 broadcast ops.
    T2 = fd.ops.broadcast_in_dim(T0, shape=bcast_shape, broadcast_dims=[1 - bcast_axis])
    T3 = fd.ops.broadcast_in_dim(T2, shape=T1.shape(), broadcast_dims=[0, 1])

    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T3 = fd.ops.cast(T3, dtype=DataType.Float)

    T4 = fd.ops.add(T1, T3)
    if dtype in PROMOTE_DTYPES:
        T4 = fd.ops.cast(T4, dtype=dtype)
    fd.add_output(T4)


def bcast_add_fwd_fn(inputs: list):  # bias, x, bcast_dim
    bias, x, bcast_axis = inputs
    return x + bias.unsqueeze(bcast_axis)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bcast_axis", [0, 1], ids=["outer", "inner"])
@pytest.mark.parametrize(
    "contiguous", [True, False], ids=["contiguous", "non-contiguous"]
)
def test_bcast_add_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    bcast_axis: int,
    contiguous: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    bias = torch.randn(size[1 - bcast_axis], dtype=dtype, device="cuda")

    input_shape = size if contiguous else (size[1], size[0])
    x = torch.randn(input_shape, dtype=dtype, device="cuda")
    if not contiguous:
        x = x.t()
    assert x.is_contiguous() == contiguous
    with FusionDefinition() as fd:
        bcast_add_fusion(
            fd,
            torch_dtype_to_nvfuser_dtype(dtype),
            bcast_axis=bcast_axis,
            contiguous=contiguous,
        )

    if not disable_validation:
        eager_output = bcast_add_fwd_fn([bias, x, bcast_axis])
        fd.validate([bias, x], [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [bias, x])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("bcast_axis", [0, 1], ids=["outer", "inner"])
@pytest.mark.parametrize(
    "contiguous", [True, False], ids=["contiguous", "non-contiguous"]
)
def test_bcast_add_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    bcast_axis: int,
    contiguous: bool,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    bias = torch.randn(size[1 - bcast_axis], dtype=dtype, device="cuda")
    input_shape = size if contiguous else (size[1], size[0])
    x = torch.randn(input_shape, dtype=dtype, device="cuda")
    if not contiguous:
        x = x.t()
    assert x.is_contiguous() == contiguous

    benchmark_fn = with_executor(executor, bcast_add_fwd_fn)

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        benchmark_fn,
        [bias, x, bcast_axis],
    )
