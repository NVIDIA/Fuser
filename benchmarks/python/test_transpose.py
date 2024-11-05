# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def transpose_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    axes: list,
):
    T0 = fd.define_tensor(
        shape=[-1, -1, -1], contiguity=[True, True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1, -1], contiguity=[True, True, True], dtype=dtype, is_cpu=False
    )

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T4 = fd.ops.add(T0, T1)
    T5 = fd.ops.permute(T4, dims=axes)

    if dtype in PROMOTE_DTYPES:
        T5 = fd.ops.cast(T5, dtype=dtype)

    S6 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T7 = fd.ops.gt(T5, S6)
    T9 = fd.ops.where(T7, T5, S6)

    fd.add_output(T9)


def transpose_fwd_fn(inputs: list):  # [input1, input2, dim0, dim1]
    return torch.nn.functional.relu(
        torch.transpose(inputs[0] + inputs[1], inputs[2], inputs[3])
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=3))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
def test_transpose_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    axes: list,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    input2 = torch.randn(size, device="cuda", dtype=dtype)
    permute_axes = list(range(len(size)))
    permute_axes[axes[0]], permute_axes[axes[1]] = (
        permute_axes[axes[1]],
        permute_axes[axes[0]],
    )

    with FusionDefinition() as fd:
        transpose_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), permute_axes)

    if not disable_validation:
        eager_output = transpose_fwd_fn([input1, input2, axes[0], axes[1]])
        fd.validate([input1, input2], [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [input1, input2])


@pytest.mark.parametrize("executor", ["eager", "torchcompile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=3))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
def test_transpose_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    axes: list,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    input1 = torch.randn(size, device="cuda", dtype=dtype)
    input2 = torch.randn(size, device="cuda", dtype=dtype)

    benchmark_fn = {
        "eager": transpose_fwd_fn,
        "torchcompile": torch.compile(transpose_fwd_fn),
    }

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        benchmark_fn[executor],
        [input1, input2, axes[0], axes[1]],
    )
