# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype, clear_cuda_cache
from .core import run_benchmark, clear_dynamo_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def split_silu_fwd_fusion(fd: FusionDefinition, dtype: DataType, size: tuple):
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=dtype,
        is_cpu=False,
        stride_order=[1, 0],
    )
    S0, S1 = size
    T1 = fd.ops.slice(
        T0, start_indices=[0, 0], end_indices=[S0, S1 // 2], strides=[1, 1]
    )
    T2 = fd.ops.slice(
        T0, start_indices=[0, S1 // 2], end_indices=[S0, S1], strides=[1, 1]
    )
    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)
    T3 = fd.ops.neg(T1)
    T4 = fd.ops.exp(T3)
    S5 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T6 = fd.ops.add(S5, T4)
    T7 = fd.ops.reciprocal(T6)
    T8 = fd.ops.mul(T1, T7)
    T9 = fd.ops.mul(T8, T2)
    if dtype in PROMOTE_DTYPES:
        T9 = fd.ops.cast(T9, dtype=dtype)
    fd.add_output(T9)


def split_silu_fwd_fn(inputs: list):  # [in_tensor]
    x, y = torch.chunk(inputs[0], 2, -1)
    return torch.nn.functional.silu(x) * y


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_split_silu_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        split_silu_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), size=size)
    if not disable_validation:
        eager_output = split_silu_fwd_fn(inputs)
        fd.validate(inputs, [eager_output])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_split_silu_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    compile: bool,
):
    clear_cuda_cache()
    if compile:
        clear_dynamo_cache()
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        torch.compile(split_silu_fwd_fn) if compile else split_silu_fwd_fn,
        inputs,
    )
