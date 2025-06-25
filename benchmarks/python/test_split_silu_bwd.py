# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype, clear_cuda_cache
from .core import run_benchmark, clear_dynamo_cache, unary_bwd_torch
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np


def split_silu_bwd_fusion(fd: FusionDefinition, dtype: DataType, size: tuple):
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False, stride_order=[1, 0])
    
    S0, S1 = size
    T2 = fd.ops.slice(T0, start_indices=[0, 0], end_indices=[S0, S1 // 2], strides=[1, 1])
    T4 = fd.ops.slice(T0, start_indices=[0, S1 // 2], end_indices=[S0, S1], strides=[1, 1])
    
    if dtype in PROMOTE_DTYPES:
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)
    
    T3 = fd.ops.neg(T2)
    T5 = fd.ops.exp(T3)
    T6 = fd.ops.mul(T4, T1)
    S7 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T8 = fd.ops.add(S7, T5)
    T9 = fd.ops.mul(T2, T6)
    T10 = fd.ops.reciprocal(T8)
    T11 = fd.ops.neg(T9)
    T12 = fd.ops.mul(T11, T10)
    T13 = fd.ops.mul(T12, T10)
    T14 = fd.ops.mul(T13, T5)
    T15 = fd.ops.mul(T2, T10)
    T16 = fd.ops.neg(T14)
    T17 = fd.ops.mul(T10, T6)
    T18 = fd.ops.mul(T15, T1)
    T19 = fd.ops.add(T17, T16)
    T20 = fd.ops.cat([T19, T18], dim=-1)
    if dtype in PROMOTE_DTYPES:
        T20 = fd.ops.cast(T20, dtype=dtype)
    fd.add_output(T20)


def split_silu_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Total IO bytes = input (size[0], size[1]) + grad_out (size[0], size[1]//2)+ grad_input(size[0], size[1])
    return int(dtype.itemsize * (2 * np.prod(size) + np.prod(size) // 2))


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_split_silu_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    input = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size[0], size[1] // 2, device="cuda", dtype=dtype)
    with FusionDefinition() as fd:
        split_silu_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype), size)
    if not disable_validation:
        x, y = torch.split(input, size[1] // 2, -1)
        eager_output = torch.nn.functional.silu(x) * y
        eager_output.backward(grads)
        fd.validate([input, grads], [input.grad])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [input, grads])


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_split_silu_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    compile: bool,
):
    clear_cuda_cache()
    if compile:
        clear_dynamo_cache()
    input = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(size[0], size[1] // 2, device="cuda", dtype=dtype)
    x, y = torch.split(input, size[1] // 2, -1)
    eager_output = torch.nn.functional.silu(x) * y

    run_benchmark(
        benchmark,
        torch.compile(unary_bwd_torch) if compile else unary_bwd_torch,
        [eager_output, grads],
        iobytes=split_silu_bwd_iobytes(size, dtype),
    )
