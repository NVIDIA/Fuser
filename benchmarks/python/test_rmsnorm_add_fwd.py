# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES
from .torch_ops import rmsnorm_add


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_persistent
def test_rmsnorm_add_fwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=False)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=False)
    residual = torch.randn(size, device="cuda", dtype=dtype, requires_grad=False)

    benchmark_fn = with_executor(executor, rmsnorm_add)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, weights, residual],
    )
