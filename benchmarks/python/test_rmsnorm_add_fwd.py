# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import (
    run_benchmark,
    clear_dynamo_cache,
    with_executor,
    DEFAULT_EXECUTORS,
    check_module_available,
)
import torch
from .global_params import FLOAT_DTYPES, BENCHMARK_CONFIG
from .torch_ops import rmsnorm_add
import itertools
from random import sample
from typing import List, Tuple

if check_module_available("flashinfer"):
    from flashinfer import fused_add_rmsnorm


def generate_input_sizes_rmsnorm_add(dims: int = 2) -> List[Tuple]:
    """
    Local version of generate_input_sizes specifically for rmsnorm_add_fwd tests.
    Uses batch sizes [1, 2, 4, 8, 16, 32, ..., 16384] (powers of 2).
    Uses hidden sizes [256, 512, 768, 1024, ..., 16384] (step size 256).
    """
    inputs = []

    if dims == 2:
        input_ranges = []

        step_size = 256
        # Use powers of 2 from 1 to 16384 for batch sizes
        batch_range = [2**i for i in range(0, 15)]  # [1, 2, 4, 8, 16, 32, ..., 16384]

        # Hidden range is always [256, 512, 768, 1024, ..., 16384]
        hidden_range = list(range(step_size, 16384 + 1, step_size))

        # Use the same hidden range for all batch sizes since it's already capped at 16384
        input_ranges.append((batch_range, hidden_range))

        for batch_range_subset, hidden_range_subset in input_ranges:
            inputs.extend(
                list(itertools.product(batch_range_subset, hidden_range_subset))
            )
    else:
        raise NotImplementedError(
            f"Generating input sizes of dimension {dims} is not implemented for rmsnorm_add"
        )

    if BENCHMARK_CONFIG["num_inputs"] is not None:
        inputs = sample(inputs, BENCHMARK_CONFIG["num_inputs"])

    return inputs


# flash infer does inplace update of inputs and residual tensors
# needs to explicitly return the updated tensors to correctly compute IO bytes
# https://github.com/flashinfer-ai/flashinfer/blob/ba2b4aa636c4ecf99981794767ffbf89267720cd/flashinfer/norm.py#L106
def flashinfer_rmsnorm_add_wrapper(inputs_list):
    from flashinfer import fused_add_rmsnorm

    inputs, weights, residual = inputs_list
    fused_add_rmsnorm(inputs, residual, weights)
    return inputs, residual


@pytest.mark.parametrize(
    "executor",
    DEFAULT_EXECUTORS
    + [
        pytest.param(
            "flashinfer",
            marks=pytest.mark.skipif(
                not check_module_available("flashinfer"),
                reason="flashinfer executor is not available on this device",
            ),
        )
    ],
)
@pytest.mark.parametrize("size", generate_input_sizes_rmsnorm_add(dims=2))
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
    # requires flashinfer to be installed
    # `pip install flashinfer-python` works in pjnl container
    if executor == "flashinfer":
        benchmark_fn = flashinfer_rmsnorm_add_wrapper
    else:
        benchmark_fn = with_executor(executor, rmsnorm_add)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, weights, residual],
    )
