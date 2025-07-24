# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache, with_executor, DEFAULT_EXECUTORS
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np

# purpose: test fusion with mixed input precisions.
# inputs: [bfloat16, float32]
# output: [bfloat16]
def rmsnorm_x_fp16_weight_fp32_out_fp16(inputs: list):
    inp, weights = inputs
    squared_mean = (inp**2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + 1e-5)
    output = weights * (inp / rms_eps)
    return output.to(dtype=inp.dtype)

@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
def test_rmsnorm_fwd_thunder_xfloat16_weightfp32(
    benchmark,
    size: tuple,
):
    inputs = torch.randn(size, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    weights = torch.randn(size[1], device="cuda", dtype=torch.float32, requires_grad=False)

    benchmark_fn = with_executor("thunder", rmsnorm_x_fp16_weight_fp32_out_fp16)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [inputs, weights]
    )
