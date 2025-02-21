# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES
from .normalization import norm_fwd_nvf_benchmark, norm_fwd_baseline_benchmark


@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_instancenorm_fwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    norm_fwd_nvf_benchmark(
        benchmark,
        size,
        dtype,
        "instance_norm",
        channels_last,
        disable_validation,
        disable_benchmarking,
    )


@pytest.mark.parametrize("executor", ["eager", "torchcompile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_instancenorm_fwd_baseline_benchmark(
    benchmark, size: tuple, dtype: torch.dtype, channels_last: bool, executor: str
):
    norm_fwd_baseline_benchmark(
        benchmark, size, dtype, channels_last, executor, "instance_norm"
    )
