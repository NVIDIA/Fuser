# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .cross_entropy_loss import cross_entropy_loss_setup

@pytest.mark.parametrize(
    "variation",
    [
        "hf_qwen2",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
def test_rope_fwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    model, gen_inputs, _, _ = cross_entropy_loss_setup[variation](dtype=torch.bfloat16)
    inputs = gen_inputs()

    def fwd_call(inp):
        return model(**inp)

    # Compile the fwd fn for torchcompile
    benchmark_fn = with_executor(executor, fwd_call, **kwargs)
    run_benchmark(benchmark, benchmark_fn, inputs)