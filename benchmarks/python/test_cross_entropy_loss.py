# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch
import thunder

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .cross_entropy_loss import cross_entropy_loss_setup


@pytest.mark.parametrize(
    "variation",
    [
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
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


@pytest.mark.parametrize(
    "variation",
    [
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
def test_rope_bwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    model, gen_inputs, grad, iobytes = cross_entropy_loss_setup[variation](
        dtype=torch.bfloat16
    )
    fwd_inputs = gen_inputs()

    def fwd_call(inp):
        return model(**inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs)

    assert len(outputs) == 1

    # NOTE: the iobytes is computed based on how thunder autograd worked. So this is just
    # a reference point for torchcompile and eager executor for comparison.
    # NOTE: passing in *list(model.parameters()), so we would clear all computed grad before
    # calling backwards, this avoid the accumulation kernel
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs[0], grad(), *list(model.parameters())],
        iobytes=iobytes(),
    )
