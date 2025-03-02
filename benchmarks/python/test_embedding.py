# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .embedding_ops import embedding_setup


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
def test_embedding_fwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()
    if executor == "thunder":
        kwargs["nv_enable_embedding"] = True

    test_case = embedding_setup[variation](dtype=torch.bfloat16)
    inputs = test_case.inputs()
    model = test_case.model()

    def fwd_call(inp):
        return model(**inp)

    # Compile the fwd fn
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
def test_embedding_bwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()
    if executor == "thunder":
        kwargs["nv_enable_embedding"] = True

    test_case = embedding_setup[variation](dtype=torch.bfloat16)
    fwd_inputs = test_case.inputs()
    model = test_case.model()

    def fwd_call(inp):
        return model(**inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs)

    assert len(outputs) == 1

    # NOTE: passing in *list(model.parameters()), so we would clear all computed grad before
    # calling backwards, this avoid the accumulation kernel
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs[0], test_case.grads(), *list(model.parameters())],
        iobytes=test_case.grad_iobytes(),
    )
