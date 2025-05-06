# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache

from .rope_ops import rope_setup, SEQ_LENGTHS


@pytest.mark.parametrize(
    "variation",
    [
        "llama_2_7b_hf",
        "llama_3_8B",
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
        "litgpt-gemma-2-9b",
        "litgpt-mistral-7b",
        "litgpt-meta-llama-3-8B",
        "litgpt-phi3.5-mini",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.resize
def test_rope_fwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
    seq_length: int | None,
):
    kwargs = {}
    if executor == "thunder":
        kwargs["nv_enable_matmul"] = True
    elif executor == "torchcompile":
        clear_dynamo_cache()

    model, gen_inputs, _, _ = rope_setup[variation](seq_length)
    inputs = gen_inputs()

    def fwd_call(inp):
        return model(*inp)

    # Compile the fwd fn for torchcompile
    benchmark_fn = with_executor(executor, fwd_call, **kwargs)
    run_benchmark(benchmark, benchmark_fn, inputs)


@pytest.mark.parametrize(
    "variation",
    [
        "llama_2_7b_hf",
        "llama_3_8B",
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
        "litgpt-gemma-2-9b",
        "litgpt-mistral-7b",
        "litgpt-meta-llama-3-8B",
        "litgpt-phi3.5-mini",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.resize
def test_rope_bwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
    seq_length: int | None,
):
    kwargs = {}
    if executor == "thunder":
        kwargs["nv_enable_matmul"] = True
    elif executor == "torchcompile":
        clear_dynamo_cache()

    model, gen_inputs, grad, iobytes = rope_setup[variation](seq_length)
    fwd_inputs = gen_inputs()

    def fwd_call(inp):
        return model(*inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs)

    # accumulate all output, so we can feed a single grad and use the unary bwd function
    output = outputs[0]
    for i in range(1, len(outputs)):
        output += outputs[i]

    # NOTE: the iobytes is computed based on how thunder autograd worked. So this is just
    # a reference point for torchcompile and eager executor for comparison.
    run_benchmark(
        benchmark, unary_bwd_torch, [output, grad(), *fwd_inputs], iobytes=iobytes()
    )
