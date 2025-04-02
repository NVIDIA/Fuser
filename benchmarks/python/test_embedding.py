# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch

from .core import (
    run_benchmark,
    clear_dynamo_cache,
    unary_bwd_torch,
    with_executor,
    DEFAULT_EXECUTORS,
)
from .global_params import FLOAT_DTYPES, PROMOTE_DTYPES
from .embedding_ops import embedding_setup, EMBEDDING_CONFIGS, SEQ_LENGTHS
from .torch_ops import embedding


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

@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("vocab_hidden", EMBEDDING_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding_fwd_baseline_benchmark(
    benchmark,
    seq_length: int,
    vocab_hidden: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    
    indices = torch.randint(0, vocab_hidden[0], (seq_length,), device="cuda")
    embedding_table = torch.randn(vocab_hidden, device="cuda", dtype=dtype)

    benchmark_fn = with_executor(executor, embedding)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [indices, embedding_table],
    )

@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("vocab_hidden", EMBEDDING_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding_bwd_baseline_benchmark(
    benchmark,
    seq_length: int,
    vocab_hidden: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    indices = torch.randint(0, vocab_hidden[0], (seq_length,), device="cuda")
    grads = torch.randn((seq_length, vocab_hidden[1]), device="cuda", dtype=dtype)
    embedding_table = torch.randn(vocab_hidden, device="cuda", dtype=dtype)

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, embedding)
    fwd_inputs = [indices, embedding_table]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs[0], grads, *fwd_inputs],
        #iobytes=rmsnorm_bwd_iobytes(size, dtype),
    )
