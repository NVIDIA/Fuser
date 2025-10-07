# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .cross_entropy_loss import (
    cross_entropy_loss_setup,
    SyntheticMiniModel,
)
from .torch_ops import cross_entropy as torch_cross_entropy_fwd
from quack.cross_entropy import cross_entropy_fwd as quack_cross_entropy_fwd


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
def test_cross_entropy_fwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    test_case = cross_entropy_loss_setup[variation](dtype=torch.bfloat16)
    inputs = test_case.inputs()
    model = test_case.model()

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
def test_cross_entropy_bwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    test_case = cross_entropy_loss_setup[variation](dtype=torch.bfloat16)
    fwd_inputs = test_case.inputs()
    model = test_case.model()

    def fwd_call(inp):
        return model(**inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs)

    assert len(outputs) == 1

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs[0], test_case.grads(), *fwd_inputs, *list(model.parameters())],
        iobytes=test_case.grad_iobytes(),
    )


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_cross_entropy_mini_benchmark_fwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    # picking a value that doesn't OOM for large vocab sizes
    batch_size = 4096

    def fwd_call(inp):
        return SyntheticMiniModel.mini_model(*inp)

    inputs = SyntheticMiniModel.inputs(int(batch_size), int(vocab_size))

    fwd_fn = with_executor(executor, fwd_call)
    run_benchmark(benchmark, fwd_fn, inputs)


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_cross_entropy_mini_benchmark_bwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    # picking a value that doesn't OOM for large vocab sizes
    batch_size = 4096

    def fwd_call(inp):
        return SyntheticMiniModel.mini_model(*inp)

    inputs = SyntheticMiniModel.inputs(batch_size, vocab_size)

    fwd_fn = with_executor(executor, fwd_call)
    outputs = fwd_fn(inputs)
    grads = SyntheticMiniModel.grads()
    run_benchmark(benchmark, unary_bwd_torch, [outputs, grads, *inputs])


# Simple test of F.cross_entropy without final reduction
def quack_cross_entropy_fwd_wrapper(inputs: list):
    return quack_cross_entropy_fwd(*inputs, return_dx=False)


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile", "quack"]
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.sizes_from_models)
def test_function_cross_entropy_fwd_benchmark(
    benchmark, executor: str, vocab_size: int
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    batch_size = 4096
    # vocab_size is large, scale by 0.1 to avoid overflow and represent realistic values, same used in quack test
    logits = 0.1 * torch.randn(
        batch_size, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=False
    )
    labels = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        device="cuda",
        dtype=torch.int64,
        requires_grad=False,
    )

    if executor == "quack":
        fwd_fn = quack_cross_entropy_fwd_wrapper
    else:
        fwd_fn = with_executor(executor, torch_cross_entropy_fwd)

    run_benchmark(benchmark, fwd_fn, [logits, labels])
