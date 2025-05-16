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


problem_sizes = [8000, 32064, 132035, 180000]


def mini_model(logits, labels):
    labels = torch.nn.functional.pad(labels, (0, 1))
    labels = labels[1 : labels.shape[-1]]
    return torch.nn.functional.cross_entropy(logits, labels)


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", problem_sizes)
def test_cross_entropy_mini_benchmark_fwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    def fwd_call(inp):
        return mini_model(*inp)

    inputs = torch.rand((8192, vocab_size), device="cuda", requires_grad=True)
    target = torch.randint(
        0, vocab_size - 1, (8192,), device="cuda", requires_grad=False
    )

    kwargs = {}
    inputs = [inputs, target]
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    run_benchmark(benchmark, fwd_fn, inputs)


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", problem_sizes)
def test_cross_entropy_mini_benchmark_bwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    def fwd_call(inp):
        return mini_model(*inp)

    input = torch.rand((8192, vocab_size), device="cuda", requires_grad=True)
    target = torch.randint(
        0, vocab_size - 1, (8192,), device="cuda", requires_grad=False
    )

    kwargs = {}
    inputs = [input, target]
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(inputs)
    grads = torch.tensor(1, device="cuda", dtype=torch.float32, requires_grad=False)
    run_benchmark(benchmark, unary_bwd_torch, [outputs, grads, inputs])
