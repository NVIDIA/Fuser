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
from .global_params import FLOAT_DTYPES
from .torch_ops import scatter_reduce

# (top_k, hidden) configurations seen in models.
TEST_CONFIGS = [
    (8, 7168),  # deepseek r1
]

SEQ_LENGTHS = [
    1024,
    2048,
    3072,
    4096,
    8192,
    12288,
    16384,
    20480,
    24576,
    28672,
    32768,
]


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("topk_hidden", TEST_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_fwd_baseline_benchmark(
    benchmark,
    seq_length: int,
    topk_hidden: tuple,
    dtype: torch.dtype,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    hidden_states = torch.randn((seq_length * topk_hidden[0], topk_hidden[1]), device="cuda", requires_grad=True))
    logits = torch.randn(seq_length * topk_hidden[0], device="cuda")
    idxs = logits.argsort()
    topk_weight = torch.randn((seq_length, topk_hidden[0]), device="cuda")

    benchmark_fn = with_executor(executor, scatter_reduce, **kwargs)
    run_benchmark(
        benchmark,
        benchmark_fn,
        [hidden_states, idxs, topk_weight],
    )


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("topk_hidden", TEST_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_reduce_bwd_baseline_benchmark(
    benchmark,
    seq_length: int,
    topk_hidden: tuple,
    dtype: torch.dtype,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    hidden_states = torch.randn((seq_length * topk_hidden[0], topk_hidden[1]), device="cuda", requires_grad=True)
    logits = torch.randn(seq_length * topk_hidden[0], device="cuda")
    idxs = logits.argsort()
    topk_weight = torch.randn((seq_length, topk_hidden[0]), device="cuda")
    grads = torch.randn((seq_length, topk_hidden[1]), device="cuda")

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, scatter_reduce, **kwargs)
    fwd_inputs = [hidden_states, idxs, topk_weight]
    outputs = fwd_fn(fwd_inputs)

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
    )
