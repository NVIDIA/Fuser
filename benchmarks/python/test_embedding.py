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
from .torch_ops import embedding


# (vocab, hidden) configurations seen in models.
EMBEDDING_CONFIGS = [
    (152064, 3584),  # hf_qwen2
    (32064, 3072),  # hf_phi3
    (131072, 5120),  # hf_mistral_nemo
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
@pytest.mark.parametrize("vocab_hidden", EMBEDDING_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding_fwd_baseline_benchmark(
    benchmark,
    seq_length: int,
    vocab_hidden: tuple,
    dtype: torch.dtype,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()
    if executor == "thunder":
        kwargs["nv_enable_embedding"] = True

    indices = torch.randint(0, vocab_hidden[0], (seq_length,), device="cuda")
    embedding_table = torch.randn(vocab_hidden, device="cuda", dtype=dtype)

    benchmark_fn = with_executor(executor, embedding, **kwargs)
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
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()
    if executor == "thunder":
        kwargs["nv_enable_embedding"] = True

    indices = torch.randint(0, vocab_hidden[0], (seq_length,), device="cuda")
    grads = torch.randn((seq_length, vocab_hidden[1]), device="cuda", dtype=dtype)
    embedding_table = torch.randn(
        vocab_hidden, device="cuda", dtype=dtype, requires_grad=True
    )

    # Compile the fwd fn for torchcompile
    fwd_fn = with_executor(executor, embedding, **kwargs)
    fwd_inputs = [indices, embedding_table]
    outputs = fwd_fn(fwd_inputs)

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
    )
