# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import torch.distributed as dist
from enum import auto, Enum

import transformer_engine.pytorch as te

import multidevice


multidevice_test = multidevice.multidevice_test


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@pytest.mark.mpi
@pytest.mark.parametrize("compute_type", [ComputeType.FORWARD, ComputeType.BACKWARD])
def test_transformer_layer(multidevice_test, benchmark, compute_type):
    # Hyperparameters for GPT-3
    hidden_size = 12288
    num_heads = 96
    ffn_hidden_size = hidden_size * 4
    batch_size = 1
    sequence_length = 2048
    dtype = torch.bfloat16

    size = multidevice_test.size
    rank = multidevice_test.rank

    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=size,
        rank=rank,
    )
    tp_group = dist.new_group()

    transformer_layer = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        set_parallel_mode=True,
        tp_group=tp_group,
    )
    transformer_layer.to(dtype).to("cuda")

    x = torch.randn(
        batch_size, sequence_length, hidden_size, dtype=dtype, device="cuda"
    )

    match compute_type:
        case ComputeType.FORWARD:

            def benchmark_fn():
                return transformer_layer(x)

            y = benchmark(benchmark_fn)
            assert y.size() == torch.Size([batch_size, sequence_length, hidden_size])
        case ComputeType.BACKWARD:
            # Due to
            # https://github.com/Lightning-AI/lightning-thunder/issues/701, a
            # limitation in TransformerEngine, we can't repeatedly call
            # torch.autograd.backward to benchmark just backprop. As a
            # workaround, the code below runs forward before each backprop but
            # only measure the backprop time.
            def setup_fn():
                y = transformer_layer(x)
                dy = torch.rand_like(y)
                return (y, dy), {}

            def benchmark_fn(y, dy):
                torch.autograd.backward(y, dy)

            benchmark.pedantic(
                benchmark_fn, setup=setup_fn, warmup_rounds=2, iterations=1, rounds=5
            )

    dist.destroy_process_group()
