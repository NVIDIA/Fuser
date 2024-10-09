# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import torch.distributed as dist
from enum import auto, Enum
from functools import partial


import transformer_engine.pytorch as te

import mpi_fixtures


mpi_test = mpi_fixtures.mpi_test


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@pytest.fixture(scope="module")
def setup_process_group(mpi_test) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    # The default port as used by https://github.com/pytorch/pytorch/blob/45a8b5682eb69d865cbf68c7f2f689b56b4efd53/torch/csrc/distributed/c10d/TCPStore.hpp#L51.
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=mpi_test.size,
        rank=mpi_test.rank,
    )
    yield
    dist.destroy_process_group()


# This benchmark is instrumented with cudaProfilerStart/Stop. Therefore, one
# can collect stats of the first few non-warmup benchmark iterations using
#
# ```bash
# mpirun -np <processes> nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:<iterations> pytest tests/python/test_transformer_engine.py -k <filter> --only-mpi
# ```
@pytest.mark.mpi
@pytest.mark.parametrize(
    "compute_type",
    [ComputeType.FORWARD, ComputeType.BACKWARD],
    ids=["forward", "backward"],
)
def test_transformer_layer(mpi_test, setup_process_group, benchmark, compute_type):
    # Hyperparameters for GPT-3
    hidden_size = 12288
    num_heads = 96
    ffn_hidden_size = hidden_size * 4
    batch_size = 1
    sequence_length = 2048
    dtype = torch.bfloat16

    size = mpi_test.size
    rank = mpi_test.rank

    torch.cuda.set_device(rank)
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

            def benchmark_fn(profile):
                if profile:
                    torch.cuda.cudart().cudaProfilerStart()

                y = transformer_layer(x)
                torch.cuda.synchronize()

                if profile:
                    torch.cuda.cudart().cudaProfilerStop()
                return y

            # Warmup.
            y = benchmark_fn(False)
            assert y.size() == torch.Size([batch_size, sequence_length, hidden_size])

            benchmark.pedantic(benchmark_fn, args=(True,), rounds=5)
        case ComputeType.BACKWARD:
            # Due to
            # https://github.com/Lightning-AI/lightning-thunder/issues/701, a
            # limitation in TransformerEngine, we can't repeatedly call
            # torch.autograd.backward to benchmark just backprop. As a
            # workaround, the code below runs forward before each backprop but
            # only measure the backprop time.
            def setup_fn(profile):
                y = transformer_layer(x)
                dy = torch.rand_like(y)
                torch.cuda.synchronize()
                # Unlike for forward, I can't pass `profile` directly to
                # `benchmark_fn` because `benchmark.pedantic` is not allowed to
                # take both `setup` and `args`. Therefore, we pass `profile` to
                # `setup_fn`, which in turn passes iit through to
                # `benchmark_fn`.
                return (y, dy, profile), {}

            def benchmark_fn(y, dy, profile):
                if profile:
                    torch.cuda.cudart().cudaProfilerStart()

                torch.autograd.backward(y, dy)
                torch.cuda.synchronize()

                if profile:
                    torch.cuda.cudart().cudaProfilerStop()

            # Warmup.
            args, kwargs = setup_fn(False)
            benchmark_fn(*args, **kwargs)

            benchmark.pedantic(
                benchmark_fn,
                setup=partial(setup_fn, True),
                rounds=5,
            )

    dist.destroy_process_group(group=tp_group)
