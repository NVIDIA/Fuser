# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from enum import auto, Enum
from functools import partial


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class Parallelism(Enum):
    # https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#tensor-parallelism
    TENSOR_PARALLEL = auto()
    # https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#sequence-parallelism
    SEQUENCE_PARALLEL = auto()


# This benchmark is instrumented with cudaProfilerStart/Stop. Therefore, one
# can collect stats of the first few non-warmup benchmark iterations using
# ```bash
# mpirun -np <processes> nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:<iterations> pytest tests/python/test_transformer_engine.py -k <filter> --only-mpi
# ```
# and then display the status using e.g. `nsys stats --report=cuda_gpu_kern_sum report1.nsys-rep`.
@pytest.mark.mpi
@pytest.mark.parametrize(
    "compute_type",
    [ComputeType.FORWARD, ComputeType.BACKWARD],
    ids=["forward", "backward"],
)
@pytest.mark.parametrize(
    "parallelism",
    [Parallelism.TENSOR_PARALLEL, Parallelism.SEQUENCE_PARALLEL],
    ids=["tp", "sp"],
)
@pytest.mark.parametrize(
    "overlap",
    [False, True],
    ids=["nonoverlap", "overlap"],
)
def test_transformer_layer(
    setup_default_process_group,
    monkeypatch,
    benchmark,
    compute_type: ComputeType,
    parallelism: Parallelism,
    overlap: bool,
):
    if overlap and parallelism == Parallelism.TENSOR_PARALLEL:
        pytest.skip("Tensor parallelism doesn't support overlapping.")

    # Hyperparameters for GPT-3
    hidden_size = 12288
    num_heads = 96
    ffn_hidden_size = hidden_size * 4
    batch_size = 1
    sequence_length = 2048
    dtype = torch.bfloat16

    size = dist.get_world_size()

    transformer_layer = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        # According to https://github.com/NVIDIA/TransformerEngine/issues/1350,
        # `attn_input_format` has to match the format of `transformer_layer`'s
        # input.
        attn_input_format="bshd",
        set_parallel_mode=True,
        sequence_parallel=(parallelism == Parallelism.SEQUENCE_PARALLEL),
        ub_tp_comm_overlap=overlap,
        tp_group=dist.group.WORLD,
    )
    transformer_layer.to(dtype).to("cuda")

    match parallelism:
        case Parallelism.TENSOR_PARALLEL:
            local_sequence_length = sequence_length
        case Parallelism.SEQUENCE_PARALLEL:
            assert sequence_length % size == 0
            local_sequence_length = sequence_length // size

    x = torch.randn(
        batch_size, local_sequence_length, hidden_size, dtype=dtype, device="cuda"
    )

    if overlap:
        # Similar to https://github.com/NVIDIA/TransformerEngine/blob/e7bfc0c547d63332e4f8d65e606dc69f4c22ffbe/examples/pytorch/comm_gemm_overlap/te_layer_with_overlap.py#L27-L29
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        if not te.cpp_extensions.device_supports_multicast():
            monkeypatch.setenv("UB_SKIPMC", "1")

        te.module.base.initialize_ub(
            # Instructed by https://github.com/NVIDIA/TransformerEngine/blob/e7bfc0c547d63332e4f8d65e606dc69f4c22ffbe/transformer_engine/pytorch/module/base.py#L96-L99
            [batch_size * sequence_length, hidden_size],
            size,
            dtype=dtype,
            bootstrap_backend="nccl",
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
            assert y.size() == torch.Size(
                [batch_size, local_sequence_length, hidden_size]
            )

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

    if overlap:
        te.module.base.destroy_ub()
