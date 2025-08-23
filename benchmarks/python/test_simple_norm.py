# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_simple_norm_fusion(
    fd: FusionDefinition, batch_size: int, vocab_size: int
) -> None:
    T0 = fd.define_tensor(
        shape=[batch_size, vocab_size],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    T3 = fd.ops.sum(T2, dims=[1], keepdim=False, dtype=DataType.Null)
    T4 = fd.ops.broadcast_in_dim(T3, shape=[batch_size, 1], broadcast_dims=[0])
    T11 = fd.ops.broadcast_in_dim(
        T4, shape=[batch_size, vocab_size], broadcast_dims=[0, 1]
    )
    T12 = fd.ops.add(T2, T11)
    T13 = fd.ops.cast(T12, dtype=DataType.BFloat16)
    fd.add_output(T13)


@pytest.mark.parametrize("batch_size", [1024, 4096, 32768])
@pytest.mark.parametrize("vocab_size", [65536, 100352, 131072, 152064, 202048, 262144])
# @pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_simple_norm_nvf_static_benchmark(
    benchmark,
    batch_size: int,
    vocab_size: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    with FusionDefinition() as fd:
        nvfuser_simple_norm_fusion(fd, batch_size, vocab_size)

    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.bfloat16)
    inputs = [logits]
    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
