# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This test is yet to be added to CI. For now, run it manually with `mpirun -n
# 2 pytest transformer_engine_test.py`.

import os
import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te

from multidevice import multidevice_test


@pytest.mark.mpi
def test_transformer_layer(multidevice_test):
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
    y = transformer_layer(x)
    assert y.size() == torch.Size([batch_size, sequence_length, hidden_size])

    dist.destroy_process_group()
