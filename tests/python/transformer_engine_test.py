# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This test is yet to be added to CI. For now, run it manually with `mpirun -n
# 2 pytest transformer_engine_test.py`.

import os
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te


def test_transformer_layer():
    # Hyperparameters for GPT-3
    hidden_size = 12288
    num_heads = 96
    ffn_hidden_size = hidden_size * 4
    batch_size = 1
    sequence_length = 2048
    dtype = torch.bfloat16

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        world_size=world_size,
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
