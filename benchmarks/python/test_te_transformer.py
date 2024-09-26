# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import torch
import torch.distributed as dist
from enum import auto, Enum
import time
import transformer_engine.pytorch as te


warmup_iters = 10
num_iters = 10
# Hyperparameters for GPT-3
hidden_size = 12288
num_heads = 96
ffn_hidden_size = hidden_size * 4
batch_size = 1
sequence_length = 2048
dtype = torch.bfloat16

rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
torch.cuda.set_device(rank)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
)
tp_group = dist.new_group()

def benchmark_loop(model, input, profile):
    forward_time = 0.0
    backward_time = 0.0

    for i in range(num_iters + warmup_iters):
        if i == num_iters and profile:
            torch.cuda.cudart().cudaProfilerStart()
        if i >= warmup_iters:
            start = time.time()
            if profile:
                torch.cuda.nvtx.range_push(f"iteration{i}")
                torch.cuda.nvtx.range_push("forward")

        output = model(input)
        torch.cuda.synchronize()

        if i >= warmup_iters:
            end = time.time()
            forward_time += end - start
            if profile:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("backward")
            start = time.time()

        output.sum().backward()
        torch.cuda.synchronize()

        if i >= warmup_iters:
            end = time.time()
            backward_time += end - start
            if profile:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    forward_time = forward_time / num_iters * 1000.0
    backward_time = backward_time / num_iters * 1000.0
    return forward_time, backward_time

def transformer_engine(profile):
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
    forward_time, backward_time = benchmark_loop(transformer_layer, x, profile)
    print(f'{rank}: Average forward time:{forward_time} ms, backward time:{backward_time} ms')


parser = argparse.ArgumentParser()
parser.add_argument(
    "--profile",
    dest="profile",
    default=False,
    action="store_true",
    help="Adds cuda profiling ranges",
)
args = parser.parse_args()

transformer_engine(args.profile)
dist.destroy_process_group()
