# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.distributed as dist
from enum import auto, Enum
import time
import transformer_engine.pytorch as te

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

def profile_loop(model, input, num_iters=10):
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(num_iters):
        torch.cuda.nvtx.range_push(f'iteration{i}')
        torch.cuda.nvtx.range_push("forward")
        output = model(input)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        output.sum().backward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def benchmark_loop(model, input, num_iters=10):
    forward_time = 0.0
    backward_time = 0.0

    for _ in range(num_iters):
        start = time.time()
        output = model(input)
        torch.cuda.synchronize()
        end = time.time()
        forward_time += end - start

        start = time.time()
        output.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        backward_time += end - start

    forward_time /= num_iters
    backward_time /= num_iters
    return forward_time, backward_time   

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
# warm up
warmup_itrs = 10
benchmark_loop(transformer_layer, x, warmup_itrs)
forward_time, backward_time = benchmark_loop(transformer_layer, x)
print(f'Average forward time:{forward_time}s, backward time:{backward_time}s')
profile_loop(transformer_layer, x)
dist.destroy_process_group()