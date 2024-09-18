# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmarks Tensor parallel NanoGPT block using Pytorch TP API.

Usage: torchrun --nproc-per-node=<number of processes> test_pytorch_transformer.py
"""

import os
import time

from nanogpt import *

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor.device_mesh import init_device_mesh

# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 64
sequence_length = 2048
dtype = torch.bfloat16
world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
rank = device_mesh.get_rank()

assert (
    world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {world_size} gpus"


def benchmark_loop(model, input):
    forward_time = 0.0
    backward_time = 0.0

    for i in range(num_iters):
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


def benchmark_model():
    if rank != 0:
        return
    config = GPTConfig()
    model = Block(config).to(dtype).to("cuda")

    input = torch.rand(
        batch_size, sequence_length, config.n_embd, dtype=dtype, device="cuda"
    )
    forward_time, backward_time = benchmark_loop(model, input)
    print(f"Average forward time {forward_time}s, backward time {backward_time}s")


def benchmark_tensor_parallel():
    config = GPTConfig()
    config.n_devices = world_size
    tp_model = Block(config).to(dtype).to("cuda")

    # Parallelization plan. Tensor parallel
    tp_model = parallelize_module(
        module=tp_model,
        device_mesh=device_mesh,
        parallelize_plan={
            "attn.c_attn_key": ColwiseParallel(),
            "attn.c_attn_query": ColwiseParallel(),
            "attn.c_attn_value": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(),
            "mlp.c_fc": ColwiseParallel(),
            "mlp.c_proj": RowwiseParallel(),
        },
    )
    input = torch.rand(
        batch_size, sequence_length, config.n_embd, dtype=dtype, device="cuda"
    )

    forward_time, backward_time = benchmark_loop(tp_model, input)
    print(
        f"{rank}: Average tensor parallel forward time {forward_time}s, backward time {backward_time}s"
    )


benchmark_model()
benchmark_tensor_parallel()
dist.destroy_process_group()
