# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmarks Tensor parallel NanoGPT block using Pytorch TP API.

Usage: torchrun --nproc-per-node=<number of processes> test_pytorch_transformer.py <--compile>

To capture annotated nsight trace:
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu
--capture-range-end stop --capture-range=cudaProfilerApi
--cudabacktrace=true -o <trace> torchrun --nproc-per-node <> test_pytorch_transformer.py --profile
"""

import argparse
import os
import time

from nanogpt import Block, GPTConfig

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor.device_mesh import init_device_mesh


warmup_iters = 10
num_iters = 10
batch_size = 1
sequence_length = 2048
dtype = torch.bfloat16
world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
rank = device_mesh.get_rank()

assert (
    world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {world_size} gpus"


def benchmark_loop(model, input, profile):
    forward_time = 0.0
    backward_time = 0.0

    if profile:
        torch.cuda.cudart().cudaProfilerStart()

    for i in range(num_iters + warmup_iters):
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


def benchmark_model(use_torch_compile, profile):
    if rank != 0:
        return
    config = GPTConfig()
    model = Block(config).to(dtype).to("cuda")
    if use_torch_compile:
        model = torch.compile(model)

    input = torch.rand(
        batch_size, sequence_length, config.n_embd, dtype=dtype, device="cuda"
    )

    forward_time, backward_time = benchmark_loop(model, input, profile)
    print(
        f"torch.compile {not use_torch_compile}, Average forward time {forward_time}ms, backward time {backward_time}ms"
    )


def benchmark_tensor_parallel(use_torch_compile, profile):
    config = GPTConfig()
    config.n_devices = world_size
    tp_model = Block(config).to(dtype).to("cuda")

    # Parallelization plan. Tensor parallel
    tp_model = parallelize_module(
        module=tp_model,
        device_mesh=device_mesh,
        parallelize_plan={
            "attn.c_attn": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(),
            "mlp.c_fc": ColwiseParallel(),
            "mlp.c_proj": RowwiseParallel(),
        },
    )
    if use_torch_compile:
        tp_model = torch.compile(tp_model)

    input = torch.rand(
        batch_size, sequence_length, config.n_embd, dtype=dtype, device="cuda"
    )

    forward_time, backward_time = benchmark_loop(tp_model, input, profile)
    print(
        f"{rank}: torch.compile {use_torch_compile}, Average tensor parallel forward time {forward_time}ms, backward time {backward_time}ms"
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--compile",
    dest="compile",
    default=False,
    action="store_true",
    help="Use torch.compile",
)
parser.add_argument(
    "--profile",
    dest="profile",
    default=False,
    action="store_true",
    help="Adds cuda profiling ranges",
)
args = parser.parse_args()

# benchmark_model(args.compile)
benchmark_tensor_parallel(args.compile, args.profile)
dist.destroy_process_group()
