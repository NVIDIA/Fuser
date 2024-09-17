# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

'''
Benchmarks Tensor parallel NanoGPT with Pytorch.

Usage: torchrun --nproc-per-node=<number of processes> transformer.py 
'''

import os
import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor.device_mesh import init_device_mesh

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

# Usage: torchrun --nproc-per-node=<number of processes> transformer.py 

def get_logger():
    return logging.getLogger(__name__)


def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Splitting key, query, and value projections 
        self.c_attn_key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # NOTE: The original Karpathy's script hides bias registration behind a flag
        # but we don't do that here. We always register bias due to a now-fixed
        # bug in thunder.
        # TODO: Move the bias registration to be happening `if not self.flash` once the bug is fixed.
        # if not self.flash:
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = self.c_attn_query(x)
        k = self.c_attn_key(x)
        v = self.c_attn_value(x)

        # TODO: It looks like view needs to take in the sharded size.
        k = k.view(B, T, self.n_head, C // self.n_head // _world_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head // _world_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head // _world_size).transpose(1, 2)  # (B, nh, T, hs)
        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C // _world_size)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # NOTE: The original Karpathy's script doesn't set the approximate flag,
        # probably by mistake.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


logger = get_logger()

# DeviceMesh creation
_world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

dtype = torch.bfloat16
config = GPTConfig()
tp_model = Block(config).to(dtype).to("cuda")

# Create an optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

# Parallelization plan 
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
    }
)
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10
rank_log(_rank, logger, "Tensor Parallel training starting...")

for i in range(num_iters):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(64, 2048, config.n_embd, dtype=dtype, device="cuda")
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"Tensor Parallel iter {i} completed")

rank_log(_rank, logger, "Tensor Parallel training completed!")