# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class Config:
    max_seq_len: int = 4096
    hidden_size: int = 7168
    num_heads: int = 128
    down_projected_q_size: int = 1536
    down_projected_kv_size: int = 512
    up_projected_head_size: int = 128
    rope_head_size: int = 64


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        t = torch.arange(self.max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor):
        _, _, seq_len, head_size = x.size()

        x1 = x[..., : head_size // 2]
        x2 = x[..., head_size // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        # sin/cos: [seq_len, head_size]
        return x * self.cos[:seq_len] + rotated * self.sin[:seq_len]


class MultiheadLatentAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.down_projection_q = nn.Linear(config.hidden_size, config.down_projected_q_size)
        self.down_projection_kv = nn.Linear(config.hidden_size, config.down_projected_kv_size)
        self.up_projection_q = nn.Linear(config.down_projected_q_size, config.num_heads * config.up_projected_head_size)
        self.up_projection_kv = nn.Linear(config.down_projected_kv_size, config.num_heads * config.up_projected_head_size * 2)
        self.rope_projection_q = nn.Linear(config.down_projected_q_size, config.num_heads * config.rope_head_size)
        self.rope_projection_k = nn.Linear(config.hidden_size, config.rope_head_size)
        self.rope = RotaryPositionEmbedding(config.rope_head_size, config.max_seq_len)
        self.out_projection = nn.Linear(config.num_heads * config.up_projected_head_size, config.hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Variable names follow the notation of https://arxiv.org/abs/2412.19437
        batch_size, seq_len, _ = h.size()
        num_heads = self.config.num_heads

        # Down projections
        c_q = self.down_projection_q(h)
        c_kv = self.down_projection_kv(h)

        # Up projections
        q_c = self.up_projection_q(c_q).view(batch_size, seq_len, num_heads, self.config.up_projected_head_size).transpose(1, 2)
        kv_c = self.up_projection_kv(c_kv).view(batch_size, seq_len, num_heads, self.config.up_projected_head_size * 2).transpose(1, 2)
        k_c, v = kv_c.split(self.config.up_projected_head_size, -1)

        # RoPE
        q_r = self.rope_projection_q(c_q).view(batch_size, seq_len, num_heads, self.config.rope_head_size).transpose(1, 2)
        k_r = self.rope_projection_k(h).view(batch_size, seq_len, 1, self.config.rope_head_size).transpose(1, 2)
        q_r = self.rope(q_r)
        k_r = self.rope(k_r)
        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r.expand(-1, num_heads, -1, -1)], dim=-1)

        # Attention
        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection
        return self.out_projection(o)


def test_transformer_layer():
    config = Config()
    model = MultiheadLatentAttention(config)
    model.to("cuda").to(torch.bfloat16)

    batch_size, seq_len = 1, 4096
    inp = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    out = model(inp)

    assert out.size() == (batch_size, seq_len, config.hidden_size)
    assert out.dtype == torch.bfloat16
