# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    max_seq_len: int = 4096
    hidden_size: int = 7168
    # Intermediate size per expert in MoE
    intermediate_size: int = 2048
    num_heads: int = 128
    down_projected_q_size: int = 1536
    down_projected_kv_size: int = 512
    up_projected_head_size: int = 128
    rope_head_size: int = 64
    num_routed_experts: int = 256
    num_shared_experts: int = 1
    k: int = 8


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(self.max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        self.down_proj_q = nn.Linear(
            config.hidden_size, config.down_projected_q_size, bias=False
        )
        self.norm_q = nn.RMSNorm(config.down_projected_q_size)
        self.up_proj_q = nn.Linear(
            config.down_projected_q_size,
            config.num_heads * (config.up_projected_head_size + config.rope_head_size),
            bias=False,
        )

        self.down_proj_kv_with_mqa = nn.Linear(
            config.hidden_size,
            config.down_projected_kv_size + config.rope_head_size,
            bias=False,
        )
        self.norm_kv = nn.RMSNorm(config.down_projected_kv_size)
        self.up_proj_kv = nn.Linear(
            config.down_projected_kv_size,
            config.num_heads * config.up_projected_head_size * 2,
            bias=False,
        )

        self.rope = RotaryPositionEmbedding(config.rope_head_size, config.max_seq_len)

        self.out_proj = nn.Linear(
            config.num_heads * config.up_projected_head_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Variable names follow the notation of https://arxiv.org/abs/2412.19437
        batch_size, seq_len, _ = hidden_states.size()
        num_heads = self.config.num_heads

        # Query
        c_q = self.down_proj_q(hidden_states)
        q_c_and_q_r = self.up_proj_q(self.norm_q(c_q))
        q_c, q_r = (
            q_c_and_q_r.view(batch_size, seq_len, num_heads, -1)
            .transpose(1, 2)
            .split([self.config.up_projected_head_size, self.config.rope_head_size], -1)
        )

        # Key and value
        c_kv_and_k_r = self.down_proj_kv_with_mqa(hidden_states)
        c_kv, k_r = c_kv_and_k_r.split(
            [self.config.down_projected_kv_size, self.config.rope_head_size], -1
        )
        k_r = k_r.view(batch_size, seq_len, 1, -1).transpose(1, 2)
        kv_c = (
            self.up_proj_kv(self.norm_kv(c_kv))
            .view(batch_size, seq_len, num_heads, -1)
            .transpose(1, 2)
        )
        k_c, v = kv_c.split(self.config.up_projected_head_size, -1)

        # RoPE
        q_r = self.rope(q_r)
        k_r = self.rope(k_r)
        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r.expand(-1, num_heads, -1, -1)], dim=-1)

        # Attention
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection
        return self.out_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, config: Config, intermediate_size: int = None):
        super().__init__()
        self.config = config
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MixtureOfExperts(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.shared_experts = SwiGLU(
            config, config.intermediate_size * config.num_shared_experts
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLU(config) for _ in range(config.num_routed_experts)]
        )

    def run_routed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO:
        # - auxiliary-loss-free load balancing
        # - multiple expert groups
        batch_size, seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        scores = self.gate(hidden_states).sigmoid()
        topk_weight, topk_ids = scores.topk()
        topk_weight /= topk_weight.sum(-1)

        counts = topk_ids.new_zeros((topk_ids.size(0), config.num_routed_experts))
        counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = counts.sum(0)

        indices_sorted_by_expert_id = topk_ids.view(-1).argsort()
        token_ids_sorted_by_expert_id = indices_sorted_by_expert_id // topk_ids.size(1)
        tokens_sorted_by_expert_id = hidden_states[token_ids_sorted_by_expert_id]

        outs_per_expert = []
        start_index = 0
        for expert_id, num_tokens in enumerate(tokens_per_expert):
            end_index = start_index + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[expert_id]
            expert_out = expert(tokens_sorted_by_expert_id[start_index:end_index])
            outs_per_expert.append(expert_out)
            start_index = end_index
        outs_sorted_by_expert_id = torch.cat(outs_per_expert, dim=0)

        outs_sorted_by_token_id = torch.empty_like(outs_sorted_by_expert_id)
        outs_sorted_by_token_id[indices_sorted_by_expert_id] = outs_sorted_by_expert_id
        return (
            (
                outs_sorted_by_token_id.view(*topk_ids.size(), -1)
                * topk_weight.unsqueeze(-1)
            )  # [batch_size * seq_len, k, hidden_size]
            .sum(1)  # [batch_size * seq_len, hidden_size]
            .view(batch_size, seq_len, -1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states) + self.run_routed_experts(
            hidden_states
        )


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.input_norm = nn.RMSNorm(config.hidden_size)
        self.mla = MultiheadLatentAttention(config)
        self.post_attention_norm = nn.RMSNorm(config.hidden_size)
        self.moe = MixtureOfExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.mla(hidden_states)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states += residual

        return hidden_states


def test_transformer_layer():
    config = Config()
    model = MultiheadLatentAttention(config)
    model.to("cuda").to(torch.bfloat16)

    batch_size, seq_len = 1, 4096
    inp = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    out = model(inp)

    assert out.size() == (batch_size, seq_len, config.hidden_size)
    assert out.dtype == torch.bfloat16
