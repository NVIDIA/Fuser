# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass


# Sizes used in Llama 4 Maverick
@dataclass
class Config:
    hidden_size: int = 5120
    intermediate_size: int = 8192
    num_routed_experts: int = 128
    num_shared_experts: int = 1


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Llama4MoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.shared_experts = SwiGLU(
            config.hidden_size, config.intermediate_size * config.num_shared_experts
        )
        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_routed_experts)
            ]
        )

    def run_routed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))  # [s, h]

        router_logits = self.gate(hidden_states)  # [s, n]
        topk_weight, topk_ids = router_logits.topk(1)  # [s, 1]
        router_scores = topk_weight.sigmoid()  # [s, 1]
        hidden_states = hidden_states * router_scores  # [s, h]

        counts = topk_ids.new_zeros(
            (topk_ids.size(0), self.config.num_routed_experts)
        )  # [s, n]
        counts.scatter_(1, topk_ids, 1)  # [s, n]
        tokens_per_expert = counts.sum(0)  # [n]

        token_ids_sorted_by_expert_id = topk_ids.view(-1).argsort()  # [s]
        tokens_sorted_by_expert_id = hidden_states[
            token_ids_sorted_by_expert_id
        ]  # [s, h]

        # The following code block should be replaced with a grouped gemm.
        # However, torch._grouped_mm is yet to be usable because it requires
        # offsets being multiples of 16.
        outs_per_expert = []
        start_index = 0
        for expert_id, num_tokens in enumerate(tokens_per_expert):
            end_index = start_index + num_tokens
            if num_tokens == 0:
                continue
            expert = self.routed_experts[expert_id]
            expert_out = expert(tokens_sorted_by_expert_id[start_index:end_index])
            outs_per_expert.append(expert_out)
            start_index = end_index
        outs_sorted_by_expert_id = torch.cat(outs_per_expert, dim=0)  # [s, h]

        outs_sorted_by_token_id = torch.empty_like(outs_sorted_by_expert_id)  # [s, h]
        outs_sorted_by_token_id[
            token_ids_sorted_by_expert_id
        ] = outs_sorted_by_expert_id
        return outs_sorted_by_token_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states) + self.run_routed_experts(
            hidden_states
        )


@contextmanager
def default_tensor_type(dtype=torch.float32, device="cpu"):
    # Save
    prev_dtype = torch.get_default_dtype()
    prev_device = torch.get_default_device()

    # Set
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    yield

    # Restore
    torch.set_default_dtype(prev_dtype)
    torch.set_default_device(prev_device)


def test_llama4_moe():
    config = Config()

    # This is much faster than creating the module with CPU float parameters
    # and then doing `.to("cuda").to(torch.bfloat16)`.
    with default_tensor_type(dtype=torch.bfloat16, device="cuda"):
        model = Llama4MoE(config)

    batch_size, seq_len = 1, 2048
    inp = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    out = model(inp)

    assert out.size() == (batch_size, seq_len, config.hidden_size)
    assert out.dtype == torch.bfloat16
    assert out.is_cuda
