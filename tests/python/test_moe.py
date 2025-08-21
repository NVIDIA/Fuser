# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from thunder.dynamo import thunderfx


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


def _group_sizes_from_offsets(offsets: torch.Tensor) -> list[int]:
    group_sizes = []
    prev = 0
    for offset in offsets:
        group_sizes.append(offset - prev)
        prev = offset
    return group_sizes


# Required otherwise, there is a graph-break.
_grouped_mm = torch.compiler.allow_in_graph(torch._grouped_mm)


# This function should be replaced with torch._grouped_mm.  However,
# torch._grouped_mm is yet to be usable because it requires offsets being
# multiples of 16.
def grouped_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _grouped_mm(a, b, offsets)

    group_sizes = _group_sizes_from_offsets(offsets)
    group_outs = []
    for group_a, group_b in zip(a.split(group_sizes), b.unbind()):
        group_outs.append(group_a @ group_b)
    return torch.cat(group_outs)


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight, offsets)


class GroupedSwiGLU(nn.Module):
    def __init__(self, groups: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.up_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.down_proj = GroupedLinear(groups, intermediate_size, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states, offsets))
            * self.up_proj(hidden_states, offsets),
            offsets,
        )


class Llama4MoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.shared_experts = SwiGLU(
            config.hidden_size, config.intermediate_size * config.num_shared_experts
        )
        self.routed_experts = GroupedSwiGLU(
            config.num_routed_experts, config.hidden_size, config.intermediate_size
        )

    def run_routed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))  # [s, h]

        router_logits = self.gate(hidden_states)  # [s, n]
        topk_weight, topk_ids = router_logits.topk(1)  # [s, 1]
        router_scores = topk_weight.sigmoid()  # [s, 1]
        hidden_states = hidden_states * router_scores  # [s, h]

        counts = torch.zeros(
            topk_ids.size(0),
            self.config.num_routed_experts,
            device=topk_ids.device,
            dtype=torch.int32,
        )  # [s, n]
        counts = counts.scatter(1, topk_ids, 1)  # [s, n]
        tokens_per_expert = counts.sum(0)  # [n]

        token_ids_sorted_by_expert_id = topk_ids.view(-1).argsort()  # [s]
        tokens_sorted_by_expert_id = hidden_states[
            token_ids_sorted_by_expert_id
        ]  # [s, h]

        # Without `torch.int32`, we see `RuntimeError: Offsets tensor must be integer (int32) tensor, but got torch.int64.`
        # from PyTorch when calling _grouped_mm.
        offsets = torch.cumsum(tokens_per_expert, 0, dtype=torch.int32)  # [n]
        outs_sorted_by_expert_id = self.routed_experts(
            tokens_sorted_by_expert_id, offsets
        )  # [s, h]

        token_ids_sorted_by_expert_inverse_id = torch.argsort(
            token_ids_sorted_by_expert_id
        )
        outs_sorted_by_token_id = outs_sorted_by_expert_id[
            token_ids_sorted_by_expert_inverse_id
        ]

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


def test_llama4_moe_thunderfx():
    config = Config()

    # This is much faster than creating the module with CPU float parameters
    # and then doing `.to("cuda").to(torch.bfloat16)`.
    with default_tensor_type(dtype=torch.bfloat16, device="cuda"):
        model = Llama4MoE(config)

    # Without this, `thunderfx` falls back to `inductor` for `_grouped_mm`
    # as it doesn't have a grad-rule for the same.
    model.requires_grad_(False)

    batch_size, seq_len = 1, 2048
    inp = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    expected = model(inp)

    assert expected.size() == (batch_size, seq_len, config.hidden_size)
    assert expected.dtype == torch.bfloat16
    assert expected.is_cuda

    tmodel = thunderfx(model, nv_enable_linear=True)

    with torch.no_grad():
        actual = tmodel(inp)

    assert len(tmodel._backend.subgraph_infos) == 1
    assert len(tmodel._backend.subgraph_infos[0].split_reasons) == 0
    # Uncomment to view thunder traces
    # print(tmodel.last_traces)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
