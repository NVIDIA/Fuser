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
from thunder.executors.nvfuserex_impl import getnv
from thunder.torch.custom_op import _register_custom_op, _register_nvfuser_translator

from nvfuser_direct import DataType

from python.direct_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_EPS,
    FLOAT8_E4M3_MAX,
    pytorch_nvfp4_quantize,
    linear_to_swizzled_128_4,
    round_up,
)


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


def _group_sizes_from_offsets(offsets: torch.Tensor, total_tokens: int) -> list[int]:
    group_sizes = []
    prev = 0
    for offset in offsets[1:]:
        group_sizes.append(offset - prev)
        prev = offset
    group_sizes.append(total_tokens - prev)
    return group_sizes


# Required otherwise, there is a graph-break.
_grouped_mm = torch.compiler.allow_in_graph(torch._grouped_mm)


# This function should be replaced with torch._grouped_mm.  However,
# torch._grouped_mm is yet to be usable because it requires offsets being
# multiples of 16.
def grouped_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _grouped_mm(a, b, offsets)

    group_sizes = _group_sizes_from_offsets(offsets, a.size(0))
    group_outs = []
    for group_a, group_b in zip(a.split(group_sizes), b.unbind()):
        group_outs.append(group_a @ group_b)
    return torch.cat(group_outs)


@torch.library.custom_op(
    "nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm", mutates_args=()
)
def nvfuser_f16a_nvfp4weight_scaled_grouped_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    global_scale: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    dropme: torch.Tensor,
) -> torch.Tensor:
    # assert False # dequantize_to_dtype is way too slow to be usable.
    # hp_weight = torch.empty(
    #    (fp4_weight.size(0), fp4_weight.size(1), fp4_weight.size(2) * 2),
    #    device=activation.device,
    #    dtype=activation.dtype,
    # )
    # for i in range(fp4_weight.size(0)):
    #    hp_weight[i] = dequantize_to_dtype(
    #        fp4_weight[i], weight_scaling_factor[i], global_scale[i], activation.dtype, fp4_weight.device, 16
    #    )
    return grouped_mm(activation, dropme, offsets)


@torch.library.register_fake("nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    global_scale: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    dropme: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (activation.size(0), fp4_weight.size(2)),
        device=activation.device,
        dtype=activation.dtype,
    )


def gmm_nvfuser(
    activation,
    fp4_weight,
    weight_scaling_factor,
    global_scale,
    offsets,
    blockscale_offsets,
    problem_sizes,
    dropme,
    *,
    fd,
    lc_to_nv_map,
):
    nv_act = getnv(activation, fd, lc_to_nv_map)
    nv_fp4_w = getnv(fp4_weight, fd, lc_to_nv_map)
    nv_sf_w = getnv(weight_scaling_factor, fd, lc_to_nv_map)
    nv_alpha = getnv(global_scale, fd, lc_to_nv_map)
    nv_offsets = getnv(offsets, fd, lc_to_nv_map)
    nv_blocksf_offsets = getnv(blockscale_offsets, fd, lc_to_nv_map)
    nv_problem_sizes = getnv(problem_sizes, fd, lc_to_nv_map)
    # dynamic shape support has some concretization issue
    m_size = activation.shape[0]
    k_size = activation.shape[1]
    k_tile_size = k_size // 16

    reshaped_mat1 = fd.ops.reshape(nv_act, [m_size, k_tile_size, 16])
    scale1 = fd.ops.abs(reshaped_mat1)
    scale1 = fd.ops.max(scale1, 2)
    scale1 = fd.ops.div(scale1, FLOAT4_E2M1_MAX)
    scale1 = fd.ops.clamp(scale1, FLOAT8_E4M3_EPS, FLOAT8_E4M3_MAX)

    broadcast_scale1 = fd.ops.broadcast(scale1, [False, False, True])
    reshaped_scaled_mat1 = fd.ops.div(reshaped_mat1, broadcast_scale1)
    reshaped_scaled_mat1 = fd.ops.clamp(
        reshaped_scaled_mat1, -FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX
    )

    scaled_mat1 = fd.ops.reshape(reshaped_scaled_mat1, [m_size, k_size])
    fp4_mat1 = fd.ops.cast(scaled_mat1, DataType.Float4_e2m1fn)
    fp8_scale1 = fd.ops.cast(scale1, DataType.Float8_e4m3fn)
    layout_fp8_scale1 = fd.ops.preprocess_grouped_matmul_input_sf(
        fp8_scale1, nv_offsets, nv_blocksf_offsets
    )
    out = fd.ops.cutlass_nvfp4_grouped_mm(
        fp4_mat1,
        nv_fp4_w,
        layout_fp8_scale1,
        nv_sf_w,
        nv_alpha,
        # NOTE: we might need to call contiguous on problem_sizes
        nv_problem_sizes,
        nv_offsets,
        nv_blocksf_offsets,
        DataType.BFloat16,
    )
    return out


_sym_of_nvfp4_scaled_grouped_mm = _register_custom_op(
    nvfuser_f16a_nvfp4weight_scaled_grouped_mm
)

_register_nvfuser_translator(_sym_of_nvfp4_scaled_grouped_mm, gmm_nvfuser)


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        alpha = torch.empty((groups,), dtype=torch.float32, requires_grad=False)
        fp4_weight = torch.empty(
            (groups, out_features, in_features // 2),
            dtype=torch.float4_e2m1fn_x2,
            requires_grad=False,
        )
        b_sf = torch.empty(
            (groups, round_up(out_features, 128), round_up(in_features // 16, 4)),
            dtype=torch.float8_e4m3fn,
            requires_grad=False,
        )

        self.k = (
            torch.tensor(in_features, dtype=torch.int32, requires_grad=False)
            .unsqueeze(-1)
            .expand((groups, 1))
        )
        self.n = (
            torch.tensor(out_features, dtype=torch.int32, requires_grad=False)
            .unsqueeze(-1)
            .expand((groups, 1))
        )

        transposed_weight = self.weight.transpose(-1, -2).contiguous()
        for i in range(groups):
            global_factor = (
                FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / transposed_weight[i].max()
            )
            alpha[i] = 1.0 / global_factor
            scaled_mat2_i, bs_mat2_i = pytorch_nvfp4_quantize(
                transposed_weight[i], global_factor
            )
            fp4_weight[i] = scaled_mat2_i
            b_sf[i] = linear_to_swizzled_128_4(bs_mat2_i)

        self.alpha = nn.Parameter(alpha)
        self.fp4_weight = nn.Parameter(fp4_weight.transpose(-1, -2))
        self.b_sf = nn.Parameter(b_sf)

    def forward(
        self,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            problem_sizes = torch.cat(
                (tokens_per_expert.unsqueeze(-1), self.n, self.k), dim=1
            ).to(torch.int32)
            return torch.ops.nvf_cutlass.f16a_nvfp4weight_scaled_grouped_mm(
                hidden_states,
                self.fp4_weight,
                self.b_sf,
                self.alpha,
                offsets,
                blockscale_offsets,
                problem_sizes,
                self.weight,
            )

        return grouped_mm(hidden_states, self.weight, offsets)


class GroupedSwiGLU(nn.Module):
    def __init__(self, groups: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.up_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.down_proj = GroupedLinear(groups, intermediate_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.down_proj(
            F.silu(
                self.gate_proj(
                    hidden_states, offsets, blockscale_offsets, tokens_per_expert
                )
            )
            * self.up_proj(
                hidden_states, offsets, blockscale_offsets, tokens_per_expert
            ),
            offsets,
            blockscale_offsets,
            tokens_per_expert,
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
        shifted_tokens_per_expert = torch.nn.functional.pad(tokens_per_expert, [1, 0])

        offsets = torch.cumsum(shifted_tokens_per_expert, 0, dtype=torch.int32)  # [n]
        rounded_tokens_per_expert = (shifted_tokens_per_expert + 127) // 128 * 128
        blockscale_offsets = torch.cumsum(
            rounded_tokens_per_expert, 0, dtype=torch.int32
        )  # [n]

        outs_sorted_by_expert_id = self.routed_experts(
            tokens_sorted_by_expert_id,
            offsets[:-1],
            blockscale_offsets[:-1],
            tokens_per_expert,
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

    tmodel = thunderfx(model, nv_enable_linear=True, nv_enable_scatter=True)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        actual = tmodel(inp)
    peak = torch.cuda.max_memory_allocated()
    print(f"Peak memory allocated: {peak / 1024**2:.2f} MB")

    # assert len(tmodel._backend.subgraph_infos) == 1
    # assert len(tmodel._backend.subgraph_infos[0].split_reasons) == 0
    # Uncomment to view thunder traces
    print(tmodel.last_traces)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
