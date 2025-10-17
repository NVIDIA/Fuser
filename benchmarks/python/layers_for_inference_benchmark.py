# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTE: `down_size`, and `pack_uint4` are copied from PyTorch's test code.
#
# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import Optional, Tuple
import math

from looseversion import LooseVersion
import torch
import torch.nn as nn
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked

from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextMoe,
    Llama4PreTrainedModel,
    ROPE_INIT_FUNCTIONS,
    ALL_ATTENTION_FUNCTIONS,
)


__all__ = [
    "GroupedLinear",
    "Llama4MoE",
    "NVFP4InferenceGroupedLinear",
    "NVFP4InferenceLinear",
    "nvfuser_f16a_nvfp4weight_scaled_grouped_mm",
    "nvfuser_f16a_nvfp4weight_scaled_mm",
]


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L972-L974
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L977-L982
def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# Ref: Based on `_bfloat16_to_float4_e2m1fn_x2` of https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L985-L990
def to_fp4(x: torch.Tensor) -> torch.Tensor:
    x = _f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L8-L10
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L125-L148
def pytorch_nvfp4_quantize(a, a_global_scale):
    BLOCK_SIZE = 16
    assert (
        a.size(-1) % BLOCK_SIZE == 0
    ), "The inner-most dim must be divisible by block_size; Padding is not implemented."
    assert a.is_contiguous(), "Only contiguous tensors are supported."

    original_shape = a.shape
    a_fp32 = a.float().reshape(original_shape[0], -1, BLOCK_SIZE)

    # Find absolute maximum along blockwise dimension
    max_abs = torch.amax(torch.abs(a_fp32), dim=-1)
    block_scale_fp32 = (max_abs / FLOAT4_E2M1_MAX).float()

    scaled_block_scale_fp32 = block_scale_fp32 * a_global_scale
    scaled_block_scale_fp8 = torch.clamp(
        scaled_block_scale_fp32,
        min=FLOAT8_E4M3_EPS,
        max=FLOAT8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    scaled_block_scale_fp8_fp32 = scaled_block_scale_fp8.to(torch.float)
    total_scale = scaled_block_scale_fp8_fp32 / a_global_scale
    a_scaled = a_fp32 / total_scale.unsqueeze(-1)
    a_scaled = torch.clamp(a_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    a_scaled = a_scaled.view(original_shape)
    return to_fp4(a_scaled), scaled_block_scale_fp8


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L63-L82
# apply swizzled on block scaling factor:
# 1. apply padding to [mn_t * 128 , k_t * 4]
# 2. apply swizzle
def linear_to_swizzled_128_4(a_sf_linear: torch.Tensor):
    mn, sf_k = a_sf_linear.shape
    m_tiles = (mn + 128 - 1) // 128
    mn_padded = m_tiles * 128
    k_tiles = (sf_k + 4 - 1) // 4
    k_padded = k_tiles * 4
    if mn_padded != mn or k_padded != sf_k:
        a_sf_padded = torch.empty(
            mn_padded, k_padded, dtype=a_sf_linear.dtype, device=a_sf_linear.device
        )
        a_sf_padded[0:mn, 0:sf_k] = a_sf_linear
    else:
        a_sf_padded = a_sf_linear
    # details about layout requirement on block-wise scaling factor
    # https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
    tmp = torch.reshape(a_sf_padded, (m_tiles, 4, 32, k_tiles, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, k_padded)[:mn, :sf_k]


@torch.inference_mode()
def quantize_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize weight to nvfp4, returning (packed) e2m1 weight, e4m3 scale factor, fp32 global scale."""
    global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight.float().abs().amax()
    ).to(torch.float32)
    fp4_weight, weight_scaling_factor = pytorch_nvfp4_quantize(weight, global_scale)
    weight_scale_interleaved = linear_to_swizzled_128_4(weight_scaling_factor)
    return fp4_weight, weight_scale_interleaved, global_scale


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L151-L152
def round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L13-L22
kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L25-L32
# Convert FP4 into FP32
def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L35-L49
# Unpack float4_e2m1fn_x2 into two separate fp32 values
def unpack_fp4_bytes(a, dtype):
    assert a.dtype == torch.float4_e2m1fn_x2
    m, n = a.shape
    a = a.view(torch.uint8).flatten()
    upper_half_byte = (a & 0xF0) >> 4
    lower_half_byte = a & 0x0F
    upper_half_float = torch.tensor([e2m1_to_fp32(x) for x in upper_half_byte]).to(
        a.device
    )
    lower_half_float = torch.tensor([e2m1_to_fp32(x) for x in lower_half_byte]).to(
        a.device
    )
    out = torch.stack((lower_half_float, upper_half_float), dim=-1).reshape(m, n * 2)
    return out


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L55-L60
# restore swizzled on block scaling factor:
# 1. restore swizzle
# 2. removes padding via slicing to [:mn, :k]
def swizzled_to_linear_128_4(a_sf_swizzled: torch.Tensor, mn, k):
    mn_padded, sf_k_padded = a_sf_swizzled.shape
    m_tiles = mn_padded // 128
    k_tiles = sf_k_padded // 4
    tmp = torch.reshape(a_sf_swizzled, (m_tiles, k_tiles, 32, 4, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, sf_k_padded)[:mn, :k]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L85-L101
def dequantize_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.float4_e2m1fn_x2
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = unpack_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = swizzled_to_linear_128_4(tensor_sf, m, k)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


# TODO: Update this accordingly to the progress of nvfp4 kernel implementation.
# An alternative is to use `_register_nvfuser_translator` of https://github.com/Lightning-AI/lightning-thunder/pull/2481
# instead of updating this function itself.
@torch.library.custom_op("nvf_cutlass::f16a_nvfp4weight_scaled_mm", mutates_args=())
def nvfuser_f16a_nvfp4weight_scaled_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    hp_weight = dequantize_to_dtype(
        fp4_weight,
        weight_scaling_factor,
        weight_global_scale,
        activation.dtype,
        fp4_weight.device,
        16,
    )
    return activation @ hp_weight + bias


@torch.library.register_fake("nvf_cutlass::f16a_nvfp4weight_scaled_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty(
        (activation.size(0), fp4_weight.size(0)),
        device=activation.device,
        dtype=activation.dtype,
    )


# TODO: Update this accordingly to the progress of nvfp4 kernel implementation.
# An alternative is to use `_register_nvfuser_translator` of https://github.com/Lightning-AI/lightning-thunder/pull/2481
# instead of updating this function itself.
@torch.library.custom_op(
    "nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm", mutates_args=()
)
def nvfuser_f16a_nvfp4weight_scaled_grouped_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    ab_strides: torch.Tensor,
    c_strides: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    hp_weight = torch.empty(
        (fp4_weight.size(0), fp4_weight.size(1), fp4_weight.size(2) * 2),
        device=activation.device,
        dtype=activation.dtype,
    )
    for i in range(fp4_weight.size(0)):
        hp_weight[i] = dequantize_to_dtype(
            fp4_weight[i],
            weight_scaling_factor[i],
            weight_global_scale[i],
            activation.dtype,
            fp4_weight.device,
            16,
        )
    return grouped_mm(activation, hp_weight, offsets)


@torch.library.register_fake("nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    ab_strides: torch.Tensor,
    c_strides: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (activation.size(0), fp4_weight.size(1)),
        device=activation.device,
        dtype=activation.dtype,
    )


class NVFP4InferenceLinear(nn.Module):
    """NVFP4 Linear layer for Inference.

    Weight, its scaling factor, its global scale, and bias are registered as a buffer, not a parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        fp4_weight: torch.Tensor | nn.Parameter,
        weight_scaling_factor: torch.Tensor | nn.Parameter,
        weight_global_scale: torch.Tensor | nn.Parameter | None,
        bias: torch.Tensor | nn.Parameter | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("fp4_weight", fp4_weight)
        self.register_buffer("weight_scaling_factor", weight_scaling_factor)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.nvf_cutlass.f16a_nvfp4weight_scaled_mm(
            x,
            self.fp4_weight,
            self.weight_scaling_factor,
            self.weight_global_scale,
            self.bias,
        )

    @staticmethod
    def from_linear(linear: nn.Linear) -> NVFP4InferenceLinear:
        weight = linear.weight
        bias = linear.bias
        out_features, in_features = weight.size()
        (
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
        ) = quantize_linear_weight_to_nvfp4(weight)
        return NVFP4InferenceLinear(
            in_features,
            out_features,
            fp4_weight=fp4_weight,
            weight_scaling_factor=weight_scaling_factor,
            weight_global_scale=weight_global_scale,
            bias=bias,
        )


class SwiGLU(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str
    ):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype, device=device
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _group_sizes_from_offsets(offsets: torch.Tensor) -> list[int]:
    group_sizes = []
    prev = 0
    for offset in offsets:
        group_sizes.append(offset - prev)
        prev = offset
    return group_sizes


if LooseVersion(torch.__version__) >= LooseVersion("2.8.0"):
    # Required otherwise, there is a graph-break.
    _grouped_mm = torch.compiler.allow_in_graph(torch._grouped_mm)


# This function should be replaced with torch._grouped_mm.  However,
# torch._grouped_mm is yet to be usable because it requires offsets being
# multiples of 16.
def grouped_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        # NOTE: This path also works for `thunder.jit` as it has a lookaside for `torch.compiler.is_compiling`.
        return _grouped_mm(a, b, offsets)

    group_sizes = _group_sizes_from_offsets(offsets)
    group_outs = []
    for group_a, group_b in zip(a.split(group_sizes), b.unbind()):
        group_outs.append(group_a @ group_b)
    return torch.cat(group_outs)


class GroupedLinear(nn.Module):
    def __init__(
        self,
        groups: int,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(groups, in_features, out_features, dtype=dtype, device=device)
        )
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight, offsets)


@torch.inference_mode()
def quantize_grouped_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize grouped linear's weight to nvfp4

    Args:
        weight: Parameter of `GroupedLinear` of [g, n, k]
        m: hidden_states.size(0)
        tokens_per_expert_neg_one:

    Returns:
        fp4_weight: [g, n, k // 2]
        scale_factors: [g, n, k // 16]
        global_scales: [g]
        ab_strides: [g]
        c_strides: [g]
    """
    assert weight.ndim == 3, "Weight must be a 3D tensor"

    device: torch.device = weight.device
    g, n, k = weight.size()

    with device:
        ab_strides = torch.full((g,), k, dtype=torch.int32)
        c_strides = torch.full((g,), n, dtype=torch.int32)

        fp4_weight = torch.empty((g, n, k // 2), dtype=torch.float4_e2m1fn_x2)
        global_scales = torch.empty((g,), dtype=torch.float32)
        scale_factors = torch.empty((g, n, k // 16), dtype=torch.float8_e4m3fn)

    for i in range(g):
        cur_weight = weight[i]
        global_scales[i] = cur_weight.abs().amax()
        cur_fp4_weight, cur_scale_factors = pytorch_nvfp4_quantize(
            cur_weight, global_scales[i]
        )
        fp4_weight[i] = cur_fp4_weight
        scale_factors[i] = linear_to_swizzled_128_4(cur_scale_factors)

    return fp4_weight, scale_factors, global_scales, ab_strides, c_strides


class NVFP4InferenceGroupedLinear(nn.Module):
    def __init__(
        self,
        fp4_weight: torch.Tensor,
        weight_scaling_factor: torch.Tensor,
        weight_global_scale: torch.Tensor,
        ab_strides: torch.Tensor,
        c_strides: torch.Tensor,
    ) -> None:
        self.register_buffer("fp4_weight", fp4_weight)
        self.register_buffer("weight_scaling_factor", weight_scaling_factor)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("ab_strides", ab_strides)
        self.register_buffer("c_strides", c_strides)

    # TODO: Update this accordingly to the progress of nvfp4 kernel implementation.
    def forward(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        tokens_per_group = offsets[1:] - offsets[:-1]
        problem_sizes = torch.stack(
            [
                tokens_per_group,
                torch.full_like(tokens_per_group, hidden_states.size(0)),
                torch.full_like(tokens_per_group, self.fp4_weight.size(2) * 2),
            ],
            dim=1,
        )
        blockscale_offsets = torch.cumsum(torch.ceil(tokens_per_group, 128) * 128)
        return torch.ops.nvf_cutlass.f16a_nvfp4weight_scaled_grouped_mm(
            hidden_states,
            self.fp4_weight,
            self.weight_scaling_factor,
            self.weight_global_scale,
            self.ab_strides,
            self.c_strides,
            offsets,
            blockscale_offsets,
            problem_sizes,
        )

    @staticmethod
    def from_grouped_linear(
        grouped_linear: GroupedLinear,
    ) -> NVFP4InferenceGroupedLinear:
        weight = grouped_linear.weight
        (
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
            ab_strides,
            c_strides,
        ) = quantize_grouped_linear_weight_to_nvfp4(weight)
        return NVFP4InferenceGroupedLinear(
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
            ab_strides=ab_strides,
            c_strides=c_strides,
        )


class GroupedSwiGLU(nn.Module):
    def __init__(
        self,
        groups: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.gate_proj = GroupedLinear(
            groups, hidden_size, intermediate_size, dtype, device
        )
        self.up_proj = GroupedLinear(
            groups, hidden_size, intermediate_size, dtype, device
        )
        self.down_proj = GroupedLinear(
            groups, intermediate_size, hidden_size, dtype, device
        )

    def forward(
        self, hidden_states: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states, offsets))
            * self.up_proj(hidden_states, offsets),
            offsets,
        )


# Slightly modified version of `thunder.tests.test_networks.Llama4MoE`
# to have the same singature as transformers' Llama4TextMoe -- in this file
# return values include `router_logits`.
# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L147-L165
class Llama4MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_routed_experts,
            bias=False,
            dtype=config.dtype,
            device=config.device,
        )
        self.shared_experts = SwiGLU(
            config.hidden_size,
            config.intermediate_size * config.num_shared_experts,
            config.dtype,
            config.device,
        )
        self.routed_experts = GroupedSwiGLU(
            config.num_routed_experts,
            config.hidden_size,
            config.intermediate_size,
            config.dtype,
            config.device,
        )

    @staticmethod
    def from_transformers_llama4textmoe(moe: Llama4TextMoe) -> Llama4MoE:
        """[CAUTION] A converter written by Gemini 2.5."""
        # This is defined in `thunder.tests.test_networks`
        from thunder.tests.test_networks import Config

        # 1. Create a config for the Llama4MoE model from the transformers config
        config = Config(
            hidden_size=moe.config.hidden_size,
            intermediate_size=moe.config.intermediate_size,
            num_routed_experts=moe.config.num_local_experts,
            num_shared_experts=1,  # Based on HF implementation having one shared_expert
            dtype=moe.router.weight.dtype,
            device=moe.router.weight.device,
        )

        # 2. Create an instance of our Llama4MoE
        new_moe = Llama4MoE(config)

        # 3. Copy the router weights (called 'gate' in our implementation)
        new_moe.gate.weight.data.copy_(moe.router.weight.data)

        # 4. Copy the shared expert weights
        new_moe.shared_experts.gate_proj.weight.data.copy_(
            moe.shared_expert.gate_proj.weight.data
        )
        new_moe.shared_experts.up_proj.weight.data.copy_(
            moe.shared_expert.up_proj.weight.data
        )
        new_moe.shared_experts.down_proj.weight.data.copy_(
            moe.shared_expert.down_proj.weight.data
        )

        # 5. For the routed experts, we need to handle the combined gate_up_proj
        # and permute the weight dimensions to match GroupedLinear
        # HF format: (groups, in_features, out_features)
        # Our format: (groups, out_features, in_features)

        # Permute from (num_experts, hidden_size, 2 * intermediate_size) to
        # (num_experts, 2 * intermediate_size, hidden_size)
        gate_up_proj_permuted = moe.experts.gate_up_proj.permute(0, 2, 1)

        # Split into gate and up projections
        gate_proj_w, up_proj_w = gate_up_proj_permuted.chunk(2, dim=1)

        new_moe.routed_experts.gate_proj.weight.data.copy_(gate_proj_w)
        new_moe.routed_experts.up_proj.weight.data.copy_(up_proj_w)

        # Permute down_proj from (num_experts, intermediate_size, hidden_size) to
        # (num_experts, hidden_size, intermediate_size)
        down_proj_permuted = moe.experts.down_proj.permute(0, 2, 1)
        new_moe.routed_experts.down_proj.weight.data.copy_(down_proj_permuted)

        return new_moe

    def run_routed_experts(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        return outs_sorted_by_token_id, router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs_sorted_by_token_id, router_logits = self.run_routed_experts(hidden_states)
        return (
            self.shared_experts(hidden_states) + outs_sorted_by_token_id,
            router_logits,
        )


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L749-L760
def copied_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L213-L222
def copied_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L226-L249
def copied_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = copied_repeat_kv(key, module.num_key_value_groups)
    value_states = copied_repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L279-L370
class copied_Llama4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Llama4TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.no_rope_layers[layer_idx]
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = copied_apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(
                    torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand(
                (*input_shape, 1, 1)
            )  # batch size > 1
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = copied_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Ref: Llama4TextMLP of https://docs.google.com/document/d/1Nd8jp-hC00Y8hIZFsl2Sy2_zF32C-_2tB1y0JaGqyf8/edit?tab=t.0
class copied_Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(down_proj)


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L112C1-L129C61
class copied_Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        Llama4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L46-L75
class copied_Llama4TextExperts(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L147-L165
class copied_Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = copied_Llama4TextExperts(config)
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.shared_expert = copied_Llama4TextMLP(config)

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits, float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device)
            .view(1, -1)
            .expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out.scatter_add_(
            dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim)
        )
        return out, router_scores


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L168C7-L198
class copied_Llama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(
                1, 2
            )
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # Convert to complex representation
            freqs_cis = freqs_cis * self.attention_scaling

        return freqs_cis


class copied_Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = copied_Llama4TextAttention(config, layer_idx)
        self.use_chunked_attention = config.attention_chunk_size is not None and bool(
            config.no_rope_layers[layer_idx]
        )
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = copied_Llama4TextMoe(config)
        else:
            self.feed_forward = copied_Llama4TextMLP(
                config, intermediate_size=config.intermediate_size_mlp
            )

        self.input_layernorm = copied_Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = copied_Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        # Self Attention
        attention_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L467-L563
class copied_Llama4TextModel(Llama4PreTrainedModel):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "model"
    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                copied_Llama4TextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = copied_Llama4TextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = copied_Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids.to(self.embed_tokens.weight.device)
            )

        if use_cache and past_key_values is None:
            if self.config.get_text_config().attention_chunk_size is not None:
                past_key_values = HybridChunkedCache(
                    self.config, inputs_embeds.shape[0], inputs_embeds.shape[1]
                )
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask, chunk_causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            use_cache=use_cache,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    chunk_causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    False,  # output_router_logits is False
                    use_cache,
                    cache_position,
                    freq_cis,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    chunk_causal_mask=chunk_causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=freq_cis,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @torch.compiler.disable(
        recursive=False
    )  # the operations in this method are not compilable
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        chunked_attention_mask=None,
        use_cache=True,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return (
                    attention_mask,
                    attention_mask,
                )  # flash does not support chunked attn TODO support flash
            return None, None

        if self.config._attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        sequence_length = input_tensor.shape[1]
        cache_position = cache_position.to(self.device)
        attention_chunk_size = self.config.attention_chunk_size
        using_chunked_attention = attention_chunk_size is not None

        first_cache_position = cache_position[0]

        if past_key_values is not None:
            full_cache_length = past_key_values.get_max_cache_shape() or sequence_length
        else:
            full_cache_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else sequence_length
            )

        if using_chunked_attention:
            cond1 = first_cache_position >= attention_chunk_size
            cond2 = (first_cache_position < attention_chunk_size) & (
                first_cache_position + sequence_length > attention_chunk_size
            )
            key_length = (
                torch.where(
                    cond1,
                    attention_chunk_size + sequence_length - 1,
                    torch.where(
                        cond2,
                        first_cache_position + sequence_length,
                        attention_chunk_size,
                    ),
                )
                if use_cache
                else full_cache_length
            )

        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                if using_chunked_attention:
                    offsets = (
                        first_cache_position,
                        max(first_cache_position - attention_chunk_size + 1, 0),
                    )
                    chunked_attention_mask = make_flex_block_causal_mask(
                        attention_mask,
                        attention_chunk_size,
                        sequence_length,
                        key_length,
                        offsets=offsets,
                    )
                attention_mask = make_flex_block_causal_mask(
                    attention_mask,
                    query_length=sequence_length,
                    key_length=full_cache_length,
                    offsets=(first_cache_position, 0),
                )
                return attention_mask, chunked_attention_mask
            if isinstance(attention_mask, BlockMask):
                return attention_mask, chunked_attention_mask

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        target_length = (
            max(full_cache_length, attention_chunk_size)
            if using_chunked_attention
            else full_cache_length
        )
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        if using_chunked_attention and full_cache_length > attention_chunk_size:
            start_idx = max(first_cache_position - attention_chunk_size + 1, 0)
            end_idx = start_idx + key_length
            chunked_attention_mask = self.create_chunked_attention_mask(
                self.config.attention_chunk_size,
                start=start_idx,  # same offset as with flex
                end=end_idx,
                device=device,
            )

            local_attention_mask = attention_mask[
                :, start_idx:end_idx
            ]  # offset here as well
            # It may be smaller than attention_chunk_size -> pad it
            requires_padding = local_attention_mask.shape[-1] < attention_chunk_size
            if requires_padding:
                local_attention_mask = nn.functional.pad(
                    local_attention_mask,
                    (0, attention_chunk_size - local_attention_mask.shape[-1]),
                )
            # Depending on the padding, take the query tokens from the end or the cache_position
            if not requires_padding:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, -sequence_length:, :
                ]
            else:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, cache_position, :
                ]

            chunked_attention_mask = chunked_attention_mask.expand(
                input_tensor.shape[0], -1, -1, -1
            )
            chunked_attention_mask = (
                chunked_attention_mask * local_attention_mask[:, None, None, :]
            )
            if self.config._attn_implementation == "eager":
                min_dtype = torch.finfo(dtype).min
                chunked_attention_mask = torch.where(
                    chunked_attention_mask == 0, min_dtype, 0.0
                ).to(dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and attention_mask.ndim == 4
            and not output_attentions  # Only unmask for 4d masks
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and chunked_attention_mask is not None
        ):
            chunked_attention_mask = chunked_attention_mask.bool()
            causal_mask = causal_mask != torch.finfo(dtype).min
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=first_cache_position,
                is_training=self.training,
            ):
                causal_mask = None
        return causal_mask, chunked_attention_mask

    def create_chunked_attention_mask(
        self, attention_chunk_size: int, start: int, end: int, device: torch.device
    ) -> torch.Tensor:
        arange_vector = torch.arange(start, end, device=device)
        block_pos = torch.abs(
            arange_vector.unsqueeze(0) // attention_chunk_size
            - arange_vector.unsqueeze(1) // attention_chunk_size
        )
        token_pos = arange_vector.unsqueeze(0) - arange_vector.unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask.to(device)

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=cache_position.device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(cache_position.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L566-L644
class copied_Llama4ForCausalLM(Llama4PreTrainedModel, GenerationMixin):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        super().__init__(config)
        self.model = copied_Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
