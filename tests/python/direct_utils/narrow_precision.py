# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

# Portions of this code are derived from NVIDIA Transformer Engine
# https://github.com/NVIDIA/TransformerEngine
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0

import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Map the 7 values of e2m1 to corresponding positive fp32 value
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


# Convert FP4 into FP32
def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


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


# restore swizzled on block scaling factor:
# 1. restore swizzle
# 2. removes padding via slicing to [:mn, :k]
def swizzled_to_linear_128_4(a_sf_swizzled: torch.Tensor, mn, k):
    mn_padded, sf_k_padded = a_sf_swizzled.shape
    m_tiles = mn_padded // 128
    k_tiles = sf_k_padded // 4
    tmp = torch.reshape(a_sf_swizzled, (m_tiles, k_tiles, 32, 4, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, sf_k_padded)[:mn, :k]


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
    return tmp.transpose(1, 3).reshape(mn_padded, k_padded)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


_FP4_LUT = torch.tensor(
    [
        0.0,  # 0: 0000 - zero
        0.5,  # 1: 0001 - smallest positive normal
        1.0,  # 2: 0010
        1.5,  # 3: 0011
        2.0,  # 4: 0100
        3.0,  # 5: 0101
        4.0,  # 6: 0110
        6.0,  # 7: 0111 - largest positive normal
        -0.0,  # 8: 1000 - negative zero
        -0.5,  # 9: 1001 - smallest negative normal
        -1.0,  # 10: 1010
        -1.5,  # 11: 1011
        -2.0,  # 12: 1100
        -3.0,  # 13: 1101
        -4.0,  # 14: 1110
        -6.0,  # 15: 1111 - largest negative normal
    ],
    dtype=torch.float32,
)


def fp4_to_fp32(fp4: torch.Tensor) -> torch.Tensor:
    # Convert FP4 indices to their corresponding floating point values
    # Each index (0-15) represents a 4-bit FP4 value in E2M1 format
    # Values based on the FP4 E2M1 specification
    fp4_lut = _FP4_LUT.to(fp4.device)
    return fp4_lut[fp4.to(torch.long)]


def dequantize_fp4(
    qx: torch.Tensor, sx: torch.Tensor, amax: torch.Tensor
) -> torch.Tensor:
    sf = sx.repeat_interleave(16, dim=1).view(torch.float8_e4m3fn).to(torch.float32)
    dqx = fp4_to_fp32(unpack_fp4(qx))
    sf = sf[: dqx.shape[0], : dqx.shape[1]]
    dequant = dqx * sf * (amax / (6.0 * 448))
    return dequant


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


# NOTE: This is from pytorch nvfp4 gemm tests.
def to_fp4(x):
    def down_size(size):
        assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
        return (*size[:-1], size[-1] // 2)

    def pack_uint4(uint8_data) -> torch.Tensor:
        # converting to uint8 for operations
        shape = uint8_data.shape
        assert shape[-1] % 2 == 0
        uint8_data = uint8_data.contiguous().view(-1)
        return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))

    from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked

    x = _f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


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
    scaled_block_scale_fp32 = torch.clamp(
        scaled_block_scale_fp32, min=FLOAT8_E4M3_EPS, max=FLOAT8_E4M3_MAX
    )
    scaled_block_scale_fp8 = scaled_block_scale_fp32.to(torch.float8_e4m3fn)
    total_scale = scaled_block_scale_fp32 / a_global_scale
    a_scaled = a_fp32 / total_scale.unsqueeze(-1)
    a_scaled = torch.clamp(a_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    a_scaled = a_scaled.view(original_shape)
    return to_fp4(a_scaled), scaled_block_scale_fp8


def round_up(x, y):
    return (x + y - 1) // y * y


def activation_scale_to_nvfp4(x, g_sf, offsets, blockscale_offsets, block_size):
    m = x.size(0)
    k = x.size(1)
    g = g_sf.size(0)
    padded_m_size = blockscale_offsets[g - 1] + round_up(m - offsets[g - 1], 128)
    block_scale = torch.empty(
        (padded_m_size, k // block_size), dtype=torch.float8_e4m3fn, device="cuda:0"
    )
    v_scaled = torch.empty((m, k // 2), dtype=torch.float4_e2m1fn_x2, device="cuda:0")
    for i in range(len(g_sf)):
        l = offsets[i]
        if i == g - 1:
            r = m
        else:
            r = offsets[i + 1]
        l_sf = blockscale_offsets[i]
        r_sf = l_sf + (r - l + 127) // 128 * 128
        v, b_sf = pytorch_nvfp4_quantize(x[l:r], g_sf[i])
        v_scaled[l:r] = v
        block_scale[l_sf:r_sf] = linear_to_swizzled_128_4(b_sf)

    return v_scaled, block_scale
