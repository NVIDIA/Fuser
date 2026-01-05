# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
from nvfuser_direct import nvf_cutlass

compute_cap = torch.cuda.get_device_capability()
if compute_cap < (10, 0) or compute_cap >= (12, 0):
    pytest.skip(
        reason="MxFp8 Requires compute capability 10.",
        allow_module_level=True,
    )

from python.direct_utils import (
    linear_to_swizzled_128_4,
    swizzled_to_linear_128_4,
)


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def scale_shape(shape, group_shape):
    return tuple(ceildiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def dequantize_mxfp8(tensor_fp8, tensor_sf):
    """Dequantize the fp8 tensor back to high precision."""
    m, k = tensor_fp8.shape
    BLOCK_SIZE = 32
    tensor_sf_linear = swizzled_to_linear_128_4(tensor_sf, m, k)
    # Apply scale factor to all elements in the same block
    sf = tensor_sf_linear.repeat_interleave(BLOCK_SIZE, dim=1).to(torch.float32)
    dqx = tensor_fp8.to(torch.float32)
    # Account for padding of scale factor
    sf = sf[: dqx.shape[0], : dqx.shape[1]]
    dequant = dqx * sf
    return dequant.reshape(m, k)


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def pytorch_mxfp8_quantize(a):
    BLOCK_SIZE = 32
    assert (
        a.size(-1) % BLOCK_SIZE == 0
    ), "The inner-most dim must be divisible by block_size; Padding is not implemented."
    assert a.is_contiguous(), "Only contiguous tensors are supported."

    # Find absolute maximum along blockwise dimension
    original_shape = a.shape
    a_fp32 = a.float().reshape(original_shape[0], -1, BLOCK_SIZE)
    max_abs = torch.amax(torch.abs(a_fp32), dim=-1)

    # Get fp32 block scale factor for fp8
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    block_scale_fp32 = (max_abs / FLOAT8_E4M3_MAX).float()

    # Clamp scale factor within UE8M0
    FLOAT8_UE8M0_EPS = torch.finfo(torch.float8_e8m0fnu).tiny
    FLOAT8_UE8M0_MAX = torch.finfo(torch.float8_e8m0fnu).max
    block_scale_fp32 = torch.clamp(
        block_scale_fp32, min=FLOAT8_UE8M0_EPS, max=FLOAT8_UE8M0_MAX
    )

    # Apply block conversion factor
    a_scaled = a_fp32 / block_scale_fp32.unsqueeze(-1)
    a_scaled = a_scaled.view(original_shape)

    return to_fp8(a_scaled), block_scale_fp32.to(torch.float8_e8m0fnu)


def get_ref_results(
    a_fp8,
    b_fp8,
    a_sf,
    b_sf,
    m,
    n,
):
    _, m_k = a_fp8.shape
    _, n_k = b_fp8.shape
    assert m_k == n_k
    a_in_dtype = dequantize_mxfp8(a_fp8, a_sf)
    b_in_dtype = dequantize_mxfp8(b_fp8, b_sf)
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(128, 128, 128), (128, 128, 256), (256, 128, 128), (128, 256, 256)]
)
@torch.inference_mode()
def test_mxfp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    m, n, k = shape
    block_size = 32
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    alpha = torch.tensor(1.0, device="cuda")
    a_fp8, a_scale_linear = pytorch_mxfp8_quantize(a_dtype)
    b_fp8, b_scale_linear = pytorch_mxfp8_quantize(b_dtype)
    a_scale_interleaved = linear_to_swizzled_128_4(a_scale_linear)
    b_scale_interleaved = linear_to_swizzled_128_4(b_scale_linear)

    expected_out = get_ref_results(
        a_fp8,
        b_fp8,
        a_scale_interleaved,
        b_scale_interleaved,
        m,
        n,
    )
    out = nvf_cutlass.mxfp8_scaled_mm(
        a_fp8, b_fp8, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype))


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    return torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)


@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_mxfp8_scaled_grouped_mm(config, tokens_per_expert_neg_one, out_dtype):
    device = "cuda"
    alignment = 16

    # k dimension is multiple of 128 to avoid padding
    m, n_g, k_g = config
    tokens_per_expert = list(tokens_per_expert_neg_one)
    tokens_per_expert.append(m - sum(tokens_per_expert))
    num_experts = len(tokens_per_expert)

    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)

    expert_offsets = torch.zeros((num_experts + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)
    layout_sfa = torch.zeros((num_experts, 5), device=device, dtype=torch.int32)
    layout_sfb = torch.zeros((num_experts, 5), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    baseline_tensors = []

    for g in range(num_experts):
        m_g = tokens_per_expert[g]
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        b_g = to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        scale_a_shape = scale_shape(a_g.shape, scale_a_group_shape)
        scale_b_shape = scale_shape(b_g.shape, scale_b_group_shape)

        a_scales_tensors.append(torch.randn(scale_a_shape, device=device) * 0.001)
        b_scales_tensors.append(torch.randn(scale_b_shape, device=device) * 0.001)

        baseline = baseline_scaled_mm(
            a_g, b_g, a_scales_tensors[-1], b_scales_tensors[-1], out_dtype
        )
        baseline_tensors.append(baseline)

    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn
    )
    b_stack = torch.empty(
        (num_experts, n_g, k_g), device=device, dtype=torch.float8_e4m3fn
    )

    for g in range(num_experts):
        a_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)

    a_scale_stack = torch.empty(
        (expert_offsets[-1], k_g // 128), device=device, dtype=torch.float32
    )
    b_scale_stack = torch.empty(
        (num_experts, n_g // 128, k_g // 128), device=device, dtype=torch.float32
    )

    for g in range(num_experts):
        a_scale_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_scales_tensors[g]
        b_scale_stack[g] = b_scales_tensors[g].t()
    b_scale_stack = b_scale_stack.transpose(1, 2)

    c_out = torch.empty((expert_offsets[-1], n_g), device=device, dtype=out_dtype)
    a_strides = torch.full(
        (num_experts,), a_stack.stride(0), device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_experts,), c_out.stride(0), device=device, dtype=torch.int64
    )

    a_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    b_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    out_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    a_scales_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    b_scales_ptrs = torch.empty((num_experts,), device=device, dtype=torch.int64)
    workspace = torch.empty((1024 * 1024 * 1024), device=device, dtype=torch.uint8)

    nvf_cutlass.mxfp8_scaled_grouped_mm(
        c_out,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a_stack,
        b_stack,
        a_scale_stack,
        b_scale_stack,
        a_strides,
        a_strides,
        c_strides,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets[:-1],
        workspace,
    )

    for g in range(num_experts):
        baseline = baseline_tensors[g]
        actual = c_out[expert_offsets[g] : expert_offsets[g + 1]]
        torch.testing.assert_close(actual, baseline, rtol=1e-2, atol=5e-4)
        print(f"num_experts={num_experts}, out_dtype={out_dtype}: OK")
