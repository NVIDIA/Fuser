# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
from nvfuser_direct import nvf_cutlass

if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

from python.utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_to_dtype,
    linear_to_swizzled_128_4,
    pytorch_nvfp4_quantize,
    unpack_fp4_bytes,
    round_up,
    activation_scale_to_nvfp4,
)


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    a_fp4, a_scale_linear = pytorch_nvfp4_quantize(a_dtype, a_global_scale)
    b_fp4, b_scale_linear = pytorch_nvfp4_quantize(b_dtype, b_global_scale)
    a_scale_interleaved = linear_to_swizzled_128_4(a_scale_linear)
    b_scale_interleaved = linear_to_swizzled_128_4(b_scale_linear)

    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        "cuda",
    )
    out = nvf_cutlass.nvfp4_scaled_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
)
@torch.inference_mode()
def test_nvfp4_gemm_epilogue(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    global_normconst = torch.tensor(2, dtype=torch.float, device="cuda")

    a_fp4, a_scale_linear = pytorch_nvfp4_quantize(a_dtype, a_global_scale)
    b_fp4, b_scale_linear = pytorch_nvfp4_quantize(b_dtype, b_global_scale)
    a_scale_interleaved = linear_to_swizzled_128_4(a_scale_linear)
    b_scale_interleaved = linear_to_swizzled_128_4(b_scale_linear)

    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        "cuda",
    )

    expected_out_fp4, expected_out_scale_linear = pytorch_nvfp4_quantize(
        expected_out, global_normconst
    )
    expected_out_scale_interleaved = linear_to_swizzled_128_4(expected_out_scale_linear)

    out_fp4, out_scale_interleaved = nvf_cutlass.nvfp4_scaled_mm_blockscale(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, global_normconst
    )

    # Convert to unpacked fp32 to check nvfp4 tensors.
    expected_out_fp32 = unpack_fp4_bytes(expected_out_fp4, torch.float32)
    out_fp32 = unpack_fp4_bytes(out_fp4, torch.float32)

    # The absolute max difference is 2.0.
    abs_diff = torch.abs(expected_out_fp32 - out_fp32)
    assert torch.max(abs_diff) <= 2.0

    # The percentage of mismatched values is 1%.
    nonzero = torch.count_nonzero(torch.ne(abs_diff, 0.0))
    assert (nonzero / abs_diff.numel()) < 0.01

    # Compare scale factors
    # rtol = epsilon = 2**(-3) for fp8_m4e3
    # atol = 2**(num_exponent_bits + smallest_exponent_value)
    torch.testing.assert_close(
        out_scale_interleaved.to(torch.float),
        expected_out_scale_interleaved.to(torch.float),
        atol=2e-3,
        rtol=0.125,
    )


@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_grouped_mm(
    config,
    tokens_per_expert_neg_one,
    out_dtype,
):
    nvfp4_block_size = 16

    # k dimension is multiple of 128 to avoid padding
    m, n, k = config
    tokens_per_expert = list(tokens_per_expert_neg_one)
    tokens_per_expert.append(m - sum(tokens_per_expert))
    g = len(tokens_per_expert)

    mat1 = torch.testing.make_tensor((m, k), dtype=torch.float32, device="cuda:0")
    mat2 = torch.testing.make_tensor((g, n, k), dtype=torch.float32, device="cuda:0")
    ab_strides = torch.full((g,), k, dtype=torch.int64, device="cuda:0")
    c_strides = torch.full((g,), n, dtype=torch.int64, device="cuda:0")

    offsets = torch.empty((g,), dtype=torch.int32, device="cuda:0")
    blockscale_offsets = torch.empty((g,), dtype=torch.int32, device="cuda:0")
    problem_sizes = torch.empty((g, 3), dtype=torch.int32, device="cuda:0")
    mat2_gs = torch.empty((g,), dtype=torch.float32, device="cuda:0")
    scale2 = torch.empty(
        (g, n, k // nvfp4_block_size), dtype=torch.float8_e4m3fn, device="cuda:0"
    )

    accumulated_tokens = 0
    rounded_accumulated_tokens = 0
    mat2_fp4 = torch.empty(
        (g, n, k // 2), dtype=torch.float4_e2m1fn_x2, device="cuda:0"
    )

    # Use tokens_per_expert to calculate offsets into m dimension of input tensor.
    # The blockscale offset per expert is rounded to 128 to match the alignment
    # requirements for blockscale factor.
    for i in range(g):
        mat2_gs[i] = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2[i].max()
        offsets[i] = accumulated_tokens
        blockscale_offsets[i] = rounded_accumulated_tokens
        accumulated_tokens += tokens_per_expert[i]
        # Note: we technically don't need to round up, since k is perfectly sized.
        rounded_accumulated_tokens += round_up(tokens_per_expert[i], 128)

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

        mat2_fp4_i, mat2_sf_i = pytorch_nvfp4_quantize(mat2[i], mat2_gs[i])
        mat2_fp4[i] = mat2_fp4_i
        scale2[i] = linear_to_swizzled_128_4(mat2_sf_i)

    # prepare quantization for mat1
    # note: following sglang implementation, not computing global scaling factor for mat1
    #       similarly, we don't need to apply mat1_gs to alpha
    mat1_gs = torch.ones((g,), dtype=torch.float32, device="cuda:0")
    mat1_fp4, scale1 = activation_scale_to_nvfp4(
        mat1, mat1_gs, offsets, blockscale_offsets, nvfp4_block_size
    )

    out = nvf_cutlass.nvfp4_scaled_grouped_mm(
        mat1_fp4,
        mat2_fp4,
        scale1,
        scale2,
        mat2_gs,
        ab_strides,
        c_strides,
        problem_sizes,
        offsets,
        blockscale_offsets,
        out_dtype,
    )

    # Create pytorch expected output reference
    # For each expert, apply nvfp4 gemm. Slice the input matrix given the tokens_per_expert.
    # C[start:stop] = A[start:stop] @ B[expert].
    out_decomposed_ref = torch.empty(m, n, dtype=out_dtype, device="cuda:0")
    for i in range(g):
        l = offsets[i]
        l_sf = blockscale_offsets[i]
        if i == g - 1:
            r = m
        else:
            r = offsets[i + 1]
        r_sf = round_up(tokens_per_expert[i], 128) + l_sf
        # A cublas invalid value error occurs when passing in mat2_gs[i] as
        # alpha in the torch kernel.
        out_decomposed_ref[l:r] = (
            torch._scaled_mm(
                mat1_fp4[l:r],
                mat2_fp4[i].transpose(-1, -2),
                scale1[l_sf:r_sf],
                scale2[i],
                None,
                None,
                out_dtype,
            )
            * mat2_gs[i]
        )

    assert torch.allclose(out_decomposed_ref, out, atol=1e-2, rtol=1e-2)
