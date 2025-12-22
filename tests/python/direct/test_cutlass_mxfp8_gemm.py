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
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_to_dtype,
)


def activation_scale_to_nvfp8(x, g_sf, offsets, blockscale_offsets, block_size):
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


def get_ref_results(
    a_fp8,
    b_fp8,
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
    _, m_k = a_fp8.shape
    _, n_k = b_fp8.shape
    assert m_k == n_k
    a_in_dtype = dequantize_to_dtype(
        a_fp8, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_to_dtype(
        b_fp8, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(128, 128, 128), (128, 128, 256), (256, 128, 128), (128, 256, 256)]
)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    m, n, k = shape
    block_size = 32
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    a_fp8, a_scale_linear = pytorch_mxfp8_quantize(a_dtype, a_global_scale)
    b_fp8, b_scale_linear = pytorch_mxfp8_quantize(b_dtype, b_global_scale)
    a_scale_interleaved = linear_to_swizzled_128_4(a_scale_linear)
    b_scale_interleaved = linear_to_swizzled_128_4(b_scale_linear)

    expected_out = get_ref_results(
        a_fp8,
        b_fp8,
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
    out = nvf_cutlass.mxfp8_scaled_mm(
        a_fp8, b_fp8, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)
