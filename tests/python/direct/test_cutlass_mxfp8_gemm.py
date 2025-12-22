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
    FLOAT8_E4M3_MAX,
    dequantize_to_dtype,
)


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

    alpha = 1.0
    a_fp8, a_scale_linear = pytorch_mxfp8_quantize(a_dtype)
    b_fp8, b_scale_linear = pytorch_mxfp8_quantize(b_dtype)
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
