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
