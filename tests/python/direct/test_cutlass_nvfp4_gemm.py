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
