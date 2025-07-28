# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

from nvfuser import (
    FusionDefinition,
    DataType,
)
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from python.utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    pytorch_nvfp4_quantize,
    is_pre_blackwell,
    linear_to_swizzled_128_4,
)

import pytest


def nvfp4_quantize(x):
    x_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max()).to(
        torch.float32
    )

    x_u8, x_scale = pytorch_nvfp4_quantize(x, x_global_scale)
    return x_u8, x_scale, x_global_scale


# cannot use opinfo test, because the input tensor dtype and fusion definition dtype doesn't match
@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.parametrize("config", [[128, 256, 512], [128, 256, 512]])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_scaled_mm(
    config,
    out_dtype,
):
    in_dtype = torch.float4_e2m1fn_x2
    quantization = nvfp4_quantize

    m, k, n = config
    mat1_ref = torch.randn((m, k), dtype=torch.float32, device="cuda")
    mat2_ref = torch.randn((n, k), dtype=torch.float32, device="cuda")

    mat1, scale1, global_sf1 = quantization(mat1_ref)
    mat2, scale2, global_sf2 = quantization(mat2_ref)
    alpha = 1.0 / (global_sf1 * global_sf2)

    inputs = [
        mat1,
        mat2.t(),
        linear_to_swizzled_128_4(scale1),
        linear_to_swizzled_128_4(scale2),
        alpha,
    ]

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        mat1 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float4_e2m1fn, is_cpu=False
        )
        mat2 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
            stride_order=[0, 1],
        )
        scale1 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False
        )
        scale2 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False
        )
        alpha = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        out, _, _ = fd.ops.scaled_mm(
            mat1,
            mat2,
            scale1,
            scale2,
            alpha,
            None,
            None,
            torch_dtype_to_nvfuser_dtype(out_dtype),
        )
        fd.add_output(out)

    with FusionDefinition() as fd:
        nvfuser_fusion_id0(fd)

    o = fd.execute(inputs)[0]

    # error on reference implementation is too large. maybe use the ref implementation instead.
    # mat1_ref = mat1_ref.reshape(m, k//16, 16).to(torch.bfloat16) * scale1.unsqueeze(-1).to(torch.bfloat16)
    # mat2_ref = mat2_ref.reshape(m, k//16, 16).to(torch.bfloat16) * scale2.unsqueeze(-1).to(torch.bfloat16)
    # ref_o = mat1_ref @ mat2_ref.t() * alpha
    ref_o = (
        torch._scaled_mm(
            mat1,
            mat2.t(),
            linear_to_swizzled_128_4(scale1),
            linear_to_swizzled_128_4(scale2),
            None,
            None,
            torch.bfloat16,
        )
        * alpha
    )
    assert o.allclose(ref_o, 1e-2, 1e-2)
