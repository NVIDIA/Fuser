# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype
from python.direct_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    pytorch_nvfp4_quantize,
    is_pre_blackwell,
    microarchitecture_is_pre,
    linear_to_swizzled_128_4,
    round_up,
    activation_scale_to_nvfp4,
    to_fp4,
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
@pytest.mark.skipif(
    not microarchitecture_is_pre(12), reason="Does not support blackwell compute 12.0"
)
@pytest.mark.parametrize("config", [[128, 256, 512], [128, 256, 512]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_scaled_mm(
    nvfuser_direct_test,
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
            bias=None,
            beta=None,
            dtype=torch_dtype_to_nvfuser_dtype(out_dtype),
        )
        fd.add_output(out)

    outputs, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)

    ref_outputs = (
        torch._scaled_mm(
            mat1,
            mat2.t(),
            linear_to_swizzled_128_4(scale1),
            linear_to_swizzled_128_4(scale2),
            None,
            None,
            out_dtype,
        )
        * alpha
    )
    assert outputs[0].allclose(ref_outputs, 1e-2, 1e-2)


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.skipif(
    not microarchitecture_is_pre(12), reason="Does not support blackwell compute 12.0"
)
@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_cutlass_nvfp4_grouped_mm(
    nvfuser_direct_test,
    config,
    tokens_per_expert_neg_one,
    out_dtype,
):
    BLOCK_SIZE = 16

    # k dimension is multiple of 128 to avoid padding
    m, n, k = config
    tokens_per_expert = tokens_per_expert_neg_one
    tokens_per_expert.append(m - sum(tokens_per_expert))
    g = len(tokens_per_expert)

    mat1_ref = torch.testing.make_tensor((m, k), dtype=torch.float32, device="cuda:0")
    # format is g, n, k instead of g, k, n
    mat2_ref = torch.testing.make_tensor(
        (g, n, k), dtype=torch.float32, device="cuda:0"
    )

    offsets = torch.empty((g,), dtype=torch.int32, device="cuda:0")
    blockscale_offsets = torch.empty((g,), dtype=torch.int32, device="cuda:0")
    problem_sizes = torch.empty((g, 3), dtype=torch.int32, device="cuda:0")

    # prepare quantization for mat2
    mat2_gs = torch.empty((g,), dtype=torch.float32, device="cuda:0")
    scale2 = torch.empty(
        (g, n, k // BLOCK_SIZE), dtype=torch.float8_e4m3fn, device="cuda:0"
    )

    acc_tokens = 0
    rounded_acc_tokens = 0
    mat2_scaled = torch.empty(
        (g, n, k // 2), dtype=torch.float4_e2m1fn_x2, device="cuda:0"
    )

    for i in range(g):
        global_sf = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2_ref[i].max()
        offsets[i] = acc_tokens
        blockscale_offsets[i] = rounded_acc_tokens
        acc_tokens += tokens_per_expert[i]
        # Note: we technically don't need to round up, since k is perfectly sized.
        rounded_acc_tokens += round_up(tokens_per_expert[i], 128)

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

        scaled_mat2_i, bs_mat2_i = pytorch_nvfp4_quantize(mat2_ref[i], global_sf)
        mat2_gs[i] = 1.0 / global_sf
        mat2_scaled[i] = scaled_mat2_i
        scale2[i] = linear_to_swizzled_128_4(bs_mat2_i)

    # prepare quantization for mat1
    # note: following sglang implementation, not computing global scaling factor for mat1
    #       similarly, we don't need to apply mat1_gs to alpha
    mat1_gs = torch.ones((g,), dtype=torch.float32, device="cuda:0")
    mat1, scale1 = activation_scale_to_nvfp4(
        mat1_ref, mat1_gs, offsets, blockscale_offsets, BLOCK_SIZE
    )

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        mat1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
        )
        mat2 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        scale1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float8_e4m3fn,
            is_cpu=False,
        )
        scale2 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=True,
            dtype=DataType.Float8_e4m3fn,
            is_cpu=False,
        )
        alpha = fd.define_tensor(
            shape=[-1], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        problem_sizes = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Int32, is_cpu=False
        )
        offsets = fd.define_tensor(
            shape=[-1], contiguity=True, dtype=DataType.Int32, is_cpu=False
        )
        blockscale_offsets = fd.define_tensor(
            shape=[-1], contiguity=True, dtype=DataType.Int32, is_cpu=False
        )
        out = fd.ops.cutlass_nvfp4_grouped_mm(
            mat1,
            mat2,
            scale1,
            scale2,
            alpha,
            problem_sizes,
            offsets,
            blockscale_offsets,
            DataType.BFloat16,
        )
        fd.add_output(out)

    inputs = [
        mat1.view(torch.float4_e2m1fn_x2),
        mat2_scaled.view(torch.float4_e2m1fn_x2).transpose(-1, -2),
        scale1,
        scale2,
        mat2_gs,
        problem_sizes,
        offsets,
        blockscale_offsets,
    ]

    outputs, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)

    o_decomposed_ref = torch.empty(m, n, dtype=torch.bfloat16, device="cuda:0")
    for i in range(g):
        l = offsets[i]
        l_sf = blockscale_offsets[i]
        if i == g - 1:
            r = m
        else:
            r = offsets[i + 1]
        r_sf = round_up(tokens_per_expert[i], 128) + l_sf
        # For some reason I cannot feed mat2_gs[i] as alpha in the torch kernel.
        # This triggers a cublas invalid value error.
        o_decomposed_ref[l:r] = (
            torch._scaled_mm(
                mat1[l:r],
                mat2_scaled[i].transpose(-1, -2),
                scale1[l_sf:r_sf],
                scale2[i],
                None,
                None,
                torch.bfloat16,
            )
            * mat2_gs[i]
        )

    assert torch.allclose(o_decomposed_ref, outputs[0], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.skipif(
    not microarchitecture_is_pre(12), reason="Does not support blackwell compute 12.0"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_fp4_vectorization(
    nvfuser_direct_test,
    dtype,
):
    inputs = [
        torch.ones(4, 8, dtype=dtype, device="cuda"),
        torch.ones(4, dtype=dtype, device="cuda"),
    ]

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T1 = fd.from_pytorch(inputs[1])
        T2 = fd.ops.cast(T0, DataType.Float)
        cast_T1 = fd.ops.cast(T1, DataType.Float)
        broadcast_T1 = fd.ops.broadcast(cast_T1, [False, True])
        T3 = fd.ops.div(T2, broadcast_T1)
        T4 = fd.ops.cast(T3, DataType.Float4_e2m1fn)
        T5 = fd.ops.reshape(T4, [32])
        fd.add_output(T5)

    outputs, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)

    ref_outputs = to_fp4(inputs[0].to(torch.float) / inputs[1].unsqueeze(-1)).reshape(-1)
    assert outputs[0].view(dtype=torch.uint8).allclose(ref_outputs.view(dtype=torch.uint8))