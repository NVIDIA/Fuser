# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from python.utils import set_env
from python.direct_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_EPS,
    FLOAT8_E4M3_MAX,
    pytorch_nvfp4_quantize,
    is_pre_blackwell,
    microarchitecture_is_pre,
    linear_to_swizzled_128_4,
    round_up,
    activation_scale_to_nvfp4,
)

import pytest


# FIXME: this test needs to be merged back into test_narrow_precision.py.
# We have indexer issue: https://github.com/NVIDIA/Fuser/issues/5200, which
# forces the adoption of environment variable in order to avoid codegen
# assertion. Having this as a separate test file would avoid environment
# variable contamination from others.
@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.skipif(
    not microarchitecture_is_pre(12), reason="Does not support blackwell compute 12.0"
)
@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_layout_op_and_cutlass_nvfp4_grouped_mm(
    nvfuser_direct_test,
    config,
    tokens_per_expert_neg_one,
    out_dtype,
):
    BLOCK_SIZE = 16

    # k dimension is multiple of 4 * 16 to avoid padding on block scaling factor
    m, n, k = config
    assert k % 64 == 0
    tokens_per_expert = list(tokens_per_expert_neg_one)
    tokens_per_expert.append(m - sum(tokens_per_expert))
    g = len(tokens_per_expert)

    mat1 = torch.testing.make_tensor((m, k), dtype=torch.float32, device="cuda:0")
    # format is g, n, k instead of g, k, n
    mat2 = torch.testing.make_tensor((g, n, k), dtype=torch.float32, device="cuda:0")

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
        global_sf = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2[i].max()
        offsets[i] = acc_tokens
        blockscale_offsets[i] = rounded_acc_tokens
        acc_tokens += tokens_per_expert[i]
        # Note: we technically don't need to round up, since k is perfectly sized.
        rounded_acc_tokens += round_up(tokens_per_expert[i], 128)

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

        scaled_mat2_i, bs_mat2_i = pytorch_nvfp4_quantize(mat2[i], global_sf)
        mat2_gs[i] = 1.0 / global_sf
        mat2_scaled[i] = scaled_mat2_i
        scale2[i] = linear_to_swizzled_128_4(bs_mat2_i)

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        mat1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float,
            is_cpu=False,
        )
        mat2 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
            stride_order=[2, 0, 1],
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
        # TODO: fix dynamic shape in issue https://github.com/NVIDIA/Fuser/issues/5199
        # m_size = fd.ops.size(mat1, 0)
        # k_size = fd.ops.size(mat1, 1)
        # k_tile_size = fd.ops.div(k_size, 16)
        # use static shape as a temporary WAR.
        m_size = m
        k_size = k
        k_tile_size = k_size // 16
        # using primitive operations to handle quantization
        reshaped_mat1 = fd.ops.reshape(mat1, [m_size, k_tile_size, 16])

        # quantization math to compute block scaling factor
        scale1 = fd.ops.abs(reshaped_mat1)
        scale1 = fd.ops.max(scale1, 2)
        scale1 = fd.ops.div(scale1, FLOAT4_E2M1_MAX)
        scale1 = fd.ops.clamp(scale1, FLOAT8_E4M3_EPS, FLOAT8_E4M3_MAX)
        broadcast_scale1 = fd.ops.broadcast(scale1, [False, False, True])
        reshaped_scaled_mat1 = fd.ops.div(reshaped_mat1, broadcast_scale1)
        reshaped_scaled_mat1 = fd.ops.clamp(
            reshaped_scaled_mat1, -FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX
        )

        scaled_mat1 = fd.ops.reshape(reshaped_scaled_mat1, [m_size, k_size])

        # cast the quantized tv and block sf to proper dtype
        fp4_mat1 = fd.ops.cast(scaled_mat1, DataType.Float4_e2m1fn)
        fp8_scale1 = fd.ops.cast(scale1, DataType.Float8_e4m3fn)

        # swizzle & pad block sf
        layout_fp8_scale1 = fd.ops.preprocess_grouped_matmul_input_sf(
            fp8_scale1, offsets, blockscale_offsets
        )
        out = fd.ops.cutlass_nvfp4_grouped_mm(
            fp4_mat1,
            mat2,
            layout_fp8_scale1,
            scale2,
            alpha,
            problem_sizes,
            offsets,
            blockscale_offsets,
            DataType.BFloat16,
        )
        fd.add_output(out)

    inputs = [
        mat1,
        mat2_scaled.view(torch.float4_e2m1fn_x2).transpose(-1, -2),
        scale2,
        mat2_gs,
        problem_sizes,
        offsets,
        blockscale_offsets,
    ]

    # FIXME: force indexing to use IdModel indexer to avoid indexing error.
    # see issue: https://github.com/NVIDIA/Fuser/issues/5200
    with set_env(NVFUSER_ENABLE="id_model(all)"):
        o, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)

    # quantization for activation is needed for reference.
    # note: following sglang implementation, not computing global scaling factor for mat1
    #       similarly, we don't need to apply mat1_gs to alpha
    mat1_gs = torch.ones((g,), dtype=torch.float32, device="cuda:0")
    mat1_fp4, scale1 = activation_scale_to_nvfp4(
        mat1, mat1_gs, offsets, blockscale_offsets, BLOCK_SIZE
    )
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
                mat1_fp4[l:r],
                mat2_scaled[i].transpose(-1, -2),
                scale1[l_sf:r_sf],
                scale2[i],
                None,
                None,
                torch.bfloat16,
            )
            * mat2_gs[i]
        )

    torch.testing.assert_close(o_decomposed_ref, o[0], atol=1e-2, rtol=1e-2)
