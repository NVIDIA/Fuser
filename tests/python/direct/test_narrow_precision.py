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
#from python.direct_utils import (
from narrow_precision import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_EPS,
    FLOAT8_E4M3_MAX,
    pytorch_nvfp4_quantize,
    #is_pre_blackwell,
    linear_to_swizzled_128_4,
    round_up,
    activation_scale_to_nvfp4,
)

import pytest


def nvfp4_quantize(x):
    x_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max()).to(
        torch.float32
    )

    x_u8, x_scale = pytorch_nvfp4_quantize(x, x_global_scale)
    return x_u8, x_scale, x_global_scale


## cannot use opinfo test, because the input tensor dtype and fusion definition dtype doesn't match
#@pytest.mark.skipif(
#    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
#)
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

    o, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)

    ref_o = (
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
    assert o[0].allclose(ref_o, 1e-2, 1e-2)


#@pytest.mark.skipif(
#    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
#)
#@pytest.mark.parametrize("config", [[1024, 128, 256]])
#@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
#@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
#def test_cutlass_nvfp4_grouped_mm(
def test1(
    nvfuser_direct_test,
    config,
    tokens_per_expert_neg_one,
    out_dtype,
):
    INPUT_DTYPE = torch.uint8
    BLOCK_SIZE = 16

    # k dimension is multiple of 128 to avoid padding
    m, n, k = config
    tokens_per_expert = tokens_per_expert_neg_one
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
        mat2_gs[i] = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2[i].max()
        offsets[i] = acc_tokens
        blockscale_offsets[i] = rounded_acc_tokens
        acc_tokens += tokens_per_expert[i]
        # Note: we technically don't need to round up, since k is perfectly sized.
        rounded_acc_tokens += round_up(tokens_per_expert[i], 128)

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

        scaled_mat2_i, bs_mat2_i = pytorch_nvfp4_quantize(mat2[i], mat2_gs[i])
        mat2_scaled[i] = scaled_mat2_i
        scale2[i] = linear_to_swizzled_128_4(bs_mat2_i)

    # prepare quantization for mat1
    # note: following sglang implementation, not computing global scaling factor for mat1
    #       similarly, we don't need to apply mat1_gs to alpha
    mat1_gs = torch.ones((g,), dtype=torch.float32, device="cuda:0")
    mat1, scale1 = activation_scale_to_nvfp4(
        mat1, mat1_gs, offsets, blockscale_offsets, BLOCK_SIZE
    )

    ab_strides = torch.full((g,), k, dtype=torch.int64, device="cuda:0")
    c_strides = torch.full((g,), n, dtype=torch.int64, device="cuda:0")

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

    #o, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)
    o, _ = nvfuser_direct_test(nvfuser_fusion_id0, inputs)

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

    assert torch.allclose(o_decomposed_ref, o[0], atol=1e-2, rtol=1e-2)


#@pytest.mark.skipif(
#    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
#)
#@pytest.mark.parametrize("config", [[1024, 128, 16*9]])
#@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
#@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
#def test_layout_op_and_cutlass_nvfp4_grouped_mm(
def test(
    nvfuser_direct_test,
    config,
    tokens_per_expert_neg_one,
    out_dtype,
):
    INPUT_DTYPE = torch.uint8
    BLOCK_SIZE = 16

    # k dimension is multiple of 128 to avoid padding
    m, n, k = config
    tokens_per_expert = tokens_per_expert_neg_one
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
        mat2_gs[i] = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2[i].max()
        offsets[i] = acc_tokens
        blockscale_offsets[i] = rounded_acc_tokens
        acc_tokens += tokens_per_expert[i]
        # Note: we technically don't need to round up, since k is perfectly sized.
        rounded_acc_tokens += round_up(tokens_per_expert[i], 128)

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

        scaled_mat2_i, bs_mat2_i = pytorch_nvfp4_quantize(mat2[i], mat2_gs[i])
        mat2_scaled[i] = scaled_mat2_i
        scale2[i] = linear_to_swizzled_128_4(bs_mat2_i)


    ab_strides = torch.full((g,), k, dtype=torch.int64, device="cuda:0")
    c_strides = torch.full((g,), n, dtype=torch.int64, device="cuda:0")

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
        m_size = m
        k_size = k
        k_tile_size = k_size //  16;
        # m_size = fd.ops.size(mat1, 0)
        # k_size = fd.ops.size(mat1, 1)
        # k_tile_size = fd.ops.div(k_size, 16)
        # using primitive operations to handle quantization
        reshaped_mat1 = fd.ops.reshape(mat1, [m_size, k_tile_size, 16])

        scale1 = fd.ops.abs(reshaped_mat1)
        scale1 = fd.ops.max(scale1, 2)
        scale1 = fd.ops.div(scale1, FLOAT4_E2M1_MAX)
        scale1 = fd.ops.clamp(scale1, FLOAT8_E4M3_EPS, FLOAT8_E4M3_MAX)

        broadcast_scale1 = fd.ops.broadcast(scale1, [False, False, True])
        reshaped_scaled_mat1 = fd.ops.div(reshaped_mat1, broadcast_scale1)
        reshaped_scaled_mat1 = fd.ops.clamp(reshaped_scaled_mat1, -FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)

        scaled_mat1 = fd.ops.reshape(reshaped_scaled_mat1, [m_size, k_size])
        # should I clamp here before cast?!
        fp4_mat1 = fd.ops.cast(scaled_mat1, DataType.Float4_e2m1fn)
        fp8_scale1 = fd.ops.cast(scale1, DataType.Float8_e4m3fn)
        # NOTE: I need to add an entry for translation rule to print out this
        layout_fp8_scale1 = fd.ops.preprocess_grouped_matmul_input_sf(fp8_scale1, offsets, blockscale_offsets)
        # NOTE: it's not working with the grouped_mm. Looks like segmentation is a bit different. But I think it's also exposing some dependency issue above.
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
        fd.add_output(fp4_mat1)
        fd.add_output(layout_fp8_scale1)
        fd.add_output(fp8_scale1)

    inputs = [
        mat1,
        mat2_scaled.view(torch.float4_e2m1fn_x2).transpose(-1, -2),
        scale2,
        mat2_gs,
        problem_sizes,
        offsets,
        blockscale_offsets,
    ]

    #o, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, inputs)
    o, _ = nvfuser_direct_test(nvfuser_fusion_id0, inputs)

    # quantization for activation is needed for reference.
    # note: following sglang implementation, not computing global scaling factor for mat1
    #       similarly, we don't need to apply mat1_gs to alpha
    mat1_gs = torch.ones((g,), dtype=torch.float32, device="cuda:0")
    mat1_fp4, scale1, vanilla_s1 = activation_scale_to_nvfp4(
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
    assert mat1_fp4.view(torch.uint8).equal(o[1].view(torch.uint8))
    assert o[3].equal(vanilla_s1)

    # a very rough and wrong way to validate
    mask = scale1 != 0
    buffer_s1 = o[2].as_strided(scale1.size(), scale1.stride())
    masked_s1 = torch.where(mask, buffer_s1, 0)

    breakpoint()
    assert torch.allclose(o_decomposed_ref, o[0], atol=1e-2, rtol=1e-2)

def fn(
    fusion_func,
    inputs,
    *,
    device=None,
):
    # Copy inputs because aliased outputs can modify inputs when running
    # FusionDefinition

    # Execute a fusion function and capture the string python definition
    with FusionDefinition() as fd:
        fusion_func(fd)
    torch.manual_seed(0)
    out = fd.execute(
        inputs,
        device=device,
    )
    return out, fd

# misaligned memory access
#test(fn, [1024, 128, 16*9], [115, 144, 8], [torch.bfloat16])
test(fn, [1024, 128, 256], [115, 144, 8], [torch.bfloat16])
