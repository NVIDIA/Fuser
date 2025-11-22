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
    torch.testing.assert_close(outputs[0], ref_outputs, rtol=1e-1, atol=1e-2)


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.parametrize("swizzle_scales", [False])
def test_nv_block_quantization(nvfuser_direct_test, swizzle_scales):
    x = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    x_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max()).to(
        torch.float32
    )

    x_u8, x_scale = pytorch_nvfp4_quantize(x, x_global_scale)

    if swizzle_scales:
        x_scale = linear_to_swizzled_128_4(x_scale)

    def nvfuser_fusion_id0(fd: FusionDefinition):
        x_tv = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.BFloat16, is_cpu=False
        )
        global_scale_tv = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        vals_, scales_ = fd.ops.nv_block_quantize(
            x_tv, global_scale_tv, swizzle_scales, 16
        )
        fd.add_output(vals_)
        fd.add_output(scales_)

    o, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, [x, x_global_scale])

    # # Move tensor from GPU to CPU and get raw data as uint8
    o0_cpu = o[0].cpu()  # Move to CPU
    o1_cpu = o[1].cpu()  # Move to CPU
    o0_uint8 = o0_cpu.view(torch.uint8)  # Reinterpret bytes as uint8
    o1_uint8 = o1_cpu.view(torch.uint8)  # Reinterpret bytes as uint8

    # Move reference tensors to CPU and convert to uint8 for comparison
    x_u8_cpu = x_u8.cpu()  # Move x_u8 to CPU
    x_scale_cpu = x_scale.cpu()  # Move x_scale to CPU
    x_u8_uint8 = x_u8_cpu.view(torch.uint8)  # Reinterpret x_u8 bytes as uint8
    x_scale_uint8 = x_scale_cpu.view(torch.uint8)  # Reinterpret x_scale bytes as uint8

    print("\n--- Comparison Results ---")
    print(f"o[0] (nvfuser quantized values) shape: {o0_uint8.shape}")
    print(f"x_u8 (reference quantized values) shape: {x_u8_uint8.shape}")
    print(f"o[1] (nvfuser scales) shape: {o1_uint8.shape}")
    print(f"x_scale (reference scales) shape: {x_scale_uint8.shape}")

    # Compare first 20 bytes of quantized values (o[0] vs x_u8)
    print("\n--- Quantized Values Comparison (first 20 bytes) ---")
    print(f"nvfuser o[0]: {[hex(x.item()) for x in o0_uint8.flatten()[:20]]}")
    print(f"reference x_u8: {[hex(x.item()) for x in x_u8_uint8.flatten()[:20]]}")

    # Compare first 20 bytes of scales (o[1] vs x_scale)
    print("\n--- Scale Values Comparison (first 20 bytes) ---")
    print(f"nvfuser o[1]: {[hex(x.item()) for x in o1_uint8.flatten()[:20]]}")
    print(f"reference x_scale: {[hex(x.item()) for x in x_scale_uint8.flatten()[:20]]}")

    # Check if quantized values match
    if o0_uint8.flatten().size() == x_u8_uint8.flatten().size():
        values_match = torch.equal(o0_uint8.flatten(), x_u8_uint8.flatten())
        print(f"Quantized values match: {values_match}")
        if not values_match:
            diff_mask = o0_uint8.flatten() != x_u8_uint8.flatten()
            diff_count = diff_mask.sum().item()
            print(f"Number of differing bytes: {diff_count}")
            diff_indices = torch.where(diff_mask)[0][:20]  # Show first 20 mismatches
            print("\n--- Quantized Values Mismatches (first 20) ---")
            for idx in diff_indices:
                print(
                    f"Index {idx.item()}: nvfuser={hex(o0_uint8.flatten()[idx].item())}, reference={hex(x_u8_uint8.flatten()[idx].item())}"
                )
    else:
        print(
            f"Quantized values have different sizes after flattening: {o0_uint8.flatten().size()} vs {x_u8_uint8.flatten().size()}"
        )

    # Check if scale values match
    if o1_uint8.flatten().size() == x_scale_uint8.flatten().size():
        scales_match = torch.equal(o1_uint8.flatten(), x_scale_uint8.flatten())
        print(f"Scale values match: {scales_match}")
        if not scales_match:
            diff_mask = o1_uint8.flatten() != x_scale_uint8.flatten()
            diff_count = diff_mask.sum().item()
            print(f"Number of differing bytes: {diff_count}")
            diff_indices = torch.where(diff_mask)[0][:20]  # Show first 20 mismatches
            print("\n--- Scale Values Mismatches (first 20) ---")
            for idx in diff_indices:
                print(
                    f"Index {idx.item()}: nvfuser={hex(o1_uint8.flatten()[idx].item())}, reference={hex(x_scale_uint8.flatten()[idx].item())}"
                )
    else:
        print(
            f"Scale values have different sizes after flattening: {o1_uint8.flatten().size()} vs {x_scale_uint8.flatten().size()}"
        )


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.parametrize("config", [[1024, 1024, 1024]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_scaled_mm_new(
    nvfuser_direct_test,
    config,
    out_dtype,
):
    in_dtype = torch.float4_e2m1fn_x2
    quantization = nvfp4_quantize

    m, k, n = config
    mat1_ref = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    mat2_ref = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    mat1, scale1, global_sf1 = quantization(mat1_ref)
    mat2, scale2, global_sf2 = quantization(mat2_ref)
    alpha = 1.0 / (global_sf1 * global_sf2)

    inputs = [
        mat1_ref,
        mat2.t(),
        global_sf1,
        linear_to_swizzled_128_4(scale2),
        alpha,
    ]

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        mat1 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.BFloat16, is_cpu=False
        )
        mat2_ = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
            stride_order=[0, 1],
        )

        global_scale_tv = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )

        mat1_, scale1_ = fd.ops.nv_block_quantize(mat1, global_scale_tv, True, 16)

        scale2_ = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False
        )
        alpha = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        out, _, _ = fd.ops.scaled_mm(
            mat1_,
            mat2_,
            scale1_,
            scale2_,
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

    # Check for values that don't match the tolerance
    if not o[0].allclose(ref_o, 1e-2, 1e-2):
        diff = torch.abs(o[0] - ref_o)
        tolerance = 1e-2 + 1e-2 * torch.abs(ref_o)
        mismatch_mask = diff > tolerance

        mismatch_indices = torch.where(mismatch_mask)
        num_mismatches = mismatch_mask.sum().item()
        print(
            f"\n--- Mismatches found: {num_mismatches} out of {o[0].numel()} elements ---"
        )

        # Show first 20 mismatches
        for i in range(min(20, num_mismatches)):
            idx = tuple(ind[i].item() for ind in mismatch_indices)
            nvfuser_val = o[0][idx].item()
            ref_val = ref_o[idx].item()
            diff_val = diff[idx].item()
            tol_val = tolerance[idx].item()
            print(
                f"Index {idx}: nvfuser={nvfuser_val:.6f}, reference={ref_val:.6f}, diff={diff_val:.6f}, tolerance={tol_val:.6f}"
            )

    assert o[0].allclose(ref_o, 1e-2, 1e-2)


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

    torch.testing.assert_close(o_decomposed_ref, outputs[0], atol=1e-2, rtol=1e-2)


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

    ref_outputs = to_fp4(inputs[0].to(torch.float) / inputs[1].unsqueeze(-1)).reshape(
        -1
    )

    torch.testing.assert_close(
        outputs[0].view(dtype=torch.uint8),
        ref_outputs.view(dtype=torch.uint8),
        rtol=1e-1,
        atol=1e-2,
    )
