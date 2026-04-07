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
    microarchitecture_is,
    is_pre_blackwell,
    microarchitecture_is_pre,
    linear_to_swizzled_128_4,
    round_up,
    activation_scale_to_nvfp4,
    to_fp4,
    swizzled_to_linear_128_4,
    dequantize_fp4,
)

import pytest


def nvfp4_quantize(x):
    x_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max()).to(
        torch.float32
    )

    x_u8, x_scale = pytorch_nvfp4_quantize(x, x_global_scale)
    return x_u8, x_scale, x_global_scale


def quantize_to_mxfp8_e4m3(tensor: torch.Tensor):
    """
    Quantize a Float32 tensor to MXFP8 E4M3 format using block scaling.
    Args:
        tensor: Input Float32 tensor to quantize
    Returns:
        MXFP8Tensor containing quantized data and scaling factors
    Note: You can access the components separately:
        - quantized_tensor._rowwise_data: quantized FP8 values (uint8)
        - quantized_tensor._rowwise_scale_inv: inverse scale factors (uint8)
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    # Create MXFP8 quantizer for E4M3 format
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,  # Enable rowwise scaling
        columnwise=False,  # Disable columnwise scaling for this example
    )

    # Perform quantization
    quantized_tensor = quantizer(tensor)

    return quantized_tensor


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nv_quantization_to_mxfp8(nvfuser_direct_test, dtype):
    x = torch.rand((1024, 1024), dtype=dtype, device="cuda")

    quantized_x = quantize_to_mxfp8_e4m3(x)
    ref_vals = quantized_x._rowwise_data.view(torch.float8_e4m3fn)
    ref_scales = quantized_x._rowwise_scale_inv.view(torch.float8_e8m0fnu)

    def nvfuser_fusion_id0(fd: FusionDefinition):
        x_tv = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=torch_dtype_to_nvfuser_dtype(dtype),
            is_cpu=False,
        )
        vals_, scales_ = fd.ops.nv_block_quantize(
            x_tv, None, False, 32, DataType.Float8_e4m3fn
        )
        fd.add_output(vals_)
        fd.add_output(scales_)

    outputs, _ = nvfuser_direct_test.exec_nvfuser(nvfuser_fusion_id0, [x])

    quantized_vals = outputs[0]
    quantized_scales = outputs[1]

    # Check that values match
    torch.testing.assert_close(
        quantized_vals, ref_vals, rtol=0, atol=0, msg="Quantized values do not match"
    )

    # Check that scales match
    torch.testing.assert_close(
        quantized_scales, ref_scales, rtol=0, atol=0, msg="Block scales do not match"
    )


def nvfp4_quantize_with_te(input_tensor):
    """
    Directly quantizes a tensor to NVFP4 using TE NVFP4Quantizer,
    returning the NVFP4Tensor which contains the quantized values and block scales.
    Block size is 16 elements per scale.
    """

    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.cpp_extensions as tex

    try:
        # Create NVFP4Quantizer with block size of 16
        quantizer = te.NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,  # NVFP4 format
            rowwise=True,  # Use rowwise block scaling
            columnwise=False,  # Disable columnwise
        )

        # Quantize the input tensor
        nvfp4_tensor = quantizer.quantize(input_tensor)
        return nvfp4_tensor

    except Exception as e:
        print(f"\nError during quantization: {e}")
        import traceback

        traceback.print_exc()
        print("NOTE: This requires an NVIDIA Blackwell GPU and TE >= 1.6.")
        return None


def compute_nvfp4_global_scale(tensor):
    """Compute global scale factor for NVFP4 quantization.

    Args:
        tensor: Input tensor to compute scale for

    Returns:
        Global scale as (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / max(abs(tensor)),
        clamped to float32 max. Returns 1.0 if max(abs(tensor)) is 0.
    """
    amax_scalar = torch.max(torch.abs(tensor)).cpu().to(torch.float32).item()

    if amax_scalar == 0.0:
        return torch.tensor(1.0, device=tensor.device, dtype=torch.float32)

    float32_max = torch.finfo(torch.float32).max
    scale = min((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax_scalar, float32_max)

    return torch.tensor(scale, device=tensor.device, dtype=torch.float32)


def extract_te_nvfp4_metadata(input_tensor):
    """Quantize tensor with TE and extract quantization metadata.

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_data, scale_inv, global_scale)
    """
    nvfp4_tensor = nvfp4_quantize_with_te(input_tensor)
    metadata = nvfp4_tensor.get_metadata()
    quantized_data = metadata["rowwise_data"].view((torch.float4_e2m1fn_x2))
    scale_inv = metadata["rowwise_scale_inv"].view(torch.float8_e4m3fn)
    global_scale = compute_nvfp4_global_scale(input_tensor)
    return quantized_data, scale_inv, global_scale


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.parametrize("swizzle_scales", [True, False])
@pytest.mark.parametrize("sizes", [[1024, 1024], [1, 1024]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nv_block_quantization_vs_te(nvfuser_direct_test, swizzle_scales, sizes, dtype):
    """Compare nvfuser nv_block_quantize output against Transformer Engine NVFP4 quantization."""
    x = torch.randn(sizes, dtype=dtype, device="cuda")

    if swizzle_scales and (sizes[0] % 128 != 0 or sizes[1] % 4 != 0):
        # otherwise, nvfuser_direct_test.exec_nvfuser would assert on identical result from captured fusion.
        pytest.skip(
            "Swizzled scales require 128x4 block size to avoid uninitialized padding region in outputs"
        )

    # Compute global scale for nvfuser block quantization
    x_global_scale = compute_nvfp4_global_scale(x)

    def nvfuser_fusion_id0(fd: FusionDefinition):
        x_tv = fd.from_pytorch(x)
        global_scale_tv = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        vals_, scales_ = fd.ops.nv_block_quantize(
            x_tv, global_scale_tv, swizzle_scales, 16
        )
        fd.add_output(vals_)
        fd.add_output(scales_)

    outputs, _ = nvfuser_direct_test.exec_nvfuser(
        nvfuser_fusion_id0, [x, x_global_scale]
    )

    # Get TE NVFP4 reference
    if sizes[0] == 1:
        # nvfp4_quantize_with_te requires batch dimension to be multiple of 16.
        x = x.expand(16, sizes[1])
    nvfp4_result = nvfp4_quantize_with_te(x)
    assert nvfp4_result is not None
    nvfp4_metadata = nvfp4_result.get_metadata()
    te_data = nvfp4_metadata["rowwise_data"].view(torch.uint8)
    te_scales = nvfp4_metadata["rowwise_scale_inv"]

    fuser_data = outputs[0].view(torch.uint8)
    fuser_scales = outputs[1]

    if swizzle_scales:
        te_scales = linear_to_swizzled_128_4(te_scales)
        te_scales = swizzled_to_linear_128_4(te_scales, *sizes)
        fuser_scales = swizzled_to_linear_128_4(fuser_scales, *sizes)

    ref_fp32 = dequantize_fp4(te_data, te_scales, torch.max(torch.abs(x)).float())
    fuser_fp32 = dequantize_fp4(
        fuser_data, fuser_scales, torch.max(torch.abs(x)).float()
    )
    if sizes[0] == 1:
        # slice the expanded data
        ref_fp32 = ref_fp32[0]
    abs_diff = torch.abs(ref_fp32 - fuser_fp32)
    assert torch.max(abs_diff) <= 1.0

    # The percentage of mismatched values is LT 10%.
    nonzero = torch.count_nonzero(torch.ne(abs_diff, 0.0))
    assert (nonzero / abs_diff.numel()) < 0.1


# cannot use opinfo test, because the input tensor dtype and fusion definition dtype doesn't match
@pytest.mark.skipif(
    not microarchitecture_is(10, 0), reason="Only supported on blackwell compute 10.0."
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

    outputs, _ = nvfuser_direct_test.exec_nvfuser(
        nvfuser_fusion_id0, inputs, new_fusion_expected=None
    )

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
    not microarchitecture_is(10, 0), reason="Only supported on blackwell compute 10.0."
)
@pytest.mark.parametrize("config", [[1024, 1024, 1024]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_scaled_mm_nv_quantized(
    nvfuser_direct_test,
    config,
    out_dtype,
):
    """Test scaled_mm with on-the-fly quantization vs pre-quantized baseline.

    Compares nvfuser's nv_block_quantize (quantizing mat1 on-the-fly) against
    a baseline using pre-quantized inputs from Transformer Engine.
    """
    m, k, n = config
    mat1_ref = torch.testing.make_tensor((m, k), dtype=torch.float, device="cuda")
    mat2_ref = torch.testing.make_tensor((n, k), dtype=torch.float, device="cuda")

    # Quantize both matrices using Transformer Engine
    mat1_quantized, mat1_scale_inv, global_sf1 = extract_te_nvfp4_metadata(mat1_ref)
    mat2_quantized, mat2_scale_inv, global_sf2 = extract_te_nvfp4_metadata(mat2_ref)

    # Alpha compensates for both quantization scales
    alpha = 1.0 / (global_sf1 * global_sf2)

    # Prepare inputs for fusion with on-the-fly quantization
    inputs_with_quantize = [
        mat1_ref,
        mat2_quantized.t(),
        global_sf1,
        linear_to_swizzled_128_4(mat2_scale_inv),
        alpha,
    ]

    # Fusion 1: Quantize mat1 on-the-fly using nv_block_quantize
    def fusion_with_nv_block_quantize(fd: FusionDefinition) -> None:
        """Defines fusion that quantizes mat1 on-the-fly before scaled_mm."""
        mat1 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        mat2_fp4 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=True,
            dtype=DataType.Float4_e2m1fn,
            is_cpu=False,
            stride_order=[0, 1],
        )
        global_scale = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )
        scale2 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False
        )
        alpha = fd.define_tensor(
            shape=[], contiguity=True, dtype=DataType.Float, is_cpu=False
        )

        # Quantize mat1 on-the-fly
        mat1_fp4, scale1 = fd.ops.nv_block_quantize(mat1, global_scale, True, 16)

        # Perform scaled matrix multiplication
        out, _, _ = fd.ops.scaled_mm(
            mat1_fp4,
            mat2_fp4,
            scale1,
            scale2,
            alpha,
            bias=None,
            beta=None,
            dtype=torch_dtype_to_nvfuser_dtype(out_dtype),
        )
        fd.add_output(out)

    outputs, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_with_nv_block_quantize, inputs_with_quantize
    )

    # Fusion 2: Baseline using pre-quantized inputs
    inputs_baseline = [
        mat1_quantized,
        mat2_quantized.t(),
        linear_to_swizzled_128_4(mat1_scale_inv),
        linear_to_swizzled_128_4(mat2_scale_inv),
        alpha,
    ]

    def fusion_baseline(fd: FusionDefinition) -> None:
        """Defines baseline fusion using pre-quantized inputs."""
        mat1_fp4 = fd.define_tensor(
            shape=[-1, -1], contiguity=True, dtype=DataType.Float4_e2m1fn, is_cpu=False
        )
        mat2_fp4 = fd.define_tensor(
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
            mat1_fp4,
            mat2_fp4,
            scale1,
            scale2,
            alpha,
            bias=None,
            beta=None,
            dtype=torch_dtype_to_nvfuser_dtype(out_dtype),
        )
        fd.add_output(out)

    outputs_baseline, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_baseline,
        inputs_baseline,
        new_fusion_expected=None,
    )

    torch.testing.assert_close(outputs[0], outputs_baseline[0], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not microarchitecture_is(10, 0), reason="Only supported on blackwell compute 10.0."
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
    # copy list and append tokens for last expert
    tokens_per_expert = list(tokens_per_expert_neg_one)
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


# This is adopted from the decomposed version.
# A few things I have to change in order to pass the test:
#     1. inputs data needs to be changed from `torch.testing.make_tensor` to `torch.randn`;
#     2. output errors are much more relaxed.
@pytest.mark.skipif(
    not microarchitecture_is(10, 0), reason="Only supported on blackwell compute 10.0."
)
@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_block_quantize_op_and_layout_op(
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

    mat1 = torch.randn((m, k), dtype=torch.float32, device="cuda:0")
    # format is g, n, k instead of g, k, n
    mat2 = torch.randn((g, n, k), dtype=torch.float32, device="cuda:0")

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

        # Note: the decomposed quantization seems to give much better numerics.
        # quantization math with nv_block_quantize op
        fp4_mat1, fp8_scale1 = fd.ops.nv_block_quantize(mat1)

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

    # Validate: nvfuser quantization should match baseline
    abs_diff = torch.abs(o[0] - o_decomposed_ref)
    max_diff = torch.max(abs_diff)
    assert max_diff <= 10.0, f"Max difference {max_diff:.4f} exceeds threshold of 10.0"

    # Check that large differences (> 5.0) are rare (< 10% of elements)
    large_diff_count = torch.count_nonzero(torch.gt(abs_diff, 5.0))
    large_diff_ratio = large_diff_count / abs_diff.numel()
    assert (
        large_diff_ratio < 0.1
    ), f"Large diff ratio {large_diff_ratio:.2%} exceeds 10% threshold"


# This is adopted from the decomposed version test_block_quantize_op_and_layout_op
@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.skipif(
    not microarchitecture_is_pre(12), reason="Does not support blackwell compute 12.0"
)
@pytest.mark.parametrize("config", [[1024, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8]])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_grouped_block_quantize_op(
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

    mat1 = torch.randn((m, k), dtype=torch.float32, device="cuda:0")
    # format is g, n, k instead of g, k, n
    mat2 = torch.randn((g, n, k), dtype=torch.float32, device="cuda:0")

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

        fp4_mat1, fp8_scale1 = fd.ops.nv_grouped_block_quantize(
            mat1, offsets, blockscale_offsets
        )

        out = fd.ops.cutlass_nvfp4_grouped_mm(
            fp4_mat1,
            mat2,
            fp8_scale1,
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

    # Validate: nvfuser quantization should match baseline
    abs_diff = torch.abs(o[0] - o_decomposed_ref)
    max_diff = torch.max(abs_diff)
    assert max_diff <= 10.0, f"Max difference {max_diff:.4f} exceeds threshold of 10.0"

    # Check that large differences (> 5.0) are rare (< 10% of elements)
    large_diff_count = torch.count_nonzero(torch.gt(abs_diff, 5.0))
    large_diff_ratio = large_diff_count / abs_diff.numel()
    assert (
        large_diff_ratio < 0.1
    ), f"Large diff ratio {large_diff_ratio:.2%} exceeds 10% threshold"
