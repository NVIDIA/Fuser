// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Testing fusions used in low precision recipes

#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <preseg_passes/pre_segmenter.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using FP4RecipeTest = NVFuserTest;

// Testing the following function:
// https://github.com/pytorch/ao/blob/b1163dc63dfa22d403586672fd3648cd661c5003/torchao/prototype/mx_formats/nvfp4_tensor.py#L545-L617
//
// clang-format off
//
// def nvfp4_quantize(
//     data_hp: torch.Tensor,
//     block_size: int = 16,
//     per_tensor_scale: Optional[torch.Tensor] = None,
// ) -> tuple[torch.Tensor, torch.Tensor]:
//     """NVIDIA FP4 quantization with UE4M3 scales.
//
//     Implements the NVIDIA algorithm for quantizing tensors to FP4 format
//     with unsigned E4M3 (UE4M3) scales.
//
//     Args:
//         data_hp: High precision input tensor (bfloat16 or float32)
//         block_size: Block size for quantization (must be 16)
//         per_tensor_amax: Optional pre-computed absolute maximum for calibration.
//             If provided, uses per-tensor scaling. If None, uses block-wise scaling only.
//
//     Returns:
//         tuple: A tuple containing:
//             - total_scale_fp8: Blockwise scales in float8_e4m3fn format
//             - per_tensor_scale: Global per-tensor scale if per_tensor_amax provided, else None
//             - data_lp: Packed FP4 data (2 values per byte)
//
//     Raises:
//         AssertionError: If input dtype is not supported, tensor size is not
//             divisible by block_size, tensor is not contiguous, or block_size != 16
//     """
//     assert data_hp.dtype in (torch.bfloat16, torch.float), (
//         f"{data_hp.dtype} not supported"
//     )
//     assert data_hp.numel() % block_size == 0, "unsupported"
//     assert data_hp.is_contiguous(), "unsupported"
//     assert block_size == 16, "NVFP4 requires block_size=16"
//
//     orig_shape = data_hp.shape
//     data_hp = data_hp.reshape(orig_shape[0], -1, block_size)
//
//     max_abs = torch.amax(torch.abs(data_hp), dim=-1)
//     # These scales are currently in fp32, we are going to `quantize` them to e4m3
//     block_scale = max_abs / F4_E2M1_MAX
//
//     out_scales = None
//     if per_tensor_scale is None:
//         # We are doing single level scaling
//         block_scale_fp8 = torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
//             torch.float8_e4m3fn
//         )
//         block_scale_fp32 = block_scale_fp8.to(torch.float32)
//         data_scaled = data_hp / block_scale_fp32.unsqueeze(-1)
//         out_scales = block_scale_fp8
//     else:
//         # We are doing two level scaling,
//         # This will likely be calibrated but
//         # we want the per_tensor_scale ~= amax of the block_scale_fp32
//         block_scale_fp32 = block_scale.to(torch.float32)
//         # Quantize the blockwise scales w/ the per_tensor_scale
//         scaled_block_scales = block_scale_fp32 / per_tensor_scale
//         scaled_block_scales_fp8 = torch.clamp(
//             scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
//         ).to(torch.float8_e4m3fn)
//         scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
//         # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
//         # To apply to data
//         total_scale = per_tensor_scale * scaled_block_scales_fp32
//         data_scaled = data_hp / total_scale.unsqueeze(-1)
//         out_scales = scaled_block_scales_fp8
//
//     data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
//     data_scaled = data_scaled.view(orig_shape)
//     data_lp = f32_to_f4_unpacked(data_scaled.float())
//     # TODO: NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
//     # data_lp = pack_uint4(data_lp).view(torch.float4_e2m1fn_x2)
//     data_lp = pack_uint4(data_lp)
//     return out_scales, data_lp
//
// clang-format on

constexpr int64_t block_size = 16;
constexpr double F4_E2M1_MAX = 6.0;
constexpr double E4M3_EPS = 0.015625;
constexpr double F8E4M3_MAX = 448.0;

class NVFP4QuantizeTest : public FP4RecipeTest,
                          public ::testing::WithParamInterface<DataType> {};

TEST_P(NVFP4QuantizeTest, WithoutPerTensorAmax) {
  auto data_hp_dtype = GetParam();

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion.addInput(tv_data_hp);

  // Unfortunately reshape uses outer-split, but I wanted inner split.
  // So here I just use an arbitrary shape to create a ViewOp. I will manually
  // modify the rFactor domain later, so the shape is here not important.
  auto tv_data_hp_reshaped = reshape(tv_data_hp, {5, 7 * 16}, {5, -1, 16});
  tv_data_hp_reshaped->setLoopDomain(tv_data_hp_reshaped->getRootDomain());
  tv_data_hp_reshaped->split(-1, block_size);
  tv_data_hp_reshaped->commitLeafToLogical();

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});
  // These scales are currently in fp32, we are going to `quantize` them to e4m3
  // Note: in the torchao implementation, tv_block_scale is bf16 if the input is
  // bf16 But in our case, tv_block_scale is always fp32, regardless of the
  // input dtype.
  auto tv_block_scale = div(
      tv_data_hp_amax, IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));
  auto tv_block_scale_clamp = clamp(
      tv_block_scale,
      IrBuilder::create<Val>(E4M3_EPS, DataType::Float),
      IrBuilder::create<Val>(F8E4M3_MAX, DataType::Float));
  auto tv_block_scale_fp8 =
      castOp(DataType::Float8_e4m3fn, tv_block_scale_clamp);
  // TODO: should we just use auto tv_block_scale_fp32 = tv_block_scale_clamp?
  auto tv_block_scale_fp32 = castOp(DataType::Float, tv_block_scale_fp8);
  auto tv_block_scale_fp32_unsqueeze = unsqueeze(tv_block_scale_fp32, -1);
  auto tv_data_scaled = div(tv_data_hp_reshaped, tv_block_scale_fp32_unsqueeze);
  auto tv_data_scaled_clamp = clamp(
      tv_data_scaled,
      IrBuilder::create<Val>(-F4_E2M1_MAX, DataType::Float),
      IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));
  // Arbitrarily choose 5, 7, and 16 to generate a merge for reshape
  auto tv_data_lp_fp4 = castOp(DataType::Float4_e2m1fn, tv_data_scaled_clamp);
  std::cout << "cast:" << tv_data_lp_fp4->toString() << std::endl;
  auto tv_data_lp = reshape(tv_data_lp_fp4, {5, 7, 16}, {5, 7 * 16});

  fusion.addOutput(tv_block_scale_fp8);
  fusion.addOutput(tv_data_lp);

  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      &fusion);

  fusion.printMath();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    NVFP4QuantizeTest,
    ::testing::Values(DataType::BFloat16, DataType::Float),
    [](auto info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

} // namespace nvfuser
