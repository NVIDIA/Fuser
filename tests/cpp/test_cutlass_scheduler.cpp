// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <cutlass/block_scaling.h>
#include <cutlass/evt.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/cutlass_executor.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <cstdlib>

namespace nvfuser {

using CutlassExecutorTest = NVFuserTest;

struct QuantizedTensor {
  at::Tensor elts;
  at::Tensor block_scale;
  at::Tensor global_scale;
};

at::Tensor packUint4(at::Tensor uint8_data) {
  std::vector<int64_t> down_shape = uint8_data.sizes().vec();
  down_shape.back() /= 2;

  // converting to uint8 for operations
  NVF_ERROR(uint8_data.size(-1) % 2 == 0);
  uint8_data = uint8_data.contiguous().view(-1);

  at::indexing::TensorIndex shifted_range = at::indexing::Slice(
      /*start_index=*/1, /*stop_index=*/std::nullopt, /*step_index=*/2);
  at::indexing::TensorIndex unshifted_range = at::indexing::Slice(
      /*start_index=*/0, /*stop_index=*/std::nullopt, /*step_index=*/2);
  return (uint8_data.index({shifted_range}).bitwise_left_shift(4) |
          uint8_data.index({unshifted_range}))
      .view(down_shape);
}

constexpr int64_t lowBitsMask(int64_t n) {
  return (1 << n) - 1;
}

at::Tensor unpackAnyFloatingPointToFp32(
    at::Tensor x,
    int64_t ebits,
    int64_t mbits) {
  NVF_ERROR(x.scalar_type() == at::kFloat);
  NVF_ERROR(1 + ebits + mbits <= 8);

  constexpr int64_t EBITS_F32 = 8;
  constexpr int64_t MBITS_F32 = 23;
  constexpr int64_t F32_EXP_BIAS = lowBitsMask(EBITS_F32 - 1);

  // calculate constants
  const int64_t exp_bias = lowBitsMask(ebits - 1);
  const int64_t max_int = lowBitsMask(ebits + mbits);
  const int64_t sign_mask = 1 << (ebits + mbits);

  // TODO document this better
  const int64_t magic_adder = lowBitsMask(MBITS_F32 - mbits - 1);

  // all E bits and M bits are 1s
  const int64_t max_normal = (1 << (lowBitsMask(ebits) - exp_bias)) *
      (lowBitsMask(mbits + 1) / (1 << mbits));

  // E bits = 1, M bits = 0
  const int64_t min_normal = 1 << (1 - exp_bias);

  const int64_t denorm_exp = (
      // exp bias conversion between formats
      (F32_EXP_BIAS - exp_bias)
      // mantissa length difference between formats
      + (MBITS_F32 - mbits)
      // add one to encoded exponent for denormalized numbers
      + 1);
  const int64_t denorm_mask_int = denorm_exp << MBITS_F32;

  // reinterpret int32 as float32
  auto options = at::TensorOptions().dtype(at::kInt).device(x.device());
  at::Tensor denorm_mask_float =
      at::scalar_tensor(denorm_mask_int, options.dtype(at::kInt))
          .view(at::kFloat);

  // save the sign
  // Note that we have torch.uint32, but some ops like cpu bit shifts
  // do not work on it. So, we stay in int32.
  x = x.view(at::kInt);
  at::Tensor sign = x.bitwise_and(0x80000000);

  // set everything to positive, will add sign back at the end
  x = x.bitwise_xor(sign);

  // TODO: can the branch floating point comparisons below be done without
  // converting to float? probably but need to verify
  x = x.view(at::kFloat);

  // rewrite saturate/denorm/norm branches without explicit data dependent
  // control flow, to be more compiler friendly
  const at::Tensor saturate_mask = x >= max_normal;
  const at::Tensor denormal_mask =
      saturate_mask.logical_not().logical_and(x < min_normal);
  const at::Tensor normal_mask =
      saturate_mask.logical_or(denormal_mask).logical_not();

  //
  // branch 1: saturate to max val - handled later in the code which combines
  //   the branches
  //

  //
  // branch 2: to conversion to denormal as well as rounding up to normal
  //
  at::Tensor denormal_x = x + denorm_mask_float;
  denormal_x = denormal_x.view(at::kInt);
  denormal_x -= denorm_mask_int;
  denormal_x = denormal_x.to(at::kByte);

  //
  // branch 3: stay in normal range, adjust the exponent and round
  //
  at::Tensor normal_x = x.view(at::kInt);
  // resulting mantissa is odd
  at::Tensor mant_odd =
      normal_x.bitwise_right_shift(MBITS_F32 - mbits).bitwise_and(1);
  // update exponent, rounding bias part 1
  int64_t val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder;
  normal_x += val_to_add;
  // rounding bias part 2
  normal_x += mant_odd;
  // take the bits!
  normal_x = normal_x.bitwise_right_shift(MBITS_F32 - mbits);
  normal_x = normal_x.to(at::kByte);

  //
  // combine the branches
  //
  x = at::full_like(x, max_int, options);
  x = at::where(denormal_mask, denormal_x, x);
  x = at::where(normal_mask, normal_x, x);

  // add sign back
  at::Tensor sign_lp =
      sign.bitwise_right_shift(MBITS_F32 + EBITS_F32 - mbits - ebits);
  sign_lp = sign_lp.to(at::kByte);
  // Right shift of a negative signed integer can fill the least significant
  // bits with either 1s or 0s, depending on the implementation. Since PyTorch
  // doesn't have an uint32 dtype, we mask out these bits to get just the
  // f4 sign bit
  sign_lp = sign_lp.bitwise_and(sign_mask);
  x = x.bitwise_or(sign_lp);

  return x.to(at::kByte);
}

at::Tensor toFp4(at::Tensor x) {
  x = unpackAnyFloatingPointToFp32(x.to(at::kFloat), /*ebits=*/2, /*mbits=*/1);
  x = packUint4(x);
  x = x.view(at::kFloat4_e2m1fn_x2);
  return x;
}

std::pair<at::Tensor, at::Tensor> pytorchNvfp4Quantize(
    const at::Tensor a,
    const at::Tensor a_global_scale) {
  constexpr double FLOAT8_E4M3_EPS = 0.125;
  constexpr double FLOAT8_E4M3_MAX = 0.015625;
  constexpr double FLOAT4_E2M1_MAX = 6.0;
  constexpr int64_t BLOCK_SIZE = 16;
  NVF_ERROR(
      a.size(-1) % BLOCK_SIZE == 0,
      "The inner-most dim must be divisible by block_size; Padding is not "
      "implemented.");
  NVF_ERROR(a.is_contiguous(), "Only contiguous tensors are supported.");

  const auto& original_shape = a.sizes();
  const auto a_fp32 =
      a.to(at::kFloat).reshape({original_shape[0], -1, BLOCK_SIZE});

  // Find absolute maximum along blockwise dimension
  const auto max_abs = a_fp32.abs().amax(/*dim=*/-1);
  const auto block_scale_fp32 = (max_abs / FLOAT4_E2M1_MAX).to(at::kFloat);

  const auto scaled_block_scale_fp32 = block_scale_fp32 * a_global_scale;
  const auto scaled_block_scale_fp8 = at::clamp(
                                          scaled_block_scale_fp32,
                                          /*min=*/FLOAT8_E4M3_EPS,
                                          /*max=*/FLOAT8_E4M3_MAX)
                                          .to(at::kFloat8_e4m3fn);
  const auto scaled_block_scale_fp8_fp32 =
      scaled_block_scale_fp8.to(at::kFloat);
  const auto total_scale = scaled_block_scale_fp8_fp32 / a_global_scale;
  auto a_scaled = a_fp32 / total_scale.unsqueeze(-1);
  a_scaled = at::clamp(a_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX);
  a_scaled = a_scaled.view(original_shape);
  return {toFp4(a_scaled), scaled_block_scale_fp8};
}

QuantizedTensor quantizeNvfp4(const at::Tensor x) {
  constexpr double FLOAT8_E4M3_MAX = 0.015625;
  constexpr double FLOAT4_E2M1_MAX = 6.0;

  auto x_global_scale =
      ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max()).to(at::kFloat);

  auto [x_u8, x_scale] = pytorchNvfp4Quantize(x, x_global_scale);
  return {x_u8, x_scale, x_global_scale};
}

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(CutlassExecutorTest, Nvfp4ScaledGemm_Executor) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  if (!std::getenv("CUTLASS_PATH")) {
    GTEST_SKIP() << "The CUTLASS_PATH environment variable must be set in "
                 << "order to run this test";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
  TensorView* b = makeContigTensor(2, DataType::Float4_e2m1fn);
  // B has K inner
  b->setAllocationDomain({b->axis(1), b->axis(0)}, /*new_contiguity=*/true);
  TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* b_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* alpha = makeContigTensor(0, DataType::Float);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(alpha);

  // TODO: support more output dtypes, specifically nvfp4
  auto smm = scaled_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
      /*bias=*/nullptr,
      /*beta=*/nullptr,
      /*dtype=*/DataType::BFloat16);

  fusion->addOutput(smm.tv);

  // Note that K is the actual problem size independent of data type, not the
  // packed size.
  constexpr int64_t M = 8192, N = 8192, K = 8192;

  // Create actual tensor data for inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  QuantizedTensor qa = quantizeNvfp4(at::randn({M, K}, options));
  QuantizedTensor qb = quantizeNvfp4(at::randn({N, K}, options));

  at::Tensor at_a = qa.elts;
  at::Tensor at_b = qb.elts.t();

  at::Tensor at_a_sf = qa.block_scale;
  at::Tensor at_b_sf = qb.block_scale;

  // Create scalar tensors
  at::Tensor at_alpha = 1.0 / (qa.global_scale * qb.global_scale);

  std::vector<c10::IValue> inputs{at_a, at_b, at_a_sf, at_b_sf, at_alpha};

  CutlassParams params;

  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  testValidate(fusion.get(), outputs, inputs, __LINE__, __FILE__);
}

TEST_F(CutlassExecutorTest, Nvfp4MatmulReLU) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10 ||
      at::cuda::getCurrentDeviceProperties()->major > 11) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  if (!std::getenv("CUTLASS_PATH")) {
    GTEST_SKIP() << "The CUTLASS_PATH environment variable must be set in "
                 << "order to run this test";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
  TensorView* b = makeContigTensor(2, DataType::Float4_e2m1fn);
  // B has K inner
  b->setAllocationDomain({b->axis(1), b->axis(0)}, /*new_contiguity=*/true);
  TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* b_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* alpha = makeContigTensor(0, DataType::Float);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(alpha);

  // TODO: support more output dtypes, specifically nvfp4
  auto smm = scaled_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
      /*bias=*/nullptr,
      /*beta=*/nullptr,
      /*dtype=*/DataType::BFloat16);

  TensorView* out_tv = relu(smm.tv);

  fusion->addOutput(out_tv);

  // Note that K is the actual problem size independent of data type, not the
  // packed size.
  constexpr int64_t M = 8192, N = 8192, K = 8192;

  // Create actual tensor data for inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  QuantizedTensor qa = quantizeNvfp4(at::randn({M, K}, options));
  QuantizedTensor qb = quantizeNvfp4(at::randn({N, K}, options));

  at::Tensor at_a = qa.elts;
  at::Tensor at_b = qb.elts.t();

  at::Tensor at_a_sf = qa.block_scale;
  at::Tensor at_b_sf = qb.block_scale;

  // Create scalar tensors
  at::Tensor at_alpha = 1.0 / (qa.global_scale * qb.global_scale);

  std::vector<c10::IValue> inputs{at_a, at_b, at_a_sf, at_b_sf, at_alpha};

  CutlassParams params;

  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  testValidate(fusion.get(), outputs, inputs, __LINE__, __FILE__);
}

// Test findBlockScaledOutputs pattern matching
TEST_F(CutlassExecutorTest, FindBlockScaledOutputs_WithoutGlobalScale) {
  constexpr int64_t block_size = 16;
  constexpr double F4_E2M1_MAX = 6.0;
  constexpr double E4M3_EPS = 0.015625;
  constexpr double F8E4M3_MAX = 448.0;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});
  auto tv_block_scale = div(
      tv_data_hp_amax, IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));
  auto tv_block_scale_clamp = clamp(
      tv_block_scale,
      IrBuilder::create<Val>(E4M3_EPS, DataType::Float),
      IrBuilder::create<Val>(F8E4M3_MAX, DataType::Float));
  auto tv_block_scale_fp8 =
      castOp(DataType::Float8_e4m3fn, tv_block_scale_clamp);
  auto tv_block_scale_fp32 = castOp(DataType::Float, tv_block_scale_fp8);
  auto tv_block_scale_fp32_unsqueeze = unsqueeze(tv_block_scale_fp32, -1);
  auto tv_data_scaled = div(tv_data_hp_reshaped, tv_block_scale_fp32_unsqueeze);
  auto tv_data_scaled_clamp = clamp(
      tv_data_scaled,
      IrBuilder::create<Val>(-F4_E2M1_MAX, DataType::Float),
      IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));

  auto tv_data_lp_fp4 = castOp(DataType::Float4_e2m1fn, tv_data_scaled_clamp);
  auto tv_data_lp = reshape(tv_data_lp_fp4, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_block_scale_fp8);
  fusion->addOutput(tv_data_lp);

  fusion->printMath();

  // Test pattern matching
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].output, tv_data_lp);
  EXPECT_EQ(patterns[0].prescaled_output, tv_data_scaled_clamp);
  EXPECT_EQ(patterns[0].block_scale_factors, tv_block_scale_fp8);
  EXPECT_EQ(patterns[0].global_scale_factor, nullptr);
  EXPECT_EQ(patterns[0].block_size, block_size);
}

TEST_F(CutlassExecutorTest, FindBlockScaledOutputs_WithGlobalScale) {
  constexpr int64_t block_size = 16;
  constexpr double F4_E2M1_MAX = 6.0;
  constexpr double E4M3_EPS = 0.015625;
  constexpr double F8E4M3_MAX = 448.0;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  auto tv_per_tensor_scale = makeContigTensor(0, DataType::Float);
  fusion->addInput(tv_data_hp);
  fusion->addInput(tv_per_tensor_scale);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});
  auto tv_block_scale = div(
      tv_data_hp_amax, IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));

  auto tv_scaled_block_scales = div(tv_block_scale, tv_per_tensor_scale);
  auto tv_scaled_block_scales_clamp = clamp(
      tv_scaled_block_scales,
      IrBuilder::create<Val>(E4M3_EPS, DataType::Float),
      IrBuilder::create<Val>(F8E4M3_MAX, DataType::Float));
  auto tv_scaled_block_scales_fp8 =
      castOp(DataType::Float8_e4m3fn, tv_scaled_block_scales_clamp);
  auto tv_scaled_block_scales_fp32 =
      castOp(DataType::Float, tv_scaled_block_scales_fp8);

  auto tv_total_scale = mul(tv_per_tensor_scale, tv_scaled_block_scales_fp32);
  auto tv_total_scale_unsqueeze = unsqueeze(tv_total_scale, -1);
  auto tv_data_scaled = div(tv_data_hp_reshaped, tv_total_scale_unsqueeze);

  auto tv_data_scaled_clamp = clamp(
      tv_data_scaled,
      IrBuilder::create<Val>(-F4_E2M1_MAX, DataType::Float),
      IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));

  auto tv_data_lp_fp4 = castOp(DataType::Float4_e2m1fn, tv_data_scaled_clamp);
  auto tv_data_lp = reshape(tv_data_lp_fp4, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_scaled_block_scales_fp8);
  fusion->addOutput(tv_data_lp);

  // Test pattern matching
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].output, tv_data_lp);
  EXPECT_EQ(patterns[0].prescaled_output, tv_data_scaled_clamp);
  EXPECT_EQ(patterns[0].block_scale_factors, tv_scaled_block_scales_fp8);
  EXPECT_EQ(patterns[0].global_scale_factor, tv_per_tensor_scale);
  EXPECT_EQ(patterns[0].block_size, block_size);
}

TEST_F(CutlassExecutorTest, FindBlockScaledOutputs_MXFP8) {
  constexpr int64_t block_size = 32;
  constexpr double FP8_MAX = 448.0;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});
  auto tv_block_scale =
      div(tv_data_hp_amax, IrBuilder::create<Val>(FP8_MAX, DataType::Float));
  auto tv_block_scale_fp8 = castOp(DataType::Float8_e4m3fn, tv_block_scale);
  auto tv_block_scale_fp32 = castOp(DataType::Float, tv_block_scale_fp8);
  auto tv_block_scale_fp32_unsqueeze = unsqueeze(tv_block_scale_fp32, -1);
  auto tv_data_scaled = div(tv_data_hp_reshaped, tv_block_scale_fp32_unsqueeze);

  auto tv_data_lp_fp8 = castOp(DataType::Float8_e4m3fn, tv_data_scaled);
  auto tv_data_lp = reshape(tv_data_lp_fp8, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_block_scale_fp8);
  fusion->addOutput(tv_data_lp);

  // Test pattern matching for FP8 output (MXFP8)
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].output, tv_data_lp);
  EXPECT_EQ(patterns[0].prescaled_output, tv_data_scaled);
  EXPECT_EQ(patterns[0].block_scale_factors, tv_block_scale_fp8);
  EXPECT_EQ(patterns[0].global_scale_factor, nullptr);
  EXPECT_EQ(patterns[0].block_size, block_size);
}

} // namespace nvfuser
