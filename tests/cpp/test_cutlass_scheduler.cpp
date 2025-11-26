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
#include <nvf_cutlass.h>
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

class CutlassExecutorTest : public NVFuserTest {
 public:
  void SetUp() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::CutlassScheduler);
  }

 private:
  EnableOptionsGuard eog_;
};

struct QuantizedTensorView {
  TensorView* elts;
  TensorView* block_scale;
};

QuantizedTensorView quantizeTv(
    TensorView* unquantized,
    TensorView* global_scale_factor,
    int64_t block_size,
    DataType quantized_dtype,
    DataType block_scale_dtype,
    double quantized_max_norm,
    double block_scale_min,
    double block_scale_max,
    bool clamp_before_casts) {
  // Upcast to Float first if necessary
  unquantized = maybeCastOp(
      promoteType(DataType::Float, unquantized->dtype()), unquantized);

  TensorView* reshaped =
      reshape(unquantized, [block_size](auto& x) { x.split(-1, block_size); });

  TensorView* mag = abs(reshaped);
  TensorView* block_max = max(mag, {-1});
  TensorView* block_sf = div(
      block_max, IrBuilder::create<Val>(quantized_max_norm, DataType::Float));

  if (global_scale_factor != nullptr) {
    block_sf = div(block_sf, global_scale_factor);
  }
  if (clamp_before_casts) {
    block_sf = clamp(
        block_sf,
        IrBuilder::create<Val>(block_scale_min, DataType::Float),
        IrBuilder::create<Val>(block_scale_max, DataType::Float));
  }
  TensorView* block_sf_cast = castOp(DataType::Float8_e4m3fn, block_sf);
  TensorView* block_sf_fp32 = castOp(DataType::Float, block_sf_cast);

  if (global_scale_factor != nullptr) {
    block_sf_fp32 = mul(global_scale_factor, block_sf_fp32);
  }
  TensorView* unsqueezed = unsqueeze(block_sf_fp32, -1);
  TensorView* scaled = div(reshaped, unsqueezed);

  if (clamp_before_casts) {
    scaled = clamp(
        scaled,
        IrBuilder::create<Val>(-quantized_max_norm, DataType::Float),
        IrBuilder::create<Val>(quantized_max_norm, DataType::Float));
  }

  TensorView* casted = castOp(DataType::Float4_e2m1fn, scaled);
  TensorView* quantized = reshape(casted, [](auto& x) { x.merge(-2); });

  return {.elts = quantized, .block_scale = block_sf_cast};
}

QuantizedTensorView quantizeTvNvfp4(
    TensorView* unquantized,
    TensorView* global_scale_factor = nullptr) {
  // max = (2 – 2^(–M)) * 2^(2^(E-1))  (no nans for fp4)
  constexpr double F4E2M1_MAX = 6.0;
  // eps = 2^(1-M-E)
  constexpr double F8E4M3_EPS = 0.015625;
  constexpr double F8E4M3_MAX = 448.0;
  return quantizeTv(
      unquantized,
      global_scale_factor,
      /*block_size=*/16,
      /*quantized_dtype=*/DataType::Float4_e2m1fn,
      /*block_scale_factor_dtype=*/DataType::Float8_e4m3fn,
      /*quantized_max_norm=*/F4E2M1_MAX,
      /*block_scale_min=*/F8E4M3_EPS,
      /*block_scale_max=*/F8E4M3_MAX,
      /*clamp_before_casts=*/true);
}

QuantizedTensorView quantizeTvMxfp8(
    TensorView* unquantized,
    TensorView* global_scale_factor = nullptr) {
  // eps = 2^(1-M-E)
  constexpr double F8E4M3_EPS = 0.015625;
  // https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
  constexpr double F8E4M3_MAX = 448.0;
  return quantizeTv(
      unquantized,
      global_scale_factor,
      /*block_size=*/32,
      /*quantized_dtype=*/DataType::Float4_e2m1fn,
      /*block_scale_factor_dtype=*/DataType::Float8_e4m3fn,
      /*quantized_max_norm=*/F8E4M3_MAX,
      /*block_scale_min=*/F8E4M3_EPS,
      /*block_scale_max=*/F8E4M3_MAX,
      /*clamp_before_casts=*/false);
}

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
  std::vector<int64_t> new_shape = original_shape.vec();
  new_shape.back() = original_shape.back() / BLOCK_SIZE;
  new_shape.push_back(BLOCK_SIZE);

  const auto a_fp32 = a.to(at::kFloat).reshape(new_shape);

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

//! Convert FP4 into FP32
at::Tensor e2m1ToFp32(const at::Tensor& int4_value) {
  NVF_ERROR_EQ(int4_value.dtype(), at::kByte);
  const at::Tensor signBit = int4_value & 0b1000;
  const at::Tensor index = int4_value & 0b0111;

  // Map the 8 possible non-negative values of e2m1 to corresponding positive
  // fp32 value
  const at::Tensor kE2M1ToFloatArray = at::tensor(
      {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f},
      int4_value.options().dtype(at::kFloat));

  const at::Tensor abs_float_result =
      at::take(kE2M1ToFloatArray, index.to(at::kLong));
  return at::where(signBit != 0, -abs_float_result, abs_float_result);
}

//! Unpack float4_e2m1fn_x2 into two separate fp32 values
at::Tensor unpackFp4Bytes(const at::Tensor& a) {
  NVF_ERROR_EQ(a.dtype(), at::kFloat4_e2m1fn_x2);
  int64_t m = a.size(0);
  int64_t n = a.size(1);
  const at::Tensor a_byte = a.view(at::kByte);
  const at::Tensor a_flat = a_byte.flatten();
  const at::Tensor upper_half_byte =
      at::bitwise_right_shift(a_byte & 0b11110000, 4);
  const at::Tensor lower_half_byte = a_byte & 0b00001111;
  const at::Tensor upper_half_float = e2m1ToFp32(upper_half_byte);
  const at::Tensor lower_half_float = e2m1ToFp32(lower_half_byte);
  return at::stack({lower_half_float, upper_half_float}, /*dim=*/-1)
      .reshape({m, n * 2});
}

// apply swizzled on block scaling factor:
// 1. apply padding to [mn_t * 128 , k_t * 4]
// 2. apply swizzle
at::Tensor linearToSwizzled128by4(const at::Tensor& a_sf_linear) {
  NVF_ERROR_EQ(a_sf_linear.dim(), 2);
  int64_t mn = a_sf_linear.size(0);
  int64_t sf_k = a_sf_linear.size(1);
  int64_t m_tiles = ceilDiv(mn, 128);
  int64_t mn_padded = m_tiles * 128;
  int64_t k_tiles = ceilDiv(sf_k, 4);
  int64_t k_padded = k_tiles * 4;
  at::Tensor a_sf_padded;
  if (mn_padded != mn || k_padded != sf_k) {
    a_sf_padded = at::empty({mn_padded, k_padded}, a_sf_linear.options());
    a_sf_padded.slice(0, 0, mn).slice(1, 0, sf_k) = a_sf_linear;
  } else {
    a_sf_padded = a_sf_linear;
  }
  at::Tensor tmp = at::reshape(a_sf_padded, {m_tiles, 4, 32, k_tiles, 4});
  return tmp.transpose(1, 3).reshape({mn_padded, k_padded});
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

  EXPECT_TRUE(SchedulerEntry::makeSchedulerInstance(SchedulerType::Cutlass)
                  ->canScheduleCompileTime(fusion.get()));

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

  EXPECT_TRUE(SchedulerEntry::makeSchedulerInstance(SchedulerType::Cutlass)
                  ->canScheduleCompileTime(fusion.get()));

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
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto input = makeContigTensor(2, DataType::Float);
  fusion->addInput(input);

  auto unquantized_output = exp(input);

  const QuantizedTensorView qtv =
      quantizeTvNvfp4(unquantized_output, /*global_scale_factor=*/nullptr);

  fusion->addOutput(qtv.block_scale);
  fusion->addOutput(qtv.elts);

  // Test pattern matching
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].quantized_output, qtv.elts);
  EXPECT_EQ(patterns[0].unquantized_output, unquantized_output);
  EXPECT_EQ(patterns[0].block_scale_factors, qtv.block_scale);
  EXPECT_EQ(patterns[0].global_scale_factor, nullptr);
  EXPECT_EQ(patterns[0].block_size, 16);
}

TEST_F(CutlassExecutorTest, FindBlockScaledOutputs_WithGlobalScale) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto input = makeContigTensor(2, DataType::Float);
  auto per_tensor_scale = makeContigTensor(0, DataType::Float);
  fusion->addInput(input);
  fusion->addInput(per_tensor_scale);

  TensorView* unquantized_output = exp(input);

  const QuantizedTensorView qtv =
      quantizeTvNvfp4(unquantized_output, per_tensor_scale);

  fusion->addOutput(qtv.block_scale);
  fusion->addOutput(qtv.elts);

  // Test pattern matching
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].quantized_output, qtv.elts);
  EXPECT_EQ(patterns[0].unquantized_output, unquantized_output);
  EXPECT_EQ(patterns[0].block_scale_factors, qtv.block_scale);
  EXPECT_EQ(patterns[0].global_scale_factor, per_tensor_scale);
  EXPECT_EQ(patterns[0].block_size, 16);
}

TEST_F(CutlassExecutorTest, FindBlockScaledOutputs_MXFP8) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto input = makeContigTensor(2, DataType::Float);
  fusion->addInput(input);

  TensorView* unquantized_output = exp(input);

  const QuantizedTensorView qtv =
      quantizeTvMxfp8(unquantized_output, /*global_scale_factor=*/nullptr);

  fusion->addOutput(qtv.block_scale);
  fusion->addOutput(qtv.elts);

  // Test pattern matching for FP8 output (MXFP8)
  auto patterns = cutlass_codegen::findBlockScaledOutputs(fusion.get());

  ASSERT_EQ(patterns.size(), 1);
  EXPECT_EQ(patterns[0].quantized_output, qtv.elts);
  EXPECT_EQ(patterns[0].unquantized_output, unquantized_output);
  EXPECT_EQ(patterns[0].block_scale_factors, qtv.block_scale);
  EXPECT_EQ(patterns[0].global_scale_factor, nullptr);
  EXPECT_EQ(patterns[0].block_size, 32);
}

// Test GEMM + ReLU with nvfp4 block-scaled inputs and output
TEST_F(CutlassExecutorTest, Nvfp4BlockScaledGemmReLU) {
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

  // Input tensors: nvfp4 block-scaled A and B matrices
  TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
  TensorView* b = makeContigTensor(2, DataType::Float4_e2m1fn);
  // B has K inner for optimal memory access
  b->setAllocationDomain({b->axis(1), b->axis(0)}, /*new_contiguity=*/true);
  TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* b_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* alpha = makeContigTensor(0, DataType::Float);
  TensorView* global_normconst = makeContigTensor(0, DataType::Float);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(alpha);
  fusion->addInput(global_normconst);

  // Perform block-scaled matmul
  auto smm = scaled_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
      /*bias=*/nullptr,
      /*beta=*/nullptr,
      /*dtype=*/DataType::Float);

  const QuantizedTensorView qtv = quantizeTvNvfp4(smm.tv, global_normconst);

  fusion->addOutput(qtv.block_scale);
  fusion->addOutput(qtv.elts);

  EXPECT_TRUE(SchedulerEntry::makeSchedulerInstance(SchedulerType::Cutlass)
                  ->canScheduleCompileTime(fusion.get()));

  // Test dimensions
  constexpr int64_t M = 4096, N = 4096, K = 4096;

  // Create test data
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  QuantizedTensor qa = quantizeNvfp4(at::randn({M, K}, options));
  QuantizedTensor qb = quantizeNvfp4(at::randn({N, K}, options));

  at::Tensor at_a = qa.elts;
  at::Tensor at_b = qb.elts.t();
  at::Tensor at_a_sf = qa.block_scale;
  at::Tensor at_b_sf = qb.block_scale;

  // Compute alpha to combine global scales
  at::Tensor at_alpha = 1.0 / (qa.global_scale * qb.global_scale);
  at::Tensor at_global_normconst = at::full({}, 2.0f, options);

  std::vector<c10::IValue> inputs{
      at_a, at_b, at_a_sf, at_b_sf, at_alpha, at_global_normconst};

  // Compile and run
  CutlassParams params;
  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  EXPECT_EQ(outputs.size(), 2);

#if NVFUSER_ENABLE_CUTLASS
  std::pair<torch::Tensor, torch::Tensor> aot_result =
      cutlass_kernels::nvfp4_scaled_mm_blockscale(
          at_a, at_b.t(), at_a_sf, at_b_sf, at_alpha, at_global_normconst);

  EXPECT_TRUE(at::allclose(
      outputs[0].as<at::Tensor>().to(at::kFloat),
      aot_result.second.to(at::kFloat),
      /*rtol=*/0.001,
      /*atol=*/0.001));
  EXPECT_TRUE(at::allclose(
      unpackFp4Bytes(outputs[1].as<at::Tensor>()),
      unpackFp4Bytes(aot_result.first),
      /*rtol=*/0.001,
      /*atol=*/0.001));
#endif
}

TEST_F(CutlassExecutorTest, Nvfp4Matmul_BiasEpilogue) {
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
  TensorView* beta = makeContigTensor(0, DataType::Float);
  TensorView* bias = makeContigTensor(2, DataType::BFloat16);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(bias);
  fusion->addInput(alpha);
  fusion->addInput(beta);

  auto smm = scaled_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
      bias,
      beta,
      // NOTE: We support DataType::Float output in the Cutlass executor, but
      // we use BFloat16 here in order to compare fairly against our
      // precompiled kernels which only support Half and BFloat16
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

  at::Tensor at_bias = at::randn({M, N}, options.dtype(at::kBFloat16));

  // Create scalar tensors
  at::Tensor at_alpha = 1.0 / (qa.global_scale * qb.global_scale);
  at::Tensor at_beta = at::randn({}, options);

  std::vector<c10::IValue> inputs{
      at_a, at_b, at_a_sf, at_b_sf, at_bias, at_alpha, at_beta};

  CutlassParams params;

  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  // The pre-compiled kernels do not support beta or bias, so instead, I'll
  // create a separate fusion that computes those separately in order to create
  // a reference.
  {
    auto ref_fusion = std::make_unique<Fusion>();
    FusionGuard fg(ref_fusion.get());

    TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
    TensorView* b = makeContigTensor(2, DataType::Float4_e2m1fn);
    // B has K inner
    b->setAllocationDomain({b->axis(1), b->axis(0)}, /*new_contiguity=*/true);
    TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
    TensorView* b_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
    TensorView* alpha = makeContigTensor(0, DataType::Float);
    TensorView* beta = makeContigTensor(0, DataType::Float);
    TensorView* bias = makeContigTensor(2, DataType::BFloat16);

    ref_fusion->addInput(a);
    ref_fusion->addInput(b);
    ref_fusion->addInput(a_sf);
    ref_fusion->addInput(b_sf);
    ref_fusion->addInput(bias);
    ref_fusion->addInput(alpha);
    ref_fusion->addInput(beta);

    auto smm = scaled_mm(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        /*bias=*/nullptr,
        /*beta=*/nullptr,
        /*dtype=*/DataType::BFloat16);

    TensorView* beta_bias = mul(beta, bias);
    TensorView* lincomb = add(smm.tv, beta_bias);

    TensorView* out_tv = relu(lincomb);

    ref_fusion->addOutput(out_tv);

    ExpressionEvaluator expr_eval;

    ASSERT_EQ(ref_fusion->inputs().size(), inputs.size());
    for (auto [val, at_tens] : zip(ref_fusion->inputs(), inputs)) {
      expr_eval.bind(val, at_tens.toTensor());
    }

    PolymorphicValue out_pv = expr_eval.evaluate(out_tv);

    ASSERT_TRUE(out_pv.is<at::Tensor>());

    testValidate(
        fusion.get(),
        outputs,
        inputs,
        {out_pv.as<at::Tensor>()},
        __LINE__,
        __FILE__);
  }
}

// Test Grouped GEMM + ReLU with nvfp4 block-scaled inputs and output
TEST_F(CutlassExecutorTest, Nvfp4BlockScaledGroupedGemmReLU) {
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

  // Input tensors: nvfp4 block-scaled A and B matrices
  TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
  TensorView* b = makeContigTensor(3, DataType::Float4_e2m1fn);
  // B has K inner and is of logical shape [E, K, N]
  b->setAllocationDomain(
      {b->axis(0), b->axis(2), b->axis(1)}, /*new_contiguity=*/true);
  TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* b_sf = makeContigTensor(3, DataType::Float8_e4m3fn);
  TensorView* alpha = makeContigTensor(1, DataType::Float);
  TensorView* problem_sizes = makeContigTensor(2, DataType::Int32);
  TensorView* expert_offsets = makeContigTensor(1, DataType::Int32);
  TensorView* sf_offsets = makeContigTensor(1, DataType::Int32);

  // TensorView* global_normconst = makeContigTensor(0, DataType::Float);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(alpha);
  fusion->addInput(problem_sizes);
  fusion->addInput(expert_offsets);
  fusion->addInput(sf_offsets);
  // fusion->addInput(global_normconst);

  // Perform block-scaled matmul
  TensorView* gmmtv = cutlass_nvfp4_grouped_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
      problem_sizes,
      expert_offsets,
      sf_offsets,
      /*dtype=*/DataType::BFloat16);

  TensorView* unquantized_output = relu(gmmtv);
  fusion->addOutput(unquantized_output);

  /*
  const QuantizedTensorView qtv =
      quantizeTvNvfp4(unquantized_output, global_normconst);

  fusion->addOutput(qtv.block_scale);
  fusion->addOutput(qtv.elts);
  */

  EXPECT_TRUE(SchedulerEntry::makeSchedulerInstance(SchedulerType::Cutlass)
                  ->canScheduleCompileTime(fusion.get()));

  const std::vector<int64_t> tokens_per_expert{115, 144, 8, 757};

  int64_t num_experts = tokens_per_expert.size();
  constexpr int64_t N = 128, K = 256;

  // Create test data
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_offsets = at::empty({num_experts}, options.dtype(at::kInt));
  at::Tensor at_sf_offsets = at::empty({num_experts}, options.dtype(at::kInt));
  at::Tensor at_problem_sizes =
      at::empty({num_experts, 3}, options.dtype(at::kInt));
  int32_t accumulated_tokens = 0;
  int32_t rounded_accumulated_tokens = 0;

  std::vector<at::Tensor> as, bs, a_sfs, b_sfs, alphas;
  as.reserve(num_experts);
  bs.reserve(num_experts);
  a_sfs.reserve(num_experts);
  b_sfs.reserve(num_experts);
  alphas.reserve(num_experts);
  for (auto [i, Mi] : enumerate(tokens_per_expert)) {
    // Record size of this subproblem Mi, N, K
    at_problem_sizes.index({(int64_t)i, 0}).fill_(Mi);
    at_problem_sizes.index({(int64_t)i, 1}).fill_(N);
    at_problem_sizes.index({(int64_t)i, 2}).fill_(K);

    // Generate A and B with for subproblem
    const QuantizedTensor qa = quantizeNvfp4(at::randn({Mi, K}, options));
    const QuantizedTensor qb = quantizeNvfp4(at::randn({N, K}, options));

    // For the A operand we need to swizzle and pad the scale factor
    at::Tensor a_sf = linearToSwizzled128by4(qa.block_scale);

    as.push_back(qa.elts);
    bs.push_back(qb.elts);
    a_sfs.push_back(a_sf);
    b_sfs.push_back(qb.block_scale);
    alphas.push_back(1.0 / (qa.global_scale * qb.global_scale));

    at_offsets[i] = accumulated_tokens;
    at_sf_offsets[i] = rounded_accumulated_tokens;

    accumulated_tokens += Mi;
    rounded_accumulated_tokens += roundUpToMultiple(Mi, 128);
  }

  at::Tensor at_a = at::concatenate(as, /*dim=*/0);
  at::Tensor at_b = at::stack(bs, /*dim=*/0).permute({0, 2, 1});
  at::Tensor at_a_sf = at::concatenate(a_sfs, /*dim=*/0);
  at::Tensor at_b_sf = at::stack(b_sfs, /*dim=*/0);
  at::Tensor at_alpha = at::stack(alphas, /*dim=*/0);

  // at::Tensor at_global_normconst = at::full({}, 2.0f, options);

  std::vector<c10::IValue> inputs{
      at_a,
      at_b,
      at_a_sf,
      at_b_sf,
      at_alpha,
      at_problem_sizes,
      at_offsets,
      at_sf_offsets
      // at_global_normconst
  };

  // Compile and run
  CutlassParams params;

  // We cannot use 2SM for grouped GEMM currently
  params.mma_tile = {128, 128, 128};
  params.per_sm_tile = params.mma_tile;
  params.cluster_shape = {1, 1, 1};

  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  testValidate(fusion.get(), outputs, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
