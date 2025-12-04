// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Testing fusions used in low precision recipes

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {
using testing::UnorderedElementsAre;

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

class NVFP4QuantizeTest : public BlackwellBase,
                          public ::testing::WithParamInterface<DataType> {};
namespace {

void createMXFP8QuantizationFusion(Fusion* fusion, DataType data_hp_dtype) {
  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion->addInput(tv_data_hp);

  constexpr int64_t fp8_block_size = 32;
  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, fp8_block_size); });

  tv_data_hp_reshaped = castOp(DataType::Float, tv_data_hp_reshaped);
  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});

  static constexpr float max_norm_rcp = 1.0f / 448;
  auto tv_block_scale = mul(
      tv_data_hp_amax, IrBuilder::create<Val>(max_norm_rcp, DataType::Float));

  auto tv_block_scale_fp8 = castOp(DataType::Float8_e8m0fnu, tv_block_scale);

  auto tv_unsqueeze = unsqueeze(tv_block_scale_fp8, -1);
  auto exponent_scale = reciprocal(exp2(tv_unsqueeze));
  auto tv_data_scaled = mul(tv_data_hp_reshaped, exponent_scale);

  auto tv_data_lp = castOp(DataType::Float8_e4m3fn, tv_data_scaled);
  tv_data_lp = reshape(tv_data_lp, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_data_lp);
  fusion->addOutput(tv_block_scale_fp8);
}

void createNVFP4QuantizationFusion(
    Fusion* fusion,
    DataType data_hp_dtype,
    bool use_global_scale = false,
    bool swizzle_block_scales = false) {
  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion->addInput(tv_data_hp);

  auto tv_global_scale =
      use_global_scale ? makeContigTensor(0, DataType::Float) : nullptr;
  if (use_global_scale) {
    fusion->addInput(tv_global_scale);
  }

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  // cast it to FP32
  tv_data_hp_reshaped = castOp(DataType::Float, tv_data_hp_reshaped);

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});

  constexpr float rcp_6f = 1.0f / 6.0f;
  auto tv_block_scale = tv_data_hp_amax;
  if (use_global_scale) {
    tv_block_scale = mul(tv_block_scale, tv_global_scale);
  }
  tv_block_scale =
      mul(tv_block_scale, IrBuilder::create<Val>(rcp_6f, DataType::Float));

  auto tv_block_scale_fp8 = castOp(DataType::Float8_e4m3fn, tv_block_scale);

  auto tv_block_scale_fp32 = castOp(DataType::Float, tv_block_scale_fp8);
  if (use_global_scale) {
    tv_block_scale_fp32 = div(tv_global_scale, tv_block_scale_fp32);
  }

  auto tv_block_scale_fp32_unsqueeze = unsqueeze(tv_block_scale_fp32, -1);
  auto tv_data_scaled = mul(tv_data_hp_reshaped, tv_block_scale_fp32_unsqueeze);

  auto tv_data_lp_fp4 = castOp(DataType::Float4_e2m1fn, tv_data_scaled);
  auto tv_data_lp = reshape(tv_data_lp_fp4, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_block_scale_fp8);
  fusion->addOutput(tv_data_lp);

  if (swizzle_block_scales) {
    nvfuser::ir_utils::swizzleBlockScales(tv_block_scale_fp8);
  }
}
} // namespace

class MXFP8QuantizationTest : public BlackwellBase {};

TEST_F(MXFP8QuantizationTest, Basic) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  createMXFP8QuantizationFusion(fusion.get(), DataType::Float);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  inputs.push_back(
      at::randn({1024, 1024}, at::device(at::kCUDA).dtype(at::kFloat)));
  auto outputs = fec.runFusionWithInputs(inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();

  // Check that the fusion is segmented into two groups.
  // The normalization scheduler is used for the first group
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::InnerPersistent)));
}

TEST_P(NVFP4QuantizeTest, WithoutPerTensorAmax) {
  auto data_hp_dtype = GetParam();

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  createNVFP4QuantizationFusion(fusion.get(), data_hp_dtype);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  inputs.push_back(
      at::randn({1024, 1024}, at::device(at::kCUDA).dtype(at::kFloat))
          .to(data_type_to_aten(data_hp_dtype)));
  auto outputs = fec.runFusionWithInputs(inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();

  // Check that the fusion is segmented into two groups.
  // The normalization scheduler is used for the first group
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::InnerPersistent)));
}

class BlockQuantizationTest
    : public BlackwellBase,
      public ::testing::WithParamInterface<std::tuple<DataType, int64_t>> {};

TEST_P(BlockQuantizationTest, ScheduleAsPointwise) {
  auto data_hp_dtype = std::get<0>(GetParam());
  auto group_width = std::get<1>(GetParam());

  // Baseline implementation
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QuantizationFusion(fusion.get(), data_hp_dtype);

  FusionExecutorCache fec(std::move(fusion));

  const int m = 1024;
  const int n = 1024;
  std::vector<at::Tensor> inputs;
  inputs.push_back(at::randn({m, n}, at::device(at::kCUDA).dtype(at::kFloat))
                       .to(data_type_to_aten(data_hp_dtype)));
  auto outputs_baseline = fec.runFusionWithInputs(inputs);

  auto baseline_block_scales = outputs_baseline[0].as<at::Tensor>();
  auto baseline_quantized_tensor = outputs_baseline[1].as<at::Tensor>();

  // Move baseline tensors from GPU to CPU
  auto baseline_block_scales_cpu = baseline_block_scales.cpu();
  auto baseline_quantized_tensor_cpu = baseline_quantized_tensor.cpu();

  const uint8_t* baseline_block_scales_data =
      static_cast<const uint8_t*>(baseline_block_scales_cpu.data_ptr());
  const uint8_t* baseline_quantized_data =
      static_cast<const uint8_t*>(baseline_quantized_tensor_cpu.data_ptr());

  std::unique_ptr<Fusion> fusion_new_op = std::make_unique<Fusion>();
  FusionGuard fg2(fusion_new_op.get());

  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion_new_op->addInput(tv_data_hp);

  auto t0 = set(tv_data_hp);
  auto quantization_results = blockQuantize(t0);
  auto t_out = set(quantization_results.quantized_tensor);

  fusion_new_op->addOutput(quantization_results.block_scales);
  fusion_new_op->addOutput(t_out);

  auto vectorization_factor = group_width;

  for (auto t :
       {tv_data_hp,
        t0,
        quantization_results.quantized_tensor,
        quantization_results.block_scales,
        t_out}) {
    // Merge all dims.
    t->merge(-2);
    if (t->getLoopDomain().size() >= 2) {
      t->merge(-2);
    }

    // split by 4 (or 2, 8).
    // I -> I/4, 4
    t->split(-1, vectorization_factor);
    // I//4, 4 -> I/4, 1, 4
    t->split(-2, 1);
    // I//4, 1, 4 -> I/512, 128, 1, 4
    t->split(-3, 128);

    if (t != tv_data_hp) {
      if (t == quantization_results.block_scales ||
          t == quantization_results.quantized_tensor) {
        t->axis(-1)->parallelize(ParallelType::Group);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
      }
      t->axis(-3)->parallelize(ParallelType::TIDx);
      t->axis(-4)->parallelize(ParallelType::BIDx);
    }
  }

  // Execute the fusion
  KernelExecutor ke;
  ke.compile(fusion_new_op.get(), inputs);
  auto outputs_new_op = ke.run(inputs);

  // Verify we got the expected outputs
  auto block_scales_output = outputs_new_op[0].as<at::Tensor>();
  auto quantized_tensor_output = outputs_new_op[1].as<at::Tensor>();

  // Move tensors from GPU to CPU
  auto block_scales_cpu = block_scales_output.cpu();
  auto quantized_tensor_cpu = quantized_tensor_output.cpu();

  auto block_scales_bytes = (m * n) / block_size;
  auto quantized_tensor_bytes = (m * n) / 2;

  const uint8_t* block_scales_data =
      static_cast<const uint8_t*>(block_scales_cpu.data_ptr());
  for (int i = 0; i < block_scales_bytes; ++i) {
    EXPECT_EQ(block_scales_data[i], baseline_block_scales_data[i]);
  }

  const uint8_t* quantized_data =
      static_cast<const uint8_t*>(quantized_tensor_cpu.data_ptr());
  for (int i = 0; i < quantized_tensor_bytes; ++i) {
    EXPECT_EQ(quantized_data[i], baseline_quantized_data[i]);
  }

  // Basic shape checks
  EXPECT_EQ(block_scales_output.dim(), 2);
  EXPECT_EQ(quantized_tensor_output.dim(), 2);
}

TEST_P(BlockQuantizationTest, ScheduleAsPointwise2D) {
  auto data_hp_dtype = std::get<0>(GetParam());
  auto group_width = std::get<1>(GetParam());

  // Baseline  implementation
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QuantizationFusion(fusion.get(), data_hp_dtype);

  FusionExecutorCache fec(std::move(fusion));

  const int m = 1024;
  const int n = 1024;
  std::vector<at::Tensor> inputs;
  inputs.push_back(at::randn({m, n}, at::device(at::kCUDA).dtype(at::kFloat))
                       .to(data_type_to_aten(data_hp_dtype)));
  auto outputs_baseline = fec.runFusionWithInputs(inputs);

  // Print baseline outputs
  auto baseline_block_scales = outputs_baseline[0].as<at::Tensor>();
  auto baseline_quantized_tensor = outputs_baseline[1].as<at::Tensor>();

  // Move baseline tensors from GPU to CPU
  auto baseline_block_scales_cpu = baseline_block_scales.cpu();
  auto baseline_quantized_tensor_cpu = baseline_quantized_tensor.cpu();

  const uint8_t* baseline_block_scales_data =
      static_cast<const uint8_t*>(baseline_block_scales_cpu.data_ptr());
  const uint8_t* baseline_quantized_data =
      static_cast<const uint8_t*>(baseline_quantized_tensor_cpu.data_ptr());

  std::unique_ptr<Fusion> fusion_new_op = std::make_unique<Fusion>();
  FusionGuard fg2(fusion_new_op.get());

  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion_new_op->addInput(tv_data_hp);

  // t0 is 2D
  auto t0 = set(tv_data_hp);
  auto quantization_results = blockQuantize(t0);
  auto t_out = set(quantization_results.quantized_tensor);

  // outputs are 3D
  fusion_new_op->addOutput(quantization_results.block_scales);
  fusion_new_op->addOutput(t_out);

  t0->setMemoryType(MemoryType::Local);

  auto vectorization_factor = group_width;

  for (auto t :
       {tv_data_hp,
        t0,
        quantization_results.quantized_tensor,
        quantization_results.block_scales,
        t_out}) {
    // We split by 4 as an example, but can also be 2 or 8(fp16/bf16 on;y)
    // (m, n) -> (m, n/4, 4)
    // (m, n/4, 4) -> (m, n/128, 32, 4)
    t->split(-1, vectorization_factor); // V
    t->split(-2, 32); // BDx

    // (m, n/128, 32, 4) -> (m, 1, n/128, 32, 4)
    // (m, 1, n/128, 32, 4) -> (m/4, 4, 1, n/128, 32, 4)
    t->split(0, 1);
    t->split(0, 4);

    // (m/4(bidy), 4(tidy), 1, n*k/128(bidx), 32(tidx), 4(v))
    if (t != tv_data_hp) {
      if (t == quantization_results.block_scales ||
          t == quantization_results.quantized_tensor) {
        t->axis(-1)->parallelize(ParallelType::Group);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
      }
      t->axis(-2)->parallelize(ParallelType::TIDx);
      t->axis(-3)->parallelize(ParallelType::BIDx);
      t->axis(-5)->parallelize(ParallelType::TIDy);
      t->axis(-6)->parallelize(ParallelType::BIDy);
    }
  }

  // Execute the fusion
  KernelExecutor ke;
  ke.compile(fusion_new_op.get(), inputs);
  auto outputs_new_op = ke.run(inputs);

  // Verify we got the expected outputs
  auto block_scales_output = outputs_new_op[0].as<at::Tensor>();
  auto quantized_tensor_output = outputs_new_op[1].as<at::Tensor>();

  // Move tensors from GPU to CPU
  auto block_scales_cpu = block_scales_output.cpu();
  auto quantized_tensor_cpu = quantized_tensor_output.cpu();

  auto block_scales_bytes = (m * n) / block_size;
  auto quantized_tensor_bytes = (m * n) / 2;

  const uint8_t* block_scales_data =
      static_cast<const uint8_t*>(block_scales_cpu.data_ptr());
  for (int i = 0; i < block_scales_bytes; ++i) {
    EXPECT_EQ(
        block_scales_data[i],
        baseline_block_scales_data[i]); // Compare with baseline
  }

  const uint8_t* quantized_data =
      static_cast<const uint8_t*>(quantized_tensor_cpu.data_ptr());
  for (int i = 0; i < quantized_tensor_bytes; ++i) {
    EXPECT_EQ(
        quantized_data[i],
        baseline_quantized_data[i]); // Compare with baseline
  }

  // Basic shape checks
  EXPECT_EQ(block_scales_output.dim(), 2);
  EXPECT_EQ(quantized_tensor_output.dim(), 2);
}

class BlockQuantizationValidationTest : public BlackwellBase {
 protected:
  struct FusionSetup {
    std::unique_ptr<Fusion> fusion;
    TensorView* tv_data_hp;
    TensorView* t0;
    TensorView* quantized_tensor;
    TensorView* block_scales;
    TensorView* t_out;
  };

  // Helper function to create a fusion with blockQuantize and apply scheduling
  FusionSetup createBlockQuantizeFusion(int64_t dim = 2) {
    FusionSetup setup;
    setup.fusion = std::make_unique<Fusion>();
    FusionGuard fg(setup.fusion.get());

    setup.tv_data_hp = makeContigTensor(dim, DataType::Float);
    setup.fusion->addInput(setup.tv_data_hp);

    setup.t0 = set(setup.tv_data_hp);
    auto quantization_results = blockQuantize(setup.t0);
    setup.quantized_tensor = quantization_results.quantized_tensor;
    setup.block_scales = quantization_results.block_scales;
    setup.t_out = set(setup.quantized_tensor);

    setup.fusion->addOutput(setup.block_scales);
    setup.fusion->addOutput(setup.t_out);

    return setup;
  }

  // Helper to apply common merge and split operations
  // This is limited for the tests with 2D tv inputs.
  void applyMergeAndSplit(
      TensorView* t,
      int64_t split_factor,
      int64_t inner_split = 1,
      int64_t thread_split = 128) {
    // Merge all dims
    // (I0, I1) -> (I0*I1) == (I)
    t->merge(-2);

    // Apply splits: I -> I/split_factor, split_factor
    t->split(-1, split_factor);
    // I/split_factor, split_factor -> I/split_factor/inner_split, inner_split,
    // split_factor
    t->split(-2, inner_split);
    // I/split_factor/inner_split, inner_split, split_factor  ->
    // I/split_factor/inner_split/thread_split, thread_split, inner_split,
    // split_factor
    t->split(-3, thread_split);
  }
};

// Input is in global memory - not valid
TEST_F(BlockQuantizationValidationTest, InputMustBeInLocalMemory) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  auto quantization_results = blockQuantize(tv_data_hp);
  auto t_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(quantization_results.block_scales);
  fusion->addOutput(t_out);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Input must be a local memory tensor")));
}

// Quantized output is written to global memory - not valid
TEST_F(BlockQuantizationValidationTest, QuantizedOutputMustBeInLocalMemory) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  tv_data_hp = set(tv_data_hp);
  auto quantization_results = blockQuantize(tv_data_hp);

  fusion->addOutput(quantization_results.block_scales);
  fusion->addOutput(quantization_results.quantized_tensor);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Quantized output must be a local memory tensor")));
}

// Block scaling factor is written to local memory - not valid
TEST_F(
    BlockQuantizationValidationTest,
    BlockScalingFactorMustBeInGlobalMemory) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  tv_data_hp = set(tv_data_hp);
  auto quantization_results = blockQuantize(tv_data_hp);
  auto tv_block_scales = set(quantization_results.block_scales);
  auto tv_quantized_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(tv_block_scales);
  fusion->addOutput(tv_quantized_out);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Block scaling factor must be a global memory tensor")));
}

namespace {

// A helper function to create a BlockQuantizationOp in the IR.
// It'll set the swizzle scales flag, but won't actually swizzle it.
// This will help us run the validation tests.
BlockQuantizationResults createBlockQuantizationOpWithSwizzleFlag(
    TensorView* input) {
  auto inp_domain = TensorDomain::noReductions(input->getLogicalDomain());

  std::vector<IterDomain*> quantized_out_domain;
  quantized_out_domain.reserve(inp_domain.size());

  for (auto inp_domain_ptr : inp_domain) {
    quantized_out_domain.push_back(inp_domain_ptr->cloneWithoutRFactor());
  }

  std::vector<IterDomain*> scales_out_domain;
  scales_out_domain.reserve(inp_domain.size());

  for (auto inp_id : inp_domain) {
    if (inp_id == inp_domain.back()) {
      scales_out_domain.push_back(
          IterDomainBuilder(
              inp_id->start(),
              SimplifyingIrBuilder::divExpr(
                  inp_id->extent(),
                  IrBuilder::create<Val>(block_size, DataType::Index)))
              .build());

    } else {
      scales_out_domain.push_back(inp_id->cloneWithoutRFactor());
    }
  }

  // Create output tensors
  TensorView* quantized_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          quantized_out_domain,
          TensorDomain::getContiguityFilledWith(quantized_out_domain, true)),
      DataType::Float4_e2m1fn);

  TensorView* block_scales = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          scales_out_domain,
          TensorDomain::getContiguityFilledWith(scales_out_domain, true)),
      DataType::Float8_e4m3fn);

  IrBuilder::create<BlockQuantizationOp>(
      block_scales,
      quantized_tensor,
      input,
      /*logical_index=*/nullptr,
      /*global_scaling_factor=*/nullptr,
      /*block_size=*/16,
      /*swizzle_scales=*/true);

  return BlockQuantizationResults(quantized_tensor, block_scales);
}
} // namespace

TEST_F(
    BlockQuantizationValidationTest,
    InvalidSwizzlePermutationOnBlockScales) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  tv_data_hp = set(tv_data_hp);
  auto quantization_results =
      createBlockQuantizationOpWithSwizzleFlag(tv_data_hp);
  auto tv_quantized_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(tv_quantized_out);
  fusion->addOutput(quantization_results.block_scales);

  quantization_results.block_scales->split(0, 128);
  quantization_results.block_scales->split(1, 32);
  quantization_results.block_scales->split(3, 4);

  // Bad permutation.
  std::vector<IterDomain*> tv_block_scale_alloc{
      quantization_results.block_scales->axis(0),
      quantization_results.block_scales->axis(1),
      quantization_results.block_scales->axis(2),
      quantization_results.block_scales->axis(3),
      quantization_results.block_scales->axis(4)};
  quantization_results.block_scales->setAllocationDomain(
      tv_block_scale_alloc, true);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Block scale swizzle permutation is invalid")));
}

TEST_F(
    BlockQuantizationValidationTest,
    SwizzleInnermostSplitMustHaveExtentOfFour) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  tv_data_hp = set(tv_data_hp);
  auto quantization_results =
      createBlockQuantizationOpWithSwizzleFlag(tv_data_hp);
  auto tv_quantized_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(tv_quantized_out);
  fusion->addOutput(quantization_results.block_scales);

  quantization_results.block_scales->split(0, 128);
  quantization_results.block_scales->split(1, 32);
  // Bad inner split.
  quantization_results.block_scales->split(3, 8);

  std::vector<IterDomain*> tv_block_scale_alloc{
      quantization_results.block_scales->axis(0),
      quantization_results.block_scales->axis(3),
      quantization_results.block_scales->axis(2),
      quantization_results.block_scales->axis(1),
      quantization_results.block_scales->axis(4)};
  quantization_results.block_scales->setAllocationDomain(
      tv_block_scale_alloc, true);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("The innermost split in block scale swizzle must "
                             "have an extent of 4")));
}

TEST_F(
    BlockQuantizationValidationTest,
    SwizzleAllocationDomainMustHaveAtMostFiveIterDomains) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  tv_data_hp = set(tv_data_hp);
  auto quantization_results =
      createBlockQuantizationOpWithSwizzleFlag(tv_data_hp);
  auto tv_quantized_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(tv_quantized_out);
  fusion->addOutput(quantization_results.block_scales);

  quantization_results.block_scales->split(0, 128);
  quantization_results.block_scales->split(1, 32);
  // Too many splits - allocation domain is now more than 5.
  quantization_results.block_scales->split(2, 4);
  quantization_results.block_scales->split(4, 4);

  std::vector<IterDomain*> tv_block_scale_alloc{
      quantization_results.block_scales->axis(0),
      quantization_results.block_scales->axis(3),
      quantization_results.block_scales->axis(2),
      quantization_results.block_scales->axis(1),
      quantization_results.block_scales->axis(5),
      quantization_results.block_scales->axis(4)};
  quantization_results.block_scales->setAllocationDomain(
      tv_block_scale_alloc, true);

  EXPECT_THAT(
      [&]() { GpuLower(fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Block scale swizzle must have 2D logical domain "
                             "and 5D allocation domain")));
}

// Group ID must be the innermost of all splits from logical domains to loop
// domains
TEST_F(BlockQuantizationValidationTest, GroupIDMustBeInnermost) {
  auto setup = createBlockQuantizeFusion();
  FusionGuard fg(setup.fusion.get());

  std::vector<TensorView*> tensors = {
      setup.tv_data_hp,
      setup.t0,
      setup.quantized_tensor,
      setup.block_scales,
      setup.t_out};

  for (auto t : tensors) {
    applyMergeAndSplit(
        t, /*split_factor=*/128, /*inner_split=*/1, /*thread_split=*/4);

    if (t != setup.tv_data_hp) {
      // Mark outer ID as Group for quantized outputs (should fail)
      // instead of innermost ID
      if (t == setup.block_scales || t == setup.quantized_tensor) {
        t->axis(-3)->parallelize(ParallelType::Group);
        t->axis(-1)->parallelize(ParallelType::TIDx);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
        t->axis(-3)->parallelize(ParallelType::TIDx);
      }
      t->axis(-4)->parallelize(ParallelType::BIDx);
    }
  }

  EXPECT_THAT(
      [&]() { GpuLower(setup.fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "The grouped ID must correspond to the innermost of all splits from "
          "logical domains to loop domains for BlockQuantizationOp")));
}

// We do not allow IDs of types serial, unroll, unswitch to have extent > 1
// We do not want the runtime kernel which implement block quantization to be
// called multiple times in a kernel as yet
TEST_F(BlockQuantizationValidationTest, NonParallelizedIDsMustHaveExtentOfOne) {
  auto setup = createBlockQuantizeFusion();
  FusionGuard fg(setup.fusion.get());

  std::vector<TensorView*> tensors = {
      setup.tv_data_hp,
      setup.t0,
      setup.quantized_tensor,
      setup.block_scales,
      setup.t_out};

  for (auto t : tensors) {
    // There will be a non-parallelized ID with a trip count of 2
    applyMergeAndSplit(t, /*split_factor=*/4, /*inner_split=*/2);

    if (t != setup.tv_data_hp) {
      if (t == setup.block_scales || t == setup.quantized_tensor) {
        t->axis(-1)->parallelize(ParallelType::Group);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
      }
      t->axis(-3)->parallelize(ParallelType::TIDx);
      t->axis(-4)->parallelize(ParallelType::BIDx);
    }
  }

  EXPECT_THAT(
      [&]() { GpuLower(setup.fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Expected non-TID/BID/Group ID to have extent of 1 for "
          "BlockQuantizationOp")));
}

// The runtime kernel for block quantization expects TIDx to access contiguous
// memory locations - just 16, but to be safe we enforce all memory locations of
// TIDx are contiguous. To enforce this, TIDx must be the second innermost ID
// after Group ID. By that we mean if we derive this ID from the logical domain,
// there should be no other IDs between Group ID and TIDx except for IDs with
// extent of 1.
TEST_F(BlockQuantizationValidationTest, TIDxMustBeSecondInnermostAfterGroupID) {
  auto setup = createBlockQuantizeFusion();
  FusionGuard fg(setup.fusion.get());

  std::vector<TensorView*> tensors = {
      setup.tv_data_hp,
      setup.t0,
      setup.quantized_tensor,
      setup.block_scales,
      setup.t_out};

  for (auto t : tensors) {
    applyMergeAndSplit(t, /*split_factor=*/4);

    if (t != setup.tv_data_hp) {
      if (t == setup.block_scales || t == setup.quantized_tensor) {
        t->axis(-1)->parallelize(ParallelType::Group);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
      }
      // TIDx is "outer" compared to BIDx causing a failure
      t->axis(-3)->parallelize(ParallelType::BIDx);
      t->axis(-4)->parallelize(ParallelType::TIDx);
    }
  }

  EXPECT_THAT(
      [&]() { GpuLower(setup.fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Expected IDs between Group ID and TIDx to have extent of 1 for "
          "BlockQuantizationOp:")));
}

// When running validation checks we traverse from loop to logical domain
// and vice-versa. During this traversal, when we encounter a merge operation,
// we find all input IDs to the merge (traced back to the logical domain of the
// quantized output). The input IDs in the logical domain need to be contiguous.
TEST_F(BlockQuantizationValidationTest, MergesMustBeContiguous) {
  auto setup = createBlockQuantizeFusion(/*dim=*/3);
  FusionGuard fg(setup.fusion.get());

  std::vector<TensorView*> tensors = {
      setup.tv_data_hp,
      setup.t0,
      setup.quantized_tensor,
      setup.block_scales,
      setup.t_out};

  for (auto t : tensors) {
    // Merge first two dims instead of last two
    // This will cause a failure as the merged IDs are not contiguous
    t->reorder({{0, 1}, {1, 0}}); // (i0, i1, i2) -> (i1, i0, i2)
    t->merge(1);

    // split I1 by 4
    t->split(-1, 4);
    // I/4, 4 -> I/4, 1, 4
    t->split(-2, 1);
    // I/4, 1, 4 -> I/512, 128, 1, 4
    t->split(-3, 128);

    if (t != setup.tv_data_hp) {
      if (t == setup.block_scales || t == setup.quantized_tensor) {
        t->axis(-1)->parallelize(ParallelType::Group);
      } else {
        t->axis(-1)->parallelize(ParallelType::Vectorize);
      }
      t->axis(-3)->parallelize(ParallelType::TIDx);
      t->axis(-4)->parallelize(ParallelType::BIDx);
      t->axis(-5)->parallelize(ParallelType::BIDy);
    }
  }

  EXPECT_THAT(
      [&]() { GpuLower(setup.fusion.get()).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "All merge operations deriving the grouped ID must combine "
          "contiguous "
          "IDs from the logical domain for BlockQuantizationOp")));
}

class BlockQuantizationSchedulingTest
    : public BlackwellBase,
      public ::testing::WithParamInterface<
          std::tuple<DataType, std::pair<int, int>, bool, bool>> {};

TEST_P(BlockQuantizationSchedulingTest, AutoScheduleSingleOp) {
  const auto data_type = std::get<0>(GetParam());
  const auto dimensions = std::get<1>(GetParam());
  const auto use_global_scale = std::get<2>(GetParam());
  const auto swizzle_block_scales = std::get<3>(GetParam());
  const int m = dimensions.first;
  const int n = dimensions.second;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QuantizationFusion(
      fusion.get(), data_type, use_global_scale, swizzle_block_scales);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  auto in_tensor = at::randn({m, n}, at::device(at::kCUDA).dtype(at::kFloat))
                       .to(data_type_to_aten(data_type));
  inputs.push_back(in_tensor);
  if (use_global_scale) {
    // Calculate the max value in the in_tensor.
    auto max_value = 4.480000000e+02f /*FLOAT8_E4M3_MAX*/
        * 6.0f /*FLOAT4_E2M1_MAX*/
        / in_tensor.max().to(at::kFloat);
    inputs.push_back(max_value);
  }
  auto outputs_baseline = fec.runFusionWithInputs(inputs);

  auto baseline_block_scales = outputs_baseline[0].as<at::Tensor>();
  auto baseline_quantized_tensor = outputs_baseline[1].as<at::Tensor>();

  auto baseline_block_scales_cpu = baseline_block_scales.cpu();
  auto baseline_quantized_tensor_cpu = baseline_quantized_tensor.cpu();

  const uint8_t* baseline_block_scales_data =
      static_cast<const uint8_t*>(baseline_block_scales_cpu.data_ptr());
  const uint8_t* baseline_quantized_data =
      static_cast<const uint8_t*>(baseline_quantized_tensor_cpu.data_ptr());

  std::unique_ptr<Fusion> fusion_new_op = std::make_unique<Fusion>();
  FusionGuard fg2(fusion_new_op.get());

  auto tv_in_1 = makeContigTensor(2, data_type);
  auto tv_global_scale =
      use_global_scale ? makeContigTensor(0, DataType::Float) : nullptr;
  fusion_new_op->addInput(tv_in_1);
  if (use_global_scale) {
    fusion_new_op->addInput(tv_global_scale);
  }

  auto quantization_results = blockQuantize(
      tv_in_1, tv_global_scale, /*block_size=*/16, swizzle_block_scales);

  fusion_new_op->addOutput(quantization_results.block_scales);
  fusion_new_op->addOutput(quantization_results.quantized_tensor);

  FusionExecutorCache executor_cache(std::move(fusion_new_op));
  auto outputs_new_op = executor_cache.runFusionWithInputs(inputs);

  // Verify we got the expected outputs
  auto block_scales_output = outputs_new_op[0].as<at::Tensor>();
  auto quantized_tensor_output = outputs_new_op[1].as<at::Tensor>();

  // Move tensors from GPU to CPU
  auto block_scales_cpu = block_scales_output.cpu();
  auto quantized_tensor_cpu = quantized_tensor_output.cpu();

  auto block_scales_bytes = (m * n) / block_size;
  auto quantized_tensor_bytes = (m * n) / 2;

  const uint8_t* block_scales_data =
      static_cast<const uint8_t*>(block_scales_cpu.data_ptr());
  for (int i = 0; i < block_scales_bytes; ++i) {
    EXPECT_EQ(
        block_scales_data[i],
        baseline_block_scales_data[i]); // Compare with baseline
  }

  const uint8_t* quantized_data =
      static_cast<const uint8_t*>(quantized_tensor_cpu.data_ptr());
  for (int i = 0; i < quantized_tensor_bytes; ++i) {
    EXPECT_EQ(
        quantized_data[i],
        baseline_quantized_data[i]); // Compare with baseline
  }
}

class BlockQuantizationCanScheduleTests : public BlackwellBase {};

TEST_F(
    BlockQuantizationCanScheduleTests,
    CanRuntimeScheduleFailFromNoVectorization) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv_data_hp);

  auto t0 = set(tv_data_hp);
  auto quantization_results = blockQuantize(t0);
  auto t_out = set(quantization_results.quantized_tensor);

  fusion->addOutput(quantization_results.block_scales);
  fusion->addOutput(t_out);

  // Create misaligned tensor directly on GPU using custom CUDA allocation
  size_t element_size = 4;
  int m = 1024;
  int n = 1024;

  size_t total_elements = m * n;
  size_t buffer_size =
      total_elements * element_size + 16; // Extra bytes for misalignment

  // Allocate GPU memory with extra space
  void* gpu_ptr;
  cudaMalloc(&gpu_ptr, buffer_size);

  // Create tensor from GPU memory at offset of 4 bytes
  void* misaligned_ptr = static_cast<char*>(gpu_ptr) + 4;
  auto misaligned_gpu_tensor = at::from_blob(
      misaligned_ptr,
      {m, n},
      at::TensorOptions()
          .dtype(data_type_to_aten(DataType::Float))
          .device(at::kCUDA));

  auto good_input = at::randn({m, n}, at::device(at::kCUDA).dtype(at::kFloat));

  // Expect failure as the input tensor can't be vectorized
  // and we need vectorization > 2
  SchedulerRuntimeInfo runtime_info(fusion.get(), {misaligned_gpu_tensor});
  ASSERT_FALSE(Schedule::canSchedule(
      SchedulerType::PointWise, fusion.get(), runtime_info));

  SchedulerRuntimeInfo runtime_info_new(fusion.get(), {good_input});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::PointWise, fusion.get(), runtime_info_new));

  if (gpu_ptr)
    cudaFree(gpu_ptr);
}

TEST_P(NVFP4QuantizeTest, SwizzledOuputAndWithoutPerTensorAmax) {
  auto data_hp_dtype = GetParam();

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion->addInput(tv_data_hp);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

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

  auto tv_data_lp_fp4 = castOp(DataType::Float4_e2m1fn, tv_data_scaled_clamp);
  auto tv_data_lp = reshape(tv_data_lp_fp4, [](auto& x) { x.merge(-2); });

  fusion->addOutput(tv_block_scale_fp8);
  fusion->addOutput(tv_data_lp);

  nvfuser::ir_utils::swizzleBlockScales(tv_block_scale_fp8);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  inputs.push_back(
      at::randn({1024, 1024}, at::device(at::kCUDA).dtype(at::kFloat))
          .to(data_type_to_aten(data_hp_dtype)));
  auto outputs = fec.runFusionWithInputs(inputs);

  // Check that the fusion is segmented into two groups.
  // The normalization scheduler is used for the first group
  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::InnerPersistent)));
}

TEST_P(NVFP4QuantizeTest, WithPerTensorAmax) {
  auto data_hp_dtype = GetParam();

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  auto tv_per_tensor_scale = makeContigTensor(0, DataType::Float);
  fusion->addInput(tv_data_hp);
  fusion->addInput(tv_per_tensor_scale);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  auto tv_data_hp_abs = abs(tv_data_hp_reshaped);
  auto tv_data_hp_amax = max(tv_data_hp_abs, {-1});
  // These scales are currently in fp32, we are going to `quantize` them to e4m3
  // Note: in the torchao implementation, tv_block_scale is bf16 if the input is
  // bf16 But in our case, tv_block_scale is always fp32, regardless of the
  // input dtype.
  auto tv_block_scale = div(
      tv_data_hp_amax, IrBuilder::create<Val>(F4_E2M1_MAX, DataType::Float));

  auto tv_scaled_block_scales = div(tv_block_scale, tv_per_tensor_scale);
  auto tv_scaled_block_scales_clamp = clamp(
      tv_scaled_block_scales,
      IrBuilder::create<Val>(E4M3_EPS, DataType::Float),
      IrBuilder::create<Val>(F8E4M3_MAX, DataType::Float));
  auto tv_scaled_block_scales_fp8 =
      castOp(DataType::Float8_e4m3fn, tv_scaled_block_scales_clamp);

  // TODO: should we just use auto tv_block_scale_fp32 = tv_block_scale_clamp?
  auto tv_scaled_block_scales_fp32 =
      castOp(DataType::Float, tv_scaled_block_scales_fp8);

  // Temporary dequant the scaled_block_scales_fp32 to get the per_tensor_scale
  // To apply to data
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

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  inputs.push_back(
      at::randn({1024, 1024}, at::device(at::kCUDA).dtype(at::kFloat))
          .to(data_type_to_aten(data_hp_dtype)));
  inputs.push_back(at::randn({}, at::device(at::kCUDA).dtype(at::kFloat)));
  auto outputs = fec.runFusionWithInputs(inputs);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    NVFP4QuantizeTest,
    ::testing::Values(DataType::BFloat16, DataType::Float),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    ,
    BlockQuantizationTest,
    ::testing::Values(
        std::make_tuple(DataType::Float, 2),
        std::make_tuple(DataType::Float, 4),
        std::make_tuple(DataType::BFloat16, 2),
        std::make_tuple(DataType::BFloat16, 4),
        std::make_tuple(DataType::BFloat16, 8),
        std::make_tuple(DataType::Half, 2),
        std::make_tuple(DataType::Half, 4),
        std::make_tuple(DataType::Half, 8)),
    [](const testing::TestParamInfo<std::tuple<DataType, int64_t>>& info) {
      std::ostringstream os;
      os << std::get<0>(info.param) << "_GroupWidth" << std::get<1>(info.param);
      return os.str();
    });

INSTANTIATE_TEST_SUITE_P(
    BlockQuantizationSchedulingTestSuite,
    BlockQuantizationSchedulingTest,
    ::testing::Combine(
        ::testing::Values(DataType::Float, DataType::BFloat16),
        ::testing::Values(
            std::make_pair(1024, 1024),
            std::make_pair(128, 64),
            std::make_pair(2048, 128),
            std::make_pair(2048, 2048)),
        ::testing::Bool(),
        ::testing::Bool()),
    [](const testing::TestParamInfo<
        std::tuple<DataType, std::pair<int, int>, bool, bool>>& info) {
      const auto data_type = std::get<0>(info.param);
      const auto dimensions = std::get<1>(info.param);
      const auto use_global_scale = std::get<2>(info.param);
      const auto swizzle_block_scales = std::get<3>(info.param);

      std::ostringstream name;
      name << data_type << "_" << dimensions.first << "x" << dimensions.second;
      name << (use_global_scale ? "_WithGlobalScale" : "_NoGlobalScale");
      name << (swizzle_block_scales ? "_WithSwizzle" : "_NoSwizzle");
      return name.str();
    });

} // namespace nvfuser
