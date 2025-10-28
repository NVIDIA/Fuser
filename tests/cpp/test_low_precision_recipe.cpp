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
void createNVFP4QunatizationFusion(Fusion* fusion, DataType data_hp_dtype) {
  auto tv_data_hp = makeContigTensor(2, data_hp_dtype);
  fusion->addInput(tv_data_hp);

  auto tv_data_hp_reshaped =
      reshape(tv_data_hp, [](auto& x) { x.split(-1, block_size); });

  // cast it to FP32
  tv_data_hp_reshaped = castOp(DataType::Float, tv_data_hp_reshaped);

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
}
} // namespace

TEST_P(NVFP4QuantizeTest, WithoutPerTensorAmax) {
  auto data_hp_dtype = GetParam();

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  createNVFP4QunatizationFusion(fusion.get(), data_hp_dtype);

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

struct BlockQuantizationTestParams {
  DataType data_type;
  int vectorization_width;
};

class BlockQuantizationTest
    : public BlackwellBase,
      public ::testing::WithParamInterface<BlockQuantizationTestParams> {};

TEST_P(BlockQuantizationTest, ScheduleAsPointwise) {
  auto params = GetParam();
  auto data_hp_dtype = params.data_type;
  auto vectorization_factor = params.vectorization_width;

  // Baseline implementation
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QunatizationFusion(fusion.get(), data_hp_dtype);

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

    // split by 4 (or 8).
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
  auto params = GetParam();
  auto data_hp_dtype = params.data_type;
  auto vectorization_factor = params.vectorization_width;

  // Baseline  implementation
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QunatizationFusion(fusion.get(), data_hp_dtype);

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

  for (auto t :
       {tv_data_hp,
        t0,
        quantization_results.quantized_tensor,
        quantization_results.block_scales,
        t_out}) {
    // (m, n) -> (m, n/4, 4) (or (m, n/8, 8) if bfloat16)
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

struct BlockQuantizationSchedulingTestParams {
  DataType data_type;
  int m;
  int n;
};

class BlockQuantizationSchedulingTest
    : public BlackwellBase,
      public ::testing::WithParamInterface<
          BlockQuantizationSchedulingTestParams> {};

TEST_P(BlockQuantizationSchedulingTest, AutoScheduleSingleOp) {
  auto params = GetParam();
  auto data_type = params.data_type;
  const int m = params.m;
  const int n = params.n;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  createNVFP4QunatizationFusion(fusion.get(), data_type);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> inputs;
  inputs.push_back(at::randn({m, n}, at::device(at::kCUDA).dtype(at::kFloat))
                       .to(data_type_to_aten(data_type)));
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
  fusion_new_op->addInput(tv_in_1);

  // t0 is 2D
  auto quantization_results = blockQuantize(tv_in_1);

  // outputs are 3D
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

  tv_block_scale_fp8->split(0, 128);
  // m/128, 128, k
  tv_block_scale_fp8->split(1, 32);
  // m/128, 4(m_o), 32(m_i), k
  tv_block_scale_fp8->split(3, 4);
  // m/128, 4(m_o), 32(m_i), k/4, 4(k)
  std::vector<IterDomain*> tv_block_scale_fp8_alloc{
      tv_block_scale_fp8->axis(0),
      tv_block_scale_fp8->axis(3),
      tv_block_scale_fp8->axis(2),
      tv_block_scale_fp8->axis(1),
      tv_block_scale_fp8->axis(4)};
  // m/128, k/4, 32(m_i), 4(m_o), 4(k)
  tv_block_scale_fp8->setAllocationDomain(tv_block_scale_fp8_alloc, true);

  // back to a 2D logical domain.
  tv_block_scale_fp8->merge(0);
  tv_block_scale_fp8->merge(0);
  tv_block_scale_fp8->merge(-1);

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
        BlockQuantizationTestParams{DataType::BFloat16, 2},
        BlockQuantizationTestParams{DataType::BFloat16, 4},
        BlockQuantizationTestParams{DataType::BFloat16, 8},
        BlockQuantizationTestParams{DataType::Float, 2},
        BlockQuantizationTestParams{DataType::Float, 4},
        BlockQuantizationTestParams{DataType::Half, 2},
        BlockQuantizationTestParams{DataType::Half, 4},
        BlockQuantizationTestParams{DataType::Half, 8}),
    [](const testing::TestParamInfo<BlockQuantizationTestParams>& info) {
      std::ostringstream name;
      name << info.param.data_type << "_VecWidth_"
           << info.param.vectorization_width;
      return name.str();
    });

INSTANTIATE_TEST_SUITE_P(
    ,
    BlockQuantizationSchedulingTest,
    ::testing::Values(
        BlockQuantizationSchedulingTestParams{DataType::Float, 1024, 1024},
        BlockQuantizationSchedulingTestParams{DataType::Float, 128, 64},
        BlockQuantizationSchedulingTestParams{DataType::Float, 2048, 128},
        BlockQuantizationSchedulingTestParams{DataType::Float, 2048, 2048},
        BlockQuantizationSchedulingTestParams{DataType::BFloat16, 1024, 1024},
        BlockQuantizationSchedulingTestParams{DataType::BFloat16, 128, 64},
        BlockQuantizationSchedulingTestParams{DataType::BFloat16, 2048, 128},
        BlockQuantizationSchedulingTestParams{DataType::BFloat16, 2048, 2048}),
    [](const testing::TestParamInfo<BlockQuantizationSchedulingTestParams>&
           info) {
      std::ostringstream name;
      name << info.param.data_type << "_" << info.param.m << "x"
           << info.param.n;
      return name.str();
    });

} // namespace nvfuser
