// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationDomainTest : public NVFuserTest {};

TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 = t0_wrong_format.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutorCache fec(std::move(fusion_ptr));

  EXPECT_THAT(
      [&]() { fec.runFusionWithInputs({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Contiguity not match")));

  auto cg_outputs = fec.runFusionWithInputs({t0});

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  ASSERT_TRUE(hasVectorizeOutput(fec.getMostRecentKernelRuntime()));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NHWC1d_To_NHWC4d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_1d = {tv0->axis(0)};
  tv0->setAllocationDomain(tv0_1d, true);
  tv0->split(0, IrBuilder::create<Int>()); // C
  tv0->split(0, IrBuilder::create<Int>()); // W
  tv0->split(0, IrBuilder::create<Int>()); // H
  tv0->reorder({{-1, 1}});
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 = t0_wrong_format.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutorCache fec(std::move(fusion_ptr));

  EXPECT_THAT(
      [&]() { fec.runFusionWithInputs({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fec.runFusionWithInputs({t0});

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  ASSERT_TRUE(hasVectorizeOutput(fec.getMostRecentKernelRuntime()));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NHWC4d_To_NHWC1d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  // [N, C, H, W] -> [N, H, W, C]
  tv1->reorder({{1, -1}});
  // [N, H, W, C] -> [N*H*W*C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);

  std::vector<IterDomain*> tv1_1d = {tv1->axis(0)};
  tv1->setAllocationDomain(tv1_1d, true);

  tv1->split(0, 8);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 = t0_wrong_format.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Contiguity not match")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NHWC1d_To_NHWC1d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_1d = {tv0->axis(0)};
  tv0->setAllocationDomain(tv0_1d, true);
  tv0->split(0, IrBuilder::create<Int>()); // C
  tv0->split(0, IrBuilder::create<Int>()); // W
  tv0->split(0, IrBuilder::create<Int>()); // H
  // [N, H, W, C]
  tv0->reorder({{-1, 1}});
  // [N, C, H, W]
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  // [N, C, H, W] -> [N, H, W, C]
  tv1->reorder({{1, -1}});
  // [N, H, W, C] -> [N*H*W*C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);

  std::vector<IterDomain*> tv1_1d = {tv1->axis(0)};
  tv1->setAllocationDomain(tv1_1d, true);

  tv1->split(0, 8);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 = t0_wrong_format.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h / 8, 8 * w * c});
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_2d = {tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_2d, true);
  tv0->merge(0);
  tv0->split(0, IrBuilder::create<Int>()); // C
  tv0->split(0, IrBuilder::create<Int>()); // W
  tv0->split(0, IrBuilder::create<Int>()); // H
  // [N, H, W, C]
  tv0->reorder({{-1, 1}});
  // [N, C, H, W]
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);
  // [N*H*W*C]

  tv1->split(0, 8);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  std::vector<IterDomain*> tv1_2d = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_2d, true);

  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 = t0_wrong_format.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NCHW4d_To_NHWC4d_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(1), tv0->axis(2), tv0->axis(3)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);
  at::Tensor t0_wrong_format = t0.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutorCache fec(std::move(fusion_ptr));

  EXPECT_THAT(
      [&]() { fec.runFusionWithInputs({t0_wrong_format}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fec.runFusionWithInputs({t0});

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Transpose});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  ASSERT_TRUE(hasVectorizeOutput(fec.getMostRecentKernelRuntime()));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, TransposedIntermediate_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  auto bc = fusion.bankConflictInfo();
  ASSERT_TRUE(bc.size() == 1);
  auto [read, write] = bc.at(tv1);
  ASSERT_EQ(read, std::vector<int>{32});
  ASSERT_EQ(write, std::vector<int>{32});

  std::vector<IterDomain*> tv1_transposed = {tv0->axis(1), tv0->axis(0)};
  tv0->setAllocationDomain(tv1_transposed, true);

  ASSERT_TRUE(fusion.bankConflictInfo().empty());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
