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

TEST_F(AllocationDomainTest, NHWC2NHWC_CUDA) {
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
          ::testing::HasSubstr("The memory format of input tensor")));

  auto cg_outputs = fec.runFusionWithInputs({t0});

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  // TODO: assert vectorized

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, NCHW2NHWC_CUDA) {
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

  // TODO: assert vectorized

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, Unspecified2NHWC_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_nchw = at::randn({n, c, h, w}, options);
  at::Tensor t0_nhwc = t0_nchw.contiguous(at::MemoryFormat::ChannelsLast);

  FusionExecutorCache fec(std::move(fusion_ptr));

  // run with nchw input
  auto cg_outputs = fec.runFusionWithInputs({t0_nchw});
  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Transpose});
  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));
  // TODO: assert vectorized
  testValidate(&fusion, cg_outputs, {t0_nchw}, {t0_nchw}, __LINE__, __FILE__);

  // run with nhwc input
  cg_outputs = fec.runFusionWithInputs({t0_nhwc});
  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});
  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));
  // TODO: assert vectorized
  testValidate(&fusion, cg_outputs, {t0_nhwc}, {t0_nhwc}, __LINE__, __FILE__);
}

} // namespace nvfuser
