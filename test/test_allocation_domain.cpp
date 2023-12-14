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
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationDomainTest : public NVFuserTest {};

using ::testing::ElementsAre;

// A global->shared->global copy kernel, shared memory allocated transposed to
// avoid bank conflict.
TEST_F(AllocationDomainTest, TransposedIntermediate) {
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

  std::vector<IterDomain*> tv1_transposed = {tv1->axis(1), tv1->axis(0)};
  tv1->setAllocationDomain(tv1_transposed, true);

  ASSERT_TRUE(fusion.bankConflictInfo().empty());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 4d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC4d) {
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

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 1d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC1d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  tv1->setAllocationDomain({tv1->axis(0)}, true);
  // [N*H*W*C]
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 2d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC2d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 128);
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(1)}, true);
  // [N*H*W*C/128, 128]
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 104, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Reshape and transpose a 3d tensor into an NHWC tensor with a 3d allocation
// domain in fusion output.
TEST_F(AllocationDomainTest, Tensor3d_To_NHWC3d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n1 = 31, n2 = 29, h = 64, w = 104, c = 21;

  auto tv0 = makeContigTensor(3); // [N1, N2, H*W*C]
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(1), tv1->axis(2)}, true);
  tv1->merge(0);
  tv1->split(1, c);
  tv1->split(1, w);
  // [N, H, W, C]
  tv1->reorder({{-1, 1}});
  tv1->commitLeafToRFactor();
  // [N, C, H, W]

  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 128);
  // [N*H*W*C/128, 128]
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({n1, n2, h * w * c}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {t0.view({n1 * n2, h, w, c}).permute({0, 3, 1, 2})},
      __LINE__,
      __FILE__);
}

// Reshape a 3d tensor into an NHWC tensor with a 4d allocation domain in fusion
// output. The allocation domain is on both the producer and the consumer side
// of the rFactor domain.
TEST_F(AllocationDomainTest, Tensor3d_To_NHWC4d_FwdBwd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n1 = 31, n2 = 29, h = 64, w = 104, c = 21;

  auto tv0 = makeContigTensor(3); // [N1, N2, H*W*C]
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_alloc{tv1->axis(0), tv1->axis(1)};
  tv1->merge(0);
  tv1->split(1, w);
  tv1->split(1, h);
  tv1->commitLeafToRFactor();
  // [N, C, H, W]

  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(1);
  // [N, H*W, C]
  tv1_alloc.emplace_back(tv1->axis(1));
  tv1_alloc.emplace_back(tv1->axis(2));
  // tv1_alloc = [N1, N2, H*W, C]
  tv1->setAllocationDomain(tv1_alloc, true);

  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 128);
  // [N*H*W*C/128, 128]
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({n1, n2, c * h * w}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {t0.view({n1 * n2, c, h, w})},
      __LINE__,
      __FILE__);
}

// A global->global copy kernel where both inputs and outputs are NHWC memory
// format
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d) {
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

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 4);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx, V]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC4d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h * w * c});
  fusion.addInput(tv0);
  std::vector<IterDomain*> tv0_1d = {tv0->axis(0)};
  tv0->setAllocationDomain(tv0_1d, true);
  tv0->split(0, c);
  tv0->split(0, w);
  tv0->split(0, h);
  tv0->reorder({{-1, 1}});
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(0);
  // [N*H*W*C]
  tv1->split(0, 4);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx, V]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain of the output view the output as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC1d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

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

  tv1->split(0, 4);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx, V]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view both the input and the output as a 1d tensors.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC1d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h * w * c});
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_1d = {tv0->axis(0)};
  tv0->setAllocationDomain(tv0_1d, true);
  tv0->split(0, c);
  tv0->split(0, w);
  tv0->split(0, h);
  tv0->reorder({{-1, 1}});
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W] -> [N, H, W, C]
  tv1->reorder({{1, -1}});
  // [N, H, W, C] -> [N*H*W*C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);

  std::vector<IterDomain*> tv1_1d = {tv1->axis(0)};
  tv1->setAllocationDomain(tv1_1d, true);

  tv1->split(0, 4);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx, V]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 2d tensor of shape [N*H/8, 8*W*C], and
// view the output as a 2d tensor of shape [N*H*W*C/4, 4]
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h / 8, 8 * w * c});
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_2d = {tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_2d, true);
  tv0->merge(0);
  tv0->split(0, c);
  tv0->split(0, w);
  tv0->split(0, h);
  // [N, H, W, C]
  tv0->reorder({{-1, 1}});
  // [N, C, H, W]
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);
  // [N*H*W*C]

  tv1->split(0, 4);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  // [N*H*W*C/4, 4]

  std::vector<IterDomain*> tv1_2d = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_2d, true);

  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx, V]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheBefore) {
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

  auto tv2 = tv1->cacheBefore();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_nhwc);
  ASSERT_EQ(tv1->getAllocationDomain(), expected_new_allocation_domain);
  ASSERT_EQ(tv2->getAllocationDomain(), tv1_nhwc);

  for (auto tv : {tv1, tv2}) {
    // [N, C, H, W]
    tv->reorder({{1, -1}});
    // [N, H, W, C]
    tv->merge(0);
    tv->merge(0);
    tv->merge(0);
    // [N*H*W*C]
    tv->split(0, 4);
    tv->axis(1)->parallelize(ParallelType::Vectorize);
    tv->split(0, 128);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC2d_To_NHWC2d, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheBefore) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h / 8, 8 * w * c});
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_2d = {tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_2d, true);
  tv0->merge(0);
  tv0->split(0, c);
  tv0->split(0, w);
  tv0->split(0, h);
  // [N, H, W, C]
  tv0->reorder({{-1, 1}});
  // [N, C, H, W]
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(0);
  // [N*H*W*C]

  tv1->split(0, 4);
  // [N*H*W*C/4, 4]

  std::vector<IterDomain*> tv1_2d = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_2d, true);

  auto tv2 = tv1->cacheBefore();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv1->axis(0), tv1->axis(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_2d);
  ASSERT_EQ(tv1->getAllocationDomain(), expected_new_allocation_domain);
  ASSERT_EQ(tv2->getAllocationDomain(), tv1_2d);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 128);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::Vectorize);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheAfter
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheAfter) {
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

  auto tv2 = tv0->cacheAfter();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv2->axis(0), tv2->axis(2), tv2->axis(3), tv2->axis(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_nhwc);
  ASSERT_EQ(tv1->getAllocationDomain(), tv1_nhwc);
  ASSERT_EQ(tv2->getAllocationDomain(), expected_new_allocation_domain);

  for (auto tv : {tv1, tv2}) {
    // [N, C, H, W]
    tv->reorder({{1, -1}});
    // [N, H, W, C]
    tv->merge(0);
    tv->merge(0);
    tv->merge(0);
    // [N*H*W*C]
    tv->split(0, 4);
    tv->axis(1)->parallelize(ParallelType::Vectorize);
    tv->split(0, 128);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// NOT similar to NHWC2d_To_NHWC2d, because cacheAfter requires the
// allocation tensor to be between rFactor domain and leaf domain, which is not
// the case for NHWC2d_To_NHWC2d
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheAfter) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  for (auto tv : {tv0, tv1}) {
    // [N, C, H, W]
    tv->reorder({{1, -1}});
    // [N, H, W, C]
    tv->merge(0);
    tv->merge(1);
    tv->merge(0);
    // [N*H*W*C]

    tv->split(0, 4);
    // [N*H*W*C/4, 4]
  }

  std::vector<IterDomain*> tv0_2d = {tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_2d, true);
  std::vector<IterDomain*> tv1_2d = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_2d, true);

  auto tv2 = tv0->cacheAfter();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv2->axis(0), tv2->axis(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_2d);
  ASSERT_EQ(tv1->getAllocationDomain(), tv1_2d);
  ASSERT_EQ(tv2->getAllocationDomain(), expected_new_allocation_domain);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 128);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::Vectorize);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "merging of discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheFork
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheFork) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  std::vector<IterDomain*> tv2_nhwc = {
      tv2->axis(0), tv2->axis(2), tv2->axis(3), tv2->axis(1)};
  tv2->setAllocationDomain(tv2_nhwc, true);

  auto tv3 = tv1->cacheFork();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv3->axis(0), tv3->axis(2), tv3->axis(3), tv3->axis(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_nhwc);
  ASSERT_EQ(tv1->getAllocationDomain(), tv1_nhwc);
  ASSERT_EQ(tv2->getAllocationDomain(), tv2_nhwc);
  ASSERT_EQ(tv3->getAllocationDomain(), expected_new_allocation_domain);

  for (auto tv : {tv1, tv2, tv3}) {
    // [N, C, H, W]
    tv->reorder({{1, -1}});
    // [N, H, W, C]
    tv->merge(0);
    tv->merge(0);
    tv->merge(0);
    // [N*H*W*C]
    tv->split(0, 4);
    tv->axis(1)->parallelize(ParallelType::Vectorize);
    tv->split(0, 128);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC2d_To_NHWC2d, but does a cacheFork
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheFork) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int n = 31, h = 64, w = 103, c = 21;

  auto tv0 = makeContigConcreteTensor({n * h / 8, 8 * w * c});
  fusion.addInput(tv0);

  std::vector<IterDomain*> tv0_2d = {tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_2d, true);
  tv0->merge(0);
  tv0->split(0, c);
  tv0->split(0, w);
  tv0->split(0, h);
  // [N, H, W, C]
  tv0->reorder({{-1, 1}});
  // [N, C, H, W]
  tv0->commitLeafToRFactor();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  for (auto tv : {tv1, tv2}) {
    // [N, C, H, W]
    tv->reorder({{1, -1}});
    // [N, H, W, C]
    tv->merge(0);
    tv->merge(1);
    tv->merge(0);
    // [N*H*W*C]

    tv->split(0, 4);
    // [N*H*W*C/4, 4]
  }

  std::vector<IterDomain*> tv2_2d = {tv2->axis(0), tv2->axis(1)};
  tv2->setAllocationDomain(tv2_2d, true);

  auto tv3 = tv1->cacheFork();

  std::vector<IterDomain*> expected_new_allocation_domain{
      tv3->getMaybeRFactorDomain().at(0),
      tv3->getMaybeRFactorDomain().at(2),
      tv3->getMaybeRFactorDomain().at(3),
      tv3->getMaybeRFactorDomain().at(1)};

  ASSERT_EQ(tv0->getAllocationDomain(), tv0_2d);
  ASSERT_EQ(tv1->getAllocationDomain(), tv1_nhwc);
  ASSERT_EQ(tv2->getAllocationDomain(), tv2_2d);
  ASSERT_EQ(tv3->getAllocationDomain(), expected_new_allocation_domain);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 128);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::Vectorize);
    // [BIDx, TIDx, V]
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0_wrong_format = at::randn({n, c, h, w}, options);
  at::Tensor t0 =
      t0_wrong_format.as_strided({n, c, h, w}, {h * w * c, 1, w * c, c});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});

  EXPECT_THAT(
      [&]() { fe.runFusion({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, VectorizationIssue902) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({16, 16, 512, 64});

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<nvfuser::IterDomain*> aloc_domain;
  aloc_domain.push_back(tv1->axis(0));
  aloc_domain.push_back(tv1->axis(2));
  aloc_domain.push_back(tv1->axis(3));
  aloc_domain.push_back(tv1->axis(1));
  tv1->setAllocationDomain(aloc_domain, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  ASSERT_TRUE(cg_outputs[0].equal(t0));
}

TEST_F(AllocationDomainTest, TransposeMatrix) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3});

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = permute(tv0, {1, 0});
  fusion->addOutput(tv1);

  tv1->setAllocationDomain(
      {tv1->axis(1), tv1->axis(0)}, /*new_contiguity=*/true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(in_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  std::vector<at::Tensor> outputs = executor_cache.runFusionWithInputs({t0});
  at::Tensor t1 = outputs[0];

  auto get_data = [](const at::Tensor& t) -> std::vector<float> {
    const float* base = t.data_ptr<float>();
    return std::vector<float>(base, base + t.numel());
  };
  std::vector<float> t0_data = get_data(t0.cpu());
  std::vector<float> t1_data = get_data(t1.cpu());
  EXPECT_EQ(t0_data, t1_data)
      << "Although t1 is logically a transpose of t0, their underlying data "
      << "should be the same due to setAllocationDomain. They can even be "
      << "alias.";
}

TEST_F(NVFuserTest, AllocationDomainContiguityIssue1021) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(2)
                 .shape({-1, -1})
                 .contiguity({false, true})
                 .strideOrder({0, 1})
                 .build();
  fusion->addInput(tv0);

  auto s0 = IrBuilder::create<Val>(5, DataType::Float);
  auto tv1 = add(tv0, s0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8, 8}, options).as_strided({4, 8}, {1, 8});
  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, AllocationDomainContiguityForBroadcast) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(2)
                 .shape({1, 1})
                 .contiguity({std::nullopt, std::nullopt})
                 .strideOrder({0, 1})
                 .build();
  fusion->addInput(tv0);

  auto s0 = IrBuilder::create<Val>(5, DataType::Float);
  auto tv1 = add(tv0, s0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options).as_strided({1, 1}, {0, 3});
  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, AllocationDomainContiguityForExplicitBroadcast) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(3)
                 .shape({-1, -1, -1})
                 .contiguity({true, true, std::nullopt})
                 .expanded({true, false, false})
                 .strideOrder({0, 1, 2})
                 .build();
  fusion->addInput(tv0);

  auto s0 = IrBuilder::create<Val>(5, DataType::Float);
  auto tv1 = add(tv0, s0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 8}, options).as_strided({3, 8, 4}, {0, 1, 8});
  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Test that allocation domain can be used to vectorize overlapping tensors,
// by making the allocation domain deviate from the stride order. Note that
// this test is only a demo "hey, we can do this", instead of checking for
// a real use case. Supporting overlapping tensor is a gray area for framework,
// and we are not actively using the trick in this test to generate a better
// kernel for overlapping tensors. The only reason why this test exists is
// because I think it is a good sign that we have a good design (a good design
// automatically supports all kinds of use cases, even those that we don't have
// an active plan to support on).
TEST_F(AllocationDomainTest, VectorizeOverlappingTensor) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // According to the stride order below, the allocation domain should be the
  // same as the root domain. However, here we intentionally make the allocation
  // domain [axis(1), axis(0), axis(2)] because doing so allows us to vectorize
  // by 4.
  tv0->setAllocationDomain(
      {tv0->axis(1), tv0->axis(0), tv0->axis(2)}, {false, true, true});

  for (auto tv : {tv2, tv1}) {
    // [I0, I1, I2]
    tv->reorder({{0, 1}});
    // [I1, I0, I2]
    tv->merge(0);
    // [I1*I0, I2]
    tv->merge(0);
    // [I1*I0*I2]
    tv->split(0, 4);
    // [I1*I0*I2/4, 4]
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  // Note that the stride of the second dimension of the input tensor must be a
  // multiple of 4, otherwise we will have misaligned address access.
  at::Tensor t0 =
      at::randn({4 * 5 * 7}).cuda().as_strided({4, 5, 7}, {7, 4, 1});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, Issue1290_ContiguityWasMissing) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .ndims(2)
                       .dtype(DataType::Float)
                       .contiguity({false, true})
                       .shape({-1, -1})
                       .build();
  fusion->addInput(in);
  TensorView* out1 = permute(in, {1, 0});
  fusion->addOutput(out1);
  TensorView* out2 = add(out1, fusion->oneVal());
  fusion->addOutput(out2);

  at::Tensor in_tensor = at::randn({2 * 4}).cuda().as_strided({2, 3}, {4, 1});

  FusionExecutorCache fec(std::move(fusion));
  fec.runFusionWithInputs({in_tensor});

  // The initial issue was detected in the pointwise scheduler, so I added these
  // checks to make sure it's a valid regression test. The transpose scheduler
  // could accept this but decided not to because of a small problem size.
  const std::vector<SegmentedGroup*>& groups =
      fec.getMostRecentKernelRuntime()->fusionSegments()->groups();
  ASSERT_EQ(groups.size(), 1);
  SegmentedGroup* group = groups[0];
  EXPECT_EQ(group->heuristic(), ScheduleHeuristic::PointWise);
}

TEST_F(AllocationDomainTest, Issue1290_ReplayCasPFailedDueToDifferentRanks) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = sum(in, {1});
  fusion.addInput(in);
  fusion.addOutput(out);

  out->setAllocationDomain({out->axis(0), out->axis(1)}, true);
  out->cacheBefore();

  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  FusionExecutor fe;
  fe.compileFusion(&fusion, {in_tensor});
  at::Tensor out_tensor = fe.runFusion({in_tensor})[0];
  EXPECT_THAT(out_tensor.sizes(), ElementsAre(2));
}

// This test is meant to verify that trivial stride order is dropped by
// TensorViewBuilder. See issue: https://github.com/NVIDIA/Fuser/issues/1399
TEST_F(AllocationDomainTest, TrivialStrideOrderTensorViewBuilder) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = TensorViewBuilder().ndims(2).strideOrder({0, 1}).build();
  EXPECT_TRUE(tv0->hasAllocation());
  // trivial stride order would be dropped by TensorViewbuilder
  tv0 = TensorViewBuilder().ndims(2).strideOrder({1, 0}).build();
  // confirming that stride order is dropped and allocation domain is empty
  EXPECT_TRUE(!tv0->hasAllocation());
}

TEST_F(AllocationDomainTest, Bug) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* permute_out = permute(in, {1, 2, 0}); // {3, 5, 2}
  permute_out = segment_set(permute_out);
  TensorView* add_out = add(permute_out, permute_out);

  fusion->addInput(in);
  fusion->addOutput(add_out);

  permute_out->setAllocationDomain(
      {permute_out->axis(2), permute_out->axis(0), permute_out->axis(1)}, true);

  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  FusionExecutorCache fec(std::move(fusion));
  fec.runFusionWithInputs({in_tensor});
}

} // namespace nvfuser
