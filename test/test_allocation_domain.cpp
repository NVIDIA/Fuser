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

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationDomainTest : public NVFuserTest {};

// A global->shared->global copy kernel, shared memory allocated transposed to
// avoid bank conflict.
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

  std::vector<IterDomain*> tv1_transposed = {tv1->axis(1), tv1->axis(0)};
  tv1->setAllocationDomain(tv1_transposed, true);

  ASSERT_TRUE(fusion.bankConflictInfo().empty());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 4d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC4d_CUDA) {
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

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 1d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC1d_CUDA) {
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

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 2d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC2d_CUDA) {
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

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Reshape and transpose a 3d tensor into an NHWC tensor with a 3d allocation
// domain in fusion output.
TEST_F(AllocationDomainTest, Tensor3d_To_NHWC3d_CUDA) {
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
TEST_F(AllocationDomainTest, Tensor3d_To_NHWC4d_FwdBwd_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC4d_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain of the output view the output as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC1d_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view both the input and the output as a 1d tensors.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC1d_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 2d tensor of shape [N*H/8, 8*W*C], and
// view the output as a 2d tensor of shape [N*H*W*C/4, 4]
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
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d_CUDA, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheBefore_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC2d_To_NHWC2d_CUDA, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheBefore_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d_CUDA, but does a cacheAfter
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheAfter_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// NOT similar to NHWC2d_To_NHWC2d_CUDA, because cacheAfter requires the
// allocation tensor to be between rFactor domain and leaf domain, which is not
// the case for NHWC2d_To_NHWC2d_CUDA
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheAfter_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
          "merging of discontiguous dimensions is not allowed in allocation domain")));

  auto cg_outputs = fe.runFusion({t0});

  ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
