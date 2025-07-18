// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>
#include <scheduler/tools/inlining.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <torch/torch.h>

namespace nvfuser {

using AllocationDomainTest = NVFuserTest;

using ::testing::ElementsAre;

// A global->shared->global copy kernel, shared memory allocated transposed to
// avoid bank conflict.
TEST_F(AllocationDomainTest, TransposedIntermediate) {
  Fusion fusion;
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
  ASSERT_EQ(read, std::vector<int64_t>{32});
  ASSERT_EQ(write, std::vector<int64_t>{32});

  std::vector<IterDomain*> tv1_transposed = {tv1->axis(1), tv1->axis(0)};
  tv1->setAllocationDomain(tv1_transposed, true);

  ASSERT_TRUE(fusion.bankConflictInfo().empty());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32, 32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 4d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC4d) {
  Fusion fusion;
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
  tv1->flatten();
  // [N*H*W*C]
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 1d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC1d) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->flatten();
  tv1->setAllocationDomain({tv1->axis(0)}, true);
  // [N*H*W*C]
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int n = 31, h = 64, w = 103, c = 21;

  at::Tensor t0 = at::randn({n, c, h, w}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 2d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC2d) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Reshape and transpose a 3d tensor into an NHWC tensor with a 3d allocation
// domain in fusion output.
TEST_F(AllocationDomainTest, Tensor3d_To_NHWC3d) {
  Fusion fusion;
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
  tv1->commitLeafToLogical();
  // [N, C, H, W]

  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->flatten();
  // [N*H*W*C]
  tv1->split(0, 128);
  // [N*H*W*C/128, 128]
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, TIDx]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({n1, n2, h * w * c}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

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
  Fusion fusion;
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
  tv1->commitLeafToLogical();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

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
  Fusion fusion;
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
  tv1->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC4d) {
  Fusion fusion;
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
  tv0->commitLeafToLogical();

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  // [N, C, H, W]
  tv1->reorder({{1, -1}});
  // [N, H, W, C]
  tv1->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not "
          "allowed in allocation domain")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain of the output view the output as a 1d tensor.
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC1d) {
  Fusion fusion;
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view both the input and the output as a 1d tensors.
TEST_F(AllocationDomainTest, NHWC1d_To_NHWC1d) {
  Fusion fusion;
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
  tv0->commitLeafToLogical();

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// A global->global copy kernel where both inputs are NHWC memory format. The
// allocation domain view the input as a 2d tensor of shape [N*H/8, 8*W*C], and
// view the output as a 2d tensor of shape [N*H*W*C/4, 4]
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d) {
  Fusion fusion;
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
  tv0->commitLeafToLogical();

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not "
          "allowed in allocation domain")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheBefore) {
  Fusion fusion;
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
    tv->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC2d_To_NHWC2d, but does a cacheBefore
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheBefore) {
  Fusion fusion;
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
  tv0->commitLeafToLogical();

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not "
          "allowed in allocation domain")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheAfter
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheAfter) {
  Fusion fusion;
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
    tv->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// NOT similar to NHWC2d_To_NHWC2d, because cacheAfter requires the
// allocation tensor to be between rFactor domain and loop domain, which is not
// the case for NHWC2d_To_NHWC2d
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheAfter) {
  Fusion fusion;
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Merging of discontiguous dimensions")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC4d_To_NHWC4d, but does a cacheFork
TEST_F(AllocationDomainTest, NHWC4d_To_NHWC4d_cacheFork) {
  Fusion fusion;
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
    tv->flatten();
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to NHWC2d_To_NHWC2d, but does a cacheFork
TEST_F(AllocationDomainTest, NHWC2d_To_NHWC2d_cacheFork) {
  Fusion fusion;
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
  tv0->commitLeafToLogical();

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
      tv3->getLogicalDomain().at(0),
      tv3->getLogicalDomain().at(2),
      tv3->getLogicalDomain().at(3),
      tv3->getLogicalDomain().at(1)};

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.run({t0_wrong_format}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "splitting one dimension into discontiguous dimensions is not "
          "allowed in allocation domain")));

  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, VectorizationIssue902) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> shape({16, 16, 512, 64});

  auto tv0 = makeContigTensor(4);
  fusion->addInput(tv0);

  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  std::vector<nvfuser::IterDomain*> alloc_domain;
  alloc_domain.push_back(tv1->axis(0));
  alloc_domain.push_back(tv1->axis(2));
  alloc_domain.push_back(tv1->axis(3));
  alloc_domain.push_back(tv1->axis(1));
  tv1->setAllocationDomain(alloc_domain, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().equal(t0));
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
  auto outputs = executor_cache.runFusionWithInputs({t0});
  at::Tensor t1 = outputs[0].as<at::Tensor>();

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

TEST_F(AllocationDomainTest, ContiguityIssue1021) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

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
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, ContiguityForBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

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
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, ContiguityForExplicitBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

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
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.add(5.0);
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
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
  Fusion fusion;
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

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

  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.runFusionWithInputs({in_tensor});

  // The initial issue was detected in the pointwise scheduler, so I added these
  // checks to make sure it's a valid regression test. The transpose scheduler
  // could accept this but decided not to because of a small problem size.
  const std::vector<SegmentedGroup*>& groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();
  ASSERT_EQ(groups.size(), 1);
  SegmentedGroup* group = groups[0];
  EXPECT_EQ(group->schedulerType(), SchedulerType::PointWise);
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
  KernelExecutor ke;
  ke.compile(&fusion, {in_tensor});
  at::Tensor out_tensor = ke.run({in_tensor})[0].as<at::Tensor>();
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

TEST_F(AllocationDomainTest, Issue1524) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* permute_out = permute(in, {1, 0});
  permute_out = segment_set(permute_out);
  TensorView* add_out = add(permute_out, permute_out);

  fusion->addInput(in);
  fusion->addOutput(permute_out);
  fusion->addOutput(add_out);

  permute_out->setAllocationDomain(
      {permute_out->axis(1), permute_out->axis(0)}, true);

  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.runFusionWithInputs({in_tensor});
}

TEST_F(AllocationDomainTest, EmptyAllocationDomainApi) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = TensorViewBuilder().ndims(0).build();
  tv0->setAllocationDomain({}, true);
}

TEST_F(AllocationDomainTest, ReductionSchedulerIssue1895) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  long x = 2L, y = 8L, z = 8L, w = 16L, h = 512L;
  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .shape({-1, -1, -1, -1, -1})
                 .contiguity({true, true, true, true, true})
                 .strideOrder({4, 3, 2, 0, 1})
                 .build();
  fusion->addInput(tv0);
  auto tv1 = full(
      {IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(y),
       IrBuilder::create<Val>(z),
       IrBuilder::create<Val>(w),
       IrBuilder::create<Val>(h)},
      fusion->oneVal(),
      DataType::Float);
  auto tv2 = mul(tv0, tv1);
  auto tv3 = sum(tv2, {2, 4});
  fusion->addOutput(tv3);
  std::vector<IterDomain*> tv3_dom = {
      tv3->axis(0), tv3->axis(1), tv3->axis(2), tv3->axis(4), tv3->axis(3)};
  tv3->setAllocationDomain(tv3_dom, true);

  // tv1 is a constant tensor, and its domains are constant.
  // Its constant domains are used in ExactMappedExtentSubstitutionPass
  // to substitute the domains of tv0.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 =
      at::randn({x, y, z, w, h}, options)
          .as_strided({x, y, z, w, h}, {w * h * z * y, w * h * z, w * h, 1, w});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.to(at::kDouble).sum({2, 4});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, ReductionVectorization) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  long x = 2L, y = 2L, z = 2L;
  auto tv0 = TensorViewBuilder()
                 .ndims(3)
                 .shape({-1, 1, -1})
                 .contiguity({true, std::nullopt, true})
                 .build();
  fusion->addInput(tv0);
  auto tv1 = TensorViewBuilder()
                 .ndims(3)
                 .shape({-1, -1, -1})
                 .contiguity({true, true, true})
                 .strideOrder({0, 1, 2})
                 .build();
  fusion->addInput(tv1);
  auto s0 = IrBuilder::create<Val>(2);
  auto tv2 = expand(tv0, {s0, s0, s0});
  auto tv3 = mul(tv2, tv1);
  auto tv4 = sum(tv3, {2});
  fusion->addOutput(tv4);
  std::vector<IterDomain*> tv4_dom = {tv4->axis(2), tv4->axis(1), tv4->axis(0)};
  tv4->setAllocationDomain(tv4_dom, true);

  // tv1 is a constant tensor, and its domains are constant.
  // Its constant domains are used in ExactMappedExtentSubstitutionPass
  // to substitute the domains of tv0.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({x, 1, z}, options);
  auto t1 = at::randn({x, y, z}, options).as_strided({x, y, z}, {1, x, x * y});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, ClearReductionIterDomainsPatch) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(3)
                 .shape({-1, 1, -1})
                 .contiguity({true, std::nullopt, true})
                 .build();
  auto tv1 = sum(tv0, {2});

  tv1->setAllocationDomain(
      {tv1->axis(1), tv1->axis(2), tv1->axis(0)},
      {std::nullopt, std::nullopt, true});

  // copy entries from old domain for validation later
  std::vector<IterDomain*> logical_copy = tv1->getLogicalDomain();
  std::vector<IterDomain*> alloc_copy = tv1->getAllocationDomain();
  std::vector<std::optional<bool>> contig_copy = tv1->getContiguity();
  // clear reduction iter domain removed reduction iter domain from both root
  // and allocation domain and adjusting contiguity flag as well
  tv1->clearReductionIterDomains();
  // entry 2 is removed since tv1->axis(2) is a reduction iter domain in tv1's
  // root domain
  EXPECT_THAT(
      tv1->getLogicalDomain(), ElementsAre(logical_copy[0], logical_copy[1]));
  // entry 1 is removed since tv1->axis(2) is a reduction iter domain and tv1's
  // allocation domain looks like {tv1->axis(1), tv1->axis(2), tv1->axis(0)},
  EXPECT_THAT(
      tv1->getAllocationDomain(), ElementsAre(alloc_copy[0], alloc_copy[2]));
  EXPECT_THAT(
      tv1->getContiguity(), ElementsAre(contig_copy[0], contig_copy[2]));
}

TEST_F(AllocationDomainTest, InputAllocationIsSplit_Concrete) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({6});
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  // Ideally, loop should stay the same as logical because a fusion input comes
  // from outside and isn't generated by a loop in the containing kernel (cf.
  // #3479).
  in->split(0, 2);
  in->setAllocationDomain(in->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({6}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, InputAllocationIsSplitReorderContiguous) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({6});
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->split(0, 2);
  in->reorder({{1, 0}});
  // new_contiguity=True is the problem here. After reordering, the two
  // IterDomains are no longer contiguous.
  in->setAllocationDomain(in->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({6}, options);
  EXPECT_THAT(
      [&]() { executor_cache.runFusionWithInputs({in_tensor}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Stride mismatch with contiguity info")));
}

TEST_F(AllocationDomainTest, InputAllocationIsSplitReorderMerge) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({6});
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->split(0, 2);
  in->merge(1, 0);
  in->setAllocationDomain(in->getLoopDomain(), false);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({6}, options);
  EXPECT_THAT(
      [&]() { executor_cache.runFusionWithInputs({in_tensor}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Merging of discontiguous dimensions")));
}

// https://github.com/NVIDIA/Fuser/issues/3480
TEST_F(AllocationDomainTest, InputAllocationIsSplit_Symbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(1);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->split(0, 2);
  in->setAllocationDomain(in->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({6}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, ManualReplayLoopToAllocation) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t x = 2L, y = 12L, z = 16L;
  auto tv0 = makeContigConcreteTensor({x, y, z});
  fusion->addInput(tv0);
  std::vector<IterDomain*> tv0_dom = {tv0->axis(1), tv0->axis(0), tv0->axis(2)};
  tv0->setAllocationDomain(tv0_dom, true);
  auto tv2 = add(tv0, tv0);
  fusion->addOutput(tv2);

  auto tv1 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::Bulk);

  for (auto tv : fusion->allTvs()) {
    // [2, 3, 4, 16]
    tv->split(1, 4);
  }

  inlineSelectedAt({tv1}, tv1, /*reference_pos=*/2);

  // Before fix, we have:

  // T2_s_float[iS6{2}, iS11{3}, iS12{4}, iB8{16}] ca_pos( 2 )
  // logical domain : (iS6{2}, iS7{12}, iB8{16})
  // allocation domain : (iS7{12}, iS6{2}, iB8{16})
  // contiguity: t t t
  //  Split: iS7{12} by factor 4 -> iS11{3}, iS12{4}
  // loop domain : (iS6{2}, iS11{3}, iS12{4}, iB8{16})

  // T2 is computed at pos 2, we don't need to allocate domains iS6{2} and
  // iS11{3} nvFuser tries to exclude these two domains from the allocation
  // domain, however, iS11{3} doesn't exist in the allocation domain, so it's
  // not excluded and this is considered a failed case.

  // To fix, we can reaplay transforms on the allocation domain.
  // How to split the allocation domain?
  // Create AbstractTensor from current allocation domain
  // Apply the same split transformation to the allocation domain
  // Update the allocation domain
  AbstractTensor alloc_tensor(tv1->getAllocationDomain());
  alloc_tensor.split(0, 4);
  tv1->setAllocationDomain(alloc_tensor.as<IterDomain*>(), true);
  // after this change to allocation domain, we have:
  // T2_s_float[iS6{2}, iS11{3}, iS12{4}, iB8{16}] ca_pos( 2 )
  // logical domain : (iS6{2}, iS7{12}, iB8{16})
  // allocation domain : (iS15{3}, iS16{4}, iS6{2}, iB8{16})
  // contiguity: t t t t
  //  Split: iS7{12} by factor 4 -> iS15{3}, iS16{4}
  //  Split: iS7{12} by factor 4 -> iS11{3}, iS12{4}
  // loop domain : (iS6{2}, iS11{3}, iS12{4}, iB8{16})

  // Based on loop domain and compute pos, we don't need to allocate iS6{2} and
  // iS11{3}. However, the corresponding allocation domain of iS11{3} is
  // iS15{3}. How do we map them in getAllocationDomainsAndContiguity()? use
  // IdModel if pointer comparison fails IdModel maintains a disjointValSets
  // id_sets: disjoint sets{
  //   { iS3{2}; iS6{2}; iS0{2} }
  //   { iS4{12}; iS7{12}; iS1{12} }
  //   { iS13{3}; iS11{3}; iS15{3}; iS9{3} }
  //   { iS14{4}; iS12{4}; iS16{4}; iS10{4} }
  //   { iS5{16}; iB8{16}; iS2{16} }
  // }
  // where iS11{3} and iS15{3} are in the same set.

  fusion->print();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  // shape: (x, y, z), alloc: (y, x, z), stride: (z, x * z, 1)
  auto t0 = at::randn({x, y, z}, options).as_strided({x, y, z}, {z, x * z, 1});
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});
  auto outputs = ke.run({t0});
  testValidate(fusion.get(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(AllocationDomainTest, selfReplayLoopToAllocation) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t x = 2L, y = 12L, z = 16L;
  auto tv0 = makeContigConcreteTensor({x, y, z});
  fusion->addInput(tv0);
  std::vector<IterDomain*> tv0_dom = {tv0->axis(1), tv0->axis(0), tv0->axis(2)};
  tv0->setAllocationDomain(tv0_dom, true);
  auto tv2 = add(tv0, tv0);
  fusion->addOutput(tv2);

  auto tv1 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::Bulk);

  for (auto tv : fusion->allTvs()) {
    // [2, 3, 4, 16]
    tv->split(1, 4);
  }

  inlineSelectedAt({tv1}, tv1, /*reference_pos=*/2);
  selfReplayLoopToAllocation(tv1);

  fusion->print();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  // shape: (x, y, z), alloc: (y, x, z), stride: (z, x * z, 1)
  auto t0 = at::randn({x, y, z}, options).as_strided({x, y, z}, {z, x * z, 1});
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});
  auto outputs = ke.run({t0});
  testValidate(fusion.get(), outputs, {t0}, __LINE__, __FILE__);
}
} // namespace nvfuser
