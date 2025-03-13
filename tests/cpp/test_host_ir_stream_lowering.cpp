// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <host_ir/lower.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <kernel_ir.h>
#include <multidevice/executor.h>
#include <ops/all_ops.h>
#include <preseg_passes/stream_parallel_type.h>
#include <tests/cpp/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

namespace hir {

using HirLowerStreamTest = NVFuserTest;

TEST_F(HirLowerStreamTest, InputsAreNotStreamParallelized) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv = makeContigTensor(2);
  hic->addInput(tv);
  tv->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(preseg_passes::OptimizationPass<
                   preseg_passes::StreamParallelType>::runPass(hic.get()));
}

TEST_F(HirLowerStreamTest, Split) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv1->split(0, 2);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(preseg_passes::OptimizationPass<
                   preseg_passes::StreamParallelType>::runPass(hic.get()));
}

TEST_F(HirLowerStreamTest, Merge) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv1->merge(0, 1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(preseg_passes::OptimizationPass<
                   preseg_passes::StreamParallelType>::runPass(hic.get()));
}

TEST_F(HirLowerStreamTest, SingleUnaryOp) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, SingleUnaryOpNonOutermost) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(1)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, SingleBinaryOp) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  TensorView* tv2 = add(tv0, tv1);
  hic->addInput(tv0);
  hic->addInput(tv1);
  hic->addOutput(tv2);
  hic->pushBackTopLevelExprs(tv2->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor tv0_input = at::rand({4, 4}, options);
  at::Tensor tv1_input = at::rand({4, 4}, options);
  // std::unordered_map<Val*, PolymorphicValue> inputs = {{tv0, input}};
  auto output = hie.runWithInput({{tv0, tv0_input}, {tv1, tv1_input}})[0].as<at::Tensor>();
  auto expected_output = tv0_input + tv1_input;
  EXPECT_TRUE(output.equal(expected_output)) << "Output: " << output << "Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, TwoUnaryOps) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  hic->addInput(tv0);
  hic->addOutput(tv2);
  hic->pushBackTopLevelExprs(tv1->definition());
  hic->pushBackTopLevelExprs(tv2->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 3);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, ThreeUnaryOpsWithDisjointsForLoops) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  hic->addInput(tv0);
  hic->addOutput(tv3);
  hic->pushBackTopLevelExprs(tv1->definition());
  hic->pushBackTopLevelExprs(tv2->definition());
  hic->pushBackTopLevelExprs(tv3->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv3->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 5);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<LoadStoreOp>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(4)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, ReductionUnsupported) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {0});
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get()));
}

TEST_F(HirLowerStreamTest, Reduction) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = sum(tv0, {2});
  hic->addInput(tv0);
  hic->addOutput(tv1);
  hic->pushBackTopLevelExprs(tv1->definition());
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8, 2}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = input.sum(2);
  EXPECT_TRUE(output.equal(expected_output))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, Matmul_M) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  hic->addInput(a);
  hic->addInput(b);
  hic->addOutput(c);
  hic->pushBackTopLevelExprs(c->definition());
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, BatchedMatmul) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  hic->addInput(a);
  hic->addInput(b);
  hic->addOutput(c);
  hic->pushBackTopLevelExprs(c->definition());
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t B=16, M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({B, M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, Matmul_N) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  hic->addInput(a);
  hic->addInput(b);
  hic->addOutput(c);
  hic->pushBackTopLevelExprs(c->definition());
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(1)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 2);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, Matmul_K) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  hic->addInput(a);
  hic->addInput(b);
  hic->addOutput(c);
  hic->pushBackTopLevelExprs(c->definition());
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(-1)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(preseg_passes::OptimizationPass<preseg_passes::StreamParallelType>::runPass(
      hic.get()));
}

} // namespace hir

using MultiDeviceExecutorLowerStreamTest = NVFuserFixtureParamTest<bool>;

TEST_P(MultiDeviceExecutorLowerStreamTest, InputsAreNotStreamParallelized) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv = makeContigTensor(2);
  fusion->addInput(tv);
  tv->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Split) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->split(0, 2);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Merge) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->merge(0, 1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_P(MultiDeviceExecutorLowerStreamTest, SingleUnaryOp) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  // tv0->setMemoryType(MemoryType::Global);
  // tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, SingleUnaryOpNonOutermost) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(1)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, SingleBinaryOp) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  TensorView* tv2 = add(tv0, tv1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);

  at::Tensor tv0_input = at::rand({4, 4}, options);
  at::Tensor tv1_input = at::rand({4, 4}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({tv0_input, tv1_input}))[0].as<at::Tensor>();
  auto expected_output = tv0_input + tv1_input;
  EXPECT_TRUE(output.equal(expected_output)) << "Output: " << output << "Expected: " << expected_output;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, TwoUnaryOps) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 3);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, ThreeUnaryOpsWithDisjointsForLoops) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  fusion->addInput(tv0);
  fusion->addOutput(tv3);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv3->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 5);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<LoadStoreOp>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(4)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, ReductionUnsupported) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Reduction) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = sum(tv0, {2});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv0->setMemoryType(MemoryType::Global);
  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8, 2}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = input.sum(2);
  EXPECT_TRUE(output.equal(expected_output))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Matmul_M) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  constexpr int64_t M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, BatchedMatmul) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  constexpr int64_t B=16, M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({B, M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Matmul_N) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(1)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  constexpr int64_t M=8, K=4, N=2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_P(MultiDeviceExecutorLowerStreamTest, Matmul_K) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  a->setMemoryType(MemoryType::Global);
  b->setMemoryType(MemoryType::Global);
  c->setMemoryType(MemoryType::Global);
  c->axis(-1)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MultiDeviceExecutorLowerStreamTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) -> std::string {
      return info.param ? "useFusionExecutorCache"
                                     : "useFusionExecutor";
    });


} // namespace nvfuser
