// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <host_ir/host_ir.h>
#include <host_ir/lower.h>
#include <host_ir/pass/stream_parallel_type.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <kernel_ir.h>
#include <multidevice/executor.h>
#include <ops/all_ops.h>
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

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
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

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
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

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
}

TEST_F(HirLowerStreamTest, SingleSetOp) {
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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, SingleSetOpNonOutermost) {
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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor tv0_input = at::rand({4, 4}, options);
  at::Tensor tv1_input = at::rand({4, 4}, options);
  auto output = hie.runWithInput({{tv0, tv0_input}, {tv1, tv1_input}})[0]
                    .as<at::Tensor>();
  auto expected_output = tv0_input + tv1_input;
  EXPECT_TRUE(output.equal(expected_output))
      << "Output: " << output << "Expected: " << expected_output;
}

TEST_F(HirLowerStreamTest, TwoSetOps) {
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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 5);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(4)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output = hie.runWithInput({{tv0, input}})[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(HirLowerStreamTest, ThreeSetOpsWithDisjointsForLoops) {
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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 9);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(4)->isA<LoadStoreOp>());
  EXPECT_TRUE(hic->topLevelExprs().at(5)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(6)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(7)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(8)->isA<ForLoop>());

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

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output =
      hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t B = 16, M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({B, M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output =
      hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

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

  hir_pass::StreamParallelType().runPass(hic.get());

  EXPECT_EQ(hic->topLevelExprs().size(), 4);
  EXPECT_TRUE(hic->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(hic->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(hic->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(hic->topLevelExprs().at(3)->isA<ForLoop>());

  HostIrEvaluator hie(std::move(hic));

  constexpr int64_t M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output =
      hie.runWithInput({{a, a_aten}, {b, b_aten}})[0].as<at::Tensor>();

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

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
}

// We don's support PostOnStream because it does not support well pre-allocated
// outputs. There is no strong motivation to support PostOnStream
TEST_F(HirLowerStreamTest, DoNotSupportPostOnStream) {
  const std::vector<int64_t> input_sizes = {4, 8, 32};
  const std::vector<int64_t> output_sizes = {
      input_sizes.at(1), input_sizes.at(2)};

  auto get_fusion = [input_sizes]() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeConcreteTensor(input_sizes);
    auto tv1 = add(tv0, tv0);
    auto tv2 = sum(tv1, {0});
    fusion->addInput(tv0);
    fusion->addOutput(tv2);
    return fusion;
  };

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto host_unit = IrBuilder::create<HostUnit>(get_fusion());

  IrCloner ir_cloner(hic.get());
  TensorView* input =
      ir_cloner.clone(host_unit->fusion_to_execute()->inputs().at(0))
          ->as<TensorView>();
  TensorView* output =
      ir_cloner.clone(host_unit->fusion_to_execute()->outputs().at(0))
          ->as<TensorView>();

  std::vector<Val*> inputs = {input};
  std::vector<Val*> outputs = {output};
  auto post_on_stream =
      IrBuilder::create<PostOnStream>(host_unit, inputs, outputs);

  hic->pushBackTopLevelExprs(post_on_stream);

  hic->addInput(input);
  hic->addOutput(output);

  output->axis(-1)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(hir_pass::StreamParallelType().runPass(hic.get()));
}

} // namespace hir

using MultiDeviceExecutorLowerStreamTest = NVFuserTest;

TEST_F(MultiDeviceExecutorLowerStreamTest, InputsAreNotStreamParallelized) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv = makeContigTensor(2);
  fusion->addInput(tv);
  tv->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_F(MultiDeviceExecutorLowerStreamTest, Split) {
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

TEST_F(MultiDeviceExecutorLowerStreamTest, Merge) {
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

TEST_F(MultiDeviceExecutorLowerStreamTest, SingleSetOp) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, SingleSetOpNonOutermost) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->axis(1)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, SingleBinaryOp) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  TensorView* tv2 = add(tv0, tv1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor tv0_input = at::rand({4, 4}, options);
  at::Tensor tv1_input = at::rand({4, 4}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({tv0_input, tv1_input}))[0]
          .as<at::Tensor>();
  auto expected_output = tv0_input + tv1_input;
  EXPECT_TRUE(output.equal(expected_output))
      << "Output: " << output << "Expected: " << expected_output;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, TwoSetOps) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 5);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(4)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, ThreeSetOpsWithDisjointsForLoops) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  fusion->addInput(tv0);
  fusion->addOutput(tv3);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv3->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 9);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(4)->isA<LoadStoreOp>());
  EXPECT_TRUE(container->topLevelExprs().at(5)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(6)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(7)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(8)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  EXPECT_TRUE(output.equal(input))
      << "Output: " << output << " Expected: " << input;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, ReductionUnsupported) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

TEST_F(MultiDeviceExecutorLowerStreamTest, Reduction) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = sum(tv0, {2});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::rand({4, 8, 2}, options);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = input.sum(2);
  EXPECT_TRUE(output.equal(expected_output))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, Matmul_M) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  c->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  constexpr int64_t M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0]
                    .as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, BatchedMatmul) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  c->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  constexpr int64_t B = 16, M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({B, M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0]
                    .as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, Matmul_N) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  c->axis(1)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), Communicator::getInstance());

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 4);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<hir::GetCurrentStream>());
  EXPECT_TRUE(container->topLevelExprs().at(2)->isA<ForLoop>());
  EXPECT_TRUE(container->topLevelExprs().at(3)->isA<ForLoop>());

  constexpr int64_t M = 8, K = 4, N = 2;
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor a_aten = at::rand({M, K}, options);
  at::Tensor b_aten = at::rand({K, N}, options);
  auto output = executor.runWithInput(KernelArgumentHolder({a_aten, b_aten}))[0]
                    .as<at::Tensor>();

  torch::cuda::synchronize();
  auto expected_output = at::matmul(a_aten, b_aten);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << " Expected: " << expected_output;
}

TEST_F(MultiDeviceExecutorLowerStreamTest, Matmul_K) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* a = makeContigTensor(2);
  TensorView* b = makeContigTensor(2);
  TensorView* c = matmul(a, b);
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);
  c->axis(-1)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

// We only support Stream parallel type on ops that support pre-allocated
// output, which means they need a special handle in HostIrEvaluator and they
// need to be lowered as a Host Ir Op in the TopLevelExpression, no a
// PostOnStream(HostUnit(.)) See HostIrLower::isLoweredAsStandaloneHostOp and
// the test HirLowerStreamTest.DoNotSupportPostOnStream
TEST_F(MultiDeviceExecutorLowerStreamTest, DoNotSupportPostOnStream) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 =
      abs(tv0); // arbitrary example of an unsupported op. There is no deep
                // reason why we not support it -- if needed we could widen the
                // support. But I wanna make sure that an unsupported op do not
                // silently fails
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  EXPECT_ANY_THROW(
      MultiDeviceExecutor(std::move(fusion), Communicator::getInstance()));
}

} // namespace nvfuser
