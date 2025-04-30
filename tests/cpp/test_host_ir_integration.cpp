// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gmock/gmock-more-matchers.h>

#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using HostIrEvaluatorTest = NVFuserTest;

// This test manually creates a HostIrContainer with LaunchKernels and runs it
// using HostIrEvaluator.
TEST_F(HostIrEvaluatorTest, LaunchKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(2);
  fusion.addInput(in);

  TensorView* out = set(in);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  auto ke = std::make_unique<KernelExecutor>();
  ke->compile(&fusion, {t0});

  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  hic->setKernelExecutor(0, std::move(ke));

  IrCloner ir_cloner(hic.get());
  auto hic_in = ir_cloner.clone(in);
  auto hic_out = ir_cloner.clone(out);

  hic->addInput(hic_in);
  hic->addOutput(hic_out);

  auto launch_kernel = IrBuilder::create<LaunchKernel>(
      0,
      LaunchParams(),
      CompileParams(),
      std::vector<Val*>{hic_in},
      std::vector<Val*>{hic_out});

  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({{hic_in, t0}});

  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

class HostIrIntegrationTest : public NVFuserTest {
 protected:
  HostIrIntegrationTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

TEST_F(HostIrIntegrationTest, Set_Kernel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrIntegrationTest, Sum_Kernel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

// Same as AliasTest.ViewPermute but via host IR. This test verifies that Exprs
// in SchedulerType::ExprEval segments are cloned to the top level and produce
// aliases.
TEST_F(HostIrIntegrationTest, ViewPermute_ExprEval) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  out = permute(out, {1, 0});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(HostIrIntegrationTest, Deallocate) {
  const std::vector<int64_t> sizes = {8, 64};
  c10::DeviceIndex device_index = 0;

  resetPeakMemoryStats(device_index);
  at::cuda::clearCublasWorkspaces();
  nvfuser::releaseZeroedMemory();
  ASSERT_EQ(memoryAllocated(device_index), 0)
      << "Previous tests leaked memory.";

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  for (int i = 0; i < 10; i++) {
    TensorView* tv = makeConcreteTensor(sizes);
    tv->setMemoryType(MemoryType::Global);
    auto* allocate = IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
    auto* deallocate = IrBuilder::create<Deallocate>(tv);

    hic->pushBackTopLevelExprs(allocate);
    hic->pushBackTopLevelExprs(deallocate);
  }

  HostIrEvaluator hie(std::move(hic));

  hie.runWithInput({});

  EXPECT_EQ(memoryAllocated(device_index), 0);
}

TEST_F(HostIrIntegrationTest, InsertDeallocations) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* t0 = add(in, in);
  auto t1 = segment_set(t0);
  TensorView* t2 = add(t1, t1);
  auto t3 = segment_set(t2);
  TensorView* t4 = add(t3, t0);
  TensorView* out = add(t4, t4);
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->getHostIrEvaluator().canRun(), "");
  auto hicExprs =
      runtime->getHostIrEvaluator().getHostIrContainer().topLevelExprs();
  using testing::Contains;
  using testing::IsTrue;
  using testing::Property;
  EXPECT_THAT(
      hicExprs,
      Contains(Property(&Expr::isA<Deallocate>, IsTrue()))
          .Times(testing::Eq(4)))
      << "host ir container should have 4 deallocate ops";

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

} // namespace hir

} // namespace nvfuser
