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
#include <host_ir/jit.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <random>

namespace nvfuser {

namespace hir {

using HostIrJitTest = NVFuserTest;
// Build with: python setup.py install --build-with-host-ir-jit
// Run with: NVFUSER_ENABLE=host_ir_lowering ./bin/test_host_ir_jit
// --gtest_filter=HostIrJitTest.TestJITAtenCall

TEST_F(HostIrJitTest, LaunchKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(2);
  fusion.addInput(in);

  TensorView* out = set(in);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  auto ke = std::make_unique<KernelExecutor>();
  ke->setGroupId(0);
  ke->compile(&fusion, {t0});

  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  hic->addKernelExecutor(std::move(ke));

  IrCloner ir_cloner(hic.get());
  auto hic_in = ir_cloner.clone(in);
  auto hic_out = ir_cloner.clone(out);

  hic->addInput(hic_in);
  hic->addOutput(hic_out);

  auto allocate = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  auto launch_kernel = IrBuilder::create<LaunchKernel>(
      0,
      LaunchParams(),
      CompileParams(),
      std::vector<Val*>{hic_in},
      std::vector<Val*>{hic_out},
      cache_id);

  hic->pushBackTopLevelExprs(allocate);
  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrJit jit(std::move(hic));
  KernelArgumentHolder args;
  args.setCacheId(1);
  args.push(t0);

  auto outputs = jit.runWithInputs(args);

  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

TEST_F(HostIrJitTest, HostIrMatmulOut1) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = makeContigTensor(3);
  TensorView* tv2 = matmul(tv0, tv1);

  hic->addInput(tv0);
  hic->addInput(tv1);
  hic->addOutput(tv2);

  hic->pushBackTopLevelExprs(tv2->definition());

  HostIrJit hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  auto t0 = at::randn({32, 64, 128}, options);
  auto t1 = at::randn({32, 128, 256}, options);
  std::unordered_map<Val*, PolymorphicValue> concrete_input_buffers = {
      {tv0, t0}, {tv1, t1}};

  auto output = hie.runWithInput(concrete_input_buffers)[0].as<at::Tensor>();

  // validate
  auto ref_output = at::matmul(t0, t1);

  EXPECT_TRUE(ref_output.allclose(output));
}


TEST_F(HostIrJitTest, HostIrMatmulOut2) {
  constexpr int64_t H = 32;
  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 256;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = makeContigTensor(3);
  TensorView* tv2 = makeContigTensor(3);
  auto* matmul = IrBuilder::create<MatmulOp>(tv2, tv0, tv1);

  hic->addInput(tv0);
  hic->addInput(tv1);
  hic->addInput(tv2);
  hic->addOutput(tv2);

  hic->pushBackTopLevelExprs(matmul);

  HostIrJit hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  at::Tensor t0 = at::randn({H, M, K}, options);
  at::Tensor t1 = at::randn({H, K, N}, options);
  at::Tensor t2 = at::randn({H, M, N}, options);
  std::unordered_map<Val*, PolymorphicValue> concrete_input_buffers = {
      {tv0, t0}, {tv1, t1}, {tv2, t2}};

  hie.runWithInput(concrete_input_buffers);

  // validate
  auto ref_output = at::matmul(t0, t1);

  EXPECT_TRUE(ref_output.allclose(t2));
}

TEST_F(HostIrJitTest, HostIrLinearOut) {
  constexpr int64_t B = 32;
  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 256;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* in = makeContigTensor(3);
  TensorView* weight = makeContigTensor(2);
  TensorView* bias = makeContigTensor(1);
  TensorView* out = makeContigTensor(3);

  auto linear_op = IrBuilder::create<LinearOp>(out, in, weight, bias);

  hic->addInput(in);
  hic->addInput(weight);
  hic->addInput(bias);
  hic->addInput(out);

  hic->pushBackTopLevelExprs(linear_op);

  HostIrJit hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  auto in_at = at::randint(5, {B, M, K}, options);
  auto weight_at = at::randint(5, {N, K}, options);
  auto bias_at = at::randint(5, {N}, options);
  auto out_at = at::empty({B, M, N}, options);
  std::unordered_map<Val*, PolymorphicValue> concrete_input_buffers = {
      {hie.inputs().at(0), in_at},
      {hie.inputs().at(1), weight_at},
      {hie.inputs().at(2), bias_at},
      {hie.inputs().at(3), out_at}};

  hie.runWithInput(concrete_input_buffers);

  // validate
  auto ref_output = at::linear(in_at, weight_at, bias_at);

  EXPECT_TRUE(ref_output.allclose(out_at));
}

TEST_F(HostIrJitTest, Deallocate) {
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

  HostIrJit hie(std::move(hic));

  hie.runWithInput({});

  EXPECT_EQ(memoryAllocated(device_index), 0);
}

TEST_F(HostIrJitTest, Allocation) {
  const std::vector<int64_t> sizes = {8, 64};

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* tv = makeConcreteTensor(sizes);
  tv->setMemoryType(MemoryType::Global);
  auto* allocate = IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
  hic->addOutput(tv);
  hic->pushBackTopLevelExprs(allocate);

  HostIrJit hie(std::move(hic));

  auto outputs = hie.runWithInput({});

  EXPECT_EQ(sizes, outputs[0].as<at::Tensor>().sizes());
}


TEST_F(HostIrJitTest, Set_Kernel) {
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

TEST_F(HostIrJitTest, Sum_Kernel) {
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


}

} // namespace nvfuser
