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
#include <host_ir/compile_to_llvm.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using testing::Contains;
using HostIrLLVMTest = NVFuserTest;
// Build with: python setup.py install --build-with-llvm
// NVFUSER_ENABLE=host_ir_lowering ./bin/test_llvm_compile
// --gtest_filter=HostIrLLVMTest.TestLLVMJITAtenCall
TEST_F(HostIrLLVMTest, TestLLVMJITAtenCall) {
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

  HostIrLlvmJit jit;
  jit.compile(hic.get());
  // Test single allocate with different sizes
  auto t1 = jit.allocate(allocate, {1});
  EXPECT_EQ(t1.sizes(), at::IntArrayRef({1}));
  auto t2 = jit.allocate(allocate, {1, 2, 3});
  EXPECT_EQ(t2.sizes(), at::IntArrayRef({1, 2, 3}));
  auto t3 = jit.allocate(allocate, {1, 2, 3, 4});
  EXPECT_EQ(t3.sizes(), at::IntArrayRef({1, 2, 3, 4}));
  auto t4 = jit.allocate(allocate, {32, 32});
  EXPECT_EQ(t4.sizes(), at::IntArrayRef({32, 32}));
  auto t5 = jit.allocate(allocate, {32, 32, 32});
  EXPECT_EQ(t5.sizes(), at::IntArrayRef({32, 32, 32}));

  //Test multiple allocates with different sizes
  auto allocate1 = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto allocate2 = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto allocate3 = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto allocate4 = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto allocate5 = IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  hic->pushBackTopLevelExprs(allocate1);
  hic->pushBackTopLevelExprs(allocate2);
  hic->pushBackTopLevelExprs(allocate3);
  hic->pushBackTopLevelExprs(allocate4);
  hic->pushBackTopLevelExprs(allocate5);

  HostIrLlvmJit jit_multi;
  jit_multi.compile(hic.get());
  auto t6 = jit_multi.allocate(allocate1, {1});
  EXPECT_EQ(t6.sizes(), at::IntArrayRef({1}));
  auto t7 = jit_multi.allocate(allocate2, {1, 2, 3});
  EXPECT_EQ(t7.sizes(), at::IntArrayRef({1, 2, 3}));
  auto t8 = jit_multi.allocate(allocate3, {1, 2, 3, 4});
  EXPECT_EQ(t8.sizes(), at::IntArrayRef({1, 2, 3, 4}));
  auto t9 = jit_multi.allocate(allocate4, {32, 32});
  EXPECT_EQ(t9.sizes(), at::IntArrayRef({32, 32}));
  auto t10 = jit_multi.allocate(allocate5, {32, 32, 32});
  EXPECT_EQ(t10.sizes(), at::IntArrayRef({32, 32, 32}));
}

} // namespace hir

} // namespace nvfuser
