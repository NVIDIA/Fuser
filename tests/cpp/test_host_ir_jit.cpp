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

using HostIrJitTest = NVFuserTest;
// Build with: python setup.py install --build-with-llvm
// NVFUSER_ENABLE=host_ir_jit ./bin/test_jit
// --gtest_filter=HostIrJitTest.TestJITAtenCall
TEST_F(HostIrJitTest, TestJITAtenCall) {
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

  // Test single compile with multiple allocates with different sizes
  auto* allocate0 =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto* allocate1 =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto* allocate2 =
      IrBuilder::create<kir::Allocate>(hic_out, MemoryType::Global);
  auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
  auto launch_kernel = IrBuilder::create<LaunchKernel>(
      0,
      LaunchParams(),
      CompileParams(),
      std::vector<Val*>{hic_in},
      std::vector<Val*>{hic_out},
      cache_id);

  hic->pushBackTopLevelExprs(allocate0);
  hic->pushBackTopLevelExprs(allocate1);
  hic->pushBackTopLevelExprs(allocate2);
  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrJit jit;
  jit.compile(hic.get());
  auto allocated_t0 = jit.allocate(allocate0, {32, 32});
  auto allocated_t1 = jit.allocate(allocate1, {64, 64});
  auto allocated_t2 = jit.allocate(allocate2, {16, 128});
  EXPECT_EQ(allocated_t0.sizes(), at::IntArrayRef({32, 32}));
  EXPECT_EQ(allocated_t1.sizes(), at::IntArrayRef({64, 64}));
  EXPECT_EQ(allocated_t2.sizes(), at::IntArrayRef({16, 128}));
}

} // namespace hir

} // namespace nvfuser
