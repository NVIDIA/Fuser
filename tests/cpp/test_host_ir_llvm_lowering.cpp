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
#include <host_ir/lower_to_llvm.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using testing::Contains;
using HostIrLLVMTest = NVFuserTest;
// Build with: python setup.py install --build-with-llvm
// NVFUSER_ENABLE=host_ir_lowering ./bin/test_host_ir
// --gtest_filter=HostIrLLVMTest.TestLLVMJIT
TEST_F(HostIrLLVMTest, TestLLVMJIT) {
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

  HostIrLlvmJit::getInstance(4).compile(hic.get());
  auto t = HostIrLlvmJit::getInstance(4).allocate(allocate, {32, 32});
  EXPECT_TRUE(t.equal(t0));

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({{hic_in, t0}});

  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

} // namespace hir

} // namespace nvfuser
