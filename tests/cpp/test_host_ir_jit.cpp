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

TEST_F(HostIrJitTest, Matmul) {
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

}

} // namespace nvfuser
