// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using HostIrIntegrationTest = NVFuserTest;

TEST_F(HostIrIntegrationTest, LaunchKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(2);
  fusion.addInput(in);

  TensorView* out = set(in);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  std::vector<c10::IValue> aten_inputs = {t0};
  auto ke = std::make_unique<KernelExecutor>();
  ke->compile(&fusion, aten_inputs);

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  hic->pushBackKernelExecutor(std::move(ke));

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

  EXPECT_TRUE(outputs[0].equal(t0));
}

using EnableHostIrCodepath = bool;

class HostIrCodepathSetTest
    : public NVFuserTest,
      public testing::WithParamInterface<EnableHostIrCodepath> {};

TEST_P(HostIrCodepathSetTest, ) {
  const auto& use_codepath = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  if (use_codepath) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

INSTANTIATE_TEST_SUITE_P(, HostIrCodepathSetTest, ::testing::Bool());

class HostIrCodepathSumTest
    : public NVFuserTest,
      public testing::WithParamInterface<EnableHostIrCodepath> {};

TEST_P(HostIrCodepathSumTest, ) {
  const auto& use_codepath = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  if (use_codepath) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

INSTANTIATE_TEST_SUITE_P(, HostIrCodepathSumTest, ::testing::Bool());

} // namespace hir

} // namespace nvfuser
