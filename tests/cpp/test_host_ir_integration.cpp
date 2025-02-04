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
      0, LaunchParams(), CompileParams(), std::vector<Val*>{hic_in}, std::vector<Val*>{hic_out});

  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({{hic_in, t0}});

  EXPECT_TRUE(outputs[0].equal(t0));
}

TEST_F(HostIrIntegrationTest, HostIrCodepathSet) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__, "");
}

TEST_F(HostIrIntegrationTest, HostIrCodepathSum) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__, "");
}

TEST_F(HostIrIntegrationTest, NonHostIrCodepathSum) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  //EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__, "");
}

TEST_F(HostIrIntegrationTest, NonHostIrCodepath) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});

  testValidate(executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__, "");
}

} // namespace hir

} // namespace nvfuser
