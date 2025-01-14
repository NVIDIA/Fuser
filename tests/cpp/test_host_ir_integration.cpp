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

namespace nvfuser {

namespace hir {

using HostIrIntegrationTest = NVFuserTest;

TEST_F(HostIrIntegrationTest, LaunchKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  std::vector<c10::IValue> aten_inputs = {t0};
  auto ke = std::make_unique<KernelExecutor>();
  ke->compile(&fusion, aten_inputs);

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  hic->pushBackKernelExecutor(std::move(ke));

  IrCloner ir_cloner(hic.get());
  auto tv2 = ir_cloner.clone(tv0);
  auto tv3 = ir_cloner.clone(tv1);

  std::vector<Val*> launch_kernel_inputs = {tv2};
  std::vector<Val*> launch_kernel_outputs = {tv3};

  hic->addInput(launch_kernel_inputs.back());
  hic->addOutput(launch_kernel_outputs.back());

  auto launch_kernel = IrBuilder::create<LaunchKernel>(
      0, launch_kernel_inputs, launch_kernel_outputs);

  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrEvaluator hie(std::move(hic));

  at::Tensor output = at::empty({32, 32}, options);
  auto outputs = hie.runWithInput({{tv2, t0}, {tv3, output}});

  EXPECT_TRUE(outputs[0].equal(t0));
}

} // namespace hir

} // namespace nvfuser
