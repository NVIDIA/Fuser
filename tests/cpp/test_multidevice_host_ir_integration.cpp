// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

namespace hir {

TEST_F(MultiDeviceTest, LaunchKernel) {
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

  hic->pushBackKernelExecutor(ke.get());

  IrCloner ir_cloner(hic.get());
  auto tv2 = ir_cloner.clone(tv0);
  auto tv3 = ir_cloner.clone(tv1);

  std::vector<Val*> lk_inputs = {tv2};
  std::vector<Val*> lk_outputs = {tv3};

  hic->addInput(lk_inputs.back());
  hic->addOutput(lk_outputs.back());

  auto launch_kernel =
      IrBuilder::create<LaunchKernel>(0, lk_inputs, lk_outputs);

  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrEvaluatorParams params;
  params.use_fusion_executor_cache = false;
  HostIrEvaluator hie(std::move(hic), communicator_, params);

  at::Tensor output = at::empty({32, 32}, options);
  auto outputs =
      hie.runWithInput({{lk_inputs.back(), t0}, {lk_outputs.back(), output}});

  ASSERT_TRUE(outputs[0].equal(t0));
}

} // namespace hir

} // namespace nvfuser
