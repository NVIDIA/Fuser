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

using MultiDeviceHostIrIntegrationTestParams = std::tuple<bool, bool>;

class MultiDeviceHostIrIntegrationTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<MultiDeviceHostIrIntegrationTestParams> {};

TEST_P(MultiDeviceHostIrIntegrationTest, test_kernel) {
  //auto [use_fusion_executor_cache, with_sharding_annotations] = GetParam();
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

  // [Step 5)a.] Create PostOnStream Irs representing executing the Fusion
  std::vector<Val*> lk_inputs = {tv0};
  std::vector<Val*> lk_outputs = {tv1};
  auto launch_kernel =
      IrBuilder::create<LaunchKernel>(0, lk_inputs, lk_outputs); // todo: change to segment index instead of hardcoding index 0 in the kernel_executors_ vector

  // [Step 6)] Define the Host program
  hic->pushBackTopLevelExprs(launch_kernel);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(lk_inputs.back());
  hic->addOutput(lk_outputs.back());

  // [Step 8)] Evaluate the Host program
  HostIrEvaluatorParams params;
  params.use_fusion_executor_cache = false;
  HostIrEvaluator hie(std::move(hic), communicator_, params);

  at::Tensor output = at::empty({32, 32}, options);
  auto outputs = hie.runWithInput({{inputs.back(), input}, {outputs.back(), output}});

  // validate the obtained results
  ASSERT_TRUE(outputs[0].equal(t0));
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    MultiDeviceHostIrIntegrationTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<MultiDeviceHostIrIntegrationTestParams>& info)
        -> std::string {
      std::string s;
      s += std::get<0>(info.param) ? "useFusionExecutorCache"
                                   : "useFusionExecutor";
      s += "_";
      s += std::get<1>(info.param) ? "withShardingAnnotations"
                                   : "withoutShardingAnnotations";
      return s;
    });

} // namespace hir

} // namespace nvfuser
