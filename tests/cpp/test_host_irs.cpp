// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <executor_kernel_arg.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <host_ir_container.h>
#include <host_ir_executor.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

namespace hir {

using HostIrTestParams = std::tuple<bool>;
class HostIrTest:
    public NVFuserTest,
    public testing::WithParamInterface<HostIrTestParams> {};


TEST_P(HostIrTest, SingleFusion) {
  auto [use_fusion_executor_cache] = GetParam();

  auto hic = std::make_unique<HostIrContainer>();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_sizes = {4, 8, 32};

  auto tv0 = makeConcreteTensor(input_sizes);
  auto tv1 = add(tv0, tv0);
  auto tv2 = sum(tv1, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  FusionGuard::setCurFusion(hic.get());
  auto eu = IrBuilder::create<ExecutableUnit>(static_cast<IrContainer*>(hic.get()), std::move(fusion));
  std::cout << "EU: " << eu << std::endl;

  IrCloner ir_cloner(hic.get());

  std::vector<Val*> post_inputs;
  post_inputs.reserve(eu->fusion_to_execute()->inputs().size());
  for (auto input: eu->fusion_to_execute()->inputs()) {
    auto post_input = ir_cloner.clone(input);
    post_inputs.push_back(post_input);
  }

  std::vector<Val*> post_outputs;
  post_outputs.reserve(eu->fusion_to_execute()->outputs().size());
  for (auto output: eu->fusion_to_execute()->outputs()) {
    auto post_output = ir_cloner.clone(output);
    post_outputs.push_back(post_output);
  }

  auto post = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), eu, std::move(post_inputs), std::move(post_outputs));

  hic->top_level_exprs.push_back(post);

  // add global IO to the HostIrContainer. This step could potentially be infered automatically
  for (auto input: post->inputs()){
    hic->addInput(input);
  }
  for (auto output: post->outputs()){
    hic->addOutput(output);
  }

  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes, options);
  auto ref_output = at::sum(input.toTensor() * 2, {0});

  auto outputs = hie.runWithInput({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(0)));
}

TEST_P(HostIrTest, TwoFusions) {
  auto [use_fusion_executor_cache] = GetParam();

  auto hic = std::make_unique<HostIrContainer>();
  Fusion fusion_0, fusion_1;

  std::vector<int64_t> input_sizes_0 = {4, 8, 32};
  std::vector<int64_t> input_sizes_1;
  for (int i = 1; i<3; i++) {
    input_sizes_1.push_back(input_sizes_0[i]);
  }

  FusionGuard fg(&fusion_0);
  auto tv0_0 = makeConcreteTensor(input_sizes_0);
  auto tv1_0 = relu(tv0_0);
  auto tv2_0 = sum(tv1_0, {0});
  fusion_0.addInput(tv0_0);
  fusion_0.addOutput(tv2_0);

  FusionGuard::setCurFusion(&fusion_1);
  auto tv0_1 = makeConcreteTensor(input_sizes_1);
  auto tv1_1 = add(tv0_1, tv0_1);
  auto tv2_1 = sum(tv1_1, {0});
  fusion_1.addInput(tv0_1);
  fusion_1.addOutput(tv2_1);


  FusionGuard::setCurFusion(hic.get());
  IrCloner ir_cloner(hic.get());

  auto eu_0 = IrBuilder::create<ExecutableUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_0));
  std::cout << "EU_0: " << eu_0 << std::endl;
  auto eu_1 = IrBuilder::create<ExecutableUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_1));
  std::cout << "EU_1: " << eu_1 << std::endl;

  std::vector<Val*> post_inputs_0;
  for (auto input: eu_0->fusion_to_execute()->inputs()) {
    post_inputs_0.push_back(ir_cloner.clone(input));
  }
  std::vector<Val*> post_outputs_0;
  for (auto output: eu_0->fusion_to_execute()->outputs()) {
    post_outputs_0.push_back(ir_cloner.clone(output));
  }
  auto post_0 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), eu_0, std::move(post_inputs_0), post_outputs_0);

  auto& post_inputs_1 = post_outputs_0;
  std::vector<Val*> post_outputs_1;
  for (auto output: eu_1->fusion_to_execute()->outputs()) {
    post_outputs_1.push_back(ir_cloner.clone(output));
  }
  auto post_1 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), eu_1, std::move(post_inputs_1), post_outputs_1);


  hic->top_level_exprs.push_back(post_0);
  hic->top_level_exprs.push_back(post_1);

  // add global IO to the HostIrContainer. This step could potentially be infered automatically
  for (auto input: post_0->inputs()){
    hic->addInput(input);
  }
  for (auto output: post_1->outputs()){
    hic->addOutput(output);
  }

  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes_0, options);
  auto ref_output = at::sum(at::relu(input.toTensor()), at::OptionalIntArrayRef({0,1})) * 2;

  auto outputs = hie.runWithInput({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(0))) << "ref_outputs:\n " << ref_output << "\noutputs:\n " << outputs.at(0);
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    HostIrTest,
    testing::Combine(testing::Bool()));

class FusionExecutorWithExternalForLoop {
public:
  FusionExecutorWithExternalForLoop(
      std::unique_ptr<Fusion> fusion) : fec_(std::move(fusion)) {}

  at::Tensor runWithInputs_v0(const at::ArrayRef<c10::IValue>& inputs) {
    auto input = inputs.at(0);
    auto aten_input = input.toTensor();
    auto for_loop_extent = aten_input.sizes().at(0);
    std::vector<at::Tensor> outputs;
    for (int for_loop_index = 0; for_loop_index < for_loop_extent; for_loop_index++) {
      c10::IValue input_i = input.toTensor().index({at::indexing::Slice(for_loop_index, for_loop_index + 1), "..."});
      outputs.push_back(fec_.runFusionWithInputs({input_i}).at(0));
    }
    return at::concat(outputs);
  }

  at::Tensor runWithInputs(const at::ArrayRef<c10::IValue>& inputs) {
    auto input = inputs.at(0);
    auto aten_input = input.toTensor();
    auto for_loop_extent = aten_input.sizes().at(0);
    std::vector<at::Tensor> outputs;
    for (int for_loop_index = 0; for_loop_index < for_loop_extent; for_loop_index++) {
      std::cout << "for_loop_index=" << for_loop_index 
                << ", for_loop_extent=" << for_loop_extent
                << ", running kernel:\n";
      fec_.fusion()->printKernel();
      std::cout << std::endl;
      c10::IValue input_i = input.toTensor().index({at::indexing::Slice(for_loop_index, for_loop_index + 1), "..."});
      outputs.push_back(fec_.runFusionWithInputs({input_i}).at(0));
    }
    return at::concat(outputs);
  }

private:
  FusionExecutorCache fec_;
};

class CpuForLoopTest: public NVFuserTest {};

TEST_F(CpuForLoopTest, pointwiseKernelSingleIO) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor({2});
  fusion->addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn({4,8}, options);
  auto ref_output = at::sum(input.toTensor() * 2, {1});

  FusionExecutorWithExternalForLoop executor(std::move(fusion));
  auto output = executor.runWithInputs_v0({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, output));
}

TEST_F(CpuForLoopTest, kernelSingleIO) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor({2});
  fusion->addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn({4,8}, options);
  auto ref_output = at::sum(input.toTensor() * 2, {1});

  tv0->axis(0)->parallelize(ParallelType::Host);
  tv1->axis(0)->parallelize(ParallelType::Host);
  tv2->axis(0)->parallelize(ParallelType::Host);
  fusion->print();

  FusionExecutorWithExternalForLoop executor(std::move(fusion));
  auto output = executor.runWithInputs({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, output));
}

} // namespace hir

} // namespace nvfuser
