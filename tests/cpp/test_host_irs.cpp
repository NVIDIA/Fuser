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
  auto hu = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::move(fusion));

  IrCloner ir_cloner(hic.get());

  std::vector<Val*> post_inputs = {ir_cloner.clone(hu->fusion_to_execute()->inputs().at(0))};
  std::vector<Val*> post_outputs = {ir_cloner.clone(hu->fusion_to_execute()->outputs().at(0))};
  auto post = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu, std::move(post_inputs), std::move(post_outputs));

  hic->top_level_exprs.push_back(post);

  // add global IO to the HostIrContainer. This step could potentially be infered automatically
  hic->addInput(post->inputs().at(0));
  hic->addOutput(post->outputs().at(0));

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
  std::vector<int64_t> input_sizes_1 = {input_sizes_0[1], input_sizes_0[2]};

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

  auto hu_0 = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_0));
  auto hu_1 = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_1));

  std::vector<Val*> post_inputs_0 = {ir_cloner.clone(hu_0->fusion_to_execute()->inputs().at(0))};
  std::vector<Val*> post_outputs_0 = {ir_cloner.clone(hu_0->fusion_to_execute()->outputs().at(0))};
  auto post_0 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu_0, std::move(post_inputs_0), post_outputs_0);

  auto& post_inputs_1 = post_outputs_0;
  std::vector<Val*> post_outputs_1 = {ir_cloner.clone(hu_1->fusion_to_execute()->outputs().at(0))};
  auto post_1 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu_1, std::move(post_inputs_1), post_outputs_1);


  hic->top_level_exprs.push_back(post_0);
  hic->top_level_exprs.push_back(post_1);

  hic->addInput(post_0->inputs().at(0));
  hic->addOutput( post_1->outputs().at(0));

  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes_0, options);
  auto ref_output = at::sum(at::relu(input.toTensor()), at::OptionalIntArrayRef({0,1})) * 2;

  auto outputs = hie.runWithInput({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(0)));
}

TEST_P(HostIrTest, ThreeFusions) {
  auto [use_fusion_executor_cache] = GetParam();

  auto hic = std::make_unique<HostIrContainer>();
  Fusion fusion_0, fusion_1, fusion_2;

  std::vector<int64_t> input_sizes_0 = {4, 8, 32};
  std::vector<int64_t> input_sizes_1 = {input_sizes_0[1], input_sizes_0[2]};

  FusionGuard fg(&fusion_0);
  auto tv0_0 = makeConcreteTensor(input_sizes_0);
  auto tv1_0 = add(tv0_0, tv0_0);
  auto tv2_0 = sum(tv1_0, {0});
  fusion_0.addInput(tv0_0);
  fusion_0.addOutput(tv1_0);
  fusion_0.addOutput(tv2_0);

  FusionGuard::setCurFusion(&fusion_1);
  auto tv0_1 = makeConcreteTensor(input_sizes_0);
  auto tv1_1 = mul(tv0_1, tv0_1);
  auto tv2_1 = sum(tv1_1, {0});
  fusion_1.addInput(tv0_1);
  fusion_1.addOutput(tv2_1);

  FusionGuard::setCurFusion(&fusion_2);
  auto tv0_2 = makeConcreteTensor(input_sizes_1);
  auto tv1_2 = makeConcreteTensor(input_sizes_1);
  auto tv2_2 = add(tv0_2, tv1_2);
  fusion_2.addInput(tv0_2);
  fusion_2.addInput(tv1_2);
  fusion_2.addOutput(tv2_2);


  FusionGuard::setCurFusion(hic.get());
  IrCloner ir_cloner(hic.get());
  auto clone = [&] (std::vector<Val*> vals)
    {
      std::vector<Val*> ret;
      for (auto val: vals) {
        ret.push_back(ir_cloner.clone(val));
      }
      return ret;
    };

  auto hu_0 = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_0));
  auto hu_1 = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_1));
  auto hu_2 = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_2));

  std::vector<Val*> post_inputs_0 = clone({tv0_0});
  std::vector<Val*> post_outputs_0 = clone({tv1_0, tv2_0});
  auto post_0 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu_0, std::move(post_inputs_0), post_outputs_0);

  std::vector<Val*> post_inputs_1 = {post_outputs_0.at(0)};
  std::vector<Val*> post_outputs_1 = clone({tv2_1});
  auto post_1 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu_1, std::move(post_inputs_1), post_outputs_1);

  std::vector<Val*> post_inputs_2 = {post_outputs_0.at(1), post_outputs_1.at(0)};
  std::vector<Val*> post_outputs_2= clone({tv2_2});
  auto post_2 = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu_2, std::move(post_inputs_2), post_outputs_2);

  hic->top_level_exprs.push_back(post_0);
  hic->top_level_exprs.push_back(post_1);
  hic->top_level_exprs.push_back(post_2);

  // add global IO to the HostIrContainer. This step could potentially be infered automatically
  hic->addInput(post_0->inputs().at(0));
  hic->addOutput(post_2->outputs().at(0));

  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue tv0_0_ref_ivalue = at::randn(input_sizes_0, options);
  at::Tensor tv0_0_ref = tv0_0_ref_ivalue.toTensor();
  auto tv1_0_ref = tv0_0_ref + tv0_0_ref;
  auto tv2_0_ref = at::sum(tv1_0_ref, {0});
  auto tv0_1_ref = tv1_0_ref;
  auto tv2_1_ref = at:: sum(tv0_1_ref * tv0_1_ref, {0});
  auto tv0_2_ref = tv2_0_ref;
  auto tv1_2_ref = tv2_1_ref;
  auto tv2_2_ref = tv0_2_ref + tv1_2_ref;

  auto outputs = hie.runWithInput({tv0_0_ref_ivalue});

  GTEST_EXPECT_TRUE(torch::allclose(tv2_2_ref, outputs.at(0)));
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
