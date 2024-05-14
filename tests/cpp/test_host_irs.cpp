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
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

namespace hir {

using HostIrTestParams = std::tuple<bool>;
using HostIrTest = NVFuserFixtureParamTest<HostIrTestParams>;

/*
    We propose a series of test illustrate how to manually write a Host program.

    The syntax can seem cumbersome at first sight, because every detail need to
   be set manually. Let us recall that Host program will typically be generated
   automatically. Here however we show that they can be manually implemented,
   and we make sure that they provide the expected result. Even though it can
   seem cumbersome, we believe that current design is as simple as can be to
   allow for the level of flexibility we will need in the future.

    The first test (HostIrTest.SingleFusion) illustrates the simplest
   non-trivial host program possible: compiling and running a single Fusion. It
   is a good starting point to understand the Host Ir semantics. The host
   program could be illustrated as follows:

    tv0: input

    tv1 = Fusion0 (tv0)

    tv1: output

    Here is a summary of the different steps:
    1) We define the fusion we want to execute

    2) We instantiate a (empty for now) HostIrContainer. This container will be
   used to 1)register the Host IRs, and 2) to represent the Host program through
   its top_level_exprs_.

    3) We create a HostUnit Ir holding the created fusion. (this IR is
   registered in the HostIrContainer)

    4) We create TensorViews that represents, at the Host level, the I/O of the
   Fusion we want to execute. On the one hand, those TensorViews are involved in
   the Host program, so they need to be registered in the HostIrContainer. On
   the other hand, they need to match the I/O of the Fusion we want to execute.
   Therefore we use IrCloner to create those TensorView from the Fusion's I/O.

    5) We create a PostOnStream Ir, taking as argument the HostUnit and the I/O
   TensorView. This IR represents the instruction of executing the Fusion with
   the given I/O.

    6) We define the Host program by adding PostOnStream to the container's top
   level expression. In the current simple example, the HostProgram only
   consists of this single instruction.

    7) We define the Host program's global I/O, using the `addInput` `addOuput`
   methods. In the present simple example, those global I/Os match the
   PostOnStream's I/O, which themselves were cloned from the Fusion's I/O. Note:
   this step could probably be automated from a data dependency analysis

    8) We instantiate HostIrExecutor and run the Host program with concrete
   inputs using HostIrExecutor::runWithInput
*/

TEST_P(HostIrTest, SingleFusion) {
  // [Step 1)] Define the Fusion we want to execute
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_sizes = {4, 8, 32};

  auto tv0 = makeConcreteTensor(input_sizes);
  auto tv1 = add(tv0, tv0);
  auto tv2 = sum(tv1, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  // [Step 2)] Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());
  // [Step 3)] Create a HostUnit Ir holding the created fusion
  auto host_unit = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::move(fusion));

  // [Step 4)] Create TensorViews representing the Fusion's I/O at the Host
  // level
  IrCloner ir_cloner(hic.get());
  std::vector<Val*> post_on_stream_inputs = {
      ir_cloner.clone(host_unit->fusion_to_execute()->inputs().at(0))};
  std::vector<Val*> post_on_stream_outputs = {
      ir_cloner.clone(host_unit->fusion_to_execute()->outputs().at(0))};

  // [Step 5)] Create a PostOnStream Ir representing executing the Fusion with
  // given I/O
  auto post_on_stream = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit,
      post_on_stream_inputs,
      post_on_stream_outputs);

  // [Step 6)] Define the Host program by adding PostOnStream to the container's
  // top level expression
  hic->pushBackTopLevelExprs(post_on_stream);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_on_stream->inputs().at(0));
  hic->addOutput(post_on_stream->outputs().at(0));

  // [Step 8)] Execute the Host program
  HostIrExecutorParams params;
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), params);

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes, options);
  auto ref_output = at::sum(input.toTensor() * 2, {0});

  auto outputs = hie.runWithInput({input});

  // validate the obtained results
  GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(0)));
}

/*
  In the second test, we build upon the previous test by writing a host program
  where we execute to Fusion in a pipeline fashion. The host program could be
  illustrated as follows:

  tv0: input

  tv1 = Fusion0 (tv0)

  tv2 = Fusion1 (tv1)

  tv2: output
*/

TEST_P(HostIrTest, TwoFusions) {
  // [Step 1)] Define the two Fusions we want to execute
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

  // [Step 2)] Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  // [Step 3)] Create two HostUnit Irs holding the fusions
  auto host_unit_0 = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_0));
  auto host_unit_1 = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_1));

  // [Step 4)a.] Create TensorViews representing the first Fusions I/O at the
  // Host level
  IrCloner ir_cloner(hic.get());
  std::vector<Val*> post_on_stream_inputs_0 = {
      ir_cloner.clone(host_unit_0->fusion_to_execute()->inputs().at(0))};
  std::vector<Val*> post_on_stream_outputs_0 = {
      ir_cloner.clone(host_unit_0->fusion_to_execute()->outputs().at(0))};
  // [Step 5)a.] Create a PostOnStream Ir representing executing the first
  // Fusion with given I/O
  auto post_on_stream_0 = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit_0,
      std::move(post_on_stream_inputs_0),
      post_on_stream_outputs_0);

  // [Step 4)b.] Create/reuse TensorViews to represent the second Fusions I/O at
  // the Host level
  auto& post_on_stream_inputs_1 = post_on_stream_outputs_0;
  std::vector<Val*> post_on_stream_outputs_1 = {
      ir_cloner.clone(host_unit_1->fusion_to_execute()->outputs().at(0))};
  // [Step 5)b.] Create a PostOnStream Ir representing executing the second
  // Fusion with given I/O
  auto post_on_stream_1 = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit_1,
      std::move(post_on_stream_inputs_1),
      post_on_stream_outputs_1);

  // [Step 6)] Define the Host program by adding the PostOnStream IRs to the
  // container's top level expression
  hic->pushBackTopLevelExprs(post_on_stream_0);
  hic->pushBackTopLevelExprs(post_on_stream_1);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_on_stream_0->inputs().at(0));
  hic->addOutput(post_on_stream_1->outputs().at(0));

  // [Step 8)] Execute the Host program
  HostIrExecutorParams params;
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes_0, options);
  auto ref_output =
      at::sum(at::relu(input.toTensor()), at::OptionalIntArrayRef({0, 1})) * 2;

  auto outputs = hie.runWithInput({input});

  // validate the obtained results
  GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(0)));
}

/*
  The third test is an example of a nonlinear Host program. It implements a
  situation where we run three Fusion with a nonlinear dependency between I/O.
  The host program could be illustrated as follows:
  tv0: input
  (tv1, tv2) = Fusion0 (tv0)
  tv3 = Fusion2 (tv1)
  tv4 = Fusion3 (tv2, tv3)
  tv4: output
*/

TEST_P(HostIrTest, ThreeFusions) {
  // [Step 1)] Define the Fusions we want to execute
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

  // [Step 2)] Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());
  // [Step 3)] Create HostUnit Irs holding the fusions
  auto host_unit_0 = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_0));
  auto host_unit_1 = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_1));
  auto host_unit_2 = IrBuilder::create<HostUnit>(
      static_cast<IrContainer*>(hic.get()), std::make_unique<Fusion>(fusion_2));

  // [Step 4)a.] Create TensorViews representing the first Fusions I/O at the
  // Host level
  IrCloner ir_cloner(hic.get());
  auto clone = [&](std::vector<Val*> vals) {
    std::vector<Val*> ret;
    for (auto val : vals) {
      ret.push_back(ir_cloner.clone(val));
    }
    return ret;
  };
  std::vector<Val*> post_on_stream_inputs_0 = clone({tv0_0});
  std::vector<Val*> post_on_stream_outputs_0 = clone({tv1_0, tv2_0});
  // [Step 5)a.] Create a PostOnStream Ir representing executing the first
  // Fusion with given I/O
  auto post_on_stream_0 = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit_0,
      std::move(post_on_stream_inputs_0),
      post_on_stream_outputs_0);

  // [Step 4)b.] Create TensorViews representing the second Fusions I/O at the
  // Host level
  std::vector<Val*> post_on_stream_inputs_1 = {post_on_stream_outputs_0.at(0)};
  std::vector<Val*> post_on_stream_outputs_1 = clone({tv2_1});
  // [Step 5)b.] Create a PostOnStream Ir representing executing the first
  // Fusion with given I/O
  auto post_on_stream_1 = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit_1,
      std::move(post_on_stream_inputs_1),
      post_on_stream_outputs_1);

  // [Step 4)c.] Create TensorViews representing the third Fusions I/O at the
  // Host level
  std::vector<Val*> post_on_stream_inputs_2 = {
      post_on_stream_outputs_0.at(1), post_on_stream_outputs_1.at(0)};
  std::vector<Val*> post_on_stream_outputs_2 = clone({tv2_2});
  // [Step 5)c.] Create a PostOnStream Ir representing executing the first
  // Fusion with given I/O
  auto post_on_stream_2 = IrBuilder::create<PostOnStream>(
      static_cast<IrContainer*>(hic.get()),
      host_unit_2,
      std::move(post_on_stream_inputs_2),
      post_on_stream_outputs_2);

  // [Step 6)] Define the Host program by adding the PostOnStream IRs to the
  // container's top level expression
  hic->pushBackTopLevelExprs(post_on_stream_0);
  hic->pushBackTopLevelExprs(post_on_stream_1);
  hic->pushBackTopLevelExprs(post_on_stream_2);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_on_stream_0->inputs().at(0));
  hic->addOutput(post_on_stream_2->outputs().at(0));

  // [Step 8)] Execute the Host program
  HostIrExecutorParams params;
  // we test two different modes of the HostIrExecutor: using FusionExecutor or
  // FusionExecutorCache
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrExecutor hie(std::move(hic), std::move(params));

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue tv0_0_ref_ivalue = at::randn(input_sizes_0, options);
  at::Tensor tv0_0_ref = tv0_0_ref_ivalue.toTensor();
  auto tv1_0_ref = tv0_0_ref + tv0_0_ref;
  auto tv2_0_ref = at::sum(tv1_0_ref, {0});
  auto tv0_1_ref = tv1_0_ref;
  auto tv2_1_ref = at::sum(tv0_1_ref * tv0_1_ref, {0});
  auto tv0_2_ref = tv2_0_ref;
  auto tv1_2_ref = tv2_1_ref;
  auto tv2_2_ref = tv0_2_ref + tv1_2_ref;

  auto outputs = hie.runWithInput({tv0_0_ref_ivalue});

  // validate the obtained results
  GTEST_EXPECT_TRUE(torch::allclose(tv2_2_ref, outputs.at(0)));
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    HostIrTest,
    testing::Combine(testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<bool>>& info) -> std::string {
      return (
          std::get<0>(info.param) ? "use_fusion_executor_cache"
                                  : "use_fusion_executor");
    });

} // namespace hir

} // namespace nvfuser
