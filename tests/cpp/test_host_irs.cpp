// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <fusion_segmenter.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <host_ir/lower.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor_kernel_arg.h>
#include <tests/cpp/utils.h>

#include <algorithm>
#include <iostream>

#include <c10/cuda/CUDAStream.h>

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

    8) We instantiate HostIrEvaluator and run the Host program with concrete
   inputs using HostIrEvaluator::runWithInput
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
  auto host_unit = IrBuilder::create<HostUnit>(std::move(fusion));

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
      host_unit, post_on_stream_inputs, post_on_stream_outputs);

  // [Step 6)] Define the Host program by adding PostOnStream to the container's
  // top level expression
  hic->pushBackTopLevelExprs(post_on_stream);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_on_stream->inputs().at(0));
  hic->addOutput(post_on_stream->outputs().at(0));

  // [Step 8)] Evaluate the Host program
  HostIrEvaluatorParams params;
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrEvaluator hie(std::move(hic), nullptr, params);

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes, options);
  auto ref_output = at::sum(input.toTensor() * 2, {0});

  auto outputs = hie.runWithInput({{post_on_stream->inputs().at(0), input}});

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
  auto host_unit_0 =
      IrBuilder::create<HostUnit>(std::make_unique<Fusion>(fusion_0));
  auto host_unit_1 =
      IrBuilder::create<HostUnit>(std::make_unique<Fusion>(fusion_1));

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

  // [Step 8)] Evaluate the Host program
  HostIrEvaluatorParams params;
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrEvaluator hie(std::move(hic), nullptr, std::move(params));

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes_0, options);
  auto ref_output =
      at::sum(at::relu(input.toTensor()), at::OptionalIntArrayRef({0, 1})) * 2;

  auto outputs = hie.runWithInput({{post_on_stream_0->inputs().at(0), input}});

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
  auto host_unit_0 =
      IrBuilder::create<HostUnit>(std::make_unique<Fusion>(fusion_0));
  auto host_unit_1 =
      IrBuilder::create<HostUnit>(std::make_unique<Fusion>(fusion_1));
  auto host_unit_2 =
      IrBuilder::create<HostUnit>(std::make_unique<Fusion>(fusion_2));

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

  // [Step 8)] Evaluate the Host program
  HostIrEvaluatorParams params;
  // we test two different modes of the HostIrEvaluator: using KernelExecutor or
  // FusionExecutorCache
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrEvaluator hie(std::move(hic), nullptr, std::move(params));

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

  auto outputs =
      hie.runWithInput({{post_on_stream_0->inputs().at(0), tv0_0_ref_ivalue}});

  // validate the obtained results
  GTEST_EXPECT_TRUE(torch::allclose(tv2_2_ref, outputs.at(0)));
}

// This unit test the for-loop IR by implementing a program that could be
// summarized as
//   |  int buf = kInitialValue;
//   |  for (int j = kForLoopStart; j < kForLoopStop; j += kForLoopStep) {
//   |    buf += j;
//   |  }
// where buf is the ouput.
TEST_P(HostIrTest, ForLoops) {
  constexpr int64_t kInitialValue = 21;
  constexpr int64_t kForLoopStart = 1;
  constexpr int64_t kForLoopStop = 7;
  constexpr int64_t kForLoopStep = 2;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  auto* index = IrBuilder::create<Val>(DataType::Index);
  auto* start = IrBuilder::create<Val>(kForLoopStart, DataType::Index);
  auto* stop = IrBuilder::create<Val>(kForLoopStop, DataType::Index);
  auto* step = IrBuilder::create<Val>(kForLoopStep, DataType::Index);
  auto* for_loop = IrBuilder::create<ForLoop>(
      /*IterDomain=*/makeContigConcreteTensor({0})->axis(0), // unused
      index,
      start,
      stop,
      step,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto arange = iota(
      IrBuilder::create<Val>(kForLoopStop),
      IrBuilder::create<Val>(0),
      IrBuilder::create<Val>(1),
      DataType::Int);
  auto* i = IrBuilder::create<Val>(DataType::Index);
  Slice s = {i, add(i, IrBuilder::create<Val>(1)), IrBuilder::create<Val>(1)};
  auto n = slice(arange, {s});
  auto acc_in = makeContigConcreteTensor({1}, DataType::Int);
  auto acc_out = add(acc_in, n);

  fusion->addInput(i);
  fusion->addInput(acc_in);
  fusion->addOutput(acc_out);
  fusion->aliasOutputToInput(acc_out, acc_in, AllocationType::ReuseBuffer);

  FusionGuard::setCurFusion(hic.get());

  auto buffer_input = makeContigConcreteTensor({1}, DataType::Int);
  auto buffer_ouput = makeContigConcreteTensor({1}, DataType::Int);

  IrCloner ir_cloner(hic.get());
  std::vector<Val*> post_on_stream_inputs = {index, buffer_input};
  std::vector<Val*> post_on_stream_outputs = {buffer_ouput};
  auto* host_unit = IrBuilder::create<HostUnit>(std::move(fusion));
  auto* post_on_stream = IrBuilder::create<PostOnStream>(
      host_unit, post_on_stream_inputs, post_on_stream_outputs);

  for_loop->body().push_back(post_on_stream);

  hic->addInput(buffer_input);
  hic->pushBackTopLevelExprs(for_loop);

  HostIrEvaluatorParams params;
  auto [use_fusion_executor_cache] = GetParam();
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrEvaluator hie(std::move(hic), /*communicator=*/nullptr, params);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor buffer_at = torch::tensor({kInitialValue}, options);

  hie.runWithInput({{buffer_input, buffer_at}});

  // Compute expected result for validation
  int64_t expected_result_data = kInitialValue;
  for (int j = kForLoopStart; j < kForLoopStop; j += kForLoopStep) {
    expected_result_data += j;
  }
  at::Tensor expected_result = torch::tensor({expected_result_data}, options);

  EXPECT_TRUE(expected_result.equal(buffer_at));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    HostIrTest,
    testing::Combine(testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<bool>>& info) -> std::string {
      return std::get<0>(info.param) ? "useFusionExecutorCache"
                                     : "useFusionExecutor";
    });

using StreamTest = NVFuserTest;

// The following test simply demonstrate how to change current CUDA stream in
// the host program
TEST_F(StreamTest, HostIrSetStream) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  auto stream = IrBuilder::create<Stream>();
  auto set_stream = IrBuilder::create<SetCurrentStream>(stream);
  hic->pushBackTopLevelExprs(set_stream);

  HostIrEvaluator hie(std::move(hic));
  setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream(0));
  hie.runWithInput({});
  EXPECT_NE(
      c10::cuda::getDefaultCUDAStream(0), c10::cuda::getCurrentCUDAStream(0));
}

// The following test simply demonstrate how to change current CUDA stream in
// the host program
TEST_F(StreamTest, HostIrDefaultStream) {
  auto change_stream = [](bool use_default_stream) {
    auto hic = std::make_unique<HostIrContainer>();
    FusionGuard fg(hic.get());
    Stream* stream;
    if (use_default_stream) {
      stream = hic->getDefaultStream();
    } else {
      stream = IrBuilder::create<Stream>();
    }
    auto set_stream = IrBuilder::create<SetCurrentStream>(stream);
    hic->pushBackTopLevelExprs(set_stream);
    HostIrEvaluator hie(std::move(hic));
    hie.runWithInput({});
  };

  setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream(0));
  change_stream(/*use_default_stream=*/false);
  EXPECT_NE(
      c10::cuda::getDefaultCUDAStream(0), c10::cuda::getCurrentCUDAStream(0));
  change_stream(/*use_default_stream=*/true);
  EXPECT_EQ(
      c10::cuda::getDefaultCUDAStream(0), c10::cuda::getCurrentCUDAStream(0));
}

TEST_F(StreamTest, HostIrGetCurrentStream) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  auto get_stream = IrBuilder::create<GetCurrentStream>();
  auto current_stream = get_stream->stream();
  auto other_stream = IrBuilder::create<Stream>();
  hic->pushBackTopLevelExprs(get_stream);
  hic->pushBackTopLevelExprs(IrBuilder::create<SetCurrentStream>(other_stream));
  hic->pushBackTopLevelExprs(
      IrBuilder::create<SetCurrentStream>(current_stream));

  auto cuda_stream = c10::cuda::getStreamFromPool();
  setCurrentCUDAStream(cuda_stream);

  HostIrEvaluator hie(std::move(hic));
  hie.runWithInput({});

  EXPECT_EQ(cuda_stream, c10::cuda::getCurrentCUDAStream(0));
}

TEST_F(StreamTest, ByIndex) {
  constexpr int64_t kStreamIndex1 = 2;
  constexpr int64_t kStreamIndex2 = 3;
  static_assert(kStreamIndex1 != kStreamIndex2);

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());
  auto stream1 =
      IrBuilder::create<Stream>(IrBuilder::create<Val>(kStreamIndex1));
  auto stream1_prime =
      IrBuilder::create<Stream>(IrBuilder::create<Val>(kStreamIndex1));
  auto stream2 =
      IrBuilder::create<Stream>(IrBuilder::create<Val>(kStreamIndex2));

  hic->pushBackTopLevelExprs(IrBuilder::create<SetCurrentStream>(stream1));
  hic->pushBackTopLevelExprs(
      IrBuilder::create<SetCurrentStream>(stream1_prime));
  hic->pushBackTopLevelExprs(IrBuilder::create<SetCurrentStream>(stream2));

  HostIrEvaluator hie(std::move(hic));
  hie.runWithInput({});

  const std::unordered_map<
      std::variant<int64_t, Stream*>,
      c10::cuda::CUDAStream>& streams = hie.getCudaStreams();
  // This stream hashtable should contain the default stream and only rwo extra
  // streams, cached with the integer index "2" and "3" as keys
  EXPECT_EQ(streams.size(), 3);
  for (auto it : streams) {
    auto key = it.first;
    if (std::holds_alternative<int64_t>(key)) {
      EXPECT_NE(streams.at(key), c10::cuda::getDefaultCUDAStream(0))
          << "newly created stream should not coincide with default stream";
      auto index = std::get<int64_t>(key);
      if (index == kStreamIndex1) {
        EXPECT_NE(streams.at(key), c10::cuda::getCurrentCUDAStream(0))
            << "Stream " << index << " should not be the current active stream";
      } else if (index == kStreamIndex2) {
        EXPECT_EQ(streams.at(key), c10::cuda::getCurrentCUDAStream(0))
            << "Stream " << index << " should be the current active stream";
      } else {
        FAIL() << "stream's index " << index << "should be " << kStreamIndex1
               << " or " << kStreamIndex2;
      }
    } else if (std::holds_alternative<Stream*>(key)) {
      EXPECT_EQ(streams.at(key), c10::cuda::getDefaultCUDAStream(0));
    } else {
      FAIL() << "stream key of unsupported type";
    }
  }
}

using StreamHostIrTestParams = std::tuple<bool, int, int>;
using StreamHostIrTest = NVFuserFixtureParamTest<StreamHostIrTestParams>;

// The following test execute the same fusion `n_iterations` times by posting
// the kernels on `n_streams` different streams in a Round-Robin fashion. We
// thus produce `n_iterations` outputs from the same input, with a potential
// overlap of n_streams/n_iterations
TEST_P(StreamHostIrTest, SingleFusionMultipleStreams) {
  auto [use_fusion_executor_cache, n_streams, n_iterations] = GetParam();

  // [Step 1)] Define the Fusion we want to execute
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_sizes = {4, 8, 32};

  auto tv0 = makeConcreteTensor(input_sizes);
  auto tv1 = add(tv0, tv0);
  auto tv2 = sum(tv1, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  // [Step 2)] Instantiate an HostIroCntainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  // Create N different Streams
  std::vector<Stream*> streams;
  for (int i = 0; i < n_streams; i++) {
    streams.push_back(IrBuilder::create<Stream>());
  }

  // [Step 3)] Create a HostUnit Ir holding the created fusion
  auto host_unit = IrBuilder::create<HostUnit>(std::move(fusion));

  // [Step 4)] Create TensorViews representing the Fusion's inputs at the Host
  // level
  IrCloner ir_cloner_input(hic.get());
  std::vector<Val*> post_on_stream_inputs = {
      ir_cloner_input.clone(host_unit->fusion_to_execute()->inputs().at(0))};
  hic->addInput(post_on_stream_inputs.at(0));

  for (int i = 0; i < n_iterations; i++) {
    // [Step 4)] Create TensorViews representing the Fusion's ouputs at the Host
    // level
    IrCloner ir_cloner_output(hic.get());
    std::vector<Val*> post_on_stream_outputs = {ir_cloner_output.clone(
        host_unit->fusion_to_execute()->outputs().at(0))};

    // [Step 5)] Create a PostOnStream Ir representing executing the Fusion with
    // given I/O
    auto post_on_stream = IrBuilder::create<PostOnStream>(
        host_unit, post_on_stream_inputs, post_on_stream_outputs);

    // Set the Stream
    auto set_stream =
        IrBuilder::create<SetCurrentStream>(streams[i % streams.size()]);

    // [Step 6)] Define the Host program by adding PostOnStream to the
    // container's top level expression
    hic->pushBackTopLevelExprs(set_stream);
    hic->pushBackTopLevelExprs(post_on_stream);

    // [Step 7)] Define the Host program's global I/O
    hic->addOutput(post_on_stream->outputs().at(0));
  }

  // [Step 8)] Evaluate the Host program
  HostIrEvaluatorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  HostIrEvaluator hie(std::move(hic), nullptr, params);

  // define concrete inputs and compute ref output for validation
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn(input_sizes, options);
  auto ref_output = at::sum(input.toTensor() * 2, {0});

  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {post_on_stream_inputs.at(0), input}};

  setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream(0));

  auto outputs = hie.runWithInput(concrete_input_buffers);

  // validate the obtained results
  for (int i = 0; i < n_iterations; i++) {
    GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.at(i)));
  }
  EXPECT_NE(
      c10::cuda::getDefaultCUDAStream(0), c10::cuda::getCurrentCUDAStream(0));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    StreamHostIrTest,
    testing::Combine(
        testing::Values(true),
        testing::Values(1, 4),
        testing::Values(1, 8)),
    [](const testing::TestParamInfo<StreamHostIrTestParams>& info)
        -> std::string {
      std::stringstream ss;
      ss
          << (std::get<0>(info.param) ? "useFusionExecutorCache"
                                      : "useFusionExecutor");
      ss << "_";
      ss << "NStreams" << std::get<1>(info.param);
      ss << "_";
      ss << "NIterations" << std::get<2>(info.param);
      return ss.str();
    });

using SliceHostIrTestParams = bool;
using SliceHostIrTest = NVFuserFixtureParamTest<SliceHostIrTestParams>;

TEST_P(SliceHostIrTest, SlicingTensor) {
  constexpr int64_t ndims = 2;
  constexpr int64_t axis = 1;
  constexpr int64_t start = 3;
  constexpr int64_t stop = 13;
  constexpr int64_t step = 1;
  const std::vector<int64_t> input_sizes = {32, 32};

  ASSERT_LT(axis, ndims);
  ASSERT_LT(start, stop);
  ASSERT_EQ(
      step,
      1); // only "1" is supported at the moment,
          // https://github.com/NVIDIA/Fuser/blob/bad998ae277ffc2f43fdc28dca07d01d737a1623/csrc/ops/alias.cpp#L764
  ASSERT_EQ(input_sizes.size(), ndims);

  const bool put_slice_op_in_top_level_expr = GetParam();

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* tv = makeContigTensor(ndims);
  auto* start_val = IrBuilder::create<Val>(start, DataType::Index);
  auto* stop_val = IrBuilder::create<Val>(stop, DataType::Index);
  auto* step_val = IrBuilder::create<Val>(step, DataType::Index);
  Slice range = {.start = start_val, .stop = stop_val, .step = step_val};
  std::vector<Slice> ranges(ndims);
  ranges.at(axis) = range;
  TensorView* sliced_tv = slice(tv, ranges);

  hic->addInput(tv);
  hic->addOutput(sliced_tv);

  if (put_slice_op_in_top_level_expr) {
    hic->pushBackTopLevelExprs(sliced_tv->definition());
  }

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  c10::IValue input = at::randn(input_sizes, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {hie.inputs().at(0), input}};

  auto output = hie.runWithInput(concrete_input_buffers).at(0);

  // validate
  at::Tensor input_aten = input.toTensor();
  std::vector<at::indexing::TensorIndex> ranges_aten(
      input_aten.dim(), at::indexing::Slice());
  ranges_aten.at(axis) = at::indexing::Slice(start, stop, step);
  auto ref_output = input_aten.index(ranges_aten);
  if (put_slice_op_in_top_level_expr) {
    EXPECT_TRUE(ref_output.equal(output));
  } else {
    EXPECT_EQ(output.numel(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SliceHostIrTest,
    testing::Bool(),
    [](const testing::TestParamInfo<SliceHostIrTestParams>& info)
        -> std::string {
      std::stringstream ss;
      ss << "SliceOp";
      if (!info.param) {
        ss << "Not";
      }
      ss << "InTopLevelExpr";
      return ss.str();
    });

using MatmulHostIrTest = NVFuserTest;

TEST_F(MatmulHostIrTest, HostIr) {
  constexpr int64_t H = 32;
  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 256;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(3);
  TensorView* c = matmul(a, b);

  hic->addInput(a);
  hic->addInput(b);
  hic->addOutput(c);

  hic->pushBackTopLevelExprs(c->definition());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  at::Tensor a_tensor = at::randn({H, M, K}, options);
  at::Tensor b_tensor = at::randn({H, K, N}, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {hie.inputs().at(0), a_tensor}, {hie.inputs().at(1), b_tensor}};

  auto output = hie.runWithInput(concrete_input_buffers).at(0);

  // validate
  auto ref_output = at::matmul(a_tensor, b_tensor);

  EXPECT_TRUE(ref_output.allclose(output));
}

TEST_F(MatmulHostIrTest, HostIrMatmulOut) {
  constexpr int64_t H = 32;
  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 256;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(3);
  TensorView* c = makeContigTensor(3);
  auto* matmul = IrBuilder::create<MatmulOp>(c, a, b);

  hic->addInput(a);
  hic->addInput(b);
  hic->addInput(c);
  hic->addOutput(c);

  hic->pushBackTopLevelExprs(matmul);

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  at::Tensor a_tensor = at::randn({H, M, K}, options);
  at::Tensor b_tensor = at::randn({H, K, N}, options);
  at::Tensor c_tensor = at::randn({H, M, N}, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {a, a_tensor}, {b, b_tensor}, {c, c_tensor}};

  hie.runWithInput(concrete_input_buffers);

  // validate
  auto ref_output = at::matmul(a_tensor, b_tensor);

  EXPECT_TRUE(ref_output.allclose(c_tensor));
}

using SelectHostIrTestParams = bool;
using SelectHostIrTest = NVFuserFixtureParamTest<SelectHostIrTestParams>;

TEST_P(SelectHostIrTest, SelectingTensor) {
  constexpr int64_t ndims = 2;
  constexpr int64_t dim = 1;
  constexpr int64_t index = 3;
  const std::vector<int64_t> input_sizes = {32, 32};

  ASSERT_LT(dim, ndims);
  ASSERT_EQ(input_sizes.size(), ndims);
  ASSERT_LT(index, input_sizes.at(dim));

  const bool put_select_op_in_top_level_expr = GetParam();

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* tv = makeContigTensor(ndims);
  auto* index_val = IrBuilder::create<Val>(index, DataType::Index);
  TensorView* selected_tv = select(tv, dim, index_val);

  hic->addInput(tv);
  hic->addOutput(selected_tv);

  if (put_select_op_in_top_level_expr) {
    hic->pushBackTopLevelExprs(selected_tv->definition());
  }

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  c10::IValue input = at::randn(input_sizes, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {hie.inputs().at(0), input}};

  auto output = hie.runWithInput(concrete_input_buffers).at(0);

  // validate
  at::Tensor input_aten = input.toTensor();
  auto ref_output = input_aten.select(dim, index);
  if (put_select_op_in_top_level_expr) {
    EXPECT_TRUE(ref_output.equal(output));
  } else {
    EXPECT_EQ(output.numel(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SelectHostIrTest,
    testing::Bool(),
    [](const testing::TestParamInfo<SelectHostIrTestParams>& info)
        -> std::string {
      std::stringstream ss;
      ss << "SelectOp";
      if (!info.param) {
        ss << "Not";
      }
      ss << "InTopLevelExpr";
      return ss.str();
    });

using ViewTest = NVFuserTest;

TEST_F(ViewTest, SimpleReshape) {
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto input = makeContigTensor(2);
  Val* x = input->axis(0)->extent();
  Val* y = input->axis(1)->extent();
  Val* xy = mul(x, y);
  auto flattened_input = reshape(input, {xy});
  auto transposed_intput = reshape(flattened_input, {y, x});

  hic->addInput(input);
  hic->addOutput(flattened_input);
  hic->addOutput(transposed_intput);
  hic->pushBackTopLevelExprs(flattened_input->definition());
  hic->pushBackTopLevelExprs(transposed_intput->definition());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  constexpr int64_t kX = 32;
  constexpr int64_t kY = 64;
  auto input_aten = at::randn({kX, kY}, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {
      {input, input_aten}};

  auto outputs = hie.runWithInput(concrete_input_buffers);

  // validate
  EXPECT_TRUE(outputs[0].equal(at::reshape(input_aten, {kX * kY})));
  EXPECT_TRUE(outputs[1].equal(at::reshape(input_aten, {kY, kX})));
}

using ReductionHostIrTest = NVFuserTest;

TEST_F(ReductionHostIrTest, Sum) {
  constexpr int64_t kTensorSize = 32;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* a = makeContigTensor(1);
  TensorView* b = sum(a, {0});

  hic->addInput(a);
  hic->addOutput(b);
  hic->pushBackTopLevelExprs(b->definition());

  HostIrEvaluator hie(std::move(hic));

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
  auto a_aten = at::randn({kTensorSize}, options);
  std::unordered_map<Val*, c10::IValue> concrete_input_buffers = {{a, a_aten}};

  auto outputs = hie.runWithInput(concrete_input_buffers);

  // validate
  EXPECT_TRUE(outputs[0].equal(at::sum(a_aten, 0)));
}

using IfThenElseTest = NVFuserTest;

TEST_F(IfThenElseTest, HostIr) {
  auto create_fusion_add_one = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto input = makeContigTensor(1);
    auto output = add(input, fusion->oneVal());
    fusion->addInput(input);
    fusion->addOutput(output);
    fusion->aliasOutputToInput(output, input, AllocationType::ReuseBuffer);
    return fusion;
  };

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* input_bool = IrBuilder::create<Val>(DataType::Bool);
  auto* predicate = IrBuilder::create<kir::Predicate>(input_bool);
  auto* if_then_else = IrBuilder::create<kir::IfThenElse>(predicate);

  std::vector<Val*> shape = {hic->oneVal()};
  auto* input_buffer = makeContigTensor(1);
  auto* output_buffer = makeContigTensor(1);

  auto add_one_to_buffer = IrBuilder::create<PostOnStream>(
      IrBuilder::create<HostUnit>(create_fusion_add_one()),
      std::vector<Val*>({input_buffer}),
      std::vector<Val*>({output_buffer}));

  if_then_else->thenBody().push_back(add_one_to_buffer);
  if_then_else->thenBody().push_back(add_one_to_buffer);
  if_then_else->elseBody().push_back(add_one_to_buffer);

  hic->addInput(input_bool);
  hic->addOutput(input_buffer);
  hic->addOutput(output_buffer);
  hic->pushBackTopLevelExprs(if_then_else);

  HostIrEvaluator hie(std::move(hic));

  for (auto boolean : {true, false}) {
    c10::IValue input_bool_c10 = c10::ivalue::from(boolean);
    auto options =
        at::TensorOptions().device(at::kCUDA, 0).dtype(torch::kFloat);
    c10::IValue input_buffer_c10 = at::ones(1, options);
    std::unordered_map<Val*, c10::IValue> concrete_inputs = {
        {input_bool, input_bool_c10}, {input_buffer, input_buffer_c10}};

    auto outputs = hie.runWithInput(concrete_inputs);

    // validate
    auto ref_output =
        at::ones_like(input_buffer_c10.toTensor()) + (1 + (int)boolean);
    EXPECT_TRUE(outputs.at(0).equal(ref_output));
  }
}

using AllocationTest = NVFuserTest;

TEST_F(AllocationTest, HostIr) {
  const std::vector<int64_t> sizes = {8, 64};

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* tv = makeConcreteTensor(sizes);
  tv->setMemoryType(MemoryType::Global);
  auto* allocate = IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
  hic->addOutput(tv);
  hic->pushBackTopLevelExprs(allocate);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({});

  EXPECT_EQ(sizes, outputs.at(0).sizes());
}

TEST_F(AllocationTest, inHostForLoop) {
  constexpr int64_t kForLoopStop = 4;
  const std::vector<int64_t> sizes = {8, 64};

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* for_loop = IrBuilder::create<ForLoop>(
      /*IterDomain=*/makeContigConcreteTensor({0})->axis(0), // unused
      /*index=*/IrBuilder::create<Val>(DataType::Index),
      /*start=*/hic->zeroVal(),
      /*stop=*/IrBuilder::create<Val>(kForLoopStop, DataType::Index),
      /*step=*/hic->oneVal(),
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  TensorView* tv0 = makeConcreteTensor(sizes);
  tv0->setMemoryType(MemoryType::Global);
  auto* allocate = IrBuilder::create<kir::Allocate>(tv0, MemoryType::Global);
  TensorView* tv1 = abs(tv0);

  for_loop->body().push_back(allocate);
  for_loop->body().push_back(tv1->definition());

  hic->pushBackTopLevelExprs(for_loop);
  hic->addOutput(tv1);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({});

  EXPECT_EQ(sizes, outputs.at(0).sizes());
}

} // namespace hir

} // namespace nvfuser
