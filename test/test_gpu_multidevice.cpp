// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// utility function for validation
void testValidateMultidevice(
    std::unique_ptr<Fusion> fusion_ptr,
    MultiDeviceRuntime& runtime,
    const at::ArrayRef<c10::IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    DeviceIdxType tester = 0) {
  Communicator& comm = runtime.comm();
  std::vector<at::Tensor> buffer;

  // gathering all the inputs at tester
  std::vector<c10::IValue> input_tensors;
  for (auto i : c10::irange(inputs.size())) {
    auto sender = runtime.pipeline()
                      ->inputs()
                      .at(i)
                      ->as<PipelineVal>()
                      ->getStage()
                      ->descriptor()
                      ->mesh.deviceIndices()
                      .at(0);
    buffer = {inputs.at(i).toTensor()};
    comm.sendRecv(tester, sender, buffer);
    input_tensors.push_back(buffer.at(0));
  }

  // gathering all the outputs at tester
  std::vector<at::Tensor> output_tensors;
  for (auto i : c10::irange(outputs.size())) {
    auto sender = runtime.pipeline()
                      ->outputs()
                      .at(i)
                      ->as<PipelineVal>()
                      ->getStage()
                      ->descriptor()
                      ->mesh.deviceIndices()
                      .at(0);
    buffer = {outputs.at(i)};
    comm.sendRecv(tester, sender, buffer);
    output_tensors.push_back(buffer.at(0));
  }

  if (comm.deviceId() == tester) {
    // execute the fusion on one device without pipeline scheduling
    Fusion& fusion = *fusion_ptr.get();
    FusionExecutorCache fec(std::move(fusion_ptr));
    auto ref_outputs = fec.runFusionWithInputs(inputs);

    // validation
    testValidate(
        &fusion,
        output_tensors,
        input_tensors,
        ref_outputs,
        __LINE__,
        __FILE__);
  }

  comm.barrier();
}

/* To run the following tests on several devices, pytorch must be installed
   with the flag USE_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun,
   e.g.: mpirun -np 6 ./build/bin/nvfuser_tests
   --gtest_filter=NVFuserTest.FusionMultiGPU_CUDA
   For now, we only support setups with one node.
*/

TEST_F(NVFuserTest, FusionMultiGPU_CUDA) {
  // ===========================================================
  //        FUSION
  // ===========================================================
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0_ = makeContigTensor(2);
  fusion.addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion.addOutput(tv3_);

  TensorView* tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv2a = set(tv2);
  TensorView* tv3 = sum(tv2a, {0});

  TensorView* tv4 = set(tv1);
  TensorView* tv4a = set(tv4);
  TensorView* tv5 = sum(tv4a, {0});

  TensorView* tv6 = set(tv2a);
  TensorView* tv7 = sum(tv6, {0});
  fusion.addOutput(tv7);

  TensorView* tv8 = set(tv3);
  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv7);
  TensorView* tv11 = add(tv8, tv9);
  TensorView* tv12 = add(tv11, tv10);
  TensorView* tv13 = sum(tv12, {0});
  fusion.addOutput(tv13);

  // ===========================================================
  //        PIPELINE SCHEDULING
  // ===========================================================
  /* Each TensorView must be assigned to one and only one stage
     WAR: if an intermediate TensorView is automatically added
          in the Fusion during Fusion definition,
          it also needs to be assigned manually to a stage */
  PipelineStageDescriptor stage0_, stage1_, stage0, stage1, stage2, stage3,
      stage4;
  stage0_.addVal({tv0_, tv1_});
  stage1_.addVal({tv2_, tv3_});
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv2a, tv3});
  stage2.addVal({tv4, tv4a, tv5});
  stage3.addVal({tv6, tv7});
  stage4.addVal({tv8, tv9, tv10, tv11, tv12, tv13});

  // binding each stage to a device mesh
  stage0_.mesh.set({5});
  stage1_.mesh.set({2, 4});
  stage0.mesh.set({0});
  stage1.mesh.set({0, 1, 4});
  stage2.mesh.set({1, 3});
  stage3.mesh.set({2});
  stage4.mesh.set({4, 5});

  PipelineDescriptor descriptor{.stage_descriptors{
      std::move(stage0_),
      std::move(stage1_),
      std::move(stage0),
      std::move(stage1),
      std::move(stage2),
      std::move(stage3),
      std::move(stage4)}}; // the order doesn't matter

  Pipeline pipeline(&fusion, std::move(descriptor));

  // ===========================================================
  //        COMMUNICATOR
  // ===========================================================

  int requested_world_size = 6;
  if (!comm.is_available() || comm.size() < requested_world_size) {
    GTEST_SKIP() << "This test needs distributed setting with at least "
                 << requested_world_size << " ranks";
  }

  // ===========================================================
  //        RUNTIME
  // ===========================================================

  MultiDeviceRuntime runtime(&pipeline, comm);

  // Create input tensors.
  // Note: each rank is binded to a different GPU
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  // Note: the concrete values are only used at the relevant ranks
  std::vector<c10::IValue> inputs{
      at::randn({3, 11}, options), at::randn({2, 7, 8}, options)};

  // Run the pipeline
  auto outputs = runtime.runWithInput(inputs);

  // ===========================================================
  //        VALIDATION
  // ===========================================================

  testValidateMultidevice(std::move(fusion_ptr), runtime, inputs, outputs);
}

} // namespace nvfuser

#endif
