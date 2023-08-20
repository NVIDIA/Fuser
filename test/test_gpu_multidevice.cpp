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

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// We create the communicator globally for all the tests
Communicator comm;

// Send a possibly sharded tensor represented by a PipelineVal
// to one "tester" device
void SendToTester(
    PipelineVal* pVal,
    at::Tensor tensor,
    DeviceIdxType tester,
    Communicator& comm) {
  std::vector<at::Tensor> buffer;
  auto& mesh = pVal->getStage()->descriptor()->mesh;
  if (isParallelTypeDeviceDim(pVal->getOriginalVal()
                                  ->as<TensorView>()
                                  ->getRootDomain()
                                  .at(0)
                                  ->getParallelType())) {
    for (DeviceIdxType j : c10::irange(mesh.vector().size())) {
      buffer = {tensor.index({j, "..."})};
      auto sender = mesh.vector().at(j);
      comm.sendRecv(tester, sender, buffer);
    }
  } else {
    buffer = {tensor};
    auto sender = mesh.vector().at(0);
    comm.sendRecv(tester, sender, buffer);
  }
}

// Utility function used for validation in the tests
// It compares the given (possibly sharded) output with the result of the Fusion
// run on a single device with the given (possibly sharded) inputs
void testValidateMultidevice(
    std::unique_ptr<Fusion> fusion_ptr,
    MultiDeviceRuntime& runtime,
    const at::ArrayRef<c10::IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    bool print = true,
    DeviceIdxType tester = 1,
    bool validate = true,
    bool set_mem_type_to_global = true,
    bool auto_schedule = false) {
  // gathering all the inputs at tester
  for (auto i : c10::irange(inputs.size())) {
    SendToTester(
        runtime.pipeline()->inputs().at(i)->as<PipelineVal>(),
        inputs.at(i).toTensor(),
        tester,
        runtime.comm());
  }

  // gathering all the outputs at tester
  for (auto i : c10::irange(outputs.size())) {
    SendToTester(
        runtime.pipeline()->outputs().at(i)->as<PipelineVal>(),
        outputs.at(i),
        tester,
        runtime.comm());
  }

  if (runtime.comm().deviceId() == tester) {
    if (print) {
      std::stringstream ss;
      std::string indent = "  ";
      ss << "Obtained final outputs:{\n";
      for (auto& t : outputs) {
        ss << indent << t;
      }
      ss << "\n}";
      std::cout << ss.str() << std::endl;
    }

    // sets all the memory type to global to avoid an execution error
    if (set_mem_type_to_global) {
      for (auto tv : ir_utils::filterByType<TensorView>(fusion_ptr->vals())) {
        tv->setMemoryType(MemoryType::Global);
      }
    }

    // execute the fusion on one device without pipeline scheduling
    std::vector<at::Tensor> ref_outputs;
    Fusion& fusion = *fusion_ptr.get();
    if (auto_schedule) {
      FusionExecutorCache fec(std::move(fusion_ptr));
      ref_outputs = fec.runFusionWithInputs(inputs);
    } else {
      FusionExecutor fe;
      fe.compileFusion(&fusion, inputs);
      ref_outputs = fe.runFusion(inputs);
    }

    if (print) {
      std::stringstream ss;
      std::string indent = "  ";
      ss << "Expected outputs:{\n";
      for (auto& t : ref_outputs) {
        ss << indent << t;
      }
      ss << "\n}";
      std::cout << ss.str() << std::endl;
    }

    if (validate) {
      testValidate(&fusion, outputs, inputs, ref_outputs, __LINE__, __FILE__);
    }
  }
}

// Utility function used in the test to run and validate a given pipeline
// with given (possibly sharded) inputs
void executeAndTestPipeline(
    std::unique_ptr<Fusion> fusion_ptr,
    Pipeline& pipeline,
    Communicator& comm,
    std::vector<c10::IValue>& inputs,
    bool print = false) {
  if (print && !comm.deviceId()) {
    fusion_ptr->printKernel();
    std::cout << pipeline.toString() << std::endl;
  }

  MultiDeviceRuntime runtime(&pipeline, comm);
  auto error_msg = runtime.validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }

  auto outputs = runtime.runWithInput(inputs);

  if (print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << comm.deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  testValidateMultidevice(
      std::move(fusion_ptr), runtime, inputs, outputs, print);

  comm.barrier();
}

/* To run the following tests on several devices, pytorch must be installed
   with the flag USE_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun,
   e.g., on one node with 6 devices:
   mpirun -np 6 ./build/bin/nvfuser_tests
   --gtest_filter=NVFuserTest.FusionMultiGPU*
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
  stage0_.mesh = {5};
  stage1_.mesh = {2, 4};
  stage0.mesh = {0};
  stage1.mesh = {0, 1, 4};
  stage2.mesh = {1, 3};
  stage3.mesh = {2};
  stage4.mesh = {4, 5};

  PipelineDescriptor descriptor{.stage_descriptors{
      std::move(stage0_),
      std::move(stage1_),
      std::move(stage0),
      std::move(stage1),
      std::move(stage2),
      std::move(stage3),
      std::move(stage4)}}; // the order doesn't matter

  Pipeline pipeline(&fusion, std::move(descriptor));

  // Create input tensors.
  // Note: each process is binded to a different GPU
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  // Note: the concrete values are only used at the relevant ranks
  std::vector<c10::IValue> inputs{
      at::randn({3096, 1123}, options), at::randn({2048, 73, 81}, options)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionDidx_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  fusion.addOutput(tv1);

  // fusion.printKernel();

  if (!comm.is_available())
    GTEST_SKIP() << "distributed setting not available";

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4}, options) + comm.deviceId()};

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), inputs);
  auto ref_outputs = fe.runFusion(inputs);

  if (comm.is_available()) {
    comm.barrier();
  }
}

TEST_F(NVFuserTest, FusionMultiGPU_Gather_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv0->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0});
  stage1.addVal({tv1});

  stage0.mesh = {0, 1};
  stage0.auto_schedule = true;
  stage1.mesh = {0};
  stage1.auto_schedule = true;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({2, 11}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_Gather2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage0.auto_schedule = false;
  stage1.mesh = {0};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{
      at::ones({4, 3, 2, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_GatherMultipleDestinations_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage0.mesh = {0, 1, 2, 3};
  stage0.auto_schedule = false;
  stage1.mesh = {0, 4};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{
      at::ones({4, 3, 2, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_GatherToExternalRoot_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {1, 2, 3};
  stage0.auto_schedule = false;
  stage1.mesh = {0};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{
      at::ones({3, 3, 2, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_AllGather_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage0.auto_schedule = false;
  stage1.mesh = {0, 1, 2, 3};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{
      at::ones({4, 3, 2, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_Scatter_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0});
  stage1.addVal({tv1});
  stage0.mesh = {0};
  stage0.auto_schedule = true;
  stage1.mesh = {0, 1, 2, 3};
  stage1.auto_schedule = true;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_ScatterToExternalRoot_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0});
  stage1.addVal({tv1});
  stage0.mesh = {0};
  stage0.auto_schedule = true;
  stage1.mesh = {1, 2, 3};
  stage1.auto_schedule = true;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({3, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_BcastBothSidesParallelized_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0});
  stage1.addVal({tv1});
  stage0.mesh = {0, 1, 3, 4};
  stage0.auto_schedule = true;
  stage1.mesh = {1, 2, 3, 4};
  stage1.auto_schedule = true;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4, 5}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(NVFuserTest, FusionMultiGPU_BcastLocalCopies_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(5);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {1});
  TensorView* tv6 = set(tv5);
  TensorView* tv7 = sum(tv6, {1});
  fusion.addOutput(tv7);

  PipelineStageDescriptor stage0, stage1, stage2, stage3;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage2.addVal({tv4, tv5});
  stage3.addVal({tv6, tv7});
  stage0.mesh = {0, 1};
  stage1.mesh = {0};
  stage2.mesh = {0, 1, 2, 3};
  stage3.mesh = {0, 1, 2, 3};

  PipelineDescriptor descriptor{.stage_descriptors{
      std::move(stage0),
      std::move(stage1),
      std::move(stage2),
      std::move(stage3)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{
      at::ones({4, 5, 7, 3, 2}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

} // namespace nvfuser

#endif
