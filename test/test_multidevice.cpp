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
#include <test/multidevice.h>
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
Communicator MultiDeviceTest::comm = {};

// Send a possibly sharded tensor represented by a PipelineVal
// to one "tester" device
void SendToTester(
    PipelineVal* pVal,
    at::Tensor tensor,
    DeviceIdxType tester,
    Communicator& comm) {
  std::vector<at::Tensor> buffer;
  auto& mesh = pVal->getStage()->descriptor()->mesh;
  //check if the tv is sharded accross devices
  if (pVal->getOriginalVal()->as<TensorView>()->axis(0)->isDevice()) {
    // If needed, we first do a local copy at the tester of a slice of the tensor
    auto it = std::find(mesh.vector().begin(), mesh.vector().end(), tester);
    if (it != mesh.vector().end() && comm.deviceId() == tester) {
      auto i = std::distance(mesh.vector().begin(), it);
      tensor.select(0, i) = tensor.select(0, 0);
    }
    // We send/recv to the tester all the tensor slices
    std::cout << "RANK " << comm.deviceId() << " sendrecv about to index " << pVal->getOriginalVal() <<" with buf "<< tensor << std::endl;
    for (DeviceIdxType j : c10::irange(mesh.vector().size())) {
      buffer = {tensor.index({comm.deviceId() == tester ? j : 0, "..."})};
      auto sender = mesh.vector().at(j);
      if (tester != sender && (comm.deviceId() == sender || comm.deviceId() == tester)) {
        comm.sendRecv(tester, sender, buffer)->wait();
      }
    }
  } else {
    // If the tensor is not sharded, we send/recv the whole buffer
    buffer = {tensor};
    auto sender = mesh.vector().at(0);
    if (tester != sender && (comm.deviceId() == sender || comm.deviceId() == tester)) {
      comm.sendRecv(tester, sender, buffer)->wait();
    }
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
    bool print,
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

  if (runtime.comm().deviceId() != tester) {
    return;
  }

  if (print) {
    fusion_ptr->printKernel();

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
      for (auto i : c10::irange(tv->domain()->nDims())) {
        tv->axis(i)->parallelize(ParallelType::Serial);
      }
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
    // testValidate(&fusion, outputs, inputs, ref_outputs, __LINE__, __FILE__);
    for (auto i : c10::irange(outputs.size())) {
      auto obtained = outputs.at(i);
      auto ref = ref_outputs.at(i);
      EXPECT_TRUE(obtained.allclose(ref, 1e-2))
      << "Device "
      << runtime.comm().deviceId() 
      << " expected tensor "
      << ref
      << "\nbut obtained tensor: "
      << obtained;
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
    bool print = true) {
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
  std::cout << "!!!RANK " << comm.deviceId() << " ABOUT TO TEST" << std::endl;

  testValidateMultidevice(
      std::move(fusion_ptr), runtime, inputs, outputs, print);

  comm.barrier();
}

/* To run the following tests on several devices, pytorch must be installed
   with the flag USE_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun
   on a node having at least 6 GPUs,
   e.g.: mpirun -np 6 build/nvfuser_tests
   --gtest_filter=MultiDeviceTest.Pipeline
*/

TEST_F(MultiDeviceTest, Pipeline) {
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
      at::randn({9, 7}, options), at::randn({3, 5, 11}, options)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(MultiDeviceTest, Didx) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(3);
  // TensorView* tv0_ = makeContigTensor(4);
  fusion.addInput(tv0);
  // fusion.addInput(tv0_);
  TensorView* tv1 = add(tv0, tv0);
  // TensorView* tv2 = sum(tv0, {0});
  // TensorView* tv2 = add(tv0_, tv0_);
  // TensorView* tv3 = add(tv0_, tv0_);
  // TensorView* tv4 = add(tv0_, tv0_);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  fusion.addOutput(tv1);
  // fusion.addOutput(tv2);
  // fusion.addOutput(tv3);
  // fusion.addOutput(tv4);

  fusion.printKernel();

  if (!comm.is_available())
    GTEST_SKIP() << "distributed setting not available";

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4, 7, 11}, options) + comm.deviceId()};

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), inputs);
  auto ref_outputs = fe.runFusion(inputs);

  if (comm.is_available()) {
    comm.barrier();
  }
}

TEST_F(MultiDeviceTest, Pipeline_Gather) {
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

TEST_F(MultiDeviceTest, Pipeline_Gather2) {
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

TEST_F(MultiDeviceTest, Pipeline_GatherMultipleDestinations) {
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

TEST_F(MultiDeviceTest, Pipeline_GatherToExternalRoot) {
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

TEST_F(MultiDeviceTest, Pipeline_AllGather) {
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

TEST_F(MultiDeviceTest, Pipeline_Scatter) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv2->axis(0)->parallelize(ParallelType::DIDx);
  tv3->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage0.mesh = {0};
  stage0.auto_schedule = false;
  stage1.mesh = {0, 1, 2, 3};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4, 5, 6, 7}, options) * comm.deviceId()};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(MultiDeviceTest, Pipeline_ScatterToExternalRoot) {
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

TEST_F(MultiDeviceTest, Pipeline_BcastBothSidesParallelized) {
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

TEST_F(MultiDeviceTest, Pipeline_BcastLocalCopies) {
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

TEST_F(MultiDeviceTest, Pipeline_Reduce) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage0.auto_schedule = false;
  stage1.mesh = {0, 1};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({4, 3, 1, 2}, options) * (comm.deviceId() + 1)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(MultiDeviceTest, Pipeline_ReduceToExternalRoot) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0, stage1;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1};
  stage0.auto_schedule = false;
  stage1.mesh = {2};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones({2, 3, 1, 2}, options) * (comm.deviceId() + 1)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(MultiDeviceTest, Pipeline_Allreduce) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
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
  std::vector<c10::IValue> inputs{at::ones({4, 3, 1, 2}, options) * (comm.deviceId() + 1)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}

TEST_F(MultiDeviceTest, Pipeline_ReduceScatter) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  tv2->axis(1)->parallelize(ParallelType::DIDx); //axis(0) is the "reduce" axis from previous tensor
  tv3->axis(0)->parallelize(ParallelType::DIDx);

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
  std::vector<c10::IValue> inputs{at::ones({4, 4, 1, 2}, options) * (comm.deviceId() + 1)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs);
}


TEST_F(MultiDeviceTest, Overlap) {
  // In this example we demonstrate how we can apply the optimization
  // described in 
  // Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models (acm.org)
  // https://dl.acm.org/doi/pdf/10.1145/3567955.3567959
  // We simplify the setting as much as possible by considering a multi-device Pipeline with
  // a simple "Gather" followed by a dependent compute. The paper suggest to slice those
  // two operation and to interleave them to achieve better overlap. Consider the following Pipeline:

  // /* Stage 0 */
  // TensorView* tv0 = makeContigTensor(3);
  // fusion.addInput(tv0);
  // TensorView* tv1 = sum(tv0, {2});
  // /* Stage 1 */
  // TensorView* tv2 = set(tv1); // is lowered to a "Gather" communication
  // TensorView* tv3 = sum(tv2, {1});
  // fusion.addOutput(tv3);

  // tv0->axis(0)->parallelize(ParallelType::DIDx);
  // tv1->axis(0)->parallelize(ParallelType::DIDx);

  // PipelineStageDescriptor stage0, stage1;
  // stage0.addVal({tv0, tv1});
  // stage1.addVal({tv2, tv3});

  // stage0.mesh = {0, 1, 2, 3, 4, 5, 6, 7};
  // stage0.auto_schedule = false;
  // stage1.mesh = {0};
  // stage1.auto_schedule = false;

  const int64_t number_of_devices = comm.size();
  constexpr int64_t number_of_slices = 4;
  constexpr int64_t extent_of_axis2 = 1024;
  constexpr int64_t extent_of_slice = extent_of_axis2 / number_of_slices;
  const std::vector<int64_t> input_extents = {number_of_devices, 7, extent_of_axis2, 3};
  assert(!(extent_of_axis2 % number_of_slices)); // for simplicity

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  PipelineStageDescriptor stage0, stage1;
  // containers used later for adding ranges of tvs directly to the stages
  std::unordered_set<Val*> from_stage0, from_stage1;
  std::vector<Val*> to_stage0, to_stage1;

  TensorView* tv0 = makeConcreteTensor(input_extents);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {3});
  from_stage0.insert(tv0);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  TensorView *tv1x, *tv2x, *tv3x;
  std::vector<TensorView*> tv3_slices;
  std::vector<Slice> slices {3};
  for (int i = 0; i < number_of_slices; i++) {
    slices.at(2).start = IrBuilder::create<Val>(i * extent_of_slice);
    slices.at(2).stop = IrBuilder::create<Val>((i+1) * extent_of_slice);
    tv1x = slice(tv1, slices);
    tv1x->axis(0)->parallelize(ParallelType::DIDx);
    to_stage0.push_back(tv1x);

    tv2x = set(tv1x);
    from_stage1.insert(tv2x);
    tv3x = sum(tv2x, {1});
    tv3_slices.push_back(tv3x);
  }
  TensorView* tv3 = cat(tv3_slices, 1);
  fusion.addOutput(tv3);
  to_stage1.push_back(tv3);

  //instead of using "slice/cat" it would be nicer to split the dimension and use "select/stack", but "stack" is not implemented in nvFuser at the moment

  stage0.addRange(fusion_ptr.get(), from_stage0, to_stage0);
  stage1.addRange(fusion_ptr.get(), from_stage1, to_stage1);
  std::vector<DeviceIdxType> devices(number_of_devices);
  std::iota(devices.begin(), devices.end(), 0);
  stage0.mesh = devices;
  stage0.auto_schedule = false;
  stage1.mesh = {0};
  stage1.auto_schedule = false;

  PipelineDescriptor descriptor {
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  Pipeline pipeline(&fusion, std::move(descriptor));

  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());
  std::vector<c10::IValue> inputs{at::ones(input_extents, options) * (comm.deviceId() + 1)};

  executeAndTestPipeline(std::move(fusion_ptr), pipeline, comm, inputs, true);
}


} // namespace nvfuser

#endif
