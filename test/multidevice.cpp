// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/all_nodes.h>
#include <fusion_segmenter.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
#include <test/multidevice.h>
#include <test/validator.h>
#include <torch/cuda.h>

namespace nvfuser {

auto multidevice_env = static_cast<MultiDeviceEnvironment*>(
    testing::AddGlobalTestEnvironment(new MultiDeviceEnvironment));

void MultiDeviceEnvironment::SetUp() {
  communicator_ = std::make_unique<Communicator>();
  if (getNvFuserEnv("MULTIDEVICE_DEBUG_PRINT")) {
    debug_print_ = true;
  }
  if (getNvFuserEnv("MULTIDEVICE_DEBUG_BARRIER")) {
    do_barrier_at_test_ = true;
  }
}

void MultiDeviceEnvironment::TearDown() {
  if (communicator_->is_available()) {
    communicator_->barrier();
  }
  communicator_.reset();
}

void MultiDeviceTest::SetUp() {
  NVFuserTest::SetUp();
  communicator = multidevice_env->communicator();
  if (!communicator->is_available() || communicator->size() < 2 ||
      torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks";
  }
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
  debug_print = multidevice_env->debugPrint();
  do_barrier_at_test = multidevice_env->doBarrierAtTest();
}

void MultiDeviceTest::TearDown() {
  if (do_barrier_at_test && communicator->is_available()) {
    communicator->barrier();
  }
  NVFuserTest::TearDown();
}

void CommunicationTest::SetUp() {
  MultiDeviceTest::SetUp();
  if (!communicator->isBackendAvailable(GetParam())) {
    GTEST_SKIP() << "Backend not available";
  }
  all_ranks = std::vector<DeviceIdxType>(communicator->size());
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
}

void CommunicationTest::validate(at::Tensor obtained, at::Tensor expected) {
  NVF_ERROR(
      obtained.equal(expected),
      "Device ",
      communicator->deviceId(),
      " expected tensor:\n",
      expected,
      "\nbut obtained tensor:\n",
      obtained);
}

void CommunicationTest::resetDstBuffers() {
  for (auto& buf : params.dst_bufs) {
    buf.copy_(at::full(tensor_size, nan(""), tensor_options));
  }
}

namespace {

void unshardTv(TensorView* tv) {
  for (IterDomain* id : tv->getLeafDomain()) {
    if (id->isDeviceDim()) {
      id->parallelize(ParallelType::Serial);
    }
  }
}

void doSendRecv(
    DeviceIdxType sender,
    DeviceIdxType receiver,
    at::Tensor send_buf,
    at::Tensor recv_buf,
    Communicator* communicator) {
  CommParams params;
  params.root = sender;
  if (sender == receiver) {
    params.team = {sender};
  } else {
    params.team = {sender, receiver};
  }
  if (send_buf.numel()) {
    params.src_bufs = {send_buf};
  }
  if (recv_buf.numel()) {
    params.dst_bufs = {recv_buf};
  }
  auto work = SendRecv(params).post(*communicator);
  if (work) {
    work->wait();
  }
}

// Send a possibly sharded tensor represented by a PipelineVal
// to one "tester" device
void SendToTester(
    PipelineVal* pVal,
    at::Tensor tensor,
    at::Tensor tester_tensor,
    DeviceIdxType tester,
    Communicator* communicator,
    bool debug_print) {
  auto& mesh = pVal->getStage()->descriptor()->mesh;
  if (isSharded(pVal->getOriginalVal()->as<TensorView>())) {
    for (DeviceIdxType j : c10::irange(mesh.vector().size())) {
      at::Tensor send_buf, recv_buf;
      auto sender = mesh.vector().at(j);
      if (communicator->deviceId() == sender ||
          communicator->deviceId() == tester) {
        if (communicator->deviceId() == sender) {
          send_buf = tensor.index({0, "..."});
        }
        if (communicator->deviceId() == tester) {
          recv_buf = tester_tensor.index({j, "..."});
        }
        doSendRecv(sender, tester, send_buf, recv_buf, communicator);
      }
    }
  } else {
    at::Tensor send_buf, recv_buf;
    auto sender = mesh.vector().at(0);
    if (tester != sender &&
        (communicator->deviceId() == sender ||
         communicator->deviceId() == tester)) {
      if (communicator->deviceId() == sender) {
        send_buf = tensor;
      }
      if (communicator->deviceId() == tester) {
        recv_buf = tester_tensor;
      }
      doSendRecv(sender, tester, send_buf, recv_buf, communicator);
    }
  }
}

// Utility function used for validation in the tests
// It compares the given (possibly sharded) output with the result of the Fusion
// run on a single device with the given (possibly sharded) inputs
void testValidateMultidevice(
    MultiDeviceRuntime& runtime,
    const at::ArrayRef<c10::IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    Communicator* communicator,
    bool debug_print,
    DeviceIdxType tester = 0,
    bool validate = true,
    bool auto_schedule = false) {
  std::vector<c10::IValue> unsharded_inputs;
  std::vector<at::Tensor> unsharded_outputs;

  // gathering all the inputs at tester
  for (auto i : c10::irange(inputs.size())) {
    c10::IValue unsharded_input = inputs.at(i).deepcopy();
    unsharded_inputs.push_back(unsharded_input);
    SendToTester(
        runtime.pipeline()->inputs().at(i)->as<PipelineVal>(),
        inputs.at(i).toTensor(),
        unsharded_inputs.at(i).toTensor(),
        tester,
        communicator, debug_print);
  }

  // gathering all the outputs at tester
  for (auto i : c10::irange(outputs.size())) {
    at::Tensor unsharded_output = at::clone(outputs.at(i));
    unsharded_outputs.push_back(unsharded_output);
    SendToTester(
        runtime.pipeline()->outputs().at(i)->as<PipelineVal>(),
        outputs.at(i),
        unsharded_outputs.at(i),
        tester,
        communicator, debug_print);
  }

  if (communicator->deviceId() == tester) {
    if (debug_print) {
      std::stringstream ss;
      std::string indent = "  ";
      ss << "Obtained final outputs:{\n";
      for (auto& t : unsharded_outputs) {
        ss << indent << t;
      }
      ss << "\n}\n";
      ss << "Reference (unsharded) input:{\n";
      for (auto& t : unsharded_inputs) {
        ss << indent << t;
      }
      ss << "\n}";
      std::cout << ss.str() << std::endl;
    }

    auto fusion_ptr = runtime.pipeline()->originalFusion();
    for (auto tv : ir_utils::filterByType<TensorView>(fusion_ptr->vals())) {
      unshardTv(tv);
    }

    // execute the fusion on one device without pipeline scheduling
    std::vector<at::Tensor> ref_outputs;
    if (auto_schedule) {
      auto fusion_unique_ptr = std::make_unique<Fusion>(*fusion_ptr);
      FusionExecutorCache fec(std::move(fusion_unique_ptr));
      ref_outputs = fec.runFusionWithInputs(unsharded_inputs);
    } else {
      FusionExecutor fe;
      fe.compileFusion(fusion_ptr, unsharded_inputs);
      ref_outputs = fe.runFusion(unsharded_inputs);
    }

    if (debug_print) {
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
      testValidate(fusion_ptr, unsharded_outputs, unsharded_inputs, ref_outputs, __LINE__, __FILE__);
    }
  }
}

// Run and validate a pipeline
// with given (possibly sharded) inputs
void executeAndValidatePipeline(
    Pipeline& pipeline,
    std::vector<c10::IValue>& inputs,
    Communicator* communicator,
    bool debug_print) {
  if (debug_print && !communicator->deviceId()) {
    pipeline.originalFusion()->printKernel();
    std::cout << pipeline.toString() << std::endl;
  }

  MultiDeviceRuntime runtime(&pipeline, *communicator);
  auto error_msg = runtime.validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }

  auto outputs = runtime.runWithInput(inputs);

  if (debug_print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator->deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  testValidateMultidevice(runtime, inputs, outputs, communicator, debug_print);
}

} // namespace

void PipelineTest::SetUp() {
  MultiDeviceTest::SetUp();
  fusion = std::make_unique<Fusion>();
  communicator->setDefaultBackend(CommunicatorBackend::nccl);
}

void PipelineTest::validate() {
  executeAndValidatePipeline(*pipeline, inputs, communicator, debug_print);
}

void PipelineTestTwoStages::SetUp() {
  PipelineTest::SetUp();
  auto [backend, mesh0, mesh1, is_stage0_sharded, is_stage1_sharded] =
      GetParam();
  if (!communicator->isBackendAvailable(backend)) {
    GTEST_SKIP() << "Backend not available";
  }
  communicator->setDefaultBackend(backend);

  int64_t first_axis_extent = 16;
  if (is_stage0_sharded) {
    first_axis_extent = mesh0.vector().size();
  } else if (is_stage1_sharded) {
    first_axis_extent = mesh1.vector().size();
  }
  const std::vector<int64_t> input_sizes = {first_axis_extent, 4, 3, 5};

  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeConcreteTensor(input_sizes);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {2});
  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  PipelineStageDescriptor stage0(false), stage1(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage0.mesh = mesh0;
  stage1.mesh = mesh1;
  tv0->setDeviceMesh(&mesh0);
  tv1->setDeviceMesh(&mesh0);
  tv2->setDeviceMesh(&mesh1);
  tv3->setDeviceMesh(&mesh1);
  if (is_stage0_sharded) {
    tv0->axis(0)->parallelize(ParallelType::DIDx);
    tv1->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_stage1_sharded) {
    tv2->axis(0)->parallelize(ParallelType::DIDx);
    tv3->axis(0)->parallelize(ParallelType::DIDx);
  }

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};
  pipeline = std::make_unique<Pipeline>(std::move(fusion), std::move(descriptor));

  inputs = {
      at::ones(input_sizes, tensor_options) *
      communicator->deviceId()};

  if (!communicator->deviceId() && debug_print) {
    std::cout << pipeline->toString()
      << toString(pipeline->sf_.get())
      << std::endl;
  }
  validate();
}

} // namespace nvfuser

#endif
