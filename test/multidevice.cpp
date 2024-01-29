// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
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
  if (getNvFuserEnv("MULTIDEVICE_TIME_PRINT")) {
    time_print_ = true;
  }
}

void MultiDeviceTest::SetUp() {
  NVFuserTest::SetUp();
  communicator = multidevice_env->communicator();
  debug_print = multidevice_env->debugPrint();
  do_barrier_at_test =
      multidevice_env->doBarrierAtTest() && communicator->is_available();
  time_print = multidevice_env->timePrint() && communicator->is_available();
  if (!communicator->is_available() || communicator->size() < 2 ||
      torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks";
  }
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
  recordEvent("init");
}

void MultiDeviceTest::TearDown() {
  if (do_barrier_at_test) {
    recordEvent("final barrier");
    communicator->barrier();
  }
  recordEvent("cleanup");
  if (time_print) {
    printTimes();
  }
  NVFuserTest::TearDown();
}

void MultiDeviceTest::recordEvent(const std::string name) {
  times.push_back(
      std::make_pair(name, std::chrono::high_resolution_clock::now()));
}

void MultiDeviceTest::printTimes() {
  std::stringstream ss;
  auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  ss << "Rank " << communicator->deviceId() << " -- test "
     << test_info->test_suite_name() << "." << test_info->name()
     << " -- Timestamps: {\n";
  for (auto i : c10::irange(times.size() - 1)) {
    auto [event_name, time] = times[i];
    auto [_, next_time] = times[i + 1];
    ss << "  " << event_name << ": "
       << std::chrono::duration_cast<std::chrono::milliseconds>(
              next_time - time)
              .count()
       << " ms\n";
  }
  ss << "}";
  std::cout << ss.str() << std::endl;
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

// Send a possibly sharded tensor to one "tester" device
void SendToTester(
    TensorView* tv,
    at::Tensor tensor,
    at::Tensor tester_tensor,
    DeviceIdxType tester,
    Communicator* communicator,
    bool debug_print) {
  auto mesh = tv->getDeviceMesh();
  if (isSharded(tv)) {
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
    if (communicator->deviceId() == sender ||
        communicator->deviceId() == tester) {
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

c10::IValue allocate_unsharded_input(
    DeviceIdxType tester,
    TensorView* tv,
    at::Tensor sharded_input) {
  // TODO: Extend to multi-dimension mesh
  std::vector<int64_t> unsharded_sizes;
  for (size_t i = 0; i < tv->nDims(); i++) {
    if (tv->axis(i)->isDeviceDim()) {
      unsharded_sizes.push_back(tv->getDeviceMesh().vector().size());
    } else {
      unsharded_sizes.push_back(sharded_input.size(i));
    }
  }
  at::Tensor unsharded_input =
      at::rand(unsharded_sizes, sharded_input.options());
  unsharded_input.index_put_({tester, "..."}, sharded_input.index({0, "..."}));
  return unsharded_input;
}
} // namespace

// Utility function used for validation in the tests
// It compares the given (possibly sharded) output with the result of the Fusion
// run on a single device with the given (possibly sharded) inputs
void PipelineTest::validate(DeviceIdxType tester, bool auto_schedule) {
  recordEvent("gather inputs at tester");
  // gathering all the inputs at tester
  std::vector<c10::IValue> unsharded_inputs;
  for (auto i : c10::irange(inputs.size())) {
    TensorView* tv = runtime->fusion()->inputs().at(i)->as<TensorView>();
    c10::IValue unsharded_input = isSharded(tv)
        ? allocate_unsharded_input(tester, tv, inputs.at(i).toTensor())
        : inputs.at(i).deepcopy();
    unsharded_inputs.push_back(unsharded_input);

    SendToTester(
        tv,
        inputs.at(i).toTensor(),
        unsharded_inputs.at(i).toTensor(),
        tester,
        runtime->comm(),
        debug_print);
  }

  std::unique_ptr<FusionExecutorCache> unsharded_fec;
  // allocate output buffers for the tester
  std::vector<at::Tensor> unsharded_outputs;
  std::unique_ptr<Fusion> fusion_copy;
  if (runtime->comm()->deviceId() == tester) {
    recordEvent("compile unsharded fusion and alloc output");
    fusion_copy = std::make_unique<Fusion>(*runtime->fusion());
    unshard(fusion_copy.get());
    unsharded_fec =
        std::make_unique<FusionExecutorCache>(std::move(fusion_copy));
    unsharded_outputs = unsharded_fec->allocOutputSpace(unsharded_inputs);
  } else {
    // On non-tester devices, these tensors won't be used.
    // we copy the local outputs for convenience
    unsharded_outputs = outputs;
  }

  recordEvent("gather outputs at tester");
  // gathering all the outputs at tester
  for (auto i : c10::irange(outputs.size())) {
    SendToTester(
        runtime->fusion()->outputs().at(i)->as<TensorView>(),
        outputs.at(i),
        unsharded_outputs.at(i),
        tester,
        runtime->comm(),
        debug_print);
  }

  if (runtime->comm()->deviceId() == tester) {
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

    // execute the fusion on one device without pipeline scheduling
    std::vector<at::Tensor> ref_outputs;
    if (auto_schedule) {
      recordEvent("run unsharded fusion");
      ref_outputs = unsharded_fec->runFusionWithInputs(unsharded_inputs);
    } else {
      recordEvent("compile unsharded fusion");
      FusionExecutor fe;
      fe.compileFusion(unsharded_fec->fusion(), unsharded_inputs);
      recordEvent("run unsharded fusion");
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

    recordEvent("validate unsharded fusion");
    testValidate(
        unsharded_fec->fusion(),
        unsharded_outputs,
        unsharded_inputs,
        ref_outputs,
        __LINE__,
        __FILE__);
  }
}

// Run and validate a pipeline
// with given (possibly sharded) inputs
void PipelineTest::execute() {
  if (debug_print && !communicator->deviceId()) {
    fusion->printKernel();
  }

  recordEvent("runtime instantiation");
  runtime =
      std::make_unique<MultiDeviceExecutor>(std::move(fusion), *communicator);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }

  recordEvent("run the multidevice fusion");
  outputs = runtime->runWithInput(inputs);

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
}

void PipelineTest::SetUp() {
  MultiDeviceTest::SetUp();
  fusion = std::make_unique<Fusion>();
  communicator->setDefaultBackend(CommunicatorBackend::nccl);
}

} // namespace nvfuser

#endif
