// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/builder.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <tests/cpp/multidevice.h>

#include <iostream>

namespace nvfuser {

class CommunicationTest
    : public MultiDeviceTest,
      public ::testing::WithParamInterface<CommunicatorBackend> {
 protected:
  CommunicationTest();
  void SetUp() override;

  void validate(at::Tensor obtained, at::Tensor expected);

  static constexpr DeviceIdxType root = 0;
  static constexpr int tensor_size = 1024;
  // This is so we test having multiple inflights collectives on the same
  // buffers. This emulates more accurately the type of workload we are
  // targeting.
  static constexpr int num_repetitions = 8;
  // TODO: test other reduction op types.
  static constexpr c10d::ReduceOp::RedOpType red_op =
      c10d::ReduceOp::RedOpType::SUM;
  CommParams params;
  std::vector<DeviceIdxType> all_ranks;
  c10::intrusive_ptr<c10d::Backend> backend;
  IrContainer container;
};

CommunicationTest::CommunicationTest() {
  all_ranks = std::vector<DeviceIdxType>(communicator->size());
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
  backend = communicator->getBackendForTeam(all_ranks, GetParam());
}

void CommunicationTest::SetUp() {
  MultiDeviceTest::SetUp();

  if (!communicator->isBackendAvailable(GetParam())) {
    GTEST_SKIP() << "Backend not available";
  }
}

void CommunicationTest::validate(at::Tensor obtained, at::Tensor expected) {
  EXPECT_TRUE(obtained.equal(expected))
      << "Device " << communicator->deviceId() << " expected tensor:\n"
      << expected << "\nbut obtained tensor:\n"
      << obtained;
}

TEST_P(CommunicationTest, Gather) {
  params.type = CommunicationType::Gather;
  params.root = root;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor = at::empty({1, tensor_size}, tensor_options);
  at::Tensor output_tensor =
      at::empty({communicator->size(), tensor_size}, tensor_options);
  for (auto repetition : c10::irange(num_repetitions)) {
    input_tensor.copy_(
        at::arange(tensor_size, tensor_options).unsqueeze(0) +
        (communicator->deviceId() + 1) * repetition);
    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    if (communicator->deviceId() == root) {
      at::Tensor ref = at::arange(tensor_size, tensor_options).unsqueeze(0) +
          at::arange(1, communicator->size() + 1, tensor_options).unsqueeze(1) *
              repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, Allgather) {
  params.type = CommunicationType::Allgather;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor = at::empty({1, tensor_size}, tensor_options);
  at::Tensor output_tensor =
      at::empty({communicator->size(), tensor_size}, tensor_options);
  for (auto repetition : c10::irange(num_repetitions)) {
    input_tensor.copy_(
        at::arange(tensor_size, tensor_options).unsqueeze(0) +
        (communicator->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    at::Tensor ref = at::arange(tensor_size, tensor_options).unsqueeze(0) +
        at::arange(1, communicator->size() + 1, tensor_options).unsqueeze(1) *
            repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Scatter) {
  params.type = CommunicationType::Scatter;
  params.root = root;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor;
  if (communicator->deviceId() == root) {
    input_tensor =
        at::empty({communicator->size(), tensor_size}, tensor_options);
  }
  at::Tensor output_tensor = at::empty({1, tensor_size}, tensor_options);

  for (auto repetition : c10::irange(num_repetitions)) {
    if (communicator->deviceId() == root) {
      input_tensor.copy_(
          at::arange(tensor_size, tensor_options).unsqueeze(0) +
          at::arange(1, communicator->size() + 1, tensor_options).unsqueeze(1) *
              repetition);
    }

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    auto ref = at::arange(tensor_size, tensor_options).unsqueeze(0) +
        (communicator->deviceId() + 1) * repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Broadcast) {
  params.type = CommunicationType::Broadcast;
  params.root = root;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor;
  if (communicator->deviceId() == root) {
    input_tensor = at::empty({tensor_size}, tensor_options);
  }
  at::Tensor output_tensor = at::empty({tensor_size}, tensor_options);
  for (auto repetition : c10::irange(num_repetitions)) {
    if (communicator->deviceId() == root) {
      input_tensor.copy_(at::arange(tensor_size, tensor_options) + repetition);
    }

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    if (work != nullptr) {
      work->wait();
    }

    auto ref = at::arange(tensor_size, tensor_options) + repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, SendRecv) {
  if (GetParam() == CommunicatorBackend::ucc) {
    GTEST_SKIP() << "Disabling because of UCC hangs, see issue #2091";
  }
  if (communicator->size() < 2 || torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks.";
  }

  constexpr DeviceIdxType sender = 0;
  constexpr DeviceIdxType receiver = 1;
  if (communicator->deviceId() > 1) {
    // Only devices 0 and 1 participate.
    return;
  }

  params.type = CommunicationType::SendRecv;
  params.root = sender;
  params.team = {0, 1};
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor;
  at::Tensor output_tensor;
  if (communicator->deviceId() == sender) {
    input_tensor = at::empty({tensor_size}, tensor_options);
  } else {
    NVF_ERROR(communicator->deviceId() == receiver);
    output_tensor = at::empty({tensor_size}, tensor_options);
  }

  for (auto repetition : c10::irange(num_repetitions)) {
    if (communicator->deviceId() == sender) {
      input_tensor.copy_(at::arange(tensor_size, tensor_options) + repetition);
    }

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    if (communicator->deviceId() == receiver) {
      auto ref = at::arange(tensor_size, tensor_options) + repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, SendRecvToSelf) {
  constexpr DeviceIdxType sender = 0;
  if (communicator->deviceId() > 0) {
    // Only device 0 participates.
    return;
  }

  params.type = CommunicationType::SendRecv;
  params.root = sender;
  params.team = {0};
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor = at::empty({tensor_size}, tensor_options);
  at::Tensor output_tensor = at::empty_like(input_tensor);

  for (auto repetition : c10::irange(num_repetitions)) {
    input_tensor.copy_(at::arange(tensor_size, tensor_options) + repetition);

    postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);

    auto ref = at::arange(tensor_size, tensor_options) + repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Reduce) {
  params.type = CommunicationType::Reduce;
  params.redOp = red_op;
  params.root = root;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor = at::empty({1, tensor_size}, tensor_options);
  at::Tensor output_tensor = at::empty({tensor_size}, tensor_options);

  for (auto repetition : c10::irange(num_repetitions)) {
    input_tensor.copy_(
        at::arange(tensor_size, tensor_options).unsqueeze(0) +
        (communicator->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    if (communicator->deviceId() == root) {
      const int s = communicator->size();
      auto ref = at::arange(tensor_size, tensor_options) * s +
          s * (s + 1) / 2 * repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, Allreduce) {
  params.type = CommunicationType::Allreduce;
  params.redOp = red_op;
  params.team = all_ranks;
  auto communication = IrBuilder::create<Communication>(&container, params);

  at::Tensor input_tensor = at::empty({1, tensor_size}, tensor_options);
  at::Tensor output_tensor = at::empty({tensor_size}, tensor_options);
  for (auto repetition : c10::irange(num_repetitions)) {
    input_tensor.copy_(
        at::arange(tensor_size, tensor_options).unsqueeze(0) +
        (communicator->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    const int s = communicator->size();
    auto ref = at::arange(tensor_size, tensor_options) * s +
        s * (s + 1) / 2 * repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, ReduceScatter) {
  params.type = CommunicationType::ReduceScatter;
  params.redOp = red_op;
  params.root = root;
  params.team = all_ranks;
  params.scattered_axis = 1;
  auto communication = IrBuilder::create<Communication>(&container, params);

  const int num_devices = communicator->size();
  const int device_id = communicator->deviceId();
  at::Tensor unsharded_input_tensor =
      at::empty({num_devices, num_devices, tensor_size}, tensor_options);
  at::Tensor input_tensor =
      unsharded_input_tensor.slice(0, device_id, device_id + 1);
  at::Tensor output_tensor = at::empty({1, tensor_size}, tensor_options);

  for (auto repetition : c10::irange(num_repetitions)) {
    std::ignore = repetition;

    // Create a tensor with integer values to avoid rounding error so we can
    // validate using `equal` for more confidence.
    unsharded_input_tensor.copy_(at::randint(
        2, {num_devices, num_devices, tensor_size}, tensor_options));

    auto work = postSingleCommunication(
        communication,
        communicator->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    work->wait();

    auto ref =
        unsharded_input_tensor.sum({0}).slice(0, device_id, device_id + 1);
    validate(output_tensor, ref);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    CommunicationTest,
    testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
    testing::PrintToStringParamName());

} // namespace nvfuser
