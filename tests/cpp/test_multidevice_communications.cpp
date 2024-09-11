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

#include <ops/arith.h>
#include <ops/utils.h>

#include <iostream>

namespace nvfuser {

class CommunicationTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<CommunicatorBackend> {
 protected:
  CommunicationTest();
  void SetUp() override;

  void validate(at::Tensor obtained, at::Tensor expected);

  static constexpr DeviceIdxType kRoot = 0;
  static constexpr int kTensorSize = 1024;
  // This is so we test having multiple inflights collectives on the same
  // buffers. This emulates more accurately the type of workload we are
  // targeting.
  static constexpr int kNumRepetitions = 8;
  // TODO: test other reduction op types.
  static constexpr c10d::ReduceOp::RedOpType kReductionOp =
      c10d::ReduceOp::RedOpType::SUM;
  const DeviceMesh full_mesh_;
  const Team all_ranks_;
  c10d::Backend* backend_ = nullptr;
};

CommunicationTest::CommunicationTest()
    : full_mesh_(DeviceMesh::createForNumDevices(communicator_->size())),
      all_ranks_(full_mesh_.vector()) {}

void CommunicationTest::SetUp() {
  MultiDeviceTest::SetUp();

  const CommunicatorBackend backend_type = GetParam();
  if (!communicator_->isBackendAvailable(backend_type)) {
    GTEST_SKIP() << "Backend not available: " << backend_type;
  }
  // getBackendForTeam throws an error if the requested backend type isn't
  // available. Therefore, we call it after the isBackendAvailable check.
  backend_ = communicator_->getBackendForTeam(all_ranks_, backend_type);
}

void CommunicationTest::validate(at::Tensor obtained, at::Tensor expected) {
  EXPECT_TRUE(obtained.equal(expected))
      << "Device " << communicator_->deviceId() << " expected tensor:\n"
      << expected << "\nbut obtained tensor:\n"
      << obtained;
}

TEST_P(CommunicationTest, Gather) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto* communication = IrBuilder::create<Communication>(
      CommunicationType::Gather, out, in, all_ranks_, kRoot);

  at::Tensor input_tensor = at::empty({1, kTensorSize}, tensor_options);
  at::Tensor output_tensor =
      at::empty({communicator_->size(), kTensorSize}, tensor_options);
  for (auto repetition : c10::irange(kNumRepetitions)) {
    input_tensor.copy_(
        at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        (communicator_->deviceId() + 1) * repetition);
    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    work->wait();

    if (communicator_->deviceId() == kRoot) {
      at::Tensor ref = at::arange(kTensorSize, tensor_options).unsqueeze(0) +
          at::arange(1, communicator_->size() + 1, tensor_options)
                  .unsqueeze(1) *
              repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, Allgather) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Allgather, out, in, all_ranks_);

  at::Tensor input_tensor = at::empty({1, kTensorSize}, tensor_options);
  at::Tensor output_tensor =
      at::empty({communicator_->size(), kTensorSize}, tensor_options);
  for (auto repetition : c10::irange(kNumRepetitions)) {
    input_tensor.copy_(
        at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        (communicator_->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    work->wait();

    at::Tensor ref = at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        at::arange(1, communicator_->size() + 1, tensor_options).unsqueeze(1) *
            repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Scatter) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Scatter, out, in, all_ranks_, kRoot);

  at::Tensor input_tensor;
  if (communicator_->deviceId() == kRoot) {
    input_tensor =
        at::empty({communicator_->size(), kTensorSize}, tensor_options);
  }
  at::Tensor output_tensor = at::empty({1, kTensorSize}, tensor_options);

  for (auto repetition : c10::irange(kNumRepetitions)) {
    if (communicator_->deviceId() == kRoot) {
      input_tensor.copy_(
          at::arange(kTensorSize, tensor_options).unsqueeze(0) +
          at::arange(1, communicator_->size() + 1, tensor_options)
                  .unsqueeze(1) *
              repetition);
    }

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    work->wait();

    auto ref = at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        (communicator_->deviceId() + 1) * repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Broadcast) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Broadcast, out, in, all_ranks_, kRoot);

  at::Tensor input_tensor;
  if (communicator_->deviceId() == kRoot) {
    input_tensor = at::empty({kTensorSize}, tensor_options);
  }
  at::Tensor output_tensor = at::empty({kTensorSize}, tensor_options);
  for (auto repetition : c10::irange(kNumRepetitions)) {
    if (communicator_->deviceId() == kRoot) {
      input_tensor.copy_(at::arange(kTensorSize, tensor_options) + repetition);
    }

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    if (work != nullptr) {
      work->wait();
    }

    auto ref = at::arange(kTensorSize, tensor_options) + repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, SendRecv) {
  if (communicator_->size() < 2 || torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks.";
  }

  constexpr DeviceIdxType sender = 1;
  constexpr DeviceIdxType receiver = 0;

  const DeviceIdxType rank = communicator_->deviceId();
  if (rank != sender && rank != receiver) {
    return;
  }

  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::SendRecv, out, in, Team({sender, receiver}), sender);

  at::Tensor input_tensor;
  at::Tensor output_tensor;
  if (rank == sender) {
    input_tensor = at::empty({kTensorSize}, tensor_options);
  } else {
    NVF_ERROR(rank == receiver);
    output_tensor = at::empty({kTensorSize}, tensor_options);
  }

  c10d::Backend* backend =
      communicator_->getBackendForTeam(communication->team(), GetParam());
  for (auto repetition : c10::irange(kNumRepetitions)) {
    if (rank == sender) {
      input_tensor.copy_(at::arange(kTensorSize, tensor_options) + repetition);
    }

    auto work = postSingleCommunication(
        communication, rank, backend, input_tensor, output_tensor);
    work->wait();

    if (rank == receiver) {
      auto ref = at::arange(kTensorSize, tensor_options) + repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, SendRecvToSelf) {
  constexpr DeviceIdxType sender = 0;
  if (communicator_->deviceId() > 0) {
    // Only device 0 participates.
    return;
  }

  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = ops::newValLike(in, in->dtype())->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::SendRecv, out, in, Team({sender}), sender);

  at::Tensor input_tensor = at::empty({kTensorSize}, tensor_options);
  at::Tensor output_tensor = at::empty_like(input_tensor);

  c10d::Backend* backend =
      communicator_->getBackendForTeam(communication->team(), GetParam());
  for (auto repetition : c10::irange(kNumRepetitions)) {
    input_tensor.copy_(at::arange(kTensorSize, tensor_options) + repetition);

    postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend,
        input_tensor,
        output_tensor);

    auto ref = at::arange(kTensorSize, tensor_options) + repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, Reduce) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = newForReduction(in, {0});
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Reduce, out, in, all_ranks_, kRoot, kReductionOp);

  at::Tensor input_tensor = at::empty({1, kTensorSize}, tensor_options);
  at::Tensor output_tensor = at::empty({kTensorSize}, tensor_options);

  for (auto repetition : c10::irange(kNumRepetitions)) {
    input_tensor.copy_(
        at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        (communicator_->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    work->wait();

    if (communicator_->deviceId() == kRoot) {
      const int s = communicator_->size();
      auto ref = at::arange(kTensorSize, tensor_options) * s +
          s * (s + 1) / 2 * repetition;
      validate(output_tensor, ref);
    }
  }
}

TEST_P(CommunicationTest, Allreduce) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(2);
  in->setDeviceMesh(full_mesh_);
  auto* out = newForReduction(in, {0});
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Allreduce,
      out,
      in,
      all_ranks_,
      /*root=*/-1,
      kReductionOp);

  at::Tensor input_tensor = at::empty({1, kTensorSize}, tensor_options);
  at::Tensor output_tensor = at::empty({kTensorSize}, tensor_options);
  for (auto repetition : c10::irange(kNumRepetitions)) {
    input_tensor.copy_(
        at::arange(kTensorSize, tensor_options).unsqueeze(0) +
        (communicator_->deviceId() + 1) * repetition);

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
        input_tensor,
        output_tensor);
    work->wait();

    const int s = communicator_->size();
    auto ref = at::arange(kTensorSize, tensor_options) * s +
        s * (s + 1) / 2 * repetition;
    validate(output_tensor, ref);
  }
}

TEST_P(CommunicationTest, ReduceScatter) {
  hir::HostIrContainer container;
  FusionGuard fg(&container);
  auto* in = makeContigTensor(3);
  in->setDeviceMesh(full_mesh_);
  auto* out = newForReduction(in, {0});
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      out,
      in,
      all_ranks_,
      /*root=*/-1,
      kReductionOp,
      /*scattered_axis=*/1);

  const int num_devices = communicator_->size();
  const int device_id = communicator_->deviceId();
  at::Tensor unsharded_input_tensor =
      at::empty({num_devices, num_devices, kTensorSize}, tensor_options);
  at::Tensor input_tensor =
      unsharded_input_tensor.slice(0, device_id, device_id + 1);
  at::Tensor output_tensor = at::empty({1, kTensorSize}, tensor_options);

  for (auto repetition : c10::irange(kNumRepetitions)) {
    std::ignore = repetition;

    // Create a tensor with integer values to avoid rounding error so we can
    // validate using `equal` for more confidence.
    unsharded_input_tensor.copy_(at::randint(
        2, {num_devices, num_devices, kTensorSize}, tensor_options));

    auto work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend_,
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

class HangTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<CommunicatorBackend> {};

TEST_P(HangTest, MinimalTestHangSendRecv) {
  if (communicator_->size() != 2) {
    GTEST_SKIP() << "only supports 2 devices";
  }
  auto my_rank = communicator_->deviceId();
  auto peer_rank = 1 - communicator_->deviceId();

  auto options = at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  at::Tensor src_buffer = at::ones({1}, options) * my_rank;
  at::Tensor dst_buffer = at::empty({1}, options);
  std::vector<at::Tensor> src = {src_buffer};
  std::vector<at::Tensor> dst = {dst_buffer};

  std::vector<int64_t> all_devices = {0,1};
  c10d::Backend* world_communicator_ = communicator_->getBackendForTeam(all_devices, GetParam());
  c10::intrusive_ptr<c10d::Work> recv_h, send_h;
  if (my_rank == 0) {
    recv_h = world_communicator_->recv(dst, peer_rank, 1);
    world_communicator_->send(src, peer_rank, 0);
  } else {
    recv_h = world_communicator_->recv(dst, peer_rank, 0);
    world_communicator_->send(src, peer_rank, 1);
  }

  std::cout << "rank " <<  my_rank << " has finished posting" << std::endl;
  recv_h->wait();
  std::cout << "rank " <<  my_rank << " src = " << src << " dst = " << dst << std::endl;
}

TEST_P(HangTest, ThreeRanksTestHangSendRecv) {
  if (communicator_->size() != 3) {
    GTEST_SKIP() << "only supports 3 devices";
  }
  auto my_rank = communicator_->deviceId();
  auto send_peer_rank = (my_rank + 1) % 3;
  auto recv_peer_rank = (my_rank - 1 + 3) % 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  at::Tensor src_buffer = at::ones({1}, options) * my_rank;
  at::Tensor dst_buffer = at::empty({1}, options);
  at::Tensor allreduce_buffer = at::ones({1}, options);
  std::vector<at::Tensor> src = {src_buffer};
  std::vector<at::Tensor> dst = {dst_buffer};
  std::vector<at::Tensor> allreduce_buf = {allreduce_buffer};

  std::vector<int64_t> all_devices = {0,1,2};
  c10d::Backend* world_communicator_ = communicator_->getBackendForTeam(all_devices, GetParam());
  world_communicator_->allreduce(allreduce_buf)->wait();

  c10::intrusive_ptr<c10d::Work> recv_h, send_h;
  std::cout << "rank " <<  my_rank << " starts posting" << std::endl;
  if (my_rank == 0) {
    world_communicator_->send(src, send_peer_rank, my_rank);
    recv_h = world_communicator_->recv(dst, recv_peer_rank, recv_peer_rank);
  } else if (my_rank == 1) {
    recv_h = world_communicator_->recv(dst, recv_peer_rank, recv_peer_rank);
    world_communicator_->send(src, send_peer_rank, my_rank);
  } else {
    world_communicator_->send(src, send_peer_rank, my_rank);
    recv_h = world_communicator_->recv(dst, recv_peer_rank, recv_peer_rank);
  }

  std::cout << "rank " <<  my_rank << " has finished posting" << std::endl;
  std::cout << "rank=" <<  my_rank << ", send_peer_rank=" << send_peer_rank << ", recv_peer_rank=" << recv_peer_rank << std::endl;
  recv_h->wait();
  std::cout << "rank " <<  my_rank << " src = " << src << " dst = " << dst << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    HangTest,
    testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
    testing::PrintToStringParamName());

} // namespace nvfuser
