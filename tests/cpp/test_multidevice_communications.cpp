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
#include <tests/cpp/validator.h>

#include <ops/all_ops.h>
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
  for (auto repetition : arange(kNumRepetitions)) {
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
  for (auto repetition : arange(kNumRepetitions)) {
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

  for (auto repetition : arange(kNumRepetitions)) {
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
  for (auto repetition : arange(kNumRepetitions)) {
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

  if (GetParam() == CommunicatorBackend::kUcc) {
    GTEST_SKIP() << "TODO(#3120): investigate why this test hangs on H100";
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
  for (auto repetition : arange(kNumRepetitions)) {
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
  for (auto repetition : arange(kNumRepetitions)) {
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

  for (auto repetition : arange(kNumRepetitions)) {
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
  for (auto repetition : arange(kNumRepetitions)) {
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
      kReductionOp);

  const int num_devices = communicator_->size();
  const int device_id = communicator_->deviceId();
  at::Tensor unsharded_input_tensor =
      at::empty({num_devices, num_devices, kTensorSize}, tensor_options);
  at::Tensor input_tensor =
      unsharded_input_tensor.slice(0, device_id, device_id + 1);
  at::Tensor output_tensor = at::empty({1, kTensorSize}, tensor_options);

  for (auto repetition : arange(kNumRepetitions)) {
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
    testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
    testing::PrintToStringParamName());

using P2PCommunicationTest = MultiDeviceTest;

TEST_F(P2PCommunicationTest, CudaComm) {
  static constexpr int kTensorSize = 8;
  static constexpr int kNumRepetitions = 32;

  if (communicator_->size() < 2 || torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks.";
  }

  const DeviceIdxType my_rank = communicator_->deviceId();
  const DeviceIdxType size = communicator_->size();
  const DeviceIdxType send_peer = (my_rank + 1) % size;
  const DeviceIdxType recv_peer = (size + my_rank - 1) % size;

  auto container = std::make_unique<hir::HostIrContainer>();
  FusionGuard fg(container.get());

  auto* send_peer_val = IrBuilder::create<Val>(send_peer, DataType::Int);
  auto* recv_peer_val = IrBuilder::create<Val>(recv_peer, DataType::Int);

  auto* send_tv = makeContigTensor(1);
  auto* recv_tv = makeContigTensor(1);
  container->addInput(send_tv);
  container->addInput(recv_tv);

  auto send = IrBuilder::create<P2PCommunication>(
      P2PCommunicationType::SEND,
      send_tv,
      send_peer_val,
      CommunicatorBackend::kCuda);
  auto recv = IrBuilder::create<P2PCommunication>(
      P2PCommunicationType::RECV,
      recv_tv,
      recv_peer_val,
      CommunicatorBackend::kCuda);
  std::vector<P2PCommunication*> grouped_communications = {send, recv};
  auto share_mem_handles = IrBuilder::create<hir::ShareMemHandles>(
      std::move(grouped_communications));
  auto wait_send = IrBuilder::create<hir::Wait>(send);
  auto wait_recv = IrBuilder::create<hir::Wait>(recv);

  container->pushBackTopLevelExprs(share_mem_handles);
  container->pushBackTopLevelExprs(send);
  container->pushBackTopLevelExprs(recv);
  container->pushBackTopLevelExprs(wait_send);
  container->pushBackTopLevelExprs(wait_recv);

  hir::HostIrEvaluator executor(std::move(container), communicator_);

  at::Tensor send_tensor = at::empty({kTensorSize}, tensor_options);
  at::Tensor recv_tensor = at::empty({kTensorSize}, tensor_options);

  std::unordered_map<Val*, PolymorphicValue> inputs = {
      {send_tv, send_tensor}, {recv_tv, recv_tensor}};

  for (auto repetition : c10::irange(kNumRepetitions)) {
    send_tensor.copy_(
        at::arange(kTensorSize, tensor_options) + repetition * 10 +
        100 * my_rank);

    executor.runWithInput(inputs);

    auto ref = at::arange(kTensorSize, tensor_options) + repetition * 10 +
        100 * recv_peer;
    EXPECT_TRUE(torch::allclose(recv_tensor, ref))
        << "Rank " << my_rank << " failed at repetition " << repetition
        << " with recv tensor " << recv_tensor << " and ref " << ref;
  }
}

} // namespace nvfuser
