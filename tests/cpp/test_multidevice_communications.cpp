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
    // testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
    //
    // UCC triggered the following segfault in CI:
    //
    // clang-format off
    // 00:00:18 [1,0]<stdout>:[ RUN      ] CommunicationTest.Gather/UCC
    // 00:00:18 [1,1]<stderr>:[7859fcd3f8f9:338  :0:338] Caught signal 11 (Segmentation fault: address not mapped to object at address 0x55b7566aaf90)
    // 00:00:18 [1,1]<stderr>:==== backtrace (tid:    338) ====
    // 00:00:18 [1,1]<stderr>: 0  /opt/hpcx/ucx/lib/libucs.so.0(ucs_handle_error+0x2e4) [0x7f558fbc8654]
    // 00:00:18 [1,1]<stderr>: 1  /opt/hpcx/ucx/lib/libucs.so.0(+0x3684c) [0x7f558fbc884c]
    // 00:00:18 [1,1]<stderr>: 2  /opt/hpcx/ucx/lib/libucs.so.0(+0x36a88) [0x7f558fbc8a88]
    // 00:00:18 [1,1]<stderr>: 3  /usr/lib/x86_64-linux-gnu/libc.so.6(+0x45330) [0x7f558fc47330]
    // 00:00:18 [1,1]<stderr>: 4  /opt/hpcx/ucc/lib/ucc/libucc_tl_ucp.so(+0x2ece8) [0x7f5443fa5ce8]
    // 00:00:18 [1,1]<stderr>: 5  /opt/hpcx/ucc/lib/ucc/libucc_tl_ucp.so(ucc_tl_ucp_allreduce_knomial_progress+0x573) [0x7f5443fa7123]
    // 00:00:18 [1,1]<stderr>: 6  /opt/hpcx/ucc/lib/ucc/libucc_tl_ucp.so(ucc_tl_ucp_allreduce_knomial_start+0x1bd) [0x7f5443fa57fd]
    // 00:00:18 [1,1]<stderr>: 7  /opt/hpcx/ucc/lib/ucc/libucc_tl_ucp.so(ucc_tl_ucp_service_allreduce+0x267) [0x7f5443f88357]
    // 00:00:18 [1,1]<stderr>: 8  /opt/hpcx/ucc/lib/libucc.so.1(ucc_service_allreduce+0x107) [0x7f558fb69ee7]
    // 00:00:18 [1,1]<stderr>: 9  /opt/hpcx/ucc/lib/libucc.so.1(ucc_team_create_test_single+0x8f5) [0x7f558fb66455]
    // 00:00:18 [1,1]<stderr>:10  /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so(+0xd9a498) [0x7f5591a05498]
    // 00:00:18 [1,1]<stderr>:11  /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so(_ZN4c10d15ProcessGroupUCC8initCommEN3c106DeviceE+0x388) [0x7f5591a0c248]
    // 00:00:18 [1,1]<stderr>:12  /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so(_ZN4c10d15ProcessGroupUCC6gatherERSt6vectorIS1_IN2at6TensorESaIS3_EESaIS5_EERS5_RKNS_13GatherOptionsE+0x73) [0x7f5591a15463]
    // 00:00:18 [1,1]<stderr>:13  bin/test_multidevice(+0x6b81e5) [0x55b4f38871e5]
    // 00:00:18 [1,1]<stderr>:14  bin/test_multidevice(+0x6b9ba6) [0x55b4f3888ba6]
    // 00:00:18 [1,1]<stderr>:15  bin/test_multidevice(+0xa84aec) [0x55b4f3c53aec]
    // 00:00:18 [1,1]<stderr>:16  bin/test_multidevice(+0xb90d51) [0x55b4f3d5fd51]
    // 00:00:18 [1,1]<stderr>:17  bin/test_multidevice(+0xb77f5a) [0x55b4f3d46f5a]
    // 00:00:18 [1,1]<stderr>:18  bin/test_multidevice(+0xb78512) [0x55b4f3d47512]
    // 00:00:18 [1,1]<stderr>:19  bin/test_multidevice(+0xb78b51) [0x55b4f3d47b51]
    // 00:00:18 [1,1]<stderr>:20  bin/test_multidevice(+0xb866fa) [0x55b4f3d556fa]
    // 00:00:18 [1,1]<stderr>:21  bin/test_multidevice(+0xb78d4a) [0x55b4f3d47d4a]
    // 00:00:18 [1,1]<stderr>:22  bin/test_multidevice(+0x192edb) [0x55b4f3361edb]
    // 00:00:18 [1,1]<stderr>:23  /usr/lib/x86_64-linux-gnu/libc.so.6(+0x2a1ca) [0x7f558fc2c1ca]
    // 00:00:18 [1,1]<stderr>:24  /usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x8b) [0x7f558fc2c28b]
    // 00:00:18 [1,1]<stderr>:25  bin/test_multidevice(+0x19cae5) [0x55b4f336bae5]
    // 00:00:18 [1,1]<stderr>:=================================
    // clang-format on
    testing::Values(CommunicatorBackend::kNccl),
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
