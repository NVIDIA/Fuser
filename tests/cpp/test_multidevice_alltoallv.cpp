// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>

#include "multidevice/cuda_p2p.h"
#include "multidevice/symmetric_tensor.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {
namespace hir {

class AlltoallvCudaTest : public MultiDeviceTest {};

TEST_F(AlltoallvCudaTest, AlltoallvAsymmetric) {
  if (!communicator_->is_available() || communicator_->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks.";
  }

  const int64_t world_size = communicator_->size();
  const int64_t my_rank = communicator_->deviceId();

  auto int_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kLong);

  auto count_for = [](int64_t sender, int64_t dest) {
    return (sender + dest) % 3 + 1;
  };
  auto send_counts = at::empty({world_size}, int_options);
  for (int64_t dest = 0; dest < world_size; ++dest) {
    send_counts.index_put_({dest}, count_for(my_rank, dest));
  }

  auto metadata = prepareAlltoallvMetadata(send_counts, "test_alltoallv_counts");
  const int64_t max_recv = metadata.max_recv;
  const int64_t total_send = send_counts.sum().item<int64_t>();
  auto send_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, at::kLong, communicator_->device());
  send_sym.narrow(0, 0, total_send)
      .copy_(at::arange(total_send, int_options) + my_rank * 1000);

  auto recv_sym = SymmetricTensor::allocate(
      {max_recv}, at::kLong, communicator_->device());
  SymmetricTensor recv_handle(recv_sym);
  recv_handle.setupRemoteHandles("test_alltoallv_recv");

  std::vector<void*> recv_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    recv_ptrs[rank] = recv_handle.remoteTensor(rank).data_ptr();
  }

  auto stream = at::cuda::getDefaultCUDAStream().stream();
  alltoallvWithCudaBackend(send_sym, recv_sym, metadata, recv_ptrs, stream);
  alltoallvBarrier("test_alltoallv_counts");

  auto recv_view = recv_sym.narrow(0, 0, metadata.total_recv);
  std::vector<int64_t> expected_vec;
  expected_vec.reserve(static_cast<size_t>(metadata.total_recv));
  for (int64_t sender = 0; sender < world_size; ++sender) {
    int64_t offset = 0;
    for (int64_t dest = 0; dest < my_rank; ++dest) {
      offset += count_for(sender, dest);
    }
    const int64_t count = count_for(sender, my_rank);
    for (int64_t i = 0; i < count; ++i) {
      expected_vec.push_back(offset + i + sender * 1000);
    }
  }
  auto expected = at::tensor(expected_vec, int_options);
  EXPECT_TRUE(at::equal(recv_view, expected))
      << "Alltoallv mismatch on rank " << my_rank;
}

} // namespace hir
} // namespace nvfuser
