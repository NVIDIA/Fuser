// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
//
// Unit tests for TMA (Tensor Memory Accelerator) 1D bulk copy
// (cp.async.bulk) across different memory sources:
//   1. Local device memory
//   2. VMM-mapped peer device memory (inter-device P2P)
//   3. NVLS multicast pointers
//
// Uses the production launchTmaCopy() from cuda_p2p.cpp, which
// compiles csrc/multidevice/tma_copy.cu at runtime via NVRTC.

#include <cuda.h>

#include "cuda_utils.h"
#include "driver_api.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/symmetric_tensor.h"
#include "multidevice/utils.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {

using TmaTest = MultiDeviceTest;

TEST_F(TmaTest, TmaLocalCopy) {
  const int64_t local_rank = communicator_->local_rank();

  int major;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, local_rank));
  if (major < 9) {
    GTEST_SKIP() << "Requires Hopper (SM90+)";
  }

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(local_rank));

  constexpr int kNumElems = 256;
  constexpr int kSizeBytes = kNumElems * sizeof(uint32_t);
  static_assert(kSizeBytes % 16 == 0);

  auto options =
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, local_rank);
  at::Tensor src = at::arange(kNumElems, options);
  at::Tensor dst = at::zeros({kNumElems}, options);

  launchTmaCopy(dst.data_ptr(), src.data_ptr(), kSizeBytes, nullptr);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());

  EXPECT_TRUE(dst.equal(src));
}

TEST_F(TmaTest, TmaInterDeviceCopy) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t local_rank = communicator_->local_rank();
  const int64_t world_size = communicator_->size();

  int major;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, local_rank));
  if (major < 9) {
    GTEST_SKIP() << "Requires Hopper (SM90+)";
  }

  constexpr int kNumElems = 256;
  constexpr int kSizeBytes = kNumElems * sizeof(int32_t);
  static_assert(kSizeBytes % 16 == 0);

  at::Tensor local =
      SymmetricTensor::allocate({kNumElems}, at::kInt, communicator_->device());
  local.fill_(static_cast<int>(rank * 10000));
  SymmetricTensor sym(local);
  sym.setupRemoteHandles("tma_p2p");

  const int64_t peer_rank = (rank + 1) % world_size;
  at::Tensor peer = sym.remoteTensor(peer_rank);

  at::Tensor output = at::zeros(
      {kNumElems},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, local_rank));

  launchTmaCopy(output.data_ptr(), peer.data_ptr(), kSizeBytes, nullptr);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());

  at::Tensor expected = at::full(
      {kNumElems},
      static_cast<int>(peer_rank * 10000),
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, local_rank));
  EXPECT_TRUE(output.equal(expected))
      << "Rank " << rank << " TMA read from peer " << peer_rank
      << " returned wrong data";
}

#if (CUDA_VERSION >= 13000)

TEST_F(TmaTest, TmaMulticastWrite) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t local_rank = communicator_->local_rank();

  int major;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, local_rank));
  if (major < 9) {
    GTEST_SKIP() << "Requires Hopper (SM90+)";
  }

  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  if (is_multicast_supported == 0) {
    GTEST_SKIP() << "Device does not support Multicast Objects; skipping.";
  }

  constexpr int64_t kNumElems = 524288;
  constexpr int64_t root = 0;
  constexpr int kTmaBytes = 4096;
  static_assert(kTmaBytes % 16 == 0);
  constexpr int kTmaElems = kTmaBytes / sizeof(int32_t);

  at::Tensor local =
      SymmetricTensor::allocate({kNumElems}, at::kInt, communicator_->device());
  local.zero_();
  SymmetricTensor sym(local);
  sym.setupMulticast(root, "tma_mcast");

  auto opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, local_rank);

  if (rank == root) {
    at::Tensor src = at::arange(kTmaElems, opts);
    launchTmaCopy(sym.multicastPtr(), src.data_ptr(), kTmaBytes, nullptr);
    NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
  }

  communicator_->barrier();

  at::Tensor readback = sym.localTensor().slice(0, 0, kTmaElems).clone();
  at::Tensor expected = at::arange(kTmaElems, opts);
  EXPECT_TRUE(readback.equal(expected))
      << "Rank " << rank << " did not receive multicast data written by TMA";
}

#endif // CUDA_VERSION >= 13000

} // namespace nvfuser
