// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/symmetric_tensor.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

using SymmetricTensorTest = MultiDeviceTest;

TEST_F(SymmetricTensorTest, BasicAllocation) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t world_size = communicator_->size();

  // Create a symmetric tensor
  at::Tensor local_tensor = SymmetricTensor::allocate(
      {256, 512}, at::ScalarType::Float, communicator_->device());
  SymmetricTensor sym_tensor(local_tensor);

  // Validate local tensor
  EXPECT_TRUE(local_tensor.is_cuda());
  EXPECT_EQ(local_tensor.scalar_type(), at::ScalarType::Float);
  EXPECT_EQ(local_tensor.numel(), 256 * 512);
  EXPECT_EQ(local_tensor.sizes()[0], 256);
  EXPECT_EQ(local_tensor.sizes()[1], 512);

  // Write unique value to local tensor
  float local_value = static_cast<float>(rank + 100);
  local_tensor.fill_(local_value);

  sym_tensor.setupRemoteHandles();

  // Read from all remote tensors
  for (int64_t peer_rank = 0; peer_rank < world_size; ++peer_rank) {
    void* peer_ptr = sym_tensor.remoteTensor(peer_rank).data_ptr();
    EXPECT_NE(peer_ptr, nullptr);

    // Copy first element from peer
    float peer_value;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        &peer_value, peer_ptr, sizeof(float), cudaMemcpyDeviceToHost));

    float expected_value = static_cast<float>(peer_rank + 100);
    EXPECT_FLOAT_EQ(peer_value, expected_value)
        << "Rank " << rank << " reading from rank " << peer_rank;
  }
}

TEST_F(SymmetricTensorTest, PreallocatedTensor) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t world_size = communicator_->size();

  // Allocate tensor with symmetric memory
  at::Tensor local_tensor = SymmetricTensor::allocate(
      /*sizes=*/at::IntArrayRef({128, 256}),
      /*dtype=*/at::ScalarType::Double,
      /*device=*/c10::Device(c10::DeviceType::CUDA, rank));

  // Create SymmetricTensor from pre-allocated tensor
  SymmetricTensor sym_tensor(local_tensor);

  // Validate
  EXPECT_EQ(sym_tensor.localTensor().numel(), 128 * 256);

  // Write unique pattern to local tensor
  double local_value = static_cast<double>(rank * 1000 + 42);
  local_tensor.fill_(local_value);

  sym_tensor.setupRemoteHandles();

  // Verify remote access
  for (int64_t peer_rank = 0; peer_rank < world_size; ++peer_rank) {
    if (peer_rank == rank) {
      continue;
    }

    void* peer_ptr = sym_tensor.remoteTensor(peer_rank).data_ptr();
    double peer_value;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        &peer_value, peer_ptr, sizeof(double), cudaMemcpyDeviceToHost));

    double expected = static_cast<double>(peer_rank * 1000 + 42);
    EXPECT_DOUBLE_EQ(peer_value, expected);
  }
}

TEST_F(SymmetricTensorTest, Multicast) {
#if (CUDA_VERSION < 13000)
  GTEST_SKIP() << "Multicast requires CUDA 13.0+";
#else
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t root = 0;

  // Check multicast support
  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, rank));
  if (!is_multicast_supported) {
    GTEST_SKIP() << "Device does not support multicast";
  }

  // Create symmetric tensor (2MB to meet granularity requirements)
  constexpr int64_t kNumElems = 524288; // 2MB / 4 bytes
  at::Tensor local_tensor = SymmetricTensor::allocate(
      /*sizes=*/at::IntArrayRef({kNumElems}),
      /*dtype=*/at::ScalarType::Int,
      /*device=*/c10::Device(c10::DeviceType::CUDA, rank));
  SymmetricTensor sym_tensor(local_tensor);

  // Setup multicast
  sym_tensor.setupMulticast(root, "test_multicast");

  // Root writes data to multicast buffer
  std::vector<int> host_data(kNumElems);
  if (rank == root) {
    void* mc_ptr = sym_tensor.multicastPtr();
    EXPECT_NE(mc_ptr, nullptr);

    // Prepare pattern data
    for (int64_t i = 0; i < kNumElems; ++i) {
      host_data[i] = static_cast<int>(i * 7 + 13);
    }

    // Write to multicast buffer
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        mc_ptr,
        host_data.data(),
        kNumElems * sizeof(int),
        cudaMemcpyHostToDevice));
  }

  communicator_->barrier();

  // All ranks read from local tensor and validate
  const at::Tensor& local = sym_tensor.localTensor();
  std::vector<int> readback(kNumElems);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      readback.data(),
      local.data_ptr(),
      kNumElems * sizeof(int),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < kNumElems; ++i) {
    int expected = static_cast<int>(i * 7 + 13);
    EXPECT_EQ(readback[i], expected)
        << "Rank " << rank << " failed to read multicast data at index " << i;
  }
#endif
}

TEST_F(SymmetricTensorTest, ContiguousView) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t world_size = communicator_->size();

  // Create symmetric tensor
  at::Tensor local_tensor = SymmetricTensor::allocate(
      /*sizes=*/at::IntArrayRef({2, 262144}),
      /*dtype=*/at::ScalarType::Float,
      /*device=*/c10::Device(c10::DeviceType::CUDA, rank));
  SymmetricTensor sym_tensor(local_tensor);

  // Write rank-specific pattern to local tensor
  local_tensor.fill_(static_cast<float>(rank + 100));

  // Validate that localTensor has the correct values for this rank
  std::vector<float> local_data(local_tensor.numel());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      local_data.data(),
      local_tensor.data_ptr(),
      local_tensor.numel() * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (int64_t i = 0; i < local_tensor.numel(); ++i) {
    ASSERT_EQ(local_data[i], static_cast<float>(rank + 100))
        << "localTensor value mismatch at index " << i << " for rank " << rank;
  }

  communicator_->barrier();

  // Setup and get contiguous view of all ranks
  sym_tensor.setupContiguousView("test_contiguous");
  at::Tensor contiguous_view = sym_tensor.getContiguousView();

  // Validate shape: [world_size, 2, 262144]
  EXPECT_EQ(contiguous_view.dim(), 3);
  EXPECT_EQ(contiguous_view.size(0), world_size);
  EXPECT_EQ(contiguous_view.size(1), 2);
  EXPECT_EQ(contiguous_view.size(2), 262144);

  // Validation: copy and check each per-rank slice from host buffer
  const int64_t slice_elems = contiguous_view.size(1) * contiguous_view.size(2);
  const int64_t total_elems = world_size * slice_elems;
  std::vector<float> all_data(total_elems);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      all_data.data(),
      contiguous_view.data_ptr(),
      total_elems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (int64_t r = 0; r < world_size; ++r) {
    for (int64_t i = 0; i < slice_elems; ++i) {
      float expected = static_cast<float>(r + 100);
      size_t idx = r * slice_elems + i;
      ASSERT_EQ(all_data[idx], expected)
          << "Rank " << rank << " view checking slice for rank " << r
          << " at offset " << i << " did not match expected value";
    }
  }
}

} // namespace nvfuser
