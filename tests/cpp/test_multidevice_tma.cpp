// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
//
// Unit tests for Hopper TMA (Tensor Memory Accelerator) 1D bulk copy
// (cp.async.bulk) across different memory sources:
//   1. Local device memory (cudaMalloc)
//   2. VMM-mapped peer device memory (inter-device P2P)
//   3. NVLS multicast unicast pointers
//
// The kernel source lives in csrc/multidevice/tma_copy.cu and is
// stringified at build time. It is compiled at runtime via NVRTC,
// same pattern as csrc/multidevice/cuda_p2p.cpp.

#include <cuda.h>
#include <nvrtc.h>

#include <string>
#include <vector>

#include "cuda_utils.h"
#include "driver_api.h"
#include "exceptions.h"
#include "multidevice/symmetric_tensor.h"
#include "multidevice/utils.h"
#include "nvfuser_resources/tma_copy.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {

// ============================================================================
// NVRTC helper: compile kernel source at runtime, cache the result.
// ============================================================================

namespace {

CUfunction compileAndGetKernel(
    CUmodule& module,
    CUfunction& function,
    const char* source,
    const char* source_name,
    const char* kernel_name) {
  if (function != nullptr) {
    return function;
  }

  nvrtcProgram prog;
  NVFUSER_NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog, source, source_name, 0, nullptr, nullptr));

  int device = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device));
  cudaDeviceProp prop;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, device));

  std::string arch_arg = "--gpu-architecture=compute_" +
      std::to_string(prop.major) + std::to_string(prop.minor);
  std::vector<const char*> opts = {arch_arg.c_str(), "--std=c++17"};

  nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
  if (res != NVRTC_SUCCESS) {
    size_t logSize;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    std::vector<char> log(logSize);
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
    NVF_ERROR(
        false,
        "NVRTC compilation of '",
        source_name,
        "' failed:\n",
        log.data());
  }

  size_t ptxSize;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
  NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  NVFUSER_CUDA_SAFE_CALL(cuModuleLoadData(&module, ptx.data()));
  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, kernel_name));

  return function;
}

//! Return the NVRTC-compiled tma_copy_1d CUfunction (cached after
//! first call). The kernel uses cp.async.bulk to perform
//!   GMEM(src) -> SMEM -> GMEM(dst)
//! and requires dynamic shared memory of num_bytes + 8 (mbarrier).
CUfunction getTmaCopy1dKernel() {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;
  return compileAndGetKernel(
      module,
      kernel,
      nvfuser_resources::tma_copy_cu,
      "tma_copy.cu",
      "tma_copy_1d");
}

//! Launch the TMA 1D bulk copy kernel: GMEM(src) -> SMEM -> GMEM(dst).
//! num_bytes must be > 0 and a multiple of 16.
void launchTmaCopy1D(
    void* dst,
    const void* src,
    int num_bytes,
    CUstream stream = nullptr) {
  NVF_CHECK(num_bytes > 0 && num_bytes % 16 == 0);
  CUfunction tma_kernel = getTmaCopy1dKernel();
  int smem_size = num_bytes + static_cast<int>(sizeof(uint64_t));
  void* args[] = {&dst, &src, &num_bytes};
  NVFUSER_CUDA_SAFE_CALL(cuLaunchKernel(
      tma_kernel, 1, 1, 1, 32, 1, 1, smem_size, stream, args, nullptr));
}

} // anonymous namespace

// ============================================================================
// Tests
// ============================================================================

using TmaTest = MultiDeviceTest;

// Verify TMA 1D bulk copy on local device memory.
// The kernel uses cp.async.bulk (GMEM->SMEM) + cp.async.bulk (SMEM->GMEM)
// with mbarrier synchronization between the two phases.
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

  launchTmaCopy1D(dst.data_ptr(), src.data_ptr(), kSizeBytes);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());

  EXPECT_TRUE(dst.equal(src));
}

// Verify TMA 1D bulk copy reading from a VMM-mapped peer device
// buffer. SymmetricTensor handles the VMM allocation and IPC handle
// exchange; the test focuses on the TMA transfer itself.
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

  launchTmaCopy1D(output.data_ptr(), peer.data_ptr(), kSizeBytes);
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

// Verify TMA 1D bulk copy writing TO an NVLS multicast pointer.
// Root uses TMA to write data to the MC pointer, which broadcasts
// via NVLS hardware. All ranks then verify the data arrived by
// reading from their local UC view with a normal copy.
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

  constexpr int64_t kNumElems = 524288; // 2 MB / sizeof(int32_t)
  constexpr int64_t root = 0;

  // cp.async.bulk transfer size is limited by shared memory,
  // so we broadcast a 4 KB slice via TMA.
  constexpr int kTmaBytes = 4096;
  static_assert(kTmaBytes % 16 == 0);
  constexpr int kTmaElems = kTmaBytes / sizeof(int32_t);

  at::Tensor local =
      SymmetricTensor::allocate({kNumElems}, at::kInt, communicator_->device());
  local.zero_();
  SymmetricTensor sym(local);
  sym.setupMulticast(root, "tma_mcast");

  auto opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, local_rank);

  // Root: TMA-write source data to MC pointer (NVLS broadcasts it)
  if (rank == root) {
    at::Tensor src = at::arange(kTmaElems, opts);
    launchTmaCopy1D(sym.multicastPtr(), src.data_ptr(), kTmaBytes);
    NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
  }

  communicator_->barrier();

  // All ranks: verify data arrived via normal read of local UC tensor
  at::Tensor readback = sym.localTensor().slice(0, 0, kTmaElems).clone();
  at::Tensor expected = at::arange(kTmaElems, opts);
  EXPECT_TRUE(readback.equal(expected))
      << "Rank " << rank << " did not receive multicast data written by TMA";
}

#endif // CUDA_VERSION >= 13000

} // namespace nvfuser
