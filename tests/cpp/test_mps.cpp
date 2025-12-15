// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
// MPS SM Affinity Utilities and Tests
// =====================================
//
// This file provides utilities for experimenting with MPS (Multi-Process
// Service) SM (Streaming Multiprocessor) affinity - limiting kernels to use
// only a subset of available SMs on a GPU.
//
// Design Philosophy:
// ------------------
// Rather than integrating SM affinity deeply into nvFuser core, we provide it
// as a testing/benchmarking utility. This keeps the core compiler focused on
// fusion and scheduling logic, while still enabling important experiments:
//
//   - Understanding performance scaling with SM count
//   - Simulating gpus with higher bandwidth per SM
//
// Setup: Starting MPS, see tools/start_mps.sh
// -------------------
// MPS must be running with per-context SM partitioning enabled:
//
//   export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
//   export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
//   export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
//   mkdir -p /tmp/nvidia-mps
//   nvidia-cuda-mps-control -d
//
// To stop MPS, see tools/start_mps.sh
//   ./tools/start_mps.sh stop
//

#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_utils.h>
#include <driver_api.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <options.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {

// Utility class for managing MPS SM affinity in tests and benchmarks
// This provides a clean interface for experimenting with SM limiting
// without integrating it deeply into nvFuser core.
class MPSContextManager {
 public:
  // Default constructor
  MPSContextManager() = default;

  // Initialize MPS context with specific SM count
  // Returns true if successful, false if MPS not available or setup failed
  bool initialize(int num_sms) {
    if (num_sms <= 0) {
      std::cerr << "Invalid SM count: " << num_sms << std::endl;
      return false;
    }

    // Check if MPS is running
    int ret = system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1");
    if (ret != 0) {
      std::cerr
          << "MPS not running. See tools/start_mps.sh for setup instructions."
          << std::endl;
      return false;
    }

    // Initialize CUDA driver API
    NVFUSER_CUDA_SAFE_CALL(cuInit(0));

    // Use device 0 in Driver API enumeration
    // Note: cuDeviceGet uses Driver API device ordinals, which respect
    // CUDA_VISIBLE_DEVICES. So device 0 here is the first visible device.
    int device_id = 0;

    CUdevice dev;
    NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&dev, device_id));

    // Query total SM count
    int total_sms = 0;
    NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &total_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    if (num_sms > total_sms) {
      std::cerr << "Requested " << num_sms << " SMs but GPU only has "
                << total_sms << " SMs" << std::endl;
      return false;
    }

    // Check if context already exists
    CUcontext existing_ctx = nullptr;
    NVFUSER_CUDA_SAFE_CALL(cuCtxGetCurrent(&existing_ctx));
    if (existing_ctx != nullptr) {
      std::cerr << "CUDA context already exists. MPSContextManager must be "
                << "initialized before any CUDA operations." << std::endl;
      return false;
    }

    // Create execution affinity parameter for SM count limiting
    CUexecAffinityParam affinity = {};
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = (unsigned int)num_sms;

    // Wrap affinity params in CUctxCreateParams
    CUctxCreateParams ctx_params = {};
    ctx_params.execAffinityParams = &affinity;
    ctx_params.numExecAffinityParams = 1;

    // Create context with SM affinity
    NVFUSER_CUDA_SAFE_CALL(cuCtxCreate(&sm_limited_ctx_, &ctx_params, 0, dev));

    sm_count_ = num_sms;
    total_sms_ = total_sms;
    initialized_ = true;

    // Set our SM-limited context as current
    // This ensures PyTorch operations (tensor allocation, etc.) use our context
    NVFUSER_CUDA_SAFE_CALL(cuCtxSetCurrent(sm_limited_ctx_));

    std::cout << "MPSContextManager: Created and activated context with "
              << num_sms << " SMs (out of " << total_sms << " total)"
              << std::endl;

    return true;
  }

  // Query actual SM count assigned to current context
  int getActualSmCount() const {
    if (!initialized_) {
      return -1;
    }

    CUexecAffinityParam affinity = {};
    NVFUSER_CUDA_SAFE_CALL(
        cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT));

    return affinity.param.smCount.val;
  }

  // Get requested SM count
  int getRequestedSmCount() const {
    return sm_count_;
  }

  // Get total SMs on GPU
  int getTotalSmCount() const {
    return total_sms_;
  }

  bool isInitialized() const {
    return initialized_;
  }

  // Verify our context is still current
  // Returns true if our SM-limited context is active, false otherwise
  bool isContextCurrent() const {
    if (!initialized_ || sm_limited_ctx_ == nullptr) {
      return false;
    }

    CUcontext current_ctx = nullptr;
    NVFUSER_CUDA_SAFE_CALL(cuCtxGetCurrent(&current_ctx));

    return current_ctx == sm_limited_ctx_;
  }

  // Make our context current again (in case PyTorch switched contexts)
  bool makeContextCurrent() {
    if (!initialized_ || sm_limited_ctx_ == nullptr) {
      return false;
    }

    NVFUSER_CUDA_SAFE_CALL(cuCtxSetCurrent(sm_limited_ctx_));
    return true;
  }

  // Destructor - synchronize before cleanup
  ~MPSContextManager() {
    if (initialized_ && sm_limited_ctx_ != nullptr) {
      // Make sure our context is current for cleanup
      cuCtxSetCurrent(sm_limited_ctx_);

      // Synchronize to ensure all CUDA operations complete
      cudaDeviceSynchronize();

      // Note: We don't call cuCtxDestroy() here because:
      // 1. CUDA profiler and other resources may still need the context
      // 2. The context will be cleaned up automatically at process exit
      // 3. Explicit destruction can cause "context destroyed" errors
      //
      // In a long-running process where you need to clean up, call
      // cudaDeviceReset() after you're done with all CUDA operations.
    }
  }

  // Delete copy/move to avoid double-free
  MPSContextManager(const MPSContextManager&) = delete;
  MPSContextManager& operator=(const MPSContextManager&) = delete;
  MPSContextManager(MPSContextManager&&) = delete;
  MPSContextManager& operator=(MPSContextManager&&) = delete;

 private:
  CUcontext sm_limited_ctx_ = nullptr;
  int sm_count_ = 0;
  int total_sms_ = 0;
  bool initialized_ = false;
};

// Common test fusion: (t0 + t1) * (t0 + t1)
void runPointwiseFusion(MPSContextManager* mps_ctx = nullptr) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Float);
  auto tv1 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);
  auto tv3 = mul(tv2, tv2);
  fusion->addOutput(tv3);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16384, 16384}, options);
  auto t1 = at::randn({16384, 16384}, options);

  // If using MPS SM affinity, re-activate our context after PyTorch tensor ops
  if (mps_ctx != nullptr) {
    if (!mps_ctx->isContextCurrent()) {
      mps_ctx->makeContextCurrent();
    }
  }

  auto outputs = fec.runFusionWithInputs({t0, t1});

  auto expected = (t0 + t1) * (t0 + t1);
  testValidate(fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
}

} // namespace

// Test fixture for MPS SM limiting
// NOTE: Does NOT inherit from NVFuserTest to avoid early CUDA init.
// MPS SM-limited context can only be created ONCE per process, so
// parameterized testing (TEST_P) doesn't work - each SM count requires
// a separate process.
//
// To test different SM counts with MPSContextManager, create a benchmark
// that runs each SM count in a separate process.
class MPSSmLimitTest : public ::testing::Test {};

// Test: SM limiting with MPS using MPSContextManager
// This demonstrates the utility class usage for benchmarking.
//
// Defaults to 8 SMs if MPS_SM_COUNT is not set.
// To run with specific SM count:
//   MPS_SM_COUNT=16 ./test_mps --gtest_filter=*PointwiseWithUtility
//   MPS_SM_COUNT=32 ./test_mps --gtest_filter=*PointwiseWithUtility
// To check performance, further add:
// NVFUSER_PROF=print MPS_SM_COUNT=32 ./test_mps
// --gtest_filter=*PointwiseWithUtility
TEST_F(MPSSmLimitTest, PointwiseWithUtility) {
  // Check for MPS_SM_COUNT environment variable, default to 8
  const char* sm_count_env = std::getenv("MPS_SM_COUNT");
  int requested_sms = 8; // Default value

  if (sm_count_env != nullptr) {
    requested_sms = std::atoi(sm_count_env);
    if (requested_sms <= 0) {
      GTEST_SKIP() << "Invalid MPS_SM_COUNT: " << sm_count_env;
    }
  }

  std::cout << "Testing with " << requested_sms
            << " SMs (set MPS_SM_COUNT to override)" << std::endl;

  // Initialize MPS context with requested SM count
  // IMPORTANT: This must be done BEFORE any CUDA/PyTorch operations
  MPSContextManager mps_ctx;
  if (!mps_ctx.initialize(requested_sms)) {
    GTEST_SKIP() << "Failed to initialize MPS context. "
                 << "See file header for MPS setup instructions.";
  }

  std::cout << "Running with " << mps_ctx.getActualSmCount() << " SMs (out of "
            << mps_ctx.getTotalSmCount() << " total)" << std::endl;

  // Verify context is current before PyTorch operations
  ASSERT_TRUE(mps_ctx.isContextCurrent())
      << "SM-limited context is not current after initialization";

  // Run the fusion with limited SMs
  // Pass mps_ctx so it can re-activate context after tensor creation
  runPointwiseFusion(&mps_ctx);

  // Verify context is still current after operations
  ASSERT_TRUE(mps_ctx.isContextCurrent())
      << "SM-limited context should be current after fusion execution";

  std::cout << "Successfully executed with " << mps_ctx.getActualSmCount()
            << " SMs" << std::endl;
}

TEST_F(NVFuserTest, PointwiseReference) {
  runPointwiseFusion();
}

} // namespace nvfuser
