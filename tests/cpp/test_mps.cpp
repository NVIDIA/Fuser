// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <cuda.h>
#include <driver_api.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <cstdlib>

namespace nvfuser {

// Test fixture for MPS SM limiting tests
class MPSTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();

    // Check if we have a Volta+ GPU (required for execution affinity)
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    compute_capability_ = prop.major * 10 + prop.minor;
    total_sms_ = prop.multiProcessorCount;

    // Save original environment variable
    original_sm_count_ = getEnvVar("NVFUSER_SM_COUNT");
  }

  void TearDown() override {
    // Restore original environment variable
    restoreEnvVar("NVFUSER_SM_COUNT", original_sm_count_);

    NVFuserTest::TearDown();
  }

  // Helper to get environment variable (returns empty string if not set)
  std::string getEnvVar(const char* name) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : std::string();
  }

  // Helper to set environment variable
  void setEnvVar(const char* name, const std::string& value) {
    setenv(name, value.c_str(), 1);
  }

  // Helper to unset environment variable
  void unsetEnvVar(const char* name) {
    unsetenv(name);
  }

  // Helper to restore environment variable
  void restoreEnvVar(const char* name, const std::string& original) {
    if (original.empty()) {
      unsetenv(name);
    } else {
      setenv(name, original.c_str(), 1);
    }
  }

  // Check if GPU supports execution affinity (Volta+)
  bool supportsExecutionAffinity() const {
    return compute_capability_ >= 70;
  }

  // Check if MPS is running and per-context SM partitioning is enabled
  bool isMPSConfigured() const {
    // Check if MPS control daemon is running
    int ret = system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1");
    if (ret != 0) {
      return false;
    }

    // Check if per-context SM partitioning is enabled
    const char* mps_partition = std::getenv(
        "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING");
    return mps_partition && std::string(mps_partition) == "1";
  }

  // Create a simple fusion for testing
  std::unique_ptr<Fusion> createSimpleFusion() {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(2, DataType::Float);
    auto tv1 = makeContigTensor(2, DataType::Float);
    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);
    auto tv3 = mul(tv2, tv2);
    fusion->addOutput(tv3);

    return fusion;
  }

  int compute_capability_ = 0;
  int total_sms_ = 0;
  std::string original_sm_count_;
};

// Test: Basic context initialization without SM limiting
TEST_F(MPSTest, ContextInitializationDefault) {
  // Ensure SM limiting is disabled (no NVFUSER_SM_COUNT set)
  unsetEnvVar("NVFUSER_SM_COUNT");

  // This should succeed regardless of MPS configuration
  EXPECT_NO_THROW(executor_utils::initializeCudaContext());
}

// Test: SM limiting with valid configuration (requires MPS)
TEST_F(MPSTest, SMLimitingWithMPS) {
  // Skip if GPU doesn't support execution affinity
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity (requires Volta+)";
  }

  // Skip if MPS is not configured
  if (!isMPSConfigured()) {
    GTEST_SKIP()
        << "MPS is not configured with per-context SM partitioning. "
        << "To enable this test, run:\n"
        << "  export "
           "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1\n"
        << "  nvidia-cuda-mps-control -d";
  }

  // Set SM count to half of available SMs
  int target_sms = (total_sms_ + 1) / 2;
  setEnvVar("NVFUSER_SM_COUNT", std::to_string(target_sms));

  // Initialize context with SM limiting
  EXPECT_NO_THROW(executor_utils::initializeCudaContext());

  // Create and run a simple fusion to verify it works
  auto fusion = createSimpleFusion();
  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({128, 128}, options);
  auto t1 = at::randn({128, 128}, options);

  auto outputs = fec.runFusionWithInputs({t0, t1});

  // Verify output is correct
  auto expected = (t0 + t1) * (t0 + t1);
  testValidate(fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
}

// Test: SM limiting with specific SM count
TEST_F(MPSTest, SMLimitingWithSpecificCount) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity (requires Volta+)";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  // Set SM count to 1/4 of available SMs
  int target_sms = std::max(1, total_sms_ / 4);
  setEnvVar("NVFUSER_SM_COUNT", std::to_string(target_sms));

  EXPECT_NO_THROW(executor_utils::initializeCudaContext());

  // Run a fusion to verify functionality
  auto fusion = createSimpleFusion();
  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({64, 64}, options);
  auto t1 = at::randn({64, 64}, options);

  auto outputs = fec.runFusionWithInputs({t0, t1});
  auto expected = (t0 + t1) * (t0 + t1);
  testValidate(fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
}

// Test: Invalid SM count (too high)
TEST_F(MPSTest, InvalidSMCountTooHigh) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", std::to_string(total_sms_ * 2));

  // Should fail with invalid SM count error
  EXPECT_THROW(executor_utils::initializeCudaContext(), nvfuser::nvfError);
}

// Test: Invalid SM count (zero)
TEST_F(MPSTest, InvalidSMCountZero) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", "0");

  EXPECT_THROW(executor_utils::initializeCudaContext(), nvfuser::nvfError);
}

// Test: Invalid SM count (negative)
TEST_F(MPSTest, InvalidSMCountNegative) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", "-10");

  EXPECT_THROW(executor_utils::initializeCudaContext(), nvfuser::nvfError);
}

// Test: SM limiting with full SM count (should not create affinity context)
TEST_F(MPSTest, SMLimitingWithFullCount) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", std::to_string(total_sms_));

  // Should succeed without creating affinity context
  EXPECT_NO_THROW(executor_utils::initializeCudaContext());

  // Verify fusion still works
  auto fusion = createSimpleFusion();
  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto t1 = at::randn({32, 32}, options);

  auto outputs = fec.runFusionWithInputs({t0, t1});
  auto expected = (t0 + t1) * (t0 + t1);
  testValidate(fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
}

// Test: Complex fusion with SM limiting
TEST_F(MPSTest, ComplexFusionWithSMLimiting) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", std::to_string(total_sms_ / 2));

  executor_utils::initializeCudaContext();

  // Create a more complex fusion
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(3, DataType::Float);
  auto tv1 = makeContigTensor(3, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);
  auto tv3 = mul(tv2, IrBuilder::create<Val>(2.0, DataType::Float));
  auto tv4 = relu(tv3);
  auto tv5 = sum(tv4, {2});
  fusion->addOutput(tv5);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8, 16}, options);
  auto t1 = at::randn({4, 8, 16}, options);

  auto outputs = fec.runFusionWithInputs({t0, t1});

  auto expected = at::sum((at::relu((t0 + t1) * 2.0)), {2});
  testValidate(fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
}

// Test: Multiple fusions with same SM limiting context
TEST_F(MPSTest, MultipleFusionsWithSMLimiting) {
  if (!supportsExecutionAffinity()) {
    GTEST_SKIP() << "GPU does not support execution affinity";
  }

  if (!isMPSConfigured()) {
    GTEST_SKIP() << "MPS not configured";
  }

  setEnvVar("NVFUSER_SM_COUNT", std::to_string(total_sms_ / 2));

  executor_utils::initializeCudaContext();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // Fusion 1: Element-wise operations
  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(2, DataType::Float);
    auto tv1 = makeContigTensor(2, DataType::Float);
    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);
    fusion->addOutput(tv2);

    FusionExecutorCache fec(std::move(fusion));

    auto t0 = at::randn({64, 64}, options);
    auto t1 = at::randn({64, 64}, options);
    auto outputs = fec.runFusionWithInputs({t0, t1});

    testValidate(
        fec.fusion(), outputs, {t0, t1}, {t0 + t1}, __LINE__, __FILE__);
  }

  // Fusion 2: Reduction operations
  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(2, DataType::Float);
    fusion->addInput(tv0);

    auto tv1 = sum(tv0, {1});
    fusion->addOutput(tv1);

    FusionExecutorCache fec(std::move(fusion));

    auto t0 = at::randn({64, 64}, options);
    auto outputs = fec.runFusionWithInputs({t0});

    testValidate(
        fec.fusion(), outputs, {t0}, {at::sum(t0, {1})}, __LINE__, __FILE__);
  }

  // Fusion 3: Broadcasting operations
  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(1, DataType::Float);
    auto tv1 = makeContigTensor(2, DataType::Float);
    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = broadcast(tv0, {true, false});
    auto tv3 = add(tv2, tv1);
    fusion->addOutput(tv3);

    FusionExecutorCache fec(std::move(fusion));

    auto t0 = at::randn({64}, options);
    auto t1 = at::randn({64, 64}, options);
    auto outputs = fec.runFusionWithInputs({t0, t1});

    auto expected = t0.unsqueeze(0) + t1;
    testValidate(
        fec.fusion(), outputs, {t0, t1}, {expected}, __LINE__, __FILE__);
  }
}

// Test: Driver API availability check
TEST_F(MPSTest, DriverAPIAvailability) {
  // Test that the required driver API functions are available
  CUresult status;

  // Test cuInit
  status = nvfuser::cuInit(0);
  EXPECT_EQ(status, CUDA_SUCCESS);

  // Test cuDeviceGet
  CUdevice device;
  status = nvfuser::cuDeviceGet(&device, 0);
  EXPECT_EQ(status, CUDA_SUCCESS);

  // Test cuDeviceGetAttribute
  int sm_count;
  status = nvfuser::cuDeviceGetAttribute(
      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  EXPECT_EQ(status, CUDA_SUCCESS);
  EXPECT_EQ(sm_count, total_sms_);

  // Test cuCtxGetCurrent
  CUcontext ctx;
  status = nvfuser::cuCtxGetCurrent(&ctx);
  EXPECT_EQ(status, CUDA_SUCCESS);
}

// Test: Environment variable parsing
TEST_F(MPSTest, EnvironmentVariableParsing) {
  // Test with NVFUSER_SM_COUNT unset (should behave normally)
  unsetEnvVar("NVFUSER_SM_COUNT");
  EXPECT_NO_THROW(executor_utils::initializeCudaContext());

  // Test with NVFUSER_SM_COUNT set but without MPS
  // (should fail gracefully with error message)
  if (!isMPSConfigured()) {
    setEnvVar("NVFUSER_SM_COUNT", std::to_string(total_sms_ / 2));
    // May throw if MPS is not configured, which is expected
    // We just verify it doesn't crash
    try {
      executor_utils::initializeCudaContext();
    } catch (const nvfuser::nvfError& e) {
      // Expected if MPS is not configured
      EXPECT_TRUE(
          std::string(e.what()).find("MPS") != std::string::npos ||
          std::string(e.what()).find("cuCtxCreate") != std::string::npos);
    }
  }
}

} // namespace nvfuser
