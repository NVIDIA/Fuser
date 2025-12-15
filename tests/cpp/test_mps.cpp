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
#include <options.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <cstdlib>
#include <iostream>

namespace nvfuser {

namespace {

// Common test fusion: (t0 + t1) * (t0 + t1)
void runPointwiseFusion() {
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
// To test different SM counts, run separate processes:
//   NVFUSER_ENABLE='mps_sm_affinity(8)' ./test_mps --gtest_filter=*Pointwise
//   NVFUSER_ENABLE='mps_sm_affinity(16)' ./test_mps --gtest_filter=*Pointwise
//   etc.
class MPSSmLimitTest : public ::testing::Test {};

// Test: SM limiting with MPS
// GB200 benchmark results (16384x16384 pointwise):
// SM 8:   3.733 ms
// SM 16:  1.933 ms
// SM 32:  1.040 ms
// SM 64:  0.610 ms
// SM 76:  0.567 ms
// SM 128: 0.462 ms
// SM 152: 0.451 ms (full GPU, 90% SOL)
TEST_F(MPSSmLimitTest, Pointwise) {
  // Skip if MPS SM affinity not enabled via environment
  if (!isOptionEnabled(EnableOption::MpsSmAffinity)) {
    GTEST_SKIP() << "Set SM count via: NVFUSER_ENABLE=mps_sm_affinity(N)";
  }

  // Check if MPS is running
  int ret = system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1");
  if (ret != 0) {
    GTEST_SKIP() << "MPS not running. Start with: tools/start_mps.sh";
  }

  ASSERT_NO_THROW(executor_utils::initializeCudaContext());

  // Query and print actual SM count
  CUexecAffinityParam affinity = {};
  CUresult res =
      nvfuser::cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
  ASSERT_EQ(res, CUDA_SUCCESS) << "Failed to query execution affinity";
  std::cout << "MPS SM limiting: " << affinity.param.smCount.val << " SMs"
            << std::endl;

  runPointwiseFusion();
}

// Reference test: Same fusion WITHOUT MPS SM limiting
// GB200, 90% SOL, 0.453 ms
TEST_F(NVFuserTest, PointwiseReference) {
  runPointwiseFusion();
}

} // namespace nvfuser
