// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <chrono>

#include <gtest/gtest.h>

#include <multidevice/communicator.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

class CommunicatorTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<CommunicatorBackend> {};

// A regression test for #2499.
TEST_P(CommunicatorTest, Barrier) {
  using clock = std::chrono::high_resolution_clock;

  const auto rank = communicator_->deviceId();
  constexpr int kNumIterations = 4;
  constexpr auto kUnitDuration = std::chrono::milliseconds(500);

  std::vector<std::chrono::time_point<clock>> end_times;
  end_times.reserve(kNumIterations);
  for ([[maybe_unused]] auto _ : c10::irange(kNumIterations)) {
    // The last rank enters the barrier the last. Therefore, the duration per
    // iteration is expected to be `kUnitDuration*(num_devices - 1)`.
    std::this_thread::sleep_for(kUnitDuration * rank);
    communicator_->barrier();
    end_times.push_back(clock::now());
  }

  const auto expected_duration = kUnitDuration * (communicator_->size() - 1);
  for (int i = 1; i < kNumIterations; i++) {
    const auto duration = end_times[i] - end_times[i - 1];
    // Expects `duration` to be close enoguh to `expected_duration`.
    EXPECT_LE(duration, expected_duration + kUnitDuration / 2);
    EXPECT_GE(duration, expected_duration - kUnitDuration / 2);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    CommunicatorTest,
    testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
    testing::PrintToStringParamName());

} // namespace nvfuser
