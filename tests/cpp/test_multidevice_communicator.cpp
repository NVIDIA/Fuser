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

namespace {
template <typename Duration>
double toSeconds(const Duration& d) {
  // By default, std::chrono::duration uses ratio 1, which means seconds.
  return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
}
} // namespace

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
    // Expect `duration` to be close enoguh to `expected_duration`. Convert to
    // seconds before comparison for a better error message. EXPECT_ has
    // problems printing a duration.
    const double duration = toSeconds(end_times[i] - end_times[i - 1]);
    const double expected_upper =
        toSeconds(expected_duration + kUnitDuration / 2);
    const double expected_lower =
        toSeconds(expected_duration - kUnitDuration / 2);
    EXPECT_LE(duration, expected_upper);
    EXPECT_GE(duration, expected_lower);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    CommunicatorTest,
    testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
    testing::PrintToStringParamName());

} // namespace nvfuser
