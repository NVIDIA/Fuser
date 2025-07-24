// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <chrono>
#include <ostream>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <multidevice/communicator.h>
#include <tests/cpp/multidevice.h>

namespace std {
namespace chrono {
// Without this, EXPECT_* would print duration as bytes.
void PrintTo(const std::chrono::duration<double>& d, ostream* os) {
  *os << d.count() << " seconds";
}
} // namespace chrono
} // namespace std

namespace nvfuser {

using testing::PrintToString;

class CommunicatorTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<CommunicatorBackend> {};

namespace {
template <typename Duration>
auto toSeconds(const Duration& d) {
  // By default, std::chrono::duration uses ratio 1, which means seconds.
  return std::chrono::duration_cast<std::chrono::duration<double>>(d);
}

MATCHER_P2(
    IsBetween,
    lower,
    upper,
    std::string(negation ? "isn't" : "is") + " between " +
        PrintToString(lower) + " and " + PrintToString(upper)) {
  return lower <= arg && arg <= upper;
}
} // namespace

// A regression test for #2499.
//
// It's currently disabled for a potential flake. One way to fix that is:
// ```
// for each iteration i:
//   timestamps[i] = now()
//   barrier()
//
// prev_max = -inf
// for each iteration i:
//   min = allreduce(timestamps[i], MIN)
//   assert prev_max <= min
//   prev_max = allreduce(timestamps[i], MAX)
// ```
TEST_P(CommunicatorTest, DISABLED_Barrier) {
  using clock = std::chrono::high_resolution_clock;

  const auto rank = communicator_->deviceId();
  constexpr int kNumIterations = 4;
  constexpr auto kUnitDuration = std::chrono::milliseconds(500);

  std::vector<std::chrono::time_point<clock>> end_times;
  end_times.reserve(kNumIterations);
  for ([[maybe_unused]] auto _ : arange(kNumIterations)) {
    // The last rank enters the barrier the last. Therefore, the duration per
    // iteration is expected to be `kUnitDuration*(num_devices - 1)`.
    std::this_thread::sleep_for(kUnitDuration * rank);
    communicator_->barrier();
    end_times.push_back(clock::now());
  }

  const auto expected_duration = kUnitDuration * (communicator_->size() - 1);
  for (int i = 1; i < kNumIterations; i++) {
    // Expect `duration` to be close enoguh to `expected_duration`. Convert to
    // duration<double> (and thus seconds) before comparison for a better error
    // message.
    const auto duration = toSeconds(end_times[i] - end_times[i - 1]);
    const auto expected_upper =
        toSeconds(expected_duration + kUnitDuration / 2);
    const auto expected_lower =
        toSeconds(expected_duration - kUnitDuration / 2);
    EXPECT_THAT(duration, IsBetween(expected_lower, expected_upper))
        << "Duration of iteration " << i
        << " is outside the range of expectation.";
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    CommunicatorTest,
    testing::Values(CommunicatorBackend::kNccl),
    testing::PrintToStringParamName());

} // namespace nvfuser
