// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_kernel_runtime.h>
#include <tests/cpp/validator.h>
#include <validator_utils.h>

namespace nvfuser {
void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<SchedulerType>& expected_heuristics) {
  const auto& segment_groups = runtime->fusionSegments()->groups();

  NVF_CHECK(
      segment_groups.size() == expected_heuristics.size(),
      "Unexpected segments. Expected: ",
      expected_heuristics.size(),
      ". Actual: ",
      segment_groups.size());

  // Assumes up to two segments exist for simplicity
  NVF_ERROR(
      segment_groups.size() <= 2, "True segment order analysis is required");

  for (auto& group : segment_groups) {
    int64_t segment_order = group->producer_edges.empty() ? 0 : 1;
    NVF_CHECK(
        group->schedulerType() == expected_heuristics.at(segment_order),
        "Expected to use the ",
        expected_heuristics.at(segment_order),
        " scheduler but ",
        group->schedulerType(),
        " was used");
  }
}

} // namespace nvfuser
