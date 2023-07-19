// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/pass/hoist_to_host.h>
#include <ir/utils.h>

namespace nvfuser {

void hoistScalarComputationToHost(
    Fusion* kernel,
    std::vector<Val*>& all_known_vals) {
  all_known_vals.insert(
      all_known_vals.end(), kernel->inputs().begin(), kernel->inputs().end());
  if (!kernel->hasManaged("hoist_to_host")) {
    return;
  }
  const auto& hoist_to_host =
      kernel->getManaged<std::vector<Val*>>("hoist_to_host");
  TORCH_CHECK(
      ir_utils::dependenciesSatisfied(
          hoist_to_host, {kernel->inputs().begin(), kernel->inputs().end()}),
      "Cannot hoist to host because some inputs are not provided");
  all_known_vals.insert(
      all_known_vals.end(), hoist_to_host.begin(), hoist_to_host.end());
}

} // namespace nvfuser
