// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/device_version.h>
#include <device_lower/lower2device.h>
#include <mma_type.h>

namespace nvfuser {

void MinimumDeviceVersion::handle(Val* val) {
  if (val->dtype() == DataType::BFloat16) {
    ensureVersion({7, 0});
  }
}

void MinimumDeviceVersion::handle(MmaOp* mma_op) {
  if (isTuring(mma_op->macro())) {
    ensureVersion({7, 0});
  } else if (isAmpere(mma_op->macro())) {
    ensureVersion({8, 0});
  } else if (isHopper(mma_op->macro())) {
    ensureVersion({9, 0});
  } else {
    NVF_ERROR("MmaOp ", mma_op->toString(), " has macro ", toString(mma_op->macro()), " which does not appear to be Turing, Ampere, or Hopper");
  }
}

void MinimumDeviceVersion::ensureVersion(std::pair<int, int> version) {
  if (version > min_version_) {
    min_version_ = version;
  }
}

} // namespace nvfuser
