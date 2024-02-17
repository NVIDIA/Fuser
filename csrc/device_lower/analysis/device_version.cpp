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

void MinimumDeviceVersion::dispatch(Val* val) {
  if (val->dtype() == DataType::BFloat16) {
    ensureVersion(
        {8, 0},
        "Fusion contains BFloat16 values which was introduced in Ampere (8.0)");
  }
  IterVisitor::dispatch(val);
}

void MinimumDeviceVersion::handle(MmaOp* mma_op) {
  if (isTuring(mma_op->macro())) {
    ensureVersion({7, 5}, "Fusion contains a Turing MMA macro");
  } else if (isAmpere(mma_op->macro())) {
    ensureVersion({8, 0}, "Fusion contains an Ampere MMA macro");
  } else if (isHopper(mma_op->macro())) {
    ensureVersion({9, 0}, "Fusion contains a Hopper MMA macro");
  } else {
    NVF_ERROR(
        "MmaOp ",
        mma_op->toString(),
        " has macro ",
        toString(mma_op->macro()),
        " which does not appear to be Turing, Ampere, or Hopper");
  }
}

void MinimumDeviceVersion::handle(LoadStoreOp* ls_op) {
  if (ls_op->opType() == LoadStoreOpType::CpAsync) {
    ensureVersion(
        {8, 0}, "LoadStoreOpType::CpAsync requires Ampere (8.0) or newer");
  } else if (ls_op->opType() == LoadStoreOpType::CpAsyncBulkTensorTile) {
    ensureVersion(
        {9, 0},
        "LoadStoreOpType::CpAsyncBulkTensorTile requires Hopper (9.0) or newer");
  }
}

void MinimumDeviceVersion::ensureVersion(
    std::pair<int, int> version,
    std::string reason) {
  if (version > min_version_) {
    min_version_ = version;
    reason_ = reason;
  }
}

} // namespace nvfuser
