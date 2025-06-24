// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda.h>

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
  if (val->dtype() == DataType::Float8_e4m3fn ||
      val->dtype() == DataType::Float8_e5m2) {
// See release note
// https://docs.nvidia.com/cuda/archive/12.1.0/parallel-thread-execution/index.html#ptx-isa-version-8-1
#if (CUDA_VERSION >= 12010)
    ensureVersion(
        {8, 9},
        "Fusion contains Float8_xxx values which was introduced in Ada (8.9)");
// See release note
// https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#ptx-isa-version-7-8
#elif (CUDA_VERSION >= 11080)
    ensureVersion(
        {9, 0},
        "Fusion contains Float8_xxx values which was introduced in Hopper "
        "(9.0)");
#else
    NVF_ERROR(
        "Fusion contains Float8_xxx values which was not supported in given "
        "CUDA version");
#endif // (CUDA_VERSION >= 12010)
  }
  if (val->dtype() == DataType::Float8_e8m0fnu) {
    ensureVersion(
        {10, 0},
        "Fusion contains Float8_e8m0fnu values which was introduced in "
        "Blackwell (10.0)");
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
  } else if (
      ls_op->opType() == LoadStoreOpType::CpAsyncBulkTensorTile ||
      ls_op->opType() == LoadStoreOpType::CpAsyncBulk) {
    ensureVersion(
        {9, 0},
        "LoadStoreOpType::CpAsyncBulk{TensorTile} requires Hopper (9.0) or "
        "newer");
  }
}

void MinimumDeviceVersion::handle(TensorView* tv) {
  bool enable_register_sharing = std::holds_alternative<WarpSpecialized>(
                                     tv->circularBufferOptions().type) &&
      std::get<WarpSpecialized>(tv->circularBufferOptions().type)
          .num_registers.has_value();
  if (enable_register_sharing) {
    ensureVersion(
        {9, 0},
        "Warp Specialized Circular Buffering uses the setmaxnreg ptx "
        "instruction, which requires Hopper (9.0)");
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
