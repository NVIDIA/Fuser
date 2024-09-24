// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iostream>

#include <debug.h>
#include <multidevice/communicator.h>

namespace nvfuser {

static thread_local std::ostream* ACTIVE_STREAM = &std::cout;

DebugStreamGuard::DebugStreamGuard(std::ostream& stream)
    : prev_stream_{ACTIVE_STREAM} {
  ACTIVE_STREAM = &stream;
}

DebugStreamGuard::~DebugStreamGuard() {
  ACTIVE_STREAM = prev_stream_;
}

std::ostream& DebugStreamGuard::getCurStream() {
  return *ACTIVE_STREAM;
}
void DebugStreamGuard::setCurStream(std::ostream& stream) {
  ACTIVE_STREAM = &stream;
}

std::ostream& debug(const bool only_first_local_rank) {
  auto should_print = [&]() -> bool {
    if (!only_first_local_rank) {
      return true;
    }

    const auto& communicator = Communicator::getInstance();
    if (!communicator.is_available()) {
      return true;
    }

    if (communicator.local_rank() == 0) {
      return true;
    }

    return false;
  };

  if (should_print()) {
    return DebugStreamGuard::getCurStream();
  }

  static std::ofstream null_stream("/dev/null");
  return null_stream;
}

} // namespace nvfuser
