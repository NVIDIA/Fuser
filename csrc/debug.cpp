// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>

#include <iostream>

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

std::ostream& debug() {
  return DebugStreamGuard::getCurStream();
}

} // namespace nvfuser
