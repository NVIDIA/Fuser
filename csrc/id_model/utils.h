// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <utils.h>

#include <functional>
#include <iostream>
#include <sstream>

#define VERBOSE() verbose(__LINE__)
#define WARN() warn(__LINE__)

namespace nvfuser {

// Temporary logging utility
class DebugStream {
 public:
  DebugStream()
      : enabled_(getNvFuserEnv("ID_MODEL_VERBOSE")), out_(std::cerr) {}

  template <typename T>
  DebugStream& operator<<(const T& v) {
    if (enabled_) {
      out_ << v;
    }
    return *this;
  }

  DebugStream& operator<<(std::ostream& (*endl)(std::ostream&)) {
    if (enabled_) {
      out_ << endl;
    }
    return *this;
  }

 private:
  bool enabled_ = false;
  std::ostream& out_;
};

inline DebugStream verbose(int line) {
  return DebugStream() << "[DEBUG@" << line << "] ";
}

inline DebugStream warn(int line) {
  return DebugStream() << "[WARN@" << line << "] ";
}

} // namespace nvfuser
