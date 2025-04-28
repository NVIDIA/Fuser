// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>

#include <c10/util/ThreadLocal.h>
#include <exceptions.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

// Thread-local variable to indicate parallel context
thread_local bool g_is_parallel_compile_thread = false;

// RAII Guard to manage the flag
class ParallelCompileContextGuard {
 public:
  ParallelCompileContextGuard() {
    // Remember the previous state in case of nested parallel calls (unlikely
    // for compile but good practice)
    prev_state_ = g_is_parallel_compile_thread;
    g_is_parallel_compile_thread = true;
  }
  ~ParallelCompileContextGuard() {
    g_is_parallel_compile_thread = prev_state_;
  }
  // Disable copy/move
  ParallelCompileContextGuard(const ParallelCompileContextGuard&) = delete;
  ParallelCompileContextGuard& operator=(const ParallelCompileContextGuard&) =
      delete;

 private:
  bool prev_state_ = false; // Store previous state for nesting
};

//! This guard controls the output for debug info, such as any output resulting
//! from use of the $NVFUSER_DUMP environment variable options. Debug output can
//! be captured like so:
//!
//!   std::stringstream ss
//!   {
//!     DebugStreamGuard dsg(ss);
//!     // Unmodified original code
//!
//!     // ss.str() now holds a std::string of debug info
//!     // The guard resets the debug stream at the end of its lifetime
//!   }
//!   // Code after the dsg object is destroyed will use the previously-set
//!   // stream, which defaults to std::cout.
class DebugStreamGuard {
 public:
  NVF_API DebugStreamGuard(std::ostream& stream);

  NVF_API ~DebugStreamGuard();

  static std::ostream& getCurStream();

  void setCurStream(std::ostream& stream);

 private:
  std::ostream* prev_stream_;
};

//! This is just a short alias to avoid having to type
//! DebugStreamGuard::getCurStream() for each line we want to debug-print.
NVF_API std::ostream& debug();

} // namespace nvfuser
