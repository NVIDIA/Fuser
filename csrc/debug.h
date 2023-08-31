// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

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
class TORCH_CUDA_CU_API DebugStreamGuard {
 public:
  DebugStreamGuard(std::ostream& stream);

  ~DebugStreamGuard();

  static std::ostream& getCurStream();

  void setCurStream(std::ostream& stream);

 private:
  std::ostream* prev_stream_;
};

//! This is just a short alias to avoid having to type
//! DebugStreamGuard::getCurStream() for each line we want to debug-print.
TORCH_CUDA_CU_API std::ostream& debug();

} // namespace nvfuser
