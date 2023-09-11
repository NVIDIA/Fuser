// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <debug.h>
#include <options.h>
#include <utils.h>

#include <iostream>

namespace nvfuser {

namespace scheduler_debug_utils {

// Basic logging utility for any messages in scheduler or segmenter
template <typename... Args>
void canScheduleMessage(const Args&... args) {
  // Using builtin expect to reduce the overhead slightly,
  //  alternatively may want to allow this message in debug
  //  build only but that'd be inconvenient for user support.
  if (C10_UNLIKELY(isDebugDumpEnabled(DebugDumpOption::FusionSegmenterLog))) {
    debug() << c10::str(args...) << "\n";
  }
}

// Short-cut message for flagging why shedulers cannot schedule fusions,
//  assuming first argument is heuristic type (not actively checked).
template <typename HeuristicType, typename... Args>
void canScheduleRejectReason(HeuristicType heuristic, const Args&... args) {
  canScheduleMessage(
      "Scheduler _", heuristic, "_ ***rejected*** because : ", args...);
}

// Based on
// https://learn.microsoft.com/en-us/cpp/cpp/ellipses-and-variadic-templates?view=msvc-170#example
inline void log() {
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    debug() << std::endl;
  }
}

template <typename T>
void log(const T& t) {
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    debug() << t << std::endl;
  }
}

template <typename First, typename... Rest>
void log(const First& first, const Rest&... rest) {
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    debug() << first;
    log(rest...);
  }
}

} // namespace scheduler_debug_utils

} // namespace nvfuser
