// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <options.h>
#include <utils.h>

#include <functional>
#include <iostream>
#include <sstream>

#define VERBOSE() verbose(__LINE__)
#define WARN() warn(__LINE__)

namespace nvfuser {

enum class IdModelEnableOption {
  ConsumerIndex,
  ProducerIndex,
  InlinePredicate,
  UnswitchPredicate,
};

inline std::unordered_set<IdModelEnableOption> getIdModelEnabledOptions() {
  std::unordered_set<IdModelEnableOption> opts;

  if (hasEnableOptionArgument(EnableOption::IdModel, "consumer_index") ||
      hasEnableOptionArgument(EnableOption::IdModel, "index") ||
      hasEnableOptionArgument(EnableOption::IdModel, "all")) {
    opts.insert(IdModelEnableOption::ConsumerIndex);
  }

  if (hasEnableOptionArgument(EnableOption::IdModel, "producer_index") ||
      hasEnableOptionArgument(EnableOption::IdModel, "index") ||
      hasEnableOptionArgument(EnableOption::IdModel, "all")) {
    opts.insert(IdModelEnableOption::ProducerIndex);
  }

  if (hasEnableOptionArgument(EnableOption::IdModel, "inline_predicate") ||
      hasEnableOptionArgument(EnableOption::IdModel, "predicate") ||
      hasEnableOptionArgument(EnableOption::IdModel, "all")) {
    opts.insert(IdModelEnableOption::InlinePredicate);
  }

  if (hasEnableOptionArgument(EnableOption::IdModel, "unswitch_predicate") ||
      hasEnableOptionArgument(EnableOption::IdModel, "predicate") ||
      hasEnableOptionArgument(EnableOption::IdModel, "all")) {
    opts.insert(IdModelEnableOption::UnswitchPredicate);
  }

  return opts;
}

inline bool isIdModelOptionEnabled(IdModelEnableOption option) {
  const auto opts = getIdModelEnabledOptions();
  return opts.find(option) != opts.end();
}

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
