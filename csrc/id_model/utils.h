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

namespace nvfuser {

// Options to enable the IdModel-based tensor indexer selectively
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

} // namespace nvfuser
