// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/id_model.h>
#include <id_model/to_string.h>
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
  Inlining,
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

  if (hasEnableOptionArgument(EnableOption::IdModel, "inlining")) {
    opts.insert(IdModelEnableOption::Inlining);
  }

  return opts;
}

inline bool isIdModelOptionEnabled(IdModelEnableOption option) {
  const auto opts = getIdModelEnabledOptions();
  return opts.find(option) != opts.end();
}

// Get the promotion domain of a given loop domain.
inline IterDomain* getLoopPromotion(
    IterDomain* loop_id,
    const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(loop_id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ",
      loop_id->toString(),
      ". Loop group: ",
      nvfuser::toString(loop_group));

  return loop_promotion_map_it->second;
}

} // namespace nvfuser
