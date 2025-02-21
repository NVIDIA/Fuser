// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <expr_simplifier.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <ir/utils.h>
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
  // Uses the loop promotion to generate loops. Indexing and
  // predication need to be enabled as well.
  Loop,
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

  if (hasEnableOptionArgument(EnableOption::IdModel, "loop") ||
      hasEnableOptionArgument(EnableOption::IdModel, "all")) {
    opts.insert(IdModelEnableOption::Loop);
  }

  // Loop requires ConsumerIndex, ProducerIndex, InlinePredicate and
  // UnswitchPredicate
  if (opts.find(IdModelEnableOption::Loop) != opts.end()) {
    NVF_ERROR(
        opts.find(IdModelEnableOption::ConsumerIndex) != opts.end(),
        "ConsumerIndex required for Loop");
    NVF_ERROR(
        opts.find(IdModelEnableOption::ProducerIndex) != opts.end(),
        "ProducerIndex required for Loop");
    NVF_ERROR(
        opts.find(IdModelEnableOption::InlinePredicate) != opts.end(),
        "InlinePredicate required for Loop");
    NVF_ERROR(
        opts.find(IdModelEnableOption::UnswitchPredicate) != opts.end(),
        "UnswitchPredicate required for Loop");
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

// Get the loop domains of a given expr. Currently, they're always
// the loop domains of a consumer tensor, but in the future this
// function may return the loop domains of a producer for
// producer-based indexing.
inline std::vector<IterDomain*> getLoopIds(
    const Expr* expr,
    const IdModel& id_model) {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  NVF_ERROR(!expr->outputs().empty());
  auto output_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(output_tv != nullptr);
  auto loop_ids = output_tv->getLoopDomain();

  for (auto& loop_id : loop_ids) {
    loop_id = getLoopPromotion(loop_id, id_model);
  }

  return loop_ids;
}

inline ParallelType getParallelType(const ValGroup& loop_group) {
  ParallelType common_pt = ParallelType::Serial;
  for (const auto val : *loop_group) {
    auto pt = val->as<IterDomain>()->getParallelType();
    if (common_pt == pt || pt == ParallelType::Serial) {
      continue;
    } else if (common_pt == ParallelType::Serial) {
      common_pt = pt;
    } else {
      // Inconsistent parallelization
      NVF_THROW(
          "Inconsistent parallelization detected. ",
          "Known type: ",
          common_pt,
          "New type: ",
          pt);
    }
  }

  return common_pt;
}

// Check if the loop index of a loop group should be always
// just zero. For example, a loop group with an extent of one, i.e.,
// a broadcast-only loop group, should just use zero.
inline bool shouldUseZeroIndex(
    const ValGroup& loop_group,
    const IdModel& id_model) {
  // Trivial loop
  auto promotion_id =
      getLoopPromotion(loop_group->front()->as<IterDomain>(), id_model);

  // ExprSimplify should be disabled here as it would fail to
  // recognize size-one IterDomain.
  DisableOptionsGuard options_guard;
  DisableOptionsGuard::getCurOptions().unset(DisableOption::ExprSimplify);

  if (promotion_id->isBroadcast() ||
      simplifyExpr(promotion_id->extent())->isOneInt()) {
    return true;
  }

  return false;
}

} // namespace nvfuser
