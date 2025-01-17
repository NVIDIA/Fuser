// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <scheduler/tools/static_repeat.h>

namespace nvfuser {
namespace scheduler_tools {

std::optional<StaticRepeatInfo> getMaybeStaticRepeatInfo(
    TensorView* maybe_repeat_out) {
  // The pattern to detect:
  //
  // broadcast_out = broadcast(input)
  // expand_out = expand(broadcast_out)
  // repeat_out = reshape(expand_out)
  //
  // Additionally, since maybe_repeat_out is commonly a fusion
  // output, it is likely there's a cache tv between expand_out and
  // repeat_out, so the following pattern should also be detected.
  //
  // broadcast_out = broadcast(input)
  // expand_out = expand(broadcast_out)
  // cache_of_repeat_out = reshape(expand_out)
  // repeat_out = set(cache_of_repeat_out)

  std::unordered_set<TensorView*> repeat_tvs;
  repeat_tvs.insert(maybe_repeat_out);

  auto reshape_out = maybe_repeat_out;

  // Check if there's a cache
  if (auto ldst = dynamic_cast<LoadStoreOp*>(maybe_repeat_out->definition());
      ldst->opType() == LoadStoreOpType::Set) {
    reshape_out = ldst->in()->as<TensorView>();
    repeat_tvs.insert(reshape_out);
  }

  // Detect reshape
  auto reshape = dynamic_cast<ViewOp*>(reshape_out->definition());
  if (reshape == nullptr) {
    return std::nullopt;
  }

  // Detect expand
  auto expand_out = reshape->in();
  repeat_tvs.insert(expand_out);
  auto expand = dynamic_cast<ExpandOp*>(expand_out->definition());
  if (expand == nullptr) {
    return std::nullopt;
  }

  // Detect broadcast
  auto broadcast_out = expand->in();
  repeat_tvs.insert(broadcast_out);
  auto broadcast = dynamic_cast<BroadcastOp*>(broadcast_out->definition());
  if (broadcast == nullptr) {
    return std::nullopt;
  }

  auto inp_tv = broadcast->in();

  // Not sure if this is really necessary to check, but assume there's
  // only single chain of the ops and tensors from inp_tv to
  // maybe_reshape_out
  if (std::any_of(repeat_tvs.begin(), repeat_tvs.end(), [](TensorView* tv) {
        return tv->uses().size() > 1;
      })) {
    return std::nullopt;
  }

  // Check if the ops match with the repeat pattern. Currently only
  // one iter domain can be repeated
  IterDomain* broadcast_id = nullptr;
  int64_t broadcast_pos = -1;
  for (const auto i : c10::irange(broadcast_out->getLogicalDomain().size())) {
    if (broadcast->getBroadcastDimFlags().at(i)) {
      if (broadcast_id != nullptr) {
        // Multiple broadcast IDs not supported
        return std::nullopt;
      }
      broadcast_id = broadcast_out->getLogicalDomain().at(i);
      broadcast_pos = (int64_t)i;
    }
  }

  if (broadcast_id == nullptr) {
    return std::nullopt;
  }

  // Check if and only if the broadcast ID is expanded
  IterDomain* expanded_id = nullptr;
  for (const auto i : c10::irange(broadcast_out->getLogicalDomain().size())) {
    auto p_id = broadcast_out->getLogicalDomain().at(i);
    auto c_id = expand_out->getLogicalDomain().at(i);
    if (p_id == broadcast_id && c_id->isBroadcast() &&
        c_id->hasExpandedExtent()) {
      expanded_id = c_id;
    } else if (
        p_id->isBroadcast() && !p_id->hasExpandedExtent() &&
        c_id->isBroadcast() && c_id->hasExpandedExtent()) {
      // Expanded but this broadcast was not introduced by the
      // preceding broadcast op
      return std::nullopt;
    }
  }

  if (expanded_id == nullptr) {
    return std::nullopt;
  }

  // Only a static repeat factor is considered
  if (!expanded_id->expandedExtent()->isConstInt()) {
    return std::nullopt;
  }

  // The expanded ID should be merged with the iter domain next to it,
  // and that should be the only reshape expr
  auto reshape_exprs = DependencyCheck::getAllExprsBetween(
      {reshape_out->getRootDomain().begin(),
       reshape_out->getRootDomain().end()},
      {reshape_out->getLogicalDomain().begin(),
       reshape_out->getLogicalDomain().end()});
  if (reshape_exprs.size() != 1) {
    return std::nullopt;
  }

  auto reshape_merge = dynamic_cast<Merge*>(reshape_exprs.at(0));
  if (reshape_merge == nullptr) {
    return std::nullopt;
  }

  // The corresponding root ID of the outout tv should be one of the
  // inputs of the merge
  auto reshape_root_broadcast = reshape_out->getRootDomain().at(broadcast_pos);
  if (reshape_merge->outer() != reshape_root_broadcast &&
      reshape_merge->inner() != reshape_root_broadcast) {
    return std::nullopt;
  }

  StaticRepeatInfo info;
  info.repeat_output_tv = maybe_repeat_out;
  info.reshape_output_tv = reshape_out;
  info.reshape_repeat_id = reshape_out->getRootDomain().at(broadcast_pos);
  info.repeat_tvs = repeat_tvs;

  return info;
}

} // namespace scheduler_tools
} // namespace nvfuser
