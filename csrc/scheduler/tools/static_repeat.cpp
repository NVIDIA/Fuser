// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
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
      ldst != nullptr && ldst->opType() == LoadStoreOpType::Set) {
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
  if (inp_tv->uses().size() > 1 &&
      std::any_of(repeat_tvs.begin(), repeat_tvs.end(), [](TensorView* tv) {
        return tv->uses().size() > 1;
      })) {
    return std::nullopt;
  }

  // Check if the ops match with the repeat pattern. Currently only
  // one iter domain can be repeated
  IterDomain* broadcast_id = nullptr;
  int64_t broadcast_pos = -1;
  for (const auto i : arange(broadcast_out->getLogicalDomain().size())) {
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
  for (const auto i : arange(broadcast_out->getLogicalDomain().size())) {
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

  // Reshape of an expanded broadcast always generates a concrete
  // non-broadcast ID, so this check is not necessary, but just in
  // case in the future that may change.
  if (reshape_merge->out()->isBroadcast() ||
      reshape_merge->out()->hasExpandedExtent()) {
    return std::nullopt;
  }

  StaticRepeatInfo info;
  info.repeat_output_tv = maybe_repeat_out;
  info.reshape_output_tv = reshape_out;
  info.reshape_repeat_id = reshape_out->getRootDomain().at(broadcast_pos);
  info.repeat_tvs = repeat_tvs;

  return info;
}

std::optional<StaticRepeatInfo> getMaybeStaticRepeatInfo(
    TensorView* maybe_repeat_out_tv);

std::optional<StaticRepeatingReshapeInfo> getMaybeStaticRepeatingReshapeInfo(
    ViewOp* reshape) {
  auto reshape_in = reshape->input(0)->as<TensorView>();
  auto reshape_out = reshape->output(0)->as<TensorView>();

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

  // Reshape of an expanded broadcast always generates a concrete
  // non-broadcast ID, so this check is not necessary, but just in
  // case in the future that may change.
  if (reshape_merge->out()->isBroadcast() ||
      reshape_merge->out()->hasExpandedExtent()) {
    return std::nullopt;
  }

  StaticRepeatingReshapeInfo info;

  info.output_id = reshape_merge->out();

  const auto c2p =
      PairwiseLogicalDomainMap(reshape_in, reshape_out).mapConsumerToProducer();

  auto producer_merge_outer = c2p.at(reshape_merge->outer());
  auto producer_merge_inner = c2p.at(reshape_merge->inner());
  IterDomain* producer_factor_id = nullptr;

  if (producer_merge_outer->isBroadcast() &&
      producer_merge_outer->hasExpandedExtent() &&
      !producer_merge_inner->isBroadcast()) {
    // Inner ID is repeated by the factor of the outer extent
    info.input_id = reshape_merge->inner();
    info.factor_id = reshape_merge->outer();
    producer_factor_id = producer_merge_outer;
  } else if (
      producer_merge_inner->isBroadcast() &&
      producer_merge_inner->hasExpandedExtent() &&
      !producer_merge_outer->isBroadcast()) {
    // Outer ID is repeated by the factor of the inner extent
    info.input_id = reshape_merge->outer();
    info.factor_id = reshape_merge->inner();
    producer_factor_id = producer_merge_inner;
  } else {
    return std::nullopt;
  }

  // Check if the expanded ID has a static expanded extent
  if (!producer_factor_id->expandedExtent()->isConstInt()) {
    return std::nullopt;
  }

  return info;
}

std::optional<StaticRepeatingReshapeInfo> getMaybeStaticRepeatingReshapeInfo(
    TensorView* maybe_repeat_out_tv) {
  // Skip a set if any (e.g., inserted by caching)
  if (auto ldst = dynamic_cast<LoadStoreOp*>(maybe_repeat_out_tv->definition());
      ldst != nullptr && ldst->opType() == LoadStoreOpType::Set) {
    maybe_repeat_out_tv = ldst->in()->as<TensorView>();
  }

  // Detect reshape
  auto reshape = dynamic_cast<ViewOp*>(maybe_repeat_out_tv->definition());
  if (reshape == nullptr) {
    return std::nullopt;
  }

  return getMaybeStaticRepeatingReshapeInfo(reshape);
}

} // namespace scheduler_tools
} // namespace nvfuser
