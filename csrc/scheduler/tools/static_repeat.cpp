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

  auto current_out = maybe_repeat_out;

  // Check if there's a cache
  while (true) {
    if (auto ldst = dynamic_cast<LoadStoreOp*>(current_out->definition());
        ldst != nullptr &&
        (ldst->opType() == LoadStoreOpType::Set ||
         ldst->opType() == LoadStoreOpType::SegmenterSet)) {
      auto ldst_in = ldst->in()->as<TensorView>();
      repeat_tvs.insert(ldst_in);
      current_out = ldst_in;
    } else {
      break;
    }
  }

  auto reshape_out = current_out;

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

  auto inp_tv = expand->in()->as<TensorView>();
  BroadcastOp* broadcast = nullptr;
  TensorView* broadcast_out = nullptr;

  if (getenv("REQUIRE_BROADCAST")) {
    // Detect broadcast
    broadcast_out = expand->in();
    repeat_tvs.insert(broadcast_out);
    broadcast = dynamic_cast<BroadcastOp*>(broadcast_out->definition());
    if (broadcast == nullptr) {
      return std::nullopt;
    }

    inp_tv = broadcast->in()->as<TensorView>();
  }

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
  if (getenv("REQUIRE_BROADCAST")) {
    for (const auto i : arange(broadcast_out->getLogicalDomain().size())) {
      if (broadcast->getBroadcastDimFlags().at(i)) {
        if (broadcast_id != nullptr) {
          // Multiple broadcast IDs not supported
          return std::nullopt;
        }
        broadcast_id = broadcast_out->getLogicalDomain().at(i);
      }
    }
  }

  // Check if and only if the broadcast ID is expanded
  IterDomain* expanded_id = nullptr;
  auto expand_c2p = PairwiseLogicalDomainMap(expand->in(), expand->out())
                        .mapConsumerToProducer();
  for (const auto i : arange(expand->out()->getLogicalDomain().size())) {
    auto c_id = expand->out()->getLogicalDomain().at(i);
    auto p_id = expand_c2p.at(c_id);
    if (p_id->isBroadcast() && p_id->extent()->isOneInt() &&
        c_id->isBroadcast() && c_id->hasExpandedExtent()) {
      if (broadcast_id == p_id) {
        expanded_id = c_id;
      } else {
        expanded_id = c_id;
        broadcast_id = p_id;
      }
    }
  }

  if (expanded_id == nullptr) {
    return std::nullopt;
  }

  if (broadcast_id == nullptr) {
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
  auto reshape_root_broadcast =
      PairwiseLogicalDomainMap(expand_out, reshape_out)
          .mapProducerToConsumer()
          .at(expanded_id);
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

  IterDomain* reshape_root_inp_id =
      reshape_merge->outer() == reshape_root_broadcast ? reshape_merge->inner()
                                                       : reshape_merge->outer();

  int64_t inp_id_pos = std::distance(
      reshape_out->getRootDomain().begin(),
      std::ranges::find(reshape_out->getRootDomain(), reshape_root_inp_id));
  IterDomain* inp_tv_id = nullptr;
  if (getenv("REQUIRE_BROADCAST")) {
    auto broadcast_inp_id = broadcast_out->getLogicalDomain().at(inp_id_pos);
    inp_tv_id = PairwiseLogicalDomainMap(inp_tv, broadcast_out)
                    .mapConsumerToProducer()
                    .at(broadcast_inp_id);
  } else {
    inp_tv_id = PairwiseLogicalDomainMap(inp_tv, expand_out)
                    .mapConsumerToProducer()
                    .at(expand_out->getLogicalDomain().at(inp_id_pos));
  }

  StaticRepeatInfo info;
  info.repeat_input_tv = inp_tv;
  info.repeat_input_id = inp_tv_id;
  info.repeat_output_tv = maybe_repeat_out;
  info.reshape_output_tv = reshape_out;
  info.reshape_repeat_id = reshape_root_broadcast;
  info.repeat_tvs = repeat_tvs;

  info.input_broadcast_id = broadcast_id;

  return info;
}

} // namespace scheduler_tools
} // namespace nvfuser
