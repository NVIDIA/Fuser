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
    TensorView* maybe_repeat_out_tv) {
  // Skip set ops if any (e.g., inserted by caching). Only Set
  // or SegmenterSet are considered.
  while (auto ldst =
             dynamic_cast<LoadStoreOp*>(maybe_repeat_out_tv->definition())) {
    if (ldst->opType() != LoadStoreOpType::Set &&
        ldst->opType() != LoadStoreOpType::SegmenterSet) {
      break;
    }
    maybe_repeat_out_tv = ldst->in()->as<TensorView>();
  }

  // Detect reshape
  auto reshape = dynamic_cast<ReshapeOp*>(maybe_repeat_out_tv->definition());
  if (reshape == nullptr) {
    return std::nullopt;
  }

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

  StaticRepeatInfo info;

  info.reshape_output_tv = reshape_out;
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

} // namespace scheduler_tools
} // namespace nvfuser
