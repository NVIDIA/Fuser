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

std::optional<StaticRepeatInfo> getMaybeStaticRepeatId(TensorView* ref_tv) {
  // Assume ref is a fusion output
  NVF_ERROR(ref_tv->isFusionOutput());

  // Find the broadcast, expand and reshape pattern.

  TensorView* reshape_out_tv = ref_tv;

  // It is likely a cache is inserted
  auto load_store = dynamic_cast<LoadStoreOp*>(ref_tv->definition());
  // Only Set is considered for now
  if (load_store != nullptr) {
    if (load_store->opType() != LoadStoreOpType::Set) {
      return std::nullopt;
    }
    reshape_out_tv = load_store->input(0)->as<TensorView>();
    // Not sure if this is really problematic, but the producer of the
    // caching op should have only one consumer
    if (reshape_out_tv->uses().size() > 1) {
      return std::nullopt;
    }
  }

  std::cerr << "Reshape out: " << reshape_out_tv->toString() << "\n";

  // The pattern to detect:
  //
  // broadcast_out = broadcast(input)
  // expand_out = expand(broadcast_out)
  // reshape_out = reshape(expand_out)

  auto reshape = dynamic_cast<ViewOp*>(reshape_out_tv->definition());
  if (reshape == nullptr) {
    return std::nullopt;
  }

  std::cerr << reshape->toString();

  auto expand_out_tv = reshape->in();

  auto expand = dynamic_cast<ExpandOp*>(expand_out_tv->definition());
  if (expand == nullptr) {
    return std::nullopt;
  }

  std::cerr << expand->toString();

  auto broadcast_out_tv = expand->in();

  auto broadcast = dynamic_cast<BroadcastOp*>(broadcast_out_tv->definition());
  if (broadcast == nullptr) {
    return std::nullopt;
  }

  std::cerr << broadcast->toString();

  auto inp_tv = broadcast->in();

  std::cerr << "Inp tv: " << inp_tv->toString() << "\n";

  // Not sure if this is really necessary to check, but assume there's
  // only single chain of the ops and tensors from inp_tv to the
  // fusion outputs
  auto all_dep_vals =
      DependencyCheck::getAllValsBetween({inp_tv}, inp_tv->fusion()->outputs());
  if (std::unordered_set<Val*>{all_dep_vals.begin(), all_dep_vals.end()} !=
      std::unordered_set<Val*>{
          inp_tv, broadcast_out_tv, expand_out_tv, reshape_out_tv, ref_tv}) {
    return std::nullopt;
  }

  std::cerr << "All dep vals: " << toDelimitedString(all_dep_vals) << "\n";

  // Check if the ops match with the repeat pattern. Currently only
  // one iter domain can be repeated
  IterDomain* broadcast_id = nullptr;
  int64_t broadcast_pos = -1;
  for (const auto i :
       c10::irange(broadcast_out_tv->getLogicalDomain().size())) {
    if (broadcast->getBroadcastDimFlags().at(i)) {
      if (broadcast_id != nullptr) {
        // Multiple broadcast IDs not supported
        return std::nullopt;
      }
      broadcast_id = broadcast_out_tv->getLogicalDomain().at(i);
      broadcast_pos = (int64_t)i;
    }
  }

  if (broadcast_id == nullptr) {
    return std::nullopt;
  }

  std::cerr << "Broadcast ID: " << broadcast_id->toString() << "\n";

  // Check if and only if the broadcast ID is expanded
  IterDomain* expanded_id = nullptr;
  for (const auto i :
       c10::irange(broadcast_out_tv->getLogicalDomain().size())) {
    auto p_id = broadcast_out_tv->getLogicalDomain().at(i);
    auto c_id = expand_out_tv->getLogicalDomain().at(i);
    std::cerr << "p_id: " << p_id->toString() << ", c_id: " << c_id->toString()
              << "\n";
    if (p_id == broadcast_id && c_id->isBroadcast() &&
        c_id->hasExpandedExtent()) {
      expanded_id = c_id;
      std::cerr << "Expand: " << c_id->toString() << "\n";
    } else if (
        p_id->isBroadcast() && !p_id->hasExpandedExtent() &&
        c_id->isBroadcast() && c_id->hasExpandedExtent()) {
      // Expanded but this broadcast was not introduced by the
      // preceding broadcast op
      std::cerr << "Non-broadcast expansion: " << p_id->toString() << ", "
                << c_id->toString() << "\n";
      return std::nullopt;
    }
  }

  if (expanded_id == nullptr) {
    return std::nullopt;
  }

  std::cerr << "Expand ID: " << expanded_id->toString() << "\n";

  // Only a static repeat factor is considered
  if (!expanded_id->expandedExtent()->isConstInt()) {
    std::cerr << "Non-const expand\n";
    return std::nullopt;
  }

  // The expanded ID should be merged with the iter domain next to it,
  // and that should be the only reshape expr
  auto reshape_exprs = DependencyCheck::getAllExprsBetween(
      {reshape_out_tv->getRootDomain().begin(),
       reshape_out_tv->getRootDomain().end()},
      {reshape_out_tv->getLogicalDomain().begin(),
       reshape_out_tv->getLogicalDomain().end()});
  if (reshape_exprs.size() != 1) {
    std::cerr << "More exprs: " << reshape_exprs.size() << "\n";
    return std::nullopt;
  }

  std::cerr << reshape_exprs.at(0)->toString();

  auto reshape_merge = dynamic_cast<Merge*>(reshape_exprs.at(0));
  if (reshape_merge == nullptr) {
    return std::nullopt;
  }

  std::cerr << "Reshape merge: " << reshape_merge->toString() << "\n";

  auto reshape_root_broadcast =
      reshape_out_tv->getRootDomain().at(broadcast_pos);
  // IterDomain* ref_repeated_id = nullptr;
  if (reshape_merge->outer() != reshape_root_broadcast &&
      reshape_merge->inner() != reshape_root_broadcast) {
    std::cerr << "Invalid merge\n";
    return std::nullopt;
  }

  // When ref_tv != reshape_out_tv due to caching, assume the loop
  // domain of the reference is already transformed to cancel the
  // reshape
  NVF_ERROR(
      ref_tv->getLoopDomain().size() == reshape_out_tv->getRootDomain().size());

  StaticRepeatInfo info;
  info.ref_repeating_id = ref_tv->getLoopDomain().at(broadcast_pos);
  info.repeated_tvs =
      std::vector<TensorView*>{broadcast_out_tv, expand_out_tv, reshape_out_tv};
  if (reshape_out_tv != ref_tv) {
    info.repeated_tvs.push_back(ref_tv);
  }

  return info;
}

} // namespace scheduler_tools
} // namespace nvfuser
