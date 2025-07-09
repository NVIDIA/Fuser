// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/vectorize_helper.h>

#include <compute_at_map.h>
#include <contiguity.h>
#include <device_lower/analysis/divisible_split.h>
#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <iter_visitor.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

#include <unordered_set>

namespace nvfuser {
namespace vectorize_helper {

namespace {

// Search through the almost exact map to see if there's a compile time factor
// that can be used. Otherwise return the concrete ID extent which makes
// simplification pass easier as the same factor will be used more consistently
// where possible.
//
// TODO: This is generally useful, would be good to add it to compute at map and
// refactor lowering to use it so we consistently use compile time values or the
// same symbolic value consistently.
Val* commonOrConstExtent(
    std::shared_ptr<const ComputeAtMap> ca_map,
    IterDomain* id) {
  auto disjoint_set = ca_map->idGraph().almostExactNodes().getDisjointSetOf(id);
  for (auto entry : disjoint_set) {
    if (entry->extent()->isConstScalar()) {
      return entry->extent();
    }
  }
  return ca_map->getConcreteMappedID(id, IdMappingMode::ALMOSTEXACT)->extent();
}

} // namespace

Val* ContiguousInnerDimensionsMapper::isFullyProjected(IterDomain* id) {
  return SimplifyingIrBuilder::eqExpr(
      getProjectedExtent(id), commonOrConstExtent(ca_map_, id));
}

ContiguousInnerDimensionsMapper::ContiguousInnerDimensionsMapper(
    TensorView* reference,
    const std::vector<IterDomain*>& ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits)
    // Send null info to MaxInfoSpanning tree because we need state to compute
    // the info object for reference. It's not really needed on construction,
    // just before traversal.
    : MaxInfoSpanningTree(reference, std::make_shared<MappedDomain>()),
      ca_map_(std::move(ca_map)),
      divisible_splits_(divisible_splits) {
  FusionGuard fg(reference->fusion());

  // Exclude reduction IDs if the reference is a fusion input as they
  // don't manifest at all in the fusion. This simplifies the
  // analysis in getContigMergeOfInnerSize, which only looks at
  // non-reduction logical domains. Including reduction domains here
  // can result in incorrect ordering
  // NOTE: this is necessary to enable vectorization in
  // NVFuserTest.FusionSegmentReduceSoftmax_CUDA
  auto logical_domain = reference->getLogicalDomain();
  auto filtered_ids = ids;
  if (reference->isFusionInput()) {
    logical_domain = TensorDomain::noReductions(logical_domain);
    filtered_ids = TensorDomain::noReductions(filtered_ids);
  } else {
    NVF_ERROR(
        !TensorDomain::hasReduction(logical_domain) &&
            !TensorDomain::hasReduction(filtered_ids),
        "Unexpected reduction domain given to ContiguousInnerDimensionsMapper");
  }

  // Record while processing reference's information
  recording_ = true;
  for (auto id : filtered_ids) {
    addProjectedExtent(id, commonOrConstExtent(ca_map_, id));
  }

  // Ordering of dimensions is important in this analysis, if an ordering is
  // contiguous in the reference, but not the target tensor views, then we
  // cannot consider that a contiguous merge dimension for vectorization.
  auto projected_logical = projectId(filtered_ids, logical_domain);

  std::shared_ptr<Information> reference_information = MappedDomain::build(
      projectId(projected_logical, reference->getMaybeRootDomain()),
      projected_logical,
      reference->hasRoot() /*shouldn't matter how we initialize this*/);

  // Stop recording before traversal
  recording_ = false;

  // Set MaxInfoSpanningTree::reference_info_ before traversal
  reference_info_ = reference_information;
  // Set ContiguousInnerDimensionsMapper::tv_infos_ entry for reference
  tv_infos_[reference] = reference_information;

  traverse(this);
}

ContiguousInnerDimensionsMapper ContiguousInnerDimensionsMapper::map(
    TensorView* reference,
    const std::vector<IterDomain*>& ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits) {
  return ContiguousInnerDimensionsMapper(
      reference, ids, ca_map, divisible_splits);
}

template <typename MergeOrSplit>
void ContiguousInnerDimensionsMapper::combinePE(
    const MergeOrSplit* merge_or_split,
    bool outer_maps) {
  // Nothing to do unless recording
  if (!recording_) {
    return;
  }

  auto projected_inner_extent = getProjectedExtent(merge_or_split->inner());
  Val* projected_combined_extent = projected_inner_extent;

  if (outer_maps) {
    // We don't map the outer dimension through if the inner dimension maps
    // partially as we'd have to support mapping a non-continuous dimension.
    // For example:
    //
    // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
    //
    // With partial mapping of I1 and I3, then there'd be I2 between them
    // so it wouldn't be a continuous segment that we map through this
    // merge. Therefore we'd only consider I3 partialy mapping through this
    // operation.
    //
    // If we have the same merge
    // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
    // However, I2*I3 completely maps, and I1 partially maps, then we can
    // forward a partially mapped domain to the output of size I1*I2*I3

    auto maybe_projected_outer_extent = SimplifyingIrBuilder::whereExpr(
        isFullyProjected(merge_or_split->inner()),
        getProjectedExtent(merge_or_split->outer()),
        merge_or_split->container()->oneVal());
    projected_combined_extent = SimplifyingIrBuilder::mulExpr(
        maybe_projected_outer_extent, projected_inner_extent);
  }

  if constexpr (std::is_same_v<MergeOrSplit, Merge>) {
    addProjectedExtent(merge_or_split->out(), projected_combined_extent);
  } else {
    static_assert(std::is_same_v<MergeOrSplit, Split>);
    addProjectedExtent(merge_or_split->in(), projected_combined_extent);
  }
}

template <typename MergeOrSplit>
void ContiguousInnerDimensionsMapper::distributePE(
    const MergeOrSplit* merge_or_split) {
  // Nothing to do unless recording
  if (!recording_) {
    return;
  }

  auto inner_extent = commonOrConstExtent(ca_map_, merge_or_split->inner());
  auto outer_extent = commonOrConstExtent(ca_map_, merge_or_split->outer());
  Val* projected_combined_extent = nullptr;

  if constexpr (std::is_same_v<MergeOrSplit, Merge>) {
    projected_combined_extent = getProjectedExtent(merge_or_split->out());
  } else {
    static_assert(std::is_same_v<MergeOrSplit, Split>);
    projected_combined_extent = getProjectedExtent(merge_or_split->in());
  }

  // Propagate out mapping to inner as gcd(combined, inner)
  auto projected_inner_extent =
      SimplifyingIrBuilder::gcdExpr(projected_combined_extent, inner_extent);
  addProjectedExtent(merge_or_split->inner(), projected_inner_extent);

  // Propagate out mapping to outer as gcd(combined / inner, outer) if inner is
  // fuly projected
  auto quotient =
      SimplifyingIrBuilder::divExpr(projected_combined_extent, inner_extent);
  auto projected_outer_extent = SimplifyingIrBuilder::whereExpr(
      isFullyProjected(merge_or_split->inner()),
      SimplifyingIrBuilder::gcdExpr(quotient, outer_extent),
      FusionGuard::getCurFusion()->oneVal());
  addProjectedExtent(merge_or_split->outer(), projected_outer_extent);
}

std::vector<IterDomain*> ContiguousInnerDimensionsMapper::projectId(
    const std::vector<IterDomain*>& from,
    const std::vector<IterDomain*>& to) {
  if (from.empty()) {
    return {};
  }

  std::vector<IterDomain*> frontier = from;

  // Process `merge_or_split` and update `frontier`, where `merge_or_split` must
  // be an expr that combines two IDs into one. For forward propagation, it must
  // be a merge, and for backward propagation, it must be a split.
  auto propagateCombine = [&frontier, this](auto* merge_or_split) {
    // Initialize state
    auto find_outer_it = frontier.begin();
    auto outer_pos = frontier.size();
    auto find_inner_it = frontier.begin();
    auto inner_pos = frontier.size();

    // Removes all entries to the left of provided `it`, if `it` is not
    // frontier.begin(). Updates all state of finding outer and inner in the
    // frontier vector after erasing.
    auto clear_left_of = [&find_outer_it,
                          &outer_pos,
                          &find_inner_it,
                          &inner_pos,
                          &frontier,
                          &merge_or_split](decltype(find_outer_it) it) {
      if (it != frontier.begin()) {
        frontier.erase(frontier.begin(), it);
      }

      // Set outer it and position
      find_outer_it =
          std::find(frontier.begin(), frontier.end(), merge_or_split->outer());
      outer_pos = std::distance(frontier.begin(), find_outer_it);

      // Set inner it and position
      find_inner_it =
          std::find(frontier.begin(), frontier.end(), merge_or_split->inner());
      inner_pos = std::distance(frontier.begin(), find_inner_it);
    };

    // Dry run to fill state
    clear_left_of(frontier.begin());

    // Cannot map through non-divisible split
    if constexpr (std::is_same_v<decltype(merge_or_split), Split*>) {
      if (divisible_splits_.find(merge_or_split) == divisible_splits_.end()) {
        if (find_inner_it != frontier.end()) {
          clear_left_of(find_inner_it + 1);
        }
        if (find_outer_it != frontier.end()) {
          clear_left_of(find_outer_it + 1);
        }
        return;
      }
    }

    // Check if the domains out of the split are contiguous in the mapped
    // domain.
    if (find_outer_it == frontier.end() && find_inner_it != frontier.end()) {
      // Outer dimension was not found, but inner dimension was. Must assume
      // everything to the left of inner is not contiguously merged.
      //
      // Clear left of inner
      clear_left_of(find_inner_it);
    } else if (
        find_outer_it != frontier.end() && find_inner_it == frontier.end()) {
      // Inner dimension was not found, outer and anything left of outer are
      // definitely not contiguous.
      //
      // Clear outer and left of outer
      clear_left_of(find_outer_it + 1);
      return;
    } else if (
        find_outer_it == frontier.end() && find_inner_it == frontier.end()) {
      // Nothing mapped, just continue
      return;
    }

    if (find_outer_it != frontier.end() && find_inner_it != frontier.end()) {
      // Both outer and inner mapped.
      if (outer_pos >= inner_pos) {
        // Make sure outer is outside inner, otherwise neither could be part
        // of a continuous mapping. There are cases where we could have
        // reversible operations e.g.:
        //    [id{3} id{5} id{6}] -> merge(1, 0)
        // -> [id{5*3} id{6}] -> split(0, 3)
        // -> [id{5} id{3} id{6}] -> transpose(0, 1)
        // -> [id{3} id{5} id{6}]
        // However we don't try and capture cases like this correcly, we'd
        // just reduce this down to only the iter domain of size 6 mapping.
        //
        // Clear outer and left of outer
        clear_left_of(find_outer_it + 1);
        return;
      }

      // Find the position inner would have to have to be considered ordered
      // relative to outer
      auto pos_after_outer = outer_pos + 1;
      for (; pos_after_outer < frontier.size(); pos_after_outer++) {
        if (frontier[pos_after_outer]->isBroadcast() &&
            pos_after_outer != inner_pos) {
          // Skip broadcast axes as they must not have been concretized in
          // the reference. We remove dimensions that underwent a
          // concretization as well as the dimensions to the left of that in
          // propagateC2P.
          return;
        }
        break;
      }

      if (inner_pos != pos_after_outer) {
        // Nothing to the left of inner could be continuous.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      }
    }

    if constexpr (std::is_same_v<decltype(merge_or_split), Split*>) {
      frontier[inner_pos] = merge_or_split->in();
    } else {
      static_assert(std::is_same_v<decltype(merge_or_split), Merge*>);
      frontier[inner_pos] = merge_or_split->out();
    }
    bool outer_mapped = find_outer_it != frontier.end();
    if (outer_mapped) {
      // Remove outer
      frontier.erase(find_outer_it);
    } else {
      // Clear to the left of inner in since only inner maps
      frontier.erase(frontier.begin(), frontier.begin() + (int64_t)inner_pos);
    }

    combinePE(merge_or_split, outer_mapped);
  };

  // Process `merge_or_split` and update `frontier`, where `merge_or_split` must
  // be an expr that unflatten one ID into two IDs. For forward propagation, it
  // must be a split, and for backward propagation, it must be a merge
  auto propagateDistribute = [&frontier, this](auto* merge_or_split) {
    Val* combined = nullptr;
    if constexpr (std::is_same_v<decltype(merge_or_split), Split*>) {
      combined = merge_or_split->in();
    } else {
      static_assert(std::is_same_v<decltype(merge_or_split), Merge*>);
      combined = merge_or_split->out();
    }
    auto find_out_it = std::find(frontier.begin(), frontier.end(), combined);
    if (find_out_it == frontier.end()) {
      return;
    }

    auto out_pos = std::distance(frontier.begin(), find_out_it);
    frontier[out_pos] = merge_or_split->outer();
    frontier.insert(frontier.begin() + out_pos + 1, merge_or_split->inner());

    distributePE(merge_or_split);
  };

  auto propagateResize = [&frontier, this](Resize* resize_op, bool p2c) {
    IterDomain* id_from = p2c ? resize_op->in() : resize_op->out();
    IterDomain* id_to = p2c ? resize_op->out() : resize_op->in();

    auto it = std::find(frontier.begin(), frontier.end(), id_from);
    if (it == frontier.end()) {
      return;
    }
    auto pos = std::distance(frontier.begin(), it);

    // project resize op to frontier.
    frontier[pos] = id_to;
    // clear left of resize, since those are no long contiguous.
    frontier.erase(frontier.begin(), it);

    if (recording_) {
      // we need to check slice offset at this point.
      auto projected_extent = getProjectedExtent(id_from);

      // projected_extent == 0: return the projected_extent as-is
      // resize_extent == 0   : no resizing, return the projected_extent as-is
      // resize_extent != 0   : slicing/padding, return gcd(projected_extent,
      // abs(resize_extent)) This is a conservative analysis of the offset for
      // data accessing. A better approach needs to consider the actual start
      // pointer address and handle it in alignment analysis in runtime info. We
      // also need to consider multiple resize stacked together and how they
      // could interact with each other. Translating this to code: if
      // (resize_extent == 0 || projected_extent == 0) {
      //   return projected_extent;
      // } else {
      //   gcd(projected_extent, abs(resize_extent));
      // }
      auto comp = [](Val* projected_extent, Val* resize_extent) {
        return SimplifyingIrBuilder::whereExpr(
            SimplifyingIrBuilder::logicalOrExpr(
                SimplifyingIrBuilder::eqExpr(
                    resize_extent, resize_extent->container()->zeroVal()),
                SimplifyingIrBuilder::eqExpr(
                    projected_extent,
                    projected_extent->container()->zeroVal())),
            projected_extent,
            SimplifyingIrBuilder::gcdExpr(
                projected_extent, IrBuilder::absExpr(resize_extent)));
      };
      projected_extent = comp(projected_extent, resize_op->leftExpand());
      projected_extent = comp(projected_extent, resize_op->rightExpand());

      // cap extent by the destination, this is useful when the id_to is resized
      // to zero, where projected_extent shouldn't go beyond the total extent.
      projected_extent =
          SimplifyingIrBuilder::minExpr(projected_extent, id_to->extent());

      addProjectedExtent(id_to, projected_extent);
    }
  };

  // If `from` is [I1, I2, I3, I4], `to` is [I1, I5, I6, I7], where I2 =
  // merge(I5, I6) and I7 = merge(I3, I4), `from` is on both side of `to`. We
  // traverse the forward side and backward side separately. For this example,
  // we will have backward exprs {I2 = merge(I5, I6)} and forward exprs {I7 =
  // merge(I3, I4)}. If from is on the forward side only, then we will have
  // empty backward exprs, vice versa.

  auto backward_exprs = StmtSort::getExprsBetween(
      {to.begin(), to.end()}, {frontier.begin(), frontier.end()});

  // Mapping from logical to root, reverse expressions
  std::reverse(backward_exprs.begin(), backward_exprs.end());

  for (auto* expr : backward_exprs) {
    if (Split* split = dynamic_cast<Split*>(expr)) {
      propagateCombine(split);
    } else if (Merge* merge = dynamic_cast<Merge*>(expr)) {
      propagateDistribute(merge);
    } else if (Resize* resize = dynamic_cast<Resize*>(expr)) {
      propagateResize(resize, false);
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      NVF_THROW(
          "ProjectDimensions does not support expr type: ", expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  if (frontier.empty()) {
    return {};
  }

  auto forward_exprs = StmtSort::getExprsBetween(
      {frontier.begin(), frontier.end()}, {to.begin(), to.end()});

  // Map forward through transforms since we're going from root to logical
  for (auto* expr : forward_exprs) {
    if (Merge* merge = dynamic_cast<Merge*>(expr)) {
      propagateCombine(merge);
    } else if (Split* split = dynamic_cast<Split*>(expr)) {
      propagateDistribute(split);
    } else if (Resize* resize = dynamic_cast<Resize*>(expr)) {
      propagateResize(resize, true);
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      NVF_THROW(
          "ProjectDimensions does not support expr type: ", expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  return frontier;
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoC2P(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  auto from_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                      ->mapped_root_ids_;
  // When we propagate, we should check the resolved broadcast in the order of
  // mapped from_ids.
  //
  // If we have a case where we have a concretized broadcast that's being
  // tracked in a consumer but not concretized in the producer we should break
  // off the dimensions connected to the left of that dimension. So if we have:
  // T0[i0, i2]
  // T1[i0, b1, i2] = broadcast(T0)
  // T2[i0, i1, i2]
  // T3[i0, i1, i2] = T1 + T2
  // and we're propogating from T3 with {i0, i1, i2}
  // When we go from T3 to T0, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T3.
  //
  // Another example is that, if the last broadcast dimension resolved in
  // consumers root domain is mapped for vectorization, the merge order in
  // the vectorization axes matters.
  //
  // T0[i0, i1]
  // T1[i0, i1, b2] = broadcast(T0)
  // T2[i0, i1, i3]
  // T3[i0, i1, i2] = T1 + T2
  //
  // If the mapped ids are {i0, i2, i1}, when propagating from T3 to T1, the
  // resolved broadcast iterdomain is `i2`/`b2`, which would give clear_pos=1.
  // So we'll skip all from_ids with index < clear_pos. see issue:
  // https://github.com/NVIDIA/Fuser/issues/1567#issuecomment-1894605385
  PairwiseLogicalDomainMap logical_map(to, from);
  auto c2p_map = logical_map.mapConsumerToProducer();

  // Id's in consumer to clear from the mapped set due to broadcast
  // concretization.
  std::unordered_set<IterDomain*> consumer_ids_to_clear;
  int64_t clear_pos = 0;
  if (to->hasBroadcast()) {
    // Find the last broadcast dimension resolved in consumers through from_ids
    for (int64_t i = (int64_t)from_ids.size() - 1; i >= 0; i--) {
      auto c_id = from_ids[i];
      auto c_it = c2p_map.find(c_id);
      if (c_it == c2p_map.end()) {
        continue;
      }
      auto p_id = c_it->second;
      if ((!c_id->isBroadcast()) && p_id->isBroadcast()) {
        clear_pos = i + 1;
        break;
      }
    }
  }

  std::vector<IterDomain*> producer_logical_ids;
  for (auto i : arange(clear_pos, (int64_t)from_ids.size())) {
    auto from_id = from_ids[i];
    auto c2p_it = c2p_map.find(from_id);
    if (c2p_it != c2p_map.end() &&
        consumer_ids_to_clear.find(c2p_it->first) ==
            consumer_ids_to_clear.end()) {
      producer_logical_ids.push_back(c2p_it->second);
      if (recording_) {
        addProjectedExtent(c2p_it->second, getProjectedExtent(c2p_it->first));
      }
    }
  }
  return MappedDomain::build(
      projectId(producer_logical_ids, to->getMaybeRootDomain()),
      producer_logical_ids,
      true);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoP2C(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  auto from_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                      ->mapped_logical_ids_;
  // If we have a case where we have a reduction that's being tracked in a
  // producer but not a consumer we should break off the dimensions connected to
  // the left of that reduction unless the producer is a fusion
  // input. So if we have:
  // T0[i0, i1, i2]
  // T1[i0, r1, i2] = sum(T0)
  // T2[i0, i2] = T1
  // and we're propogating from T0 with {i0, i1, i2}
  // When we go from T1 to T2, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T0. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the reduction.
  // So if we have:
  // T0[i0, i1, i2]
  // T1[i1, i0, i2] = transpose(T0)
  // T2[i1, r0, i2] = sum(T1)
  // T3[i1, i2] = T2
  // Then i1 and i2 are contiguous in both T0 and T3, but due to the sum on T1
  // we will have removed i1.
  PairwiseLogicalDomainMap logical_map(from, to);
  auto p2c_map = logical_map.mapProducerToConsumer();
  std::vector<IterDomain*> consumer_root_ids;

  // Id's in producer to clear from the mapped set due to reductions.
  std::unordered_set<IterDomain*> producer_ids_to_clear;
  if (!from->isFusionInput() && from->hasReduction()) {
    // Find the last reduction dimension in the logical domain.
    int64_t clear_pos = -1;
    for (auto i : arange((int64_t)from->getLogicalDomain().size())) {
      if (from->getLogicalDomain()[i]->isReduction()) {
        clear_pos = i;
      }
    }
    // Clear everything to the left of the inner most reduction dimension.
    if (clear_pos >= 0) {
      producer_ids_to_clear.insert(
          from->getLogicalDomain().begin(),
          from->getLogicalDomain().begin() + clear_pos + 1);
    }
  }

  for (auto from_id : from_ids) {
    auto p2c_it = p2c_map.find(from_id);
    if (p2c_it != p2c_map.end() &&
        producer_ids_to_clear.find(p2c_it->first) ==
            producer_ids_to_clear.end()) {
      consumer_root_ids.push_back(p2c_it->second);
      if (recording_) {
        addProjectedExtent(p2c_it->second, getProjectedExtent(p2c_it->first));
      }
    }
  }
  return MappedDomain::build(
      consumer_root_ids,
      projectId(consumer_root_ids, to->getLogicalDomain()),
      false);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoSibling(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  NVF_ERROR(
      from->getMaybeRootDomain().size() == to->getMaybeRootDomain().size(),
      "Siblings of different root sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave root sizes of ",
      from->getMaybeRootDomain().size(),
      " and ",
      to->getMaybeRootDomain().size());

  auto from_root_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                           ->mapped_root_ids_;
  std::vector<IterDomain*> sibling_root_ids;

  for (auto from_root_id : from_root_ids) {
    auto from_it = std::find(
        from->getMaybeRootDomain().begin(),
        from->getMaybeRootDomain().end(),
        from_root_id);
    NVF_ERROR(
        from_it != from->getMaybeRootDomain().end(),
        "Expected ",
        from_root_id->toString(),
        " to be in the root of ",
        from->toString());
    auto pos = std::distance(from->getMaybeRootDomain().begin(), from_it);
    sibling_root_ids.push_back(to->getMaybeRootDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getMaybeRootDomain()[pos],
          getProjectedExtent(from->getMaybeRootDomain()[pos]));
    }
  }

  if (!from->hasRoot()) {
    return MappedDomain::build(
        sibling_root_ids,
        sibling_root_ids,
        false /*shouldn't matter how we initialize this*/);
  }

  NVF_ERROR(
      from->getLogicalDomain().size() == to->getLogicalDomain().size(),
      "Siblings of different logical sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave logical sizes of ",
      from->getLogicalDomain().size(),
      " and ",
      to->getLogicalDomain().size());

  auto from_logical_ids =
      std::dynamic_pointer_cast<const MappedDomain>(from_info)
          ->mapped_logical_ids_;
  std::vector<IterDomain*> sibling_logical_ids;

  for (auto from_logical_id : from_logical_ids) {
    auto from_it = std::find(
        from->getLogicalDomain().begin(),
        from->getLogicalDomain().end(),
        from_logical_id);
    NVF_ERROR(
        from_it != from->getLogicalDomain().end(),
        "Expected ",
        from_logical_id->toString(),
        " to be in the logical of ",
        from->toString());
    auto pos = std::distance(from->getLogicalDomain().begin(), from_it);
    sibling_logical_ids.push_back(to->getLogicalDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getLogicalDomain()[pos],
          getProjectedExtent(from->getLogicalDomain()[pos]));
    }
  }

  return MappedDomain::build(sibling_root_ids, sibling_logical_ids, false);
}

// MaxInfoSpanningTree functions
void ContiguousInnerDimensionsMapper::propagateC2P(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoC2P(from, to, from_info);
  tv_infos_[to] = to_info;
}

void ContiguousInnerDimensionsMapper::propagateP2C(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoP2C(from, to, from_info);
  tv_infos_[to] = to_info;
}

void ContiguousInnerDimensionsMapper::propagateSibling(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoSibling(from, to, from_info);
  tv_infos_[to] = to_info;
}

Val* ContiguousInnerDimensionsMapper::getContigMergeOfInnerSize(
    TensorView* of_tv) {
  FusionGuard fg(of_tv->fusion());
  Val* product_of_inner_extents = of_tv->container()->oneVal();
  auto of_tv_alloc = of_tv->getMaybeAllocationDomain();

  NVF_ERROR(hasMappedDims(of_tv));

  const std::vector<IterDomain*>& projected_dims = mappedLogicalIds(of_tv);
  auto of_tv_alloc_no_reductions = TensorDomain::noReductions(of_tv_alloc);

  auto contiguity = of_tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == of_tv_alloc.size() &&
      contiguity.size() != of_tv_alloc_no_reductions.size()) {
    std::vector<std::optional<bool>> new_contiguity;
    for (auto i : arange(of_tv_alloc.size())) {
      if (!of_tv_alloc[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }

  auto of_tv_alloc_no_reductions_size = of_tv_alloc_no_reductions.size();

  // Filter out 0-dim tensors
  if (of_tv_alloc_no_reductions_size < 1) {
    return product_of_inner_extents;
  }

  NVF_ERROR(
      of_tv_alloc_no_reductions_size == contiguity.size(),
      "Contiguity mismatch found.");

  // Order is important, need to make sure dimensions match up correctly with
  // what was propogated through the mapper. The mapper's dimensions is
  // propogated in the order of the reference, if that order doesn't match the
  // tensor we're mapping too then a transpose interfered with expanded the
  // vectorize dimension.
  size_t projected_dims_i = projected_dims.size();

  for (auto i : arange(of_tv_alloc_no_reductions_size)) {
    if (projected_dims_i == 0) {
      break;
    }
    auto alloc_ii = of_tv_alloc_no_reductions_size - i - 1;
    auto alloc_iid = of_tv_alloc_no_reductions.at(alloc_ii);

    if (alloc_iid->extent()->isOneInt() || alloc_iid->isBroadcast()) {
      if (projected_dims[projected_dims_i - 1] == alloc_iid) {
        --projected_dims_i;
      }
      continue;
    }

    auto contiguity_i = contiguity.at(alloc_ii);
    if (!contiguity_i.has_value()) {
      NVF_THROW("contiguity flag at alloc_ii can't be null");
    } else {
      // Not contiguous
      if (!contiguity_i.value()) {
        break;
      }
    }

    // Mapping order isn't correct, cannot expand vectorization dimension.
    if (projected_dims[--projected_dims_i] != alloc_iid) {
      break;
    }

    product_of_inner_extents = SimplifyingIrBuilder::mulExpr(
        product_of_inner_extents, getProjectedExtent(alloc_iid));
  }
  return simplifyExpr(product_of_inner_extents);
}

std::unordered_map<TensorView*, Val*> ContiguousInnerDimensionsMapper::
    getTvToContigMergeOfInnerSizeMap() {
  std::unordered_map<TensorView*, Val*> result;
  for (auto& [tv, _] : tv_infos_) {
    result[tv] = getContigMergeOfInnerSize(tv);
  }
  return result;
}

namespace {

// Returns Mappings of all dims in reference starting from inner most position
// to outer most position. e.g. T0[i0, r1, b2] will return 3 Mapper instances
// associated with:
// {{i0, r1, b2}, {r1, b2}, {b2}}
// Note that the reference `ref` will be reordered per `logical_reorder_map`
std::vector<std::unordered_map<TensorView*, Val*>> getTvToContigInnerSizeMapsOf(
    TensorView* ref,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map) {
  std::vector<std::unordered_map<TensorView*, Val*>> mappers;
  auto logical_dom = ref->getLogicalDomain();
  if (!logical_reorder_map.empty()) {
    logical_dom = TensorDomain::orderedAs(logical_dom, logical_reorder_map);
  }
  while (!logical_dom.empty()) {
    mappers.push_back(
        ContiguousInnerDimensionsMapper::map(ref, logical_dom)
            .getTvToContigMergeOfInnerSizeMap());
    logical_dom.erase(logical_dom.begin());
  }
  return mappers;
}

// Check if a traversal from vectorized reference IDs may reach the
// IDs of a resize expr without visiting the Resize expr itself. That's
// problematic for the vectorization analysis as the spanning-tree
// based analysis may miss the constraint by the Resize expr.
//
// For this analysis, we start a traversal from the vectorized
// reference IDs to both the input and output of the Resize expr but
// disallow visiting the Resize expr itself. If the traversal is still
// successful, it means there's a path from the reference IDs to the
// resize input and output IDs without visiting the Resize expr.
//
// Permissive BFS is used in this traversal as the vectorized
// reference IDs may not have all the dependencies for the
// traversal. For example, suppose there's a split resshape, and only
// the innermost ID is vectorized. The standard BFS is not able to
// move forward if only the vectorized ID is give as the backward
// split requires both outputs to be presented.
class CanSkipResize : public ValGraphPermissiveBFS {
 public:
  static bool run(
      const ValGraph& graph,
      const ValGroups& ref_groups,
      Resize* resize) {
    ValGroups resize_in_out_groups;
    resize_in_out_groups.pushBack(graph.toGroup(resize->in()));
    resize_in_out_groups.pushBack(graph.toGroup(resize->out()));
    CanSkipResize bfs(graph, ref_groups, resize_in_out_groups, resize);
    bfs.traverse();
    return bfs.allToNodesVisited();
  }

  CanSkipResize(
      const ValGraph& graph,
      const ValGroups& ref_groups,
      const ValGroups& resize_in_out_groups,
      Resize* resize)
      : ValGraphPermissiveBFS(
            graph,
            {ref_groups.begin(), ref_groups.end()},
            {resize_in_out_groups.begin(), resize_in_out_groups.end()},
            /*require_all_to_visited=*/false,
            /*allowed_direction=*/Direction::Undefined),
        resize_(resize) {}

  bool excludeFromTraversal(const NodeType& node) const override {
    const ExprGroup* e = std::get_if<ExprGroup>(&node);
    return e != nullptr && (*e)->has(resize_);
  }

 private:
  Resize* resize_ = nullptr;
};

// This is a WAR for vectorizing through resized iter domains. The
// spanning tree based analysis is not guaranteed to take all resize
// ops into considerations (issue
// https://github.com/NVIDIA/Fuser/issues/3640). To workaround the
// limitation, grab all factors that must be divisible by a
// vectorization factors.
std::unordered_set<Val*> getResizeVectorizationFactors(
    TensorView* reference_tv,
    int64_t break_point) {
  Fusion* fusion = reference_tv->fusion();
  const auto resize_based_ops = scheduler_tools::getResizeBasedOps(fusion);

  if (resize_based_ops.empty()) {
    return {};
  }

  IdModel id_model(fusion);
  const auto& graph = id_model.buildExactGraph();

  std::unordered_set<Val*> resize_factors;

  auto add_resize_factors = [&](Resize* resize) {
    if (!resize->leftExpand()->isZeroInt()) {
      resize_factors.insert(resize->leftExpand());
    }
    if (!resize->rightExpand()->isZeroInt()) {
      resize_factors.insert(resize->rightExpand());
    }
  };

  const ValGroups ref_vec_groups = graph.toGroups(
      std::vector<Val*>{
          reference_tv->getLogicalDomain().begin() + break_point,
          reference_tv->getLogicalDomain().end()});

  // For each of Resize exprs, if it's reachable from the reference
  // vectorized IDs without visiting the Resize expr itself, its
  // constraint may not be reflectd in the inner sizes.
  for (auto resize : resize_based_ops) {
    auto resize_out_tv = resize->output(0)->as<TensorView>();
    for (const auto logical_id : resize_out_tv->getLogicalDomain()) {
      auto resize = dynamic_cast<Resize*>(logical_id->definition());
      if (resize == nullptr) {
        continue;
      }

      if (CanSkipResize::run(graph, ref_vec_groups, resize)) {
        add_resize_factors(resize);
      }
    }
  }

  return resize_factors;
}

} // namespace

int64_t getVectorizationFactor(
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference_tv,
    HeuristicDataCache* data_cache,
    int64_t break_point,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map) {
  FUSER_PERF_SCOPE("vectorize_helper::getVectorizationFactor");

  auto vectorizable_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizableInputsAndOutputs>(
      data_cache, [&reference_tv]() {
        return std::make_unique<std::vector<TensorView*>>(
            scheduler_utils::getInputsOutputsWithInnerDim(
                reference_tv, true, true));
      });

  auto& vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  auto vectorize_maps_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::TvToContigInnerSizeMaps>(
          data_cache, [&reference_tv, &logical_reorder_map]() {
            return std::make_unique<
                std::vector<std::unordered_map<TensorView*, Val*>>>(
                getTvToContigInnerSizeMapsOf(
                    reference_tv, logical_reorder_map));
          });

  if (vectorizable_inputs_outputs.empty()) {
    return 1;
  }

  // break point is beyond the range of vectorize_maps_entry, no vectorization.
  if (break_point >= static_cast<int64_t>(vectorize_maps_entry.get().size())) {
    return 1;
  }

  auto resize_factors_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ResizeVectorizationFactors>(
          data_cache, [&reference_tv, &break_point]() {
            return std::make_unique<std::unordered_set<Val*>>(
                getResizeVectorizationFactors(reference_tv, break_point));
          });

  const auto& resize_factors = resize_factors_entry.get();

  int64_t max_vec_size = SchedulerRuntimeInfo::max_alignment_size_in_bit;
  const auto& tv_to_inner_size_map = vectorize_maps_entry.get().at(break_point);

  for (auto inp_or_out : vectorizable_inputs_outputs) {
    // factor <= max_factor / dtype_size_bit
    const auto dtype_size_bit =
        dataTypeSizeBit(inp_or_out->dtype(), runtime_info.getIndexType());
    max_vec_size = std::min(
        max_vec_size,
        SchedulerRuntimeInfo::max_alignment_size_in_bit / dtype_size_bit);

    // factor <= alignment / dtype_size_bit
    int64_t alignment_size_bit =
        (int64_t)runtime_info.getAlignmentSizeBit(inp_or_out);
    NVF_ERROR(alignment_size_bit % dtype_size_bit == 0);
    max_vec_size = std::min(max_vec_size, alignment_size_bit / dtype_size_bit);

    // factor <= projected_extent
    auto inner_size_it = tv_to_inner_size_map.find(inp_or_out);
    if (inner_size_it == tv_to_inner_size_map.end()) {
      // If we don't have info for a tensor that is supposed to be
      // vectorized, that means the tensor has no projected
      // vectorizable extent, i.e., not vectorizable.
      // TODO: Instead of competely disabling vectorization for all
      // tensors, just disable the problematic tensor and keep the
      // other tensors vectorized
      return 1;
    }
    auto inner_size_opt =
        runtime_info.expressionEvaluator().evaluate(inner_size_it->second);
    NVF_ERROR(
        inner_size_opt.hasValue(),
        "Vectorization heuristic could not evaluate inner most size: ",
        inner_size_it->second);

    max_vec_size = std::min(
        scheduler_utils::maxVectorizationWidth(inner_size_opt.as<int64_t>()),
        max_vec_size);
  }

  // This is a WAR for vectorization through resize as the spanning
  // tree based traversal is not guaranteed to reflect all resize ops
  // that may affect vectorization. This is a safe but conservative
  // analysis since it should only be necessary for innermost IDs.
  for (const auto resize_factor : resize_factors) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(resize_factor);
    if (!inferred_val.hasValue()) {
      return 1;
    }
    auto inferred_val_int = inferred_val.as<int64_t>();
    if (inferred_val_int == 0) {
      continue;
    }
    max_vec_size = std::gcd(max_vec_size, inferred_val_int);
  }

  return max_vec_size;
}

int64_t getVectorizationFactorTransposeGroup(
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference,
    int64_t inner_most_dim,
    const std::vector<int64_t>& dims_to_merge,
    const std::vector<TensorView*>& vec_tv,
    int64_t max_vectorization) {
  max_vectorization = scheduler_utils::maxVectorizationWidth(max_vectorization);
  std::vector<IterDomain*> virtual_innermost_dim;
  // find the virtual_innermost_dim in reference so we can later map
  // that to individual TensorView in vec_tv.
  for (const auto dim : dims_to_merge) {
    virtual_innermost_dim.insert(
        virtual_innermost_dim.begin(), reference->axis(dim));
  }
  virtual_innermost_dim.push_back(reference->axis(inner_most_dim));

  if (reference->axis(inner_most_dim)->isParallelized()) {
    std::ostringstream oss;
    IrTransformPrinter printer(oss);
    printer.printTransforms(reference);
    NVF_THROW(
        "Loop axis ", inner_most_dim, " is expected to be Serial: ", oss.str());
  }

  // NOTE: do I need to consider stride here?! sounds like
  // ContiguousInnerDimensionsMapper::map requires reference to be
  // contiguous, but does it handle stride order?
  auto contig_inner_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          reference, virtual_innermost_dim)
          .getTvToContigMergeOfInnerSizeMap();
  for (auto tv : vec_tv) {
    auto inner_size_it = contig_inner_map.find(tv);
    int64_t tv_vectorize_factor = inner_size_it == contig_inner_map.end()
        ? 1
        : runtime_info.expressionEvaluator()
              .evaluate(inner_size_it->second)
              .as<int64_t>();
    max_vectorization = std::min(
        max_vectorization,
        scheduler_utils::maxVectorizationWidth(tv_vectorize_factor));
  }

  return max_vectorization;
}

int64_t getVectorizationBreakPointOfReductionProducer(
    TensorView* reduction_consumer,
    TensorView* reduction_producer,
    int64_t consumer_innermost_ndims) {
  FUSER_PERF_SCOPE(
      "vectorize_helper::getVectorizationBreakPointOfReductionProducer");

  NVF_ERROR(
      reduction_consumer->definition() != nullptr &&
          ir_utils::isReductionOp(reduction_consumer->definition()) &&
          reduction_consumer->definition()->input(0) == reduction_producer,
      "Invalid reduction consumer and producer. ",
      reduction_consumer->toString(),
      ". ",
      reduction_producer->toString());

  // Find the conrresponding producer break point. To the right of the
  // break point, there must be only the producer innermost IDs or
  // reduction IDs
  int64_t break_point = reduction_producer->nDims();

  // short-cut to to return break point when no c2p mapping is going to be
  // performed
  if (consumer_innermost_ndims == 0) {
    return break_point;
  }

  const auto c2p =
      PairwiseLogicalDomainMap(reduction_producer, reduction_consumer)
          .mapConsumerToProducer();

  // Grab all the corresponding producer IDs that are mapped with the
  // innermost consumer IDs
  std::unordered_set<IterDomain*> producer_innermost_ids;
  for (auto it = reduction_consumer->getMaybeRootDomain().end() -
           consumer_innermost_ndims;
       it != reduction_consumer->getMaybeRootDomain().end();
       ++it) {
    IterDomain* consumer_id = *it;
    auto c2p_it = c2p.find(consumer_id);
    // Since this is for a reduction op, there must be a mapped
    // producer ID
    NVF_ERROR(c2p_it != c2p.end());
    IterDomain* producer_id = c2p_it->second;
    producer_innermost_ids.insert(producer_id);
  }

  int num_detected_producer_innermost_ids = 0;
  for (auto it = reduction_producer->getLogicalDomain().rbegin();
       it != reduction_producer->getLogicalDomain().rend();
       ++it) {
    auto producer_rf_id = *it;

    // If the mapped producer ID is also a reduction domain, the
    // producer should be a fusion input as our
    // reduction/normalization scheduler do not support fusing
    // multiple back-to-back reductions
    if (producer_rf_id->isReduction()) {
      NVF_ERROR(
          reduction_producer->isFusionInput(),
          "Unexpected producer of reduction: ",
          reduction_producer->toString());
      --break_point;
      continue;
    }

    if (producer_innermost_ids.count(producer_rf_id)) {
      --break_point;
      ++num_detected_producer_innermost_ids;
      // If all innermost IDs are found, stop shifting the break point
      // further
      if (num_detected_producer_innermost_ids ==
          (int64_t)producer_innermost_ids.size()) {
        break;
      }
      continue;
    }

    // Neither reduction nor mapped to consumer innermost IDs.
    // This should not happen
    NVF_THROW("Unexpected producer RF ID: ", producer_rf_id->toString())
  }

  return break_point;
}

} // namespace vectorize_helper
} // namespace nvfuser
