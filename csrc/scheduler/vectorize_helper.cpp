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
#include <ir/builder.h>
#include <iter_visitor.h>
#include <scheduler/registry.h>

#include <c10/util/irange.h>

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

Bool* ContiguousInnerDimensionsMapper::isFullyProjected(IterDomain* id) {
  return SimplifyingIrBuilder::eqExpr(
      getProjectedExtent(id), commonOrConstExtent(ca_map_, id));
}

ContiguousInnerDimensionsMapper::ContiguousInnerDimensionsMapper(
    TensorView* reference,
    const std::vector<IterDomain*>& reference_ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits)
    // Send null info to MaxInfoSpanning tree because we need state to compute
    // the info object for reference. It's not really needed on construction,
    // just before traversal.
    : MaxInfoSpanningTree(reference, std::make_shared<MappedDomain>()),
      ca_map_(std::move(ca_map)),
      divisible_splits_(divisible_splits) {
  FusionGuard fg(reference->fusion());
  // Check which domain of tensor view we should be looking at. All IDs must be
  // found in the the rfactor domain.
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          reference_ids.begin(),
          reference_ids.end(),
          [reference](IterDomain* id) {
            return (
                std::find(
                    reference->getMaybeRFactorDomain().begin(),
                    reference->getMaybeRFactorDomain().end(),
                    id) != reference->getMaybeRFactorDomain().end());
          }),
      "\nIterDomains passed in to ContiguousInnerDimensionsMapper passed in to ",
      "ContiguousInnerDimensionsMapper must all exist in the rfactor domain.\n",
      "Reference: ",
      reference->toString());

  // Record while processing reference's information
  recording_ = true;
  std::shared_ptr<Information> reference_information;

  // Ordering of dimensions is important in this analysis, if an ordering is
  // contiguous in the reference, but not the target tensor views, then we
  // cannot consider that a contiguous merge dimension for vectorization.
  std::vector<IterDomain*> reordered_rfactor;
  for (auto id : reference->getMaybeRFactorDomain()) {
    if (std::find(reference_ids.begin(), reference_ids.end(), id) !=
        reference_ids.end()) {
      reordered_rfactor.push_back(id);
      // Initiailze the extent for the mapped iter domain
      addProjectedExtent(id, commonOrConstExtent(ca_map_, id));
    } else if (!id->isBroadcast()) {
      // Ignore broadcasts in the reference. Otherwise, remove non-contiguous
      // IDs in the reference tensor as this is the contiguous mapper.
      reordered_rfactor.clear();
    }
  }

  reference_information = MappedDomain::build(
      projectId(reordered_rfactor, reference->getRootDomain()),
      reordered_rfactor,
      reference->hasRFactor() /*shouldn't matter how we initialize this*/);

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
    const std::vector<IterDomain*>& reference_ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits) {
  return ContiguousInnerDimensionsMapper(
      reference, reference_ids, ca_map, divisible_splits);
}

template <typename MergeOrSplit>
void ContiguousInnerDimensionsMapper::combinePE(
    const MergeOrSplit* merge_or_split,
    bool outer_maps) {
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

  auto clear_left_of = [&frontier](IterDomain* id) {
    auto it = std::find(frontier.begin(), frontier.end(), id);
    if (it != frontier.end()) {
      frontier.erase(frontier.begin(), it + 1);
    }
  };

  // If `from` is [I1, I2, I3, I4], `to` is [I1, I5, I6, I7], where I2 =
  // merge(I5, I6) and I7 = merge(I3, I4), `from` is on both side of `to`. We
  // traverse the forward side and backward side separately. For this example,
  // we will have backward exprs {I2 = merge(I5, I6)} and forward exprs {I7 =
  // merge(I3, I4)}. If from is on the forward side only, then we will have
  // empty backward exprs, vice versa.

  auto backward_exprs = StmtSort::getExprsBetween(
      frontier.front()->fusion(),
      {to.begin(), to.end()},
      {frontier.begin(), frontier.end()});

  // Mapping from rfactor to root, reverse expressions
  std::reverse(backward_exprs.begin(), backward_exprs.end());

  for (auto* expr : backward_exprs) {
    if (Split* split = dynamic_cast<Split*>(expr)) {
      propagateCombine(split);
    } else if (Merge* merge = dynamic_cast<Merge*>(expr)) {
      propagateDistribute(merge);
    } else if (Resize* resize = dynamic_cast<Resize*>(expr)) {
      // Cannot vectorize through resize
      clear_left_of(resize->out());
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  if (frontier.empty()) {
    return {};
  }

  auto forward_exprs = StmtSort::getExprsBetween(
      frontier.front()->fusion(),
      {frontier.begin(), frontier.end()},
      {to.begin(), to.end()});

  // Map forward through transforms since we're going from root to rfactor
  for (auto* expr : forward_exprs) {
    if (Merge* merge = dynamic_cast<Merge*>(expr)) {
      propagateCombine(merge);
    } else if (Split* split = dynamic_cast<Split*>(expr)) {
      propagateDistribute(split);
    } else if (Resize* resize = dynamic_cast<Resize*>(expr)) {
      // Cannot vectorize through resize
      clear_left_of(resize->in());
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
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
  // If we have a case where we have a concretized broadcast that's being
  // tracked in a consumer but not concretized in the producer we should break
  // off the dimensions connected to the left of that dimension. So if we have:
  // T0[i0, i2]
  // T1[i0, b1, i2] = broadcast(T0)
  // T2[i0, i1, i2]
  // T3[i0, i1, i2] = T1 + T2
  // and we're propogating from T3 with {i0, i1, i2}
  // When we go from T3 to T0, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T3. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the concretized broadcast.
  // So if we have:
  // T0[i1, i2]
  // T1[b0, i1, i2] = broadcast(T0)
  // T2[i1, b0, i2] = transpose(T1)
  // T3[i1, i0, i2]
  // T4[i1, i0, i2] = T2 + T3
  // T5[i0, i1, i2] = transpose(T4)
  // Then i1 and i2 are contiguous in both T0 and T5, but due to the realization
  // of the broadcast on T4 we will have removed i1 from the mapped set.
  PairwiseRootDomainMap root_map(to, from);
  auto c2p_map = root_map.mapConsumerToProducer(from->domain(), to->domain());

  // Id's in consumer to clear from the mapped set due to broadcast
  // concretization.
  std::unordered_set<IterDomain*> consumer_ids_to_clear;
  if (to->hasBroadcast()) {
    // Find the last broadcast dimension resolved in consumers root domain
    int clear_pos = -1;
    for (auto i : c10::irange(from->getRootDomain().size())) {
      auto c_id = from->getRootDomain()[i];
      auto c_it = c2p_map.find(c_id);
      if (c_it == c2p_map.end()) {
        continue;
      }
      auto p_id = c_it->second;
      if ((!c_id->isBroadcast()) && p_id->isBroadcast()) {
        clear_pos = (int)i;
      }
    }
    // Clear everything to the left of the inner most resolved broadcast
    // dimension, including the broadcasted domain.
    if (clear_pos >= 0) {
      consumer_ids_to_clear.insert(
          from->getRootDomain().begin(),
          from->getRootDomain().begin() + clear_pos + 1);
    }
  }

  std::vector<IterDomain*> producer_rfactor_ids;
  for (auto from_id : from_ids) {
    auto c2p_it = c2p_map.find(from_id);
    if (c2p_it != c2p_map.end() &&
        consumer_ids_to_clear.find(c2p_it->first) ==
            consumer_ids_to_clear.end()) {
      producer_rfactor_ids.push_back(c2p_it->second);
      if (recording_) {
        addProjectedExtent(c2p_it->second, getProjectedExtent(c2p_it->first));
      }
    }
  }
  return MappedDomain::build(
      projectId(producer_rfactor_ids, to->getRootDomain()),
      producer_rfactor_ids,
      true);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoP2C(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  auto from_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                      ->mapped_rfactor_ids_;
  // If we have a case where we have a reduction that's being tracked in a
  // producer but not a consumer we should break off the dimensions connected to
  // the left of that reduction. So if we have:
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
  PairwiseRootDomainMap root_map(from, to);
  auto p2c_map = root_map.mapProducerToConsumer(from->domain(), to->domain());
  std::vector<IterDomain*> consumer_root_ids;

  // Id's in producer to clear from the mapped set due to reductions.
  std::unordered_set<IterDomain*> producer_ids_to_clear;
  if (from->hasReduction()) {
    // Find the last reduction dimension in the rfactor domain.
    int clear_pos = -1;
    for (auto i : c10::irange(from->getMaybeRFactorDomain().size())) {
      if (from->getMaybeRFactorDomain()[i]->isReduction()) {
        clear_pos = (int)i;
      }
    }
    // Clear everything to the left of the inner most reduction dimension.
    if (clear_pos >= 0) {
      producer_ids_to_clear.insert(
          from->getMaybeRFactorDomain().begin(),
          from->getMaybeRFactorDomain().begin() + clear_pos + 1);
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
      projectId(consumer_root_ids, to->getMaybeRFactorDomain()),
      false);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoSibling(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  TORCH_INTERNAL_ASSERT(
      from->getRootDomain().size() == to->getRootDomain().size(),
      "Siblings of different root sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave root sizes of ",
      from->getRootDomain().size(),
      " and ",
      to->getRootDomain().size());

  auto from_root_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                           ->mapped_root_ids_;
  std::vector<IterDomain*> sibling_root_ids;

  for (auto from_root_id : from_root_ids) {
    auto from_it = std::find(
        from->getRootDomain().begin(),
        from->getRootDomain().end(),
        from_root_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRootDomain().end(),
        "Expected ",
        from_root_id->toString(),
        " to be in the root of ",
        from->toString());
    auto pos = std::distance(from->getRootDomain().begin(), from_it);
    sibling_root_ids.push_back(to->getRootDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getRootDomain()[pos],
          getProjectedExtent(from->getRootDomain()[pos]));
    }
  }

  if (!from->hasRFactor()) {
    return MappedDomain::build(
        sibling_root_ids,
        sibling_root_ids,
        false /*shouldn't matter how we initialize this*/);
  }

  TORCH_INTERNAL_ASSERT(
      from->getRFactorDomain().size() == to->getRFactorDomain().size(),
      "Siblings of different rfactor sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave rfactor sizes of ",
      from->getRFactorDomain().size(),
      " and ",
      to->getRFactorDomain().size());

  auto from_rfactor_ids =
      std::dynamic_pointer_cast<const MappedDomain>(from_info)
          ->mapped_rfactor_ids_;
  std::vector<IterDomain*> sibling_rfactor_ids;

  for (auto from_rfactor_id : from_rfactor_ids) {
    auto from_it = std::find(
        from->getRFactorDomain().begin(),
        from->getRFactorDomain().end(),
        from_rfactor_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRFactorDomain().end(),
        "Expected ",
        from_rfactor_id->toString(),
        " to be in the rfactor of ",
        from->toString());
    auto pos = std::distance(from->getRFactorDomain().begin(), from_it);
    sibling_rfactor_ids.push_back(to->getRFactorDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getRFactorDomain()[pos],
          getProjectedExtent(from->getRFactorDomain()[pos]));
    }
  }

  return MappedDomain::build(sibling_root_ids, sibling_rfactor_ids, false);
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
  Val* product_of_inner_extents = of_tv->container()->oneVal();
  auto of_tv_root = of_tv->getMaybeRFactorDomain();

  TORCH_INTERNAL_ASSERT(hasMappedDims(of_tv));

  const std::vector<IterDomain*>& projected_dims = mappedRFactorIds(of_tv);
  auto of_tv_root_no_reductions = TensorDomain::noReductions(of_tv_root);

  auto contiguity = of_tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == of_tv_root.size() &&
      contiguity.size() != of_tv_root_no_reductions.size()) {
    std::vector<std::optional<bool>> new_contiguity;
    for (auto i : c10::irange(of_tv_root.size())) {
      if (!of_tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }

  auto of_tv_root_no_reductions_size = of_tv_root_no_reductions.size();

  // Filter out 0-dim tensors
  if (of_tv_root_no_reductions_size < 1) {
    return product_of_inner_extents;
  }

  TORCH_INTERNAL_ASSERT(
      of_tv_root_no_reductions_size == contiguity.size(),
      "Contiguity mismatch found.");

  // Order is important, need to make sure dimensions match up correctly with
  // what was propogated through the mapper. The mapper's dimensions is
  // propogated in the order of the reference, if that order doesn't match the
  // tensor we're mapping too then a transpose interfered with expanded the
  // vectorize dimension.
  size_t projected_dims_i = projected_dims.size();

  for (auto i : c10::irange(of_tv_root_no_reductions_size)) {
    if (projected_dims_i == 0) {
      break;
    }
    auto root_i = of_tv_root_no_reductions_size - i - 1;
    auto root_id = of_tv_root_no_reductions.at(root_i);

    if (root_id->extent()->isOneInt() || root_id->isBroadcast()) {
      if (projected_dims[projected_dims_i - 1]->sameAs(root_id)) {
        --projected_dims_i;
      }
      continue;
    }

    auto contiguity_i = contiguity.at(root_i);
    if (!contiguity_i.has_value()) {
      TORCH_INTERNAL_ASSERT(false, "contiguity flag at root_i can't be null");
    } else {
      // Not contiguous
      if (!contiguity_i.value()) {
        break;
      }
    }

    // Mapping order isn't correct, cannot expand vectorization dimension.
    if (!projected_dims[--projected_dims_i]->sameAs(root_id)) {
      break;
    }

    product_of_inner_extents = SimplifyingIrBuilder::mulExpr(
        product_of_inner_extents, getProjectedExtent(root_id));
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
// {{i0, r1, b1}, {r1, b1}, {b1}}
std::vector<std::unordered_map<TensorView*, Val*>> getTvToContigInnerSizeMapsOf(
    TensorView* ref) {
  std::vector<std::unordered_map<TensorView*, Val*>> mappers;
  auto root_dom = ref->getMaybeRFactorDomain();
  while (!root_dom.empty()) {
    mappers.push_back(ContiguousInnerDimensionsMapper::map(ref, root_dom)
                          .getTvToContigMergeOfInnerSizeMap());
    root_dom.erase(root_dom.begin());
  }
  return mappers;
}

} // namespace

size_t getVectorizationFactor(
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference_tv,
    HeuristicSummary* data_cache,
    int break_point) {
  auto vectorizable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::VectorizableInputsAndOutputs>(
          data_cache, [&reference_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reference_tv, true, true));
          });

  auto& vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  auto vectorize_maps_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::TvToContigInnerSizeMaps>(
          data_cache, [&reference_tv]() {
            return std::make_unique<
                std::vector<std::unordered_map<TensorView*, Val*>>>(
                getTvToContigInnerSizeMapsOf(reference_tv));
          });

  if (vectorizable_inputs_outputs.empty()) {
    return 1;
  }

  size_t max_vec_size = SchedulerRuntimeInfo::max_alignment_size_in_byte;
  size_t common_alignment_size =
      SchedulerRuntimeInfo::max_alignment_size_in_byte;

  for (auto inp_or_out : vectorizable_inputs_outputs) {
    auto dtype_size =
        dataTypeSize(inp_or_out->dtype(), runtime_info.getIndexType());

    max_vec_size = std::min(
        max_vec_size,
        SchedulerRuntimeInfo::max_alignment_size_in_byte / dtype_size);
    max_vec_size = std::min(
        max_vec_size, runtime_info.getMaxVectorizableWidth(inp_or_out));
    common_alignment_size = std::min(
        common_alignment_size, runtime_info.getAlignmentSize(inp_or_out));
  }

  auto tv_to_inner_size_map = vectorize_maps_entry.get().at(break_point);
  // Initialize to max the tensors could support.
  size_t max_supported_vector_size = max_vec_size;
  for (auto inp_or_out : vectorizable_inputs_outputs) {
    auto inner_size_it = tv_to_inner_size_map.find(inp_or_out);
    auto inner_size_val = inner_size_it != tv_to_inner_size_map.end()
        ? inner_size_it->second
        : inp_or_out->container()->oneVal();
    auto inner_size_opt =
        runtime_info.expressionEvaluator().evaluate(inner_size_val);
    TORCH_INTERNAL_ASSERT(
        inner_size_opt.has_value(),
        "Vectorization heuristic could not evaluate inner most size.");
    int64_t inner_size = inner_size_opt->as<int64_t>();
    size_t local_max_vec_size = 1;

    while (inner_size > 1 && inner_size % 2 == 0 &&
           local_max_vec_size < max_vec_size) {
      inner_size /= 2;
      local_max_vec_size *= 2;
    }

    max_supported_vector_size =
        std::min(local_max_vec_size, max_supported_vector_size);
  }
  return max_supported_vector_size;
}

} // namespace vectorize_helper
} // namespace nvfuser
