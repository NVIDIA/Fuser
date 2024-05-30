// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <domain_view.h>

#include <utility>

namespace nvfuser {

void IterDomainLikeObjectView::split(
    int64_t axis,
    Val* factor,
    bool inner_split) {
  NVF_ERROR(false, "Not implemented yet");
}

void IterDomainLikeObjectView::split(
    int64_t axis,
    int64_t factor,
    bool inner_split) {
  return split(axis, IrBuilder::create<Val>(factor), inner_split);
}

namespace {

struct DispatchMerge {
  template <typename LHS, typename RHS>
  IDLO operator()(LHS&& lhs, RHS&& rhs) const {
    using L = std::decay_t<LHS>;
    using R = std::decay_t<RHS>;
    if constexpr (
        std::is_same_v<L, std::monostate> ||
        std::is_same_v<R, std::monostate>) {
      NVF_CHECK(false, "Unsupported type in IDLOView::merge");
      return {};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> && std::is_same_v<R, IterDomain*>) {
      return IterDomain::merge(lhs, rhs);
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      NVF_CHECK(
          lhs.graph == rhs.graph,
          "Can not merge ValGroups of different graph.");
      auto graph = lhs.graph;
      // If there is already an existing merge in the ValGraph, just use it.
      auto lhs_uses = graph->getUses(lhs.group);
      for (const ExprGroup& use : lhs_uses) {
        auto input_groups = graph->inputGroups(use);
        NVF_ERROR(input_groups.size() == 2);
        if (input_groups == std::vector<ValGroup>{lhs.group, rhs.group}) {
          auto output_groups = graph->outputGroups(use);
          NVF_ERROR(output_groups.size() == 1);
          return ValGroupAndItsGraph{output_groups[0], graph};
        }
      }
      // There is no such merge, then create one
      auto lhs_id =
          lhs.group->front()->template as<IterDomain>()->cloneWithoutRFactor();
      auto rhs_id =
          rhs.group->front()->template as<IterDomain>()->cloneWithoutRFactor();
      auto output_id = IterDomain::merge(lhs_id, rhs_id);
      graph->initializeVal(lhs_id, {}, {});
      graph->initializeVal(rhs_id, {}, {});
      graph->mapVals(lhs.group->front(), lhs_id);
      graph->mapVals(rhs.group->front(), rhs_id);
      graph->initializeVal(output_id, {}, {});
      graph->registerExpr(output_id->definition());
      return ValGroupAndItsGraph{graph->toGroup(output_id), graph};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      return (*this)(
          ValGroupAndItsGraph{rhs.graph->toGroup(lhs), rhs.graph}, rhs);
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, IterDomain*>) {
      return (*this)(
          lhs, ValGroupAndItsGraph{lhs.graph->toGroup(rhs), lhs.graph});
    } else if constexpr (
        std::is_same_v<L, std::vector<IDLO>> &&
        std::is_same_v<R, std::vector<IDLO>>) {
      NVF_CHECK(
          lhs.size() == rhs.size(),
          "Can not merge vectors of IDLO of different size.");
      std::vector<IDLO> result;
      result.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        result.emplace_back(IDLO::dispatch((*this), lhs[i], rhs[i]));
      }
      return result;
    } else if constexpr (std::is_same_v<L, std::vector<IDLO>>) {
      std::vector<IDLO> result;
      result.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        result.emplace_back(IDLO::dispatch((*this), lhs[i], rhs));
      }
      return result;
    } else if constexpr (std::is_same_v<R, std::vector<IDLO>>) {
      std::vector<IDLO> result;
      result.reserve(rhs.size());
      for (auto i : c10::irange(rhs.size())) {
        result.emplace_back(IDLO::dispatch((*this), lhs, rhs[i]));
      }
      return result;
    } else {
      NVF_CHECK(false, "Unsupported type in IDLOView::merge");
      return {};
    }
  }
};

} // namespace

void IterDomainLikeObjectView::merge(int64_t axis_o, int64_t axis_i) {
  axis_o = wrapDim(axis_o, (int64_t)domain.size());
  axis_i = wrapDim(axis_i, (int64_t)domain.size());

  auto output = IDLO::dispatch(DispatchMerge{}, domain[axis_o], domain[axis_i]);
  // axis_o is the outer input of this merge but does not
  // automatically mean it's an outer domain in this IDLOView.
  auto view_outer_pos = axis_o < axis_i ? axis_o : axis_i;
  auto view_inner_pos = axis_o < axis_i ? axis_i : axis_o;

  domain.erase(domain.begin() + view_inner_pos);
  std::swap(domain[view_outer_pos], output);
}

} // namespace nvfuser
