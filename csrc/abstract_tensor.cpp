// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <abstract_tensor.h>
#include <id_model/utils.h>
#include <utility>

namespace nvfuser {

void AbstractTensor::split(int64_t axis, Val* factor, bool inner_split) {
  NVF_ERROR(false, "Not implemented yet");
}

void AbstractTensor::split(int64_t axis, int64_t factor, bool inner_split) {
  return split(axis, IrBuilder::create<Val>(factor), inner_split);
}

namespace {

struct DispatchMerge {
  template <typename LHS, typename RHS>
  AbstractId operator()(LHS&& lhs, RHS&& rhs) const {
    using L = std::decay_t<LHS>;
    using R = std::decay_t<RHS>;
    if constexpr (
        std::is_same_v<L, std::monostate> ||
        std::is_same_v<R, std::monostate>) {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::merge");
      return {};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> && std::is_same_v<R, IterDomain*>) {
      return IterDomain::merge(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      NVF_CHECK(
          lhs.graph == rhs.graph,
          "Can not merge ValGroups of different graph.");
      auto graph = lhs.graph;
      return ValGroupAndItsGraph{merge(graph, lhs.group, rhs.group), graph};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      return (*this)(
          ValGroupAndItsGraph{rhs.graph->toGroup(lhs), rhs.graph},
          std::forward<RHS>(rhs));
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, IterDomain*>) {
      return (*this)(
          std::forward<LHS>(lhs),
          ValGroupAndItsGraph{lhs.graph->toGroup(rhs), lhs.graph});
    } else if constexpr (
        std::is_same_v<L, std::vector<AbstractId>> &&
        std::is_same_v<R, std::vector<AbstractId>>) {
      NVF_CHECK(
          lhs.size() == rhs.size(),
          "Can not merge vectors of AbstractId of different size.");
      std::vector<AbstractId> result;
      result.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        result.emplace_back(AbstractId::dispatch((*this), lhs[i], rhs[i]));
      }
      return result;
    } else if constexpr (std::is_same_v<L, std::vector<AbstractId>>) {
      std::vector<AbstractId> result;
      result.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        result.emplace_back(
            AbstractId::dispatch((*this), lhs[i], std::forward<RHS>(rhs)));
      }
      return result;
    } else if constexpr (std::is_same_v<R, std::vector<AbstractId>>) {
      std::vector<AbstractId> result;
      result.reserve(rhs.size());
      for (auto i : c10::irange(rhs.size())) {
        result.emplace_back(
            AbstractId::dispatch((*this), std::forward<LHS>(lhs), rhs[i]));
      }
      return result;
    } else {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::merge");
      return {};
    }
  }
};

} // namespace

void AbstractTensor::merge(int64_t axis_o, int64_t axis_i) {
  axis_o = wrapDim(axis_o, (int64_t)domain.size());
  axis_i = wrapDim(axis_i, (int64_t)domain.size());

  auto output =
      AbstractId::dispatch(DispatchMerge{}, domain[axis_o], domain[axis_i]);
  // axis_o is the outer input of this merge but does not
  // automatically mean it's an outer domain in this AbstractTensor.
  auto view_outer_pos = axis_o < axis_i ? axis_o : axis_i;
  auto view_inner_pos = axis_o < axis_i ? axis_i : axis_o;

  domain.erase(domain.begin() + view_inner_pos);
  std::swap(domain[view_outer_pos], output);
}

} // namespace nvfuser
