// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <abstract_tensor.h>
#include <id_model/schedule.h>
#include <ir/utils.h>
#include <utility>

namespace nvfuser {

namespace {

struct DispatchSplit {
  template <typename INPUT>
  std::pair<AbstractId, AbstractId> operator()(
      INPUT&& in,
      Val* factor,
      bool inner_split) const {
    using IN = std::decay_t<INPUT>;
    if constexpr (std::is_same_v<IN, std::monostate>) {
      return {std::monostate{}, std::monostate{}};
    } else if constexpr (std::is_same_v<IN, IterDomain*>) {
      return IterDomain::split(std::forward<INPUT>(in), factor, inner_split);
    } else if constexpr (std::is_same_v<IN, ValGroupAndItsGraph>) {
      auto graph = in.graph;
      auto [outer, inner] = split(graph, in.group, factor, inner_split);
      return {
          ValGroupAndItsGraph{outer, graph}, ValGroupAndItsGraph{inner, graph}};
    } else if constexpr (std::is_same_v<IN, std::vector<AbstractId>>) {
      std::vector<AbstractId> outer_result;
      std::vector<AbstractId> inner_result;
      outer_result.reserve(in.size());
      inner_result.reserve(in.size());
      for (auto i : c10::irange(in.size())) {
        auto [outer, inner] =
            AbstractId::dispatch((*this), in[i], factor, inner_split);
        outer_result.emplace_back(outer);
        inner_result.emplace_back(inner);
      }
      return {outer_result, inner_result};
    } else {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::split");
      return {};
    }
  }
};

} // namespace

void AbstractTensor::split(int64_t axis, Val* factor, bool inner_split) {
  axis = wrapDim(axis, (int64_t)domain.size());
  auto [outer, inner] =
      AbstractId::dispatch(DispatchSplit{}, domain[axis], factor, inner_split);
  std::swap(domain[axis], inner);
  domain.insert(domain.begin() + axis, outer);
}

void AbstractTensor::split(int64_t axis, int64_t factor, bool inner_split) {
  return split(
      axis, IrBuilder::create<Val>(factor, DataType::Index), inner_split);
}

namespace {

struct DispatchMerge {
  template <typename LHS, typename RHS>
  AbstractId operator()(LHS&& lhs, RHS&& rhs) const {
    using L = std::decay_t<LHS>;
    using R = std::decay_t<RHS>;
    if constexpr (
        std::is_same_v<L, std::monostate> &&
        std::is_same_v<R, std::monostate>) {
      return std::monostate{};
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
  auto domain_outer_pos = axis_o < axis_i ? axis_o : axis_i;
  auto domain_inner_pos = axis_o < axis_i ? axis_i : axis_o;

  domain.erase(domain.begin() + domain_inner_pos);
  std::swap(domain[domain_outer_pos], output);
}

void AbstractTensor::reorder(
    const std::unordered_map<int64_t, int64_t>& old2new) {
  NVF_ERROR(
      !domain.empty() || old2new.empty(), "Tried to reorder a 0-dim domain");

  auto new2old = ir_utils::normalizeOld2New(old2new, (int64_t)domain.size());

  std::vector<AbstractId> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [this](int64_t i) { return domain[i]; });

  domain = std::move(reordered_domain);
}

// old2new[index] = permutation[index]
void AbstractTensor::reorder(const std::vector<int64_t>& permutation) {
  std::unordered_map<int64_t, int64_t> reorder_map;
  int64_t idx = 0;
  std::transform(
      permutation.begin(),
      permutation.end(),
      std::inserter(reorder_map, reorder_map.end()),
      [&idx](int64_t v) { return std::make_pair(idx++, v); });
  return reorder(reorder_map);
}

void AbstractTensor::flatten(int64_t from, int64_t to) {
  NVF_ERROR(!domain.empty(), "Tried to do flatten on a 0-dim domains");
  from = wrapDim(from, (int64_t)domain.size());
  to = wrapDim(to, (int64_t)domain.size());
  NVF_CHECK(from <= to, "Invalid flatten range. From: ", from, " To: ", to);
  int64_t num_merges = to - from;
  for (auto _ : c10::irange(num_merges)) {
    (void)_;
    merge(from);
  }
}

namespace {

struct DispatchSwizzle {
  template <typename LHS, typename RHS>
  std::pair<AbstractId, AbstractId> operator()(
      SwizzleType swizzle_type,
      LHS&& lhs,
      RHS&& rhs) const {
    using L = std::decay_t<LHS>;
    using R = std::decay_t<RHS>;
    if constexpr (
        std::is_same_v<L, std::monostate> &&
        std::is_same_v<R, std::monostate>) {
      return {std::monostate{}, std::monostate{}};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> && std::is_same_v<R, IterDomain*>) {
      auto [out_x, out_y] = IterDomain::swizzle(
          swizzle_type, std::forward<LHS>(lhs), std::forward<RHS>(rhs));
      return {out_x, out_y};
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      NVF_CHECK(
          lhs.graph == rhs.graph,
          "Can not merge ValGroups of different graph.");
      auto graph = lhs.graph;
      auto [out_x, out_y] = swizzle(graph, swizzle_type, lhs.group, rhs.group);
      return {
          ValGroupAndItsGraph{out_x, graph}, ValGroupAndItsGraph{out_y, graph}};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      return (*this)(
          swizzle_type,
          ValGroupAndItsGraph{rhs.graph->toGroup(lhs), rhs.graph},
          std::forward<RHS>(rhs));
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, IterDomain*>) {
      return (*this)(
          swizzle_type,
          std::forward<LHS>(lhs),
          ValGroupAndItsGraph{lhs.graph->toGroup(rhs), lhs.graph});
    } else if constexpr (
        std::is_same_v<L, std::vector<AbstractId>> &&
        std::is_same_v<R, std::vector<AbstractId>>) {
      NVF_CHECK(
          lhs.size() == rhs.size(),
          "Can not merge vectors of AbstractId of different size.");
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(lhs.size());
      result_y.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        auto [out_x, out_y] =
            AbstractId::dispatch((*this), swizzle_type, lhs[i], rhs[i]);
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else if constexpr (std::is_same_v<L, std::vector<AbstractId>>) {
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(lhs.size());
      result_y.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        auto [out_x, out_y] = AbstractId::dispatch(
            (*this), swizzle_type, lhs[i], std::forward<RHS>(rhs));
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else if constexpr (std::is_same_v<R, std::vector<AbstractId>>) {
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(rhs.size());
      result_y.reserve(rhs.size());
      for (auto i : c10::irange(rhs.size())) {
        auto [out_x, out_y] = AbstractId::dispatch(
            (*this), swizzle_type, std::forward<LHS>(lhs), rhs[i]);
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::merge");
      return {};
    }
  }
};

// Copy-paste of DispatchSwizzle with s/SwizzleType/Swizzle2DType/g
// This is a temporary helper and should be removed eventually.
struct DispatchLegacySwizzle {
  template <typename LHS, typename RHS>
  std::pair<AbstractId, AbstractId> operator()(
      Swizzle2DType swizzle_type,
      LHS&& lhs,
      RHS&& rhs) const {
    using L = std::decay_t<LHS>;
    using R = std::decay_t<RHS>;
    if constexpr (
        std::is_same_v<L, std::monostate> &&
        std::is_same_v<R, std::monostate>) {
      return {std::monostate{}, std::monostate{}};
    } else if constexpr (
        std::is_same_v<L, IterDomain*> && std::is_same_v<R, IterDomain*>) {
      auto [out_x, out_y] = IterDomain::swizzle(
          swizzle_type, std::forward<LHS>(lhs), std::forward<RHS>(rhs));
      return {out_x, out_y};
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      NVF_ERROR(false, "not supported");
    } else if constexpr (
        std::is_same_v<L, IterDomain*> &&
        std::is_same_v<R, ValGroupAndItsGraph>) {
      return (*this)(
          swizzle_type,
          ValGroupAndItsGraph{rhs.graph->toGroup(lhs), rhs.graph},
          std::forward<RHS>(rhs));
    } else if constexpr (
        std::is_same_v<L, ValGroupAndItsGraph> &&
        std::is_same_v<R, IterDomain*>) {
      return (*this)(
          swizzle_type,
          std::forward<LHS>(lhs),
          ValGroupAndItsGraph{lhs.graph->toGroup(rhs), lhs.graph});
    } else if constexpr (
        std::is_same_v<L, std::vector<AbstractId>> &&
        std::is_same_v<R, std::vector<AbstractId>>) {
      NVF_CHECK(
          lhs.size() == rhs.size(),
          "Can not merge vectors of AbstractId of different size.");
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(lhs.size());
      result_y.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        auto [out_x, out_y] =
            AbstractId::dispatch((*this), swizzle_type, lhs[i], rhs[i]);
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else if constexpr (std::is_same_v<L, std::vector<AbstractId>>) {
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(lhs.size());
      result_y.reserve(lhs.size());
      for (auto i : c10::irange(lhs.size())) {
        auto [out_x, out_y] = AbstractId::dispatch(
            (*this), swizzle_type, lhs[i], std::forward<RHS>(rhs));
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else if constexpr (std::is_same_v<R, std::vector<AbstractId>>) {
      std::vector<AbstractId> result_x;
      std::vector<AbstractId> result_y;
      result_x.reserve(rhs.size());
      result_y.reserve(rhs.size());
      for (auto i : c10::irange(rhs.size())) {
        auto [out_x, out_y] = AbstractId::dispatch(
            (*this), swizzle_type, std::forward<LHS>(lhs), rhs[i]);
        result_x.emplace_back(out_x);
        result_y.emplace_back(out_y);
      }
      return {result_x, result_y};
    } else {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::merge");
      return {};
    }
  }
};

} // namespace

void AbstractTensor::swizzle(SwizzleType swizzle_type, int64_t x, int64_t y) {
  x = wrapDim(x, (int64_t)domain.size());
  y = wrapDim(y, (int64_t)domain.size());

  auto [out_x, out_y] = AbstractId::dispatch(
      DispatchSwizzle{}, swizzle_type, domain[x], domain[y]);

  std::swap(domain[x], out_x);
  std::swap(domain[y], out_y);
}

// Temporary helper for legacy swizzle, should be removed eventually.
// This is a copy-paste of AbstractTensor::swizzle(SwizzleType
void AbstractTensor::swizzle(Swizzle2DType swizzle_type, int64_t x, int64_t y) {
  x = wrapDim(x, (int64_t)domain.size());
  y = wrapDim(y, (int64_t)domain.size());

  auto [out_x, out_y] = AbstractId::dispatch(
      DispatchLegacySwizzle{}, swizzle_type, domain[x], domain[y]);

  std::swap(domain[x], out_x);
  std::swap(domain[y], out_y);
}

std::vector<AbstractTensor> AbstractTensor::unzip() const {
  std::vector<AbstractTensor> result;

  // Check and get the size of each vector
  int64_t size = -1;
  for (const auto& aid : domain) {
    if (!aid.is<std::vector>()) {
      continue;
    }
    int64_t new_size = (int64_t)aid.as<std::vector>().size();
    if (size == -1) {
      size = new_size;
    } else {
      NVF_CHECK(
          size == new_size,
          "Can not unzip an AbstractTensor with different sizes in its domains.");
    }
  }

  // unzip the AbstractTensor, broadcast the non-vector items
  result.resize(size);
  for (const auto& aid : domain) {
    for (auto i : c10::irange(size)) {
      if (!aid.is<std::vector>()) {
        result[i].domain.emplace_back(aid);
      } else {
        result[i].domain.emplace_back(aid[i]);
      }
    }
  }
  return result;
}

} // namespace nvfuser
