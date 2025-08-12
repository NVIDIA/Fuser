// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/schedule.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <type.h>
#include <val_graph.h>

#include <type_traits>
#include <utility>

#ifndef DYNAMIC_TYPE_CHECK
#define DYNAMIC_TYPE_CHECK NVF_ERROR
#endif

#include <dynamic_type/dynamic_type.h>

namespace nvfuser {

// Abstract IterDomain, or AbstractId in short, refers to objects that behave
// like a dimension of tensor. These objects can be IterDomains, ValGroups,
// vector of Abstract IterDomains, etc. See the description of AbstractTensor
// for more detail.
using AbstractId = dynamic_type::DynamicType<
    dynamic_type::Containers<std::vector>,
    IterDomain*,
    ValGroupAndItsGraph>;

inline IterDomain* representativeId(const AbstractId& abs_id) {
  if (abs_id.is<IterDomain*>()) {
    return abs_id.as<IterDomain*>();
  }
  NVF_ERROR(abs_id.is<ValGroupAndItsGraph>());
  return representativeId(abs_id.as<ValGroupAndItsGraph>().group);
}

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
      for (auto i : arange(in.size())) {
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
      for (auto i : arange(lhs.size())) {
        result.emplace_back(AbstractId::dispatch((*this), lhs[i], rhs[i]));
      }
      return result;
    } else if constexpr (std::is_same_v<L, std::vector<AbstractId>>) {
      std::vector<AbstractId> result;
      result.reserve(lhs.size());
      for (auto i : arange(lhs.size())) {
        result.emplace_back(
            AbstractId::dispatch((*this), lhs[i], std::forward<RHS>(rhs)));
      }
      return result;
    } else if constexpr (std::is_same_v<R, std::vector<AbstractId>>) {
      std::vector<AbstractId> result;
      result.reserve(rhs.size());
      for (auto i : arange(rhs.size())) {
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
      for (auto i : arange(lhs.size())) {
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
      for (auto i : arange(lhs.size())) {
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
      for (auto i : arange(rhs.size())) {
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
      NVF_THROW("not supported");
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
      for (auto i : arange(lhs.size())) {
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
      for (auto i : arange(lhs.size())) {
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
      for (auto i : arange(rhs.size())) {
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
struct DispatchParallelize {
  template <typename INPUT>
  void operator()(ParallelType parallel_type, INPUT&& in) const {
    using IN = std::decay_t<INPUT>;
    if constexpr (std::is_same_v<IN, std::monostate>) {
      return;
    } else if constexpr (std::is_same_v<IN, IterDomain*>) {
      return in->parallelize(parallel_type);
    } else if constexpr (std::is_same_v<IN, ValGroupAndItsGraph>) {
      for (auto val : *in.group) {
        auto id = dynamic_cast<IterDomain*>(val);
        NVF_ERROR(id, "Can not parallelize non-IterDomain in ValGroup.");
        id->parallelize(parallel_type);
      }
    } else if constexpr (std::is_same_v<IN, std::vector<AbstractId>>) {
      for (auto& aid : in) {
        AbstractId::dispatch((*this), parallel_type, aid);
      }
    } else {
      NVF_CHECK(false, "Unsupported type in AbstractTensor::parallelize");
    }
  }
};

} // namespace

struct EmptyInfo {
  static EmptyInfo merge(const EmptyInfo&, const EmptyInfo&) {
    return {};
  }

  static std::pair<EmptyInfo, EmptyInfo> split(const EmptyInfo& a) {
    return {{}, {}};
  }

  template <typename SwizzleT>
  static std::pair<EmptyInfo, EmptyInfo> swizzle(
      SwizzleT swizzle_type,
      const EmptyInfo& a,
      const EmptyInfo& b) {
    return {{}, {}};
  }

  bool operator==(const EmptyInfo& t) const {
    return true;
  }
};

// AbstractTensor is similar to TensorView, it has multiple dimensions, where
// each dimension is represented by an Abstract IterDomain. The interface of
// AbstractTensor is also similar to that of TensorViews, that is, it has merge,
// split, etc. However, it only has a single "domain", instead of having
// multiple domains like "logical domain", "loop domain", etc.
//
// AbstractTensor is designed to represent a virtual tensor in a developer's
// mind. AbstractTensor is typically used as follows:
//
// Example 1:
//   IterDomain *id0, *id1;
//   AbstractTensor v({id0, id1});
//   v.merge(0);
// The above code will create a new Merge object whose inputs are (id0, id1),
// and output is a newly created IterDomain, say id01. After the merge, v will
// become [id01].
//
// AbstractTensor can do transformations in batch, like shown in the following
// example:
//
// Example 2:
//   IterDomain *id0, *id1, *id2, *id3;
//   AbstractTensor v({{id0, id1}, {id2, id3}});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id2, and id1 with id3.
// The above code will create two new Merge objects whose inputs are (id0, id2),
// and (id1, id3), and outputs are newly created IterDomains, say id02 and id13.
// After the merge, v will become [{id02, id13}].
//
// AbstractTensor can also do transformations in a "broadcasting" manner, like
// shown in the following two examples:
//
// Example 3:
//   IterDomain *id0, *id1, *id2;
//   AbstractTensor v({id0, {id1, id2}});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id1, and id0 with id2.
// The above code will create two new Merge objects whose inputs are (id0, id1),
// and (id0, id2), and outputs are newly created IterDomains, say id01 and id02.
// After the merge, v will become [{id01, id02}].
//
// Example 4:
//   IterDomain *id0, *id1, *id2;
//   AbstractTensor v({{id0, id1}, id2});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id2, and id1 with id2.
// The above code will create two new Merge objects whose inputs are (id0, id2),
// and (id1, id2), and outputs are newly created IterDomains, say id02 and id12.
// After the merge, v will become [{id02, id12}].
//
// AbstractTensor also works on ValGraphs of IterDomains. For example:
//
// Example 5:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   AbstractTensor v({ValGroupAndItsGraph{g0, &g}, ValGroupAndItsGraph{g1,
//   &g}}); v.merge(0);
// If there is already a merge of g0 and g1 in graph that outputs g01, then v
// will reuse that output ValGroup and becomes [g01]. Otherwise, the above code
// will create a new ExprGroup containing a Merge of g0 and g1, and the
// output ValGroup of this ExprGroup is a newly created ValGroup, say g01. The
// newly created ExprGroups and ValGroups will be added to the ValGraph. The v
// after the merge will be [g01].
//
// Batching and broadcasting as demonstrated in Example 2, 3, 4 works for
// ValGroups as well. Besides, AbstractTensor also supports "type promotion"
// from IterDomain to ValGroup. For example:
//
// Example 6:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   AbstractTensor v({id0, ValGroupAndItsGraph{g1, &g}});
//   v.merge(0);
// This is equivalent to Example 5. You will get [g01].
//
// Example 7:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   AbstractTensor v({ValGroupAndItsGraph{g0, &g}, id1});
//   v.merge(0);
// This is also equivalent to Example 5. You will get [g01].
//
// You can always unzip an AbstractTensor to get a vector of AbstractTensors.
// For example:
//
// Example 8:
//   IterDomain *id0, *id1, *id2, *id3;
//   AbstractTensor v({{id0, id1}, {id2, id3}});
//   auto uz = v.unzip();
// Then uz will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id3}}
//
// Example 9:
//   IterDomain *id0, *id1, *id2;
//   AbstractTensor v({{id0, id1}, id2});
//   auto uz = v.unzip();
// Then uz will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id2}}
//
// The opposite operation of unzip is zip. For example:
//
// Example 10:
//   IterDomain *id0, *id1, *id2, *id3;
//   AbstractTensor v0({id0, id2});
//   AbstractTensor v1({id1, id3});
//   auto z = AbstractTensor::zip({v0, v1});
// Then z will be [{id0, id1}, {id2, id3}].
//
// Besides, you can also add a new "row" into the current
// AbstractTensor. For example:
//
// Example 11:
//   IterDomain *id0, *id1, *id2, *id3, *id4, *id5;
//   AbstractTensor v0({{id0, id1}, {id2, id3}});
//   AbstractTensor v1({id4, id5});
//   v0.addRow(v1);
// In the above example, we can visualize v0 as:
//        dim0   dim1
//   row0  id0    id2
//   row1  id1    id3
// after adding a new row v1, we will get:
//        dim0   dim1
//   row0  id0    id2
//   row1  id1    id3
//   row2  id4    id5
// In another word, v0 will become [{id0, id1, id4}, {id2, id3, id5}].
//
// AbstractId in AbstractTensor can be place holders std::monostate{}. For
// example:
//
// Example 12:
//   IterDomain *id0;
//   AbstractTensor v({{}, {}, id0}); // [null, null, id0]
//   v.split(0, 2); // [null, null, null, id0]
//   v.merge(0); // [null, null, id0]
//   v.swizzle(SwizzleType::XOR, 0, 1); // [null, null, id0]
//   auto vv = v.strip(); // [id0]

template <typename Info>
class AbstractTensorWithInfo {
 public:
  AbstractTensorWithInfo() = default;

  // These constructors use default info
  AbstractTensorWithInfo(std::vector<AbstractId> domain)
      : domain_(std::move(domain)), info_(domain_.size()) {}
  AbstractTensorWithInfo(const std::vector<IterDomain*>& domain)
      : domain_(domain.begin(), domain.end()), info_(domain_.size()) {}
  AbstractTensorWithInfo(std::initializer_list<AbstractId> domain)
      : domain_(domain), info_(domain_.size()) {}

  // These constructors use provided info
  AbstractTensorWithInfo(std::vector<AbstractId> domain, std::vector<Info> info)
      : domain_(std::move(domain)), info_(std::move(info)) {}
  AbstractTensorWithInfo(
      const std::vector<IterDomain*>& domain,
      std::vector<Info> info)
      : domain_(domain.begin(), domain.end()), info_(std::move(info)) {}
  AbstractTensorWithInfo(
      std::initializer_list<AbstractId> domain,
      std::initializer_list<Info> info)
      : domain_(domain), info_(info) {}

  virtual ~AbstractTensorWithInfo() = default;

  const Info& info(int64_t i) const {
    i = wrapDim(i, (int64_t)info_.size());
    return info_.at(i);
  }

  template <typename T>
  std::vector<T> as() const {
    std::vector<T> result;
    std::transform(
        domain_.begin(), domain_.end(), std::back_inserter(result), [](auto x) {
          return (T)x;
        });
    return result;
  }

  std::vector<std::pair<AbstractId, Info>> domainAndInfo() const {
    std::vector<std::pair<AbstractId, Info>> result;
    NVF_ERROR(domain_.size() == info_.size());
    result.reserve(domain_.size());
    for (size_t i : arange(domain_.size())) {
      result.emplace_back(domain_[i], info_[i]);
    }
    return result;
  }

  decltype(auto) operator[](int64_t i) {
    i = wrapDim(i, (int64_t)domain_.size());
    return domain_[i];
  }

  decltype(auto) operator[](int64_t i) const {
    i = wrapDim(i, (int64_t)domain_.size());
    return domain_[i];
  }

  decltype(auto) size() const {
    return domain_.size();
  }

  decltype(auto) empty() const {
    return domain_.empty();
  }

  decltype(auto) begin() {
    return domain_.begin();
  }

  decltype(auto) begin() const {
    return domain_.begin();
  }

  decltype(auto) end() {
    return domain_.end();
  }

  decltype(auto) end() const {
    return domain_.end();
  }

  decltype(auto) rbegin() {
    return domain_.rbegin();
  }

  decltype(auto) rbegin() const {
    return domain_.rbegin();
  }

  decltype(auto) rend() {
    return domain_.rend();
  }

  decltype(auto) rend() const {
    return domain_.rend();
  }

  decltype(auto) cbegin() const {
    return domain_.cbegin();
  }

  decltype(auto) cend() const {
    return domain_.cend();
  }

  decltype(auto) crbegin() const {
    return domain_.crbegin();
  }

  decltype(auto) crend() const {
    return domain_.crend();
  }

  decltype(auto) back() const {
    return domain_.back();
  }

  AbstractTensorWithInfo& pushBack(AbstractId id) {
    domain_.push_back(std::move(id));
    info_.resize(domain_.size());
    return *this;
  }

  AbstractTensorWithInfo& pushBack(AbstractId id, const Info& id_info) {
    domain_.push_back(std::move(id));
    info_.push_back(id_info);
    return *this;
  }

  template <typename... Args>
  AbstractTensorWithInfo& emplaceBack(Args&&... args) {
    domain_.emplace_back(std::forward<Args>(args)...);
    info_.emplace_back();
    return *this;
  }

  bool operator==(const AbstractTensorWithInfo& other) const {
    return domain_ == other.domain_ && info_ == other.info_;
  }

  bool operator==(const std::vector<AbstractId>& domain) const {
    return domain_ == domain;
  }

  template <typename T>
  bool operator!=(T&& t) const {
    return !operator==(std::forward<T>(t));
  }

  AbstractTensorWithInfo& parallelize(
      int64_t axis,
      ParallelType parallel_type) {
    axis = wrapDim(axis, (int64_t)domain_.size());
    AbstractId::dispatch(DispatchParallelize{}, parallel_type, domain_[axis]);
    return *this;
  }

  AbstractTensorWithInfo& split(
      int64_t axis,
      Val* factor,
      bool inner_split = true) {
    NVF_ERROR(domain_.size() == info_.size());

    axis = wrapDim(axis, (int64_t)domain_.size());
    auto [outer, inner] = AbstractId::dispatch(
        DispatchSplit{}, domain_[axis], factor, inner_split);
    std::swap(domain_[axis], inner);
    domain_.insert(domain_.begin() + axis, outer);

    auto [info_outer, info_inner] = Info::split(info_[axis]);
    info_[axis] = std::move(info_outer);
    info_.insert(info_.begin() + axis, std::move(info_inner));

    return *this;
  }

  AbstractTensorWithInfo& split(
      int64_t axis,
      int64_t factor,
      bool inner_split = true) {
    return split(
        axis, IrBuilder::create<Val>(factor, DataType::Index), inner_split);
  }

  AbstractTensorWithInfo& merge(int64_t axis_o, int64_t axis_i) {
    NVF_ERROR(domain_.size() == info_.size());

    axis_o = wrapDim(axis_o, (int64_t)domain_.size());
    axis_i = wrapDim(axis_i, (int64_t)domain_.size());

    auto output =
        AbstractId::dispatch(DispatchMerge{}, domain_[axis_o], domain_[axis_i]);
    // axis_o is the outer input of this merge but does not
    // automatically mean it's an outer domain in this AbstractTensorWithInfo.
    auto domain_outer_pos = axis_o < axis_i ? axis_o : axis_i;
    auto domain_inner_pos = axis_o < axis_i ? axis_i : axis_o;

    domain_.erase(domain_.begin() + domain_inner_pos);
    std::swap(domain_[domain_outer_pos], output);

    info_[domain_outer_pos] =
        Info::merge(info_[domain_outer_pos], info_[domain_inner_pos]);
    info_.erase(info_.begin() + domain_inner_pos);

    return *this;
  }

  AbstractTensorWithInfo& merge(int64_t axis) {
    return merge(axis, axis + 1);
  }

  AbstractTensorWithInfo& reorder(
      const std::unordered_map<int64_t, int64_t>& old2new) {
    NVF_ERROR(domain_.size() == info_.size());

    NVF_ERROR(
        !domain_.empty() || old2new.empty(), "Tried to reorder a 0-dim domain");

    auto new2old = ir_utils::normalizeOld2New(old2new, (int64_t)domain_.size());

    std::vector<AbstractId> reordered_domain;
    std::transform(
        new2old.begin(),
        new2old.end(),
        std::back_inserter(reordered_domain),
        [this](int64_t i) { return domain_[i]; });
    domain_ = std::move(reordered_domain);

    std::vector<Info> reordered_info;
    std::transform(
        new2old.begin(),
        new2old.end(),
        std::back_inserter(reordered_info),
        [this](int64_t i) { return info_[i]; });
    info_ = std::move(reordered_info);

    return *this;
  }
  AbstractTensorWithInfo& reorder(
      const std::initializer_list<std::pair<const int64_t, int64_t>>& old2new) {
    return reorder(std::unordered_map<int64_t, int64_t>(old2new));
  }
  // old2new[index] = permutation[index]
  AbstractTensorWithInfo& reorder(const std::vector<int64_t>& permutation) {
    std::unordered_map<int64_t, int64_t> reorder_map;
    int64_t idx = 0;
    std::transform(
        permutation.begin(),
        permutation.end(),
        std::inserter(reorder_map, reorder_map.end()),
        [&idx](int64_t v) { return std::make_pair(idx++, v); });
    return reorder(reorder_map);
  }
  AbstractTensorWithInfo& reorder(
      const std::initializer_list<int64_t>& permutation) {
    return reorder(std::vector<int64_t>(permutation));
  }

  // Both `from` and `to` are inclusive.

  AbstractTensorWithInfo& flatten(int64_t from = 0, int64_t to = -1) {
    NVF_ERROR(!domain_.empty(), "Tried to do flatten on a 0-dim domains");
    from = wrapDim(from, (int64_t)domain_.size());
    to = wrapDim(to, (int64_t)domain_.size());
    NVF_CHECK(from <= to, "Invalid flatten range. From: ", from, " To: ", to);
    int64_t num_merges = to - from;
    for (auto _ : arange(num_merges)) {
      (void)_;
      merge(from);
    }
    return *this;
  }

  AbstractTensorWithInfo& swizzle(
      SwizzleType swizzle_type,
      int64_t x,
      int64_t y) {
    NVF_ERROR(domain_.size() == info_.size());

    x = wrapDim(x, (int64_t)domain_.size());
    y = wrapDim(y, (int64_t)domain_.size());

    auto [out_x, out_y] = AbstractId::dispatch(
        DispatchSwizzle{}, swizzle_type, domain_[x], domain_[y]);

    std::swap(domain_[x], out_x);
    std::swap(domain_[y], out_y);

    auto [info_outer, info_inner] =
        Info::swizzle(swizzle_type, info_[x], info_[y]);
    info_[x] = std::move(info_outer);
    info_[y] = std::move(info_inner);

    return *this;
  }

  // Temporary helper for legacy swizzle, should be removed eventually.
  // This is a copy-paste of AbstractTensor::swizzle(SwizzleType
  AbstractTensorWithInfo& swizzle(
      Swizzle2DType swizzle_type,
      int64_t x,
      int64_t y) {
    NVF_ERROR(domain_.size() == info_.size());

    x = wrapDim(x, (int64_t)domain_.size());
    y = wrapDim(y, (int64_t)domain_.size());

    auto [out_x, out_y] = AbstractId::dispatch(
        DispatchLegacySwizzle{}, swizzle_type, domain_[x], domain_[y]);

    std::swap(domain_[x], out_x);
    std::swap(domain_[y], out_y);

    auto [info_outer, info_inner] =
        Info::swizzle(swizzle_type, info_[x], info_[y]);
    info_[x] = std::move(info_outer);
    info_[y] = std::move(info_inner);

    return *this;
  }

  // Unzip the AbstractTensor to separate tensors. For example, if this
  // AbstractTensor is [dim0={id0, id1}, dim1={id2, id3}], then the return value
  // will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id3}}.
  std::vector<AbstractTensorWithInfo> unzip() const {
    std::vector<AbstractTensorWithInfo> result;

    // Check and get the size of each vector
    int64_t size = -1;
    for (const auto& aid : domain_) {
      if (!aid.is<std::vector>()) {
        continue;
      }
      int64_t new_size = (int64_t)aid.as<std::vector>().size();
      if (size == -1) {
        size = new_size;
      } else {
        NVF_CHECK(
            size == new_size,
            "Can not unzip an AbstractTensor with different sizes in its "
            "domains.");
      }
    }

    // unzip the AbstractTensor, broadcast the non-vector items. Re-use info for
    // all the unzipped AbstractTensors
    result.resize(size);
    for (auto i : arange(size)) {
      for (const auto& aid : domain_) {
        if (aid.is<std::vector>()) {
          result[i].domain_.emplace_back(aid[i]);
        } else {
          result[i].domain_.emplace_back(aid);
        }
      }
      result[i].info_ = info_;
    }
    return result;
  }

  // Zip multiple AbstractTensors into a single AbstractTensor. For example, if
  // the input is {AbstractTensor{id0, id2}, AbstractTensor{id1, id3}}, then the
  // return value will be [dim0={id0, id1}, dim1={id2, id3}].
  static AbstractTensorWithInfo zip(
      std::vector<AbstractTensorWithInfo> tensors) {
    NVF_CHECK(
        !tensors.empty(), "Can not stack an empty list of AbstractTensor");

    for (const auto& tensor : tensors) {
      NVF_CHECK(
          tensor.size() == tensors[0].size(),
          "Can not stack AbstractTensors with different number of domains.");
    }

    NVF_CHECK(
        std::all_of(
            tensors.begin(),
            tensors.end(),
            [&tensors](auto t) { return t.info_ == tensors[0].info_; }),
        "Cannot zip AbstractTensors whose info does not match");

    AbstractTensorWithInfo result;
    result.info_ = std::move(tensors[0].info_);
    result.domain_.reserve(tensors[0].domain_.size());
    for (auto i : arange(tensors[0].domain_.size())) {
      std::vector<AbstractId> ids;
      ids.reserve(tensors.size());
      for (auto& tensor : tensors) {
        ids.emplace_back(std::move(tensor.domain_[i]));
      }
      result.domain_.emplace_back(std::move(ids));
    }

    return result;
  }

  // Add a new row to the current AbstractTensor. For example, if the current
  // AbstractTensor is [dim0={id0, id1}, dim1={id2, id3}], it is helpful to
  // visualize it as:
  //        dim0   dim1
  //   row0  id0    id2
  //   row1  id1    id3
  // If we add a new row [dim0=id4, dim1=id5], then the current AbstractTensor
  // will become:
  //        dim0   dim1
  //   row0  id0    id2
  //   row1  id1    id3
  //   row2  id4    id5
  // in another word, the return value will be an AbstractTensor:
  // [dim0={id0, id1, id4}, dim1={id2, id3, id5}].
  AbstractTensorWithInfo& addRow(AbstractTensorWithInfo tensor) {
    NVF_CHECK(
        domain_.size() == tensor.domain_.size(),
        "Can not add a new row with different number of domains.");
    NVF_CHECK(
        std::all_of(
            domain_.begin(),
            domain_.end(),
            [](const AbstractId& aid) { return aid.is<std::vector>(); }),
        "Can not add a new row to an AbstractTensor with non-vector domains.");

    NVF_CHECK(
        info_ == tensor.info_, "Cannot add a new row with mismatched info");

    for (auto i : arange(size())) {
      domain_[i].template as<std::vector>().emplace_back(
          std::move(tensor.domain_[i]));
    }
    return *this;
  }

  // Remove all the null elements.
  AbstractTensorWithInfo& strip() {
    AbstractTensorWithInfo result;
    for (auto& [aid, inf] : domainAndInfo()) {
      if (aid.hasValue()) {
        result.pushBack(std::move(aid), std::move(inf));
      }
    }
    std::swap(result, *this);
    return *this;
  }

  AbstractTensorWithInfo<EmptyInfo> dropInfo() const {
    return AbstractTensorWithInfo<EmptyInfo>(domain_);
  }

  void reverse() {
    std::reverse(domain_.begin(), domain_.end());
    std::reverse(info_.begin(), info_.end());
  }

 protected:
  std::vector<AbstractId> domain_;
  std::vector<Info> info_;
};

using AbstractTensor = AbstractTensorWithInfo<EmptyInfo>;

//! This is a holds set of tags of the given type. When we merging or swizzling
//! we take the union of these tag sets. Split duplicates the tag set for each
//! output axis.
template <typename Tag>
struct TagSetInfo {
  std::unordered_set<Tag> tags;

  TagSetInfo(std::unordered_set<Tag> tags_ = {}) : tags(std::move(tags_)) {}

  static TagSetInfo merge(const TagSetInfo& a, const TagSetInfo& b) {
    TagSetInfo merged_tag_info{a.tags};
    merged_tag_info.tags.insert(b.tags.begin(), b.tags.end());
    return merged_tag_info;
  }

  static std::pair<TagSetInfo, TagSetInfo> split(const TagSetInfo& a) {
    return {a, a};
  }

  //! Swizzling mixes the tags so here we re-use merge and duplicate the result
  template <typename SwizzleT>
  static std::pair<TagSetInfo, TagSetInfo> swizzle(
      SwizzleT swizzle_type,
      const TagSetInfo& a,
      const TagSetInfo& b) {
    if (swizzle_type == SwizzleT::NoSwizzle) {
      return {a, b};
    }
    TagSetInfo merged_info = merge(a, b);
    return {merged_info, merged_info};
  }

  bool operator==(const TagSetInfo& t) const {
    return tags == t.tags;
  }
};

//! This is a special case of AbstractTensorWithInfo which propagates a set of
//! tags for each axis. The tags can be any hashable type with equality;
//! typically Tag would be an enum.
template <typename Tag>
class TaggedAbstractTensor : public AbstractTensorWithInfo<TagSetInfo<Tag>> {
 public:
  TaggedAbstractTensor(
      std::vector<AbstractId> domain,
      const std::vector<std::unordered_set<Tag>>& tag_sets)
      : AbstractTensorWithInfo<TagSetInfo<Tag>>(
            domain,
            {tag_sets.begin(), tag_sets.end()}) {}
  TaggedAbstractTensor(
      const std::vector<IterDomain*>& domain,
      const std::vector<std::unordered_set<Tag>>& tag_sets)
      : AbstractTensorWithInfo<TagSetInfo<Tag>>(
            domain,
            {tag_sets.begin(), tag_sets.end()}) {}
  TaggedAbstractTensor(
      std::initializer_list<AbstractId> domain,
      std::initializer_list<std::initializer_list<Tag>> tag_sets)
      : AbstractTensorWithInfo<TagSetInfo<Tag>>(
            domain,
            {tag_sets.begin(), tag_sets.end()}) {}

  const std::unordered_set<Tag>& getTags(int64_t i) const {
    i = wrapDim(i, (int64_t)this->size());
    return this->info_[i].tags;
  }

  std::unordered_set<Tag>& getTags(int64_t i) {
    i = wrapDim(i, (int64_t)this->size());
    return this->info_[i].tags;
  }

  bool hasTag(int64_t i, Tag tag) const {
    return getTags(i).count(tag) == 1;
  }

  //! Return tag if there is a single tag, otherwise nullopt
  //!
  //! This is just a convenience function for the common case where axes with
  //! different tags have not been merged. In these cases there is a single Tag
  //! for each axis and it is cumbersome to extract it manually each time it's
  //! needed.
  std::optional<Tag> getTag(int64_t i) const {
    const auto& tags = getTags(i);
    if (tags.size() == 1) {
      return *tags.begin();
    }
    return std::nullopt;
  }
};

} // namespace nvfuser
