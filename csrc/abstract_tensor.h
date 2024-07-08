// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <val_graph.h>

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

// AbstractTensor is similar to TensorView, it has multiple dimensions, where
// each dimension is represented by an Abstract IterDomain. The interface of
// AbstractTensor is also similar to that of TesorViews, that is, it has merge,
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
//   auto ub = v.unzip();
// Then ub will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id3}}
//
// Example 9:
//   IterDomain *id0, *id1, *id2;
//   AbstractTensor v({{id0, id1}, id2});
//   auto ub = v.unzip();
// Then ub will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id2}}

struct AbstractTensor {
  std::vector<AbstractId> domain;

  AbstractTensor() = default;
  AbstractTensor(std::vector<AbstractId> domain) : domain(std::move(domain)) {}
  AbstractTensor(std::vector<IterDomain*> domain)
      : domain(domain.begin(), domain.end()) {}
  AbstractTensor(std::initializer_list<AbstractId> domain) : domain(domain) {}

  virtual ~AbstractTensor() = default;

  template <typename T>
  std::vector<T> as() const {
    std::vector<T> result;
    std::transform(
        domain.begin(), domain.end(), std::back_inserter(result), [](auto x) {
          return (T)x;
        });
    return result;
  }

  decltype(auto) operator[](int64_t i) {
    i = wrapDim(i, (int64_t)domain.size());
    return domain[i];
  }

  decltype(auto) operator[](int64_t i) const {
    i = wrapDim(i, (int64_t)domain.size());
    return domain[i];
  }

  decltype(auto) size() const {
    return domain.size();
  }

  template <typename T>
  bool operator==(T&& t) const {
    if constexpr (std::is_same_v<AbstractTensor, std::decay_t<T>>) {
      return domain == t.domain;
    } else {
      return domain == std::forward<T>(t);
    }
  }

  template <typename T>
  bool operator!=(T&& t) const {
    return !operator==(std::forward<T>(t));
  }

  virtual void split(int64_t axis, Val* factor, bool inner_split = true);
  virtual void split(int64_t axis, int64_t factor, bool inner_split = true);

  virtual void merge(int64_t axis_o, int64_t axis_i);
  void merge(int64_t axis) {
    merge(axis, axis + 1);
  }

  virtual void reorder(const std::unordered_map<int64_t, int64_t>& old2new);
  void reorder(
      const std::initializer_list<std::pair<const int64_t, int64_t>>& old2new) {
    return reorder(std::unordered_map<int64_t, int64_t>(old2new));
  }
  // old2new[index] = permutation[index]
  virtual void reorder(const std::vector<int64_t>& permutation);
  void reorder(const std::initializer_list<int64_t>& permutation) {
    reorder(std::vector<int64_t>(permutation));
  }

  // Both `from` and `to` are inclusive.
  virtual void flatten(int64_t from = 0, int64_t to = -1);

  virtual void swizzle(SwizzleType swizzle_type, int64_t x, int64_t y);

  // Temporary helper for legacy swizzle, should be removed eventually.
  // This is a copy-paste of AbstractTensor::swizzle(SwizzleType
  virtual void swizzle(Swizzle2DType swizzle_type, int64_t x, int64_t y);

  // Unzip the AbstractTensor to separate tensors. For example, if this
  // AbstractTensor is [dim0={id0, id1}, dim1={id2, id3}], then the return value
  // will be {AbstractTensor{id0, id2}, AbstractTensor{id1, id3}}.
  std::vector<AbstractTensor> unzip() const;
};

//! This is a wrapper around AbstractTensor which propagates a set of tags for
//! each axis. The tags can be any hashable type with equality; typically Tag
//! would be an enum. See AbstractMatmulTensor, in which case Tag =
//! MatmulDimRole.
template <typename Tag>
struct TaggedAbstractTensor : AbstractTensor {
  //! Holds a set of tags for each dimension
  std::vector<std::unordered_set<Tag>> tags;

  const std::unordered_set<Tag>& getTags(int64_t i) const {
    i = wrapDim(i, (int64_t)domain.size());
    return tags[i];
  }

  bool hasTag(int64_t i, Tag tag) const {
    return (bool)getTags(i).count(tag);
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

  template <typename T>
  bool operator==(T&& t) const {
    if constexpr (std::is_same_v<TaggedAbstractTensor<Tag>, std::decay_t<T>>) {
      return domain == t.domain && tags == t.tags;
    } else {
      // Only compare true for other TaggedAbstractTensors
      return false;
    }
  }

  void split(int64_t axis, Val* factor, bool inner_split = true) override {
    AbstractTensor::split(axis, factor, inner_split);
    axis = wrapDim(axis, (int64_t)domain.size());
    // copy tags from original axis
    tags.insert(tags.begin() + axis, tags[axis]);
  }
  void split(int64_t axis, int64_t factor, bool inner_split = true) override {
    AbstractTensor::split(axis, factor, inner_split);
    axis = wrapDim(axis, (int64_t)domain.size());
    // copy tags from original axis
    tags.insert(tags.begin() + axis, tags[axis]);
  }

  using AbstractTensor::merge;
  void merge(int64_t axis_o, int64_t axis_i) override {
    AbstractTensor::merge(axis_o, axis_i);
    axis_o = wrapDim(axis_o, (int64_t)domain.size());
    axis_i = wrapDim(axis_i, (int64_t)domain.size());
    // merge tags from these axes into outer position
    if (axis_o >= axis_i) {
      std::swap(axis_o, axis_i);
    }
    tags[axis_o].insert(tags[axis_i].begin(), tags[axis_o].end());
    tags.erase(tags.begin() + axis_i);
  }

  void reorder(const std::unordered_map<int64_t, int64_t>& old2new) override {
    AbstractTensor::reorder(old2new);
    auto new2old = ir_utils::normalizeOld2New(old2new, (int64_t)domain.size());
    std::vector<std::unordered_set<Tag>> reordered_tags;
    std::transform(
        new2old.begin(),
        new2old.end(),
        std::back_inserter(reordered_tags),
        [this](int64_t i) { return tags[i]; });
    tags = std::move(reordered_tags);
  }

  // Both `from` and `to` are inclusive.
  void flatten(int64_t from = 0, int64_t to = -1) override {
    from = wrapDim(from, (int64_t)domain.size());
    to = wrapDim(to, (int64_t)domain.size());
    for (size_t i : c10::irange(from, to)) {
      tags[from].insert(tags[i + 1].begin(), tags[i + 1].end());
    }
    tags.erase(tags.begin() + from + 1, tags.begin() + to + 1);
  }

  // swizzle mixes axes, so the tag sets for both x and y become the union of
  // the input tag sets
  void swizzle(SwizzleType swizzle_type, int64_t x, int64_t y) override {
    AbstractTensor::swizzle(swizzle_type, x, y);
    x = wrapDim(x, (int64_t)domain.size());
    y = wrapDim(y, (int64_t)domain.size());
    tags[x].insert(tags[y].begin(), tags[y].end());
  }

  // Temporary helper for legacy swizzle, should be removed eventually.
  // This is a copy-paste of AbstractTensor::swizzle(SwizzleType
  void swizzle(Swizzle2DType swizzle_type, int64_t x, int64_t y) override {
    AbstractTensor::swizzle(swizzle_type, x, y);
    x = wrapDim(x, (int64_t)domain.size());
    y = wrapDim(y, (int64_t)domain.size());
    tags[x].insert(tags[y].begin(), tags[y].end());
  }
};

} // namespace nvfuser
