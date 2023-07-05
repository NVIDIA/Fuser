// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <iostream>
#include <typeinfo>
#include <unordered_set>
#include <vector>

namespace nvfuser {

//! A Union-Find (by rank) data structure on integers
//! The template parameter IndexType dictates the maximum number of elements
//! that can be used.
template <typename IndexType>
class UnionFind {
 public:
  UnionFind(size_t size = 0) {
    enlarge(size);
  }

  //! Create a new partition by merging all overlapping sets in two partitions.
  //! The result is a proper "join" in the language of algebraic lattices: it
  //! is a "supremum" or least upper bound between the two involved partitions
  //! under the "refinement" order where P1 <= P2 iff any set in P1 is a subset
  //! of some subset in P2. Note that the IndexType of the argument can differ
  //! from that of "this", but that the output will always inherit IndexType
  //! from "this".
  //! This function runs in nearly linear time.
  template <typename OtherIndexType>
  UnionFind join(UnionFind<OtherIndexType>& other) {
    TORCH_CHECK(
        size() == other.size(), "Cannot join differently-sized UnionFinds");
    UnionFind<IndexType> output(*this);
    for (IndexType i = 0; i < size(); ++i) {
      // Merging with the root of the entry in "other" is sufficient to perform
      // all intersection unions.
      // We know it's safe to cast to IndexType since we've already checked
      // that the sizes are equal.
      output.merge(i, (IndexType)other.find((OtherIndexType)i));
    }
    return output;
  }

  //! Create a new partition whose sets are subsets of one in this and one in
  //! other.
  //! The result is a proper "meet" in the language of algebraic lattices: it
  //! is an "infimum" or greatest lower bound between the two involved
  //! partitions under the "refinement" order where P1 <= P2 iff any set in P1
  //! is a subset of some subset in P2. Note that the IndexType of the argument
  //! can differ from that of "this", but that the output will always inherit
  //! IndexType from "this".
  //! This function runs in quadratic time.
  template <typename OtherIndexType>
  UnionFind meet(UnionFind<OtherIndexType>& other) {
    TORCH_CHECK(
        size() == other.size(), "Cannot meet differently-sized UnionFinds");
    UnionFind<IndexType> output(size());
    for (IndexType i = 0; i < size(); ++i) {
      for (IndexType j = 0; j < size(); ++j) {
        if (equiv(i, j) && other.equiv(i, j)) {
          output.merge(i, j);
        }
      }
    }
    return output;
  }

  //! Returns true only if other is same size and every set in this partition is
  //! a subset of some set in other. This version does not do path compression.
  template <typename OtherIndexType>
  bool isRefinementOf(const UnionFind<OtherIndexType>& other) const {
    if (size() != other.size()) {
      return false;
    }
    // We do a brute-force search, caching the find() results as much as
    // possible. If we find a pair of elements that are mapped in this but not
    // in other return false
    for (IndexType i = 0; i < size(); ++i) {
      IndexType root_i_this = findConst(i);
      IndexType root_i_other = other.findConst(i);
      for (IndexType j = 0; j < size(); ++j) {
        if (i == j) {
          continue;
        }
        IndexType root_j_this = findConst(j);
        if (root_i_this != root_j_this) {
          continue;
        }
        // i and j are distinct equivalent items in "this"
        IndexType root_j_other = (IndexType)other.findConst((OtherIndexType)j);
        if (root_i_other != root_j_other) {
          return false;
        }
      }
    }
    return true;
  }

  //! Returns true only if other is same size and every set in this partition is
  //! a subset of some set in other.
  template <typename OtherIndexType>
  bool isRefinementOf(UnionFind<OtherIndexType>& other) {
    if (size() != other.size()) {
      return false;
    }
    // We do a brute-force search, caching the find() results as much as
    // possible. If we find a pair of elements that are mapped in this but not
    // in other return false
    for (IndexType i = 0; i < size(); ++i) {
      IndexType root_i_this = find(i);
      IndexType root_i_other = other.find(i);
      for (IndexType j = 0; j < size(); ++j) {
        if (i == j) {
          continue;
        }
        IndexType root_j_this = find(j);
        if (root_i_this != root_j_this) {
          continue;
        }
        // i and j are distinct equivalent items in "this"
        IndexType root_j_other = (IndexType)other.find((OtherIndexType)j);
        if (root_i_other != root_j_other) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename OtherIndexType>
  bool operator<=(UnionFind<OtherIndexType>& other) {
    return isRefinementOf(other);
  }

  template <typename OtherIndexType>
  bool operator<=(const UnionFind<OtherIndexType>& other) const {
    return isRefinementOf(other);
  }

  //! Returns true only if UnionFinds have the same size() and their equivalence
  //! classes are the same. Note that the elements returned by find(a) maybe
  //! different for two equal UnionFind objects. This operator does not perform
  //! path compression on either operand.
  template <typename OtherIndexType>
  bool operator==(const UnionFind<OtherIndexType>& other) const {
    if (size() != (IndexType)other.size()) {
      return false;
    }
    // It suffices to check that for every element, the representative from this
    // and the representative from other are mapped to one another by the
    // _other_ UnionFind.
    for (IndexType i : c10::irange(size())) {
      auto a = findConst(i);
      auto b = other.findConst((OtherIndexType)i);
      // map other's class to ours and our class to other's
      auto findb = findConst(b);
      auto finda = other.findConst((OtherIndexType)a);
      if (a != findb || finda != b) {
        return false;
      }
    }
    return true;
  }

  template <typename OtherIndexType>
  bool operator!=(const UnionFind<OtherIndexType>& other) const {
    return !operator==(other);
  }

  //! Resize the data-structure to equal or larger size than current
  virtual void enlarge(size_t new_size) {
    TORCH_CHECK(new_size >= size(), "Cannot shrink a UnionFind");
    if (new_size == 0) {
      return;
    }
    TORCH_CHECK(
        new_size - 1 <=
            static_cast<size_t>(std::numeric_limits<IndexType>::max()),
        "Tried to enlarge UnionFind to size ",
        new_size,
        " which is greater than this IndexType's capacity of ",
        std::to_string(std::numeric_limits<IndexType>::max() + 1));
    auto old_size = parent_.size();
    parent_.resize(new_size);
    rank_.resize(new_size, 0);
    for (auto i = old_size; i < new_size; ++i) {
      parent_[i] = (IndexType)i;
    }
  }

  //! Return the number of elements in this data structure.
  size_t size() const {
    return parent_.size();
  }

  //! Determine root of element a without doing path compression
  IndexType findConst(IndexType a) const {
    TORCH_CHECK(
        a < size(),
        "Tried to find root of element ",
        a,
        " but total size of UnionFind is ",
        size());
    auto p = a;
    auto root = parent_[p];
    while (p != root) {
      p = root;
      root = parent_[p];
    }
    return root;
  }

  //! Determine root of element a and do path compression
  IndexType find(IndexType a) {
    TORCH_CHECK(
        a < size(),
        "Tried to find root of element ",
        a,
        " but total size of UnionFind is ",
        size());
    // This implementation avoids recursion by doing two passes
    // The equivalent recursive definition is:
    //   auto p = parent_[a];
    //   if (p == a) {
    //     return a;
    //   } else {
    //     // Path compression step. Next call will shortcut directly to root.
    //     return parent_[a] = find(p);
    //   }

    // First find the root without path compression
    auto p = a;
    auto root = parent_[p];
    while (p != root) {
      p = root;
      root = parent_[p];
    }

    // Path compression
    // Loop again to set parents along the path equal to root.
    // On the next call, both loops will not be entered.
    while (a != root) {
      p = parent_[a];
      parent_[a] = root;
      a = p;
    }

    return root;
  }

  //! Test whether two elements are equivalent without path compression
  bool equivConst(IndexType a, IndexType b) const {
    return findConst(a) == findConst(b);
  }

  //! Test whether two elements are equivalent
  bool equiv(IndexType a, IndexType b) {
    return find(a) == find(b);
  }

  //! Merge classes of a and b so that they will share a root.
  //! Returns the new root
  virtual IndexType merge(IndexType a, IndexType b) {
    auto root_a = find(a);
    auto root_b = find(b);
    if (root_a == root_b) {
      return root_a;
    }
    // Rank is a surrogate for "height" of each subtree. It is actually an
    // upper bound on height, since path compression can reduce the height of a
    // subtree without updating rank_. When merging trees, we try to place the
    // "shorter" tree inside the "taller" one since this would not increase the
    // larger tree's height. If they are equal, we point b's root at a's root.
    // Note that in that case the rank (height) of a must be incremented as it
    // is now equal to b's height plus the new step linking it with a.
    auto rank_a = rank_[root_a];
    auto rank_b = rank_[rank_a];
    if (rank_a == rank_b) {
      rank_[root_a]++;
      return parent_[root_b] = root_a;
    } else {
      if (rank_a < rank_b) {
        std::swap(root_a, root_b);
      }
      return parent_[root_b] = root_a;
    }
  }

  //! Resize to zero losing all merge information without altering reserved
  //! capacity
  virtual void clear() {
    parent_.clear();
    rank_.clear();
  }

 private:
  std::vector<IndexType> parent_;
  std::vector<IndexType> rank_;
};

//! This union-find holds some associated data that may optionally
//! be bound to each equivalence class. For example, this can be used to
//! associate a value to a class of symbolic scalars.
template <typename IndexType, typename DataType>
class UnionFindWithData : public UnionFind<IndexType> {
 public:
  //! Resize the data-structure to equal or larger size than current
  void enlarge(size_t new_size) override {
    UnionFind<IndexType>::enlarge(new_size);
    data_.resize(new_size, std::nullopt);
  }

  //! Merge classes of a and b so that they will share a root.
  //! Returns the new root.
  IndexType merge(IndexType a, IndexType b) override {
    auto root_a = find(a);
    auto root_b = find(b);
    if (root_a == root_b) {
      return root_a;
    }
    // If root_a and root_b both have bound values, just check that they match.
    // Otherwise, hold on to whichever value is bound, and ensure that after the
    // merge it is still bound.
    if (data_[root_a].has_value() && data_[root_b].has_value()) {
      TORCH_INTERNAL_ASSERT(
          data_[root_a].value() == data_[root_b].value(),
          "Bound data values do not match in UnionFindWithData for a=",
          a,
          " and b=",
          b);
      UnionFind<IndexType>::merge(root_a, root_b);
    } else {
      auto bound_val =
          data_[root_a].has_value() ? data_[root_a] : data_[root_b];
      UnionFind<IndexType>::merge(root_a, root_b);
      data_[find(root_a)] = bound_val;
    }
  }

  //! Associate data with any element equivalent to a
  void bind(IndexType a, DataType data) {
    if (a >= UnionFind<IndexType>::size()) {
      enlarge(a + 1);
    }
    data_[find(a)] = data;
  }

  //! Resize to zero losing all merge information without altering reserved
  //! capacity
  void clear() override {
    UnionFind<IndexType>::clear();
    data_.clear();
  }

 private:
  std::vector<std::optional<DataType>> data_;
};

} // namespace nvfuser
