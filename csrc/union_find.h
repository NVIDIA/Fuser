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
      auto a = find(i);
      auto b = other.find((OtherIndexType)i);
      // map other's class to ours and our class to other's
      auto findb = find(b);
      auto finda = other.find((OtherIndexType)a);
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
  void enlarge(size_t new_size) {
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
  IndexType find(IndexType a) const {
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
    // This implementation avoids recursion by doing two passes
    // The equivalent recursive definition is:
    //   auto p = parent_[a];
    //   if (p == a) {
    //     return a;
    //   } else {
    //     // Path compression step. Next call will shortcut directly to root.
    //     return parent_[a] = find(p);
    //   }

    // Get root using const find() which does not do path compression
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const auto root = const_cast<const UnionFind<IndexType>*>(this)->find(a);

    // Path compression
    // Loop again to set parents along the path equal to root.
    // On the next call, both loops will not be entered.
    auto p = a;
    while (a != root) {
      p = parent_[a];
      parent_[a] = root;
      a = p;
    }

    return root;
  }

  //! Test whether two elements are equivalent without path compression
  bool equiv(IndexType a, IndexType b) const {
    return find(a) == find(b);
  }

  //! Test whether two elements are equivalent
  bool equiv(IndexType a, IndexType b) {
    return find(a) == find(b);
  }

  //! Merge classes of a and b so that they will share a root.
  //! Returns the new root
  IndexType merge(IndexType a, IndexType b) {
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

  //! Set a as the root of its equivalence class, so that after this call
  //! find(a) == a.
  void setAsRoot(IndexType a) {
    const auto orig_root = find(a);
    if (orig_root == a) {
      return;
    }
    parent_[orig_root] = a;
    parent_[a] = a;
    // Update rank of new root to try and remain balanced
    rank_[a] = rank_[orig_root] + 1;
  }

  //! Compute a sorted vector of all elements equivalent to a.
  std::vector<IndexType> computeEquivalenceClass(IndexType a) {
    std::vector<IndexType> c;
    const auto root_a = find(a);

    for (const auto i : c10::irange(size())) {
      const auto root_i = find(i);
      if (root_i == root_a) {
        c.push_back(i);
      }
    }

    return c;
  }

  //! Const version of computeEquivalenceClass
  std::vector<IndexType> computeEquivalenceClass(IndexType a) const {
    std::vector<IndexType> c;
    const auto root_a = find(a);

    for (const auto i : c10::irange(size())) {
      const auto root_i = find(i);
      if (root_i == root_a) {
        c.push_back(i);
      }
    }

    return c;
  }

  //! Computes all equivalence classes as sorted vectors of ints. The classes
  //! are sorted by their lowest members.
  std::vector<std::vector<IndexType>> computeEquivalenceClasses() {
    std::vector<std::vector<IndexType>> classes;
    // First pass initializes a vector for each equivalence class
    for (const auto i : c10::irange(size())) {
      const auto root = find(i);
      // Only process each class once, when passing its root
      if (root == i) {
        // Create new empty vector for this class to be filled on second pass
        classes.emplace_back(0);
      }
    }
    // root_to_class_num maps to position of a root element to index of class.
    // This is initialized to classes.size() to indicate that the class has not
    // yet been assigned a position. Those positions are assigned as next_class
    // whenever we first encounter a member of the class.
    std::vector<IndexType> root_to_class_num(size(), classes.size());
    IndexType next_class = 0;
    // Second pass inserts into class vectors in order
    for (const auto i : c10::irange(size())) {
      const auto root = find(i);
      auto class_num = root_to_class_num.at(root);
      if (class_num == classes.size()) {
        // First element in this class
        root_to_class_num.at(root) = next_class;
        class_num = next_class++;
      }
      classes.at(class_num).push_back(i);
    }
    return classes;
  }

  //! Const version of computeEquivalenceClasses
  std::vector<std::vector<IndexType>> computeEquivalenceClasses() const {
    std::vector<std::vector<IndexType>> classes;
    // First pass initializes a vector for each equivalence class
    for (const auto i : c10::irange(size())) {
      const auto root = find(i);
      // Only process each class once, when passing its root
      if (root == i) {
        // Create new empty vector for this class to be filled on second pass
        classes.emplace_back(0);
      }
    }
    // root_to_class_num maps to position of a root element to index of class.
    // This is initialized to classes.size() to indicate that the class has not
    // yet been assigned a position. Those positions are assigned as next_class
    // whenever we first encounter a member of the class.
    std::vector<IndexType> root_to_class_num(size(), classes.size());
    IndexType next_class = 0;
    // Second pass inserts into class vectors in order
    for (const auto i : c10::irange(size())) {
      const auto root = find(i);
      auto class_num = root_to_class_num.at(root);
      if (class_num == classes.size()) {
        // First element in this class
        root_to_class_num.at(root) = next_class;
        class_num = next_class++;
      }
      classes.at(class_num).push_back(i);
    }
    return classes;
  }

  //! Resize to zero losing all merge information without altering reserved
  //! capacity
  void clear() {
    parent_.clear();
    rank_.clear();
  }

  std::string toString(int indent_size = 0) const {
    std::stringstream ss;
    std::string ind = "";
    while (indent_size-- > 0) {
      ind += "  ";
    }

    ss << ind << "UnionFind:" << std::endl;
    ss << ind << "  size=" << size() << std::endl;

    const auto classes = computeEquivalenceClasses();

    ss << ind << "  classes:" << std::endl;
    for (const auto class_num : c10::irange(classes.size())) {
      ss << ind << "    " << class_num << ")" << std::endl;
      const auto& c = classes.at(class_num);
      for (const auto i : c) {
        ss << ind << "      " << std::to_string(i);
        if (find(i) == i) {
          ss << "  *"; // indicates root
        }
        ss << std::endl;
      }
    }

    return ss.str();
  }

 private:
  std::vector<IndexType> parent_;
  std::vector<IndexType> rank_;
};

} // namespace nvfuser
