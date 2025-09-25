// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <algorithm>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// For printing of the set when using a Statement as the type for the set
#include <ir/base_nodes.h>

namespace nvfuser {

namespace {

template <typename T>
std::string abstractToString(T* ptr) {
  return ptr->toString();
}

template <typename T>
std::string abstractToString(T ref) {
  return ref.toString();
}

} // namespace

// Vector like class that will prevent adding duplicate entries by also
// maintaing a set
//
// TODO: Can we support std::back_inserter with this class?
template <typename T, typename Hash = std::hash<T>>
class VectorOfUniqueEntries {
 public:
  // Naming not following our conventions but using the same name as
  // std::vector makes it more convenient when we want to use this
  // class as if it's like std::vector
  using value_type = T;

  VectorOfUniqueEntries() = default;

  VectorOfUniqueEntries(const std::initializer_list<T>& initializer)
      : VectorOfUniqueEntries(initializer.begin(), initializer.end()) {}

  VectorOfUniqueEntries(const VectorOfUniqueEntries& other) = default;

  VectorOfUniqueEntries& operator=(const VectorOfUniqueEntries& other) =
      default;

  template <class InputIt>
  VectorOfUniqueEntries(InputIt first, InputIt last) {
    pushBack(first, last);
  }

  template <class Container>
  VectorOfUniqueEntries(const Container& container)
      : VectorOfUniqueEntries(container.begin(), container.end()) {}

  template <class InputIt>
  void pushBack(InputIt first, InputIt last) {
    while (first != last) {
      pushBack(*first++);
    }
  }

  // Returns if a node was actually added
  bool pushBack(T entry) {
    if (set_.emplace(entry).second) {
      vector_.push_back(entry);
      return true;
    }
    return false;
  }

  // Returns true if any node was added
  bool pushBack(const VectorOfUniqueEntries<T, Hash>& other) {
    return pushBack(other.vector());
  }

  // Returns true if any node was added
  template <typename OtherType>
  bool pushBack(const std::vector<OtherType>& other) {
    bool any_added = false;
    for (const auto& entry : other) {
      auto added = pushBack(entry);
      any_added = any_added || added;
    }
    return any_added;
  }

  // Returns a new VectorOfUniqueEntries with entries that are in both this and
  // other, order is preserved as this.
  VectorOfUniqueEntries<T, Hash> computeIntersect(
      const VectorOfUniqueEntries<T, Hash>& other) const {
    VectorOfUniqueEntries<T, Hash> intersection;
    for (const auto& entry : vector()) {
      if (other.has(entry)) {
        intersection.pushBack(entry);
      }
    }
    return intersection;
  }

  bool hasIntersect(const VectorOfUniqueEntries<T, Hash>& other) const {
    return std::ranges::any_of(
        vector(), [&](const auto& entry) { return other.has(entry); });
  }

  // Returns a new VectorOfUniqueEntries with entries that are in this but not
  // in other.
  VectorOfUniqueEntries<T, Hash> computeSubtract(
      const VectorOfUniqueEntries<T, Hash>& other) const {
    VectorOfUniqueEntries<T, Hash> subtraction;
    for (const auto& entry : vector()) {
      if (!other.has(entry)) {
        subtraction.pushBack(entry);
      }
    }
    return subtraction;
  }

  // Returns a new VectorOfUniqueEntries with entries that are either in this or
  // other.
  VectorOfUniqueEntries<T, Hash> computeUnion(
      const VectorOfUniqueEntries<T, Hash>& other) const {
    VectorOfUniqueEntries<T, Hash> union_(*this);
    for (const auto& entry : other.vector()) {
      union_.pushBack(entry);
    }
    return union_;
  }

  // Returns a const vector useful for iterating on
  const std::vector<T>& vector() const {
    return vector_;
  }

  const std::unordered_set<T>& set() const {
    return set_;
  }

  bool operator==(const VectorOfUniqueEntries& other) const {
    return vector() == other.vector();
  }

  bool operator!=(const VectorOfUniqueEntries& other) const {
    return !operator==(other);
  }

  // Returns first element in vector
  T front() const {
#if !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    NVF_ERROR(!empty());
#endif // !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    return vector_.front();
  }

  // Returns last element in vector
  T back() const {
#if !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    NVF_ERROR(!empty());
#endif // !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    return vector_.back();
  }

  // Remove and returns the last element in vector
  T popBack() {
#if !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    NVF_ERROR(!empty());
#endif // !defined(NDEBUG) || defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    T v = vector_.back();
    set_.erase(v);
    vector_.pop_back();
    return v;
  }

  // Returns if this container is empty
  bool empty() const {
    return vector_.empty();
  }

  void clear() {
    vector_.clear();
    set_.clear();
  }

  // Returns the number of elements in this container
  int64_t size() const {
    return vector_.size();
  }

  // Returns if entry is in this vector
  bool has(T entry) const {
    return set_.find(entry) != set_.end();
  }

  // Erase given entry from the containers if
  //  there is a match.
  int64_t erase(T entry) {
    vector_.erase(
        std::remove_if(
            vector_.begin(),
            vector_.end(),
            [entry](T val) { return val == entry; }),
        vector_.end());

    return static_cast<int64_t>(set_.erase(entry));
  }

  // Insert elements at the end of the container.
  template <typename InputIt>
  void insert(InputIt begin, InputIt end) {
    for (auto it = begin; it != end; it++) {
      pushBack(*it);
    }
  }

  // Returns iterator pointing to the beginning of vector container
  typename std::vector<T>::const_iterator begin() const {
    return vector().begin();
  }

  // Returns iterator pointing to the end of vector container
  typename std::vector<T>::const_iterator end() const {
    return vector().end();
  }

  // Returns iterator pointing to the beginning of vector container
  typename std::vector<T>::iterator begin() {
    return vector_.begin();
  }

  // Returns iterator pointing to the end of vector container
  typename std::vector<T>::iterator end() {
    return vector_.end();
  }

  auto rbegin() const {
    return vector().rbegin();
  }

  auto rend() const {
    return vector().rend();
  }

  auto rbegin() {
    return vector_.rbegin();
  }

  auto rend() {
    return vector_.rend();
  }

  T& at(int64_t pos) {
    return vector_.at(pos);
  }

  const T& at(int64_t pos) const {
    return vector_.at(pos);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "{ ";
    for (const auto& entry : vector()) {
      ss << abstractToString(entry);
      if (entry != vector().back()) {
        ss << "; ";
      }
    }
    ss << " }";
    return ss.str();
  }

 private:
  std::vector<T> vector_;
  std::unordered_set<T, Hash> set_;
};

//! Container class DisjointSet models equivalence relationships
//!
//! Each instance of this class keeps equivalence sets
//! DisjointSet::mapEntries(a,b) makes the full set of a and b equivalent
//! DisjointSet::*AreMapped(a,b) checks if a and b belong to the same disjoint
//! set
template <typename T, typename Hash = std::hash<T>>
class NVF_API DisjointSets {
 public:
  using DisjointSet = std::shared_ptr<VectorOfUniqueEntries<T, Hash>>;
  using DisjointSetMap = std::unordered_map<T, DisjointSet, Hash>;

  DisjointSets() = default;

  DisjointSets(const DisjointSets<T, Hash>& other);

  DisjointSets(DisjointSets<T, Hash>&& other) = default;

  DisjointSets<T, Hash>& operator=(const DisjointSets<T, Hash>& other);

  DisjointSets<T, Hash>& operator=(DisjointSets<T, Hash>&& other) = default;

  friend void swap(DisjointSets<T, Hash>& sets1, DisjointSets<T, Hash>& sets2) {
    using std::swap;
    swap(sets1.disjoint_sets_, sets2.disjoint_sets_);
    swap(sets1.disjoint_set_maps_, sets2.disjoint_set_maps_);
  }

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as VectorOfUniqueEntries is not returned as a const.
  const DisjointSetMap& disjointSetMap() const {
    return disjoint_set_maps_;
  }

  // Warning: returned values should never be modified. This accessor isn't
  // strictly safe as VectorOfUniqueEntries is not returned as a const.
  const std::vector<DisjointSet>& disjointSets() const {
    return disjoint_sets_;
  }

  typename DisjointSetMap::iterator find(T entry) {
    return disjoint_set_maps_.find(entry);
  }

  typename DisjointSetMap::iterator end() {
    return disjoint_set_maps_.end();
  }

  typename DisjointSetMap::const_iterator find(T entry) const {
    return disjoint_set_maps_.find(entry);
  }

  typename DisjointSetMap::const_iterator end() const {
    return disjoint_set_maps_.end();
  }

  // Return the entire disjoint set of provided entry
  const VectorOfUniqueEntries<T, Hash>& getDisjointSetOf(T entry) const {
    auto set_it = disjoint_set_maps_.find(entry);
    NVF_ERROR(
        set_it != disjoint_set_maps_.end(),
        "Could not find entry for ",
        entry->toString());
    return *(set_it->second);
  }

  // Initializes a new set for provided entry
  std::pair<typename DisjointSetMap::iterator, bool> initializeSet(T entry) {
    auto disjoint_set_maps_it = disjoint_set_maps_.find(entry);
    if (disjoint_set_maps_it != disjoint_set_maps_.end()) {
      return std::make_pair(disjoint_set_maps_it, false);
    }

    disjoint_sets_.push_back(
        std::make_shared<VectorOfUniqueEntries<T, Hash>>());
    disjoint_sets_.back()->pushBack(entry);
    return disjoint_set_maps_.emplace(
        std::make_pair(entry, disjoint_sets_.back()));
  }

  // Adds all of the disjoint set belonging to entry1 to the disjoint set
  // belonging to entry0, maps all entries of disjoint set belonging to entry1
  // to entry0, removes original disjoint set belonging to entry1.
  void mapEntries(T entry0, T entry1) {
    auto set_it_0 = disjoint_set_maps_.find(entry0);
    auto set_it_1 = disjoint_set_maps_.find(entry1);

    auto set_0_found = set_it_0 != disjoint_set_maps_.end();
    auto set_1_found = set_it_1 != disjoint_set_maps_.end();

    // Sets already joined
    if (set_0_found && set_1_found && set_it_0->second == set_it_1->second) {
      return;
    }

    // Make and map new set
    disjoint_sets_.push_back(
        std::make_shared<VectorOfUniqueEntries<T, Hash>>());
    auto new_set = disjoint_sets_.back();

    // Add an entry to new_set along with the other entries previously
    // grouped together with the entry. The existing set is erased.
    auto mergeSets = [this](const T& entry, auto& new_set) {
      if (auto it = disjoint_set_maps_.find(entry);
          it != disjoint_set_maps_.end()) {
        auto existing_set = it->second;
        for (const auto& existing_entry : *existing_set) {
          new_set->pushBack(existing_entry);
          disjoint_set_maps_[existing_entry] = new_set;
        }
        disjoint_sets_.erase(std::find(
            disjoint_sets_.begin(), disjoint_sets_.end(), existing_set));
      } else {
        new_set->pushBack(entry);
        disjoint_set_maps_[entry] = new_set;
      }
    };

    mergeSets(entry0, new_set);

    // This should be after we enter a new set in case it doesn't exist.
    if (entry0 == entry1) {
      return;
    }

    mergeSets(entry1, new_set);
  }

  // Will assert if provided entry0 is not in any disjoint set, otherwise
  // returns if entry0 and entry1 are in the same disjoint set.
  bool strictAreMapped(T entry0, T entry1) const {
    auto entry_it = disjointSetMap().find(entry0);
    NVF_ERROR(
        entry_it != disjointSetMap().end(),
        "Strict mapping failed on element: ",
        abstractToString(entry0),
        " either an error occurred, or non strict mapping should have been "
        "used.");
    return entry_it->second->has(entry1);
  }

  // If entry0 doesn't have a disjoint set returns false, otherwise returns if
  // entry0 and entry1 are in the same disjoint set.
  bool permissiveAreMapped(T entry0, T entry1) const {
    auto entry_it = disjointSetMap().find(entry0);
    if (entry_it == disjointSetMap().end()) {
      return false;
    }
    return entry_it->second->has(entry1);
  }

  // Returns if a set exists with provided entry
  bool mappingExists(T entry) const {
    return disjoint_set_maps_.find(entry) != disjoint_set_maps_.end();
  }

  // Append a new item into an existing disjoint set, and add mapping for this
  // item
  void appendToSet(T item, DisjointSet set) {
    NVF_CHECK(!mappingExists(item), "Item already exist.");
    NVF_CHECK(
        !set->empty() && &getDisjointSetOf(set->front()) == set.get(),
        "Invalid disjoint set given.");
    set->pushBack(item);
    disjoint_set_maps_[item] = set;
  }

  // Erases element if it exists in the disjoint set. Returns true if element
  // found.
  bool erase(T entry) {
    auto entry_it = disjoint_set_maps_.find(entry);
    if (entry_it == disjoint_set_maps_.end()) {
      return false;
    }

    auto set = entry_it->second;
    if (set->size() == 1) {
      NVF_ERROR(
          set->front() == entry,
          "Disjoint set container found to be in inconsistent state.");
      disjoint_set_maps_.erase(entry);
      disjoint_sets_.erase(
          std::find(disjoint_sets_.begin(), disjoint_sets_.end(), set));
    } else {
      disjoint_set_maps_.erase(entry);
      set->erase(entry);
    }

    return true;
  }

  // Returns a deterministic list of all entries that have been added to any
  // disjoint set.
  //
  // Warning: constructed on every call, consider caching result.
  VectorOfUniqueEntries<T, Hash> getAllElements() const {
    VectorOfUniqueEntries<T, Hash> all_elements;
    for (auto set : disjoint_sets_) {
      for (auto entry : set->vector()) {
        all_elements.pushBack(entry);
      }
    }
    return all_elements;
  }

  // Completely clears all disjoint sets
  void clear() {
    disjoint_set_maps_.clear();
    disjoint_sets_.clear();
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "disjoint sets{\n";
    const std::string sep("  ");
    for (auto s_ptr : disjoint_sets_) {
      auto& set = *s_ptr;
      ss << sep << abstractToString(set) << "\n";
    }
    ss << "}";
    return ss.str();
  }

  int64_t size() const {
    return disjoint_sets_.size();
  }

 private:
  // Disjoint sets
  DisjointSetMap disjoint_set_maps_;

  // Keep a list of disjoint_sets that's deterministic to iterate over
  //
  // TODO: Should this just be a
  // VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries ?
  std::vector<DisjointSet> disjoint_sets_;
};

template <typename T, typename Hash>
DisjointSets<T, Hash>::DisjointSets(const DisjointSets<T, Hash>& other) {
  std::unordered_map<DisjointSet, int> ptr_map;

  // Deep copy the vector of the disjoint sets, keeping the same
  // ordering of the sets.
  for (const auto& other_set : other.disjoint_sets_) {
    auto new_set = std::make_shared<VectorOfUniqueEntries<T, Hash>>(*other_set);
    int new_set_index = disjoint_sets_.size();
    disjoint_sets_.emplace_back(new_set);
    NVF_ERROR(
        ptr_map.emplace(other_set, new_set_index).second,
        "Duplicated set found: ",
        other_set->toString());
  }

  // Copy the mappings using the new sets
  for (const auto& kv : other.disjoint_set_maps_) {
    const auto key = kv.first;
    const auto new_set_index = ptr_map.at(kv.second);
    disjoint_set_maps_.emplace(key, disjoint_sets_.at(new_set_index));
  }
}

template <typename T, typename Hash>
DisjointSets<T, Hash>& DisjointSets<T, Hash>::operator=(
    const DisjointSets<T, Hash>& other) {
  disjoint_set_maps_.clear();
  disjoint_sets_.clear();

  DisjointSets<T, Hash> copy(other);
  swap(*this, copy);
  return *this;
}

} // namespace nvfuser
