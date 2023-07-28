// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

//! This holds a collection of _closed_ intervals in some totally ordered time
//! described by TimeType (int, float, etc.). This has O(n) space requirements,
//! costs O(n log(n)) to build, and intersection queries cost O(log(n)).
//!
//! Internally the structure looks like a binary tree where each node represents
//! a time range with a center point and is connected to two child nodes: one
//! will consist of all input intervals fully to the left of center and the
//! other holds input intervals fully to the right of center. The recursive
//! subdivision of the right and left children is straightforward.
//!
//! Intervals overlapping the center point of a node are represented by a pair
//! of lists holding the intervals in that subset sorted by their left and right
//! endpoints.
//!
//! See https://en.wikipedia.org/wiki/Interval_tree
template <typename TimeType, typename IndexType = int16_t>
class CenteredIntervalTree {
 public:
  //! Get a vector of indices of intervals that overlap a single point
  std::vector<IndexType> getIntervalsOverlappingPoint(TimeType x) const {
    std::vector<IndexType> output;
    root_.insertIntervalsOverlappingPoint(output, x);
    return output;
  }

  //! Get a vector of indices of intervals that overlap an arbitrary given
  //! interval
  std::vector<IndexType> getIntervalsOverlappingPoint(
      TimeType start,
      TimeType stop) const {
    std::vector<IndexType> output;
    root_.insertIntervalsOverlappingPoint(output, start, stop);
    return output;
  }

  //! Get a vector of indices of intervals that overlap a given interval in the
  //! original set of intervals.
  std::vector<IndexType> getIntervalsOverlappingPoint(
      IndexType interval_index) const {
    TORCH_CHECK(
        interval_index < root_.intervals_.size() &&
            (std::is_signed_v<IndexType> == true && interval_index >= 0),
        "Provided interval_index must be between 0 and intervals_.size() - 1 = ",
        root_.intervals_.size() - 1,
        ", inclusive. Found ",
        interval_index);
    auto [start, stop] = root_.intervals_.at(interval_index);
    return getIntervalsOverlappingPoint(start, stop);
  }

 private:
  class CenteredIntervalTreeNode {
   public:
    //! Inserts any intervals [a, b] in this subtree such that x \in [a, b].
    //! Notice that intervals are treated as closed intervals in this
    //! comparison.
    void insertIntervalsContainingPoint(
        std::vector<IndexType>& output,
        TimeType x) const {
      if (x == center_point_) {
        // if x is the center point, we just insert the entire center subset
        output.reserve(output.size() + local_to_global_.size());
        output.insert(
            output.end(), local_to_global_.begin(), local_to_global_.end());
        return;
      }

      // This will hold the sorted start or stop points
      CenteredIntervalTreeNode* child_node;

      if (x < center_point_) {
        // Since x is left of center and all center intervals overlap the
        // center, we need only look at the sorted starting points; any that
        // are less than x will overlap x.
        for (auto local_idx : center_start_sorted_) {
          auto global_idx = local_to_global_.at(local_idx);
          auto start = intervals_.at(global_idx).first;
          if (start > x) {
            break;
          }
          output.push_back(global_idx);
        }
        child_node = left_;
      } else {
        for (auto local_idx : center_stop_reverse_sorted_) {
          auto global_idx = local_to_global_.at(local_idx);
          auto stop = intervals_.at(global_idx).second;
          if (stop < x) {
            break;
          }
          output.push_back(global_idx);
        }
        child_node = right_;
      }

      // Recurse to left or right subset and merge results
      if (child_node) {
        child_node->insertInterval(output, x);
      }
    }

    //! Inserts any intervals [a, b] in this subtree that overlap [start, stop].
    //! Specifically, this inserts all intervals satisfying
    //!   (a >= start && a <= stop) ||
    //!   (b >= start && b <= stop) ||
    //!   (a < start && b > stop)
    void insertOverlappingIntervals(
        std::vector<IndexType>& output,
        TimeType start,
        TimeType stop) const {}

   private:
    TimeType center_point_;
    CenteredIntervalTreeNode* left_;
    CenteredIntervalTreeNode* right_;

    //! Reference to the global vector of intervals
    const std::vector<std::pair<TimeType, TimeType>>& intervals_;

    //! This is the local collection of global interval IDs
    std::vector<IndexType> local_to_global_;

    //! These hold locations within local_to_global. These are not the global
    //! IDs held in intervals.
    std::vector<IndexType> center_start_sorted_;
    //! Stop values are reverse sorted so that we can insert results starting
    //! with the first elements before breaking early.
    std::vector<IndexType> center_stop_reverse_sorted_;
  };

  CenteredIntervalTreeNode root_;
};

} // namespace nvfuser

