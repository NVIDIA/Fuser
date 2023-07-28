// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/irange.h>

#include <memory>
#include <ostream>
#include <sstream>
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
  CenteredIntervalTree(
      const std::vector<std::pair<TimeType, TimeType>>& intervals) {
    std::vector<IndexType> all_indices;
    all_indices.reserve(intervals.size());
    for (IndexType i : c10::irange(intervals.size())) {
      all_indices.push_back(i);
    }
    root_ = std::make_unique<CenteredIntervalTreeNode>(intervals, all_indices);
  }

  //! Get a vector of indices of intervals that contain a point
  std::vector<IndexType> getIntervalsContainingPoint(TimeType x) const {
    std::vector<IndexType> output;
    root_->insertIntervalsContainingPoint(output, x);
    return output;
  }

  //! Get a vector of indices of intervals that overlap an arbitrary given
  //! interval
  std::vector<IndexType> getOverlappingIntervals(TimeType start, TimeType stop)
      const {
    std::vector<IndexType> output;
    root_->insertOverlappingIntervals(output, start, stop);
    return output;
  }

  //! Get a vector of indices of intervals that overlap a given interval in the
  //! original set of intervals.
  std::vector<IndexType> getIntervalsOverlappingKnownInterval(
      IndexType interval_index) const {
    TORCH_CHECK(
        interval_index < root_.intervals_.size() &&
            (std::is_signed_v<IndexType> == true && interval_index >= 0),
        "Provided interval_index must be between 0 and intervals_.size() - 1 = ",
        root_.intervals_.size() - 1,
        ", inclusive. Found ",
        interval_index);
    auto [start, stop] = root_.intervals_.at(interval_index);
    return getOverlappingIntervals(start, stop);
  }

  void print(std::ostream& out) const {
    out << "CenteredIntervalTree {\n";
    if (root_) {
      root_->print(out, /* indent */ 1);
    } else {
      out << "  null\n";
    }
    out << "}";
  }

  std::string toString() const {
    std::stringstream ss;
    print(ss);
    return ss.str();
  }

 private:
  class CenteredIntervalTreeNode {
   public:
    //! Construct a node with global interval list intervals restricted to the
    //! given subset of indices
    CenteredIntervalTreeNode(
        const std::vector<std::pair<TimeType, TimeType>>& intervals,
        const std::vector<IndexType>& indices)
        : intervals_(intervals) {
      TORCH_CHECK(
          !indices.empty(), "Cannot initialize node with zero intervals");

      TORCH_CHECK(
          intervals_.size() < std::numeric_limits<IndexType>::max(),
          "IndexType is too small to fit ",
          intervals_.size(),
          " intervals.");

      // Decide on center point.
      selectCenterPoint(indices);

      // Create left, right, and center index lists
      std::vector<IndexType> left_indices;
      std::vector<IndexType> right_indices;
      for (const auto ind : indices) {
        TORCH_CHECK(ind < (IndexType)intervals_.size());
        const auto [start, stop] = intervals_.at(ind);
        TORCH_CHECK(start <= stop);
        if (stop < center_point_) {
          left_indices.push_back(ind);
        } else if (start > center_point_) {
          right_indices.push_back(ind);
        } else {
          // append to the center subset
          center_start_sorted_.push_back(ind);
        }
      }

      // sort center intervals by start point
      std::sort(
          center_start_sorted_.begin(),
          center_start_sorted_.end(),
          [this](IndexType const& a, IndexType const& b) -> bool {
            return intervals_.at(a).first < intervals_.at(b).first;
          });

      // copy from the start-sorted list, then reverse sort by stop
      center_stop_reverse_sorted_.insert(
          center_stop_reverse_sorted_.end(),
          center_start_sorted_.begin(),
          center_start_sorted_.end());
      std::sort(
          center_stop_reverse_sorted_.begin(),
          center_stop_reverse_sorted_.end(),
          [this](IndexType const& a, IndexType const& b) -> bool {
            return intervals_.at(a).second > intervals_.at(b).second;
          });

      // Recurse to initialize left and right
      if (!left_indices.empty()) {
        left_ =
            std::make_unique<CenteredIntervalTreeNode>(intervals, left_indices);
        // early free to make more space for allocationsin right branch
        left_indices.clear();
        left_indices.shrink_to_fit();
      }
      if (!right_indices.empty()) {
        right_ = std::make_unique<CenteredIntervalTreeNode>(
            intervals, right_indices);
      }
    }

    //! Inserts any intervals [a, b] in this subtree such that x \in [a, b].
    //! Notice that intervals are treated as closed intervals in this
    //! comparison.
    void insertIntervalsContainingPoint(
        std::vector<IndexType>& output,
        TimeType x) const {
      if (x == center_point_) {
        // if x is the center point, we just insert the entire center subset
        output.reserve(output.size() + center_start_sorted_.size());
        output.insert(
            output.end(),
            center_start_sorted_.begin(),
            center_start_sorted_.end());
        return;
      }

      // This will hold the sorted start or stop points
      CenteredIntervalTreeNode* child_node = nullptr;

      if (x < center_point_) {
        // Since x is left of center and all center intervals overlap the
        // center, we need only look at the sorted starting points; any that
        // are less than x will overlap x.
        for (auto idx : center_start_sorted_) {
          auto start = intervals_.at(idx).first;
          if (start > x) {
            break;
          }
          output.push_back(idx);
        }
        child_node = left_.get();
      } else {
        for (auto idx : center_stop_reverse_sorted_) {
          auto stop = intervals_.at(idx).second;
          if (stop < x) {
            break;
          }
          output.push_back(idx);
        }
        child_node = right_.get();
      }

      // Recurse to left or right subset and merge results
      if (child_node) {
        child_node->insertIntervalsContainingPoint(output, x);
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

    void print(std::ostream& out, int indent = 0) const {
      std::stringstream indent_ss;
      for ([[maybe_unused]] auto i : c10::irange(indent)) {
        indent_ss << "  ";
      }
      auto indent_str = indent_ss.str();
      out << indent_str << "center point: " << center_point_ << "\n";
      out << indent_str << "center intervals:\n";
      for (auto ind : center_start_sorted_) {
        const auto [start, stop] = intervals_.at(ind);
        out << indent_str << "  [ " << start << " , " << stop << " ]\n";
      }
      if (left_) {
        out << indent_str << "left branch:\n";
        left_->print(out, indent + 1);
      }
      if (right_) {
        out << indent_str << "right branch:\n";
        right_->print(out, indent + 1);
      }
    }

   private:
    //! This method picks the approximate midpoint of the span of all intervals
    //! in a subset as the center point. A more detailed approach would result
    //! in a more balanced and efficient tree.
    void selectCenterPoint(const std::vector<IndexType>& indices) {
      TimeType min_start = 0;
      TimeType max_stop = 0;
      bool init = true;
      for (const auto ind : indices) {
        const auto [start, stop] = intervals_.at(ind);
        if (init) {
          min_start = start;
          max_stop = stop;
          init = false;
        } else {
          min_start = std::min(start, min_start);
          max_stop = std::max(stop, max_stop);
        }
      }
      center_point_ = (min_start + max_stop) / 2;
    }

   private:
    //! Reference to the global vector of intervals
    const std::vector<std::pair<TimeType, TimeType>>& intervals_;

    //! This is the collection of indices held in this node (all those in this
    //! branch that contain center_point_). This will be sorted ascending by
    //! start point.
    std::vector<IndexType> center_start_sorted_;
    //! Stop values are reverse sorted so that we can insert results starting
    //! with the first elements before breaking early.
    std::vector<IndexType> center_stop_reverse_sorted_;

    std::unique_ptr<CenteredIntervalTreeNode> left_ = nullptr;
    std::unique_ptr<CenteredIntervalTreeNode> right_ = nullptr;

    TimeType center_point_ = 0;
  };

 private:
  std::unique_ptr<CenteredIntervalTreeNode> root_ = nullptr;
};

} // namespace nvfuser
