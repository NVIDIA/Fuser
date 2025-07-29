// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stack>
#include <type_traits>
#include <vector>

#include <exceptions.h>

namespace nvfuser {

//! Centered Interval Tree implementation
//!
//! A centered interval tree is a data structure for efficiently querying
//! intervals that overlap with a given point or range. The tree is built
//! by recursively partitioning intervals around their center points.
//!
//! The tree supports:
//! - O(log n) insertion and deletion
//! - O(log n + k) query for overlapping intervals, where k is the number
//!   of overlapping intervals
//! - Efficient range queries and point queries
//!
//! Template parameters:
//! - CoordT: Type for start/end coordinates (must be comparable)
//! - PayloadT: Type of object associated with each interval
template <typename CoordT, typename PayloadT>
class CenteredIntervalTree {
 public:
  //! Represents an interval with associated data
  struct Interval {
    CoordT start;
    CoordT end;
    PayloadT data;

    Interval(CoordT start, CoordT end, const PayloadT& data)
        : start(start), end(end), data(data) {}

    //! Check if this interval overlaps with another interval
    bool overlaps(const Interval& other) const {
      return start <= other.end && end >= other.start;
    }

    //! Check if this interval contains a point
    bool contains(CoordT point) const {
      return start <= point && point <= end;
    }

    //! Get the center point of this interval
    //! For numeric types, use arithmetic mean; otherwise use start point
    CoordT center() const {
      if constexpr (std::is_arithmetic_v<CoordT>) {
        return (start + end) / 2;
      } else {
        // For non-numeric types, use start point as a fallback
        return start;
      }
    }
  };

  //! Node in the interval tree
  struct Node {
    std::vector<Interval> intervals;  // All intervals at this node
    std::vector<Interval> forward_sorted;  // Sorted by start point (ascending)
    std::vector<Interval> reverse_sorted;  // Sorted by end point (descending)
    CoordT center;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node(CoordT center) : center(center), left(nullptr), right(nullptr) {}
    
    //! Add an interval and maintain sorted vectors
    void addInterval(const Interval& interval) {
      intervals.push_back(interval);
      
      // Add to forward sorted (by start point)
      auto forward_it = std::lower_bound(
          forward_sorted.begin(), forward_sorted.end(), interval,
          [](const Interval& a, const Interval& b) {
            return a.start < b.start;
          });
      forward_sorted.insert(forward_it, interval);
      
      // Add to reverse sorted (by end point, descending)
      auto reverse_it = std::lower_bound(
          reverse_sorted.begin(), reverse_sorted.end(), interval,
          [](const Interval& a, const Interval& b) {
            return a.end > b.end;  // Reverse order
          });
      reverse_sorted.insert(reverse_it, interval);
    }
    
    //! Remove an interval and maintain sorted vectors
    void removeInterval(const Interval& interval) {
      // Remove from main intervals
      auto it = std::find_if(
          intervals.begin(), intervals.end(),
          [&](const Interval& existing) {
            return existing.start == interval.start &&
                existing.end == interval.end &&
                existing.data == interval.data;
          });
      if (it != intervals.end()) {
        intervals.erase(it);
      }
      
      // Remove from forward sorted
      auto forward_it = std::find_if(
          forward_sorted.begin(), forward_sorted.end(),
          [&](const Interval& existing) {
            return existing.start == interval.start &&
                existing.end == interval.end &&
                existing.data == interval.data;
          });
      if (forward_it != forward_sorted.end()) {
        forward_sorted.erase(forward_it);
      }
      
      // Remove from reverse sorted
      auto reverse_it = std::find_if(
          reverse_sorted.begin(), reverse_sorted.end(),
          [&](const Interval& existing) {
            return existing.start == interval.start &&
                existing.end == interval.end &&
                existing.data == interval.data;
          });
      if (reverse_it != reverse_sorted.end()) {
        reverse_sorted.erase(reverse_it);
      }
    }
  };

  CenteredIntervalTree() = default;
  ~CenteredIntervalTree() = default;
  
  //! Copy constructor
  CenteredIntervalTree(const CenteredIntervalTree& other) {
    if (other.root_) {
      root_ = copyTree(other.root_.get());
    }
  }
  
  //! Copy assignment operator
  CenteredIntervalTree& operator=(const CenteredIntervalTree& other) {
    if (this != &other) {
      if (other.root_) {
        root_ = copyTree(other.root_.get());
      } else {
        root_.reset();
      }
    }
    return *this;
  }

  //! Insert an interval into the tree
  void insert(CoordT start, CoordT end, const PayloadT& data) {
    Interval interval(start, end, data);
    root_ = insert(std::move(root_), interval);
  }

  //! Remove an interval from the tree
  void remove(CoordT start, CoordT end, const PayloadT& data) {
    Interval interval(start, end, data);
    root_ = remove(std::move(root_), interval);
  }

  //! Find all intervals that overlap with the given point
  std::vector<PayloadT> query(CoordT point) const {
    std::vector<PayloadT> result;
    query(root_.get(), point, result);
    return result;
  }

  //! Find all intervals that overlap with the given range
  std::vector<PayloadT> query(CoordT start, CoordT end) const {
    std::vector<PayloadT> result;
    queryRange(root_.get(), start, end, result);
    return result;
  }

  //! Check if the tree is empty
  bool empty() const {
    return root_ == nullptr;
  }

  //! Get the number of intervals in the tree
  size_t size() const {
    return size(root_.get());
  }

  //! Clear all intervals from the tree
  void clear() {
    root_.reset();
  }

  //! Build a tree from a list of intervals
  static CenteredIntervalTree fromIntervals(
      const std::vector<Interval>& intervals) {
    CenteredIntervalTree tree;
    for (const auto& interval : intervals) {
      tree.insert(interval.start, interval.end, interval.data);
    }
    return tree;
  }

 private:
  std::unique_ptr<Node> root_;

  //! Insert an interval into the tree using explicit stack
  std::unique_ptr<Node> insert(
      std::unique_ptr<Node> root,
      const Interval& interval) {
    if (!root) {
      auto new_node = std::make_unique<Node>(interval.center());
      new_node->addInterval(interval);
      return new_node;
    }

    std::stack<Node*> stack;
    Node* current = root.get();

    // Find the appropriate position for the interval
    while (current) {
      if (interval.center() < current->center) {
        if (!current->left) {
          // Create new left child
          current->left = std::make_unique<Node>(interval.center());
          current->left->intervals.push_back(interval);
          return root;
        }
        current = current->left.get();
      } else if (interval.center() > current->center) {
        if (!current->right) {
          // Create new right child
          current->right = std::make_unique<Node>(interval.center());
          current->right->intervals.push_back(interval);
          return root;
        }
        current = current->right.get();
      } else {
        // Same center, add to this node's intervals
        current->addInterval(interval);
        return root;
      }
    }

    return root;
  }

  //! Remove an interval from the tree using explicit stack
  std::unique_ptr<Node> remove(
      std::unique_ptr<Node> root,
      const Interval& interval) {
    if (!root) {
      return nullptr;
    }

    // Use a recursive approach to traverse the entire tree
    return removeRecursive(std::move(root), interval);
  }

  //! Recursive helper for remove
  std::unique_ptr<Node> removeRecursive(
      std::unique_ptr<Node> root,
      const Interval& interval) {
    if (!root) {
      return nullptr;
    }

    // Check if this node contains the interval
    root->removeInterval(interval);

    // Recursively check left and right subtrees
    if (root->left) {
      root->left = removeRecursive(std::move(root->left), interval);
    }
    if (root->right) {
      root->right = removeRecursive(std::move(root->right), interval);
    }

    // If no intervals left and no children, remove the node
    if (root->intervals.empty() && !root->left && !root->right) {
      return nullptr;
    }

    return root;
  }

  //! Query for intervals containing a point using explicit stack
  void query(
      const Node* root,
      CoordT point,
      std::vector<PayloadT>& result) const {
    if (!root) {
      return;
    }

    std::stack<const Node*> stack;
    stack.push(root);

    while (!stack.empty()) {
      const Node* current = stack.top();
      stack.pop();

      if (!current) {
        continue;
      }

      // Check intervals in current node
      for (const auto& interval : current->intervals) {
        if (interval.contains(point)) {
          result.push_back(interval.data);
        }
      }

      // Always traverse both subtrees since intervals can overlap
      // regardless of their center points
      if (current->left) {
        stack.push(current->left.get());
      }
      if (current->right) {
        stack.push(current->right.get());
      }
    }
  }

  //! Query for intervals overlapping with a range using explicit stack
  void queryRange(
      const Node* root,
      CoordT start,
      CoordT end,
      std::vector<PayloadT>& result) const {
    if (!root) {
      return;
    }

    std::stack<const Node*> stack;
    stack.push(root);

    while (!stack.empty()) {
      const Node* current = stack.top();
      stack.pop();

      if (!current) {
        continue;
      }

      // Check intervals in current node
      for (const auto& interval : current->intervals) {
        if (interval.overlaps(Interval(start, end, PayloadT()))) {
          result.push_back(interval.data);
        }
      }

      // Always traverse both subtrees since intervals can overlap
      // regardless of their center points
      if (current->left) {
        stack.push(current->left.get());
      }
      if (current->right) {
        stack.push(current->right.get());
      }
    }
  }

  //! Count the number of intervals in the tree using explicit stack
  size_t size(const Node* root) const {
    if (!root) {
      return 0;
    }

    size_t count = 0;
    std::stack<const Node*> stack;
    stack.push(root);

    while (!stack.empty()) {
      const Node* current = stack.top();
      stack.pop();

      if (!current) {
        continue;
      }

      count += current->intervals.size();

      if (current->left) {
        stack.push(current->left.get());
      }
      if (current->right) {
        stack.push(current->right.get());
      }
    }

    return count;
  }

  //! Helper method to copy a tree node and its children
  std::unique_ptr<Node> copyTree(const Node* node) const {
    if (!node) {
      return nullptr;
    }
    
    auto new_node = std::make_unique<Node>(node->center);
    new_node->intervals = node->intervals;
    new_node->forward_sorted = node->forward_sorted;
    new_node->reverse_sorted = node->reverse_sorted;
    
    if (node->left) {
      new_node->left = copyTree(node->left.get());
    }
    if (node->right) {
      new_node->right = copyTree(node->right.get());
    }
    
    return new_node;
  }
};

} // namespace nvfuser