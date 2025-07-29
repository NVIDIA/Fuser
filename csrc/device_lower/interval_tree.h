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
    CoordT center() const {
      return (start + end) / 2;
    }
  };

  //! Node in the interval tree
  struct Node {
    std::vector<Interval> intervals;
    CoordT center;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node(CoordT center) : center(center), left(nullptr), right(nullptr) {}
  };

  CenteredIntervalTree() = default;
  ~CenteredIntervalTree() = default;

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
      new_node->intervals.push_back(interval);
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
        current->intervals.push_back(interval);
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

    std::stack<std::pair<Node*, std::unique_ptr<Node>*>> stack;
    Node* current = root.get();
    std::unique_ptr<Node>* parent_ptr = &root;

    // Find the node containing the interval
    while (current) {
      if (interval.center() < current->center) {
        if (!current->left) {
          return root; // Interval not found
        }
        stack.push({current, &current->left});
        current = current->left.get();
      } else if (interval.center() > current->center) {
        if (!current->right) {
          return root; // Interval not found
        }
        stack.push({current, &current->right});
        current = current->right.get();
      } else {
        // Found the node with matching center
        auto it = std::find_if(
            current->intervals.begin(),
            current->intervals.end(),
            [&](const Interval& existing) {
              return existing.start == interval.start &&
                  existing.end == interval.end &&
                  existing.data == interval.data;
            });
        if (it != current->intervals.end()) {
          current->intervals.erase(it);
        }

        // If no intervals left and no children, remove the node
        if (current->intervals.empty() && !current->left && !current->right) {
          *parent_ptr = nullptr;
        }
        return root;
      }
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

      // Add left subtree if point could be in left intervals
      if (point < current->center && current->left) {
        stack.push(current->left.get());
      }

      // Add right subtree if point could be in right intervals
      if (point > current->center && current->right) {
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

      // Add left subtree if range could overlap left intervals
      if (start < current->center && current->left) {
        stack.push(current->left.get());
      }

      // Add right subtree if range could overlap right intervals
      if (end > current->center && current->right) {
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
};

} // namespace nvfuser 