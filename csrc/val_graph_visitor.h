// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <ir/all_nodes.h>
#include <val_graph.h>

namespace nvfuser {

// Iterates through a Val Graph in topological order, calling handle on
// all Val and all Expr groups in a forward topological order.
//
// Warning: A ValGraph is not guaranteed to be a DAG. In fact, the
// AlmostExact and Permissive graphs would have cycles with a ValGroup
// and an ExprGroup. For example:
//
// [i0, 1]
// merge
// [i0*1]
// Current ValGroups: {{i0}, {1}, {i0*1}}
// map i0 and i0*1 as they effectively have the same extent
// Final ValGroups: {{i0, i0*1}, {1}}
//
// Here, the merge expr is the user of i0 and the definition of
// i0*1. Since i0 and i0*1 are mapped, the dependency chain looks
// like:
//
// {i0, i0*1} ----> {merge} ----> {i0, i0*1}
//             use           def
//
// These ExprGroups are called trivial ExprGroups (see also
// ValGraph::isTrivialExprGroup).
//
// Strictly speaking, these cycles mean there's no valid topological
// order anymore. In our use cases for IdModel, however, it's likely
// sufficient to return an ordering such as:
//
// {i0, i0*1} -> {merge}
//
// I.e., we visit {i0, i0*1} first even though {merge} is technically
// a definition.
//
// Another alternative may be simply giving up when such a cycle is
// detected, which may be more preferrable as it would be less
// confusing. At this moment, this visitor is only used with graphs
// with no such cycle. Should be revisited when necessary.
//
// Warning: This is not a great iterator if there's a desire to minimize paths
// traveled to simply visit all ValGroups in order. See ExprsBetween to see how
// we might minimize paths.
class ValGraphVisitor {
 public:
  ValGraphVisitor() = delete;

  ValGraphVisitor& operator=(const ValGraphVisitor& other) = delete;

  ValGraphVisitor& operator=(ValGraphVisitor&& other) = delete;

  virtual ~ValGraphVisitor() = default;

 protected:
  ValGraphVisitor(const ValGraph& val_graph) : val_graph_(val_graph) {}

  ValGraphVisitor(const ValGraphVisitor& other) = default;

  ValGraphVisitor(ValGraphVisitor&& other) = default;

  virtual void handle(const ValGroup& val_group) = 0;
  virtual void handle(const ExprGroup& expr_group) = 0;

  void traverse();

  const ValGraph& graph() {
    return val_graph_;
  };

 private:
  const ValGraph& val_graph_;
};

// Statement sorting based on ValGraphVisitor, see warnings to ValGraph Visitor.
class ValGraphStmtSort : public ValGraphVisitor {
 public:
  ValGraphStmtSort(const ValGraph& val_graph) : ValGraphVisitor(val_graph) {
    ValGraphVisitor::traverse();
  }

  // Return non-reference so that code like below can work
  // for (auto expr_group: ValGraphStmtSort(graph).exprs())
  ExprGroups exprs() const {
    return sorted_exprs_;
  }

  ValGroups vals() const {
    return sorted_vals_;
  }

  ~ValGraphStmtSort() override = default;

 protected:
  using ValGraphVisitor::handle;

  void handle(const ValGroup& val_group) override {
    sorted_vals_.pushBack(val_group);
  }

  void handle(const ExprGroup& expr_group) override {
    sorted_exprs_.pushBack(expr_group);
  }

  ExprGroups sorted_exprs_;
  ValGroups sorted_vals_;
};

enum class ExprDirection { Forward, Backward, Undefined };

std::ostream& operator<<(std::ostream&, const ExprDirection);

using ExprPath = std::vector<std::pair<ExprGroup, ExprDirection>>;

inline ExprDirection reverse(ExprDirection direction) {
  if (direction == ExprDirection::Forward) {
    return ExprDirection::Backward;
  } else if (direction == ExprDirection::Backward) {
    return ExprDirection::Forward;
  } else {
    return ExprDirection::Undefined;
  }
}

inline ExprPath reverse(const ExprPath& path) {
  auto rev = path;
  std::reverse(rev.begin(), rev.end());
  for (auto& [eg, direction] : rev) {
    direction = reverse(direction);
  }
  return rev;
}

class ValGraphBFS {
 public:
  using GroupType = std::variant<ExprGroup, ValGroup>;

  static ExprPath getExprsBetweenVals(
      const ValGraph& graph,
      const ValGroups& from,
      const ValGroups& to);

  ValGraphBFS(
      const ValGraph& graph,
      std::vector<GroupType> from_groups,
      std::vector<GroupType> to_groups)
      : graph_(graph),
        from_groups_(std::move(from_groups)),
        to_groups_(std::move(to_groups)) {}

  virtual ~ValGraphBFS() = default;

  virtual void handle(const GroupType& group);

  virtual void handle(const ValGroup& val_group);

  virtual void handle(const ExprGroup& expr_group);

  virtual void traverse();

  virtual bool isReady(const GroupType& group) const;

  virtual bool isReady(const ExprGroup& expr_group) const;

  virtual bool isDependencySatisfied(const GroupType& group) const;

  virtual bool isReady(const ValGroup& val_group) const;

  virtual bool isVisited(const GroupType& group) const;

  virtual void setVisited(const GroupType& group);

  virtual void addNewNeighbors(const GroupType& group);

  virtual void setPrevGroup(const GroupType& group);

  virtual bool excludeFromTraversal(const GroupType& group) const {
    return false;
  }

  // Extend this to support Val paths as well
  virtual ExprPath getShortestExprPath();

 protected:
  const ValGraph& graph_;
  const std::vector<GroupType> from_groups_;
  const std::vector<GroupType> to_groups_;
  std::deque<GroupType> to_visit_;
  std::unordered_set<GroupType> visited_;
  std::unordered_map<GroupType, std::vector<GroupType>> prev_groups_;
};

} // namespace nvfuser
