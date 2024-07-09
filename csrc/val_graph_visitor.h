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

#include <variant>

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

enum class Direction { Forward, Backward, Undefined };

std::ostream& operator<<(std::ostream&, const Direction);

using ExprPath = std::vector<std::pair<ExprGroup, Direction>>;

std::ostream& operator<<(std::ostream& os, const ExprPath& path);

inline Direction reverse(Direction direction) {
  if (direction == Direction::Forward) {
    return Direction::Backward;
  } else if (direction == Direction::Backward) {
    return Direction::Forward;
  } else {
    return Direction::Undefined;
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

// Traversal for finding the shortest path from ValGroups to another
// ValGroups. The algorithm is based on the standard BFS traversal,
// however, since ValGraph is not an undirected graph, the
// dependencies of ValGroups and ExprGroups need to be
// satisfied. Specifically, when visiting an ExprGroup, either its
// inputs or outputs must be visited before. Similarly, when visiting
// a ValGroup, there must be at least one defining ExprGroup or one
// use ExprGroup that is already visited.
//
// The main use case is tensor indexing, where a typical traversal
// would be from loop domains to allocation domains. Some
// indexing-specific specialization would be needed, for example,
// dependencies with broadcast domains can be ignored as their index
// is always just zero. The indexing shortest-path traversal would be
// implemented by subclassing this class.
class ValGraphBFS {
 public:
  using GroupType = std::variant<ExprGroup, ValGroup>;

  // Find the shortest path from the from_groups_ to to_groups_ on a
  // given graph. Dependency between vals and exprs must be satisfied.
  // It is an error if no valid path is found.
  static ExprPath getExprsBetween(
      const ValGraph& graph,
      const ValGroups& from,
      const ValGroups& to);

  virtual ~ValGraphBFS() = default;

 protected:
  ValGraphBFS(
      const ValGraph& graph,
      std::vector<GroupType> from_groups,
      std::vector<GroupType> to_groups)
      : graph_(graph),
        from_groups_(std::move(from_groups)),
        to_groups_(std::move(to_groups)) {}

  // Traverse from from_groups_ to to_groups_, recording each taken
  // path to generate the shortest path after the travesal
  virtual void traverse();

  // Find the shortest path from the from_groups_ to to_groups_. This
  // must be only used once traversal is completed.
  virtual ExprPath getShortestExprPath();

  // Check if a group is ready to visit. If yes, return the direction
  // and the prev nodes that should be visited before the given group
  // is visited.
  virtual std::optional<std::pair<Direction, std::vector<GroupType>>> isReady(
      const GroupType& group) const;

  // Check if an ExprGroup is ready to visit. Either all of its inputs
  // or all of outputs must have their dependencies satisfied. If
  // ready because the inputs are already visited, return
  // Direction::Forward and all the input groups. If ready because the
  // outputs are ready, return Direction::Backward and all the output groups.
  virtual std::optional<std::pair<Direction, std::vector<GroupType>>> isReady(
      const ExprGroup& expr_group) const;

  // Check if a ValGroup is ready to visit. Either its defining or use
  // ExprGroup must have its dependency satisfied. If ready because
  // there's a visited defining expr, return Direction::Forward and
  // the group of the defining expr. If ready because there's a
  // visited use expr, return Direction::Backward and the group of the
  // use expr.
  virtual std::optional<std::pair<Direction, std::vector<GroupType>>> isReady(
      const ValGroup& val_group) const;

  // If another group depends on a given group, check if that
  // dependency is considered satisfied. If the given group is already
  // visited, that should mean the dependency is satisfied.
  virtual bool isDependencySatisfied(const GroupType& dependency) const;

  // Check if a given group is already visited
  virtual bool isVisited(const GroupType& group) const;

  // Mark a group as visited
  virtual void setVisited(const GroupType& group);

  // Add new neighbors of a given group to the to_visit list
  virtual void addNewNeighbors(const GroupType& group);

  // Check if all to_groups_ are visited
  virtual bool allToGroupsVisited() const;

  // Set the previous groups of a given group that is visited in a
  // given direction
  virtual void setPrevGroups(
      const GroupType& group,
      const std::pair<Direction, std::vector<GroupType>>& prev_groups);

  // Hook to exclude certain graph nodes. See IndexingTraversal for a
  // concrete example
  virtual bool excludeFromTraversal(const GroupType& group) const {
    return false;
  }

 protected:
  const ValGraph& graph_;
  const std::vector<GroupType> from_groups_;
  const std::vector<GroupType> to_groups_;
  std::deque<GroupType> to_visit_;
  std::unordered_set<GroupType> visited_;
  std::unordered_map<GroupType, std::pair<Direction, std::vector<GroupType>>>
      prev_groups_;
};

} // namespace nvfuser
