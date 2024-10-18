// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <bfs.h>
#include <disjoint_set.h>
#include <id_model/to_string.h>
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
  ValGraphVisitor(const ValGraph& val_graph, bool allow_cycle = true)
      : val_graph_(val_graph), allow_cycle_(allow_cycle) {}

  ValGraphVisitor(const ValGraphVisitor& other) = default;

  ValGraphVisitor(ValGraphVisitor&& other) = default;

  virtual void handle(const ValGroup& val_group) = 0;
  virtual void handle(const ExprGroup& expr_group) = 0;

  // Returns if the traversal was successful. If false, error_message_
  // should be populated.
  bool traverse();

  const ValGraph& graph() {
    return val_graph_;
  };

  const std::string& errorMessage() const {
    return error_message_;
  }

 private:
  const ValGraph& val_graph_;
  bool allow_cycle_ = true;
  std::string error_message_;
};

// Statement sorting based on ValGraphVisitor, see warnings to ValGraph Visitor.
class ValGraphStmtSort : public ValGraphVisitor {
 public:
  ValGraphStmtSort(const ValGraph& val_graph, bool allow_cycle = true)
      : ValGraphVisitor(val_graph, allow_cycle) {
    NVF_ERROR(ValGraphVisitor::traverse(), errorMessage());
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

bool isCyclic(const ValGraph& graph);

class ValGraphDefinitions {
  const ValGraph& graph_;

 public:
  ValGraphDefinitions(const ValGraph& graph) : graph_(graph) {}
  decltype(auto) operator()(const ValGroup& val_group) const {
    return graph_.getDefinitions(val_group);
  }
};

class ValGraphUses {
  const ValGraph& graph_;

 public:
  ValGraphUses(const ValGraph& graph) : graph_(graph) {}
  decltype(auto) operator()(const ValGroup& val_group) const {
    return graph_.getUses(val_group);
  }
};

class ValGraphInputs {
  const ValGraph& graph_;

 public:
  ValGraphInputs(const ValGraph& graph) : graph_(graph) {}
  decltype(auto) operator()(const ExprGroup& expr_group) const {
    return graph_.inputGroups(expr_group);
  }
};

class ValGraphOutputs {
  const ValGraph& graph_;

 public:
  ValGraphOutputs(const ValGraph& graph) : graph_(graph) {}
  decltype(auto) operator()(const ExprGroup& expr_group) const {
    return graph_.outputGroups(expr_group);
  }
};

class ValGraphBFS : public BFS<
                        ExprGroup,
                        ValGroup,
                        ValGraphDefinitions,
                        ValGraphUses,
                        ValGraphInputs,
                        ValGraphOutputs> {
 protected:
  ValGraphBFS(
      const ValGraph& graph,
      std::vector<NodeType> from_groups,
      std::vector<NodeType> to_groups,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined)
      : BFS(ValGraphDefinitions(graph),
            ValGraphUses(graph),
            ValGraphInputs(graph),
            ValGraphOutputs(graph),
            std::move(from_groups),
            std::move(to_groups),
            require_all_to_visited,
            allowed_direction) {}

 public:
  // Find the shortest path from the from_groups_ to to_groups_ on a
  // given graph. Dependency between vals and exprs must be satisfied.
  // It is an error if no valid path is found.
  static ExprPath getExprsBetween(
      const ValGraph& graph,
      std::vector<NodeType> from,
      std::vector<NodeType> to,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined) {
    ValGraphBFS bfs(
        graph,
        std::move(from),
        std::move(to),
        require_all_to_visited,
        allowed_direction);
    bfs.traverse();
    return bfs.getShortestExprPath();
  }
  static ExprPath getExprsBetween(
      const ValGraph& graph,
      const ValGroups& from,
      const ValGroups& to,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined) {
    return getExprsBetween(
        graph,
        std::vector<NodeType>{from.vector().begin(), from.vector().end()},
        std::vector<NodeType>{to.vector().begin(), to.vector().end()},
        require_all_to_visited,
        allowed_direction);
  }

  // Get all the val groups in vals that are reachable from the from groups
  static ValGroups getReachableValsFrom(
      const ValGraph& graph,
      const ValGroups& from,
      const ValGroups& vals,
      Direction allowed_direction = Direction::Undefined);

  // Given `from`, project it to `to`. This function will return a subset of
  // `to` that is connected to `from`.
  static std::unordered_set<ValGroup> projectTo(
      const ValGraph& id_graph,
      const ValGroup& from,
      const ValGroups& to,
      Direction allowed_direction = Direction::Undefined);
};

inline std::vector<ValGroup> inputGroups(
    const ValGraph& graph,
    const ExprGroup& expr,
    Direction dir) {
  return dir == Direction::Forward ? graph.inputGroups(expr)
                                   : graph.outputGroups(expr);
}

inline std::vector<ValGroup> outputGroups(
    const ValGraph& graph,
    const ExprGroup& expr,
    Direction dir) {
  return dir == Direction::Forward ? graph.outputGroups(expr)
                                   : graph.inputGroups(expr);
}

inline ValGroups getInputsOfExprPath(
    const ValGraph& graph,
    const ValGraphBFS::ExprPath& path) {
  ValGroups inputs;
  std::unordered_set<ValGroup> all_outputs;

  for (const auto& [expr_g, dir] : path) {
    for (const auto& inp : inputGroups(graph, expr_g, dir)) {
      if (all_outputs.find(inp) == all_outputs.end()) {
        inputs.pushBack(inp);
      }
    }
    for (const auto& out : outputGroups(graph, expr_g, dir)) {
      all_outputs.emplace(out);
    }
  }

  return inputs;
}

inline ValGroups getOutputsOfExprPath(
    const ValGraph& graph,
    const ValGraphBFS::ExprPath& path) {
  return getInputsOfExprPath(graph, reverse(path));
}

} // namespace nvfuser
