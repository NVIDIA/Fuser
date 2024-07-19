// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <exceptions.h>

#include <algorithm>
#include <deque>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace nvfuser {

enum class Direction { Forward, Backward, Undefined };

template <typename ExprT>
using ExprPath = std::vector<std::pair<ExprT, Direction>>;

} // namespace nvfuser

namespace std {
template <typename ExprT>
struct hash<pair<ExprT, nvfuser::Direction>> {
  std::size_t operator()(
      const std::pair<ExprT, nvfuser::Direction>& directed_expr) const {
    using std::hash;
    return hash<ExprT>()(directed_expr.first);
  }
};
} // namespace std

namespace nvfuser {

inline std::ostream& operator<<(std::ostream& os, const Direction direction) {
  switch (direction) {
    case Direction::Forward:
      os << "Forward";
      break;
    case Direction::Backward:
      os << "Backward";
      break;
    case Direction::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

template <typename ExprT>
std::ostream& operator<<(std::ostream& os, const ExprPath<ExprT>& path) {
  for (const auto& [expr, direction] : path) {
    os << direction << " " << toString(expr);
  }
  return os;
}

inline Direction reverse(Direction direction) {
  if (direction == Direction::Forward) {
    return Direction::Backward;
  } else if (direction == Direction::Backward) {
    return Direction::Forward;
  } else {
    return Direction::Undefined;
  }
}

template <typename ExprT>
inline ExprPath<ExprT> reverse(const ExprPath<ExprT>& path) {
  auto rev = path;
  std::reverse(rev.begin(), rev.end());
  for (auto& [e, direction] : rev) {
    direction = reverse(direction);
  }
  return rev;
}

template <typename ExprT, typename ValT>
inline std::string toString(const std::variant<ExprT, ValT>& n) {
  if (auto e = std::get_if<ExprT>(&n)) {
    return toString(*e);
  } else if (auto v = std::get_if<ValT>(&n)) {
    return toString(*v);
  } else {
    NVF_ERROR(false);
  }
}

// Traversal for finding the shortest path from given vals to another
// vals. For now, the vals are either Val* if we want to traverse IR nodes,
// or ValGroup if we want to traverse ValGraph. However, this algorithm is
// implement as a class template so in the future, we can extend it to support
// other types of vals and exprs. The algorithm is based on the standard BFS
// traversal, however, the traversal graph is treated as an undirected graph, so
// the traversal direction can be both forward and backward. The dependencies of
// vals and exprs need to be satisfied. Specifically, when visiting an expr,
// either its inputs or outputs must be visited before. Similarly, when visiting
// a val, there must be at least one defining expr or one use expr that is
// already visited.
template <
    typename ExprT,
    typename ValT,
    typename DefinitionT,
    typename UsesT,
    typename InputsT,
    typename OutputsT>
class BFS {
 public:
  using NodeType = std::variant<ExprT, ValT>;
  using ExprPath = std::vector<std::pair<ExprT, Direction>>;

  virtual ~BFS() = default;

 protected:
  BFS(DefinitionT definition,
      UsesT uses,
      InputsT inputs,
      OutputsT outputs,
      std::vector<NodeType> from,
      std::vector<NodeType> to,
      bool require_all_to_visited = true)
      : definition_(std::move(definition)),
        uses_(std::move(uses)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        from_(std::move(from)),
        to_(std::move(to)),
        require_all_to_visited_(require_all_to_visited) {}

  // Traverse from from_ to to_, recording each taken
  // path to generate the shortest path after the travesal
  virtual void traverse() {
    for (const auto& n : from_) {
      setVisited(n);
      addNewNeighbors(n);
    }

    while (!allToNodesVisited()) {
      bool something_was_processed = false;
      std::deque<NodeType> not_ready_;
      while (!allToNodesVisited() && !to_visit_.empty()) {
        const auto n = to_visit_.front();
        to_visit_.pop_front();

        if (isVisited(n)) {
          continue;
        }

        auto ready_direction = isReady(n);
        if (!ready_direction.has_value()) {
          // To stop an infinite loop, the not-ready node is not moved
          // back to the to_visit_ queue but kept in the separate
          // queue. This way, if all nodes in to_visit_ are not ready,
          // the queue would eventually become empty, which would then
          // break the inner while loop. The something_was_processed
          // flag is used to remember if there's any progress.
          not_ready_.emplace_back(n);
          continue;
        }

        // Visit this node and add its neighbors to to_visit if not
        // visited yet
        setVisited(n);
        setPrevGroups(n, *ready_direction);
        addNewNeighbors(n);
        something_was_processed = true;
      }

      // If nothing was processed, break out of the loop
      if (!something_was_processed) {
        break;
      }

      // Something was processed. Redo the traversal.
      to_visit_.insert(to_visit_.end(), not_ready_.begin(), not_ready_.end());
    }

    if (require_all_to_visited_ && !allToNodesVisited()) {
      std::stringstream ss;
      for (const auto& to : to_) {
        if (!isVisited(to)) {
          ss << " " << toString(to);
          if (const ExprT* e = std::get_if<ExprT>(&to)) {
            ss << " " << toString(*e);
          }
        }
      }
      NVF_ERROR(false, "BFS traversal could not visit some nodes: ", ss.str());
    }
  }

  // Find the shortest path from the from_ to to_. This
  // must be only used once traversal is completed.
  virtual ExprPath getShortestExprPath() {
    NVF_ERROR(
        !require_all_to_visited_ || allToNodesVisited(),
        "Traveral is either not done or failed");

    ExprPath path;

    std::deque<std::pair<NodeType, Direction>> to_visit;
    for (const NodeType& to : to_) {
      to_visit.emplace_back(to, Direction::Undefined);
    }

    while (!to_visit.empty()) {
      const auto [node, direction] = to_visit.front();
      to_visit.pop_front();

      if (const ExprT* e = std::get_if<ExprT>(&node)) {
        path.emplace_back(*e, direction);
      }

      if (std::find(from_.begin(), from_.end(), node) != from_.end()) {
        continue;
      }

      auto prev_nodes_it = prev_nodes_.find(node);
      NVF_ERROR(!require_all_to_visited_ || prev_nodes_it != prev_nodes_.end());
      if (prev_nodes_it != prev_nodes_.end()) {
        const Direction dir = prev_nodes_it->second.first;
        for (const auto& prev_node : prev_nodes_it->second.second) {
          to_visit.emplace_back(prev_node, dir);
        }
      }
    }

    // At this point, we have the reverse path, but it may have multiple exprs
    // that need to be filtered out. For example, if we are traversing
    // IterDomain transformations, let's say there are domains 0, 1 and 2, and
    // domains 1 and 2 are merged to produce domain 3, and then domains
    // 0 and 3 are merged to produce domain 4.
    //
    // 0       1         2
    //
    // |       |         |
    // |       |         |
    // |       +-->   <--+
    // |            3
    // |            |
    // |            |
    // +----> 4 <---+
    //
    // Suppose we want to find the shortest path from {4} to {0, 1,
    // 2}. The correct answer should be:
    //
    //   Backward merge of 0, 3 -> 4
    //   Backward merge of 1, 2 -> 3
    //
    // However, the above traversal would produce a path of:
    //
    //   Backward merge of 0, 3 -> 4
    //   Backward merge of 1, 2 -> 3
    //   Backward merge of 1, 2 -> 3
    //   Backward merge of 0, 3 -> 4
    //
    // This is because, since nodes 0, 1 and 2 are the starting nodes,
    // we would first visit 4 from 0, and then 3 from 1 and again 3 from
    // 2. Since node 3 would be visited twice, the path from 3 to 4
    // would be traversed twice as well. Obviously, just reversing this
    // path wouldn't give the correct path. There are two issues here:
    //
    // - The first visit to node 4 from node 0 should not be taken since
    //   node 4 must appear after node 3
    // - Visiting the same node multiple times is redundant and should
    //   be removed
    //
    // Both problems could be solved by taking into considerations if
    // nodes are ready to visit and also are already visited, just like
    // done in the forward traversal. However, there's an additional
    // complexity in this case because the following graph is also valid:
    //
    //         1         2
    //
    // |       |         |
    // |       |         |
    // |       +-->   <--+
    // |            3
    // |            |
    // |            |
    // +----> 4 <---+
    //
    // Notice that node 0 is missing, meaning the shortest path problem
    // in this case is  from node 4 to nodes 1 and 2, and node 0 is not
    // of interest. The correct path is still the same, i.e., first
    // backward merge of 0 and 3 and then another backward merge of 1
    // and 2. It is just node 0 is discarded as it is not of
    // interest. In this case, however, if the
    // traversal was enforced to honor the dependency of each node,
    // it would not be able to visit the backward merge of 0 and 3 as
    // node 0 is missing.
    //
    // A straightforward solution here is simply first generating the
    // path as above and for each node, take the last visit only. Note
    // that the last visit is always guaranteed to satisfy its
    // dependencies.
    //
    // Recall that the final path needs to be reversed, so instead of
    // finding the last appearance of each node, the final path can be
    // obtained by first reversing the current path and then only taking
    // the first appearance of each expr. Or, more simply, we can
    // just use VectorOfUniqueEntries with the reverse iterator.
    //
    // See the BFS2 test for a concrete example.

    VectorOfUniqueEntries<std::pair<ExprT, Direction>> unique_path(
        path.rbegin(), path.rend());

    return unique_path.vector();
  }

  // Check if a node is ready to visit. If yes, return the direction
  // and the prev nodes that should be visited before the given node
  // is visited.
  virtual std::optional<std::pair<Direction, std::vector<NodeType>>> isReady(
      const NodeType& node) const {
    if (const ExprT* e = std::get_if<ExprT>(&node)) {
      return isReady(*e);
    } else if (const ValT* v = std::get_if<ValT>(&node)) {
      return isReady(*v);
    } else {
      NVF_ERROR(false);
    }
  }

  // Check if an ExprT is ready to visit. Either all of its inputs
  // or all of outputs must have their dependencies satisfied. If
  // ready because the inputs are already visited, return
  // Direction::Forward and all the input nodes. If ready because the
  // outputs are ready, return Direction::Backward and all the output nodes.
  virtual std::optional<std::pair<Direction, std::vector<NodeType>>> isReady(
      const ExprT& expr) const {
    // Either all inputs or all outputs must have been visited
    decltype(auto) inputs = inputs_(expr);
    if (!inputs.empty() &&
        std::all_of(
            inputs.begin(), inputs.end(), [&](const ValT& input) -> bool {
              return isDependencySatisfied(input);
            })) {
      std::vector<NodeType> prev_nodes;
      std::copy_if(
          inputs.begin(),
          inputs.end(),
          std::back_inserter(prev_nodes),
          [&](const ValT& input) -> bool { return isVisited(input); });
      return std::make_pair(Direction::Forward, prev_nodes);
    }

    decltype(auto) outputs = outputs_(expr);
    if (!outputs.empty() &&
        std::all_of(
            outputs.begin(), outputs.end(), [&](const ValT& output) -> bool {
              return isDependencySatisfied(output);
            })) {
      std::vector<NodeType> prev_nodes;
      std::copy_if(
          outputs.begin(),
          outputs.end(),
          std::back_inserter(prev_nodes),
          [&](const ValT& output) -> bool { return isVisited(output); });
      return std::make_pair(Direction::Backward, prev_nodes);
    }

    return std::nullopt;
  }

  // Check if a val is ready to visit. Either its defining or use
  // expr must have its dependency satisfied. If ready because
  // there's a visited defining expr, return Direction::Forward and
  // the defining expr. If ready because there's a visited use expr, return
  // Direction::Backward and the use expr.
  virtual std::optional<std::pair<Direction, std::vector<NodeType>>> isReady(
      const ValT& v) const {
    // In the case of Val, requires just one def or use expr.
    // Check if any use is visited
    decltype(auto) uses = uses_(v);
    if (!uses.empty()) {
      auto it = std::find_if(
          uses.begin(), uses.end(), [&](const ExprT& use_e) -> bool {
            return isDependencySatisfied(use_e);
          });
      if (it != uses.end()) {
        return std::make_pair(Direction::Backward, std::vector<NodeType>{*it});
      }
    }
    // Check if any def is visited
    decltype(auto) def = definition_(v);
    if (!def.empty()) {
      auto it =
          std::find_if(def.begin(), def.end(), [&](const ExprT& def_e) -> bool {
            return isDependencySatisfied(def_e);
          });
      if (it != def.end()) {
        return std::make_pair(Direction::Forward, std::vector<NodeType>{*it});
      }
    }

    return std::nullopt;
  }

  // If another node depends on a given node, check if that
  // dependency is considered satisfied. If the given node is already
  // visited, that should mean the dependency is satisfied.
  virtual bool isDependencySatisfied(const NodeType& dependency) const {
    return isVisited(dependency);
  }

  // Check if a given node is already visited
  virtual bool isVisited(const NodeType& node) const {
    return visited_.find(node) != visited_.end();
  }

  // Mark a node as visited
  virtual void setVisited(const NodeType& node) {
    visited_.emplace(node);
  }

  // Add new neighbors of a given node to the to_visit list
  virtual void addNewNeighbors(const NodeType& node) {
    auto add_to_visit_list = [&](const NodeType& n) -> void {
      if (isVisited(n) || excludeFromTraversal(n)) {
        return;
      }
      to_visit_.emplace_back(n);
    };

    if (const ExprT* e = std::get_if<ExprT>(&node)) {
      for (const auto& v : inputs_(*e)) {
        add_to_visit_list(v);
      }
      for (const auto& v : outputs_(*e)) {
        add_to_visit_list(v);
      }
    } else if (const ValT* v = std::get_if<ValT>(&node)) {
      for (const auto& e : uses_(*v)) {
        add_to_visit_list(e);
      }
      for (const auto& e : definition_(*v)) {
        add_to_visit_list(e);
      }
    } else {
      NVF_ERROR(false);
    }
  }

  // Check if all to_ are visited
  virtual bool allToNodesVisited() const {
    return std::all_of(
        to_.begin(), to_.end(), [&](const NodeType& node) -> bool {
          return isVisited(node);
        });
  };

  // Set the previous nodes of a given node that is visited in a
  // given direction
  virtual void setPrevGroups(
      const NodeType& node,
      const std::pair<Direction, std::vector<NodeType>>& prev_nodes) {
    NVF_ERROR(
        prev_nodes_.emplace(node, prev_nodes).second,
        "Previous node already set for ",
        toString(node));
  }

  // Hook to exclude certain graph nodes. See IndexingTraversal for a
  // concrete example
  virtual bool excludeFromTraversal(const NodeType& node) const {
    return false;
  }

 protected:
  const DefinitionT definition_;
  const UsesT uses_;
  const InputsT inputs_;
  const OutputsT outputs_;
  const std::vector<NodeType> from_;
  const std::vector<NodeType> to_;
  std::deque<NodeType> to_visit_;
  std::unordered_set<NodeType> visited_;
  std::unordered_map<NodeType, std::pair<Direction, std::vector<NodeType>>>
      prev_nodes_;
  bool require_all_to_visited_ = true;
};

} // namespace nvfuser
