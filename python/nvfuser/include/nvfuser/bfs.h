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
    NVF_THROW();
  }
}

// Gives the corresponding Val type (e.g., Val* for Expr* and ValGroup for
// ExprGroup)
template <typename ExprT>
struct GetValType;

template <typename ExprT, typename InputsT, typename OutputsT>
std::vector<typename GetValType<ExprT>::type> getInputsOfExpr(
    const ExprT& expr,
    Direction dir,
    InputsT inputs,
    OutputsT outputs) {
  NVF_ERROR(dir == Direction::Forward || dir == Direction::Backward);
  return dir == Direction::Forward ? inputs(expr) : outputs(expr);
}

template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getInputsOfExpr(
    const typename BFSType::ExprType& expr,
    Direction dir,
    const AdditionalArgs&... additional_args) {
  return getInputsOfExpr(
      expr,
      dir,
      typename BFSType::InputsType(additional_args...),
      typename BFSType::OutputsType(additional_args...));
}

template <typename ExprT, typename InputsT, typename OutputsT>
std::vector<typename GetValType<ExprT>::type> getOutputsOfExpr(
    const ExprT& expr,
    Direction dir,
    InputsT inputs,
    OutputsT outputs) {
  return getInputsOfExpr(expr, reverse(dir), inputs, outputs);
}

template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getOutputsOfExpr(
    const typename BFSType::ExprType& expr,
    Direction dir,
    const AdditionalArgs&... additional_args) {
  return getInputsOfExpr<BFSType>(expr, reverse(dir), additional_args...);
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
  using ExprType = ExprT;
  using ValType = ValT;
  using NodeType = std::variant<ExprT, ValT>;
  using ExprPath = std::vector<std::pair<ExprT, Direction>>;
  using InputsType = InputsT;
  using OutputsType = OutputsT;

  virtual ~BFS() = default;

 public:
  BFS(DefinitionT definition,
      UsesT uses,
      InputsT inputs,
      OutputsT outputs,
      std::vector<NodeType> from,
      std::vector<NodeType> to,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined)
      : definition_(std::move(definition)),
        uses_(std::move(uses)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        from_(std::move(from)),
        to_(std::move(to)),
        require_all_to_visited_(require_all_to_visited),
        allowed_direction_(allowed_direction) {}

  // Traverse from from_ to to_, recording each taken
  // path to generate the shortest path after the travesal
  virtual void traverse() {
    for (const auto& n : from_) {
      setVisited(n);
      addNewNeighbors(n);
    }

    while (!allToNodesVisited()) {
      bool something_was_processed = false;
      std::deque<NodeType> not_ready;
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
          not_ready.emplace_back(n);
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
      to_visit_.insert(to_visit_.end(), not_ready.begin(), not_ready.end());
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
      ss << " (from: ";
      for (const auto& from : from_) {
        ss << " " << toString(from);
        if (const ExprT* e = std::get_if<ExprT>(&from)) {
          ss << " " << toString(*e);
        }
      }
      ss << ")";
      ss << ", visited: (";
      for (const auto& visited : visited_) {
        if (const ValT* v = std::get_if<ValT>(&visited)) {
          ss << " " << toString(visited);
        }
      }
      ss << ")";
      NVF_THROW("BFS traversal could not visit some nodes: ", ss.str());
    }
  }

  // Find the shortest path from the from_ to to_. A boolean value
  // indicating if all nodes are visited is also returned. This
  // must be only used once traversal is completed.
  virtual std::pair<ExprPath, bool> getShortestExprPath() {
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

    return std::make_pair(unique_path.vector(), allToNodesVisited());
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
      NVF_THROW();
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
    if (!inputs.empty() && allowed_direction_ != Direction::Backward &&
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
    if (!outputs.empty() && allowed_direction_ != Direction::Forward &&
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
      if (allowed_direction_ == Direction::Backward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& v : inputs_(*e)) {
          add_to_visit_list(v);
        }
      }
      if (allowed_direction_ == Direction::Forward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& v : outputs_(*e)) {
          add_to_visit_list(v);
        }
      }
    } else if (const ValT* v = std::get_if<ValT>(&node)) {
      if (allowed_direction_ == Direction::Forward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& e : uses_(*v)) {
          add_to_visit_list(e);
        }
      }
      if (allowed_direction_ == Direction::Backward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& e : definition_(*v)) {
          add_to_visit_list(e);
        }
      }
    } else {
      NVF_THROW();
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
  Direction allowed_direction_ = Direction::Undefined;
};

// Unlike the default BFS behavior, Expr is considered ready to
// visit as long as one of the inputs or outputs has any of its dependencies met
template <
    typename ExprT,
    typename ValT,
    typename DefinitionT,
    typename UsesT,
    typename InputsT,
    typename OutputsT>
class BFSWithPermissiveDependence
    : public BFS<ExprT, ValT, DefinitionT, UsesT, InputsT, OutputsT> {
 public:
  using BFSBaseType = BFS<ExprT, ValT, DefinitionT, UsesT, InputsT, OutputsT>;
  using NodeType = typename BFSBaseType::NodeType;

  BFSWithPermissiveDependence(
      DefinitionT definition,
      UsesT uses,
      InputsT inputs,
      OutputsT outputs,
      std::vector<NodeType> from,
      std::vector<NodeType> to,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined)
      : BFSBaseType(
            definition,
            uses,
            inputs,
            outputs,
            std::move(from),
            std::move(to),
            require_all_to_visited,
            allowed_direction) {}

  std::optional<std::pair<Direction, std::vector<NodeType>>> isReady(
      const ExprT& expr) const override {
    // Either any inputs or any outputs must have been visited
    decltype(auto) inputs = this->inputs_(expr);
    if (!inputs.empty() && this->allowed_direction_ != Direction::Backward &&
        std::any_of(
            inputs.begin(), inputs.end(), [&](const ValT& input) -> bool {
              return this->isDependencySatisfied(input);
            })) {
      std::vector<NodeType> prev_nodes;
      std::copy_if(
          inputs.begin(),
          inputs.end(),
          std::back_inserter(prev_nodes),
          [&](const ValT& input) -> bool { return this->isVisited(input); });
      return std::make_pair(Direction::Forward, prev_nodes);
    }

    decltype(auto) outputs = this->outputs_(expr);
    if (!outputs.empty() && this->allowed_direction_ != Direction::Forward &&
        std::any_of(
            outputs.begin(), outputs.end(), [&](const ValT& output) -> bool {
              return this->isDependencySatisfied(output);
            })) {
      std::vector<NodeType> prev_nodes;
      std::copy_if(
          outputs.begin(),
          outputs.end(),
          std::back_inserter(prev_nodes),
          [&](const ValT& output) -> bool { return this->isVisited(output); });
      return std::make_pair(Direction::Backward, prev_nodes);
    }
    return std::nullopt;
  }

  // When adding new neighbors of an expr node, if any of inputs is
  // the previous node of this expr, then don't add the remaining
  // inputs to the to-visit list. Similary, if any of the outputs is
  // the previous node of this expr, don't add the remaining
  // outputs. See BFSTest.IRBFSPermissiveTraversal2 for a concrete
  // example.
  void addNewNeighbors(const NodeType& node) override {
    const ExprT* e = std::get_if<ExprT>(&node);
    if (e == nullptr) {
      BFSBaseType::addNewNeighbors(node);
      return;
    }

    auto add_to_visit_list = [&](const NodeType& n) -> void {
      if (this->isVisited(n) || this->excludeFromTraversal(n)) {
        return;
      }
      this->to_visit_.emplace_back(n);
    };

    auto prev_nodes_it = this->prev_nodes_.find(node);

    auto is_any_already_visited = [&](const auto& inputs_or_outputs) -> bool {
      if (prev_nodes_it == this->prev_nodes_.end()) {
        return false;
      }

      const std::vector<NodeType>& prev_nodes = prev_nodes_it->second.second;

      return std::any_of(
          inputs_or_outputs.begin(),
          inputs_or_outputs.end(),
          [&](const auto& input_or_output) {
            return std::find(
                       prev_nodes.begin(),
                       prev_nodes.end(),
                       NodeType(input_or_output)) != prev_nodes.end();
          });
    };

    if (this->allowed_direction_ == Direction::Backward ||
        this->allowed_direction_ == Direction::Undefined) {
      // There's an input node that is marked as a previous node of
      // this node. Since this is permissive traversal, some of the
      // other inputs may not be visited yet, but going back to
      // the input nodes doesn't seem to make sense
      auto input_nodes = this->inputs_(*e);
      if (!is_any_already_visited(input_nodes)) {
        for (const auto& v : input_nodes) {
          add_to_visit_list(v);
        }
      }
    }
    if (this->allowed_direction_ == Direction::Forward ||
        this->allowed_direction_ == Direction::Undefined) {
      auto output_nodes = this->outputs_(*e);
      if (!is_any_already_visited(output_nodes)) {
        for (const auto& v : output_nodes) {
          add_to_visit_list(v);
        }
      }
    }
  }
};

// Unlike the default BFS behavior, Val is considered ready to
// visit only if all of definitions or uses are visited. The default
// BFS only requires one definition is visited.
template <
    typename ExprT,
    typename ValT,
    typename DefinitionT,
    typename UsesT,
    typename InputsT,
    typename OutputsT>
class BFSWithStrictDependence
    : public BFS<ExprT, ValT, DefinitionT, UsesT, InputsT, OutputsT> {
 public:
  using NodeType =
      typename BFS<ExprT, ValT, DefinitionT, UsesT, InputsT, OutputsT>::
          NodeType;

  BFSWithStrictDependence(
      DefinitionT definition,
      UsesT uses,
      InputsT inputs,
      OutputsT outputs,
      std::vector<NodeType> from,
      std::vector<NodeType> to,
      bool require_all_to_visited = true,
      Direction allowed_direction = Direction::Undefined)
      : BFS<ExprT, ValT, DefinitionT, UsesT, InputsT, OutputsT>(
            definition,
            uses,
            inputs,
            outputs,
            std::move(from),
            std::move(to),
            require_all_to_visited,
            allowed_direction) {}

  std::optional<std::pair<Direction, std::vector<NodeType>>> isReady(
      const ValT& v) const override {
    decltype(auto) uses = this->uses_(v);
    if (!uses.empty() &&
        std::all_of(uses.begin(), uses.end(), [&](const ExprT& use_e) -> bool {
          return this->isDependencySatisfied(use_e);
        })) {
      return std::make_pair(
          Direction::Backward, std::vector<NodeType>{uses.begin(), uses.end()});
    }
    decltype(auto) def = this->definition_(v);
    if (!def.empty() &&
        std::all_of(def.begin(), def.end(), [&](const ExprT& def_e) -> bool {
          return this->isDependencySatisfied(def_e);
        })) {
      return std::make_pair(
          Direction::Forward, std::vector<NodeType>{def.begin(), def.end()});
    }

    return std::nullopt;
  }
};

// Find the shortest path from the from vals to the to
// vals. Dependency between vals and exprs must be satisfied.
// It is an error if no valid path is found unless
// require_all_to_visited is false.
template <typename BFSType, typename... AdditionalArgs>
static std::pair<typename BFSType::ExprPath, bool> getExprsBetween(
    const std::vector<typename BFSType::ValType>& from,
    const std::vector<typename BFSType::ValType>& to,
    bool require_all_to_visited = true,
    Direction allowed_direction = Direction::Undefined,
    const AdditionalArgs&... additional_args) {
  BFSType bfs(
      additional_args...,
      {from.begin(), from.end()},
      {to.begin(), to.end()},
      require_all_to_visited,
      allowed_direction);
  bfs.traverse();
  return bfs.getShortestExprPath();
}

template <typename ExprT, typename InputsT, typename OutputsT>
std::vector<typename GetValType<ExprT>::type> getInputsOfExprPath(
    const std::vector<std::pair<ExprT, Direction>>& path,
    InputsT get_inputs,
    OutputsT get_outputs) {
  using ValT = typename GetValType<ExprT>::type;
  std::vector<ValT> inputs;
  std::unordered_set<ValT> all_outputs;

  for (const auto& [expr, dir] : path) {
    for (const auto& inp :
         getInputsOfExpr(expr, dir, get_inputs, get_outputs)) {
      if (all_outputs.find(inp) == all_outputs.end()) {
        inputs.push_back(inp);
      }
    }
    for (const auto& out :
         getOutputsOfExpr(expr, dir, get_inputs, get_outputs)) {
      all_outputs.emplace(out);
    }
  }

  return inputs;
}

template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getInputsOfExprPath(
    const typename BFSType::ExprPath& path,
    const AdditionalArgs&... additional_args) {
  using ValT = typename BFSType::ValType;
  std::vector<ValT> inputs;
  std::unordered_set<ValT> all_outputs;

  for (const auto& [expr, dir] : path) {
    for (const auto& inp :
         getInputsOfExpr<BFSType>(expr, dir, additional_args...)) {
      if (all_outputs.find(inp) == all_outputs.end()) {
        inputs.push_back(inp);
      }
    }
    for (const auto& out :
         getOutputsOfExpr<BFSType>(expr, dir, additional_args...)) {
      all_outputs.emplace(out);
    }
  }

  return inputs;
}

template <typename ExprT, typename InputsT, typename OutputsT>
std::vector<typename GetValType<ExprT>::type> getOutputsOfExprPath(
    const std::vector<std::pair<ExprT, Direction>>& path,
    InputsT get_inputs,
    OutputsT get_outputs) {
  return getInputsOfExprPath(reverse(path), get_inputs, get_outputs);
}

template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getOutputsOfExprPath(
    const typename BFSType::ExprPath& path,
    const AdditionalArgs&... additional_args) {
  return getInputsOfExprPath<BFSType>(reverse(path), additional_args...);
}

// Given a set of exprs and vals, get all reachable ones from another set of
// nodes
template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::NodeType> getReachableNodesFrom(
    const std::vector<typename BFSType::NodeType>& from,
    const std::vector<typename BFSType::NodeType>& nodes,
    Direction allowed_direction = Direction::Undefined,
    const AdditionalArgs&... additional_args) {
  BFSType bfs(
      additional_args...,
      from,
      nodes,
      /*require_all_to_visited=*/false,
      allowed_direction);

  bfs.traverse();

  std::vector<typename BFSType::NodeType> reachable_nodes;
  for (const auto& node : nodes) {
    if (bfs.isVisited(node) ||
        std::find(from.begin(), from.end(), node) != from.end()) {
      reachable_nodes.push_back(node);
    }
  }

  return reachable_nodes;
}

// Shortcut of getReachableNodesFrom for Vals
template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getReachableValsFrom(
    const std::vector<typename BFSType::ValType>& from,
    const std::vector<typename BFSType::ValType>& vals,
    Direction allowed_direction = Direction::Undefined,
    const AdditionalArgs&... additional_args) {
  auto reachable_nodes = getReachableNodesFrom<BFSType, AdditionalArgs...>(
      {from.begin(), from.end()},
      {vals.begin(), vals.end()},
      allowed_direction,
      additional_args...);

  std::vector<typename BFSType::ValType> reachable_vals;
  reachable_vals.reserve(reachable_nodes.size());
  std::transform(
      reachable_nodes.begin(),
      reachable_nodes.end(),
      std::back_inserter(reachable_vals),
      [](const auto& node) {
        return std::get<typename BFSType::ValType>(node);
      });

  return reachable_vals;
}

// Traverse from a given set of vals to another set of vals and
// return all vals between them. Note that if none of the Vals in the
// second set is reachable, nothing will be returned. For example,
// if a forward Merge needs to be traversed to get to the target Val
// set, both of the two inputs must be given or reachable from the
// given starting Val set.
//
// NOTE: getValsBetween(from, to) != getValsBetween(to, from). For
// example, suppose from={i0}, to={i2}, and merge(i0, i1) =
// i2. Since i1 is missing, nothing will be returned. However, if
// from={i2} and to={i0}, then the backward merge can be traversed
// as its sole input is available, so {i0} would be returned.
template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getValsBetween(
    const std::vector<typename BFSType::ValType>& from,
    const std::vector<typename BFSType::ValType>& to,
    const AdditionalArgs&... additional_args) {
  using ValType = typename BFSType::ValType;
  auto path = getExprsBetween<BFSType>(
                  from,
                  to,
                  /*require_all_to_visited=*/false,
                  /*allowed_direction=*/Direction::Undefined,
                  additional_args...)
                  .first;

  VectorOfUniqueEntries<ValType> unique_vals;
  for (auto [expr, dir] : path) {
    unique_vals.pushBack(getInputsOfExpr(
        expr,
        dir,
        // This assumes get_inputs and get_outputs take the same
        // additional arguments, which is the case with
        // ValGraphBFS. Revisit if needed.
        typename BFSType::InputsType(additional_args...),
        typename BFSType::OutputsType(additional_args...)));
    unique_vals.pushBack(getOutputsOfExpr(
        expr,
        dir,
        typename BFSType::InputsType(additional_args...),
        typename BFSType::OutputsType(additional_args...)));
  }

  // If a val in from is found in to, just copy it to the returned val
  // set since there's no corresponding expr.
  for (const auto& from_val : from) {
    if (std::find(to.begin(), to.end(), from_val) != to.end()) {
      unique_vals.pushBack(from_val);
    }
  }

  return unique_vals.vector();
}

// Get all dependencies of to in from.
template <typename BFSType, typename... AdditionalArgs>
std::vector<typename BFSType::ValType> getDependenciesTo(
    const std::vector<typename BFSType::ValType>& vals,
    const std::vector<typename BFSType::ValType>& to) {
  using ValType = typename BFSType::ValType;
  auto path = getExprsBetween<BFSType>(
                  vals,
                  to,
                  /*require_all_to_visited=*/true,
                  /*allowed_direction=*/Direction::Undefined)
                  .first;

  VectorOfUniqueEntries<ValType> unique_vals;

  std::unordered_set<ValType> val_set{vals.begin(), vals.end()};

  for (const auto& [expr, direction] : path) {
    auto inputs =
        (direction == Direction::Forward) ? expr->inputs() : expr->outputs();
    for (auto val : inputs) {
      if (val_set.find(val) != val_set.end()) {
        unique_vals.pushBack(val);
      }
    }
  }

  return unique_vals.vector();
}

// Given `from`, project it to `to`. This function will return a subset of
// `to` that is connected to `from`.
template <typename BFSType, typename... AdditionalArgs>
std::unordered_set<typename BFSType::ValType> projectTo(
    const typename BFSType::ValType& from,
    const std::vector<typename BFSType::ValType>& to,
    Direction allowed_direction = Direction::Undefined,
    const AdditionalArgs&... additional_args) {
  using ValType = typename BFSType::ValType;
  std::unordered_set<ValType> projection{from};
  // Reverse order
  auto exprs = getExprsBetween<BFSType>(
                   {to},
                   {from},
                   /*require_all_to_visited=*/false,
                   allowed_direction,
                   additional_args...)
                   .first;
  while (!exprs.empty()) {
    const auto& [expr, direction] = exprs.back();
    exprs.pop_back();
    auto from = getOutputsOfExpr(
        expr,
        direction,
        typename BFSType::InputsType(additional_args...),
        typename BFSType::OutputsType(additional_args...));
    auto to = getInputsOfExpr(
        expr,
        direction,
        typename BFSType::InputsType(additional_args...),
        typename BFSType::OutputsType(additional_args...));

    for (const auto& g : from) {
      if (projection.count(g)) {
        projection.erase(g);
        projection.insert(to.begin(), to.end());
      }
    }
  }
  // Remove items that are not in `to`. This could happen if `from` is not
  // connected to `to`.
  for (auto it = projection.begin(); it != projection.end();) {
    if (std::find(to.begin(), to.end(), *it) == to.end()) {
      it = projection.erase(it);
    } else {
      ++it;
    }
  }
  return projection;
}

} // namespace nvfuser
