// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <bfs.h>

namespace nvfuser {

template <
    typename ExprT,
    typename ValT,
    typename DefinitionT,
    typename UsesT,
    typename InputsT,
    typename OutputsT>
class FindAllPaths {
 public:
  using ExprType = ExprT;
  using ValType = ValT;
  using NodeType = std::variant<ExprT, ValT>;
  using ExprPath = std::vector<std::pair<ExprT, Direction>>;
  using InputsType = InputsT;
  using OutputsType = OutputsT;
  
  FindAllPaths(DefinitionT definition,
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

  virtual VectorOfUniqueEntries<ExprPath> get() {
    std::deque<NodeType>

    
    return VectorOfUniqueEntries<ExprPath>{};
  }

  // Traverse from from_ to to_, recording each taken
  // path to generate the shortest path after the travesal
  virtual void traverse() {
#if 0    
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
#endif
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
  bool require_all_to_visited_ = true;
  Direction allowed_direction_ = Direction::Undefined;
};


} // namespace nvfuser
