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

  struct Edge {
    NodeType from;
    NodeType to;
    Edge(const ValT& from, const ExprT& to) : from(from), to(to) {}
    Edge(const ExprT& from, const ValT& to) : from(from), to(to) {}
    bool operator==(const Edge& other) const {
      return from == other.from && to == other.to;
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "{" << nvfuser::toString(from) << " -> " << nvfuser::toString(to)
         << "}";
      return ss.str();
    }
    Edge reverse() const {
      return Edge{to, from};
    }
  };

  struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
      return std::hash<NodeType>()(edge.from) ^ std::hash<NodeType>()(edge.to);
    }
  };

  virtual ~FindAllPaths() = default;

 public:
  FindAllPaths(
      DefinitionT definition,
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
        from_nodes_(std::move(from)),
        to_nodes_(std::move(to)),
        require_all_to_visited_(require_all_to_visited),
        allowed_direction_(allowed_direction) {}

  // Traverse from from_ to to_, recording each taken
  // path to generate the shortest path after the travesal
  virtual void traverse() {
    for (const auto& from_node : from_nodes_) {
      if (const ValT* from_val = std::get_if<ValT>(&from_node)) {
        for (const auto& use_expr : uses_(*from_val)) {
          Edge e(*from_val, use_expr);
          setVisited(e);
          addNewNeighbors(e);
        }
        for (const auto& def_expr : definition_(*from_val)) {
          Edge e(*from_val, def_expr);
          setVisited(e);
          addNewNeighbors(e);
        }
      } else {
        NVF_THROW(
            "Traversal from nodes are assumed to be all Vals but found: ",
            toString(from_node));
      }
    }

    bool something_was_processed = true;
    while (something_was_processed) {
      std::cerr << "Something was progressed\n";
      std::deque<Edge> not_ready;
      something_was_processed = false;

      while (!to_visit_.empty()) {
        const auto edge_to_visit = to_visit_.front();
        to_visit_.pop_front();

        std::cerr << "Next edge: " << edge_to_visit.toString() << "\n";

        // Don't visit edges multiple times even when traversing all paths
        if (isVisited(edge_to_visit)) {
          std::cerr << "Already visited\n";
          continue;
        }

        // std::vector<std::pair<Direction, std::vector<NodeType>>>
        auto prev_edges = isReady(edge_to_visit);
        if (!prev_edges.has_value()) {
          // To stop an infinite loop, the not-ready node is not moved
          // back to the to_visit_ queue but kept in the separate
          // queue. This way, if all nodes in to_visit_ are not ready,
          // the queue would eventually become empty, which would then
          // break the inner while loop. The something_was_processed
          // flag is used to remember if there's any progress.
          not_ready.emplace_back(edge_to_visit);
          std::cerr << "Not ready\n";
          continue;
        }

        std::cerr << "Visiting " << edge_to_visit.toString() << "\n";

        // Visit this node and add its neighbors to to_visit if not
        // visited yet
        setVisited(edge_to_visit);
        setPrevEdges(edge_to_visit, *prev_edges);
        // TODO: update the edges from the to node by adding this edge
        // to their prev sets
        addNewNeighbors(edge_to_visit);
        something_was_processed = true;
      }

      // Something was processed. Redo the traversal.
      to_visit_.insert(to_visit_.end(), not_ready.begin(), not_ready.end());
    }

    if (require_all_to_visited_ && !allToNodesVisited()) {
      auto visited_nodes = getVisitedNodes();
      std::stringstream ss;
      for (const auto& to : to_nodes_) {
        if (!visited_nodes.count(to)) {
          ss << " " << toString(to);
        }
      }
      ss << " (from: ";
      for (const auto& from : from_nodes_) {
        ss << " " << toString(from);
      }
      ss << ")";
      ss << ", visited: (";
      for (const auto& visited : visited_nodes) {
        if (const ValT* v = std::get_if<ValT>(&visited)) {
          ss << " " << toString(visited);
        }
      }
      ss << ")";
      NVF_THROW("BFS traversal could not visit some nodes: ", ss.str());
    }

    std::cerr << "Traversal done\n";
  }

  // Check if a node is ready to visit. If yes, return the direction
  // and the prev nodes that should be visited before the given node
  // is visited.
  virtual std::optional<std::vector<Edge>> isReady(const Edge& edge) const {
    Direction dir = getDirection(edge);
    if ((dir == Direction::Forward &&
         allowed_direction_ == Direction::Backward) ||
        (dir == Direction::Backward &&
         allowed_direction_ == Direction::Forward)) {
      return std::nullopt;
    }

    if (const ExprT* e = std::get_if<ExprT>(&(edge.from))) {
      return isReady(*e, std::get<ValT>(edge.to), dir);
    } else if (const ValT* v = std::get_if<ValT>(&(edge.from))) {
      return isReady(*v, std::get<ExprT>(edge.to), dir);
    } else {
      NVF_THROW();
    }
  }

  // Check if an ExprT is ready to visit. Either all of its inputs
  // or all of outputs must have their dependencies satisfied. If
  // ready because the inputs are already visited, return
  // Direction::Forward and all the input nodes. If ready because the
  // outputs are ready, return Direction::Backward and all the output nodes.
  virtual std::optional<std::vector<Edge>> isReady(
      const ExprT& from_expr,
      const ValT& to_val,
      Direction dir) const {
    if (dir == Direction::Forward) {
      decltype(auto) inputs = inputs_(from_expr);
      if (std::all_of(
              inputs.begin(), inputs.end(), [&](const ValT& input) -> bool {
                return isDependencySatisfied(Edge(input, from_expr));
              })) {
        std::vector<Edge> prev_edges;
        for (const ValT& input : inputs) {
          prev_edges.push_back(Edge(input, from_expr));
        }
        return prev_edges;
      }
    } else if (dir == Direction::Backward) {
      decltype(auto) outputs = outputs_(from_expr);
      if (std::all_of(
              outputs.begin(), outputs.end(), [&](const ValT& output) -> bool {
                return isDependencySatisfied(Edge(output, from_expr));
              })) {
        std::vector<Edge> prev_edges;
        for (const ValT& output : outputs) {
          prev_edges.push_back(Edge(output, from_expr));
        }
        return prev_edges;
      }
    }

    return std::nullopt;
  }

  // Check if a val is ready to visit. Either its defining or use
  // expr must have its dependency satisfied. If ready because
  // there's a visited defining expr, return Direction::Forward and
  // the defining expr. If ready because there's a visited use expr, return
  // Direction::Backward and the use expr.
  virtual std::optional<std::vector<Edge>> isReady(
      const ValT& from_val,
      const ExprT& to_expr,
      Direction dir) const {
    // In the case of Val, requires just one def or use expr.
    // Check if any use is visited

    std::vector<Edge> prev_edges;

    // Check if any def is visited
    decltype(auto) def = definition_(from_val);
    if (!def.empty()) {
      for (const ExprT& def_e : def) {
        if (def_e != to_expr && isDependencySatisfied(Edge(def_e, from_val))) {
          prev_edges.emplace_back(Edge(def_e, from_val));
        }
      }
    }

    decltype(auto) uses = uses_(from_val);
    for (const ExprT& use_e : uses) {
      if (use_e != to_expr && isDependencySatisfied(Edge(use_e, from_val))) {
        prev_edges.emplace_back(Edge(use_e, from_val));
      }
    }

    return prev_edges.empty() ? std::nullopt : std::make_optional(prev_edges);
  }

  // If another node depends on a given node, check if that
  // dependency is considered satisfied. If the given node is already
  // visited, that should mean the dependency is satisfied.
  virtual bool isDependencySatisfied(const Edge& edge) const {
    return isVisited(edge);
  }

  // Check if a given node is already visited
  virtual bool isVisited(const Edge& edge) const {
    return visited_.find(edge) != visited_.end();
  }

  virtual void setVisited(const Edge& edge) {
    visited_.emplace(edge);
  }

  // Add new neighbors of a given node to the to_visit list
  // const std::vector<std::pair<Direction, std::vector<NodeType>>>& prev_nodes)
  // {
  virtual void addNewNeighbors(const Edge& edge) {
    // TODO: Change the signature to receipt edge?
    auto add_to_visit_list = [&](const auto& from, const auto& to) -> void {
      Edge neighbor_edge(from, to);
      // Don't traverse back
      if (edge.from == neighbor_edge.to && edge.to == neighbor_edge.from) {
        return;
      }
      addToToVisitList(neighbor_edge);
      std::cerr << "Added to new neighbor: " << neighbor_edge.toString()
                << "\n";
    };

    Direction edge_dir = getDirection(edge);

    if (const ExprT* e = std::get_if<ExprT>(&edge.to)) {
      if (edge_dir == Direction::Forward) {
        for (const auto& v : outputs_(*e)) {
          add_to_visit_list(*e, v);
        }
      } else if (edge_dir == Direction::Backward) {
        for (const auto& v : inputs_(*e)) {
          add_to_visit_list(*e, v);
        }
      } else {
        NVF_THROW();
      }
    } else if (const ValT* v = std::get_if<ValT>(&edge.to)) {
      // In the case of Val, no matter what direction this node is, it
      // should be valid to traverse both directions. Just don't
      // traverse back to the same node
      if (allowed_direction_ == Direction::Forward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& e : uses_(*v)) {
          add_to_visit_list(*v, e);
        }
      }
      if (allowed_direction_ == Direction::Backward ||
          allowed_direction_ == Direction::Undefined) {
        for (const auto& e : definition_(*v)) {
          add_to_visit_list(*v, e);
        }
      }
    } else {
      NVF_THROW();
    }
  }

  // Check if all to_ are visited
  virtual bool allToNodesVisited() const {
    auto visited_nodes = getVisitedNodes();
    return std::all_of(
        to_nodes_.begin(), to_nodes_.end(), [&](const NodeType& node) -> bool {
          return visited_nodes.count(node);
        });
  };

  // Set the previous nodes of a given node that is visited in a
  // given direction
  virtual void setPrevEdges(
      const Edge& edge,
      const std::vector<Edge>& prev_edges) {
    auto& cur_edges = prev_edge_map_[edge];
    std::cerr << "Setting prev edges of: " << edge.toString() << "\n";
    for (const auto& prev_edge : prev_edges) {
      // Avoid duplicates
      if (std::find(cur_edges.begin(), cur_edges.end(), prev_edge) ==
          cur_edges.end()) {
        std::cerr << "New prev edge: ";
        std::cerr << " " << prev_edge.toString();
        std::cerr << "\n";
        cur_edges.push_back(prev_edge);
      }
    }
  }

  virtual void addToToVisitList(const Edge& edge) {
    if (!excludeFromTraversal(edge)) {
      to_visit_.push_back(edge);
    }
  }

  // Hook to exclude certain graph edges.
  virtual bool excludeFromTraversal(const Edge& edge) const {
    return false;
  }

  Direction getDirection(const Edge& edge) const {
    if (const ExprT* from_expr = std::get_if<ExprT>(&edge.from)) {
      const ValT& to_val = std::get<ValT>(edge.to);
      decltype(auto) inputs = inputs_(*from_expr);
      if (std::find(inputs.begin(), inputs.end(), to_val) != inputs.end()) {
        return Direction::Backward;
      }
      decltype(auto) outputs = outputs_(*from_expr);
      if (std::find(outputs.begin(), outputs.end(), to_val) != outputs.end()) {
        return Direction::Forward;
      }
      NVF_THROW();
    } else if (const ValT* from_val = std::get_if<ValT>(&edge.from)) {
      const ExprT& to_expr = std::get<ExprT>(edge.to);
      decltype(auto) inputs = inputs_(to_expr);
      if (std::find(inputs.begin(), inputs.end(), *from_val) != inputs.end()) {
        return Direction::Forward;
      }
      decltype(auto) outputs = outputs_(to_expr);
      if (std::find(outputs.begin(), outputs.end(), *from_val) !=
          outputs.end()) {
        return Direction::Backward;
      }
      NVF_THROW();
    } else {
      NVF_THROW();
    }
  }

  virtual std::unordered_set<NodeType> getVisitedNodes() const {
    std::unordered_set<NodeType> visited_nodes;
    for (const auto& visited_edge : visited_) {
      visited_nodes.emplace(visited_edge.from);
      visited_nodes.emplace(visited_edge.to);
    }
    return visited_nodes;
  }

  virtual std::pair<ExprPath, bool> getOrderedExprPath() {
    NVF_ERROR(
        !require_all_to_visited_ || allToNodesVisited(),
        "Traveral is either not done or failed");

    std::cerr << "getShortestExprPath\n";
    std::deque<Edge> to_visit;

    auto add_to_to_visit_list = [&](const std::vector<Edge>& next_edges) {
      for (const Edge& edge : next_edges) {
        to_visit.emplace_back(edge);
        std::cerr << "Added to visit: " << edge.toString() << "\n";
      }
    };

    std::vector<Edge> initial_edges;
    for (const NodeType& to_node : to_nodes_) {
      if (const ValT* to_val = std::get_if<ValT>(&to_node)) {
        for (const auto& use_expr : uses_(*to_val)) {
          Edge e{use_expr, *to_val};
          if (isVisited(e)) {
            initial_edges.emplace_back(e);
          }
        }
        for (const auto& def_expr : definition_(*to_val)) {
          Edge e{def_expr, *to_val};
          if (isVisited(e)) {
            initial_edges.emplace_back(e);
          }
        }
      } else {
        NVF_THROW(
            "Traversal to nodes are assumed to be all Vals but found: ",
            toString(to_node));
      }
    }
    add_to_to_visit_list(initial_edges);

    ExprPath expr_path;

    std::unordered_set<Edge, EdgeHash> visited_edges;

    while (!to_visit.empty()) {
      const auto edge_to_visit = to_visit.front();
      to_visit.pop_front();

      if (visited_edges.count(edge_to_visit)) {
        continue;
      }

      Direction edge_dir = getDirection(edge_to_visit);

      std::cerr << "(getShortest) Visiting " << edge_to_visit.toString() << ", "
                << edge_dir << "\n";

      if (const ExprT* from_expr = std::get_if<ExprT>(&edge_to_visit.from)) {
        expr_path.emplace_back(*from_expr, edge_dir);
      }

      auto prev_edge_map_it = prev_edge_map_.find(edge_to_visit);
      if (prev_edge_map_it != prev_edge_map_.end()) {
        add_to_to_visit_list(prev_edge_map_it->second);
      }

      visited_edges.insert(edge_to_visit);
    }

    std::cerr << "Current expr path:\n";
    for (const auto& [e, d] : expr_path) {
      std::cerr << d << ", " << toString(e) << "\n";
    }

    std::unordered_set<ValT> visited_vals;
    for (const auto& from_node : from_nodes_) {
      // from_nodes_ and val_nodes_ are assume to be ValT
      visited_vals.insert(std::get<ValT>(from_node));
    }
    std::deque<int64_t> path_offsets(expr_path.size());
    std::iota(path_offsets.begin(), path_offsets.end(), 0);
    VectorOfUniqueEntries<std::pair<ExprT, Direction>> unique_sorted_path;

    while (!path_offsets.empty()) {
      int64_t offset = path_offsets.front();
      path_offsets.pop_front();

      const auto& [expr, dir] = expr_path.at(offset);
      std::cerr << "Visiting " << dir << ", " << toString(expr) << "\n";
      const auto inputs = getInputsOfExpr(expr, dir, inputs_, outputs_);
      if (std::all_of(inputs.begin(), inputs.end(), [&](const ValT& inp) {
            return visited_vals.count(inp);
          })) {
        std::cerr << "Appended to final list\n";
        unique_sorted_path.pushBack(std::make_pair(expr, dir));
        for (const auto& output :
             getOutputsOfExpr(expr, dir, inputs_, outputs_)) {
          visited_vals.insert(output);
        }
      } else {
        std::cerr << "Dep not yet satisfied\n";
        path_offsets.push_back(offset);
      }
    }

    return std::make_pair(unique_sorted_path.vector(), allToNodesVisited());
  }

 protected:
  const DefinitionT definition_;
  const UsesT uses_;
  const InputsT inputs_;
  const OutputsT outputs_;
  const std::vector<NodeType> from_nodes_;
  const std::vector<NodeType> to_nodes_;
  bool require_all_to_visited_ = true;
  Direction allowed_direction_ = Direction::Undefined;

  std::deque<Edge> to_visit_;
  std::unordered_set<Edge, EdgeHash> visited_;
  std::unordered_map<Edge, std::vector<Edge>, EdgeHash> prev_edge_map_;
};

} // namespace nvfuser
