// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <bfs.h>

namespace {
bool _debug = false;
}

namespace nvfuser {

// Find all exprs between given nodes. Edges are visitd only once,
// but nodes may be visited multiple times. Edges are always between
// ExprT and ValT and are directed, e.g., an edge from an ExprGroup to
// a ValGroup is differentiated from an edge from the ValGroup to the
// ExprGroup, and both of them may be visited.
//
// When there's a cycle, exprs in the cycle are also included. For
// example, given a graph like (each symbol represents an expr):
//
//   A -> B -> C -> D -> E
//        ^         |
//        +--- F ---+
//
// Exprs of {A_fwd, F_bwd, B_fwd, C_fwd, D_fwd, E_fwd} would be
// returened. Note that there's no guarantee of ordering, although it
// is at least partially sorted in a topological order.
template <
    typename ExprT,
    typename ValT,
    typename DefinitionT,
    typename UsesT,
    typename InputsT,
    typename OutputsT>
class FindAllExprs {
 public:
  using ExprType = ExprT;
  using ValType = ValT;
  using NodeType = std::variant<ExprT, ValT>;
  using ExprPath = std::vector<std::pair<ExprT, Direction>>;
  using InputsType = InputsT;
  using OutputsType = OutputsT;

  // Edge represents an edge in the graph. By definition, it must be
  // between an expr and a val.
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
  };

  struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
      return std::hash<NodeType>()(edge.from) ^ std::hash<NodeType>()(edge.to);
    }
  };

  using EdgeSet = std::unordered_set<Edge, EdgeHash>;

  virtual ~FindAllExprs() = default;

 public:
  FindAllExprs(
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

  virtual void traverse() {
    std::deque<Edge> to_visit_;

    for (const auto& from_node : from_nodes_) {
      if (const ValT* from_val = std::get_if<ValT>(&from_node)) {
        for (const auto& use_expr : uses_(*from_val)) {
          Edge e(*from_val, use_expr);
          setVisited(e);
          for (const auto& next_edge : getNextEdges(e, allowed_direction_)) {
            to_visit_.push_back(next_edge);
          }
        }
        for (const auto& def_expr : definition_(*from_val)) {
          Edge e(*from_val, def_expr);
          setVisited(e);
          for (const auto& next_edge : getNextEdges(e, allowed_direction_)) {
            to_visit_.push_back(next_edge);
          }
        }
      } else {
        NVF_THROW(
            "Traversal from nodes are assumed to be all Vals but found: ",
            toString(from_node));
      }
    }

    bool something_was_processed = true;
    while (something_was_processed) {
      std::deque<Edge> not_ready;
      something_was_processed = false;

      while (!to_visit_.empty()) {
        const auto edge_to_visit = to_visit_.front();
        to_visit_.pop_front();

        if (_debug)
          std::cerr << "Next edge: " << edge_to_visit.toString() << "\n";

        // Don't visit edges multiple times even when traversing all paths
        if (isVisited(edge_to_visit)) {
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
          if (_debug)
            std::cerr << "Not ready\n";
          continue;
        }

        if (_debug)
          std::cerr << "Visiting " << edge_to_visit.toString() << "\n";

        setVisited(edge_to_visit);
        for (const auto& next_edge :
             getNextEdges(edge_to_visit, allowed_direction_)) {
          to_visit_.push_back(next_edge);
        }
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

    if (_debug)
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
                return isVisited(Edge(input, from_expr));
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
                return isVisited(Edge(output, from_expr));
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
        if (def_e != to_expr && isVisited(Edge(def_e, from_val))) {
          prev_edges.emplace_back(Edge(def_e, from_val));
        }
      }
    }

    decltype(auto) uses = uses_(from_val);
    for (const ExprT& use_e : uses) {
      if (use_e != to_expr && isVisited(Edge(use_e, from_val))) {
        prev_edges.emplace_back(Edge(use_e, from_val));
      }
    }

    return prev_edges.empty() ? std::nullopt : std::make_optional(prev_edges);
  }

  // Check if a given node is already visited
  virtual bool isVisited(const Edge& edge) const {
    return visited_edges_.find(edge) != visited_edges_.end();
  }

  virtual void setVisited(const Edge& edge) {
    if (visited_edges_.emplace(edge).second) {
      partially_ordered_visited_edges_.push_back(edge);
    }
  }

  virtual std::vector<Edge> getNextEdges(
      const Edge& edge,
      Direction allowed_direction = Direction::Undefined) const {
    std::vector<Edge> neighbor_edges;

    auto add_to_neighbor_list = [&](const auto& from, const auto& to) -> void {
      Edge neighbor_edge(from, to);

      if (edge == neighbor_edge ||
          // Don't traverse back
          (edge.from == neighbor_edge.to && edge.to == neighbor_edge.from)) {
        return;
      }

      if (excludeFromTraversal(neighbor_edge)) {
        return;
      }

      auto neighbor_edge_dir = getDirection(neighbor_edge);
      if ((allowed_direction == Direction::Forward &&
           neighbor_edge_dir == Direction::Backward) ||
          (allowed_direction == Direction::Backward &&
           neighbor_edge_dir == Direction::Forward)) {
        return;
      }

      neighbor_edges.push_back(neighbor_edge);
    };

    Direction edge_dir = getDirection(edge);
    NVF_ERROR(
        edge_dir == Direction::Forward || edge_dir == Direction::Backward);

    if (const ExprT* e = std::get_if<ExprT>(&edge.to)) {
      // The from node must be a Val.

      // In the case of Expr, only consider edges of the same
      // direction
      if (edge_dir == Direction::Forward) {
        // This edge is from an input Val to its use Expr. Traverse
        // from the use Expr to its outputs.
        for (const auto& v : outputs_(*e)) {
          add_to_neighbor_list(*e, v);
        }
      } else if (edge_dir == Direction::Backward) {
        // This edge is from an output Val to its defining Expr. Traverse
        // from the defining Expr to its inputs.
        for (const auto& v : inputs_(*e)) {
          add_to_neighbor_list(*e, v);
        }
      }
    } else if (const ValT* v = std::get_if<ValT>(&edge.to)) {
      // The from node must be an Expr.

      // In the case of Val, no matter what direction this node is, it
      // should be valid to traverse both directions. Just don't
      // traverse back to the same node.

      for (const auto& e : uses_(*v)) {
        add_to_neighbor_list(*v, e);
      }

      for (const auto& e : definition_(*v)) {
        add_to_neighbor_list(*v, e);
      }
    }

    return neighbor_edges;
  }

  virtual std::vector<Edge> getPrevEdges(
      const Edge& edge,
      Direction allowed_direction = Direction::Undefined) const {
    std::vector<Edge> neighbor_edges;

    auto add_to_neighbor_list = [&](const auto& from, const auto& to) -> void {
      Edge neighbor_edge(from, to);

      if (edge == neighbor_edge ||
          // Don't traverse back
          (edge.from == neighbor_edge.to && edge.to == neighbor_edge.from)) {
        return;
      }

      if (excludeFromTraversal(neighbor_edge)) {
        return;
      }

      auto neighbor_edge_dir = getDirection(neighbor_edge);
      if ((allowed_direction == Direction::Forward &&
           neighbor_edge_dir == Direction::Backward) ||
          (allowed_direction == Direction::Backward &&
           neighbor_edge_dir == Direction::Forward)) {
        return;
      }

      neighbor_edges.push_back(neighbor_edge);
    };

    Direction edge_dir = getDirection(edge);
    NVF_ERROR(
        edge_dir == Direction::Forward || edge_dir == Direction::Backward);

    if (const ExprT* e = std::get_if<ExprT>(&edge.from)) {
      // The to node must be a Val.

      // In the case of Expr, only consider edges of the same
      // direction
      if (edge_dir == Direction::Forward) {
        // This edge is from a defining expr to one of its
        // outputs. The previous edges consist of the inputs of the
        // expr to the expr.
        for (const auto& v : inputs_(*e)) {
          add_to_neighbor_list(v, *e);
        }
      } else if (edge_dir == Direction::Backward) {
        // This edge is from a use Expr to one of its inputs. The
        // previous edges consist of the ouputs of the expr to the
        // expr.
        for (const auto& v : outputs_(*e)) {
          add_to_neighbor_list(v, *e);
        }
      }
    } else if (const ValT* v = std::get_if<ValT>(&edge.from)) {
      // The to node must be an Expr.

      // In the case of Val, no matter what direction this edge is, it
      // should be valid to traverse both directions. Just don't
      // traverse back to the same node.

      for (const auto& e : definition_(*v)) {
        add_to_neighbor_list(e, *v);
      }

      for (const auto& e : uses_(*v)) {
        add_to_neighbor_list(e, *v);
      }
    }

    return neighbor_edges;
  }

  // Check if all to_ are visited
  virtual bool allToNodesVisited() const {
    auto visited_nodes = getVisitedNodes();
    return std::all_of(
        to_nodes_.begin(), to_nodes_.end(), [&](const NodeType& node) -> bool {
          return visited_nodes.count(node);
        });
  };

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
    visited_nodes.insert(from_nodes_.begin(), from_nodes_.end());
    for (const auto& visited_edge : visited_edges_) {
      visited_nodes.emplace(visited_edge.from);
      visited_nodes.emplace(visited_edge.to);
    }
    return visited_nodes;
  }

  virtual std::pair<ExprPath, bool> getPartiallyOrderedExprs() const {
    const auto used_edges = getUsedEdges();

    VectorOfUniqueEntries<std::pair<ExprT, Direction>> expr_path;

    for (const Edge& ordered_visited_edge : partially_ordered_visited_edges_) {
      if (!used_edges.count(ordered_visited_edge)) {
        continue;
      }

      if (_debug)
        std::cerr << ordered_visited_edge.toString() << "\n";

      Direction edge_dir = getDirection(ordered_visited_edge);

      // Append the expr of this edge
      const ExprT& expr =
          std::get_if<ExprT>(&(ordered_visited_edge.from)) != nullptr
          ? std::get<ExprT>(ordered_visited_edge.from)
          : std::get<ExprT>(ordered_visited_edge.to);
      expr_path.pushBack(std::make_pair(expr, edge_dir));
    }

    return std::make_pair(expr_path.vector(), allToNodesVisited());
  }

  virtual EdgeSet getUsedEdges() const {
    NVF_ERROR(
        !require_all_to_visited_ || allToNodesVisited(),
        "Traveral is either not done or failed");

    // Traverse back from to_ nodes to from_ nodes by traversing
    // through visted edges
    std::deque<Edge> to_visit;

    // Gather all visited edges to the to_ nodes. These edges are used
    // as initial edges for the traversal below
    for (const NodeType& to_node : to_nodes_) {
      if (const ValT* to_val = std::get_if<ValT>(&to_node)) {
        for (const ExprT& use_expr : uses_(*to_val)) {
          Edge e{use_expr, *to_val};
          if (isVisited(e)) {
            to_visit.emplace_back(e);
          }
        }
        for (const ExprT& def_expr : definition_(*to_val)) {
          Edge e{def_expr, *to_val};
          if (isVisited(e)) {
            to_visit.emplace_back(e);
          }
        }
      } else {
        NVF_THROW(
            "Traversal to nodes are assumed to be all Vals but found: ",
            toString(to_node));
      }
    }

    EdgeSet used_edges;

    while (!to_visit.empty()) {
      const auto edge_to_visit = to_visit.front();
      to_visit.pop_front();

      if (used_edges.count(edge_to_visit)) {
        continue;
      }

      auto prev_edges = getPrevEdges(edge_to_visit);
      for (const Edge& prev_edge : prev_edges) {
        if (isVisited(prev_edge)) {
          to_visit.emplace_back(prev_edge);
        }
      }

      used_edges.insert(edge_to_visit);
    }

    return used_edges;
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

  EdgeSet visited_edges_;
  std::vector<Edge> partially_ordered_visited_edges_;
};

} // namespace nvfuser
