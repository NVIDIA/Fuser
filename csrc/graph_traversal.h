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

// Find all exprs reachable from from_nodes when traversing to to_nodes. Edges
// are visitd only once, but nodes may be visited multiple times. Edges are
// always between ExprT and ValT and are directed, e.g., an edge from an
// ExprGroup to a ValGroup is differentiated from an edge from the ValGroup to
// the ExprGroup, and both of them may be visited.
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
//
// The overall traversal algorithm is to start from from_nodes and
// traverse edges in both directions or only in a specified
// direction. Unlike BFS, it keeps traversing even if all
// the to_nodes are reached and stops when no further progress is
// possible. At this point, we know all the reachable edges from
// from_nodes but we are only interested in that reach to_nodes. To
// find those edges, another traversal, this time from to_ndoes, is
// done to mark all visited edges that are reachable from
// to_nodes. That gives us all the edges between from_nodes and
// to_nodes. Finally, ExprPath is returned based on the exprs of the
// edges.
//
// NOTE 1: The algorithm and the implementation is based on the BFS
// class. There's likely more efficient algorithms.
//
// NOTE 2: The returned expr path is not guaranteed to be
// topologically sorted, which is not possible for cyclic graphs.
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

  virtual void traverseAllEdges() {
    std::deque<Edge> edges_to_visit;

    for (const auto& from_node : from_nodes_) {
      if (const ValT* from_val = std::get_if<ValT>(&from_node)) {
        for (const auto& use_expr : uses_(*from_val)) {
          Edge edge(*from_val, use_expr);
          setVisited(edge);
          for (const auto& next_edge :
               getConsumerEdges(edge, allowed_direction_)) {
            edges_to_visit.push_back(next_edge);
          }
        }
        for (const auto& def_expr : definition_(*from_val)) {
          Edge edge(*from_val, def_expr);
          setVisited(edge);
          for (const auto& next_edge :
               getConsumerEdges(edge, allowed_direction_)) {
            edges_to_visit.push_back(next_edge);
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

      while (!edges_to_visit.empty()) {
        const auto edge_to_visit = edges_to_visit.front();
        edges_to_visit.pop_front();

        // Don't visit edges multiple times even when traversing all paths
        if (isVisited(edge_to_visit)) {
          continue;
        }

        auto prev_edges = isReady(edge_to_visit);
        if (prev_edges.empty()) {
          // To stop an infinite loop, the not-ready node is not moved
          // back to the to_visit_ queue but kept in the separate
          // queue. This way, if all nodes in to_visit_ are not ready,
          // the queue would eventually become empty, which would then
          // break the inner while loop. The something_was_processed
          // flag is used to remember if there's any progress.
          not_ready.emplace_back(edge_to_visit);
          continue;
        }

        setVisited(edge_to_visit);
        for (const auto& next_edge :
             getConsumerEdges(edge_to_visit, allowed_direction_)) {
          edges_to_visit.push_back(next_edge);
        }
        something_was_processed = true;
      }

      // Something was processed. Redo the traversal.
      edges_to_visit.insert(
          edges_to_visit.end(), not_ready.begin(), not_ready.end());
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
  }

  // Check if a node is ready to visit. If yes, return the direction
  // and the prev nodes that should be visited before the given node
  // is visited.
  virtual std::vector<Edge> isReady(const Edge& edge) const {
    Direction dir = getDirection(edge);

    // If a direction is specified, only that direction of edges are
    // allowed.
    if ((dir == Direction::Forward &&
         allowed_direction_ == Direction::Backward) ||
        (dir == Direction::Backward &&
         allowed_direction_ == Direction::Forward)) {
      return {};
    }

    if (const ExprT* e = std::get_if<ExprT>(&(edge.from))) {
      return isReady(*e, std::get<ValT>(edge.to), dir);
    } else if (const ValT* v = std::get_if<ValT>(&(edge.from))) {
      return isReady(*v, std::get<ExprT>(edge.to), dir);
    } else {
      NVF_THROW();
    }
  }

  // Check if an edge from an expr to a val is ready to visit. If this
  // is a forward edge, i.e., the val is an output of the expr, the
  // edge is ready to visit as long as all the inputs of the expr are
  // visited. If it's a backward edge, i.e., the val is an input of
  // the expr, it's ready if all of the outputs are visited. If ready,
  // the edges that this edge depends on are returned. For example, in
  // the case of a forward edge, all of the edges to from_expr are
  // returned.
  virtual std::vector<Edge> isReady(
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

    return {};
  }

  // Check if an edge from a val to an expr is ready to visit. In the
  // case of a val, it is ready to visit as long as there's at least
  // one def or use expr that has been already visited. However, since
  // this is an edge to an expr, the edge from the same expr to this
  // val does not make this edge ready to visit. For example, even if
  // a merge producing i0 is visited, it should not automatically mean
  // the edge from i0 to the merge expr is ready to visit. Othewise,
  // the traversal would just move back and forth.
  virtual std::vector<Edge> isReady(
      const ValT& from_val,
      const ExprT& to_expr,
      Direction dir) const {
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

    return prev_edges;
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

  // Get edges that are consumers or producers of a given edge. A
  // consumer edge of edge A->B is an edge that has node B as its from
  // node. A producer edge is an edge that has node A as its to node.
  virtual std::vector<Edge> getConsumerOrProducerEdges(
      const Edge& edge,
      bool is_consumer,
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

    const auto& node = is_consumer ? edge.to : edge.from;

    // Since the direction is forward, this edge is
    // Consumer edges are those that start from the e expr. Since
    // the direction is Forward, When grabbing consumer edges, If the node is
    // the to of the edge, the edge is from an input Val to its use Expr, so
    // traverse from the use Expr to its outputs. If the node is the from of the
    // edge, the edge is from a defining expr to one of its outputs, in that
    // case grab edges of the inputs of the expr.

    if (const ExprT* e = std::get_if<ExprT>(&node)) {
      // The from node must be a Val.

      // In the case of Expr, only consider edges of the same
      // direction
      if (edge_dir == Direction::Forward) {
        if (is_consumer) {
          // Grab consumer edges of the forward edge to the expr. The
          // edge represents a use expr of the from val. Consumers are
          // forward edges from the expr to its outputs.
          for (const auto& v : outputs_(*e)) {
            add_to_neighbor_list(*e, v);
          }
        } else {
          // Grab producer edges of the forward edge from the expr. The
          // edge represents a defining expr of the to val. Producers
          // are forward edges to the defining expr from its inputs.
          for (const auto& v : inputs_(*e)) {
            add_to_neighbor_list(v, *e);
          }
        }
      } else if (edge_dir == Direction::Backward) {
        if (is_consumer) {
          // Grab consumer edges of the backward edge to the expr. The
          // edge represents a defining expr of the from val. Consumers
          // are backward edges from the defining expr to its inputs.
          for (const auto& v : inputs_(*e)) {
            add_to_neighbor_list(*e, v);
          }
        } else {
          // Grab producer edges of the backward edge from the expr. The
          // edge represents a use expr of the from val. Produces
          // are backward edges to the use expr expr from its outputs.
          for (const auto& v : outputs_(*e)) {
            add_to_neighbor_list(v, *e);
          }
        }
      }
    } else if (const ValT* v = std::get_if<ValT>(&node)) {
      // The from node must be an Expr.

      // In the case of Val, no matter what direction this node is, it
      // should be valid to traverse both directions. Just don't
      // traverse back to the same node.

      for (const auto& e : uses_(*v)) {
        if (is_consumer) {
          // Uses of v are forward consumer edges of the edge to val v
          add_to_neighbor_list(*v, e);
        } else {
          // Uses of v are backward producer edges of the edge from val v
          add_to_neighbor_list(e, *v);
        }
      }

      for (const auto& e : definition_(*v)) {
        if (is_consumer) {
          // Defs of v are backward consumer edges of the edge to val v
          add_to_neighbor_list(*v, e);
        } else {
          // Defs of v are forward producer edges of the edge from val v
          add_to_neighbor_list(e, *v);
        }
      }
    }

    return neighbor_edges;
  }

  // Get edges that should be traversed from the to node of a given edge
  virtual std::vector<Edge> getConsumerEdges(
      const Edge& edge,
      Direction allowed_direction = Direction::Undefined) const {
    return getConsumerOrProducerEdges(
        edge, /*is_consumer=*/true, allowed_direction);
  }

  // Get edges that should be traversed before the from node of a given edge
  virtual std::vector<Edge> getProducerEdges(
      const Edge& edge,
      Direction allowed_direction = Direction::Undefined) const {
    return getConsumerOrProducerEdges(
        edge, /*is_consumer=*/false, allowed_direction);
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

  // If an edge is from a val to its use expr, it's a forward
  // edge. Similarly, it's also a forward edge if it's an expr to one
  // of its outputs. Otherwise, it's a backward edge.
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

  // Grab all visited edges that are reachable from from_nodes and
  // to_nodes. traverseAllEdges must have been completed.
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

      auto producer_edges = getProducerEdges(edge_to_visit);
      for (const Edge& producer_edge : producer_edges) {
        if (isVisited(producer_edge)) {
          to_visit.emplace_back(producer_edge);
        }
      }

      used_edges.insert(edge_to_visit);
    }

    return used_edges;
  }

  // Return ExprPath consisting of all exprs appearing between
  // from_nodes and to_ndoes. The exprs are partially topologically
  // sorted, but not completely. The ordering should be deterministic,
  // but do not assume any particular ordering.
  virtual std::pair<ExprPath, bool> getPartiallyOrderedExprs() const {
    const auto used_edges = getUsedEdges();

    VectorOfUniqueEntries<std::pair<ExprT, Direction>> expr_path;

    for (const Edge& ordered_visited_edge : partially_ordered_visited_edges_) {
      if (!used_edges.count(ordered_visited_edge)) {
        continue;
      }

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
