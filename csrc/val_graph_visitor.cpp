// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <graph_traversal.h>
#include <id_model/to_string.h>
#include <val_graph_visitor.h>

namespace nvfuser {

bool ValGraphVisitor::traverse() {
  if (graph().disjointValSets().size() == 0) {
    return true;
  }

  // Traverse from terminating inputs. When a graph is cyclic, there
  // may be no terminating inputs. Additional starting groups can be
  // specified as an option.
  ValGroups starting_groups = graph().getTerminatingInputs();
  starting_groups.pushBack(additional_starting_groups_);

  // If no terminating input is found, that should mean there's a
  // cycle.
  if (starting_groups.empty()) {
    std::stringstream ss;
    ss << "Unsupported graph: No terminating input found, likely a cyclic "
          "graph: ";
    ss << graph().toString();
    error_message_ = ss.str();
    return false;
  }

  std::deque<ValGroup> to_visit_vals(
      starting_groups.begin(), starting_groups.end());
  ValGroups visited_vals;

  std::deque<ExprGroup> to_visit_exprs;
  ExprGroups visited_exprs;

  auto is_expr_ready = [&](const ExprGroup& expr_group) -> bool {
    const auto inp_groups = graph().inputGroups(expr_group);
    return std::all_of(
        inp_groups.begin(), inp_groups.end(), [&](ValGroup val_group) {
          return visited_vals.has(val_group) || val_group->empty();
        });
  };

  // When allow_cycle_ is true, cyclic dependency is ignored. For
  // example, if any input of the def expr is mapped with the val
  // group itself, i.e., a trivial expr, allow visiting the
  // val group first. The trivial expr group will be visited
  // after the val group.
  //
  // Example:
  //
  // [i0, 1]
  // merge
  // [i0*1]
  // map i0 and i0*1
  // ValGroups: {{i0, i0*1}, {1}}
  //
  // Then, {i0, i0*1} and {1} would be visited first, then the merge
  // expr group would be visited. {i0, i0*1} is also an output group
  // of the merge but since it's already in the visited set, it would
  // not be visited again.
  //
  // Similarly, when there are five groups as shown below:
  //
  //   i0 -> i1  ->  i2 -> i3
  //          ^       |
  //          |- i4 <-+
  //
  //  (Edges: i0->i1, i1->i2, i2->i3, i2->i4, i4->i1)
  //
  // is_val_ready of i1 would become true while ignoring the incoming
  // edge from i4. The traversal order would look like:
  //
  // i0->i1, i1->i2, i2->i3, i2->i4
  //
  // See also IdModelTest.ValGraphStmtSort3 for a concrete
  // example. See IdModelTest.LoopPromotionWithCyclicGraph for some
  // use cases of this traversal for the loop promotion analysis with
  // cyclic graphs.
  auto is_val_ready = [&](const ValGroup& val_group) -> bool {
    if (starting_groups.has(val_group)) {
      return true;
    }
    const ExprGroups& unique_defs = graph().getDefinitions(val_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          if (expr_group->empty() || visited_exprs.has(expr_group)) {
            return true;
          }

          if (!allow_cycle_) {
            return false;
          }

          auto reachable_nodes = getReachableNodesFrom<ValGraphPermissiveBFS>(
              {expr_group}, {val_group}, Direction::Backward, graph());
          if (!reachable_nodes.empty()) {
            // Cycle detected.
            return true;
          }

          return false;
        });
  };

  // Detect if nothing has been processed which would put us in an infinite
  // loop
  bool something_was_processed = false;

  do {
    something_was_processed = false;

    // Process expressions first as all definitions of vals have to be
    // processed before we can process that val.

    while (!to_visit_exprs.empty()) {
      ExprGroup current_expr_group = to_visit_exprs.front();
      to_visit_exprs.pop_front();
      NVF_ERROR(!current_expr_group->empty());
      if (visited_exprs.has(current_expr_group)) {
        continue;
      }

      if (is_expr_ready(current_expr_group)) {
        handle(current_expr_group);

        something_was_processed = true;
        visited_exprs.pushBack(current_expr_group);

        for (const ValGroup& output_group :
             graph().outputGroups(current_expr_group)) {
          to_visit_vals.push_back(output_group);
        }
      }
    }

    std::deque<ValGroup> still_to_visit_vals;
    while (!to_visit_vals.empty()) {
      auto current_val_group = to_visit_vals.front();
      to_visit_vals.pop_front();
      NVF_ERROR(!current_val_group->empty());
      if (visited_vals.has(current_val_group)) {
        continue;
      }

      if (is_val_ready(current_val_group)) {
        handle(current_val_group);

        something_was_processed = true;
        visited_vals.pushBack(current_val_group);

        for (const ExprGroup& use_group : graph().getUses(current_val_group)) {
          to_visit_exprs.push_back(use_group);
        }
      } else {
        still_to_visit_vals.push_back(current_val_group);
      }
    }

    std::swap(to_visit_vals, still_to_visit_vals);

  } while (something_was_processed);

  if (!to_visit_vals.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Vals should be "
          "visited but are never ready:";
    for (const ValGroup& vg : to_visit_vals) {
      ss << " " << nvfuser::toString(vg);
    }
    ss << ". Already visited vals: ";
    for (const auto& eg : visited_vals) {
      ss << " " << nvfuser::toString(eg);
    }
    ss << ". Already visited exprs: ";
    for (const ExprGroup& eg : visited_exprs) {
      ss << " " << nvfuser::toString(eg);
    }

    error_message_ = ss.str();
    graph().dumpGraphvizDotGraph("val_graph_stmt_sort.dot");
    return false;
  }

  if (!to_visit_exprs.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be "
          "visited but are never ready:";
    for (const ExprGroup& eg : to_visit_exprs) {
      ss << " " << nvfuser::toString(eg);
    }
    ss << ". Already visited exprs: ";
    for (const ExprGroup& eg : visited_exprs) {
      ss << " " << nvfuser::toString(eg);
    }
    error_message_ = ss.str();

    graph().dumpGraphvizDotGraph("val_graph_stmt_sort.dot");
    return false;
  }

  // If not all exprs are visited, that should mean there must be a
  // cyclic subgraph. The subgraph should have no terminating input,
  // so it should not be visited at all. Note that some Val groups may
  // not be visited, which should be fine.
  if (visited_exprs.size() != graph().disjointExprSets().size()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be "
          "visited but are never ready:";
    for (const ExprGroup& eg : graph().disjointExprSets().disjointSets()) {
      if (!visited_exprs.has(eg)) {
        ss << " " << nvfuser::toString(eg);
      }
    }
    error_message_ = ss.str();
    graph().dumpGraphvizDotGraph("stmt_sort_cycle.dot");
    return false;
  }

  return true;
}

namespace {

class ValGraphCycleDetector : public ValGraphVisitor {
 public:
  ValGraphCycleDetector(const ValGraph& graph)
      : ValGraphVisitor(graph, /*allow_cycle=*/false),
        cycle_detected_(!traverse()) {}

  void handle(const ValGroup& val_group) override {}
  void handle(const ExprGroup& expr_group) override {}

  bool cycle_detected_ = false;
};

} // namespace

bool isCyclic(const ValGraph& graph) {
  return ValGraphCycleDetector(graph).cycle_detected_;
}

std::pair<ExprGroupPath, bool> getAllExprGroupsBetween(
    const ValGraph& graph,
    const ValGroups& from,
    const ValGroups& to,
    bool require_all_to_visited,
    Direction allowed_direction) {
  FindAllExprs<
      ExprGroup,
      ValGroup,
      ValGraphDefinitions,
      ValGraphUses,
      ValGraphInputs,
      ValGraphOutputs>
      finder(
          ValGraphDefinitions{graph},
          ValGraphUses{graph},
          ValGraphInputs{graph},
          ValGraphOutputs{graph},
          {from.vector().begin(), from.vector().end()},
          {to.vector().begin(), to.vector().end()},
          require_all_to_visited,
          allowed_direction);
  finder.traverseAllEdges();
  return finder.getPartiallyOrderedExprs();
}

} // namespace nvfuser
