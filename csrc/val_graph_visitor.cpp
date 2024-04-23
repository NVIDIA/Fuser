// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <val_graph_visitor.h>

#include <id_model/to_string.h>

#include <variant>

namespace nvfuser {

void ValGraphVisitor::traverse() {
  const ValGroups terminating_inputs = graph().getTerminatingInputs();
  std::deque<ValGroup> to_visit_vals(
      terminating_inputs.begin(), terminating_inputs.end());
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

  // If any input of the def expr is mapped with the val
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
  // See also IdModelTest.ValGraphStmtSort3 for a concrete example.
  auto is_val_ready = [&](const ValGroup& val_group) -> bool {
    if (terminating_inputs.has(val_group)) {
      return true;
    }
    const ExprGroups& unique_defs = graph().getDefinitions(val_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          if (expr_group->empty() || visited_exprs.has(expr_group)) {
            return true;
          }
          // Handle ExprGroups that return one or some of its input ValGroups as
          // output. This expr_group is not visited yet, which means there're
          // input ValGroups that are not yet visited. If those not-visited
          // inputs are actually the same as val_group, visit val_group at this
          // point to resolve the circular dependency.
          for (const ValGroup& input_group : graph().inputGroups(expr_group)) {
            if (input_group != val_group && !visited_vals.has(input_group) &&
                input_group->empty()) {
              return false;
            }
          }
          return true;
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
    ss << "The graph has an infinite loop. The following Vals should be visited but are never ready:";
    for (const ValGroup& vg : to_visit_vals) {
      ss << " " << nvfuser::toString(vg);
    }
    NVF_ERROR(false, ss.str());
  }

  if (!to_visit_exprs.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be visited but are never ready:";
    for (const ExprGroup& eg : to_visit_exprs) {
      ss << " " << nvfuser::toString(eg);
    }
    NVF_ERROR(false, ss.str());
  }
}

ExprGroups ValGraphBFS::getExprsBetweenVals(
    const ValGraph& graph,
    const ValGroups& from,
    const ValGroups& to) {
  ValGraphBFS bfs(
      graph,
      {from.vector().begin(), from.vector().end()},
      {to.vector().begin(), to.vector().end()});

  bfs.traverse();

  return bfs.getShortestExprPath();
}

namespace {

std::string toString(const ValGraphBFS::GroupType& g) {
  if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
    std::stringstream ss;
    ss << nvfuser::toString(*eg)
       << ", input: " << (*eg)->front()->input(0)->name();
    return ss.str();
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&g)) {
    return nvfuser::toString(*vg);
  } else {
    NVF_ERROR(false);
  }
}

} // namespace

void ValGraphBFS::traverse() {
#if 0
  std::stringstream ss;
  ss << "  Disjoint Ids:\n"
     << idGroupsString(graph_, 2)
     << "\n  Disjoint Expression groups:\n"
     << exprGroupsString(graph_, 2) << std::endl;
  ss << "   } IdGraph\n" << std::endl;

  std::cerr << ss.str();
#endif

  std::cerr << "From: ";
  for (const auto& g : from_groups_) {
    std::cerr << " " << toString(g);
  }
  std::cerr << std::endl;
  std::cerr << "To: ";
  for (const auto& g : to_groups_) {
    std::cerr << " " << toString(g);
  }
  std::cerr << std::endl;

  auto is_all_terminal_visited = [&]() -> bool {
    // std::cerr << "Is all terminal visited\n";
    bool b = std::all_of(
        to_groups_.begin(),
        to_groups_.end(),
        [&](const GroupType& group) -> bool { return isVisited(group); });
    // std::cerr << "Visited? : " << b << std::endl;
    return b;
  };

  // TODO: Make sure from_groups_ has no resize eg that are not in the
  // resize_paths_
  // TODO: Make sure adding new neighbor only considers those in the
  // resize_paths_

  for (const auto& g : from_groups_) {
    setVisited(g);
    addNewNeighbors(g);
  }

  for (const auto& x : visited_) {
    std::cerr << "Initial visited group: " << toString(x) << std::endl;
  }

  while (!is_all_terminal_visited()) {
    bool something_was_processed = false;
    std::deque<GroupType> not_ready_;
    while (!is_all_terminal_visited() && !to_visit_.empty()) {
      const auto g = to_visit_.front();
      to_visit_.pop_front();

      if (isVisited(g)) {
        // std::cerr << "Already visited: " << toString(g) << std::endl;
        continue;
      }

#if 1
      if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
        std::cerr << "Visiting EG: " << nvfuser::toString(*eg) << " "
                  << (*eg)->front()->toString();
      } else if (const ValGroup* vg = std::get_if<ValGroup>(&g)) {
        std::cerr << "Visiting VG: " << nvfuser::toString(*vg) << std::endl;
      }
#endif

      if (!isReady(g)) {
        std::cerr << "Not yet ready: " << toString(g) << std::endl;
        // To stop an infinite loop, the not-ready group is not moved
        // back to the to_visit_ queue but kept in the separate
        // queue. This way, if all groups in to_visit_ are not ready,
        // the queue would eventually become empty, which would then
        // break the inner while loop
        not_ready_.emplace_back(g);
        if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
          std::cerr << " " << (*eg)->front()->toString();
        }
        continue;
      }

      // Visit this group and add its neighbors to to_visit if not
      // visited yet
      handle(g);
      setVisited(g);
      setPrevGroup(g);
      addNewNeighbors(g);
      something_was_processed = true;
    }

    // If nothing was processed, break out of the loop
    if (!something_was_processed) {
      break;
    }

    // Something was processed. Redo the traversal.
    to_visit_.insert(to_visit_.end(), not_ready_.begin(), not_ready_.end());
  };

  if (!is_all_terminal_visited()) {
    std::stringstream ss;
    for (const auto& to_group : to_groups_) {
      if (!isVisited(to_group)) {
        ss << " " << toString(to_group);
        if (const ExprGroup* eg = std::get_if<ExprGroup>(&to_group)) {
          ss << " " << (*eg)->front()->toString();
        }
      }
    }
    NVF_ERROR(false, "BFS traversal could not visit some nodes: ", ss.str());
  }

  std::cerr << "Traversal done\n";
}

bool ValGraphBFS::isReady(const GroupType& group) const {
  if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
    return isReady(*eg);
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&group)) {
    return isReady(*vg);
  } else {
    NVF_ERROR(false);
  }
}

bool ValGraphBFS::isReady(const ExprGroup& expr_group) const {
  // Either all inputs or all outputs must have been visited
  auto inputs = graph_.inputGroups(expr_group);
  if (!inputs.empty() &&
      std::all_of(
          inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
            return isDependencySatisfied(input);
          })) {
    std::cerr << "Forward EG ready\n";
    return true;
  }
  auto outputs = graph_.outputGroups(expr_group);
  if (!outputs.empty() &&
      std::all_of(
          outputs.begin(), outputs.end(), [&](const ValGroup& output) -> bool {
            return isDependencySatisfied(output);
          })) {
    std::cerr << "Backward EG ready\n";
    return true;
  }

  return false;
}

bool ValGraphBFS::isReady(const ValGroup& val_group) const {
  // In the case of Val, requires one def or use expr.
  // Check if any use is visited
  if (!graph_.getUses(val_group).empty() &&
      std::any_of(
          graph_.getUses(val_group).begin(),
          graph_.getUses(val_group).end(),
          [&](const ExprGroup& use_eg) -> bool {
            return isDependencySatisfied(use_eg);
          })) {
    return true;
  }
  // Check if all defs are visited
  if (!graph_.getDefinitions(val_group).empty() &&
      std::any_of(
          graph_.getDefinitions(val_group).begin(),
          graph_.getDefinitions(val_group).end(),
          [&](const ExprGroup& def_eg) -> bool {
            return isDependencySatisfied(def_eg);
          })) {
    return true;
  }

  return false;
}

bool ValGraphBFS::isDependencySatisfied(const GroupType& group) const {
  return isVisited(group);
}

bool ValGraphBFS::isVisited(const GroupType& g) const {
#if 0
  std::cerr << "Is visited: " << toString(g) << ": "
            << (visited_.find(g) != visited_.end()) << std::endl;
  for (const auto& x : visited_) {
    std::cerr << "Current visited group: " << toString(x) << std::endl;
  }
#endif
  return visited_.find(g) != visited_.end();
}

void ValGraphBFS::setVisited(const GroupType& g) {
  visited_.emplace(g);
  std::cerr << "Set visited: " << toString(g) << std::endl;
}

void ValGraphBFS::addNewNeighbors(const GroupType& g) {
  std::cerr << "addNewNeighbors for " << toString(g) << std::endl;
  auto add_to_visit_list = [&](const GroupType& g) -> void {
    if (excludeFromTraversal(g)) {
      std::cerr << "Not traversing " << toString(g) << std::endl;
      return;
    }
    to_visit_.emplace_back(g);
  };

  if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
    for (const auto& vg : graph_.inputGroups(*eg)) {
#if 0
      std::cerr << "Maybe Adding neighbor: " << nvfuser::toString(vg) <<
          std::endl;
#endif
      if (!isVisited(vg)) {
#if 0
        std::cerr << "Adding neighbor: " << nvfuser::toString(vg) <<
            std::endl;
#endif
        // to_visit_.emplace_back(vg);
        add_to_visit_list(vg);
      }
    }
    for (const auto& vg : graph_.outputGroups(*eg)) {
      if (!isVisited(vg)) {
        // std::cerr << "Adding neighbor: " << nvfuser::toString(vg) <<
        // std::endl;
        // to_visit_.emplace_back(vg);
        add_to_visit_list(vg);
      }
    }
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&g)) {
    for (const auto& eg : graph_.getUses(*vg)) {
      if (!isVisited(eg)) {
        // std::cerr << "Adding neighbor: " << nvfuser::toString(eg) <<
        // std::endl;
        // to_visit_.emplace_back(eg);
        add_to_visit_list(eg);
      }
    }
    for (const auto& eg : graph_.getDefinitions(*vg)) {
      if (!isVisited(eg)) {
        // std::cerr << "Adding neighbor: " << nvfuser::toString(eg) <<
        // std::endl;
        // to_visit_.emplace_back(eg);
        add_to_visit_list(eg);
      }
    }
  } else {
    NVF_ERROR(false);
  }
}

void ValGraphBFS::setPrevGroup(const GroupType& group) {
  std::vector<GroupType> prev_groups;

  if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
    auto inputs = graph_.inputGroups(*eg);
    if (!inputs.empty() &&
        std::all_of(
            inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
              return isDependencySatisfied(input);
            })) {
      // Some groups may not be visited, e.g., broadcast groups
      for (const auto& input_group : inputs) {
        if (isVisited(input_group)) {
          prev_groups.emplace_back(input_group);
        }
      }
    } else {
      auto outputs = graph_.outputGroups(*eg);
      NVF_ERROR(
          !outputs.empty() &&
              std::all_of(
                  outputs.begin(),
                  outputs.end(),
                  [&](const ValGroup& output) -> bool {
                    return isDependencySatisfied(output);
                  }),
          "Invalid logic. Both inputs and outputs are not ready");
      for (const auto& output_group : outputs) {
        if (isVisited(output_group)) {
          prev_groups.emplace_back(output_group);
        }
      }
    }
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&group)) {
    if (auto use_it = std::find_if(
            graph_.getUses(*vg).begin(),
            graph_.getUses(*vg).end(),
            [&](const ExprGroup& use_eg) -> bool { return isVisited(use_eg); });
        use_it != graph_.getUses(*vg).end()) {
      prev_groups.emplace_back(*use_it);
    } else {
      auto def_it = std::find_if(
          graph_.getDefinitions(*vg).begin(),
          graph_.getDefinitions(*vg).end(),
          [&](const ExprGroup& def_eg) -> bool {
            return isVisited(def_eg);
          });
      NVF_ERROR(def_it != graph_.getDefinitions(*vg).end(),
                "Invalid logic. Both defs and uses are not ready");
      prev_groups.emplace_back(*def_it);
    }
  } else {
    NVF_ERROR(false);
  }

  std::cerr << "setPrevGroup: " << toString(group) << ": -> ";
  for (const auto& g : prev_groups) {
    std::cerr << " " << toString(g);
  }
  std::cerr << std::endl;

  NVF_ERROR(
      prev_groups_.emplace(group, prev_groups).second,
      "Previous group already set for ",
      toString(group));
}

void ValGraphBFS::handle(const GroupType& group) {
  if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
    handle(*eg);
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&group)) {
    handle(*vg);
  } else {
    NVF_ERROR(false);
  }
}

void ValGraphBFS::handle(const ValGroup& val_group) {
  // std::cerr << "Handle: " << nvfuser::toString(val_group) << std::endl;
}

void ValGraphBFS::handle(const ExprGroup& expr_group) {
  // std::cerr << "Handle: " << nvfuser::toString(expr_group) << std::endl;
}

ExprGroups ValGraphBFS::getShortestExprPath() const {
  std::vector<ExprGroup> path;

  std::deque<GroupType> to_visit;
  for (const auto& to_group : to_groups_) {
    to_visit.emplace_back(to_group);
  }

  while (!to_visit.empty()) {
    const auto group = to_visit.front();
    to_visit.pop_front();

    //std::cerr << "getShortestExprPath: " << toString(group) << std::endl;

    if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
      path.emplace_back(*eg);
    }

    if (std::find(from_groups_.begin(), from_groups_.end(), group) !=
        from_groups_.end()) {
      continue;
    }

    auto prev_groups_it = prev_groups_.find(group);

    // Some groups may be considered visited without actually
    // visited. No previous path set for such groups.
    if (prev_groups_it == prev_groups_.end()) {
      continue;
    }

    for (const auto& prev_group : prev_groups_it->second) {
#if 0
      std::cerr << "prev_group: "
                << toString(prev_group)
                << std::endl;
#endif
      to_visit.emplace_back(prev_group);
    }
  }

  std::reverse(path.begin(), path.end());

  return ExprGroups{path.begin(), path.end()};
}

} // namespace nvfuser
