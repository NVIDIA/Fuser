// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_graph.h>
#include <id_model/to_string.h>
#include <ir/utils.h>

namespace nvfuser {

namespace {
using UnorderedSetOfExprGroup = std::unordered_set<ExprGroup>;
using DequeOfExprGroup = std::deque<ExprGroup>;
} // namespace

IdGraph::IdGraph(const IdGraph& other)
    : disjoint_ids_(other.disjoint_ids_),
      disjoint_exprs_(other.disjoint_exprs_),
      unique_definitions_(),
      unique_uses_() {
  for (const auto& [orig_id_group, orig_expr_groups] :
       other.unique_definitions_) {
    auto new_id_group = toGroup(orig_id_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    unique_definitions_[new_id_group] = new_expr_groups;
  }

  for (const auto& [orig_id_group, orig_expr_groups] : other.unique_uses_) {
    auto new_id_group = toGroup(orig_id_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    unique_uses_[new_id_group] = new_expr_groups;
  }
}

IdGraph& IdGraph::operator=(const IdGraph& other) {
  disjoint_ids_.clear();
  disjoint_exprs_.clear();
  unique_definitions_.clear();
  unique_uses_.clear();
  IdGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

// Return if there's a group entry in the graph for this expr
bool IdGraph::hasGroup(Expr* expr) const {
  return disjoint_exprs_.mappingExists(expr);
}

// Return if there's a group entry in the graph for this id
bool IdGraph::hasGroup(IterDomain* id) const {
  return disjoint_ids_.mappingExists(id);
}

const ExprGroup& IdGraph::toGroup(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  NVF_ERROR(
      disjoint_set_it != disjoint_exprs_.disjointSetMap().end(),
      "\nExpr group could not be found in graph associated with: ",
      expr->toString());
  return disjoint_set_it->second;
}

const IdGroup& IdGraph::toGroup(IterDomain* id) const {
  auto disjoint_set_it = disjoint_ids_.disjointSetMap().find(id);
  NVF_ERROR(
      disjoint_set_it != disjoint_ids_.disjointSetMap().end(),
      "\nId group could not be found in graph associated with: ",
      id->toString(),
      "\n");
  return disjoint_set_it->second;
}

ExprGroups IdGraph::toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const {
  ExprGroups expr_groups;
  for (auto expr : exprs) {
    expr_groups.pushBack(toGroup(expr));
  }
  return expr_groups;
}

IdGroups IdGraph::toGroups(
    const VectorOfUniqueEntries<IterDomain*>& ids) const {
  IdGroups id_groups;
  for (auto id : ids) {
    id_groups.pushBack(toGroup(id));
  }
  return id_groups;
}

std::vector<IdGroup> IdGraph::outputGroups(const ExprGroup& expr) const {
  std::vector<IdGroup> output_groups;
  for (auto id_output :
       ir_utils::filterByType<IterDomain>(expr->front()->outputs())) {
    output_groups.push_back(toGroup(id_output));
  }
  return output_groups;
}

std::vector<IdGroup> IdGraph::inputGroups(const ExprGroup& expr) const {
  std::vector<IdGroup> input_groups;
  for (auto id_input :
       ir_utils::filterByType<IterDomain>(expr->front()->inputs())) {
    input_groups.push_back(toGroup(id_input));
  }
  return input_groups;
}

ExprGroups IdGraph::allUsesOf(const IdGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const IdGroup& of_id_group : of) {
    if (const auto& [group_uses, found] = getUses(of_id_group); found) {
      to_visit.insert(to_visit.end(), group_uses.begin(), group_uses.end());
    }
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const IdGroup& output_id : outputGroups(current_expr)) {
      if (const auto& [group_uses, found] = getUses(output_id); found) {
        for (const ExprGroup& group_use : group_uses) {
          if (visited.count(group_use)) {
            continue;
          }
          to_visit.push_back(group_use);
        }
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::allDefinitionsOf(const IdGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const IdGroup& of_id_group : of) {
    if (const auto& [group_defs, found] = getDefinitions(of_id_group); found) {
      to_visit.insert(to_visit.end(), group_defs.begin(), group_defs.end());
    }
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const IdGroup& input_id : inputGroups(current_expr)) {
      if (const auto& [group_defs, found] = getDefinitions(input_id); found) {
        for (const ExprGroup& group_def : group_defs) {
          if (visited.count(group_def)) {
            continue;
          }
          to_visit.push_back(group_def);
        }
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::getExprsBetween(const IdGroups& from, const IdGroups& to)
    const {
  ExprGroups all_uses_of_from = allUsesOf(from);
  ExprGroups all_definitions_of_to = allDefinitionsOf(to);

  // All of the expressions between from and to. Not all will be used as we
  // just want to define each iter domain group once.
  ExprGroups all_exprs = all_uses_of_from.intersect(all_definitions_of_to);

  // There could be IterDomains in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;
  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    IdGroups all_id_groups;

    for (const ExprGroup& expr_group : all_exprs) {
      if (isTrivialExprGroup(expr_group)) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      std::vector<IdGroup> inp_groups = inputGroups(expr_group);
      std::vector<IdGroup> out_groups = outputGroups(expr_group);

      all_id_groups.pushBack(inp_groups);
      not_outputs.pushBack(inp_groups);

      all_id_groups.pushBack(out_groups);
      not_inputs.pushBack(out_groups);
    }
    terminating_inputs = all_id_groups.subtract(not_inputs);
    terminating_outputs = all_id_groups.subtract(not_outputs);
  }

  // Track all expressions to get from outputs to this IterDomain. We
  // traverse backwards as that's the direction of indexing expressions. An
  // index is assigned to each leaf of a domain and as we traverse backwards
  // we're effectively accumulating indexing math. We'll only keep the fewest
  // expression lists to get to the iter domain.
  std::unordered_map<IdGroup, ExprGroups> required_ind_exprs_ids;
  std::unordered_map<ExprGroup, ExprGroups> required_ind_exprs_exprs;

  // Return if all output IterDomain groups of an expression group have
  // already been visited
  auto outputsVisited = [&](ExprGroup expr_group) {
    auto output_groups = outputGroups(expr_group);
    return std::all_of(
        output_groups.begin(),
        output_groups.end(),
        [&](const IdGroup& output_group) {
          return required_ind_exprs_ids.find(output_group) !=
              required_ind_exprs_ids.end();
        });
  };

  // Returns all expression groups in required_ind_exprs_ids of outputs
  auto requiredExprsOutputs = [&](ExprGroup expr_group) -> ExprGroups {
    ExprGroups all_output_required_exprs;
    for (const IdGroup& output_id_group : outputGroups(expr_group)) {
      auto id_group_exprs_it = required_ind_exprs_ids.find(output_id_group);
      NVF_ERROR(
          id_group_exprs_it != required_ind_exprs_ids.end(),
          "Failure in Iter Domain Graph index resolution, count expected for group: ",
          output_id_group->toString());
      all_output_required_exprs.pushBack(id_group_exprs_it->second);
    }
    return all_output_required_exprs;
  };

  auto processExprGroup = [&](ExprGroup expr_group) -> bool {
    if (!outputsVisited(expr_group)) {
      return false;
    }
    // Accumulate expressions from all outputs add this expression and set it
    // as current expressions required indexing expressions.
    required_ind_exprs_exprs[expr_group] = requiredExprsOutputs(expr_group);
    return true;
  };

  auto processIdGroup = [&](IdGroup id_group) -> bool {
    // Track if we've grabed any of the uses required indexing expressions.
    bool initialized = false;
    // Expression group of all indexing expressions required for this iter
    // domain coming back from any of its uses.
    ExprGroups min_groups;

    std::pair<ExprGroups, bool> uses_pair = getUses(id_group);
    if (!uses_pair.second) {
      // No expressions required for this iter domain, it must be a
      // terminating output.
      required_ind_exprs_ids[id_group] = min_groups;
      return true;
    }

    // Only worry about expressions between inputs and outputs we're
    // looking at.
    for (const ExprGroup& use_group : uses_pair.first.intersect(all_exprs)) {
      auto use_required_ind_exprs_it = required_ind_exprs_exprs.find(use_group);
      if (use_required_ind_exprs_it == required_ind_exprs_exprs.end()) {
        // If there isn't an entry for the use expression it wasn't
        // processed, so don't try to process this iter domain yet.
        return false;
      }
      if (!initialized) {
        // If first use found initialize the minimum expression group
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
        initialized = true;
      } else if (
          use_required_ind_exprs_it->second.size() + 1 < min_groups.size()) {
        // If current use has fewer expressions use that, make sure to add the
        // use expression.
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
      }
    }
    required_ind_exprs_ids[id_group] = min_groups;
    return true;
  };

  // Backward traversal from the terminating outputs
  IdGroups to_visit_ids = terminating_outputs;
  ExprGroups to_visit_exprs;

  while (!to_visit_ids.empty() || !to_visit_exprs.empty()) {
    // Process expressions first as all uses of iter domains have to be
    // processed before we can process that iter domain.

    // Try to detect when nothing has been processed which would put us in an
    // infinite loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;
    while (!to_visit_exprs.empty()) {
      ExprGroup currently_visiting_exprs = to_visit_exprs.popFront();
      if (required_ind_exprs_exprs.find(currently_visiting_exprs) !=
          required_ind_exprs_exprs.end()) {
        // currently_visiting_exprs is already visited
        continue;
      }
      if (processExprGroup(currently_visiting_exprs)) {
        something_was_processed = true;
        std::vector<IdGroup> inp_groups = inputGroups(currently_visiting_exprs);
        for (const IdGroup& inp_group : inp_groups) {
          to_visit_ids.pushBack(inp_group);
        }
      } else {
        still_to_visit_exprs.pushBack(currently_visiting_exprs);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (!to_visit_ids.empty()) {
      auto currently_visiting_ids = to_visit_ids.popFront();
      if (required_ind_exprs_ids.find(currently_visiting_ids) !=
          required_ind_exprs_ids.end()) {
        continue;
      }

      if (processIdGroup(currently_visiting_ids)) {
        something_was_processed = true;
        auto definitions_pair = getDefinitions(currently_visiting_ids);
        if (definitions_pair.second) {
          for (const ExprGroup& def : definitions_pair.first) {
            if (!all_exprs.has(def)) {
              continue;
            }
            if (required_ind_exprs_exprs.find(def) ==
                required_ind_exprs_exprs.end()) {
              to_visit_exprs.pushBack(def);
            }
          }
        }
      } else {
        still_to_visit_ids.pushBack(currently_visiting_ids);
      }
    }

    NVF_ERROR(
        something_was_processed ||
            (to_visit_ids.empty() && to_visit_exprs.empty()),
        "Infinite loop entered.");
  }

  // We want to traverse the expressions registered in required_ind_exprs_ids,
  // let's create a strict "uses path"
  std::unordered_map<IdGroup, ExprGroups> uses_path;
  for (const auto& entry : required_ind_exprs_ids) {
    const IdGroup& id = entry.first;
    const ExprGroups& traverse_exprs = entry.second;
    if (auto all_uses = getUses(id); all_uses.second) {
      uses_path[id] = traverse_exprs.intersect(all_uses.first);
    } else {
      uses_path[id] = {};
      continue;
    }
  }

  // Topologically sort the uses_path.
  ExprGroups sorted_exprs;
  ExprGroups to_visit_expr_groups;

  for (const IdGroup& inp : terminating_inputs) {
    auto use_it = uses_path.find(inp);
    if (use_it == uses_path.end()) {
      // This can happen for a trivial traversal where inputs and outputs are
      // exactly the same.
      continue;
    }
    const ExprGroups& uses = use_it->second;
    for (const ExprGroup& use : uses) {
      to_visit_expr_groups.pushBack(use);
    }
  }

  IdGroups visited = terminating_inputs;

  while (!to_visit_expr_groups.empty()) {
    bool something_processed = false;
    ExprGroups still_to_visit;
    while (!to_visit_expr_groups.empty()) {
      auto currently_visiting = to_visit_expr_groups.popFront();
      auto inputs = inputGroups(currently_visiting);
      if (std::all_of(inputs.begin(), inputs.end(), [&](IdGroup inp_id) {
            return visited.has(inp_id);
          })) {
        something_processed = true;
        sorted_exprs.pushBack(currently_visiting);
        auto outputs = outputGroups(currently_visiting);
        for (const IdGroup& out_id : outputs) {
          visited.pushBack(out_id);
          auto use_pair = getUses(out_id);
          if (!use_pair.second) {
            continue;
          }
          still_to_visit.pushBack(use_pair.first.intersect(all_exprs));
        }
      } else {
        still_to_visit.pushBack(currently_visiting);
      }
    }
    std::swap(to_visit_expr_groups, still_to_visit);
    NVF_ERROR(something_processed, "Infinite loop entered.");
  }

  return sorted_exprs;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const std::vector<IterDomain*>& from,
        const std::vector<IterDomain*>& to) const {
  std::unordered_map<IterDomain*, IdGroup> from_ids2set;

  for (auto from_id : from) {
    if (!hasGroup(from_id)) {
      continue;
    }
    from_ids2set[from_id] = toGroup(from_id);
  }

  // Map from the sets associated with the IterDomains in to, to those iter
  // domains
  std::unordered_map<IdGroup, VectorOfUniqueEntries<IterDomain*>> set2to_ids;

  for (auto to_id : to) {
    if (!hasGroup(to_id)) {
      continue;
    }
    auto to_set = toGroup(to_id);
    auto set2to_ids_it = set2to_ids.find(to_set);

    if (set2to_ids_it == set2to_ids.end()) {
      set2to_ids[to_set] = {to_id};
    } else {
      set2to_ids[to_set].pushBack(to_id);
    }
  }

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      from_ids2to_ids;
  for (auto from_id : from) {
    from_ids2to_ids[from_id] = VectorOfUniqueEntries<IterDomain*>();

    auto from_it = from_ids2set.find(from_id);
    NVF_ERROR(from_it != from_ids2set.end());

    auto from_set = from_it->second;
    auto to_entry_it = set2to_ids.find(from_set);
    if (to_entry_it == set2to_ids.end()) {
      continue;
    }
    from_ids2to_ids[from_id] = to_entry_it->second;
  }
  return from_ids2to_ids;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const VectorOfUniqueEntries<IterDomain*>& from,
        const VectorOfUniqueEntries<IterDomain*>& to) const {
  return buildMapBetween(from.vector(), to.vector());
}

std::pair<ExprGroups, bool> IdGraph::getDefinitions(
    const IdGroup& id_group) const {
  if (!id_group) {
    return {{}, false};
  }

  if (auto definitions_it = unique_definitions_.find(id_group);
      definitions_it != unique_definitions_.end()) {
    return std::make_pair(definitions_it->second, true);
  } else {
    return {{}, false};
  }
}

std::pair<ExprGroups, bool> IdGraph::getUses(const IdGroup& id_group) const {
  if (!id_group) {
    return {{}, false};
  }

  if (auto uses_it = unique_uses_.find(id_group);
      uses_it != unique_uses_.end()) {
    return std::make_pair(uses_it->second, true);
  } else {
    return {{}, false};
  }
}

std::string IdGraph::toString() const {
  std::stringstream ss;
  ss << "IdGraph { \n";
  ss << "Disjoint Ids:\n"
     << idGroupsString(*this, 1) << "\n\nDisjoint Expression groups:\n"
     << exprGroupsString(*this, 1) << std::endl;
  ss << " } IdGraph\n" << std::endl;
  return ss.str();
}

std::vector<std::vector<IterDomain*>> IdGraph::isTrivialExpr(Expr* expr) {
  std::vector<std::vector<IterDomain*>> mapped_ids;
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    if (merge->inner()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->outer(), merge->out()});
    }
    if (merge->outer()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->inner(), merge->out()});
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
        split->stopOffset()->isZeroInt()) {
      if (split->innerSplit()) {
        mapped_ids.push_back({split->in(), split->outer()});
      } else {
        mapped_ids.push_back({split->in(), split->inner()});
      }
    }
  } else if (auto swizzle = dynamic_cast<Swizzle2D*>(expr)) {
    if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle ||
        swizzle->swizzleMode() == SwizzleMode::NoSwizzle) {
      mapped_ids.push_back({swizzle->inX(), swizzle->outX()});
      mapped_ids.push_back({swizzle->inY(), swizzle->outY()});
    }
  }
  return mapped_ids;
}

bool IdGraph::transformAtributesMatch(Expr* first, Expr* second) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  NVF_ERROR(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Swizzle2D>() ||
          first->isA<Resize>(),
      "Unsupported rfactor expressions in compute at map:\n",
      first->toString());

  if (typeid(*first) != typeid(*second)) {
    return false;
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit() ||
        !first_split->startOffset()->sameAs(second_split->startOffset()) ||
        !first_split->stopOffset()->sameAs(second_split->stopOffset())) {
      return false;
    }
  }

  if (first->isA<Swizzle2D>()) {
    auto first_swizzle = first->as<Swizzle2D>();
    auto second_swizzle = second->as<Swizzle2D>();
    if (first_swizzle->swizzleMode() != second_swizzle->swizzleMode() ||
        first_swizzle->swizzleType() != second_swizzle->swizzleType()) {
      return false;
    }
  }

  // TODO: Resize properties

  return true;
}

void IdGraph::initializeId(
    IterDomain* id,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  const IdGroup& id_disjoint_set =
      disjointIdSets().initializeSet(id).first->second;

  ExprGroups def_groups;
  for (auto def : definitions) {
    const ExprGroup& expr_set =
        disjointExprSets().initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  // TODO-NM: def_groups can be empty. Should it be still mapped?
  // TODO-NM: Can this be overwritten?
  NVF_ERROR(unique_definitions_.emplace(id_disjoint_set, def_groups).second);

  ExprGroups use_groups;
  for (auto use : uses) {
    const ExprGroup& expr_set =
        disjointExprSets().initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  // TODO-NM: use_groups can be empty. Should it be still mapped?
  // TODO-NM: Can this be overwritten?
  NVF_ERROR(unique_uses_.emplace(id_disjoint_set, use_groups).second);
}

bool IdGraph::exprsMap(Expr* first, Expr* second, bool forward) const {
  if (!transformAtributesMatch(first, second)) {
    return false;
  }

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  // TODO-MN: Is this equivalent as
  // inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)) ?
  {
    for (const auto i : c10::irange(first_ids.size())) {
      if (!disjointIdSets().permissiveAreMapped(
              first_ids.at(i), second_ids.at(i))) {
        return false;
      }
    }
  }

  // Special handling for backprop of merge
  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one input actually
    // matches. This can be done on a map or extent basis.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->outer(), merge1->outer());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->inner(), merge1->inner());

    if (!(extent_0_match || extent_1_match)) {
      return false;
    }
  }

  // TODO: For now we're using same as, however we could know what val's are
  // exactly the same given the exact map. We might want to pipe that
  // information through to here.

  // TODO-NM: Should this be transformAtributesMatch?
  if (first->isA<Resize>()) {
    if (!first->as<Resize>()->leftExpand()->sameAs(
            second->as<Resize>()->leftExpand())) {
      return false;
    }

    if (!first->as<Resize>()->rightExpand()->sameAs(
            second->as<Resize>()->rightExpand())) {
      return false;
    }
  }

  return true;
}

const ExprGroups& IdGraph::getUniqueDefinitions(const IdGroup& group) const {
  auto unique_defs_it = unique_definitions_.find(group);
  NVF_ERROR(
      unique_defs_it != unique_definitions_.end(),
      "Definition not found for IdGroup: ",
      group->toString());
  return unique_defs_it->second;
}

const ExprGroups& IdGraph::getUniqueUses(const IdGroup& group) const {
  auto unique_uses_it = unique_uses_.find(group);
  NVF_ERROR(
      unique_uses_it != unique_uses_.end(),
      "Uses not found for IdGroup: ",
      group->toString());
  return unique_uses_it->second;
}

void IdGraph::mapIds(IterDomain* id0, IterDomain* id1) {
  if (id0 == id1) {
    return;
  }

  if (disjointIdSets().strictAreMapped(id0, id1)) {
    return;
  }
  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.
  IdGroup orig_id_group0 = toGroup(id0);
  IdGroup orig_id_group1 = toGroup(id1);
  const ExprGroups& orig_defs0 = getUniqueDefinitions(orig_id_group0);
  const ExprGroups& orig_defs1 = getUniqueDefinitions(orig_id_group1);
  const ExprGroups& orig_uses0 = getUniqueUses(orig_id_group0);
  const ExprGroups& orig_uses1 = getUniqueUses(orig_id_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdSets().mapEntries(id0, id1);
  auto new_id_group = toGroup(id0);

  unique_definitions_[new_id_group] = orig_defs0.computeUnion(orig_defs1);
  unique_uses_[new_id_group] = orig_uses0.computeUnion(orig_uses1);

  // Propagate on uses
  if (!orig_uses0.empty() && !orig_uses1.empty()) {
    for (const ExprGroup& use_group_1 : orig_uses1) {
      for (const ExprGroup& use_group_0 : orig_uses0) {
        if (use_group_0 == use_group_1) {
          continue;
        }
        Expr* use0 = use_group_0->front();
        Expr* use1 = use_group_1->front();
        maybeMapThroughExprs(use0, use1, true);
      }
    }
  }

  // Propagate on definitions
  if (!orig_defs0.empty() && !orig_defs1.empty()) {
    for (const ExprGroup& def_group_1 : orig_defs1) {
      for (const ExprGroup& def_group_0 : orig_defs0) {
        if (def_group_0 == def_group_1) {
          continue;
        }
        auto def0 = def_group_0->front();
        auto def1 = def_group_1->front();
        maybeMapThroughExprs(def0, def1, false);
      }
    }
  }

  unique_definitions_.erase(orig_id_group0);
  unique_definitions_.erase(orig_id_group1);
  unique_uses_.erase(orig_id_group0);
  unique_uses_.erase(orig_id_group1);
}

void IdGraph::maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward) {
  if (!exprsMap(expr0, expr1, forward)) {
    return;
  }

  // Expr inputs are mapped. If propagate_exprs_ is true, map the
  // exprs and outputs
  if (propagate_through_exprs_) {
    mapExprs(expr0, expr1);
    mapThroughExpr(expr0, expr1, forward);
  } else if (
      inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)) &&
      outputGroups(toGroup(expr0)) == outputGroups(toGroup(expr1))) {
    mapExprs(expr0, expr1);
  }
}

void IdGraph::mapExprs(Expr* expr0, Expr* expr1) {
  if (expr0 == expr1) {
    return;
  }

  if (disjointExprSets().strictAreMapped(expr0, expr1)) {
    return;
  }

  ExprGroup expr0_orig_group = toGroup(expr0);
  ExprGroup expr1_orig_group = toGroup(expr1);

  disjointExprSets().mapEntries(expr0, expr1);

  auto expr_new_group = toGroup(expr0);

  // Update unique uses of producers
  IdGroups producers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      producers.pushBack(toGroup(input_id));
    }
  }

  for (const IdGroup& producer_group : producers) {
    unique_uses_.at(producer_group).erase(expr0_orig_group);
    unique_uses_.at(producer_group).erase(expr1_orig_group);
    unique_uses_.at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  IdGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      consumers.pushBack(toGroup(output_id));
    }
  }

  for (const IdGroup& consumer_group : consumers) {
    unique_definitions_.at(consumer_group).erase(expr0_orig_group);
    unique_definitions_.at(consumer_group).erase(expr1_orig_group);
    unique_definitions_.at(consumer_group).pushBack(expr_new_group);
  }
}

bool IdGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
    return false;
  }

  NVF_ERROR(
      propagate_through_exprs_,
      "Asked to propagate expression mappings on a graph that has propagate_exprs_ disabled.");

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    mapIds(first_ids[out_i], second_ids[out_i]);
  }

  return true;
}

void IdGraph::mapThroughLoopSwizzles() {
  std::vector<Swizzle2D*> all_swizzles;

  for (const auto& expr_set : disjointExprSets().disjointSets()) {
    auto swizzles_in_expr_set = ir_utils::filterByType<Swizzle2D>(
        expr_set->vector().begin(), expr_set->vector().end());
    all_swizzles.insert(
        all_swizzles.end(),
        swizzles_in_expr_set.begin(),
        swizzles_in_expr_set.end());
  }

  for (auto swizzle : all_swizzles) {
    if (swizzle->swizzleMode() == SwizzleMode::Loop) {
      mapIds(swizzle->inX(), swizzle->outX());
      mapIds(swizzle->inY(), swizzle->outY());
    }
  }
}

void IdGraph::mapThroughTrivialExprs() {
  // Grab all expressions
  std::vector<Expr*> exprs;

  for (const auto& expr_group : disjointExprSets().disjointSets()) {
    for (auto expr : *expr_group) {
      exprs.push_back(expr);
    }
  }

  for (auto expr : exprs) {
    // If not trivial continue
    auto mapped_ids = IdGraph::isTrivialExpr(expr);
    if (mapped_ids.empty()) {
      continue;
    }

    // Map through trivial expressions
    for (auto mapped_id_group : mapped_ids) {
      for (auto id : mapped_id_group) {
        mapIds(mapped_id_group.front(), id);
      }
    }
  }
}

void IdGraph::removeTrivialExprs() {
  ExprGroups trivial_expr_groups;
  // This seems like it shouls just be a copy if.
  for (const ExprGroup& expr_group : disjointExprSets().disjointSets()) {
    if (isTrivialExprGroup(expr_group)) {
      trivial_expr_groups.pushBack(expr_group);
    }
  }

  // Clear out expressions that map inputs and outputs to the same group
  // from definitions and uses. They shouldn't be important in traversal, and
  // will break the terminal input/terminal output logic of traversal. Similar
  // to what's drafted in buildIndexGraph
  for (const ExprGroup& trivial_expr_group : trivial_expr_groups) {
    // Complexity of erase not good as both disjoint set and vector of unique
    // entries require a vector find to erase an entry.
    eraseExprGroup(trivial_expr_group);
  }
}

// Complexity here is not great. We might want a better complexity version when
// erasing multiple expr_groups.
void IdGraph::eraseExprGroup(const ExprGroup& expr_group) {
  // Erase entries that exist in unique_definitions_ and unique_uses_
  for (const IdGroup& id_group : disjointIdSets().disjointSets()) {
    // Make sure the entries exists
    NVF_ERROR(
        unique_definitions_.find(id_group) != unique_definitions_.end(),
        "Broken definitions, couldn't find entry for id group, ",
        nvfuser::toString(id_group, 0, true));
    NVF_ERROR(
        unique_uses_.find(id_group) != unique_uses_.end(),
        "Broken uses, couldn't find entry for id group, ",
        nvfuser::toString(id_group, 0, true));

    unique_definitions_[id_group].erase(expr_group);
    unique_uses_[id_group].erase(expr_group);
  }

  for (auto expr : *expr_group) {
    disjoint_exprs_.erase(expr);
  }
}

bool IdGraph::isTrivialExprGroup(const ExprGroup& expr_group) const {
  return !IdGroups(inputGroups(expr_group))
              .intersect(IdGroups(outputGroups(expr_group)))
              .empty();
}

} // namespace nvfuser
