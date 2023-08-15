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

IdGraph::IdGraph(const IdGraph& other)
    : disjoint_ids_(other.disjoint_ids_),
      disjoint_exprs_(other.disjoint_exprs_),
      view_rfactor_ids_(other.view_rfactor_ids_),
      unique_definitions_(),
      unique_uses_() {
  for (auto orig_unique_def_pair : other.unique_definitions_) {
    auto orig_id_group = orig_unique_def_pair.first;
    auto orig_expr_groups = orig_unique_def_pair.second;
    auto new_id_group = toGroup(orig_id_group->front());

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    unique_definitions_[new_id_group] = new_expr_groups;
  }

  for (auto orig_unique_use_pair : other.unique_uses_) {
    auto orig_id_group = orig_unique_use_pair.first;
    auto orig_expr_groups = orig_unique_use_pair.second;
    auto new_id_group = toGroup(orig_id_group->front());

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
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
  view_rfactor_ids_.clear();
  IdGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

const DisjointSets<IterDomain*>& IdGraph::disjointIdSets() const {
  return disjoint_ids_;
}

DisjointSets<IterDomain*>& IdGraph::disjointIdSets() {
  return disjoint_ids_;
}

const DisjointSets<Expr*>& IdGraph::disjointExprSets() const {
  return disjoint_exprs_;
}

DisjointSets<Expr*>& IdGraph::disjointExprSets() {
  return disjoint_exprs_;
}

// Return if there's a group entry in the graph for this expr
bool IdGraph::hasGroup(Expr* expr) const {
  return disjoint_exprs_.mappingExists(expr);
}

// Return if there's a group entry in the graph for this id
bool IdGraph::hasGroup(IterDomain* id) const {
  return disjoint_ids_.mappingExists(id);
}

ExprGroup IdGraph::toGroup(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  TORCH_INTERNAL_ASSERT(
      disjoint_set_it != disjoint_exprs_.disjointSetMap().end(),
      "\nExpr group could not be found in graph associated with: ",
      expr->toString());
  return disjoint_set_it->second;
}

IdGroup IdGraph::toGroup(IterDomain* id) const {
  auto disjoint_set_it = disjoint_ids_.disjointSetMap().find(id);
  TORCH_INTERNAL_ASSERT(
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

std::vector<IdGroup> IdGraph::outputGroups(ExprGroup expr) const {
  std::vector<IdGroup> output_groups;
  for (auto id_output :
       ir_utils::filterByType<IterDomain>(expr->front()->outputs())) {
    output_groups.push_back(toGroup(id_output));
  }
  return output_groups;
}

std::vector<IdGroup> IdGraph::inputGroups(ExprGroup expr) const {
  std::vector<IdGroup> input_groups;
  for (auto id_input :
       ir_utils::filterByType<IterDomain>(expr->front()->inputs())) {
    input_groups.push_back(toGroup(id_input));
  }
  return input_groups;
}

ExprGroups IdGraph::allUsesOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_uses_pair = iterDomainGroupUses(of_id_group);
    if (group_uses_pair.second) {
      to_visit.pushBack(group_uses_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto output_ids = outputGroups(current_expr);
    for (auto output_id : output_ids) {
      auto group_uses_pair = iterDomainGroupUses(output_id);
      if (!group_uses_pair.second) {
        continue;
      }
      for (auto group_use : group_uses_pair.first) {
        if (visited.has(group_use)) {
          continue;
        }
        to_visit.pushBack(group_use);
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::allDefinitionsOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_defs_pair = iterDomainGroupDefinitions(of_id_group);
    if (group_defs_pair.second) {
      to_visit.pushBack(group_defs_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto input_ids = inputGroups(current_expr);
    for (auto input_id : input_ids) {
      auto group_defs_pair = iterDomainGroupDefinitions(input_id);
      if (!group_defs_pair.second) {
        continue;
      }
      for (auto group_def : group_defs_pair.first) {
        if (visited.has(group_def)) {
          continue;
        }
        to_visit.pushBack(group_def);
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::getExprsBetween(const IdGroups& from, const IdGroups& to)
    const {
  auto all_uses_of_from = allUsesOf(from);
  auto all_definitions_of_to = allDefinitionsOf(to);

  // All of the expressions between from and to. Not all will be used as we
  // just want to define each iter domain group once.
  auto all_exprs = all_uses_of_from.intersect(all_definitions_of_to);

  // There could be IterDomains in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;
  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    IdGroups all_id_groups;

    for (auto expr_group : all_exprs) {
      auto inp_groups = inputGroups(expr_group);
      auto out_groups = outputGroups(expr_group);
      if (IdGroups(inp_groups).intersect(IdGroups(out_groups)).size() > 0) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      all_id_groups.pushBack(inp_groups);

      if (!inp_groups.empty()) {
        not_outputs.pushBack(inp_groups);
      }

      all_id_groups.pushBack(out_groups);

      if (!out_groups.empty()) {
        not_inputs.pushBack(out_groups);
      }
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
  auto outputsVisited = [&](ExprGroup expr) {
    for (auto id_group : outputGroups(expr)) {
      if (required_ind_exprs_ids.find(id_group) ==
          required_ind_exprs_ids.end()) {
        return false;
      }
    }
    return true;
  };

#if 0
  auto allIdUsesVisisted = [&](IdGroup id) {
    auto uses_pair = iterDomainGroupUses(id);
    if (!uses_pair.second) {
      return true;
    }
    for (auto use_group : uses_pair.first) {
      if (all_exprs.has(use_group)) {
        if (required_ind_exprs_exprs.find(use_group) ==
            required_ind_exprs_exprs.end()) {
          return false;
        }
      }
    }
    return true;
  };
#endif

  // Returns all expression groups in required_ind_exprs_ids of outputs
  auto requiredExprsOutputs = [&](ExprGroup expr) {
    ExprGroups all_output_required_exprs;
    for (auto id_group : outputGroups(expr)) {
      auto id_group_exprs_it = required_ind_exprs_ids.find(id_group);
      TORCH_INTERNAL_ASSERT(
          id_group_exprs_it != required_ind_exprs_ids.end(),
          "Failure in Iter Domain Graph index resolution, count expected for group: ",
          id_group->toString());
      all_output_required_exprs.pushBack(id_group_exprs_it->second);
    }
    return all_output_required_exprs;
  };

  auto processExpr = [&](ExprGroup expr) {
    if (!outputsVisited(expr)) {
      return false;
    }
    // Accumulate expressions from all outputs add this expression and set it
    // as current expressions required indexing expressions.
    required_ind_exprs_exprs[expr] = requiredExprsOutputs(expr);
    return true;
  };

  auto processId = [&](IdGroup id) {
    // Track if we've grabed any of the uses required indexing expressions.
    bool initialized = false;
    // Expression group of all indexing expressions required for this iter
    // domain coming back from any of its uses.
    ExprGroups min_groups;

    auto uses_pair = iterDomainGroupUses(id);
    if (!uses_pair.second) {
      // No expressions required for this iter domain, it must be a
      // terminating output.
      required_ind_exprs_ids[id] = min_groups;
      return true;
    }

    // Only worry about expressions between inputs and outputs we're
    // looking at.
    for (auto use_group : uses_pair.first.intersect(all_exprs)) {
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
    required_ind_exprs_ids[id] = min_groups;
    return true;
  };

  IdGroups to_visit_ids = terminating_outputs;
  ExprGroups to_visit_exprs;

  while (to_visit_ids.size() > 0 || to_visit_exprs.size() > 0) {
    // Process expressions first as all uses of iter domains have to be
    // processed before we can process that iter domain.

    // Try to detect when nothing has been processed which would put us in an
    // infinite loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;
    while (to_visit_exprs.size() > 0) {
      auto currently_visiting = to_visit_exprs.popFront();
      if (required_ind_exprs_exprs.find(currently_visiting) !=
          required_ind_exprs_exprs.end()) {
        continue;
      }
      if (processExpr(currently_visiting)) {
        something_was_processed = true;
        auto inp_groups = inputGroups(currently_visiting);
        for (auto inp_group : inp_groups) {
          to_visit_ids.pushBack(inp_group);
        }
      } else {
        still_to_visit_exprs.pushBack(currently_visiting);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (to_visit_ids.size() > 0) {
      auto currently_visiting = to_visit_ids.popFront();
      if (required_ind_exprs_ids.find(currently_visiting) !=
          required_ind_exprs_ids.end()) {
        continue;
      }

      if (processId(currently_visiting)) {
        something_was_processed = true;
        auto definitions_pair = iterDomainGroupDefinitions(currently_visiting);
        if (definitions_pair.second) {
          for (auto def : definitions_pair.first) {
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
        still_to_visit_ids.pushBack(currently_visiting);
      }
    }

    TORCH_INTERNAL_ASSERT(
        something_was_processed ||
            (to_visit_ids.size() == 0 && to_visit_exprs.size() == 0),
        "Infinite loop entered.");
  }

  // We want to traverse the expressions registered in required_ind_exprs_ids,
  // let's create a strict "uses path"
  std::unordered_map<IdGroup, ExprGroups> uses_path;
  for (auto entry : required_ind_exprs_ids) {
    auto id = entry.first;
    auto traverse_exprs = entry.second;
    auto all_uses = iterDomainGroupUses(id);
    if (all_uses.second) {
      uses_path[id] = traverse_exprs.intersect(all_uses.first);
    } else {
      uses_path[id] = {};
      continue;
    }
  }

  // Topologically sort the uses_path.
  ExprGroups sorted_exprs;
  ExprGroups to_visit;

  for (auto inp : terminating_inputs) {
    auto use_it = uses_path.find(inp);
    if (use_it == uses_path.end()) {
      // This can happen for a trivial traversal where inputs and outputs are
      // exactly the same.
      continue;
    }
    auto uses = use_it->second;
    for (auto use : uses) {
      to_visit.pushBack(use);
    }
  }

  IdGroups visited = terminating_inputs;

  while (to_visit.size() > 0) {
    bool something_processed = false;
    ExprGroups still_to_visit;
    while (to_visit.size() > 0) {
      auto currently_visiting = to_visit.popFront();
      auto inputs = inputGroups(currently_visiting);
      if (std::all_of(inputs.begin(), inputs.end(), [&](IdGroup inp_id) {
            return visited.has(inp_id);
          })) {
        something_processed = true;
        sorted_exprs.pushBack(currently_visiting);
        auto outputs = outputGroups(currently_visiting);
        for (auto out_id : outputs) {
          visited.pushBack(out_id);
          auto use_pair = iterDomainGroupUses(out_id);
          if (!use_pair.second) {
            continue;
          }
          still_to_visit.pushBack(use_pair.first.intersect(all_exprs));
        }
      } else {
        still_to_visit.pushBack(currently_visiting);
      }
    }
    std::swap(to_visit, still_to_visit);
    TORCH_INTERNAL_ASSERT(something_processed, "Infinite loop entered.");
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
    TORCH_INTERNAL_ASSERT(from_it != from_ids2set.end());

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

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupDefinitions(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto definitions_it = unique_definitions_.find(id_group);
  if (definitions_it == unique_definitions_.end()) {
    return null_return;
  }

  return std::make_pair(definitions_it->second, true);
}

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupUses(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto uses_it = unique_uses_.find(id_group);
  if (uses_it == unique_uses_.end()) {
    return null_return;
  }

  return std::make_pair(uses_it->second, true);
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

  TORCH_INTERNAL_ASSERT(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Swizzle2D>() ||
          first->isA<Resize>(),
      "Merge and split are the only expressions supported through rfactor operations in compute at map, but found:\n",
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

  return true;
}

void IdGraph::initializeId(
    IterDomain* id,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  auto id_disjoint_set = disjointIdSets().initializeSet(id).first->second;

  ExprGroups def_groups;
  for (auto def : definitions) {
    auto expr_set = disjointExprSets().initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  unique_definitions_[id_disjoint_set] = def_groups;

  ExprGroups use_groups;
  for (auto use : uses) {
    auto expr_set = disjointExprSets().initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  unique_uses_[id_disjoint_set] = use_groups;
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

  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !disjointIdSets().permissiveAreMapped(
                  id_pair.first, id_pair.second);
            })) {
      return false;
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

ExprGroups IdGraph::uniqueDefinitions(IdGroup group) const {
  auto unique_defs_it = unique_definitions_.find(group);
  TORCH_INTERNAL_ASSERT(
      unique_defs_it != unique_definitions_.end(),
      "Definition not found for IdGroup: ",
      group->toString());
  return unique_defs_it->second;
}

ExprGroups IdGraph::uniqueUses(IdGroup group) const {
  auto unique_uses_it = unique_uses_.find(group);
  TORCH_INTERNAL_ASSERT(
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
  auto orig_id_group0 = toGroup(id0);
  auto orig_id_group1 = toGroup(id1);
  ExprGroups orig_defs0 = uniqueDefinitions(orig_id_group0);
  ExprGroups orig_defs1 = uniqueDefinitions(orig_id_group1);
  ExprGroups orig_uses0 = uniqueUses(orig_id_group0);
  ExprGroups orig_uses1 = uniqueUses(orig_id_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdSets().mapEntries(id0, id1);
  auto new_id_group = toGroup(id0);

  unique_definitions_.erase(orig_id_group0);
  unique_definitions_.erase(orig_id_group1);
  unique_uses_.erase(orig_id_group0);
  unique_uses_.erase(orig_id_group1);

  unique_definitions_[new_id_group] = orig_defs0.computeUnion(orig_defs1);
  unique_uses_[new_id_group] = orig_uses0.computeUnion(orig_uses1);

  // Propagate on uses
  if (orig_uses0.size() > 0 || orig_uses1.size() > 0) {
    if (orig_uses0.size() > 0 && orig_uses1.size() > 0) {
      for (auto use_group_1 : orig_uses1) {
        if (orig_uses0.has(use_group_1)) {
          continue;
        }

        for (auto use_group_0 : orig_uses0) {
          auto use0 = use_group_0->front();
          auto use1 = use_group_1->front();
          maybeMapThroughExprs(use0, use1, true);
        }
      }
    }
  }

  // Propagate on definitions
  if (orig_defs0.size() > 0 || orig_defs1.size() > 0) {
    if (orig_defs0.size() > 0 && orig_defs1.size() > 0) {
      for (auto def_group_1 : orig_defs1) {
        if (orig_defs0.has(def_group_1)) {
          continue;
        }

        for (auto def_group_0 : orig_defs0) {
          auto def0 = def_group_0->front();
          auto def1 = def_group_1->front();
          maybeMapThroughExprs(def0, def1, false);
        }
      }
    }
  }
}

void IdGraph::maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward) {
  if (exprsMap(expr0, expr1, forward)) {
    if (propagate_exprs_) {
      mapExprs(expr0, expr1);
      mapThroughExpr(expr0, expr1, forward);
    } else if (
        inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)) &&
        outputGroups(toGroup(expr0)) == outputGroups(toGroup(expr1))) {
      mapExprs(expr0, expr1);
    }
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

  for (auto producer_group : producers) {
    uniqueUses().at(producer_group).erase(expr0_orig_group);
    uniqueUses().at(producer_group).erase(expr1_orig_group);
    uniqueUses().at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  IdGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      consumers.pushBack(toGroup(output_id));
    }
  }

  for (auto consumer_group : consumers) {
    uniqueDefinitions().at(consumer_group).erase(expr0_orig_group);
    uniqueDefinitions().at(consumer_group).erase(expr1_orig_group);
    uniqueDefinitions().at(consumer_group).pushBack(expr_new_group);
  }
}

bool IdGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      propagate_exprs_,
      "Asked to propagate expression mappings on a graph that has propagate_exprs_ disabled.");

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  TORCH_INTERNAL_ASSERT(
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

  for (auto expr_set : disjointExprSets().disjointSets()) {
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

  for (auto expr_group : disjointExprSets().disjointSets()) {
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
  for (auto expr_group : disjointExprSets().disjointSets()) {
    if (isTrivialExprGroup(expr_group)) {
      trivial_expr_groups.pushBack(expr_group);
    }
  }

  // Clear out expressions that map inputs and outputs to the same group
  // from definitions and uses. They shouldn't be important in traversal, and
  // will break the terminal input/terminal output logic of traversal. Similar
  // to what's drafted in buildIndexGraph
  for (auto trivial_expr_group : trivial_expr_groups) {
    // Complexity of erase not good as both disjoint set and vector of unique
    // entries require a vector find to erase an entry.
    eraseExprGroup(trivial_expr_group);
  }
}

// Complexity here is not great. We might want a better complexity version when
// erasing multiple expr_groups.
void IdGraph::eraseExprGroup(ExprGroup expr_group) {
  // Erase entries that exist in unique_definitions_ and unique_uses_
  for (auto id_group : disjointIdSets().disjointSets()) {
    // Make sure the entries exists
    TORCH_INTERNAL_ASSERT(
        unique_definitions_.find(id_group) != unique_definitions_.end(),
        "Broken definitions, couldn't find entry for id group, ",
        nvfuser::toString(id_group, 0, true));
    TORCH_INTERNAL_ASSERT(
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

bool IdGraph::isTrivialExprGroup(ExprGroup expr_group) const {
  return !IdGroups(inputGroups(expr_group))
              .intersect(IdGroups(outputGroups(expr_group)))
              .empty();
}

} // namespace nvfuser
