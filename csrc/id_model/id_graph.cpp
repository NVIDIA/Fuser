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
    if (const ExprGroups* group_defs = getDefinitions(of_id_group);
        group_defs != nullptr) {
      to_visit.insert(to_visit.end(), group_defs->begin(), group_defs->end());
    }
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const IdGroup& input_id : inputGroups(current_expr)) {
      if (const ExprGroups* group_defs = getDefinitions(input_id);
          group_defs != nullptr) {
        for (const ExprGroup& group_def : *group_defs) {
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

bool IdGraph::hasUses(const IdGroup& id_group) const {
  NVF_ERROR(id_group);
  return unique_uses_.find(id_group) != unique_uses_.end();
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

    auto extent_o_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->outer(), merge1->outer());

    auto extent_i_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->inner(), merge1->inner());

    if (!(extent_o_match || extent_i_match)) {
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

const ExprGroups* IdGraph::getDefinitions(const IdGroup& group) const {
  NVF_ERROR(group, "Nullptr not allowed");
  auto unique_defs_it = unique_definitions_.find(group);
  if (unique_defs_it == unique_definitions_.end()) {
    return nullptr;
  }

  return &(unique_defs_it->second);
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
  const ExprGroups* orig_defs0 = getDefinitions(orig_id_group0);
  NVF_ERROR(orig_defs0);
  const ExprGroups* orig_defs1 = getDefinitions(orig_id_group1);
  NVF_ERROR(orig_defs1);
  const ExprGroups& orig_uses0 = getUniqueUses(orig_id_group0);
  const ExprGroups& orig_uses1 = getUniqueUses(orig_id_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdSets().mapEntries(id0, id1);
  auto new_id_group = toGroup(id0);

  unique_definitions_[new_id_group] = orig_defs0->computeUnion(*orig_defs1);
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
  if (!orig_defs0->empty() && !orig_defs1->empty()) {
    for (const ExprGroup& def_group_1 : *orig_defs1) {
      for (const ExprGroup& def_group_0 : *orig_defs0) {
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

  const ExprGroup& expr0_orig_group = toGroup(expr0);
  const ExprGroup& expr1_orig_group = toGroup(expr1);

  disjointExprSets().mapEntries(expr0, expr1);

  const ExprGroup& expr_new_group = toGroup(expr0);

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

} // namespace nvfuser
