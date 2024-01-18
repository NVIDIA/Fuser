// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/to_string.h>
#include <ir/utils.h>
#include <val_graph.h>

namespace nvfuser {

namespace {
using UnorderedSetOfExprGroup = std::unordered_set<ExprGroup>;
using DequeOfExprGroup = std::deque<ExprGroup>;
} // namespace

ValGraph::ValGraph(const ValGraph& other)
    : disjoint_vals_(other.disjoint_vals_),
      disjoint_exprs_(other.disjoint_exprs_),
      unique_definitions_(),
      unique_uses_() {
  for (const auto& [orig_val_group, orig_expr_groups] :
       other.unique_definitions_) {
    auto new_val_group = toGroup(orig_val_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    NVF_ERROR(
        unique_definitions_.emplace(new_val_group, std::move(new_expr_groups))
            .second);
  }

  for (const auto& [orig_val_group, orig_expr_groups] : other.unique_uses_) {
    auto new_val_group = toGroup(orig_val_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    NVF_ERROR(
        unique_uses_.emplace(new_val_group, std::move(new_expr_groups)).second);
  }
}

ValGraph& ValGraph::operator=(const ValGraph& other) {
  ValGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

// Return if there's a group entry in the graph for this expr
bool ValGraph::hasGroup(Expr* expr) const {
  return disjoint_exprs_.mappingExists(expr);
}

// Return if there's a group entry in the graph for this val
bool ValGraph::hasGroup(Val* val) const {
  return disjoint_vals_.mappingExists(val);
}

const ExprGroup& ValGraph::toGroup(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  NVF_ERROR(
      disjoint_set_it != disjoint_exprs_.disjointSetMap().end(),
      "\nExpr group could not be found in graph associated with: ",
      expr->toString());
  return disjoint_set_it->second;
}

const ValGroup& ValGraph::toGroup(Val* val) const {
  auto disjoint_set_it = disjoint_vals_.disjointSetMap().find(val);
  NVF_ERROR(
      disjoint_set_it != disjoint_vals_.disjointSetMap().end(),
      "\nId group could not be found in graph associated with: ",
      val->toString(),
      "\n");
  return disjoint_set_it->second;
}

std::vector<ValGroup> ValGraph::outputGroups(const ExprGroup& expr) const {
  std::vector<ValGroup> output_groups;
  for (auto id_output : expr->front()->outputs()) {
    output_groups.push_back(toGroup(id_output));
  }
  return output_groups;
}

std::vector<ValGroup> ValGraph::inputGroups(const ExprGroup& expr) const {
  std::vector<ValGroup> input_groups;
  for (auto id_input : expr->front()->inputs()) {
    input_groups.push_back(toGroup(id_input));
  }
  return input_groups;
}

ExprGroups ValGraph::allUsesOf(const ValGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const ValGroup& of_val_group : of) {
    const ExprGroups& group_uses = getUses(of_val_group);
    to_visit.insert(to_visit.end(), group_uses.begin(), group_uses.end());
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const ValGroup& output_group : outputGroups(current_expr)) {
      for (const ExprGroup& group_use : getUses(output_group)) {
        if (visited.count(group_use)) {
          continue;
        }
        to_visit.push_back(group_use);
      }
    }
  }

  return visited;
}

ExprGroups ValGraph::allDefinitionsOf(const ValGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const ValGroup& of_val_group : of) {
    const ExprGroups& group_defs = getDefinitions(of_val_group);
    to_visit.insert(to_visit.end(), group_defs.begin(), group_defs.end());
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const ValGroup& input_id : inputGroups(current_expr)) {
      for (const ExprGroup& group_def : getDefinitions(input_id)) {
        if (visited.count(group_def)) {
          continue;
        }
        to_visit.push_back(group_def);
      }
    }
  }

  return visited;
}

bool ValGraph::hasDefinitions(const ValGroup& val_group) const {
  NVF_ERROR(val_group);
  return unique_definitions_.find(val_group) != unique_definitions_.end();
}

bool ValGraph::hasUses(const ValGroup& val_group) const {
  NVF_ERROR(val_group);
  return unique_uses_.find(val_group) != unique_uses_.end();
}

std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> ValGraph::buildMapBetween(
    const std::vector<Val*>& from,
    const std::vector<Val*>& to) const {
  // Map from the sets associated with the Vals in to, to those Vals
  std::unordered_map<ValGroup, VectorOfUniqueEntries<Val*>> set2to_vals;

  for (auto to_val : to) {
    if (!hasGroup(to_val)) {
      continue;
    }
    const auto& to_set = toGroup(to_val);
    set2to_vals[to_set].pushBack(to_val);
  }

  std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> from_vals2to_vals;
  for (auto from_val : from) {
    // Initialize in case no to val is mapped
    from_vals2to_vals[from_val] = VectorOfUniqueEntries<Val*>();

    if (!hasGroup(from_val)) {
      continue;
    }

    const ValGroup& from_set = toGroup(from_val);

    auto to_entry_it = set2to_vals.find(from_set);
    if (to_entry_it == set2to_vals.end()) {
      continue;
    }

    from_vals2to_vals[from_val] = to_entry_it->second;
  }
  return from_vals2to_vals;
}

std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> ValGraph::buildMapBetween(
    const VectorOfUniqueEntries<Val*>& from,
    const VectorOfUniqueEntries<Val*>& to) const {
  return buildMapBetween(from.vector(), to.vector());
}

std::string ValGraph::toString() const {
  std::stringstream ss;
  ss << "IdGraph { \n";
  ss << "Disjoint Ids:\n"
     << idGroupsString(*this, 1) << "\n\nDisjoint Expression groups:\n"
     << exprGroupsString(*this, 1) << std::endl;
  ss << " } IdGraph\n" << std::endl;
  return ss.str();
}

void ValGraph::initializeVal(
    Val* val,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  const ValGroup& val_disjoint_set =
      disjoint_vals_.initializeSet(val).first->second;

  // For now, the definition of a val should be unique. Remove this
  // assertion as necessary
  NVF_ERROR(definitions.size() <= 1);

  ExprGroups def_groups;
  for (auto def : definitions) {
    const ExprGroup& expr_set =
        disjoint_exprs_.initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  // TODO-NM: def_groups can be empty. Should it be still mapped?
  NVF_ERROR(
      unique_definitions_.emplace(val_disjoint_set, def_groups).second,
      "Multiple defining groups for ",
      nvfuser::toString(val_disjoint_set));

  ExprGroups use_groups;
  for (auto use : uses) {
    const ExprGroup& expr_set =
        disjoint_exprs_.initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  // TODO-NM: use_groups can be empty. Should it be still mapped?
  NVF_ERROR(
      unique_uses_.emplace(val_disjoint_set, use_groups).second,
      "Multiple use groups for ",
      nvfuser::toString(val_disjoint_set));
}

void ValGraph::initializeVal(Val* val) {
  VectorOfUniqueEntries<Expr*> defs;
  if (val->definition()) {
    defs.pushBack(val->definition());
  }
  VectorOfUniqueEntries<Expr*> uses;
  for (Expr* use : val->uses()) {
    uses.pushBack(use);
  }
  initializeVal(val, defs, uses);
}

bool ValGraph::exprsMap(Expr* first, Expr* second, bool forward) const {
  NVF_ERROR(first);
  NVF_ERROR(second);

  if (!first->sameOp(second)) {
    return false;
  }

  std::vector<Val*> first_vals = forward ? first->inputs() : first->outputs();
  std::vector<Val*> second_vals =
      forward ? second->inputs() : second->outputs();

  NVF_ERROR(
      first_vals.size() == second_vals.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  for (const auto i : c10::irange(first_vals.size())) {
    if (!disjointValSets().permissiveAreMapped(
            first_vals.at(i), second_vals.at(i))) {
      return false;
    }
  }

  // Special handling for backprop of merge
  if (first->isA<Merge>() && !forward) {
    if (!shouldMapMergeBackward<Val>(
            first->as<Merge>(), second->as<Merge>(), this->disjointValSets())) {
      return false;
    }
  }

  // TODO: For now we're using same as, however we could know what val's are
  // exactly the same given the exact map. We might want to pipe that
  // information through to here.

  return true;
}

const ExprGroups& ValGraph::getDefinitions(const ValGroup& val_group) const {
  NVF_ERROR(val_group, "Nullptr not allowed");
  const auto it = unique_definitions_.find(val_group);
  NVF_ERROR(
      it != unique_definitions_.end(),
      "Definition group not found for ",
      nvfuser::toString(val_group));
  return it->second;
}

const ExprGroups& ValGraph::getUses(const ValGroup& val_group) const {
  NVF_ERROR(val_group, "Nullptr not allowed");
  const auto it = unique_uses_.find(val_group);
  NVF_ERROR(
      it != unique_uses_.end(),
      "Use group not found for ",
      nvfuser::toString(val_group));
  return it->second;
}

void ValGraph::mapVals(Val* val0, Val* val1) {
  if (val0 == val1) {
    return;
  }

  if (disjointValSets().strictAreMapped(val0, val1)) {
    return;
  }

  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.
  const ValGroup orig_val_group0 = toGroup(val0);
  const ValGroup orig_val_group1 = toGroup(val1);

  // Note that getDefinitions and getUses return references, which
  // will be invalidated once unique_definitions_ and unique_uses_ are
  // updated
  const ExprGroups orig_defs0 = getDefinitions(orig_val_group0);
  const ExprGroups orig_defs1 = getDefinitions(orig_val_group1);
  const ExprGroups orig_uses0 = getUses(orig_val_group0);
  const ExprGroups orig_uses1 = getUses(orig_val_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjoint_vals_.mapEntries(val0, val1);
  auto new_val_group = toGroup(val0);

  unique_definitions_[new_val_group] = orig_defs0.computeUnion(orig_defs1);
  unique_uses_[new_val_group] = orig_uses0.computeUnion(orig_uses1);

  // Propagate on uses
  if (!orig_uses0.empty() && !orig_uses1.empty()) {
    for (const ExprGroup& use_group_1 : orig_uses1) {
      NVF_ERROR(use_group_1.get() != nullptr);
      NVF_ERROR(!use_group_1->empty());
      for (const ExprGroup& use_group_0 : orig_uses0) {
        NVF_ERROR(use_group_0.get() != nullptr);
        NVF_ERROR(!use_group_0->empty());
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
      NVF_ERROR(def_group_1.get() != nullptr);
      NVF_ERROR(!def_group_1->empty());
      for (const ExprGroup& def_group_0 : orig_defs0) {
        NVF_ERROR(def_group_0.get() != nullptr);
        NVF_ERROR(!def_group_0->empty());
        if (def_group_0 == def_group_1) {
          continue;
        }
        auto def0 = def_group_0->front();
        auto def1 = def_group_1->front();
        maybeMapThroughExprs(def0, def1, false);
      }
    }
  }

  unique_definitions_.erase(orig_val_group0);
  unique_definitions_.erase(orig_val_group1);
  unique_uses_.erase(orig_val_group0);
  unique_uses_.erase(orig_val_group1);
}

void ValGraph::maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward) {
  // By default, expressions are mapped only when everything is
  // matched, i.e., inputs, outputs and attributes are all mapped or
  // equal. When the propagation is allowed, as long as the inputs are
  // mapped and the attributes are equal, we propagate the mappings to
  // the outputs and the expressions.
  // In either case, it should be always true that when two
  // expressions are mapped, their inputs and outputs are also mapped,
  // respectively, and vice versa.

  if (!exprsMap(expr0, expr1, forward)) {
    return;
  }

  // Expr inputs are mapped. If propagate_through_exprs_ is true, map the
  // exprs and outputs. If not, map the exprs only when both inputs
  // and outputs are mapped. Since exprsMap makes sure inputs or
  // outputs are mapped, only outputs or inputs need to be checked
  if (propagate_through_exprs_) {
    mapExprs(expr0, expr1);
    mapThroughExpr(expr0, expr1, forward);
  } else if (
      (forward &&
       outputGroups(toGroup(expr0)) == outputGroups(toGroup(expr1))) ||
      (!forward &&
       inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)))) {
    mapExprs(expr0, expr1);
  }
}

void ValGraph::mapExprs(Expr* expr0, Expr* expr1) {
  if (expr0 == expr1) {
    return;
  }

  if (disjointExprSets().strictAreMapped(expr0, expr1)) {
    return;
  }

  // Note that non-reference copies are required here as they may be
  // removed by mapEntries
  const ExprGroup expr0_orig_group = toGroup(expr0);
  const ExprGroup expr1_orig_group = toGroup(expr1);

  disjoint_exprs_.mapEntries(expr0, expr1);

  const ExprGroup& expr_new_group = toGroup(expr0);

  // Update unique uses of producers
  ValGroups producers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto input : expr->inputs()) {
      producers.pushBack(toGroup(input));
    }
  }

  for (const ValGroup& producer_group : producers) {
    unique_uses_.at(producer_group).erase(expr0_orig_group);
    unique_uses_.at(producer_group).erase(expr1_orig_group);
    unique_uses_.at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  ValGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output : expr->outputs()) {
      consumers.pushBack(toGroup(output));
    }
  }

  for (const ValGroup& consumer_group : consumers) {
    unique_definitions_.at(consumer_group).erase(expr0_orig_group);
    unique_definitions_.at(consumer_group).erase(expr1_orig_group);
    unique_definitions_.at(consumer_group).pushBack(expr_new_group);
  }
}

bool ValGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
    return false;
  }

  NVF_ERROR(
      propagate_through_exprs_,
      "Asked to propagate expression mappings on a graph that has propagate_exprs_ disabled.");

  const auto& first_ids = forward ? first->outputs() : first->inputs();
  const auto& second_ids = forward ? second->outputs() : second->inputs();

  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    mapVals(first_ids[out_i], second_ids[out_i]);
  }

  return true;
}

} // namespace nvfuser
