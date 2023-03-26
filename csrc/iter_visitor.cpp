// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iter_visitor.h>

#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <type.h>

namespace nvfuser {

/* ITER VISITOR */

namespace {

// Remove any stmt in stmts that is in visited
void remove_visited(
    std::vector<Statement*>& stmts,
    const std::unordered_set<Statement*>& visited) {
  std::deque<std::vector<Statement*>::iterator> to_erase;
  for (auto it = stmts.begin(); it != stmts.end(); it++) {
    if (visited.find(*it) != visited.end()) {
      to_erase.push_back(it);
    }
  }

  while (!to_erase.empty()) {
    stmts.erase(to_erase.back());
    to_erase.pop_back();
  }
}

class MemberStatements : public OptOutDispatch {
 public:
  // Return all members of the stmt if it's a Val. For expressions it returns
  // nothing.
  static std::vector<Statement*> next(Statement* stmt) {
    MemberStatements find_next(stmt);
    return find_next.next_stmts_;
  }

 private:
  MemberStatements() = default;

  MemberStatements(Statement* stmt) {
    handle(stmt);
  }

  using OptOutDispatch::handle;

  void handle(Val* val) final {
    FusionGuard::getCurFusion()->assertInContainer(
        val,
        "IterVisitor.cpp::MemberStatements::handle(Val*) Cannot traverse val, ");
    OptOutDispatch::handle(val);
  }

  void handle(IterDomain* stmt) final {
    next_stmts_.push_back(stmt->start());
    next_stmts_.push_back(stmt->extent());
    next_stmts_.push_back(stmt->stopOffset());
  }

  void handle(TensorDomain* stmt) final {
    next_stmts_.insert(
        next_stmts_.end(), stmt->domain().begin(), stmt->domain().end());
  }

  void handle(TensorView* tv) final {
    next_stmts_.push_back(tv->domain());
  }

  std::vector<Statement*> next_stmts_;
};

} // namespace

std::vector<Statement*> IterVisitor::next(Statement* stmt) {
  if (stmt->isVal()) {
    return next(stmt->as<Val>());
  } else {
    return next(stmt->as<Expr>());
  }
}

std::vector<Statement*> IterVisitor::next(Val* v) {
  FusionGuard::getCurFusion()->assertInContainer(v, "Cannot traverse val, ");
  if (v->definition() != nullptr) {
    return {v->definition()};
  }
  return {};
}

std::vector<Statement*> IterVisitor::next(Expr* expr) {
  FusionGuard::getCurFusion()->assertInContainer(
      expr, "Cannot traverse expr, ");
  std::vector<Statement*> next_stmts{
      expr->inputs().begin(), expr->inputs().end()};
  return next_stmts;
}

// This handle functions is called on every Statement* in topological order,
// starting from outputs to inputs.
void IterVisitor::handle(Statement* s) {
  OptOutDispatch::handle(s);
}

// This handle functions is called on every Expr* in topological order,
// starting from outputs to inputs.
void IterVisitor::handle(Expr* e) {
  OptOutDispatch::handle(e);
}

// This handle functions is called on every Val* in topological order,
// starting from outputs to inputs.
void IterVisitor::handle(Val* v) {
  OptOutDispatch::handle(v);
}

// Implementation details:
// We start with an entry in stmt_stack that is the outputs we want to
// process. We cannot process these outputs untill all Stmts in their history
// have been processed, as those Stmts contain all dependencies to produce
// these values. What we will do is traverse towards inputs until we hit a
// leaf node. Once we hit a leaf node that node will be visited, then we will
// take them off the stack. Once a stack entry is empty, know everything
// needed to be visited to visit stmt_stack.back().back(). We then visit that
// node, make it as visisted and remove it from the stack.
//
// To prevent traversing all paths through a DAG (unless we want to) we have a
// function to remove visited nodes from being re-added to the stack
// (remove_visited).
void IterVisitor::traverseBetween(
    Fusion* fusion,
    const std::unordered_set<Val*>& from,
    const std::vector<Val*>& to,
    bool traverse_all_paths,
    bool traverse_into_members,
    bool traverse_attributes) {
  FusionGuard fg(fusion);

  std::unordered_set<Statement*> visited;

  stmt_stack.clear();
  stmt_stack.emplace_back(to.rbegin(), to.rend());

  bool all_inputs_visited = false;

  while (!stmt_stack.empty()) {
    auto& current_inputs = stmt_stack.back();

    // If current_inputs is empty, pop a level of the stmt_stack, mark the level
    // we pop to as having all inputs processed, the layer we processed were all
    // added inputs required for that Stmt.
    if (current_inputs.empty()) {
      stmt_stack.pop_back();
      all_inputs_visited = true;
      continue;
    }

    // Get the very last entry in the stack to process
    const auto& stmt = current_inputs.back();

    // If we just poped a stmt_stack level, we can finally visit it!
    if (all_inputs_visited) {
      // stmt may have be already visited.
      if (traverse_all_paths || visited.find(stmt) == visited.end()) {
        // Mark visited
        visited.insert(stmt);

        // Actually visit stmt
        handle(stmt);
      }

      // Remove last value just visited
      current_inputs.pop_back();

      // Mark that we need to visit a new Stmt's.
      all_inputs_visited = false;
    } else {
      // We're not ready to process this node, so add all its inputs to be
      // checked Visit input nodes.
      std::vector<Statement*> next_stmts;

      if ((stmt->isVal() && from.find(stmt->asVal()) == from.end()) ||
          stmt->isExpr()) {
        next_stmts = next(stmt);
      }

      // If stmt is an expression, traverse attributes as well
      if (traverse_attributes && stmt->isExpr()) {
        for (Statement* x : stmt->as<Expr>()->attributes()) {
          // There can be nullptr attributes, e.g., philox_index of
          // RngOp. Don't traverse into them
          if (x == nullptr) {
            continue;
          }
          // Note that attributes of type Attribute is not possible to
          // dispatch as it's a template type.
          if (x->getValType() == ValType::Attribute) {
            continue;
          }
          next_stmts.push_back(x);
        }
      }

      if (traverse_into_members) {
        auto members = MemberStatements::next(stmt);
        next_stmts.insert(next_stmts.end(), members.begin(), members.end());
      }

      // We may want to retraverse nodes, in that case revisit everything!
      if (!traverse_all_paths) {
        // If we don't want to retraverse, remove nodes we already visisted.
        remove_visited(next_stmts, visited);
      }
      if (next_stmts.empty()) {
        // If there's nothing to visit because it was all already visited, mark
        // to process
        all_inputs_visited = true;
      } else {
        // Add all these new stmts to visit to the stack.
        stmt_stack.emplace_back(next_stmts.rbegin(), next_stmts.rend());
        // We have new things to visit,
        all_inputs_visited = false;
      }
    }
  }
}

void IterVisitor::traverseTo(
    Fusion* fusion,
    const std::vector<Val*>& to,
    bool traverse_all_paths,
    bool traverse_into_members,
    bool traverse_attributes) {
  traverseBetween(
      fusion,
      {},
      to,
      traverse_all_paths,
      traverse_into_members,
      traverse_attributes);
}

void IterVisitor::traverseHelper(Fusion* fusion, bool traverse_all_paths) {
  FusionGuard fg(fusion);

  auto term_val_outs = fusion->getTerminatingOutputs();
  if (!term_val_outs.empty()) {
    traverseTo(fusion, term_val_outs, traverse_all_paths);
  }
}

void IterVisitor::traverse(Fusion* fusion) {
  traverseHelper(fusion, false);
}

void IterVisitor::traverseAllPaths(Fusion* fusion) {
  traverseHelper(fusion, true);
}

namespace {

// TODO: Also have InputsOf should pick one and remove the other.
class Inputs : public IterVisitor {
 private:
  //! Optional list of input vals. While traversing to inputs if a value in the
  //! all_inputs list is found, that value will be added to the inputs_ and
  //! traversal will not go into its definition. Otherwise traversal follows
  //! definition paths until hitting a definition that is a nullptr (i.e. a
  //! terminating input).
  const std::vector<Val*>& all_inputs_;
  std::vector<Val*> inputs_;

  Inputs(const std::vector<Val*>& all_inputs) : all_inputs_(all_inputs) {}

  std::vector<Statement*> next(Val* v) override {
    if (std::find(inputs_.begin(), inputs_.end(), v) != inputs_.end()) {
      return {};
    }
    return IterVisitor::next(v);
  }

  void handle(Val* val) override {
    // If there's no definition to val, or val is created inside the fusion, or
    // val is within the provided inputs
    if (val->definition() == nullptr || val->definition()->inputs().empty() ||
        std::find(all_inputs_.begin(), all_inputs_.end(), val) !=
            all_inputs_.end()) {
      // if not already placed in the inputs
      if (std::find(inputs_.begin(), inputs_.end(), val) == inputs_.end()) {
        inputs_.push_back(val);
      }
    }
  }

 public:
  static std::vector<Val*> getInputs(
      const std::vector<Val*>& of,
      const std::vector<Val*>& all_inputs) {
    if (of.empty()) {
      return {};
    }
    Inputs inps(all_inputs);
    inps.traverseTo(of[0]->fusion(), of);
    return inps.inputs_;
  }
};

} // namespace

std::vector<Val*> IterVisitor::getInputsTo(
    const std::vector<Val*>& vals,
    const std::vector<Val*>& inputs) {
  return Inputs::getInputs(vals, inputs);
}

namespace {

class AllVals : public IterVisitor {
 private:
  std::unordered_set<Val*> vals;

  void handle(Val* val) final {
    vals.emplace(val);
  }

 public:
  // Return all values in history of all values in from
  static std::unordered_set<Val*> get(
      Fusion* fusion,
      const std::vector<Val*>& from) {
    AllVals av;
    av.traverseTo(fusion, from, false);
    return av.vals;
  }
};

} // namespace

/* BACKWARDS VISITOR */

std::vector<Statement*> BackwardVisitor::next(Statement* stmt) {
  if (stmt->isVal()) {
    return next(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    return next(stmt->as<Expr>());
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "BackwardVisitor could not detect type in next_dispatch.");
  }
}

std::vector<Statement*> BackwardVisitor::next(Expr* expr) {
  return std::vector<Statement*>(
      expr->outputs().begin(), expr->outputs().end());
}

std::vector<Statement*> BackwardVisitor::next(Val* val) {
  // Going to sort based on relative topological position
  std::map<size_t, Statement*> exprs;

  for (auto expr : FusionGuard::getCurFusion()->unordered_uses(val)) {
    // Make sure it's an expr we can traverse
    if (traversal_exprs_.find(expr) != traversal_exprs_.end()) {
      exprs[traversal_exprs_[expr]] = expr;
    }
  }

  std::vector<Statement*> next_stmts(exprs.size());
  std::transform(
      exprs.begin(),
      exprs.end(),
      next_stmts.begin(),
      [](std::pair<size_t, Statement*> pair) { return pair.second; });

  return next_stmts;
}

void BackwardVisitor::handle(Statement* stmt) {
  OptOutDispatch::handle(stmt);
}

void BackwardVisitor::handle(Expr* expr) {
  OptOutDispatch::handle(expr);
}

void BackwardVisitor::handle(Val* val) {
  OptOutDispatch::handle(val);
}

void BackwardVisitor::traverseTo(
    Fusion* fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);

  // Reset members
  stmt_stack_.clear();
  traversal_exprs_.clear();

  if (from.empty()) {
    return;
  }

  auto vals = AllVals::get(fusion, from);
  auto exprs = StmtSort::getExprs(fusion, from);

  {
    size_t pos = 0;
    for (auto expr : exprs)
      traversal_exprs_[expr] = pos++;
  }

  // All stmts we've called handle on
  std::unordered_set<Statement*> visited_stmts_;

  if (must_cover_all_expr_outputs_) {
    for (auto traversal_pair : traversal_exprs_) {
      for (auto out : traversal_pair.first->outputs()) {
        TORCH_INTERNAL_ASSERT(
            vals.find(out) != vals.end(),
            "Invalid backward traversal found. Some output paths were not provided:",
            out);
      }
    }
  }

  auto inputs = InputsOf::getInputsTo(from);
  stmt_stack_.emplace_back(inputs.begin(), inputs.end());

  // The rest is basically copy-pasted from IterVitor:
  while (!stmt_stack_.empty()) {
    auto next_stmts = next(stmt_stack_.back().back());

    // Remove statements we already visited if we're not traversing all paths
    if (!traverseAllPaths) {
      remove_visited(next_stmts, visited_stmts_);
    }

    // Traverse down until we get to a leaf
    while (!next_stmts.empty()) {
      stmt_stack_.emplace_back(next_stmts.rbegin(), next_stmts.rend());
      next_stmts = next(stmt_stack_.back().back());
      // Remove statements we already visited if we're not traversing all paths
      if (!traverseAllPaths) {
        remove_visited(next_stmts, visited_stmts_);
      }
    }

    // Traverse back up
    // Mark visited
    visited_stmts_.emplace(stmt_stack_.back().back());
    // Handle
    handle(stmt_stack_.back().back());
    // Remove
    stmt_stack_.back().pop_back();

    while (!stmt_stack_.empty() && stmt_stack_.back().empty()) {
      stmt_stack_.pop_back();
      if (!stmt_stack_.empty()) {
        // Mark visited
        visited_stmts_.emplace(stmt_stack_.back().back());
        // Handle
        handle(stmt_stack_.back().back());
        // Remove
        stmt_stack_.back().pop_back();
      }
    }
  }
}

/* DEPENDENCY CHECKING */

namespace {

// Looks for and returns all values in between dependencies and vals, including
// them.
struct Dependencies : public IterVisitor {
 private:
  //! A given set of dependency Vals
  const std::unordered_set<Val*> dependencies_;
  //! Vals that are found between dependencies_ and of. Topologically
  //! ordered.
  std::vector<Val*> vals_;
  //! Exprs that are found between dependencies_ and of. Topologically
  //! ordered.
  std::vector<Expr*> exprs_;
  //! A set version of vals_
  std::unordered_set<Val*> dependent_vals_;
  //! A set version of exprs_
  std::unordered_set<Expr*> dependent_exprs_;

 private:
  std::vector<Statement*> next(Val* v) override {
    if (dependencies_.find(v) != dependencies_.end()) {
      return std::vector<Statement*>();
    }
    return IterVisitor::next(v);
  }

  void handle(Val* val) override {
    // val is included if:
    // 1. it is one of the dependencies, or
    // 2. its defining expression is included in the dependent expr set
    if (dependencies_.find(val) != dependencies_.end()) {
      TORCH_INTERNAL_ASSERT(
          dependent_vals_.find(val) == dependent_vals_.end(),
          "Trying to add already added val: ",
          val);
      vals_.push_back(val);
      dependent_vals_.insert(val);
    } else {
      auto def = val->definition();
      if (def != nullptr &&
          dependent_exprs_.find(def) != dependent_exprs_.end()) {
        TORCH_INTERNAL_ASSERT(
            dependent_vals_.find(val) == dependent_vals_.end(),
            "Trying to add already added val: ",
            val);
        vals_.push_back(val);
        dependent_vals_.insert(val);
      }
    }
  }

  void handle(Expr* expr) override {
    // Track which expr is depedent on the dependencies_ exprs.
    if (std::any_of(
            expr->inputs().begin(), expr->inputs().end(), [&](Val* input_val) {
              return dependent_vals_.find(input_val) != dependent_vals_.end();
            })) {
      if (!dependent_exprs_.count(expr)) {
        exprs_.push_back(expr);
        dependent_exprs_.insert(expr);
      }
    }
  }

  Dependencies(
      std::unordered_set<Val*> _dependencies,
      const std::vector<Val*>& of)
      : dependencies_(std::move(_dependencies)) {
    traverseTo(of[0]->fusion(), of, false);
  };

 public:
  static std::vector<Val*> getAllVals(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of) {
    if (of.empty()) {
      return {};
    }

    Dependencies deps(dependencies, of);
    return deps.vals_;
  }

  static std::vector<Expr*> getAllExprs(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of) {
    if (of.empty()) {
      return {};
    }

    Dependencies deps(dependencies, of);
    return deps.exprs_;
  }
};

// Looks for and returns all output values with dependencies on `of`.
struct FindOutputs : public IterVisitor {
  const std::unordered_set<Val*>& of_;
  std::unordered_set<Val*> outs_;

  void handle(Val* val) override {
    if (of_.find(val) != of_.end()) {
      Statement* out_stmt = stmt_stack.front().back();
      TORCH_INTERNAL_ASSERT(out_stmt->isVal());
      auto out_val = out_stmt->as<Val>();
      if (of_.find(out_val) == of_.end()) {
        outs_.emplace(out_val);
      }
    }
  }

  // TODO: Simply traverse through uses from of. Would be a lot faster than
  // tracing all paths like this.
  FindOutputs(const std::unordered_set<Val*>& _of) : of_(_of) {
    auto fusion = (*of_.begin())->fusion();
    traverseTo(fusion, fusion->outputs(), true);
  };

  static std::unordered_set<Val*> getAllOutputsOf(
      const std::unordered_set<Val*>& of) {
    if (of.empty()) {
      return std::unordered_set<Val*>();
    }

    FindOutputs finder(of);
    return finder.outs_;
  }
};

// Looks for and returns all values that depends on `of`.
class DependentVals : public IterVisitor {
 private:
  // Which nodes to find dependencies of
  const std::unordered_set<Val*>& of_;

  // Dependencies we have so far
  std::unordered_set<Val*> outs_;

  // Boundary where we want to stop searching beyond
  // TODO: Based on the todo below, shouldn't we stop just at the definition of?
  // If we really wanted to make this traverse left, wouldn't we first check
  // which outputs are outputs dependent on of?
  std::unordered_set<Val*> boundary_;

  std::vector<Statement*> next(Val* v) override {
    if (boundary_.find(v) != boundary_.end())
      return std::vector<Statement*>();
    return IterVisitor::next(v);
  }

  void handle(Val* val) override {
    if (val->isFusionInput() || val->definition() == nullptr ||
        of_.count(val) || outs_.count(val)) {
      return;
    }

    for (auto v : val->definition()->inputs()) {
      if (of_.count(v) || outs_.count(v)) {
        outs_.emplace(val);
        return;
      }
    }
  }

  // optimization to limit search path
  // TODO: Is this valid? Couldn't something like:
  // out0 = of + val0
  // out1 = out0 + val1
  // out2 = TernaryOp(out1, val0, of)
  // Hide the dep of out1 on of?
  void createBoundary() {
    for (auto v_of : of_) {
      for (auto v_expr : v_of->uses()) {
        for (auto v_in : v_expr->inputs()) {
          boundary_.emplace(v_in);
        }
      }
    }
  }

  DependentVals(const std::unordered_set<Val*>& _of) : of_(_of) {
    createBoundary();
    auto fusion = (*of_.begin())->fusion();
    traverseTo(fusion, fusion->outputs(), false);
  };

 public:
  static std::unordered_set<Val*> getAllDependentVals(
      const std::unordered_set<Val*>& of) {
    if (of.empty()) {
      return std::unordered_set<Val*>();
    }
    DependentVals dependencies(of);
    return dependencies.outs_;
  }
};

class DependencyChains : public IterVisitor {
 public:
  std::deque<std::deque<Val*>> dep_chains;
  bool is_dependency = false;
  std::unordered_set<Val*> dependencies_;

  void handle(Val* val) override {
    if (dependencies_.find(val) != dependencies_.end()) {
      is_dependency = true;
      std::deque<Val*> deps;
      for (auto stack : stmt_stack) {
        if (stack.back()->isVal()) {
          deps.push_back(stack.back()->as<Val>());
        }
      }
      // Order as dependency -> of
      dep_chains.emplace_back(deps.rbegin(), deps.rend());
    }
  }

  DependencyChains(Val* _dependency, Val* _of, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    traverseTo(_of->fusion(), {_of}, all_chains_);
  }

  DependencyChains(Val* _dependency, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    if (all_chains_) {
      traverseAllPaths(_dependency->fusion());
    } else {
      traverse(_dependency->fusion());
    }
  }

  DependencyChains(
      std::unordered_set<Val*> _dependencies,
      bool all_chains_ = false)
      : dependencies_(std::move(_dependencies)) {
    if (dependencies_.empty()) {
      return;
    }

    if (all_chains_) {
      traverseAllPaths((*dependencies_.begin())->fusion());
    } else {
      traverse((*dependencies_.begin())->fusion());
    }
  }

  static std::deque<Val*> getDependencyChain(Val* dependency, Val* of) {
    DependencyChains dp(dependency, of, false);
    if (dp.dep_chains.empty()) {
      return std::deque<Val*>();
    }
    return dp.dep_chains[0];
  }

  // I don't think this is actually hooked up, but leaving for now.
  static std::deque<std::deque<Val*>> getDependencyChains(
      Val* dependency,
      Val* of) {
    DependencyChains dp(dependency, of, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(Val* dependency) {
    DependencyChains dp(dependency, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(
      const std::unordered_set<Val*>& dependencies) {
    DependencyChains dp(dependencies, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
    return dp.dep_chains;
  }
};

} // namespace

bool DependencyCheck::isDependencyOf(Val* dependency, Val* of) {
  return !DependencyChains::getDependencyChain(dependency, of).empty();
}

std::deque<Val*> DependencyCheck::getSingleDependencyChain(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChain(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllDependencyChains(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChains(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllUseChains(Val* producer) {
  return DependencyChains::getAllUseChains(producer);
}

std::vector<Val*> DependencyCheck::getAllValsBetween(
    const std::unordered_set<Val*>& dependencies,
    const std::vector<Val*>& of) {
  return Dependencies::getAllVals(dependencies, of);
}

std::vector<Expr*> DependencyCheck::getAllExprsBetween(
    const std::unordered_set<Val*>& dependencies,
    const std::vector<Val*>& of) {
  return Dependencies::getAllExprs(dependencies, of);
}

std::unordered_set<Val*> DependencyCheck::getAllOutputsOf(
    const std::unordered_set<Val*>& of) {
  if (of.empty()) {
    return std::unordered_set<Val*>();
  }
  FusionGuard fg((*of.begin())->fusion());
  return FindOutputs::getAllOutputsOf(of);
}

std::unordered_set<Val*> DependencyCheck::getAllDependentVals(
    const std::unordered_set<Val*>& of) {
  if (of.empty()) {
    return std::unordered_set<Val*>();
  }
  FusionGuard fg((*of.begin())->fusion());
  return DependentVals::getAllDependentVals(of);
}

void StmtSort::handle(Statement* stmt) {
  stmts.push_back(stmt);
}

std::vector<Expr*> StmtSort::getExprs(
    Fusion* fusion,
    bool traverse_members,
    bool traverse_attributes) {
  auto terminating_outputs = fusion->getTerminatingOutputs();
  return StmtSort::getExprs(
      fusion, terminating_outputs, traverse_members, traverse_attributes);
}

std::vector<Expr*> StmtSort::getExprs(
    Fusion* fusion,
    const std::vector<Val*>& to,
    bool traverse_members,
    bool traverse_attributes) {
  auto stmts =
      StmtSort::getStmts(fusion, to, traverse_members, traverse_attributes);
  auto filter = ir_utils::filterByType<Expr>(stmts.begin(), stmts.end());
  std::vector<Expr*> exprs(filter.begin(), filter.end());
  return exprs;
}

std::vector<Expr*> StmtSort::getExprsBetween(
    Fusion* fusion,
    const std::vector<Val*>& from,
    const std::vector<Val*>& to,
    bool traverse_members,
    bool traverse_attributes) {
  auto stmts = StmtSort::getStmtsBetween(
      fusion, from, to, traverse_members, traverse_attributes);
  auto filter = ir_utils::filterByType<Expr>(stmts.begin(), stmts.end());
  std::vector<Expr*> exprs(filter.begin(), filter.end());
  return exprs;
}

std::vector<Statement*> StmtSort::getStmts(
    Fusion* fusion,
    bool traverse_members,
    bool traverse_attributes) {
  auto terminating_outputs = fusion->getTerminatingOutputs();
  return StmtSort::getStmts(
      fusion, terminating_outputs, traverse_members, traverse_attributes);
}

std::vector<Statement*> StmtSort::getStmts(
    Fusion* fusion,
    const std::vector<Val*>& to,
    bool traverse_members,
    bool traverse_attributes) {
  StmtSort es;
  es.traverseTo(fusion, to, false, traverse_members, traverse_attributes);
  return es.stmts;
}

std::vector<Statement*> StmtSort::getStmtsBetween(
    Fusion* fusion,
    const std::vector<Val*>& from,
    const std::vector<Val*>& to,
    bool traverse_members,
    bool traverse_attributes) {
  StmtSort es;
  es.traverseBetween(
      fusion,
      {from.begin(), from.end()},
      to,
      false,
      traverse_members,
      traverse_attributes);
  return es.stmts;
}

void InputsOf::handle(Val* v) {
  if (v->definition() == nullptr || v->definition()->inputs().empty()) {
    if (grabbed_inputs.emplace(v).second) {
      ordered_inputs.push_back(v);
    }
  }
}

std::vector<Val*> InputsOf::output(Fusion* fusion, Val* output_) {
  return outputs(fusion, {output_});
}

std::vector<Val*> InputsOf::outputs(
    Fusion* fusion,
    const std::vector<Val*>& outputs_) {
  InputsOf io;
  io.traverseTo(fusion, outputs_, false);
  return io.ordered_inputs;
}

} // namespace nvfuser
