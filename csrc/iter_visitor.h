// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <ir/base_nodes.h>
#include <type.h>

#include <deque>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class Fusion;

/*
 * IterVisitor starts from leaf nodes, fusion outputs, or the provided values.
 * It walks the DAG bacwkards from the starting nodes, to roots. Each node in
 * the dag will be called with handle(Statement*) in topolgical order inputs of
 * the fusion to outputs of the fusion.
 *
 * TODO: We may want a BFS version of this code to extract ILP, not implemented
 * yet.
 *
 * TODO: We may want to have ordering of outputs to inputs. I'm not sure why we
 * would want this, but seems like it would be a reasonable request.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API IterVisitor : public OptOutDispatch {
 public:
  ~IterVisitor() override = default;

  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

 protected:
  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate up through the DAG
  // to inputs based on depth first traversal. Next could be called on a node
  // multiple times.
  virtual std::vector<Statement*> next(Statement* stmt);

  virtual std::vector<Statement*> next(Val* v);

  virtual std::vector<Statement*> next(Expr* expr);

  using OptOutDispatch::handle;

  // This dispatch functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  void dispatch(Statement* s) override;

  // This dispatch functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  void dispatch(Expr* e) override;

  // This dispatch functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  void dispatch(Val* v) override;

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the outputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<Statement*>> stmt_stack;

  void traverseHelper(Fusion* fusion, bool traverse_all_paths = false);

 public:
  //! Traverses nodes in Fusion from inputs in topological order to "to". i.e.
  //! from inputs towards outputs.
  //! \param traverseAllPaths = false only call handle on each Statement* once
  //!    traverseAllPaths = true traverses all paths between expressions/values.
  //!    Calls handle on a Statement* for every path from inputs to "to".
  //! \param traverseIntoMembers = When hitting nodes like TensorView,
  //! TensorDomain, or IterDomain where there are members of the nodes that are
  //! Val's a value of "true" will also traverse into those member Val's, a
  //! value of "false" will not traverse into the members.
  //! \param traverse_attributes When true, traverse into expr
  //! attributes. Note that attributes of template type Attribute are
  //! not traversed as there's no dispatch support.
  //! \param traverse_siblings When true, traverse all outputs of
  //! active multi-output expressions, even if those Expr outputs are not used
  //! in paths to Fusion outputs.
  void traverseTo(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_all_paths = false,
      bool traverse_into_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  //! Traverses nodes in Fusion from inputs in topological order to "to". i.e.
  //! from inputs towards outputs.
  //! \param traverseAllPaths = false only call handle on each Statement* once
  //!    traverseAllPaths = true traverses all paths between expressions/values.
  //!    Calls handle on a Statement* for every path from inputs to "to".
  //! \param traverseIntoMembers = When hitting nodes like TensorView,
  //! TensorDomain, or IterDomain where there are members of the nodes that are
  //! Val's a value of "true" will also traverse into those member Val's, a
  //! value of "false" will not traverse into the members.
  //! \param from: Specified values to start traversing. If a "from" Val is not
  //! on path from inputs to "to" node it will not be visited. If there's a path
  //! from inputs to "to" that doesn't go through "from" that input and the path
  //! from it will also be traversed.
  //! \param traverse_attributes When true, traverse into expr
  //! attributes. Note that attributes of template type Attribute are
  //! not traversed as there's no dispatch support.
  //! \param traverse_siblings When true, traverse all outputs of
  //! active multi-output expressions, even if those Expr outputs are not used
  //! in paths to Fusion outputs.
  void traverseBetween(
      Fusion* fusion,
      const std::unordered_set<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_all_paths = false,
      bool traverse_into_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Iterates from terminating outputs registered with the fusion. Terminating
  // means value is not used to generate any other value used in producing
  // registered outputs.
  void traverse(Fusion* fusion);

  // Same as traverse but it traverses every edge, meaning it will traverse
  // values more than once.
  void traverseAllPaths(Fusion* fusion);

  //! Get inputs to vals. Possible input vals can be optionally
  //! given. If not, vals with no producers are returned.
  //
  // TODO: This doesn't seem to fit with IterVisitor. Should probably be moved
  // out of the class.
  static std::vector<Val*> getInputsTo(
      const std::vector<Val*>& vals,
      const std::vector<Val*>& inputs = {});
};

/*
 * Backward visitor calls handle in reverse order from outputs to inputs.
 * It would be really nice to unify this with IterVisitor, however,
 * the challenge there is that we specify traversal from outputs towards inputs
 * because it implicitly provides DCE. However, if users are not careful, they
 * could miss necessary outputs to do a backward traversal.
 *
 * BackwardVisitor checks that all outputs of an Expr is visited before visiting
 * the Expr. If we don't provide nodes to start from on all backward paths of
 * those outputs we will never visit the Expr.
 *
 * The first step of BackwardVisitor is to make sure we've specified enough
 * outputs to guarentee that we will traverse all outputs of all exprs during
 * the backward traversal. In the case where we don't require visiting all
 * outputs of some exprs, example being the `N` output of welford ops.
 * `must_cover_all_expr_outputs` is added to disable the check, and in
 * this case the visitor pass need be aware
 *  1. Exprs in the `from` list with any output that has a use chain that
 * ends with a final consumer `will be` visited.
 *  2. Vals in the `from` list that doesn't have a use chain that ends with
 * a final consumer `will not be` visited, even though its
 * definition expr might be visited. An example is if the `N` output
 * of an welford op is unused, but other outputs are, the welford op
 * will be visited but the `N` output will not.
 *
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API BackwardVisitor : public OptOutDispatch {
 public:
  // clang-tidy: cppcoreguidelines-virtual-class-destructor
  ~BackwardVisitor() override = default;

 protected:
  BackwardVisitor(bool must_cover_all_expr_outputs = true)
      : must_cover_all_expr_outputs_(must_cover_all_expr_outputs) {}

  BackwardVisitor(const BackwardVisitor& other) = default;
  BackwardVisitor& operator=(const BackwardVisitor& other) = default;

  BackwardVisitor(BackwardVisitor&& other) = default;
  BackwardVisitor& operator=(BackwardVisitor&& other) = default;

  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate up through the DAG
  // to inputs based on depth first traversal. Next could be called on a node
  // multiple times.
  virtual std::vector<Statement*> next(Statement* stmt);

  virtual std::vector<Statement*> next(Expr* expr);

  virtual std::vector<Statement*> next(Val* val);

  using OptOutDispatch::handle;

  // This handle functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void dispatch(Statement* stmt) override;

  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void dispatch(Expr* expr) override;

  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void dispatch(Val* val) override;

  // All exprs that need to be visited in this traversal. Labeled in topological
  // order (size_t).
  std::unordered_map<Expr*, size_t> traversal_exprs_;

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the inputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  std::deque<std::deque<Statement*>> stmt_stack_;

  // Starts at nodes provided in from, traverses from these nodes to inputs.
  // Calls handle on all Statement*s in topological sorted order.
  // traverseAllPaths = false only call handle on each Statement* once
  // traverseAllPaths = true traverses all paths from nodes in from to inputs.
  //   Handle on a Statement* for every path from "from" nodes, to inputs.
  void traverseTo(
      Fusion* fusion,
      const std::vector<Val*>& from,
      bool traverseAllPaths = false);

  bool must_cover_all_expr_outputs_ = true;
};

class TORCH_CUDA_CU_API DependencyCheck {
 public:
  // Returns if "dependency" is a dependency of "of".
  static bool isDependencyOf(Val* dependency, Val* of);

  // Finds a Val* path from "of" to "dependency". Returns that path.
  // deque.back() is "of", deque[0] is dependency if a chain exists.
  static std::deque<Val*> getSingleDependencyChain(Val* dependency, Val* of);

  // Finds all Val* paths from "of" to "dependency". Returns those paths.
  // deque[i].back() is "of", and deque[i][0] is "dependency". Returns an
  // empty deque if no dependency found.
  static std::deque<std::deque<Val*>> getAllDependencyChains(
      Val* dependency,
      Val* of);

  // Finds all Val* paths from all leaf nodes to "dependency". Returns those
  // paths. deque[i].back() are leaf nodes, and deque[i][0] is "dependency".
  // Returns an empty deque if there are no uses of dependency found.
  static std::deque<std::deque<Val*>> getAllUseChains(Val* dependency);

  // Grab all values that exist between and including provided
  // vals. Returned values are topologicaly ordered, and unique.
  static std::vector<Val*> getAllValsBetween(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of);

  // Returns all dependent exprs that exist between
  //  the provided vals
  static std::vector<Expr*> getAllExprsBetween(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of);

  // Return registered outputs of the fusion that are a dependency of any val of
  static std::unordered_set<Val*> getAllOutputsOf(
      const std::unordered_set<Val*>& of);

  // Return all Vals that depend on the given Vals
  static std::unordered_set<Val*> getAllDependentVals(
      const std::unordered_set<Val*>& of);
};

// Expr sort will take a fusion and return a topologically sorted list of
// expressions.
class StmtSort : public IterVisitor {
 protected:
  StmtSort() = default;

  std::vector<Statement*> stmts;

  using IterVisitor::handle;

  void dispatch(Statement* stmt) override;

 public:
  // If traverse_members it will also extract all member nodes in the sorted
  // statement list in the fusion. i.e. all IterDomains, extents, and associated
  // expressions of them. Similarly, if traverse_attributes it will
  // grab all nodes associated as Expr attributes.
  static std::vector<Statement*> getStmts(
      Fusion* fusion,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Returns ordered Statements required to produce 'to', including 'to'.
  static std::vector<Statement*> getStmtsTo(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Returns ordered Statements required to produce from, including from.
  // Stops traversal once hiting any Statements in to. Includes Statements in
  // to.
  //
  // Warning: this doesn't necessarily prevent statements before `to` from being
  // returned. e.g.
  // i1 = i0
  // i2 = i1
  // i3 = i2
  // i4 = i3 + i1
  // getExprs(fusion, {i4}, {i3})
  // will return the definition and values {i0, i1, i4}
  // i3 is dependent on i1, but since i4 also is then the traversal will go down
  // the i4->i1->i0 path, even though the i4->i3-//>i2->i1 path is blocked.
  //
  // If traverse_members it will also extract all member nodes in the sorted
  // expr list in the fusion. i.e. all expressions on IterDomains, extents, etc
  static std::vector<Statement*> getStmtsBetween(
      Fusion* fusion,
      const std::vector<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprs(
      Fusion* fusion,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprsTo(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprsBetween(
      Fusion* fusion,
      const std::vector<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_members = false,
      bool traverse_attributes = false,
      bool traverse_siblings = false);
};

class TORCH_CUDA_CU_API InputsOf : public IterVisitor {
 private:
  std::unordered_set<Val*> grabbed_inputs;
  std::vector<Val*> ordered_inputs;

  using IterVisitor::handle;

  void dispatch(Val* v) final;

 public:
  static std::vector<Val*> output(Fusion* fusion, Val* output_);
  static std::vector<Val*> outputs(
      Fusion* fusion,
      const std::vector<Val*>& outputs_);
};

//! This is a generic traversal class that is used to modify a Fusion graph by
//! replacing Vals to simplify computation or remove dead code. This differs
//! from OptOutMutator, which is built for mutating TensorViews in-place in a
//! graph by altering the associated IterDomains, and which does not easily
//! handle modifying TensorView definitions and Fusion outputs during traversal.
//!
//! Derived classes should override handle() for relevant Exprs and they should
//! make use of registerReplacement() to change the definitions of Vals in the
//! graph. Note that if replacements are made using registerReplacement(old_val,
//! new_val), then neither new_val nor any new Statements produced in creating
//! it will be traversed by this class. Also note that any Vals or Exprs that
//! are previously marked dead will not be processed by handle().
class DeadCodeRemover : BackwardVisitor {
 public:
  DeadCodeRemover(Fusion* fusion) : BackwardVisitor(false), fusion_(fusion) {}

  DeadCodeRemover(const DeadCodeRemover& other) = default;
  DeadCodeRemover& operator=(const DeadCodeRemover& other) = default;

  DeadCodeRemover(DeadCodeRemover&& other) = default;
  DeadCodeRemover& operator=(DeadCodeRemover&& other) = default;

  //! Instead of traverseTo, run() is the entry point for this class, and we
  //! always traverse from outputs backward to their inputs.
  //!
  //! Returns a bool indicating whether the Fusion was modified or not.
  bool run();

  inline Fusion* fusion() const {
    return fusion_;
  }

 protected:
  using BackwardVisitor::handle;

  void dispatch(Statement* stmt) override;
  void dispatch(Expr* expr) override;

  //! We implement this in order to remove dangling TensorViews whose uses are
  //! all dead. Note that we do not remove other ValTypes like Scalars since
  //! they might still be used as attributes or members of other objects, which
  //! is not reflected by Val::uses().
  void handle(TensorView* tv) override;

  //! Registers a Val for replacement in outputs and in all its uses.
  //!
  //! Note that replacement does not occur immediately, but will be done after
  //! the traversal is completed. This is so that any Val* and Expr* pointers
  //! may be safely dereferenced during traversal.
  //!
  //! The argument old_val is always marked Dead by this method. If old_val is a
  //! Fusion input, we do not replace it. If old_val's definition is non-null
  //! and has other outputs which are not dead, we do not remove old_val.
  //!
  //! Returns whether old_val was registered for removal from the Fusion.
  bool registerReplacement(Val* old_val, Val* new_val);

  //! Find whether a statement is not marked as live code.
  inline bool isDead(Statement* stmt) const {
    return live_statements_.find(stmt) == live_statements_.end();
  }

  //! Find whether a statement is marked as live code.
  inline bool isLive(Statement* stmt) const {
    return !isDead(stmt);
  }

  //! Check whether all outputs of an expression have been marked dead
  inline bool allOutputsDead(Expr* expr) const {
    return std::all_of(
        expr->outputs().begin(), expr->outputs().end(), [&](Val* outp) {
          return isDead(outp);
        });
  }

  //! Check whether all uses have been marked dead
  inline bool allUsesDead(Val* val) const {
    return std::all_of(val->uses().begin(), val->uses().end(), [&](Expr* use) {
      return isDead(use);
    });
  }

 private:
  //! Removes an Expr* from the Fusion, if possible.
  //!
  //! The Expr will _only_ be marked dead and removed if all of its outputs are
  //! already marked dead. In this case all the outputs will also be removed
  //! from the Fusion.
  //!
  //! Returns whether the Expr was marked dead and removed from the Fusion.
  bool maybeRemoveExpr(Expr* expr);

  //! Mark a single Statement as being alive.
  inline void markLive(Statement* stmt) {
    live_statements_.insert(stmt);
  }

  //! Ensure that a Statement and its upstream Statements are alive. If it is an
  //! Expr, ensure all its inputs are alive. If it's a Val with a definition,
  //! recursive to the definition. Newly-created Statements default to being
  //! dead, so this method is called when adding a Statement to the active path
  //! of the Fusion inside registerReplacement.
  void markLiveRecursive(Statement* stmt);

  //! Mark a single Statement as being dead. This does not remove stmt from the
  //! Fusion. It is an error to call this on a Fusion output.
  //!
  //! Returns true if the statement was previously live, and false otherwise.
  bool markDead(Statement* stmt);

  //! Register a Val for later removal.
  void registerRemoval(Val* val);

  //! Register an Expr for later removal.
  //!
  //! Note that if any of its outputs are removed, expr will be removed even if
  //! it is not marked for removal, and all its outputs will have their
  //! definitions set to nullptr.
  inline void registerRemoval(Expr* expr) {
    exprs_to_remove_.push_back(expr);
  }

  //! All modifications to the Fusion are registered during traversal then
  //! later they are committed by this method. For safety, this should only be
  //! run after traversing the graph.
  //!
  //! Returns a bool indicating whether any modifications were performed.
  bool modifyFusion() const;

 private:
  //! The Fusion associated with live_statements_
  Fusion* fusion_;

  //! Statements are marked dead by removing them from this set
  std::unordered_set<Statement*> live_statements_;

  //! Vals to be replaced in outputs and with replaceValInExpr in all uses.
  std::vector<std::pair<Val*, Val*>> vals_to_replace_;

  //! Statements that will be removed. We remove Vals before Exprs, so we track
  //! them separately here.
  std::vector<Val*> vals_to_remove_;
  std::vector<Expr*> exprs_to_remove_;
};

} // namespace nvfuser
