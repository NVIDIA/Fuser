// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <dispatch.h>
#include <id_model/id_model.h>
#include <id_model/schedule.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <val_graph_visitor.h>

#include <unordered_set>

namespace nvfuser {
namespace scheduler_tools {

namespace {

// Similar to id_model/transform_replay.h but using a given list of
// outputs in addition to inputs
class LoopDomainSchedulerReplayTransform : OptInConstDispatch {
 public:
  static Expr* replayAs(
      const std::vector<IterDomain*>& ordered_inputs,
      const std::vector<IterDomain*>& ordered_outputs,
      const Expr* expression_to_match) {
    LoopDomainSchedulerReplayTransform replay(
        ordered_inputs, ordered_outputs, expression_to_match);
    return replay.replayed_expr_;
  }

 private:
  LoopDomainSchedulerReplayTransform(
      const std::vector<IterDomain*>& ordered_inputs,
      const std::vector<IterDomain*>& ordered_outputs,
      const Expr* expression_to_match)
      : input_ids_(ordered_inputs), output_ids_(ordered_outputs) {
    for (const auto& out : ordered_outputs) {
      NVF_ERROR(
          out->definition() == nullptr,
          "Should not rewrite definition of ",
          out->toString(),
          ". Existing definition: ",
          out->definition()->toString(),
          "New definition: ",
          expression_to_match->toString());
    }
    OptOutConstDispatch::dispatch(expression_to_match);
  }

  using OptInConstDispatch::handle;

  void handle(const Split* split) final {
    NVF_ERROR(input_ids_.size() == 1);
    NVF_ERROR(output_ids_.size() == 2);
    replayed_expr_ = IrBuilder::createInContainer<Split>(
        split->fusion(),
        output_ids_[0],
        output_ids_[1],
        input_ids_[0],
        split->factor(),
        split->innerSplit());
  }

  void handle(const Merge* merge) final {
    NVF_ERROR(input_ids_.size() == 2);
    NVF_ERROR(output_ids_.size() == 1);
    replayed_expr_ = IrBuilder::createInContainer<Merge>(
        merge->fusion(), output_ids_[0], input_ids_[0], input_ids_[1]);
  }

  void handle(const Resize* resize) final {
    NVF_ERROR(input_ids_.size() == 1);
    NVF_ERROR(output_ids_.size() == 1);
    replayed_expr_ = IrBuilder::createInContainer<Resize>(
        resize->fusion(),
        output_ids_[0],
        input_ids_[0],
        resize->leftExpand(),
        resize->rightExpand());
  }

  void handle(const Swizzle2D* swizzle_2d) final {
    NVF_THROW("Unsupported");
  }

  void handle(const Swizzle* swizzle) final {
    NVF_THROW("Unsupported");
  }

 private:
  Expr* replayed_expr_ = nullptr;
  const std::vector<IterDomain*>& input_ids_;
  const std::vector<IterDomain*>& output_ids_;
};

// Replay a given IterDomain transform expression on the loop domain
// of a given tensor using specified loop IDs as its inputs.
class ReplayForwardTransformOnLoopDomain : OptInConstDispatch {
 public:
  static void replayAs(
      TensorView* tv,
      const std::vector<IterDomain*>& input_loop_ids,
      const Expr* transform) {
    ReplayForwardTransformOnLoopDomain replay(tv, input_loop_ids, transform);
  }

 private:
  ReplayForwardTransformOnLoopDomain(
      TensorView* tv,
      const std::vector<IterDomain*>& input_loop_ids,
      const Expr* transform)
      : tv_(tv), input_loop_ids_(input_loop_ids) {
    OptOutConstDispatch::dispatch(transform);
  }

  using OptInConstDispatch::handle;

  int64_t getLoopIdPosition(IterDomain* loop_id) const {
    auto it = std::find(
        tv_->getLoopDomain().begin(), tv_->getLoopDomain().end(), loop_id);
    NVF_ERROR(
        it != tv_->getLoopDomain().end(),
        "Loop ID, ",
        loop_id->toString(),
        ", not found in ",
        tv_->toString());
    return static_cast<int64_t>(
        std::distance(tv_->getLoopDomain().begin(), it));
  }

  void handle(const Split* split) final {
    NVF_ERROR(input_loop_ids_.size() == 1);
    tv_->split(
        getLoopIdPosition(input_loop_ids_.at(0)),
        split->factor(),
        split->innerSplit());
  }

  void handle(const Merge* merge) final {
    NVF_ERROR(input_loop_ids_.size() == 2);
    tv_->merge(
        getLoopIdPosition(input_loop_ids_.at(0)),
        getLoopIdPosition(input_loop_ids_.at(1)));
  }

  void handle(const Resize* resize) final {
    NVF_ERROR(input_loop_ids_.size() == 1);
    NVF_ERROR(
        resize->out()->getIterType() != IterType::Symbolic,
        "Unexpected to have a symbolic ID: ",
        resize->out()->toString());
    // Pass the iter type explicitly to avoid generating a symblic ID
    tv_->resize(
        getLoopIdPosition(input_loop_ids_.at(0)),
        resize->leftExpand(),
        resize->rightExpand(),
        resize->out()->getIterType());
  }

  void handle(const Swizzle2D* swizzle_2d) final {
    NVF_THROW("Unsupported");
  }

  void handle(const Swizzle* swizzle) final {
    NVF_THROW("Unsupported");
  }

 private:
  TensorView* tv_ = nullptr;
  const std::vector<IterDomain*>& input_loop_ids_;
};

class LoopDomainScheduler {
 public:
  LoopDomainScheduler(
      std::vector<IterDomain*> ref_loop_dom,
      bool update_loop_domain_only = false)
      : ref_loop_dom_(std::move(ref_loop_dom)),
        update_loop_domain_only_(update_loop_domain_only) {
    NVF_ERROR(!ref_loop_dom_.empty());

    Fusion* fusion = ref_loop_dom_.front()->fusion();
    id_model_ = std::make_unique<IdModel>(fusion, /*build_graphs=*/false);
    id_model_->buildExactGraph();

    ref_id_groups_ = graph().toGroups(ref_loop_dom_);

    // Get all ancestors of the reference loop domain. Used in
    // getReplayPath.
    std::vector<ValGroup> all_val_groups =
        graph().disjointValSets().disjointSets();
    all_ancestors_of_ref_ = getReachableValsFrom<ValGraphBFS>(
        ref_id_groups_.vector(), all_val_groups, Direction::Backward, graph());
  }

  // Create the loop domain of a given tensor as specified by the
  // reference. The new loop domain is connected to the existing IDs of
  // the tensor by replaying exprs found in the ValGraph.
  void schedule(TensorView* tv) const;

 private:
  ValGraph& graph() const {
    return id_model_->idGraph(IdMappingMode::EXACT);
  }

  ValGraphBFS::ExprPath getReplayPath(TensorView* tv) const;

  // Replay an ExprGroup with given lists of input and output
  // groups. NOte that inputs and outputs are based on a given
  // direction. If it's Backward, the given inputs are used as the
  // outputs of the expr.
  Expr* replay(
      const ExprGroup& expr_g,
      Direction dir,
      const ValGroups& input_groups,
      const ValGroups& output_groups,
      const std::unordered_map<ValGroup, IterDomain*>& group_to_id) const {
    std::vector<IterDomain*> inputs;
    std::vector<IterDomain*> outputs;
    std::transform(
        input_groups.begin(),
        input_groups.end(),
        std::back_inserter(inputs),
        [&](const ValGroup& input_g) -> IterDomain* {
          return group_to_id.at(input_g);
        });
    std::transform(
        output_groups.begin(),
        output_groups.end(),
        std::back_inserter(outputs),
        [&](const ValGroup& output_g) -> IterDomain* {
          return group_to_id.at(output_g);
        });
    Expr* replayed_expr = LoopDomainSchedulerReplayTransform::replayAs(
        dir == Direction::Forward ? inputs : outputs,
        dir == Direction::Forward ? outputs : inputs,
        expr_g->front());
    return replayed_expr;
  }

 private:
  std::vector<IterDomain*> ref_loop_dom_;
  // If true, uses the current loop domain as the starting domain and
  // updates it to make it look like the given reference loop domain
  bool update_loop_domain_only_ = false;
  std::unique_ptr<IdModel> id_model_;
  ValGroups ref_id_groups_;
  ValGroups all_ancestors_of_ref_;
};

void LoopDomainScheduler::schedule(TensorView* tv) const {
  // Quick shortcut
  if (ref_id_groups_ == graph().toGroups(tv->getLoopDomain())) {
    // No need to change the current loop domain
    return;
  }

  // All of the existing IDs are reused as much as possible to
  // minimize creating new IDs.

  std::unordered_map<ValGroup, IterDomain*> group_to_id;
  ValGroups all_id_groups;
  // When update_mode_ is true, only the loop domain IDs are reused as
  // we attempt to transform the current loop domain to look like the
  // reference loop domain.
  auto all_ids =
      update_loop_domain_only_ ? tv->getLoopDomain() : tv->domain()->allIDs();
  for (auto id : all_ids) {
    const auto& group = graph().toGroup(id);
    group_to_id.emplace(group, id);
    all_id_groups.pushBack(group);
  }

  // New loop domain to set for the tv
  std::vector<IterDomain*> loop_domain(ref_loop_dom_.size());

  // Find missing IDs.
  bool has_missing_ids = false;
  for (const auto i : arange(ref_loop_dom_.size())) {
    const auto& ref_id_group = ref_id_groups_.at((int64_t)i);
    if (all_id_groups.has(ref_id_group)) {
      // This loop ID already exists.
      auto it = group_to_id.find(ref_id_group);
      NVF_ERROR(it != group_to_id.end());
      loop_domain.at(i) = it->second;
    } else {
      // Need to create a new ID for the loop ID
      has_missing_ids = true;
      // TODO: Don't force mapping at this point since that may not be necessary
      auto clone = ref_loop_dom_.at(i)->cloneWithoutRFactor(true);
      loop_domain.at(i) = clone;
      group_to_id.emplace(ref_id_group, clone);
      all_id_groups.pushBack(ref_id_group);
    }
  }

  // If no new ID is created, no expr replay is necessary
  if (!has_missing_ids) {
    tv->setLoopDomain(loop_domain);
    return;
  }

  const auto path_from_ref = getReplayPath(tv);
  const ExprGroups all_existing_expr_groups = update_loop_domain_only_
      ? ExprGroups{}
      : graph().toGroups(tv->domain()->allExprs());

  // Replay the path on the target tensor
  for (const auto& [expr_g, dir] : path_from_ref) {
    // Skip if the tensor already has the expr
    if (all_existing_expr_groups.has(expr_g)) {
      continue;
    }

    const auto input_groups = getInputsOfExpr(
        expr_g, dir, ValGraphInputs(graph()), ValGraphOutputs(graph()));
    const auto output_groups = getOutputsOfExpr(
        expr_g, dir, ValGraphInputs(graph()), ValGraphOutputs(graph()));

    // All inputs must be already in all_id_groups
    auto inputs_it = std::find_if(
        input_groups.begin(),
        input_groups.end(),
        [&](const ValGroup& input_g) -> bool {
          return !all_id_groups.has(input_g);
        });
    NVF_ERROR(
        inputs_it == input_groups.end(),
        "Unknown input group found: ",
        nvfuser::toString(*inputs_it));

    // Clone outputs if not found
    for (const auto& output_g : output_groups) {
      if (all_id_groups.has(output_g)) {
        continue;
      }

      // No need to force exact mapping since this clone is going to
      // be connected with tv
      auto clone = representativeId(output_g)->cloneWithoutRFactor();
      all_id_groups.pushBack(output_g);
      group_to_id.emplace(output_g, clone);
    }

    replay(expr_g, dir, input_groups, output_groups, group_to_id);
  }

  tv->setLoopDomain(loop_domain);
}

// The replay path of a tensor is a path from the reference loop
// domain to the root domain of the tensor. This path is used to
// augment the tensor with a new loop domain as specified by the
// reference domain. Intuitively, one might think that the shortest
// path in the ExactGraph could be a valid path. However, a ValGraph
// path cannot be replayed on a single tensor if the path contains a
// ValGroup that has multiple defining ExprGroups. That is because no
// IterDomain is allowed to have multiple defining Exprs.
//
// There are only three valid patterns: 1) forward only, 2) backward
// only and 3) forward then backward. Specifically, no backward expr
// cannot be traversed after a forward expr since that would mean the
// output of the forward expr would have multiple definitions.
//
// To find a valid path, we first find all ancestors of the reference
// loop domain. These ancestors domains are used to find a forward
// path to the target tensor. This path corresponds to the forward
// path of 1) or 3). To complete the path, a backward path from the
// reference domain to the forward path is prepended to the forward
// path.
//
// See LoopDomainSchedulingTest.ReshapeTraversalDirection for a
// concrete example.
ValGraphBFS::ExprPath LoopDomainScheduler::getReplayPath(TensorView* tv) const {
  // If not with the update mode, find the path to the root domain of
  // the tensor. It is important to use the root domain if available since there
  // can be multiple forward paths to the logical domain in the ValGraph. For
  // example,
  //
  // t0 = [i0]
  // t1 = reshape(t0, {i0}, {i0/4, 4})
  // t2 = reshape(t1, {i0/4, 4}, {i0})
  // t3 = reshape(t0, {i0}, {i0/8, 8})
  // t4 = reshape(t3, {i0/8, 8}, {i0})
  // t5 = t2 + t4
  //
  // Suppose we want to set the t2 loop domain as t0. In this case,
  // since the logical doamin of t2 is mapped with the logical domains
  // of t4, there're two paths from t0 loop domain to the t2
  // logical domain: one through t1 reshape and another through t3
  // reshape. Notice that the second path cannot be used as that would
  // mean the t2 logical domain would have another definition (exactly mapped
  // with the t4 merge reshape). This issue can be avoided by using the root
  // domain of tv2 as the target of path finding.
  //
  // In the case of the update mode, the target should be just the
  // current loop domain of the tensor.
  ValGroups tv_target_domains = graph().toGroups(TensorDomain::noBroadcasts(
      update_loop_domain_only_ ? tv->getLoopDomain()
                               : tv->getMaybeRootDomain()));

  // If all the target domains are an ancestor of the reference
  // domains, just a single backward BFS should be enough to find a
  // valid path
  if (std::all_of(
          tv_target_domains.begin(),
          tv_target_domains.end(),
          [&](const ValGroup& tv_target_domain) {
            return all_ancestors_of_ref_.has(tv_target_domain);
          })) {
    return ValGraphBFS::getExprGroupsBetween(
               graph(),
               ref_id_groups_,
               tv_target_domains,
               /*require_all_to_visited=*/true,
               Direction::Backward)
        .first;
  }

  // In the case of the update mode, the path from the reference is
  // assumed to just a backward traversal path.
  if (update_loop_domain_only_) {
    std::stringstream ss;
    ss << "Missing target ID groups: ";
    for (const auto& tv_target_domain : tv_target_domains) {
      if (!all_ancestors_of_ref_.has(tv_target_domain)) {
        ss << nvfuser::toString(tv_target_domain) << " ";
      }
    }
    NVF_THROW(
        "Trying to update the current loop domain but could not find a valid path from the reference: ",
        tv->toString(),
        ". ",
        ss.str());
  }

  // Find the forward path from the ancestors to the target tensor
  auto forward_path = ValGraphBFS::getExprGroupsBetween(
                          graph(),
                          all_ancestors_of_ref_,
                          tv_target_domains,
                          /*require_all_to_visited=*/true,
                          Direction::Forward)
                          .first;

  // Find the path from the ref to the forward path.
  auto inputs_of_forward_path = getInputsOfExprPath(
      forward_path, ValGraphInputs(graph()), ValGraphOutputs(graph()));

  auto backward_path = ValGraphBFS::getExprGroupsBetween(
                           graph(),
                           ref_id_groups_,
                           inputs_of_forward_path,
                           /*require_all_to_visited=*/true,
                           Direction::Backward)
                           .first;

  // Overall replay path = backward_path + forward_path
  ValGraphBFS::ExprPath replay_path;
  replay_path.reserve(backward_path.size() + forward_path.size());
  replay_path.insert(
      replay_path.end(), backward_path.begin(), backward_path.end());
  replay_path.insert(
      replay_path.end(), forward_path.begin(), forward_path.end());

  return replay_path;
}

} // namespace

void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    const std::vector<IterDomain*>& ref_loop_dom,
    bool update_loop_domain_only) {
  if (tvs.empty()) {
    return;
  }

  LoopDomainScheduler scheduler(ref_loop_dom, update_loop_domain_only);

  for (auto tv : tvs) {
    // Loop domain of fusion inputs should have no meaning,
    // nor should the loop domain of a tensor that has no logical
    // domain.
    if (tv->isFusionInput() || tv->getLogicalDomain().empty()) {
      continue;
    }
    scheduler.schedule(tv);
  }
}

void scheduleLoopDomainsBy(
    const std::vector<TensorView*>& tvs,
    Expr* transform,
    Direction replay_dir) {
  Fusion* fusion = transform->fusion();
  IdModel id_model(fusion, /*build_graphs=*/false);
  const ValGraph& exact_graph = id_model.buildExactGraph();

  const ValGroups input_groups = exact_graph.toGroups(transform->inputs());
  const ValGroups output_groups = exact_graph.toGroups(transform->outputs());

  for (auto tv : tvs) {
    // Check if either the inputs or the outputs are mapped with the
    // loop domain.

    std::vector<IterDomain*> input_ids;
    input_ids.reserve(transform->inputs().size());
    for (const auto& input_g : input_groups) {
      for (const auto loop_id : tv->getLoopDomain()) {
        if (input_g->has(loop_id)) {
          input_ids.push_back(loop_id);
        }
      }
    }

    std::vector<IterDomain*> output_ids;
    output_ids.reserve(transform->outputs().size());
    for (const auto& output_g : output_groups) {
      for (const auto loop_id : tv->getLoopDomain()) {
        if (output_g->has(loop_id)) {
          output_ids.push_back(loop_id);
        }
      }
    }

    // If all of the inputs are found, the tranform expr is replayed as
    // a forward op.
    Direction replay_dir_tv = Direction::Undefined;
    if (replay_dir != Direction::Backward &&
        input_ids.size() == transform->inputs().size()) {
      replay_dir_tv = Direction::Forward;
    } else if (
        replay_dir != Direction::Forward &&
        output_ids.size() == transform->outputs().size()) {
      replay_dir_tv = Direction::Backward;
    } else {
      // Replay not possible since none of inputs nor outputs are connected with
      // the transform
      continue;
    }

    // When the direction is forward, the TensorView transform
    // APIs, e.g., TensorView::split, can be used, which doesn't need
    // to use TensorView::setLoopDomain. This is important as
    // setLoopDomain may result in losing extra IDs added by prior
    // scheduleLoopDomain calls, which was indeed the case with the
    // Llama 3 RoPE backward (see also
    // https://github.com/NVIDIA/Fuser/issues/3571).
    if (replay_dir_tv == Direction::Forward) {
      ReplayForwardTransformOnLoopDomain::replayAs(tv, input_ids, transform);
      continue;
    }

    NVF_ERROR(input_ids.empty());
    for (const auto& ref_id : transform->inputs()) {
      auto clone = ref_id->as<IterDomain>()->cloneWithoutRFactor();
      input_ids.push_back(clone);
    }

    // The definition of the output IDs will be set to the newly
    // created expr. This is only allowed when the output IDs have no
    // definition yet.
    LoopDomainSchedulerReplayTransform::replayAs(
        input_ids, output_ids, transform);

    // Replace the inputs of the transform with the outputs
    auto new_loop_domain = tv->getLoopDomain();
    auto outermost_pos = (int64_t)tv->getLoopDomain().size();
    for (const auto& output_id : output_ids) {
      auto it =
          std::find(new_loop_domain.begin(), new_loop_domain.end(), output_id);
      NVF_ERROR(it != new_loop_domain.end());
      auto pos = (int64_t)std::distance(new_loop_domain.begin(), it);
      outermost_pos = std::min(outermost_pos, pos);
      new_loop_domain.erase(it);
    }

    for (auto it = input_ids.rbegin(); it != input_ids.rend(); ++it) {
      IterDomain* new_id = *it;
      new_loop_domain.insert(new_loop_domain.begin() + outermost_pos, new_id);
    }

    tv->setLoopDomain(new_loop_domain);
  }

  return;
}

void cancelReshapeInLoopDomains(TensorView* from_tv, bool skip_innermost_id) {
  Fusion* fusion = from_tv->fusion();
  IdModel id_model(fusion, /*build_graphs=*/false);
  id_model.buildExactGraph();
  const auto& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  // Reshapes producing these IDs should not be cancelled
  ValGroups reshape_dependent_ids;
  for (const ExprGroup& expr_g :
       exact_graph.disjointExprSets().disjointSets()) {
    if (expr_g->front()->isA<Resize>()) {
      reshape_dependent_ids.pushBack(exact_graph.inputGroups(expr_g));
    }
  }

  for (const ValGroup& val_g : exact_graph.disjointValSets().disjointSets()) {
    if (std::any_of(val_g->begin(), val_g->end(), [](Val* val) {
          NVF_ERROR(val->isA<IterDomain>());
          return val->as<IterDomain>()->isReduction();
        })) {
      reshape_dependent_ids.pushBack(val_g);
    }
  }

  auto all_dep_exprs_from_tv =
      DependencyCheck::getAllExprsBetween({from_tv}, fusion->outputs());

  // Visit all reshapes in a reverse topological order
  for (auto exprs_it = all_dep_exprs_from_tv.rbegin();
       exprs_it != all_dep_exprs_from_tv.rend();
       ++exprs_it) {
    auto reshape = dynamic_cast<ViewOp*>(*exprs_it);
    if (reshape == nullptr) {
      continue;
    }

    auto reshape_out = reshape->out();

    auto all_dep_vals =
        DependencyCheck::getAllValsBetween({reshape_out}, fusion->outputs());
    // Exclude reshape_out. These tensors are going to be updated by
    // replaying the reshape transform exprs using
    // scheduleLoopDomainsBy. Since the reshape output
    // tensor already has the exprs, replaying with
    // scheduleLoopDomainsBy would complain if not excluded. For the
    // reshape output tensor, setLoopDomain is done with the existing
    // IDs without replaying.
    all_dep_vals.erase(all_dep_vals.begin());
    auto all_dep_tvs = ir_utils::filterByType<TensorView>(all_dep_vals);

    // Find logical IDs that do not exist in the root domain. They are
    // the new IDs that are produced by this reshape op. If a logical
    // ID is already found in the root domain, there's nothing to do
    // for it.
    std::vector<IterDomain*> new_logical_ids;
    for (const auto& logical_id : reshape_out->getLogicalDomain()) {
      if (!reshape_out->domain()->isRoot(logical_id)) {
        new_logical_ids.push_back(logical_id);
      }
    }

    if (new_logical_ids.empty()) {
      // Nothing to do with a no-op reshape. This may not happen.
      continue;
    }

    // Find logical IDs that do not need to exist in the loop domain
    std::unordered_set<Val*> cancellable_ids;
    for (const auto new_logical_id : new_logical_ids) {
      auto new_id_group = exact_graph.toGroup(new_logical_id);
      // Not cancellable if used by resize or reduced.
      auto reachable_exprs = getReachableNodesFrom<ValGraphPermissiveBFS>(
          {new_id_group},
          {reshape_dependent_ids.begin(), reshape_dependent_ids.end()},
          Direction::Forward,
          exact_graph);
      if (!reachable_exprs.empty()) {
        continue;
      }

      cancellable_ids.insert(new_logical_id);
    }

    if (cancellable_ids.empty()) {
      continue;
    }

    // Update the loop domain by each of the reshape exprs in a
    // reverse topological order.
    auto reshape_exprs = DependencyCheck::getAllExprsBetween(
        {reshape_out->getRootDomain().begin(),
         reshape_out->getRootDomain().end()},
        {reshape_out->getLogicalDomain().begin(),
         reshape_out->getLogicalDomain().end()});

    std::unordered_set<Expr*> reshape_exprs_with_innermost_logical_id_set;
    if (skip_innermost_id) {
      auto reshape_exprs_with_innermost_logical_id =
          DependencyCheck::getAllExprsBetween(
              {reshape_out->getRootDomain().begin(),
               reshape_out->getRootDomain().end()},
              {reshape_out->getLogicalDomain().back()});
      reshape_exprs_with_innermost_logical_id_set = {
          reshape_exprs_with_innermost_logical_id.begin(),
          reshape_exprs_with_innermost_logical_id.end()};
    }

    auto reshape_out_loop_domain = reshape_out->getLoopDomain();

    for (auto reshape_exprs_it = reshape_exprs.rbegin();
         reshape_exprs_it != reshape_exprs.rend();
         ++reshape_exprs_it) {
      auto reshape_expr = *reshape_exprs_it;

      if (skip_innermost_id &&
          reshape_exprs_with_innermost_logical_id_set.count(reshape_expr)) {
        continue;
      }

      // If any of the output IDs of reshape_expr is not found in
      // cancellable_ids, that means the expr cannot be cancelled.
      if (std::any_of(
              reshape_expr->outputs().begin(),
              reshape_expr->outputs().end(),
              [&](Val* reshape_expr_out) -> bool {
                return !cancellable_ids.count(reshape_expr_out);
              })) {
        continue;
      }

      // Update all of the dependent TVs by this reshape expr
      scheduleLoopDomainsBy(
          all_dep_tvs.vector(), reshape_expr, Direction::Backward);

      cancellable_ids.insert(
          reshape_expr->inputs().begin(), reshape_expr->inputs().end());

      // For the reshape output tensor itself, since it already has the
      // reshape expr, it just needs
      // tv->setLoopDomain(tv->getRootDomain()). However, since some of the
      // reshape exprs may not be cancellable, update a vector of the
      // loop IDs for each of the cancelled exprs individually and use
      // it to set the loop domain of the reshape output tensor

      // Insert the input IDs to the loop domain
      auto insert_pos = std::find(
          reshape_out_loop_domain.begin(),
          reshape_out_loop_domain.end(),
          reshape_expr->outputs().front());
      NVF_ERROR(insert_pos != reshape_out_loop_domain.end());
      for (auto inp : reshape_expr->inputs()) {
        insert_pos =
            reshape_out_loop_domain.insert(insert_pos, inp->as<IterDomain>());
        ++insert_pos;
      }

      // Remove the output IDs
      reshape_out_loop_domain.erase(
          std::remove_if(
              reshape_out_loop_domain.begin(),
              reshape_out_loop_domain.end(),
              [&](IterDomain* cur_loop_id) {
                return std::find(
                           reshape_expr->outputs().begin(),
                           reshape_expr->outputs().end(),
                           cur_loop_id) != reshape_expr->outputs().end();
              }),
          reshape_out_loop_domain.end());
    }

    reshape_out->setLoopDomain(reshape_out_loop_domain);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
