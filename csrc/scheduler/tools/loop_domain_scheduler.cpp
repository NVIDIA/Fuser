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

class LoopDomainScheduler {
 public:
  LoopDomainScheduler(std::vector<IterDomain*> ref_loop_dom)
      : ref_loop_dom_(std::move(ref_loop_dom)) {
    NVF_ERROR(!ref_loop_dom_.empty());
#if 0
    // For now, ref must not be a broadcast domain
    NVF_ERROR(
        std::none_of(
            ref_loop_dom_.begin(),
            ref_loop_dom_.end(),
            [](IterDomain* id) { return id->isBroadcast(); }),
        "Broadcast referene not supported: ",
        toDelimitedString(ref_loop_dom_));
#endif
    Fusion* fusion = ref_loop_dom_.front()->fusion();
    id_model_ = std::make_unique<IdModel>(fusion, /*build_graphs=*/false);
    id_model_->buildExactGraph();

    ref_id_groups_ = graph().toGroups(ref_loop_dom_);

    // Get all ancestors of the reference loop domain. Used in
    // getReplayPath.
    std::vector<ValGroup> all_val_groups =
        graph().disjointValSets().disjointSets();
    all_ancestors_of_ref_ = ValGraphBFS::getReachableValsFrom(
        graph(), ref_id_groups_, all_val_groups, Direction::Backward);
  }

  // Create the loop domain of a given tensor as specified by the
  // reference. The new loop domain is connected to the existing IDs of
  // the tensor by replaying exprs found in the ValGraph.
  void schedule(TensorView* tv) const;

  void replaceAndAppend(TensorView* tv) const;

 private:
  ValGraph& graph() const {
    return id_model_->idGraph(IdMappingMode::EXACT);
  }

  ValGraphBFS::ExprPath getReplayPath(
      TensorView* tv,
      bool require_all_visited = true) const;

  std::optional<ValGraphBFS::ExprPath> getReplayPathForResize(
      TensorView* tv,
      bool require_all_visited = true) const;

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
  std::unique_ptr<IdModel> id_model_;
  ValGroups ref_id_groups_;
  ValGroups all_ancestors_of_ref_;
};

void LoopDomainScheduler::schedule(TensorView* tv) const {
  std::cerr << "LoopDomainScheduler::schedule: " << tv->toString() << "\n";
  // Quick shortcut
  if (ref_id_groups_ == graph().toGroups(tv->getLoopDomain())) {
    // No need to change the current loop domain
    std::cerr << "Already equal\n";
    return;
  }

  const auto resize_path_from_ref = getReplayPathForResize(tv);
  bool resize_war = resize_path_from_ref.has_value();

  // All of the existing IDs are reused as much as possible to
  // minimize creating new IDs.
  auto all_ids = resize_war ? tv->getLoopDomain() : tv->domain()->allIDs();
  std::unordered_map<ValGroup, IterDomain*> group_to_id;
  ValGroups all_id_groups;
  for (auto id : all_ids) {
    const auto& group = graph().toGroup(id);
    group_to_id.emplace(group, id);
    all_id_groups.pushBack(group);
  }

  // New loop domain to set for the tv
  std::vector<IterDomain*> loop_domain(ref_loop_dom_.size());

  // Find missing IDs.
  bool has_missing_ids = false;
  for (const auto i : c10::irange(ref_loop_dom_.size())) {
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
    std::cerr << "No missing ids: " << toDelimitedString(loop_domain) << "\n";
    tv->setLoopDomain(loop_domain);
    return;
  }

  const auto path_from_ref =
      resize_war ? resize_path_from_ref.value() : getReplayPath(tv);
  const ExprGroups all_existing_expr_groups =
      resize_war ? ExprGroups{} : graph().toGroups(tv->domain()->allExprs());

  // Replay the path on the target tensor
  for (const auto& [expr_g, dir] : path_from_ref) {
    std::cerr << "Visiting " << expr_g->front()->toString();
    // Skip if the tensor already has the expr
    if (all_existing_expr_groups.has(expr_g)) {
      continue;
    }

    const auto input_groups = inputGroups(graph(), expr_g, dir);
    const auto output_groups = outputGroups(graph(), expr_g, dir);

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

    std::cerr << "Replaying inputs: " << nvfuser::toString(input_groups)
              << ", outputs: " << nvfuser::toString(output_groups) << "\n";
    auto replayed_expr =
        replay(expr_g, dir, input_groups, output_groups, group_to_id);
    std::cerr << "Replayed: " << replayed_expr->toString();
  }

  std::cerr << "setLoopDomain: " << toDelimitedString(loop_domain) << "\n";
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
ValGraphBFS::ExprPath LoopDomainScheduler::getReplayPath(
    TensorView* tv,
    bool require_all_visited) const {
  // Find the path to the root domain of the tensor. It is important
  // to use the root domain if available since there can be multiple
  // forward paths to the logical domain in the ValGraph. For example,
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

  ValGroups tv_loop_domains =
      graph().toGroups(TensorDomain::noBroadcasts(tv->getLoopDomain()));

  ValGroups tv_root_domains =
      graph().toGroups(TensorDomain::noBroadcasts(tv->getMaybeRootDomain()));

  // If all the target domains are an ancestor of the reference
  // domains, just a single backward BFS should be enough to find a
  // valid path
  if (std::all_of(
          tv_loop_domains.begin(),
          tv_loop_domains.end(),
          [&](const ValGroup& tv_loop_domain) {
            return all_ancestors_of_ref_.has(tv_loop_domain);
          })) {
    return ValGraphBFS::getExprsBetween(
        graph(),
        ref_id_groups_,
        tv_loop_domains,
        /*require_all_to_visited=*/true,
        Direction::Backward);
  }

  // Find the forward path from the ancestors to the target tensor
  auto forward_path_to_root = ValGraphBFS::getExprsBetween(
      graph(),
      all_ancestors_of_ref_,
      tv_root_domains,
      /*require_all_to_visited=*/require_all_visited,
      Direction::Forward);

  auto outputs_of_forward_path =
      getOutputsOfExprPath(graph(), forward_path_to_root);

  auto root_to_loop = ValGraphBFS::getExprsBetween(
      graph(),
      outputs_of_forward_path,
      tv_loop_domains,
      /*require_all_to_visited=*/require_all_visited,
      Direction::Forward);

  // Find the path from the ref to the forward path.
  auto inputs_of_forward_path =
      getInputsOfExprPath(graph(), forward_path_to_root);

  auto backward_path_from_ref = ValGraphBFS::getExprsBetween(
      graph(),
      ref_id_groups_,
      inputs_of_forward_path,
      /*require_all_to_visited=*/true,
      Direction::Backward);

  // Overall replay path = backward_path + forward_path_to_root +
  // forward_path_to_loop
  ValGraphBFS::ExprPath ref_to_loop;
  ref_to_loop.reserve(
      backward_path_from_ref.size() + forward_path_to_root.size() +
      root_to_loop.size());
  ref_to_loop.insert(
      ref_to_loop.end(),
      backward_path_from_ref.begin(),
      backward_path_from_ref.end());
  ref_to_loop.insert(
      ref_to_loop.end(),
      forward_path_to_root.begin(),
      forward_path_to_root.end());
  ref_to_loop.insert(
      ref_to_loop.end(), root_to_loop.begin(), root_to_loop.end());

  return ref_to_loop;
}

// WAR for resize
std::optional<ValGraphBFS::ExprPath> LoopDomainScheduler::
    getReplayPathForResize(TensorView* tv, bool require_all_visited) const {
  std::cerr << "getReplayPathForResize for " << tv->toString() << "\n";

  std::vector<ValGraphBFS::NodeType> ref_definitions;
  // ValGraphBFS::ExprPath path_to_parents;
  for (const auto& ref_id_group : ref_id_groups_) {
    for (const ExprGroup& eg : graph().getDefinitions(ref_id_group)) {
      ref_definitions.emplace_back(eg);
      // path_to_parents.emplace_back(eg, Direction::Backward);
    }
  }

  ValGroups tv_loop_domains =
      graph().toGroups(TensorDomain::noBroadcasts(tv->getLoopDomain()));

  ValGraphBFS::ExprPath path = ValGraphBFS::getExprsBetween(
      graph(),
      ref_definitions,
      std::vector<ValGraphBFS::NodeType>{
          tv_loop_domains.begin(), tv_loop_domains.end()},
      /*require_all_to_visited=*/false,
      Direction::Backward);

  std::cerr << "Path from parent\n";
  for (const auto& [eg, dir] : path) {
    std::cerr << eg->front()->toString();
  }

  const auto path_vals = getValsOfExprPath(graph(), path);

  bool all_ref_used = std::all_of(
      ref_id_groups_.begin(),
      ref_id_groups_.end(),
      [&](const ValGroup& ref_id_group) {
        return path_vals.has(ref_id_group) || tv_loop_domains.has(ref_id_group);
      });

  bool all_target_reached = std::all_of(
      tv_loop_domains.begin(),
      tv_loop_domains.end(),
      [&](const ValGroup& tv_taget_domain) {
        return path_vals.has(tv_taget_domain) ||
            ref_id_groups_.has(tv_taget_domain);
      });

  if (!all_ref_used && !all_target_reached) {
    return std::nullopt;
  }

  if (require_all_visited) {
    NVF_ERROR(all_target_reached);
  }

  // Valid path found. Append with upward_path
  std::cerr << "Resize WAR: taking a backward path for " << tv->toString()
            << "\n";
  for (const auto& [eg, dir] : path) {
    std::cerr << eg->front()->toString();
  }

  return path;
}

void LoopDomainScheduler::replaceAndAppend(TensorView* tv) const {
  std::cerr << "LoopDomainScheduler::replaceOrAppend: " << tv->toString()
            << "\n";

  const auto resize_path_from_ref = getReplayPathForResize(tv, false);
  bool resize_war = resize_path_from_ref.has_value();

  // All of the existing IDs are reused as much as possible to
  // minimize creating new IDs.
  auto all_ids = resize_war ? tv->getLoopDomain() : tv->domain()->allIDs();
  std::unordered_map<ValGroup, IterDomain*> group_to_id;
  ValGroups all_id_groups;
  for (auto id : all_ids) {
    const auto& group = graph().toGroup(id);
    group_to_id.emplace(group, id);
    all_id_groups.pushBack(group);
  }

  // New loop domain to set for the tv
  // std::vector<IterDomain*> loop_domain;

  // Find missing IDs.
  // bool has_missing_ids = false;
  for (const auto i : c10::irange(ref_loop_dom_.size())) {
    const auto& ref_id_group = ref_id_groups_.at((int64_t)i);
    if (all_id_groups.has(ref_id_group)) {
      // This loop ID already exists.
      auto it = group_to_id.find(ref_id_group);
      NVF_ERROR(it != group_to_id.end());
      // loop_domain.at(i) = it->second;
    } else {
      // Need to create a new ID for the loop ID
      // has_missing_ids = true;
      // TODO: Don't force mapping at this point since that may not be necessary
      auto clone = ref_loop_dom_.at(i)->cloneWithoutRFactor(true);
      // loop_domain.at(i) = clone;
      group_to_id.emplace(ref_id_group, clone);
      all_id_groups.pushBack(ref_id_group);
    }
  }

  const auto path_from_ref =
      resize_war ? resize_path_from_ref.value() : getReplayPath(tv, false);

  const ExprGroups all_existing_expr_groups =
      graph().toGroups(tv->domain()->allExprs());

  auto path_inputs = getInputsOfExprPath(graph(), path_from_ref);

  auto path_outputs = getOutputsOfExprPath(graph(), path_from_ref);

  ValGroups disconnected_ref_groups;
  for (const auto& ref_id_group : ref_id_groups_) {
    if (!path_inputs.has(ref_id_group)) {
      disconnected_ref_groups.pushBack(ref_id_group);
    }
  }

  std::vector<IterDomain*> new_loop_domain;

  for (const auto& cur_loop_id : tv->getLoopDomain()) {
    // If it's an output of the path, it's replaced by the new ref
    // ID, which is just appended to the list
    if (path_outputs.has(graph().toGroup(cur_loop_id)) ||
        ref_id_groups_.has(graph().toGroup(cur_loop_id))) {
      continue;
    } else {
      new_loop_domain.push_back(cur_loop_id);
    }
  }

  for (const auto& ref_id_group : ref_id_groups_) {
    auto it = group_to_id.find(ref_id_group);
    NVF_ERROR(it != group_to_id.end());
    new_loop_domain.push_back(it->second);
  }

  // Replay the path on the target tensor
  for (const auto& [expr_g, dir] : path_from_ref) {
    std::cerr << "Visiting " << expr_g->front()->toString();
    // Skip if the tensor already has the expr
    if (all_existing_expr_groups.has(expr_g)) {
      continue;
    }

    const auto input_groups = inputGroups(graph(), expr_g, dir);
    const auto output_groups = outputGroups(graph(), expr_g, dir);

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

    std::cerr << "Replaying inputs: " << nvfuser::toString(input_groups)
              << ", outputs: " << nvfuser::toString(output_groups) << "\n";
    auto replayed_expr =
        replay(expr_g, dir, input_groups, output_groups, group_to_id);
    std::cerr << "Replayed: " << replayed_expr->toString();
  }

  std::cerr << "setLoopDomain: " << toDelimitedString(new_loop_domain) << "\n";
  tv->setLoopDomain(new_loop_domain);
}

} // namespace

void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    const std::vector<IterDomain*>& ref_loop_dom) {
  if (tvs.empty()) {
    return;
  }

  LoopDomainScheduler scheduler(ref_loop_dom);

  for (auto tv : tvs) {
    scheduler.schedule(tv);
  }
}

void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    IterDomain* ref_loop_id) {
  if (tvs.empty()) {
    return;
  }

  LoopDomainScheduler scheduler({ref_loop_id});

  for (auto tv : tvs) {
    scheduler.replaceAndAppend(tv);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
