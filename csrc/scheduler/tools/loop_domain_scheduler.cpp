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
  LoopDomainScheduler(std::vector<IterDomain*> ref_loop_dom, int64_t pos)
      : ref_loop_ids_(std::move(ref_loop_dom)), pos_(pos) {
    NVF_ERROR(!ref_loop_ids_.empty());
    NVF_ERROR(pos_ > 0 && pos_ <= ref_loop_ids_.size());

    ref_loop_ids_ = std::vector<IterDomain*>{
        ref_loop_ids_.begin(), ref_loop_ids_.begin() + pos_};

    // For now, ref must not be a broadcast domain
    NVF_ERROR(
        std::none_of(
            ref_loop_ids_.begin(),
            ref_loop_ids_.end(),
            [](IterDomain* id) { return id->isBroadcast(); }),
        "Broadcast referene not supported: ",
        toDelimitedString(ref_loop_ids_));

    Fusion* fusion = ref_loop_ids_.front()->fusion();
    id_model_ = std::make_unique<IdModel>(fusion, /*build_graphs=*/false);
    id_model_->buildExactGraph();
    id_model_->buildBroadcastGraph();

    ref_id_groups_ = graph().toGroups(ref_loop_ids_);
  }

  // Create the loop domain of a given tensor as specified by the
  // reference. The new loop domain is connected to the existing IDs of
  // the tensor by replaying exprs found in the ValGraph.
  void schedule(TensorView* tv) const;

  std::pair<std::vector<IterDomain*>, std::vector<ValGroup>> replayReference(
      TensorView* tv) const;

 private:
  ValGraph& graph() const {
    return id_model_->idGraph(IdMappingMode::EXACT);
  }

  ValGraphBFS::ExprPath getReplayPath(
      TensorView* tv,
      const std::vector<ValGroup>& ref_id_groups) const;

  int64_t findMatchingPos(
      TensorView* tv,
      const std::vector<ValGroup>& ref_id_groups) const;

  std::vector<ValGroup> getComplimentedReferenceGroups(TensorView* tv) const;

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
  std::vector<IterDomain*> ref_loop_ids_;
  int64_t pos_;
  std::unique_ptr<IdModel> id_model_;
  ValGroups ref_id_groups_;
  // ValGroups all_ancestors_of_ref_;
};

void LoopDomainScheduler::schedule(TensorView* tv) const {
  std::cerr << "Scheduling " << tv->toString() << "\n";

  const auto current_loop_groups = graph().toGroups(tv->getLoopDomain());

  // Quick shortcut
  if (ref_id_groups_ == current_loop_groups) {
    // No need to change the current loop domain
    return;
  }

  const auto [new_loop_ids, ref_id_groups] = replayReference(tv);

  int64_t matching_pos = findMatchingPos(tv, ref_id_groups);

  std::cerr << "New loop IDs: " << toDelimitedString(new_loop_ids) << "\n";

  std::vector<IterDomain*> new_loop_domain;
  new_loop_domain.reserve(
      new_loop_ids.size() + (tv->getLoopDomain().size() - matching_pos));

  new_loop_domain.insert(
      new_loop_domain.end(), new_loop_ids.begin(), new_loop_ids.end());
  new_loop_domain.insert(
      new_loop_domain.end(),
      tv->getLoopDomain().begin() + matching_pos,
      tv->getLoopDomain().end());

  std::cerr << "New loop domain: " << toDelimitedString(new_loop_domain)
            << "\n";

  tv->setLoopDomain(new_loop_domain);
}

std::vector<ValGroup> LoopDomainScheduler::getComplimentedReferenceGroups(
    TensorView* tv) const {
  const auto& graph = id_model_->idGraph(IdMappingMode::EXACT);
  const auto ref_id_groups = graph.toGroups(ref_loop_ids_);

  std::cerr << "Ref id groups: " << nvfuser::toString(ref_id_groups) << "\n";

  const auto logical_groups = graph.toGroups(tv->getLogicalDomain());

  auto path_to_ref =
      ValGraphBFS::getExprsBetween(graph, logical_groups, ref_id_groups);

  for (const auto& [expr_g, dir] : path_to_ref) {
    std::cerr << "To ref " << dir << " " << nvfuser::toString(expr_g) << " "
              << expr_g->front()->toString();
  }

  auto outputs_of_path_to_ref = getOutputsOfExprPath(graph, path_to_ref);
  std::cerr << "Outputs: " << nvfuser::toString(outputs_of_path_to_ref) << "\n";
  std::vector<ValGroup> complimented_ref_groups = ref_id_groups.vector();
  for (const auto& output : outputs_of_path_to_ref) {
    if (std::find(ref_id_groups.begin(), ref_id_groups.end(), output) ==
        ref_id_groups.end()) {
      // missing ref ID
      complimented_ref_groups.push_back(output);
    }
  }

  std::cerr << "Complimented ref IDs for " << tv->toString() << "\n";
  for (const auto& g : complimented_ref_groups) {
    std::cerr << nvfuser::toString(g) << ": " << g->front()->toString() << "\n";
  }
  return complimented_ref_groups;
}

int64_t LoopDomainScheduler::findMatchingPos(
    TensorView* tv,
    const std::vector<ValGroup>& ref_id_groups) const {
  // Need to use the Broadcast graph, but the graph is built before
  // these new loop IDs are created
  const auto& broadcast_graph = id_model_->idGraph(IdMappingMode::BROADCAST);

  ValGroups new_loop_id_groups;
  for (const auto& exact_group : ref_id_groups) {
    new_loop_id_groups.pushBack(broadcast_graph.toGroup(exact_group->front()));
  }

  const auto current_loop_groups =
      broadcast_graph.toGroups(tv->getLoopDomain());

  auto reachable_loop_groups = ValGraphBFS::getReachableValsFrom(
      broadcast_graph, new_loop_id_groups, current_loop_groups);

  // Reachable loop groups must be outermost
  auto reachable_loop_groups_it = reachable_loop_groups.begin();
  int64_t matching_pos = 0;
  bool mismatch_found = false;
  for (const auto& loop_group : current_loop_groups) {
    if (reachable_loop_groups_it != reachable_loop_groups.end() &&
        loop_group == *reachable_loop_groups_it) {
      NVF_ERROR(
          !mismatch_found,
          "Nonadjacent matching loop group found: ",
          nvfuser::toString(loop_group));
      ++reachable_loop_groups_it;
      ++matching_pos;
    } else {
      mismatch_found = true;
    }
  }

  std::cerr << "Matching outermost loop IDs: "
            << toDelimitedString(
                   tv->getLoopDomain().begin(),
                   tv->getLoopDomain().begin() + matching_pos)
            << "\n";
  std::cerr << "Remaining innermost loop IDs: "
            << toDelimitedString(
                   tv->getLoopDomain().begin() + matching_pos,
                   tv->getLoopDomain().end())
            << "\n";
  return matching_pos;
}

// TODO: Refactor this as ValGraphReplay?
std::pair<std::vector<IterDomain*>, std::vector<ValGroup>> LoopDomainScheduler::
    replayReference(TensorView* tv) const {
  std::cerr << "Replaying on " << tv->toString() << "\n";

  // All of the existing IDs are reused as much as possible to
  // minimize creating new IDs.
  auto all_ids = tv->domain()->allIDs();
  std::unordered_map<ValGroup, IterDomain*> group_to_id;
  ValGroups all_id_groups;
  for (auto id : all_ids) {
    const auto& group = graph().toGroup(id);
    group_to_id.emplace(group, id);
    all_id_groups.pushBack(group);
  }

  const auto ref_id_groups = getComplimentedReferenceGroups(tv);

  // New loop domain to set for the tv
  std::vector<IterDomain*> loop_ids;

  bool has_missing_ids = false;
  for (const auto& ref_id_group : ref_id_groups) {
    if (all_id_groups.has(ref_id_group)) {
      // This loop ID already exists.
      auto it = group_to_id.find(ref_id_group);
      NVF_ERROR(it != group_to_id.end());
      loop_ids.push_back(it->second);
    } else {
      // Need to create a new ID for the loop ID
      has_missing_ids = true;
      // TODO: Don't force mapping at this point since that may not be necessary
      auto clone =
            representativeId(ref_id_group)->cloneWithoutRFactor(true);
      loop_ids.push_back(clone);
      group_to_id.emplace(ref_id_group, clone);
      all_id_groups.pushBack(ref_id_group);
      std::cerr << "New clone: " << clone->toString()
                << ", original: " << ref_id_group->front()->toString() << "\n";
    }
  }

  std::cerr << "Loop IDs: " << toDelimitedString(loop_ids) << "\n";

  // If no new ID is created, no expr replay is necessary
  if (!has_missing_ids) {
    return {loop_ids, ref_id_groups};
  }

  const auto path_from_ref = getReplayPath(tv, ref_id_groups);
  const ExprGroups all_existing_expr_groups =
      graph().toGroups(tv->domain()->allExprs());

  // Replay the path on the target tensor
  for (const auto& [expr_g, dir] : path_from_ref) {
    std::cerr << "Replaying " << dir << " " << nvfuser::toString(expr_g) << " "
              << expr_g->front()->toString();

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

    auto replayed_expr =
        replay(expr_g, dir, input_groups, output_groups, group_to_id);
    std::cerr << "Replayed expr: " << replayed_expr->toString();
  }

  std::cerr << "setLoopDomain: " << toDelimitedString(loop_ids) << "\n";

  return {loop_ids, ref_id_groups};
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
    const std::vector<ValGroup>& ref_id_groups) const {
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
#if 0
  ValGroups tv_target_domains =
      graph().toGroups(TensorDomain::noBroadcasts(tv->getMaybeRootDomain()));
#else
  ValGroups tv_target_domains = graph().toGroups(tv->getMaybeRootDomain());
#endif

  // Get all ancestors of the reference loop domain. Used in
  // getReplayPath.
  const auto& all_val_groups = graph().disjointValSets().disjointSets();
  const auto all_ancestors_of_ref = ValGraphBFS::getReachableValsFrom(
      graph(), ref_id_groups, all_val_groups, Direction::Backward);

  std::cerr << "All ancestors: " << nvfuser::toString(all_ancestors_of_ref)
            << "\n";

  // If all the target domains are an ancestor of the reference
  // domains, just a single backward BFS should be enough to find a
  // valid path
  if (std::all_of(
          tv_target_domains.begin(),
          tv_target_domains.end(),
          [&](const ValGroup& tv_target_domain) {
            return all_ancestors_of_ref.has(tv_target_domain);
          })) {
    return ValGraphBFS::getExprsBetween(
        graph(),
        ref_id_groups,
        tv_target_domains,
        /*require_all_to_visited=*/true,
        Direction::Backward);
  }

  // Find the forward path from the ancestors to the target
  // tensor. Not all of the target domains correspond to the reference
  // loop IDs, so it may not reach all target domains, which is fine.
  auto forward_path = ValGraphBFS::getExprsBetween(
      graph(),
      all_ancestors_of_ref,
      tv_target_domains,
      /*require_all_to_visited=*/false,
      Direction::Forward);

  // Find the path from the ref to the forward path.
  auto inputs_of_forward_path = getInputsOfExprPath(graph(), forward_path);

  auto backward_path = ValGraphBFS::getExprsBetween(
      graph(),
      ref_id_groups,
      inputs_of_forward_path,
      /*require_all_to_visited=*/true,
      Direction::Backward);

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
    int64_t pos) {
  pos = wrapDim(pos, (int64_t)ref_loop_dom.size() + 1);
  std::cerr << "Pos: " << pos << "\n";
  if (tvs.empty() || pos == 0) {
    return;
  }

  LoopDomainScheduler scheduler(ref_loop_dom, pos);

  for (auto tv : tvs) {
    scheduler.schedule(tv);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
