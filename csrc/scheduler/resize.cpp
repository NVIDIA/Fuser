// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/resize.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
  std::cerr << "ResizeScheduler::canScheduleCompileTime\n";

  if (!ir_utils::hasOpsOfType<SliceOp, PadOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No resize op to schedule");
    return false;
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  // Add more conditions to check

  // TODO: Reject padding of unsqueezeed broadcast IDs. Backward propagation
  // would fail otherwise.

  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  std::vector<Expr*> resize_ops =
      ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  for (const auto& expr_g : graph.disjointExprSets().disjointSets()) {
    auto resize = dynamic_cast<Resize*>(expr_g->front());
    if (resize == nullptr) {
      continue;
    }

    ValGroups input_groups = graph.inputGroups(expr_g);
    NVF_ERROR(input_groups.size() == 1);
    const ValGroup& input_group = input_groups.at(0);
    if (!input_group->front()->as<IterDomain>()->isBroadcast()) {
      continue;
    }

    ExprGroups def_of_input = graph.getDefinitions(input_group);
    if (!def_of_input.empty()) {
      // This should be another resize
      NVF_ERROR(def_of_input.front()->front()->isA<Resize>());
      continue;
    }

    // It should still be fine if it's part of the fusion input
    // tensors
    if (std::any_of(
            fusion->inputs().begin(), fusion->inputs().end(), [&](Val* input) {
              auto input_tv = dynamic_cast<TensorView*>(input);
              if (input_tv == nullptr) {
                return false;
              }
              for (auto input_logical_id : input_tv->getLogicalDomain()) {
                if (input_group->has(input_logical_id)) {
                  return true;
                }
              }
              return false;
            })) {
      continue;
    }

    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Resizing of unsqueezed broadcast IDs is not supported: ",
        resize->toString(),
        ", ",
        nvfuser::toString(input_group));
    return false;
  }

#if 0
  std::vector<Expr*> resize_ops =
      ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  // Find an output that is a dependent of all of the resize ops
  TensorView* ref_output = nullptr;
  if (fusion->outputs().size() == 1) {
    ref_output = fusion->outputs().at(0)->as<TensorView>();
  } else {
    std::cerr << "Multiple outputs: " << toDelimitedString(fusion->outputs())
              << "\n";
    std::vector<std::unordered_set<Val*>> all_outputs;
    all_outputs.reserve(resize_ops.size());
    for (const Expr* resize_op : resize_ops) {
      all_outputs.emplace_back(
          DependencyCheck::getAllOutputsOf({resize_op->output(0)}));
      if (resize_op->output(0)->isFusionOutput()) {
        all_outputs.back().insert(resize_op->output(0));
      }
      std::cerr << "output of " << resize_op->toString() << ": "
                << toDelimitedString(all_outputs.back()) << "\n";
    }

    for (auto output_tv :
         ir_utils::filterByType<TensorView>(fusion->outputs())) {
      if (std::all_of(
              all_outputs.begin(), all_outputs.end(), [&](const auto& outputs) {
                return outputs.count(output_tv);
              })) {
        ref_output = output_tv;
        break;
      }
    }

    if (ref_output == nullptr) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Cannot find any reference output candidate");
      return false;
    }


    // All output IDs must be connected without resize ops. This can
    // be lifted.
    IdModel id_model(fusion, /*build_models=*/false);
    const auto& graph = id_model.buildBroadcastGraph();

    for (auto output_tv :
         ir_utils::filterByType<TensorView>(fusion->outputs())) {
      if (output_tv == ref_output) {
        continue;
      }

      auto exprs_from_ref = ValGraphBFS::getExprsBetween(
          graph,
          graph.toGroups(ref_output->getLogicalDomain()),
          graph.toGroups(output_tv->getLogicalDomain()),
          /*require_all_to_visited=*/false);

      // Reject if there's any resize
      if (std::any_of(
              exprs_from_ref.begin(),
              exprs_from_ref.end(),
              [](const auto& path_eg_dir) {
                return path_eg_dir.first->front()->template isA<Resize>();
              })) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Has another output that has different resize op: ",
            output_tv->toString());
        return false;
      }

      if (!ValGraphBFS::getUnreachableValsFrom(
               graph,
               graph.toGroups(ref_output->getLogicalDomain()),
               graph.toGroups(output_tv->getLogicalDomain()),
               exprs_from_ref)
               .empty()) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Has another output that has disconnected ID: ",
            output_tv->toString());
        return false;
      }
    }

  }
#endif

  return true;
}

std::unique_ptr<HeuristicParams> ResizeScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ResizeScheduler::computeHeuristics");
  auto params = std::make_unique<HeuristicParams>(SchedulerType::Resize);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

namespace {

std::vector<std::pair<TensorView*, std::vector<TensorView*>>>
getReferenceTensors(Fusion* fusion) {
  std::vector<TensorView*> ref_candidates;

  std::cerr << "getReferenceTensors\n";
  fusion->printMath();

  const auto all_tvs = fusion->allTvs();

  DisjointSets<TensorView*> disjoint_val_sets;

  std::vector<Expr*> resize_ops =
      ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  // Group all tvs that are dependent on resize op outputs
  for (Expr* resize_op : resize_ops) {
    auto ref_tv = resize_op->output(0)->as<TensorView>();

    auto dep_vals = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {ref_tv});

    for (auto dep_tv : ir_utils::filterByType<TensorView>(dep_vals)) {
      // Don't add inputs. Inputs are not replicated nor scheduled.
      if (dep_tv->isFusionInput()) {
        continue;
      }
      std::cerr << "Mapping " << ref_tv->toString() << " and "
                << dep_tv->toString() << "\n";
      disjoint_val_sets.mapEntries(ref_tv, dep_tv);
    }
  }

  // TODO: Reuse
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& broadcast_graph = id_model.buildBroadcastGraph();

  for (const auto i : c10::irange(resize_ops.size() - 1)) {
    for (const auto j : c10::irange(i + 1, resize_ops.size())) {
      auto out_tv_i = resize_ops.at(i)->output(0)->as<TensorView>();
      auto out_tv_j = resize_ops.at(j)->output(0)->as<TensorView>();
      if (disjoint_val_sets.strictAreMapped(out_tv_i, out_tv_j)) {
        continue;
      }

      const auto out_tv_i_loop_groups =
          broadcast_graph.toGroups(out_tv_i->getLoopDomain());
      const auto out_tv_j_loop_groups =
          broadcast_graph.toGroups(out_tv_j->getLoopDomain());

      bool same_loop_domain =
          broadcast_graph.toGroups(out_tv_i->getLoopDomain()).set() ==
          broadcast_graph.toGroups(out_tv_j->getLoopDomain()).set();
      std::cerr << "Comparing " << out_tv_i->toString() << " and "
                << out_tv_j->toString() << ": " << same_loop_domain << "\n";
      if (!same_loop_domain) {
        auto [path_from_i_to_j, all_visited] = ValGraphBFS::getExprsBetween(
            broadcast_graph,
            out_tv_i_loop_groups,
            out_tv_j_loop_groups,
            /*require_all_to_visited=*/false);
        if (!all_visited) {
          // There are some unreachable loop groups
          continue;
        }

        // If there's any resize node, don't merge them
        if (std::any_of(
                path_from_i_to_j.begin(),
                path_from_i_to_j.end(),
                [](const auto& path_component) {
                  return path_component.first->front()->template isA<Resize>();
                })) {
          continue;
        }
      }

      std::cerr << "Same loop domain: " << out_tv_i->toString() << " and "
                << out_tv_j->toString() << "\n";
      disjoint_val_sets.mapEntries(out_tv_i, out_tv_j);
    }
  }

  const auto num_disjoint_resize_groups = disjoint_val_sets.size();

  std::cerr << "Number of disjoint resize groups: "
            << num_disjoint_resize_groups << "\n";

  std::cerr << "Initial disjoint grouping of tensors\n";
  for (const auto& set : disjoint_val_sets.disjointSets()) {
    std::cerr << "\t{";
    for (auto tv : *set) {
      std::cerr << " T" << tv->name();
    }
    std::cerr << "}\n";
  }

  // Include outputs
  for (Expr* resize_op : resize_ops) {
    auto resize_out = resize_op->output(0)->as<TensorView>();
    auto output_dep_vals =
        DependencyCheck::getAllValsBetween({resize_out}, fusion->outputs());
    for (auto tv : ir_utils::filterByType<TensorView>(output_dep_vals)) {
      disjoint_val_sets.mapEntries(resize_out, tv);
    }
  }

  // Output dep vals should also be disjointly grouped, so the number
  // of groups should not change
  NVF_ERROR(
      num_disjoint_resize_groups == disjoint_val_sets.size(),
      "Expected number of groups: ",
      num_disjoint_resize_groups,
      ". Actual: ",
      disjoint_val_sets.size());

  // There can still be tensors that are not producers nor consumers
  // of resize ops. They should be fine with any of the groups.
  // All of them should now be privatized.

  auto first_group_tv = resize_ops.at(0)->output(0)->as<TensorView>();

  for (auto tv : all_tvs) {
    if (tv->isFusionInput() || disjoint_val_sets.mappingExists(tv)) {
      continue;
    }

    std::cerr << "Remaining tv: " << tv->toString()
              << ". Put into the group of " << first_group_tv->toString()
              << "\n";

    auto dep_outputs = DependencyCheck::getAllOutputsOf({tv});
    NVF_ERROR(!dep_outputs.empty());

    TensorView* first_dep_output = (*(dep_outputs.begin()))->as<TensorView>();
    bool added_to_group = false;
    for (const auto& disjoint_set : disjoint_val_sets.disjointSets()) {
      if (!disjoint_set->has(first_dep_output)) {
        continue;
      }

      // Make sure all outputs are in the same set
      for (const auto dep_output : dep_outputs) {
        NVF_ERROR(disjoint_set->has(dep_output->as<TensorView>()));
      }

      disjoint_val_sets.mapEntries(tv, disjoint_set->front());
      added_to_group = true;
      break;
    }

    // Could not find any group to join
    NVF_ERROR(
        added_to_group, "Could not find any group to add ", tv->toString());
  }

  NVF_ERROR(
      num_disjoint_resize_groups == disjoint_val_sets.size(),
      "Expected number of groups: ",
      num_disjoint_resize_groups,
      ". Actual: ",
      disjoint_val_sets.size());

  std::cerr << "TV disjoint groups: " << disjoint_val_sets.size() << "\n";

  std::vector<std::pair<TensorView*, std::vector<TensorView*>>> ref_list;

  // Pick a reference in each disjoint set
  for (const auto& disjoint_set : disjoint_val_sets.disjointSets()) {
    TensorView* ref_tv = nullptr;
    // TensorView* input_tv = nullptr;
    std::unordered_set<TensorView*> resize_op_outputs;
#if 0    
    for (TensorView* tv : *disjoint_set) {
      // All of the slice/pad/cat output tensors should have the same
      // loop domain. Any of them can be equally used as the reference
      // for this group.
      // Update: But propagation could still fail due to the resize
      // cyclic mapping. Don't use resize outputs as reference for
      // now.

      if (auto def = tv->definition();
          def != nullptr && def->isOneOf<SliceOp, PadOp>()) {
        ref_tv = def->output(0)->as<TensorView>();
        break;
      }

      if (auto def = tv->definition(); std::any_of(
              def->inputs().begin(), def->inputs().end(), [](Val* input) {
                return input->isA<TensorView>() && input->isFusionInput();
              })) {
        if (input_tv == nullptr ||
            (input_tv->domain()->noBroadcasts().size() <
             tv->domain()->noBroadcasts().size())) {
          input_tv = tv;
        }
      }
    }
#endif

    for (TensorView* tv : *disjoint_set) {
      if (auto def = tv->definition();
          def != nullptr && def->isOneOf<SliceOp, PadOp>()) {
        resize_op_outputs.insert(def->output(0)->as<TensorView>());
      }
    }

    for (TensorView* tv : *disjoint_set) {
      if (!tv->isFusionOutput()) {
        continue;
      }

      // Ref if all resize_outputs have a dependency with this output
      auto all_dep_vals = DependencyCheck::getAllValsBetween(
          {fusion->inputs().begin(), fusion->inputs().end()}, {tv});
      bool all_resize_out_dependent = true;
      for (auto resize_out : resize_op_outputs) {
        auto it =
            std::find(all_dep_vals.begin(), all_dep_vals.end(), resize_out);
        if (it == all_dep_vals.end()) {
          std::cerr << "Not a dependency: " << resize_out->toString() << " of "
                    << tv->toString() << "\n";
          all_resize_out_dependent = false;
          break;
        }
      }

      if (!all_resize_out_dependent) {
        continue;
      }

      ref_tv = tv;
    }

    if (ref_tv) {
      std::cerr << "Reference: " << ref_tv->toString() << "\n";

      ref_list.emplace_back(ref_tv, std::vector<TensorView*>{});
      auto& member_list = ref_list.back().second;
      for (auto tv : all_tvs) {
        if (disjoint_set->has(tv)) {
          member_list.push_back(tv);
        }
      }

      continue;
    }

    NVF_THROW(
        "No reference found for ", toDelimitedString(disjoint_set->vector()));
  }

  std::cerr << "Disjoint grouping of tensors with representatives:\n";
  for (const auto& [ref, set] : ref_list) {
    std::cerr << "\tRepresentative: " << ref->toString() << "\n"
              << "\t{";
    for (auto tv : set) {
      std::cerr << " T" << tv->name();
    }
    std::cerr << "}\n";
  }

  return ref_list;
}

} // namespace

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  DebugStreamGuard dsg(std::cerr);

  FusionGuard fg(fusion);

  std::cerr << "ResizeScheduler::schedule\n";

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);

  fusion->printMath();

  // Privatize all first
  // TODO: don't privatize if unique

  std::unordered_map<TensorView*, TensorView*> tv_uses_to_privatize;

  for (auto expr : fusion->exprs()) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    auto producer_tv = expr->input(0)->as<TensorView>();
    if (producer_tv->isFusionInput()) {
      continue;
    }

    auto private_copy = RecomputeTv::recompute(producer_tv);

    std::cerr << "Replacing " << producer_tv->toString() << " with "
              << private_copy->toString() << "\n";
    auto updated_op =
        ir_utils::replaceValInExprInputs(expr, producer_tv, private_copy);

    std::cerr << "New op: " << updated_op->toString();

    auto all_output_dep_vals = DependencyCheck::getAllValsBetween(
        {private_copy}, {fusion->outputs().begin(), fusion->outputs().end()});

    for (auto output_dep_tv :
         ir_utils::filterByType<TensorView>(all_output_dep_vals)) {
      if (output_dep_tv == private_copy) {
        continue;
      }

      auto def = output_dep_tv->definition();
      NVF_ERROR(def != nullptr);
      if (def->inputs().size() <= 1) {
        continue;
      }

      for (const auto i : c10::irange(def->inputs().size())) {
        auto input_of_def = dynamic_cast<TensorView*>(def->input(i));
        if (input_of_def == nullptr ||
            std::find(
                all_output_dep_vals.begin(),
                all_output_dep_vals.end(),
                input_of_def) != all_output_dep_vals.end()) {
          continue;
        }

        // Privatize this input too but don't do this while looping
        // over fusion->exprs()
#if 0
        auto private_copy_of_input = RecomputeTv::recompute(input_of_def);
        auto updated_def = ir_utils::replaceValInExprInputs(
            def, input_of_def, private_copy_of_input);
        std::cerr << "Privatized non-resized input: "
                  << private_copy_of_input->toString() << " in "
                  << updated_def->toString();
        def = updated_def;
#endif
        tv_uses_to_privatize.emplace(output_dep_tv, input_of_def);
      }
    }
  }

  for (const auto& [consumer, producer] : tv_uses_to_privatize) {
    auto def = consumer->definition();
    NVF_ERROR(
        std::find(def->inputs().begin(), def->inputs().end(), producer) !=
        def->inputs().end());

    auto private_copy_of_producer = RecomputeTv::recompute(producer);
    auto updated_def = ir_utils::replaceValInExprInputs(
        def, producer, private_copy_of_producer);
    std::cerr << "Privatized non-resized input: "
              << private_copy_of_producer->toString() << " in "
              << updated_def->toString();
  }

  // Having squeezed slices uniformly seems to make things
  // simpler. Part of the reason is the reshape propagation, which
  // would remove broadcast IDs. While it shouldn't matter, losing
  // broadcast IDs makes it more complicated to enable the indexing WAR
  // for resize. Overall, enabling this seems to be the most reasonable
  // ATM.
  scheduler_tools::propagateSqueezedSliceToOutputs(fusion);
  std::cerr << "Squeezed slice propagated\n";
  fusion->printMath();

  const auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    std::cerr << "Propagating resize tensor op: " << expr->toString();
    scheduler_tools::propagateResizeTensorOpToInputs(expr);
  }

  std::cerr << "After resize propagation\n";

  for (auto tv : fusion->allTvs()) {
    std::cerr << "Scheduled TV (after all prop): " << tv->toString() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    std::cerr << "\tAdditional ids: "
              << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString(4);
    }
  }

  const auto ref_tensors = getReferenceTensors(fusion);

  for (const auto& [ref_tv, tvs_to_schedule] : ref_tensors) {
    std::cerr << "Reference: " << ref_tv->toString() << "\n";
    std::cerr << "Tvs to schedule: " << toDelimitedString(tvs_to_schedule)
              << "\n";

    ref_tv->flatten();
    ref_tv->split(0, 128);
    ref_tv->split(0, 1 << 14);
    ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
    ref_tv->axis(-2)->parallelize(ParallelType::BIDx);

    std::cerr << "Scheduled reference:\n";
    ref_tv->printTransforms();

    scheduler_tools::scheduleLoopDomainsLike(
        tvs_to_schedule, ref_tv->getLoopDomain());
  }

  std::cerr << "All done\n";
  fusion->printMath();
  for (auto tv : fusion->allTvs()) {
    std::cerr << "Final scheduled T" << tv->name() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    std::cerr << "\tAdditional ids: "
              << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString(4);
    }
  }

  inlineMost();

  fusion->printMath();

  return;
}

} // namespace nvfuser
