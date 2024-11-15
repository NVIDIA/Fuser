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

int64_t ResizeScheduler::getVersion() const {
  int64_t ver = 1;

  if (auto ver_env = getenv("RESIZE_SCHEDULER_VERSION")) {
    ver = std::atoi(ver_env);
  }

  return ver;
}

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
  switch (getVersion()) {
    case 1:
      return canScheduleCompileTimeV1(fusion);
      break;
    case 2:
      return canScheduleCompileTimeV2(fusion);
    default:
      NVF_THROW("invalid scheduler version");
  }
}

namespace {
TensorView* getReference() {
  return nullptr;
}
} // namespace

bool ResizeScheduler::canScheduleCompileTimeV1(Fusion* fusion) {
  std::cerr << "ResizeScheduler::canScheduleCompileTimeV1\n";

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

  return true;
}

bool ResizeScheduler::canScheduleCompileTimeV2(Fusion* fusion) {
  std::cerr << "ResizeScheduler::canScheduleCompileTimeV2\n";
  if (!canScheduleCompileTimeV1(fusion)) {
    return false;
  }

  // Add more conditions to check
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
TensorView* getAllocationReferenceTensor(Fusion* fusion) {
  TensorView* reference = nullptr;
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (reference == nullptr) {
      reference = input_tv;
      continue;
    }

    if (input_tv->getLogicalDomain().size() >
        reference->getLogicalDomain().size()) {
      reference = input_tv;
      continue;
    }

    if (TensorDomain::noBroadcasts(input_tv->getLogicalDomain()).size() >
        TensorDomain::noBroadcasts(reference->getLogicalDomain()).size()) {
      reference = input_tv;
      continue;
    }
  }

  std::cerr << "Allocation reference TV: " << reference->toString()
            << ", allocation: "
            << toDelimitedString(reference->getMaybeAllocationDomain()) << "\n";
  return reference;
}

TensorView* getReferenceTensor(Fusion* fusion, TensorView* largest_out) {
  TensorView* ref = nullptr;
  for (auto expr : fusion->exprs()) {
    auto cat = dynamic_cast<CatOp*>(expr);
    if (cat == nullptr) {
      continue;
    }

    auto cat_output = cat->output(0)->as<TensorView>();

    ref = cat_output;
    break;
  }

  if (ref == nullptr) {
    // ref = getAllocationReferenceTensor(fusion);
    ref = largest_out;
  }

  std::cerr << "Reference TV: " << ref->toString() << ", allocation: "
            << toDelimitedString(ref->getMaybeAllocationDomain()) << "\n";
  return ref;
}

std::vector<std::pair<TensorView*, std::vector<TensorView*>>>
getReferenceTensors(Fusion* fusion, TensorView* largest_out) {
  std::vector<TensorView*> ref_candidates;

  const auto all_tvs = fusion->allTvs();

  for (auto expr : fusion->exprs()) {
    auto cat = dynamic_cast<CatOp*>(expr);
    if (cat == nullptr) {
      continue;
    }

    auto cat_output = cat->output(0)->as<TensorView>();

    ref_candidates.emplace_back(cat_output);
  }

  if (ref_candidates.empty()) {
    // ref_candidates.emplace_back(largest_out,
    // std::vector<TensorView*>{});
    std::cerr << "getReferenceTensors: Using largest out: "
              << largest_out->toString() << "\n";
    return {std::make_pair(largest_out, all_tvs)};
  }

  if (ref_candidates.size() == 1) {
    std::cerr << "Unique reference: " << ref_candidates[0]->toString();
    return {std::make_pair(ref_candidates[0], all_tvs)};
  }

  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  auto can_schedule = [&graph](TensorView* ref, TensorView* tv) -> bool {
    ValGroups ref_groups = graph.toGroups(ref->getLoopDomain());

    // ValGroups tv_groups = graph.toGroups(tv->getLogicalDomain());
    ValGroups tv_groups = graph.toGroups(tv->getLoopDomain());

    auto path_from_ref = ValGraphBFS::getExprsBetween(
        graph, ref_groups, tv_groups, /*require_all_to_visited=*/false);

    auto path_outputs = getOutputsOfExprPath(graph, path_from_ref);
    NVF_ERROR(path_outputs.size() <= tv_groups.size());

    if (path_outputs.size() < tv_groups.size()) {
      // something is unreachable
      for (const auto& tv_group : tv_groups) {
        if (ref_groups.has(tv_group) || path_outputs.has(tv_group)) {
          continue;
        }

        // Unreachable. If it's a broadcast, ignore it. Otherwise,
        // this tensor cannot be scheduled by this reference
        if (tv_group->front()->as<IterDomain>()->isBroadcast()) {
          continue;
        }

        // tv_group is unreachable. Give up
        std::cerr << "Unreachable tv group: " << nvfuser::toString(tv_group)
                  << " of " << tv->toString() << "\n";
        return false;
      }
    }

    // if the path involves resize, don't consider a valid refernce
    // for this tensor as resize should not be propagated
    for (const auto& [expr, dir] : path_from_ref) {
      if (expr->front()->isA<Resize>()) {
        // resize found
        std::cerr << "Resize found: " << expr->front()->toString() << " of "
                  << tv->toString() << "\n";
        return false;
      }
    }

    return true;
  };

  std::unordered_map<TensorView*, std::unordered_set<TensorView*>> grouping_map;
  std::unordered_map<TensorView*, TensorView*> tv_to_ref;

  // Check duplicates and make sure completeness
  for (auto tv : all_tvs) {
    // Don't care fusion inputs
    if (tv->isFusionInput()) {
      // Mark it as grouped for convenience
      tv_to_ref.emplace(tv, nullptr);
      continue;
    }

    // Check if this tv itself is a ref candidate
    if (auto ref_candidates_it =
            std::find(ref_candidates.begin(), ref_candidates.end(), tv);
        ref_candidates_it != ref_candidates.end()) {
      grouping_map[tv].insert(tv);
      tv_to_ref.emplace(tv, tv);
      continue;
    }

    std::vector<bool> match_with_refs(false, ref_candidates.size());
    int64_t num_matches = 0;
    TensorView* matched_ref = nullptr;
    for (const auto ref_candidate : ref_candidates) {
      auto b = can_schedule(ref_candidate, tv);
      if (b) {
        ++num_matches;
        matched_ref = ref_candidate;
      }
      match_with_refs.push_back(b);
    }

    NVF_ERROR(num_matches != 0, "Uncaptured tensor: ", tv->toString());

    // If multiple refs are candidates, don't group it at this time
    if (num_matches == 1) {
      grouping_map[matched_ref].insert(tv);
      tv_to_ref.emplace(tv, matched_ref);
    }
  }

  // Group the remaining tensors based on their producers and
  // consumers. If any of them is already grouped and the group is
  // eligible, prefer that group for the tensro
  bool changed = true;
  while (changed) {
    changed = false;

    for (auto tv : all_tvs) {
      if (tv_to_ref.count(tv)) {
        continue;
      }

      for (auto producer_tv : ir_utils::producerTvsOf(tv)) {
        auto ref_it = tv_to_ref.find(producer_tv);
        if (ref_it != tv_to_ref.end()) {
          auto producer_ref = ref_it->second;
          // producer_ref may be nullptr (when it's fusion input)
          if (producer_ref == nullptr) {
            continue;
          }
          if (!can_schedule(producer_ref, tv)) {
            continue;
          }

          grouping_map[producer_ref].insert(tv);
          tv_to_ref.emplace(tv, producer_ref);
          changed = true;
          break;
        }
      }

      if (changed) {
        continue;
      }

      for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
        auto ref_it = tv_to_ref.find(consumer_tv);
        if (ref_it != tv_to_ref.end()) {
          auto consumer_ref = ref_it->second;
          if (!can_schedule(consumer_ref, tv)) {
            continue;
          }

          grouping_map[consumer_ref].insert(tv);
          tv_to_ref.emplace(tv, consumer_ref);
          changed = true;
          break;
        }
      }
    }
  }

  NVF_ERROR(tv_to_ref.size() == all_tvs.size());

  // Create a sorted grouping list
  std::vector<std::pair<TensorView*, std::vector<TensorView*>>> ref_list;
  for (const auto ref : ref_candidates) {
    auto it = grouping_map.find(ref);
    if (it == grouping_map.end()) {
      // This ref wasn't used at all
      continue;
    }

    const auto& member_set = it->second;

    ref_list.emplace_back(ref, std::vector<TensorView*>{});
    auto& member_list = ref_list.back().second;
    for (auto tv : all_tvs) {
      if (member_set.count(tv)) {
        member_list.push_back(tv);
      }
    }
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

// Copied from pointwise.cpp.
class DomainMap : public pointwise_utils::DomainMap {
 public:
  using pointwise_utils::DomainMap::DomainMap;

  // The pointwise scheduler heuristics requires a minimum number of axes.
  // The output reference tensor should respect this requirement.
  TensorView* findReferenceTensorView(int64_t minimum_num_axes = 0) const {
    TensorView* result = nullptr;
    int64_t max_dims = -1;
    for (auto output_tv :
         ir_utils::filterByType<TensorView>(fusion_->outputs())) {
      std::cerr << "findRef. " << output_tv->toString()
                << ": isValidReference: " << isValidReference(output_tv)
                << ", hasMinimum: "
                << hasMinimumSize(output_tv, minimum_num_axes)
                << ", !isInput:" << !output_tv->isFusionInput() << "\n";
      if (isValidReference(output_tv) &&
          hasMinimumSize(output_tv, minimum_num_axes) &&
          !output_tv->isFusionInput()) {
        int64_t n_dims = pointwise_utils::nRootDims(output_tv);
        if (n_dims > max_dims) {
          result = output_tv;
          max_dims = n_dims;
        }
      }
    }
    return result;
  }

 private:
  bool hasMinimumSize(TensorView* tv, int64_t num_axes) const {
    NVF_ERROR(tv != nullptr);
    return (num_axes == 0 || (int64_t)tv->getLogicalDomain().size() > num_axes);
  }
};

TensorView* getReferenceTensorView(Fusion* fusion) {
  FusionGuard fg(fusion);
  DomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensorView();
  return reference_tv;
}

} // namespace

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  switch (getVersion()) {
    case 1:
      scheduleV1(fusion, params);
      break;
    case 2:
      scheduleV2(fusion, params);
      break;
    default:
      NVF_THROW("invalid scheduler version");
  }
}

void ResizeScheduler::scheduleV1(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  std::cerr << "ResizeScheduler::scheduleV1\n";

  DebugStreamGuard dsg(std::cerr);

  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  std::cerr << "schedulePointwise\n";
  fusion->printMath();

  // Cache inputs
  // auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  // auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  // Reference for ordering
  TensorView* reference_order_tv = nullptr;
  {
    // TODO: Propagate slice first. DO it manually for now. Or use
    // scheduleLoopDomain in a forward fashion. It should be possible
    // to reverse the setting.

    std::cerr << "Before resize scheduling\n";
    fusion->printMath();

    if (!getenv("PROPAGATE_SLICE_TO_OUTPUTS")) {
      scheduler_tools::propagateSliceToInputs(fusion);

      scheduler_tools::propagateCatToInputs(fusion);

      // Need to propagate to outputs if squeezed
      scheduler_tools::propagateSqueezedSliceToOutputs(fusion);
    } else {
      scheduler_tools::propagateCatToInputs(fusion);

      std::cerr << "After cat prop\n";

      fusion->printMath();

      for (auto tv : fusion->allTvs()) {
        std::cerr << "Scheduled TV (after cat prop): " << tv->toString()
                  << "\n";
        if (tv->hasRoot()) {
          std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain())
                    << "\n";
        }
        std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
                  << "\n";
        std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain())
                  << "\n";
        std::cerr << "\tAdditional ids: "
                  << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
        for (auto expr : tv->domain()->allExprs()) {
          std::cerr << expr->toString(4);
        }
      }

      scheduler_tools::propagateSliceToOutputs(fusion);
    }

    for (auto tv : fusion->allTvs()) {
      std::cerr << "Scheduled TV (after all prop): " << tv->toString() << "\n";
      if (tv->hasRoot()) {
        std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain())
                  << "\n";
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

    reference_order_tv = getAllocationReferenceTensor(fusion);
  }

  std::cerr << "scheduling done\n";
  fusion->printMath();

  for (auto tv : fusion->allTvs()) {
    std::cerr << "Scheduled TV: " << tv->toString() << "\n";
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

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  refineCachePolicy(fusion);

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  int64_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  auto original_reference = getReferenceTensorView(fusion);
  TensorView* reference_tv = getReferenceTensor(fusion, original_reference);
  std::cerr << "Reference: " << reference_tv->toString() << "\n";

  // Make sure reference is ordered properly
  auto reorder_tv = [](TensorView* tv, TensorView* ref) {
    IdModel id_model(tv->fusion(), /*build_models=*/false);
    const auto& graph = id_model.buildExactGraph();
    const auto ordered_domains =
        scheduler_utils::getIterationDomainsOrderedLike(
            graph,
            graph.toGroups(tv->getLoopDomain()),
            graph.toGroups(ref->getMaybeAllocationDomain()));
    std::unordered_map<int64_t, int64_t> old2new;
    for (const auto i : c10::irange(tv->getLoopDomain().size())) {
      const auto& loop_group = graph.toGroup(tv->getLoopDomain().at(i));
      auto it =
          std::find(ordered_domains.begin(), ordered_domains.end(), loop_group);
      NVF_ERROR(it != ordered_domains.end());
      auto new_pos = (int64_t)std::distance(ordered_domains.begin(), it);
      old2new.emplace((int64_t)i, new_pos);
    }

    std::cerr << "Pre-reordering reference: " << tv->toString() << "\n";
    std::cerr << "old2new: ";
    for (const auto& [o, n] : old2new) {
      std::cerr << " " << o << "->" << n;
    }
    std::cerr << "\n";
    tv->reorder(old2new);
    std::cerr << "Reordered reference: " << tv->toString() << "\n";
  };

  fusion->printMath();

  NVF_ERROR(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  // TODO: Do this with a list of all references
  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  for (const auto& [reference_tv, tvs_to_schedule] :
       getReferenceTensors(fusion, original_reference)) {
    reorder_tv(reference_tv, reference_order_tv);
    // Trivial scheduling
    reference_tv->flatten();
    reference_tv->split(0, 128);
    reference_tv->split(0, 1 << 15);
    reference_tv->axis(-1)->parallelize(ParallelType::TIDx);
    reference_tv->axis(-2)->parallelize(ParallelType::BIDx);

    std::cout << "Scheduled reference:\n";
    reference_tv->printTransforms();
    std::cout << std::endl;

    // TODO: Don't try to prop to all tensors
    scheduler_tools::scheduleLoopDomainsLike(
        tvs_to_schedule, reference_tv->getLoopDomain());
  }

  std::cerr << "Reference scheduling propagated\n";
  fusion->printMath();
  fusion->print();

  for (auto tv : fusion->allTvs()) {
    std::cerr << "Final scheduled TV: " << tv->toString() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    std::cerr << "\tAdditional ids: "
              << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString();
    }
  }

  inlineMost();

  // scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);

  std::cerr << "All done\n";
  fusion->printMath();
}

namespace {

std::vector<std::pair<TensorView*, std::vector<TensorView*>>>
getReferenceTensors2(Fusion* fusion) {
  std::vector<TensorView*> ref_candidates;

  const auto all_tvs = fusion->allTvs();

  // Use resize output tensors and fusion outputs as reference
  // candidates

  // TODO: canScheduleCompileTime should do some more checks if
  // disconnected ops should be merged
  for (auto expr : fusion->exprs()) {
    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (expr->isOneOf<SliceOp, PadOp>() || output_tv->isFusionOutput()) {
        ref_candidates.emplace_back(output_tv);
      }
    }
  }

  NVF_ERROR(!ref_candidates.empty())

  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  auto can_schedule = [&graph](TensorView* ref, TensorView* tv) -> bool {
    ValGroups ref_groups = graph.toGroups(ref->getLoopDomain());

    // ValGroups tv_groups = graph.toGroups(tv->getLogicalDomain());
    ValGroups tv_groups = graph.toGroups(tv->getLoopDomain());

    auto path_from_ref = ValGraphBFS::getExprsBetween(
        graph, ref_groups, tv_groups, /*require_all_to_visited=*/false);

    auto path_outputs = getOutputsOfExprPath(graph, path_from_ref);
    NVF_ERROR(path_outputs.size() <= tv_groups.size());

    if (path_outputs.size() < tv_groups.size()) {
      // something is unreachable
      for (const auto& tv_group : tv_groups) {
        if (ref_groups.has(tv_group) || path_outputs.has(tv_group)) {
          continue;
        }

        // Unreachable. If it's a broadcast, ignore it. Otherwise,
        // this tensor cannot be scheduled by this reference
        if (tv_group->front()->as<IterDomain>()->isBroadcast()) {
          continue;
        }

        // tv_group is unreachable. Give up
        std::cerr << "Unreachable tv group: " << nvfuser::toString(tv_group)
                  << " of " << tv->toString() << "\n";
        return false;
      }
    }

    // if the path involves resize, don't consider a valid refernce
    // for this tensor as resize should not be propagated
    for (const auto& [expr, dir] : path_from_ref) {
      if (expr->front()->isA<Resize>()) {
        // resize found
        std::cerr << "Resize found: " << expr->front()->toString() << " of "
                  << tv->toString() << "\n";
        return false;
      }
    }

    return true;
  };

  std::unordered_map<TensorView*, std::unordered_set<TensorView*>> grouping_map;
  std::unordered_map<TensorView*, TensorView*> tv_to_ref;

  // Check duplicates and make sure completeness
  for (auto tv : all_tvs) {
    // Don't care fusion inputs
    if (tv->isFusionInput()) {
      // Mark it as grouped for convenience
      tv_to_ref.emplace(tv, nullptr);
      continue;
    }

    // std::vector<bool> match_with_refs(false, ref_candidates.size());
    // int64_t num_matches = 0;
    // TensorView* matched_ref = nullptr;
    for (const auto ref_candidate : ref_candidates) {
      auto b = tv == ref_candidate || can_schedule(ref_candidate, tv);
      if (b) {
        //++num_matches;
        grouping_map[ref_candidate].insert(tv);
        tv_to_ref.emplace(tv, ref_candidate);
        std::cerr << tv->toString() << " -> Ref: " << ref_candidate->toString()
                  << "\n";
        break;
      }
    }
#if 0
    NVF_ERROR(num_matches != 0, "Uncaptured tensor: ", tv->toString());

    // If multiple refs are candidates, don't group it at this time
    if (num_matches == 1) {
      grouping_map[matched_ref].insert(tv);
      tv_to_ref.emplace(tv, matched_ref);
      std::cerr << tv->toString() << " -> Ref: " << matched_ref->toString() << "\n";
    }
#endif
  }

  std::stringstream ss;
  if (tv_to_ref.size() != all_tvs.size()) {
    std::cerr << "All tvs: " << toDelimitedString(all_tvs) << "\n";
    for (const auto& [tv, ref] : tv_to_ref) {
      NVF_ERROR(tv != nullptr);
      std::cerr << tv->toString() << " -> "
                << ((ref == nullptr) ? std::string("null") : ref->toString())
                << "\n";
    }
    for (auto tv : all_tvs) {
      if (tv_to_ref.find(tv) == tv_to_ref.end()) {
        ss << "Failed to capture: " << tv->toString() << "\n";
      }
    }
  }
  NVF_ERROR(tv_to_ref.size() == all_tvs.size(), ss.str());

  // Create a sorted grouping list
  std::vector<std::pair<TensorView*, std::vector<TensorView*>>> ref_list;
  for (const auto ref : ref_candidates) {
    auto it = grouping_map.find(ref);
    if (it == grouping_map.end()) {
      // This ref wasn't used at all
      continue;
    }

    const auto& member_set = it->second;

    ref_list.emplace_back(ref, std::vector<TensorView*>{});
    auto& member_list = ref_list.back().second;
    for (auto tv : all_tvs) {
      if (member_set.count(tv)) {
        member_list.push_back(tv);
      }
    }
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

std::vector<std::pair<TensorView*, std::vector<TensorView*>>>
getReferenceTensors3(Fusion* fusion) {
  std::vector<TensorView*> ref_candidates;

  const auto all_tvs = fusion->allTvs();

  DisjointSets<TensorView*> disjoint_val_sets;
  for (auto output_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    auto dep_vals = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {output_tv});
    auto disjoint_set_it = disjoint_val_sets.initializeSet(output_tv).first;
    // Don't add inputs. Inputs are not replicated nor scheduled.
    for (auto tv : ir_utils::filterByType<TensorView>(dep_vals)) {
      if (tv->isFusionInput()) {
        continue;
      }
      if (disjoint_val_sets.mappingExists(tv)) {
        disjoint_val_sets.mapEntries(output_tv, tv);
        disjoint_set_it = disjoint_val_sets.find(tv);
      } else {
        disjoint_val_sets.appendToSet(tv, disjoint_set_it->second);
      }
    }
  }

  std::cerr << "TV disjoint groups: " << disjoint_val_sets.size() << "\n";

  std::vector<std::pair<TensorView*, std::vector<TensorView*>>> ref_list;

  // Pick a reference in each disjoint set
  for (const auto& disjoint_set : disjoint_val_sets.disjointSets()) {
    TensorView* ref_tv = nullptr;
    for (TensorView* tv : *disjoint_set) {
      // All of the slice/pad/cat output tensors should have the same
      // loop domain. Any of them can be equally used as the reference
      // for this group
      if (auto def = tv->definition();
          def != nullptr && def->isOneOf<SliceOp, PadOp>()) {
        ref_tv = def->output(0)->as<TensorView>();
        break;
      }
    }

    if (ref_tv) {
      std::cerr << "Reference selected from resize ops: " << ref_tv->toString()
                << "\n";

      ref_list.emplace_back(ref_tv, std::vector<TensorView*>{});
      auto& member_list = ref_list.back().second;
      for (auto tv : all_tvs) {
        if (disjoint_set->has(tv)) {
          member_list.push_back(tv);
        }
      }

      continue;
    }

    // No slice or pad found
    std::cerr << "No slice/pad found for "
              << toDelimitedString(disjoint_set->vector()) << "\n";
    NVF_THROW();
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

void ResizeScheduler::scheduleV2(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  DebugStreamGuard dsg(std::cerr);

  std::cerr << "ResizeScheduler::scheduleV2\n";

  scheduler_utils::clearMemorySpace(fusion);

  fusion->printMath();

  FusionGuard fg(fusion);

  // Privatize all first
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
  }

  fusion->printMath();

  scheduler_tools::propagateSqueezedSliceToOutputs(fusion);

  std::cerr << "Squeezed slice propagated\n";
  fusion->printMath();

  const auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

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

  const auto ref_tensors = getReferenceTensors3(fusion);

  for (const auto& [ref_tv, tvs_to_schedule] : ref_tensors) {
    std::cerr << "Reference: " << ref_tv->toString() << "\n";
    std::cerr << "Tvs to schedule: " << toDelimitedString(tvs_to_schedule)
              << "\n";

    ref_tv->flatten();
    ref_tv->split(0, 128);
    ref_tv->split(0, 1 << 15);
    ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
    ref_tv->axis(-2)->parallelize(ParallelType::BIDx);

    std::cerr << "Scheduled reference:\n";
    ref_tv->printTransforms();

    scheduler_tools::scheduleLoopDomainsLike(
        tvs_to_schedule, ref_tv->getLoopDomain());
  }

  std::cerr << "All done\n";
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

  return;
}

} // namespace nvfuser
