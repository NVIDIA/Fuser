// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <instrumentation.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/resize.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
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

void ResizeScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  std::cerr << "ResizeScheduler::schedule\n";

  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  std::cout << std::endl;
  std::cerr << "schedulePointwise\n";
  fusion->printMath();
  std::cout << std::endl;

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

    std::cout << "Before resize scheduling" << std::endl;
    fusion->printMath();
    std::cout << std::endl;

    if (!getenv("PROPAGATE_SLICE_TO_OUTPUTS")) {
      scheduler_tools::propagateSliceToInputs(fusion);

      scheduler_tools::propagateCatToInputs(fusion);

      // Need to propagate to outputs if squeezed
      scheduler_tools::propagateSqueezedSliceToOutputs(fusion);
    } else {
      scheduler_tools::propagateCatToInputs(fusion);

      std::cout << "After cat prop" << std::endl;

      fusion->printMath();
      std::cout << std::endl;

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

  std::cout << "scheduling done\n";
  fusion->printMath();
  std::cout << std::endl;

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
  std::cout << std::endl;

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

  std::cout << "Reference scheduling propagated\n";
  fusion->printMath();
  fusion->print();
  std::cout << std::endl;

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

  {
    IdModel id_model(fusion);
    std::cout << id_model.idGraph(IdMappingMode::EXACT).toString();
    std::cout << std::endl;
  }

  inlineMost();

  // scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);

  std::cout << "All done\n";
  fusion->printMath();
  std::cout << std::endl;
  
}

} // namespace nvfuser
