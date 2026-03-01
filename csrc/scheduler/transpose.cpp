// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose.h"

#include <ATen/cuda/CUDAContext.h>

#include "instrumentation.h"
#include "scheduler/debug_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/transpose_non_tma.h"
#include "scheduler/transpose_tma.h"
#include "scheduler/transpose_utils.h"
#include "scheduler/utils.h"

namespace nvfuser {

bool TransposeScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("TransposeScheduler::canScheduleCompileTime");
  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  for (auto select : ir_utils::getOpsOfType<SelectOp>(fusion)) {
    auto inner = TensorDomain::noReductions(
        select->input(0)->as<TensorView>()->getMaybeAllocationDomain());
    if (select->getIndexedID() == inner[inner.size() - 1]) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "SelectOp on inner dim is not supported by transpose scheduler yet."
          "In transpose scheduler, we want to leave the select dim alone, "
          "instead of creating a tile for it.");
      return false;
    }
  }
  for (auto idx_sel : ir_utils::getOpsOfType<IndexSelectOp>(fusion)) {
    auto inner = TensorDomain::noReductions(
        idx_sel->input(0)->as<TensorView>()->getMaybeAllocationDomain());
    if (idx_sel->getIndexedID() == inner[inner.size() - 1]) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "IndexSelectOp on inner dim is not supported by transpose scheduler "
          "yet."
          "In transpose scheduler, we want to leave the select dim alone, "
          "instead of creating a tile for it.");
      return false;
    }
  }
  for (auto torch_gather : ir_utils::getOpsOfType<GatherOp>(fusion)) {
    auto inner = TensorDomain::noReductions(
        torch_gather->input(0)->as<TensorView>()->getMaybeAllocationDomain());
    if (torch_gather->getIndexedID() == inner[inner.size() - 1]) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "GatherOp on inner dim is not supported by transpose scheduler yet."
          "In transpose scheduler, we want to leave the select dim alone, "
          "instead of creating a tile for it.");
      return false;
    }
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  if (!hasAtLeastTwoValidGroups(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "cannot find two mismatching inner most dimensions");
    return false;
  }

  return true;
}

namespace {

// TransposeViewPropagator doesn't propagate anything. It simply walks across
// the path of potential propagation checking if there's any incompatible
// propagation that would not be resolved.
struct TransposeViewPropagator : public MaxInfoSpanningTree::Propagator {
  void propagateC2P(TensorView* from, TensorView* to) override {}
  void propagateP2C(TensorView* from, TensorView* to) override {
    // short-cut to skip if we know we are already rejecting the fusion for
    // transpose scheduler
    if (shouldReject()) {
      return;
    }
    // checking to see if propagation would trigger producer to consumer
    // propagation travelling across view op. Note this is a conservative check,
    // since view does NOT necessarily always introduce incoherent transform
    // that would break the propagation.
    auto chain_exprs = StmtSort::getExprsBetween({from}, {to});
    if (!ir_utils::filterByType<ReshapeOp>(chain_exprs).empty()) {
      should_reject = true;
    };
  };
  void propagateSibling(TensorView* from, TensorView* to) override {}
  ~TransposeViewPropagator() override = default;

  bool shouldReject() {
    return should_reject;
  }

  bool should_reject = false;
};

HeuristicDataCacheEntry<HeuristicCompileTime::TransposeDomainMap> getDomainMap(
    HeuristicDataCache* data_cache,
    Fusion* fusion) {
  auto domain_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::TransposeDomainMap>(
          data_cache, [fusion]() {
            return std::make_unique<scheduler_tools::TransposeDomainMap>(
                fusion);
          });
  return domain_map_entry;
}

HeuristicDataCacheEntry<HeuristicCompileTime::InputsOutputsInnerDimGroups>
getInputsOutputsGroups(
    HeuristicDataCache* data_cache,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto grouped_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::InputsOutputsInnerDimGroups>(
      data_cache, [&domain_map]() {
        return std::make_unique<std::vector<std::vector<TensorView*>>>(
            domain_map.groupInputsOutputsByInnerDim());
      });
  auto& grouped_inputs_outputs = grouped_inputs_outputs_entry.get();

  NVF_ERROR(
      grouped_inputs_outputs.size() >= 2,
      "Can not find mismatched inner most dim, should use pointwise "
      "scheduler.");

  return grouped_inputs_outputs_entry;
}

HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensorsForGroups>
getReferenceTensors(
    HeuristicDataCache* data_cache,
    scheduler_tools::TransposeDomainMap& domain_map,
    std::vector<std::vector<TensorView*>>& grouped_inputs_outputs) {
  auto reference_tensors_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensorsForGroups>(
          data_cache, [&domain_map, &grouped_inputs_outputs]() {
            std::vector<TensorView*> data{
                domain_map.findReferenceFor(grouped_inputs_outputs[0]),
                domain_map.findReferenceFor(grouped_inputs_outputs[1])};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  auto& reference_tensors = reference_tensors_entry.get();
  NVF_ERROR(reference_tensors.size() == 2);
  TensorView* reference1 = reference_tensors[0];
  TensorView* reference2 = reference_tensors[1];
  NVF_ERROR(
      reference1 != nullptr, "Unable to find reference tensor for group 1");
  NVF_ERROR(
      reference2 != nullptr, "Unable to find reference tensor for group 2");
  return reference_tensors_entry;
}

std::pair<std::vector<int64_t>, int64_t> getLoopDomainSizes(
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto ref_loop = reference->getLoopDomain();
  std::vector<int64_t> shape_in_ref;
  shape_in_ref.reserve(reference->nDims());
  int64_t n_elems = 1;
  for (auto id : ref_loop) {
    auto concrete_id = domain_map.getComputeAtMap().getConcreteMappedID(
        id, IdMappingMode::EXACT);
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(concrete_id->extent());
    NVF_ERROR(
        inferred_val.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        id->extent()->toInlineString());
    int64_t size = inferred_val.as<int64_t>();
    n_elems *= size;
    shape_in_ref.push_back(size);
  }
  return {shape_in_ref, n_elems};
}

HeuristicDataCacheEntry<HeuristicCompileTime::InnerMostDimInfo>
getInnerMostDimInfoInReference(
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& group_references,
    TensorView* global_reference,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto innermost_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::InnerMostDimInfo>(
          data_cache, [&]() {
            std::vector<int64_t> data;
            data.reserve(group_references.size());
            for (auto ref_tv : group_references) {
              auto inner_most_id = scheduler_utils::innerMostAllocDim(ref_tv);
              auto inner_most_pos_in_global_ref =
                  domain_map.getInnerLeafDim(global_reference, inner_most_id);
              data.push_back(inner_most_pos_in_global_ref);
            }
            return std::make_unique<std::vector<int64_t>>(std::move(data));
          });
  return innermost_info_entry;
}

// If can schedule at runtime, returns empty string, otherwise returns the
// reason why we should not schedule at runtime.
std::string getTransposeRuntimeRejectReason(
    Fusion* fusion,
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  auto domain_map_entry = getDomainMap(data_cache, fusion);
  auto& domain_map = dynamic_cast<scheduler_tools::TransposeDomainMap&>(
      domain_map_entry.get());
  auto grouped_inputs_outputs_entry =
      getInputsOutputsGroups(data_cache, domain_map);
  auto grouped_inputs_outputs = grouped_inputs_outputs_entry.get();
  auto reference_tensors_entry =
      getReferenceTensors(data_cache, domain_map, grouped_inputs_outputs);
  auto reference_tensors = reference_tensors_entry.get();
  TensorView* reference1 = reference_tensors[0];

  auto [shape_in_ref1, n_elems] =
      getLoopDomainSizes(data_cache, runtime_info, reference1, domain_map);

  auto innermost_info_entry = getInnerMostDimInfoInReference(
      data_cache, reference_tensors, reference1, domain_map);
  auto innermost_info = innermost_info_entry.get();
  auto inner_most_pos1_in_ref1 = innermost_info[0];
  auto inner_most_pos2_in_ref1 = innermost_info[1];
  if (inner_most_pos1_in_ref1 < 0 || inner_most_pos2_in_ref1 < 0) {
    return "Transpose scheduler requires exact mapping on inner most dimension "
           "on reference tensor.";
  }

  constexpr size_t default_tile_elements =
      TransposeParams::getDefaultTileSize() *
      TransposeParams::getDefaultTileSize();

  // don't schedule with transpose scheduler if less than a full wave
  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto elements_per_wave = device_multiprocessor_count * default_tile_elements;
  if ((int64_t)elements_per_wave > n_elems) {
    return "Transpose scheduler does not perform well on small problem sizes.";
  }

  auto inner_size1 = shape_in_ref1.at(inner_most_pos1_in_ref1);
  auto inner_size2 = shape_in_ref1.at(inner_most_pos2_in_ref1);

  // For cases like
  //   transpose(T0[1000000000, 2, 2], 1, 2)
  // the pointwise scheduler should provide better performance, because it
  // provides coalesced memory access
  if (inner_size1 * inner_size2 < (int64_t)default_tile_elements) {
    auto inner_elements = inner_size1 * inner_size2;
    for (int64_t i = inner_most_pos2_in_ref1 + 1; i < inner_most_pos1_in_ref1;
         i++) {
      inner_elements *= shape_in_ref1[i];
    }
    // note that the algorithm here is only an approximation because it only
    // checks reference1. In principle, we need to check all inputs and outputs
    // to get an accurate result, but that is too much work. I think checking
    // only reference 1 is fine for now. Below is an example where the
    // approximation here will not work:
    //   T0[10000000, 2, 3] (reference 1)
    //   T1[2, 10000000, 3] input/output
    //   T2[2, 10000000, 3] input/output
    //   T3[2, 10000000, 3] input/output
    //   T4[3, 10000000, 2] input/output
    //   T5[3, 10000000, 2] input/output
    if (inner_elements < (int64_t)default_tile_elements) {
      return "Inner transpose of small dimensions should be scheduled by the "
             "pointwise scheduler because it provides better memory coalescing";
    }
  }

#if !SUPPORT_SPLITTING_INNERMOST_DIM
  if (n_elems / inner_size1 < (int64_t)TransposeParams::getDefaultTileSize() ||
      n_elems / inner_size2 < (int64_t)TransposeParams::getDefaultTileSize()) {
    return "Splitting of inner most dim for the creation of virtual inner most "
           "dim "
           "is disabled due to indexing bug, skipping this case at runtime for "
           "now"
           "See: https://github.com/csarofeen/pytorch/issues/1964";
  }
#endif

  // TODO: ideally we shouldn't have to manually match schedule transformation
  // here. It is hard to maintain consistent code logic.
  if (!scheduler_utils::getViewTVs(fusion).empty()) {
    const auto index_type = runtime_info.getIndexType();
    auto params = std::make_unique<TransposeParams>();
    params->tag = "Transpose heuristics";
    params->cparams.index_type = index_type;
    transpose::utils::maybeBuildVirtualInnerDims(
        params.get(),
        device_multiprocessor_count,
        n_elems,
        shape_in_ref1,
        inner_most_pos1_in_ref1,
        inner_most_pos2_in_ref1);

    // disallow transpose scheduler when we have a combination of:
    // 1. view op; and
    // 2. small transpose transformation
    // See note [Supporting small transpose dimensions]
    if (transpose::utils::hasSmallTransposeDimensions(params)) {
      return "Small transpose dimensions and view op cannot be currently be "
             "handled by transpose scheduler. See: "
             "https://github.com/NVIDIA/Fuser/pull/592";
    }

    // mimic transform propagation
    // NOTE: in the actual transpose scheduler, we are applying cacheBefore and
    // cacheAfter, which I think would mean different propagation is happening
    // than what's done here. So this might not be bullet proof.
    TransposeViewPropagator propagator;

    // global schedule traverse dry-run
    // see `Step 2: global schedule`
    //
    // This is the step where we create virtual innermost dimension and prepare
    // for scheduling tiling on the two groups. Propagation from P2C across view
    // is challenging at this step and could result in propagating incoherent
    // transformation which resulted in assert. Hence our dry run here examines
    // the path for propagation and conservatively rejects fusion that requires
    // propagation in the risky direction.
    //
    // NOTE: there are three traverse called during scheduling. We are only
    // doing dry-run on the first traverse. Since the following twos are only
    // used for scheduling tiling, which is not going to cause issue, since we
    // are only tiling on the merged virtual innermost dimensions.
    MaxLogicalDomainInfoSpanningTree entire_dag(reference1);
    entire_dag.traverse(&propagator);
    if (propagator.shouldReject()) {
      return "transpose scheduler could potentially trigger incoherent "
             "transform propagation";
    }
  }

  return "";
}

} // namespace

bool hasAtLeastTwoValidGroups(Fusion* fusion) {
  return scheduler_tools::TransposeDomainMap::hasAtLeastTwoValidGroups(fusion);
}

bool TransposeScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("TransposeScheduler::canScheduleRunTime");

  auto reason =
      getTransposeRuntimeRejectReason(fusion, data_cache, runtime_info);
  if (!reason.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), reason);
    return false;
  }
  return true;
}

std::unique_ptr<HeuristicParams> TransposeScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("TransposeScheduler::computeHeuristics");

  std::unique_ptr<TransposeParams> tparams = nullptr;

  // Try TMA path first if enabled
  if (isOptionEnabled(EnableOption::TmaTranspose)) {
    tparams = transpose::tma::getTransposeHeuristics(
        fusion, runtime_info, data_cache);
  }

  // Fallback to non-TMA scheduler if TMA is not applicable
  if (tparams == nullptr) {
    tparams = transpose::non_tma::getTransposeHeuristics(
        fusion, runtime_info, data_cache);
  }

  NVF_ERROR(tparams != nullptr);
  return tparams;
}

void TransposeScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("TransposeScheduler::schedule");
  auto tparams = dynamic_cast<const TransposeParams*>(params);
  NVF_ERROR(
      tparams != nullptr,
      "Incorrect parameters sent to TransposeScheduler::schedule",
      params);

  if (tparams->use_tma_load || tparams->use_tma_store) {
    transpose::tma::scheduleTranspose(fusion, tparams);
  } else {
    transpose::non_tma::scheduleTranspose(fusion, tparams);
  }
}

} // namespace nvfuser
