// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

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

// If a fusion is segmented, the segmenter will create fusions whose inputs
// contain reduction IterDomains. These reduction IterDomains on input
// TensorViews does not have any meaning, and should just be left untouched. See
// https://github.com/NVIDIA/Fuser/issues/1659#issuecomment-1907053830
//
// This function checks the inner `n` iterdomain and reorder reduction
// iterdomain to the beginning.
void moveReductionsOut(TensorView* tv, int n) {
  if (!tv->isFusionInput()) {
    return;
  }

  std::unordered_map<int64_t, int64_t> old2new;

  int64_t target = 0;
  for (int64_t i = 0; i < n; i++) {
    if (tv->axis(-1 - i)->isReduction()) {
      old2new[-1 - i] = target++;
    }
  }

  tv->reorder(old2new);
}

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
    if (!ir_utils::filterByType<ViewOp>(chain_exprs).empty()) {
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

bool hasSmallTransposeDimensions(
    const std::unique_ptr<TransposeParams>& params) {
  return !params->split_before_tiling.empty() ||
      !params->dims_merged_with_1.empty() ||
      !params->dims_merged_with_2.empty();
}

// Note: [Supporting small transpose dimensions]
// We prefer to make tiles of size 32x32 if there are enough elements to achieve
// good occupancy, otherwise, we will use tile size 8x8. In both cases, it is
// possible that the inner dimension of group 1 and/or group 2 are smaller than
// the desired tile size. If this happens, part of the threads of a block will
// be wasted, leading to bad performance. To prevent this from happening, if the
// size of the inner-most dim is smaller than the tile size, we merge other
// dimensions with the inner-most dimension to create larger "virtual inner-most
// dimension". The algorithm that we create these virtual inner-most dimensions
// is as follows:
//
// For example, if we have
//   T0[I0{2}, I1{1024*1024}, I2{2}, I3{2}, I4{2}, I5{2}, I6{2}] input
//   T1 = transpose(T0, 4, 6)
// We first try to merge each inner-most dim with the dimensions on its left:
//   T0[I0{2}, I1*I2*I3*I4{1024*1024*8}, I5*I6{4}]
// If there is/are still unsatisfied innermost dim(s) after this step (I5*I6 in
// this example), we find other dims that is not merged yet to satisfy it/them:
//   T0[I0*I5*I6{8}, I1*I2*I3*I4{1024*1024*8}]
// If after merging all the dims, there is still one of them not satisfied, this
// usually means there is one large dim that is consumed by the satisfied one.
// We will split that dim and large dim and and use the splitted ones to satisfy
// both of them:
//   T0[I0*I1o*I5*I6{1024*1024/4*8}, I1i*I2*I3*I4{32}]
void maybeBuildVirtualInnerDims(
    TransposeParams* tparams,
    int64_t device_multiprocessor_count,
    int64_t n_elems,
    const std::vector<int64_t>& shape_in_ref1,
    int64_t inner_most1,
    int64_t inner_most2) {
  int64_t merged_size1 = shape_in_ref1[inner_most1];
  int64_t merged_size2 = shape_in_ref1[inner_most2];

  int64_t actual_tile_size1 =
      std::min<int64_t>(merged_size1, (int64_t)tparams->tile_size1);
  int64_t actual_tile_size2 =
      std::min<int64_t>(merged_size2, (int64_t)tparams->tile_size2);
  int64_t wave_elements =
      device_multiprocessor_count * actual_tile_size1 * actual_tile_size2;

  if (wave_elements >= n_elems) {
    // if one full wave can handle all elements, don't create virtual inner dims
    return;
  }

  // merge inner_most1 and inner_most2 left until we are done or we can no
  // longer do so
  int64_t dim = inner_most1 - 1;
  while (dim >= 0 && dim != inner_most2 &&
         merged_size1 < (int64_t)tparams->tile_size1) {
    tparams->dims_merged_with_1.push_back(dim);
    merged_size1 *= shape_in_ref1[dim];
    dim--;
  }
  dim = inner_most2 - 1;
  while (dim >= 0 && dim != inner_most1 &&
         merged_size2 < (int64_t)tparams->tile_size2) {
    tparams->dims_merged_with_2.push_back(dim);
    merged_size2 *= shape_in_ref1[dim];
    dim--;
  }
  // If any of them are unsatisfied, then find other dims to merge
  std::unordered_set<int64_t> unavailable_dims;
  unavailable_dims.reserve(
      2 + tparams->dims_merged_with_1.size() +
      tparams->dims_merged_with_2.size());
  unavailable_dims.insert(inner_most1);
  unavailable_dims.insert(inner_most2);
  for (auto i : tparams->dims_merged_with_1) {
    unavailable_dims.insert((int64_t)i);
  }
  for (auto i : tparams->dims_merged_with_2) {
    unavailable_dims.insert((int64_t)i);
  }
  dim = (int64_t)shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size1 < (int64_t)tparams->tile_size1) {
    if (unavailable_dims.count(dim) == 0) {
      tparams->dims_merged_with_1.push_back(dim);
      merged_size1 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  dim = (int64_t)shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size2 < (int64_t)tparams->tile_size2) {
    if (unavailable_dims.count(dim) == 0) {
      tparams->dims_merged_with_2.push_back(dim);
      merged_size2 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  // If both are satisfied, then we are done. If neither are satisfied, then it
  // is impossible to satisfy both of them, also done.
  if ((merged_size1 < (int64_t)tparams->tile_size1) ==
      (merged_size2 < (int64_t)tparams->tile_size2)) {
    return; // no need to split
  }
  // If one of them are not satisfied, there might be two cases:
  // 1. The satisfied one just merged in a large dim. If this is the case, We
  //    split this large dim, so that now we have two available dims to satisfy
  //    both virtual innermost dim.
  // 2. The satisfied one did not merge in anything. For example,
  //    T0[I0{1024*1024}, I1{2}]
  //    If this is the case, this means that we need to split the large
  //    inner-most dimension to satisfy the small innermost dimension
  int64_t large_dim = -1;
  int64_t split_factor = -1;
  bool split_inner_most = false;
  if (merged_size1 < (int64_t)tparams->tile_size1) {
    if (tparams->dims_merged_with_2.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most2;
      split_factor = tparams->tile_size2;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = (int64_t)tparams->dims_merged_with_2.back();
      auto prev_merged_size2 = merged_size2 / shape_in_ref1[large_dim];
      split_factor = ceilDiv((int64_t)tparams->tile_size2, prev_merged_size2);
    }
  } else {
    if (tparams->dims_merged_with_1.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most1;
      split_factor = tparams->tile_size1;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = (int64_t)tparams->dims_merged_with_1.back();
      auto prev_merged_size1 = merged_size1 / shape_in_ref1[large_dim];
      split_factor = ceilDiv((int64_t)tparams->tile_size1, prev_merged_size1);
    }
  }
  tparams->split_before_tiling.emplace_back(large_dim, split_factor);
  // adjust all dims to after-split
  for (auto& i : tparams->dims_merged_with_1) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  for (auto& i : tparams->dims_merged_with_2) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  // Give the split-out dim to the unsatisfied one, so that both are satisfied.
  if (merged_size1 < (int64_t)tparams->tile_size1) {
    if (!split_inner_most) {
      tparams->dims_merged_with_2.pop_back();
      tparams->dims_merged_with_2.push_back(large_dim + 1);
    }
    tparams->dims_merged_with_1.push_back(large_dim);
  } else {
    if (!split_inner_most) {
      tparams->dims_merged_with_1.pop_back();
      tparams->dims_merged_with_1.push_back(large_dim + 1);
    }
    tparams->dims_merged_with_2.push_back(large_dim);
  }
}

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
    maybeBuildVirtualInnerDims(
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
    if (hasSmallTransposeDimensions(params)) {
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

std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  const auto index_type = runtime_info.getIndexType();

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
  TensorView* reference2 = reference_tensors[1];
  auto [shape_in_ref1, n_elems] =
      getLoopDomainSizes(data_cache, runtime_info, reference1, domain_map);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto innermost_info_entry = getInnerMostDimInfoInReference(
      data_cache, reference_tensors, reference1, domain_map);
  auto innermost_info = innermost_info_entry.get();

  auto inner_most_pos1_in_ref1 = innermost_info[0];
  auto inner_most_pos2_in_ref1 = innermost_info[1];
  // No exact innermost loop dimension mapping on referenc1. cannot schedule
  if (inner_most_pos1_in_ref1 < 0 || inner_most_pos2_in_ref1 < 0) {
    NVF_THROW(
        "Transpose scheduler requires exact mapping on inner most dimension on "
        "reference tensor.");
  }

  auto tparams = std::make_unique<TransposeParams>();
  tparams->tag = "Transpose heuristics";
  tparams->cparams.index_type = index_type;

  // Expand inner-most dims to virtual inner-most dims so that the inner-most
  // dims has at least tile_size elements
  // See note [Supporting small transpose dimensions]
  maybeBuildVirtualInnerDims(
      tparams.get(),
      device_multiprocessor_count,
      n_elems,
      shape_in_ref1,
      inner_most_pos1_in_ref1,
      inner_most_pos2_in_ref1);

  NVF_ERROR(
      !hasSmallTransposeDimensions(tparams) ||
          scheduler_utils::getViewTVs(fusion).empty(),
      "combination of view op with small transpose dimensions are not "
      "supported by transpose scheduler");

  // Note [vectorization and unroll of input and output]
  //
  // The choice of vectorization size, block size and tile sizes needs to be
  // consistent with each other. Consider the following:
  //
  // The number of threads in one block is
  //   num_threads = blockDim.x * blockDim.y
  // and the number of elements per each tile is
  //   num_elems_per_tile = tparams->tile_size1 * tparams->tile_size2
  // So each thread needs to process
  //   num_elems_per_thread = num_elems_per_tile / num_threads
  // elements. That is, once the tile sizes and block size are determined, the
  // `num_elems_per_thread` is determined, regardless of vectorizability of
  // input/output tensors.
  //
  // To make the selection of tile sizes othogonal to vectorizability, we
  // support having both vectorization and unrolling in the same tensor. For
  // example, if we have num_elems_per_tile == 1024 and num_threads = 256, then
  // we have num_elems_per_thread being 4. And if we have vector size 2, then we
  // will do unroll 2 * vectorize 2 at the same tensor.
  //
  // Also, since the inner most dim of different groups are not the same, it is
  // natural to consider their vectorizability separately and allow them to have
  // different vectorize/unroll sizes.

  constexpr int64_t kSixteen = 16; // clang tidy

  int64_t max_io_dtype_size = 1;
  int64_t n_io_tensors = 0;
  auto scan_max_dtype_size = [&](const auto& vals) {
    for (auto inp : ir_utils::filterByType<TensorView>(vals)) {
      max_io_dtype_size = std::max(
          max_io_dtype_size,
          dataTypeSizeByte(inp->getDataType().value(), index_type));
      n_io_tensors++;
    }
  };
  scan_max_dtype_size(fusion->inputs());
  scan_max_dtype_size(fusion->outputs());

  auto max_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      kSixteen / max_io_dtype_size,
      // Reduce max unrolling factor if we have many inputs/outputs to unroll
      // as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)grouped_inputs_outputs[0].size() +
               (int64_t)grouped_inputs_outputs[1].size()) >>
           2),
          (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  auto max_unroll_factor_occupancy = ceilDiv(
      n_elems,
      device_multiprocessor_count * (int64_t)tparams->tile_size1 *
          (int64_t)tparams->tile_size2);
  max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_occupancy);

  // Don't unroll at the cost of getting a full warp, useful for the case where
  // tile sizes are small
  auto max_unroll_factor_block =
      ceilDiv((int64_t)tparams->tile_size1 * (int64_t)tparams->tile_size2, 32l);
  max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_block);

  // Note: [Computing Vectorization Width for Transpose]
  //
  // With support of small transpose dimension (see Note [Supporting small
  // transpose dimensions]), we need to consider the transformation applied on
  // our tile to compute the safe vectorization width. e.g. For a simple
  // transpose:
  //    (i0, i1, i2, i3, i4) size (2, 3, 65536, 4, 7)
  // -> (i3, i4, i2, i0, i1)
  //
  // transpose scheduler would apply transformation and tile on virtual
  // innermost dimensions. ( (i0*i1*i2/2), (2*i3*i4)/32) So we need to use the
  // size of the virtual innermost dimensions to compute our vectorization
  // width. In the example above, we are looking at 2*3*65536/2, 2*4*7.
  //
  // Currently there's limitation on our iter domain mapping. Since we can only
  // do it on rfactor/root domain, we cannot map across `split` domains. So the
  // example above will only have vectorization size of 2 and 4 repsectively for
  // the merge virtual innermost dimensions, rather than considering the split
  // and merged i2/2 & 2.
  //
  // TODO: We use ContiguousInnerDimensionsMapper to compute the size of virtual
  // innermost dimension. The analysis right now is limited on logical domain
  // only, so we can't actually map the `split` iter domains, which limits the
  // vectorization width we can apply. We need to fix that.
  // TODO 2: Small transpose dimensions transformation should also consider the
  // vectorization impact. i.e. when split_before_tiling, we should try to split
  // on a factor that allows vectorization.
  {
    // duplicating reference1's TensorDomain, since the transformations applied
    // is not persistent and only needed for us to compute vectorization width.
    TensorDomain* cloned_1_td =
        IrBuilder::create<TensorDomain>(reference1->domain());
    // Adding a domain_guard so we can transform reference1
    ir_utils::TVDomainGuard domain_guard(reference1, cloned_1_td);

    std::vector<int64_t> to_update(
        {inner_most_pos1_in_ref1, inner_most_pos2_in_ref1});
    // we only apply split here, since we want to merge split dimensions, we can
    // simply map those merged domains via ContiguousInnerDimensionsMapper
    scheduler_utils::splitDims(
        reference1, tparams->split_before_tiling, to_update);
    inner_most_pos1_in_ref1 = to_update[0];
    inner_most_pos2_in_ref1 = to_update[1];

    tparams->vectorize_factor1 =
        vectorize_helper::getVectorizationFactorTransposeGroup(
            runtime_info,
            reference1,
            inner_most_pos1_in_ref1,
            tparams->dims_merged_with_1,
            grouped_inputs_outputs[0],
            max_unroll_factor);

    // TODO: Since group2 only has global->shared and shared->global set op, we
    // can have fine-grained control of unroll/vectorization at per tensor
    // level. We should not be using a single global vectorize factor for the
    // entire group 2
    tparams->vectorize_factor2 =
        vectorize_helper::getVectorizationFactorTransposeGroup(
            runtime_info,
            reference1,
            inner_most_pos2_in_ref1,
            tparams->dims_merged_with_2,
            grouped_inputs_outputs[1],
            max_unroll_factor);
  }

  tparams->lparams.bind(tparams->getThreadsPerBlock(), ParallelType::TIDx);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Transpose Stats ========\n"
            << "inputs: " << ir_utils::toString(fusion->inputs()) << "\n"
            << "outputs: " << ir_utils::toString(fusion->outputs()) << "\n"
            << "shape: " << shape_in_ref1 << "\n"
            << "num_elems: " << n_elems << "\n"
            << "n_io_tensors: " << n_io_tensors << "\n"
            << "max_io_dtype_size: " << max_io_dtype_size << "\n"
            << "group 1: " << ir_utils::toString(grouped_inputs_outputs[0])
            << "\n"
            << "reference1: " << reference1->toString() << "\n"
            << "inner_most_id1 position: " << inner_most_pos1_in_ref1
            << " (in reference 1)\n"
            << "group 2: " << ir_utils::toString(grouped_inputs_outputs[1])
            << "\n"
            << "reference2: " << reference2->toString() << "\n"
            << "inner_most_id2 position: " << inner_most_pos2_in_ref1
            << " (in reference 1)" << std::endl;
    if (hasSmallTransposeDimensions(tparams)) {
      debug() << "small transposed dim, needs virtual inner-most dim"
              << std::endl;
    }
    debug() << std::endl;
    debug() << tparams->toString() << std::endl;
  }

  return tparams;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  NVF_ERROR(
      !ir_utils::hasAnyReductionOps(fusion),
      "This scheduler only handles pointwise ops.");

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

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
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  NVF_ERROR(grouped_inputs_outputs.size() >= 2);

  /*
   * We need something similar to `cacheFork` for input tensors in group 2. We
   * need this because we will want to propagate to the entire DAG except group
   * 2 and its cached inputs, so we need to make sure the DAG is still connected
   * if we remove group and its cached inputs. For example
   *    t0
   *    |
   *   cache
   *   /  \
   *  t1  t2
   * if groups = {{t1, t2}, {t0}}, then removing {t0, cache} from the DAG will
   * make it disconnected.
   */
  std::unordered_set<TensorView*> group2_and_cached_inputs(
      grouped_inputs_outputs[1].begin(), grouped_inputs_outputs[1].end());
  for (auto tv : grouped_inputs_outputs[1]) {
    if (tv->isFusionInput()) {
      auto existing_cache = ir_utils::consumerTvsOf(tv)[0];
      if (ir_utils::consumerTvsOf(existing_cache).size() > 1) {
        auto new_cache = tv->cacheAfter();
        new_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(new_cache);
      } else {
        existing_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(existing_cache);
      }
    }
  }
  // set cached outputs of group 2 to shared memory
  for (auto pair : cached_outputs) {
    auto cached_output = pair.first;
    auto output = pair.second;
    if (group2_and_cached_inputs.count(output) > 0) {
      cached_output->setMemoryType(MemoryType::Shared);
    }
  }

  TensorView* reference1 =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* reference2 =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);

  NVF_ERROR(
      reference1 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the "
      "first group.");

  NVF_ERROR(
      reference2 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the "
      "second group.");

  auto inner_most_id1 = scheduler_utils::innerMostAllocDim(reference1);
  auto inner_most_id2 = scheduler_utils::innerMostAllocDim(reference2);

  //////////////////////////////////////////
  // Step 1: Make virtual inner most dims //
  //////////////////////////////////////////

  // See note [Supporting small transpose dimensions]

  // split big dims so that we have enough dimensions available to merge with
  // inner-most dims to create the virtual inner-most dim
  scheduler_utils::splitDims(reference1, tparams->split_before_tiling);

  // prepare all dimensions in merge order for group1
  std::vector<int64_t> dims_group1 = tparams->dims_merged_with_1;
  auto inner_loop_index1 =
      domain_map.getInnerLeafDim(reference1, inner_most_id1);
  NVF_ERROR(inner_loop_index1 >= 0, "getInnerLeafDim cannot be resolved");
  int64_t inner_most_pos1_in_ref1 = inner_loop_index1;
  dims_group1.insert(dims_group1.begin(), inner_most_pos1_in_ref1);

  // prepare all dimensions in merge order for group2
  std::vector<int64_t> dims_group2 = tparams->dims_merged_with_2;
  auto inner_loop_index2 =
      domain_map.getInnerLeafDim(reference1, inner_most_id2);
  int64_t inner_most_pos2_in_ref1 = inner_loop_index2;
  NVF_ERROR(inner_loop_index2 >= 0, "getInnerLeafDim cannot be resolved");
  dims_group2.insert(dims_group2.begin(), inner_most_pos2_in_ref1);

  // merge all dimensions in group1, while updating all indices for group2
  auto merged1 =
      scheduler_utils::mergeDims(reference1, dims_group1, dims_group2);
  std::vector<int64_t> merged1_vec;
  if (merged1.has_value()) {
    merged1_vec.push_back(*merged1);
  }
  // merge all dimensions in group2, while updating merged index for group1
  auto merged2 =
      scheduler_utils::mergeDims(reference1, dims_group2, merged1_vec);

  // updating merged1 & merged2 indices if applicable
  if (merged1.has_value()) {
    inner_most_pos1_in_ref1 = merged1_vec[0];
  }
  if (merged2.has_value()) {
    inner_most_pos2_in_ref1 = *merged2;
  }

  /////////////////////////////
  // Step 2: global schedule //
  /////////////////////////////

  // make tile
  // [..., I1, .., I2, ...]
  reference1->split(inner_most_pos1_in_ref1, tparams->tile_size1);
  reference1->reorder({{inner_most_pos1_in_ref1 + 1, -1}});
  reference1->split(inner_most_pos2_in_ref1, tparams->tile_size2);
  reference1->reorder({{inner_most_pos2_in_ref1 + 1, -1}});
  // [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]

  // Merge remaining dimensions ignoring reduction axes (See Issue #2317)
  // The reduction axes cannot be at any position.
  // For example: [i0, r1, i1, r2, i2] after tiling is [i0, r1, i1/tile1, r2,
  // i2/tile2, tile1, tile2] The following code merges all the outer iterdomains
  // as: [i0 * i1/tile1 * i2/tile2, r1, r2, tile1, tile2]
  int64_t rhs_i = reference1->nDims() - 3;
  for (int64_t lhs_i = reference1->nDims() - 4; lhs_i >= 0; lhs_i--) {
    if (reference1->axis(lhs_i)->isReduction() ||
        reference1->axis(lhs_i)->isDeviceDim()) {
      continue;
    }
    if (reference1->axis(rhs_i)->isReduction() ||
        reference1->axis(rhs_i)->isDeviceDim()) {
      rhs_i = lhs_i;
      continue;
    }
    reference1->merge(lhs_i, rhs_i);
    rhs_i = lhs_i;
  }

  reference1->split(rhs_i, 1);
  // [r.., merged_dim, 1, tile1, tile2]

  // parallelize non-tile dimensions
  reference1->axis(rhs_i + 1)->parallelize(ParallelType::Unswitch);
  reference1->axis(rhs_i)->parallelize(ParallelType::BIDx);
  // [BIDx, Unswitch, tile1, tile2]

  // Propagate transformations so far to the entire DAG
  TransformPropagator propagator(reference1);
  MaxLogicalDomainInfoSpanningTree entire_dag(reference1);
  entire_dag.traverse(&propagator);

  // We may be propagating a reshape during the above transformation.
  //   T0[i0 * i1]     -> View ->   T1[i0  i1] (Root=[i0*i1])
  //      / \                       / \.
  // iDID{d} i0*i1/d           iDID{d} i0/d
  // When propagating from consumer to producer for the first time here, we will
  // replay reshape split on the view input ([i0*i1] -> [i0, i1]), followed by
  // DID loop split on the view input ([i0, i1] -> [d, i0/d, i1]) and any other
  // transforms on i0/d or i1 scheduled above. This will lose the
  // parallelization on d. Hence, we also parallelize_inputs_on_did here in case
  // T0 is a fusion input. This is not needed elsewhere. Once the reshape has
  // been replayed on the producer, the DID loop split does not need to be
  // replayed. It will be consistently present in all tvs of the fusion.
  // TODO: An alternative would be to explictly propagateReshapeTransform as in
  // other schedulers.

  scheduler_utils::parallelizeAllLike(
      reference1,
      /*selected_tvs=*/{},
      /*selected_parallel_types=*/{},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // For a transpose scheduling, all we need is to bind threadIdx.x differently
  // for inputs and outputs. This swap of binding could happen at any tensor on
  // the path from input to output, especially, it does not have to be in the
  // transpose tensor. Here, we naively do the binding swap at cached
  // input/output for simplicity. We might need to find a better set of swap
  // tensors in the future to reduce shared memory usage.

  //////////////////////////////
  // Step 3: Schedule group 2 //
  //////////////////////////////

  // transform tile for vectorization/unroll
  // See note [vectorization and unroll of input and output]

  int64_t pos = reference2->nDims() - 2;
  // [..., tile1, tile2]
  moveReductionsOut(reference2, 2);
  reference2->merge(pos);
  reference2->split(pos, tparams->vectorize_factor2);
  reference2->split(pos, tparams->getThreadsPerBlock());
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations of reference2 to the entire DAG except
  // group 1. We actually only want to propagate to the fusion outputs, but
  // inputs and outputs themselves are disconnected, so we have to borrow the
  // entire DAG and use its spanning tree.
  {
    auto all_tvs_except1 = ir_utils::allTvsExcept(
        fusion,
        {grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end()});
    SetSelector selector({all_tvs_except1.begin(), all_tvs_except1.end()});
    MaxLogicalDomainInfoSpanningTree entire_dag_except1(reference2, &selector);
    TransformPropagator propagator(reference2);
    entire_dag_except1.traverse(&propagator);
  }

  // parallelize group2 and its cached inputs
  {
    if (tparams->vectorize_factor2 > 1) {
      reference2->axis(-1)->parallelize(ParallelType::Vectorize);
    }
    reference2->axis(-2)->parallelize(ParallelType::TIDx);
    reference2->axis(-3)->parallelize(ParallelType::Unroll);

    ComputeAtMap ca_map(fusion);

    scheduler_utils::parallelizeAllLike(
        reference2,
        {group2_and_cached_inputs.begin(), group2_and_cached_inputs.end()},
        {ParallelType::TIDx});

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group2_cached_inputs.push_back(gin);
      }
    }
    if (!vectorized_group2_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference2,
          vectorized_group2_cached_inputs,
          {ParallelType::Vectorize});
    }

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group2_cached_inputs.push_back(gin);
      }
    }
    if (!unrolled_group2_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference2, unrolled_group2_cached_inputs, {ParallelType::Unroll});
    }
  }

  //////////////////////////////
  // Step 4: Schedule group 1 //
  //////////////////////////////

  // schedule group 1
  reference1->reorder({{-2, -1}});
  // [..., tile2, tile1]
  pos = reference1->nDims() - 2;
  moveReductionsOut(reference1, 2);
  reference1->merge(pos);
  reference1->split(pos, tparams->vectorize_factor1);
  reference1->split(pos, tparams->getThreadsPerBlock());
  if (tparams->vectorize_factor1 > 1) {
    reference1->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  reference1->axis(-2)->parallelize(ParallelType::TIDx);
  reference1->axis(-3)->parallelize(ParallelType::Unroll);
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations, parallelization of the reference1 to the entire
  // DAG except group 2 and its corresponding cached outputs.
  {
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(fusion, group2_and_cached_inputs);
    SetSelector selector({all_tvs_except2.begin(), all_tvs_except2.end()});
    MaxLogicalDomainInfoSpanningTree entire_dag_except_outputs(
        reference1, &selector);
    TransformPropagator propagator(reference1);
    entire_dag_except_outputs.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(
        reference1, all_tvs_except2, {ParallelType::TIDx});
  }

  // vectorize and unroll group 1's output and cached input
  {
    ComputeAtMap ca_map(fusion);
    std::vector<TensorView*> group1_and_cached_inputs(
        grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end());
    for (auto tv : grouped_inputs_outputs[0]) {
      if (tv->isFusionInput()) {
        group1_and_cached_inputs.emplace_back(ir_utils::consumerTvsOf(tv)[0]);
      }
    }

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group1_cached_inputs.push_back(gin);
      }
    }
    if (!vectorized_group1_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference1,
          vectorized_group1_cached_inputs,
          {ParallelType::Vectorize});
    }

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group1_cached_inputs.push_back(gin);
      }
    }

    if (!unrolled_group1_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference1, unrolled_group1_cached_inputs, {ParallelType::Unroll});
    }
  }

  ////////////////////////////////
  // Step 5: Cleanup and inline //
  ////////////////////////////////

  // cleanup parallelization from reference1 and reference2 if they are fusion
  // inputs
  for (auto tv : {reference1, reference2}) {
    if (tv->isFusionInput()) {
      for (auto id : tv->getLoopDomain()) {
        // DIDs are given as inputs instead of artifacts of this scheduler. So
        // do not reset them.
        if (!id->isDeviceDim()) {
          id->parallelize(ParallelType::Serial);
        }
      }
    }
  }

  // Inline
  inlineMost();

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);
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
  auto tparams = getTransposeHeuristics(fusion, runtime_info, data_cache);
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
  scheduleTranspose(fusion, tparams);
}
} // namespace nvfuser
