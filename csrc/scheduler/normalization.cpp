// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/normalization.h>
#include <scheduler/registry_utils.h>

#include <c10/util/irange.h>
#include <disjoint_set.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <grouped_reduction.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tensor_metadata.h>
#include <limits>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

using ReductionType = reduction_scheduler_utils::ReductionType;

std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  auto reduction_type = reduction_scheduler_utils::getReductionType(fusion);
  switch (reduction_type) {
    case ReductionType::Inner:
      return InnerPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::Outer:
      return OuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::InnerOuter:
      return InnerOuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::None:
      NVF_ERROR(false, "No reduction detected.");
      return nullptr;
    default:
      NVF_ERROR(false, "Reduction type not defined!");
      return nullptr;
  }
}

namespace persistent_scheduler {

// used by all persistent kernels in compile time check.
// This is the first part of the compile time check.
bool checkOpsAndInputs(Fusion* fusion, ScheduleHeuristic heuristic) {
  // Needs at least one reduction to consider.
  auto reduction_ops = ir_utils::getReductionOps(fusion);
  if (reduction_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "needs a reduction op");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(fusion, heuristic)) {
    return false;
  }

  // Fusions handled by persistent kernel scheduler cannot have MmaOp.
  if (!ir_utils::getMmaOps(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "no support for mma ops.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }
  return true;
}

// used by all persistent kernels in compile time check.
bool checkReductionType(
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic) {
  auto getExpectedType = [&heuristic]() {
    switch (heuristic) {
      case ScheduleHeuristic::InnerPersistent:
        return ReductionType::Inner;
      case ScheduleHeuristic::OuterPersistent:
        return ReductionType::Outer;
      case ScheduleHeuristic::InnerOuterPersistent:
        return ReductionType::InnerOuter;
      default:
        return ReductionType::None;
    }
  };
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  if (reduction_type != getExpectedType()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "ReductionType and heuristic doesn't match.");
    return false;
  }
  return true;
}

// used by all persistent kernels in compile time check
bool checkReductionAxis(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic) {
  // Use root domain map to check the reduction ops have the same axes
  FusionGuard fg(fusion);
  ComputeAtRootDomainMap root_map;
  root_map.build(true);
  for (const auto it : c10::irange(1, reduction_tvs.size())) {
    if (!registry_utils::checkPatternEquivalence(
            reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic,
          "unmapped reduction ",
          reduction_tvs[it - 1],
          " and ",
          reduction_tvs[it]);
      return false;
    }
  }
  return true;
}

// used by all persistent kernels in compile time check.
// This is the last part of the compile time check.
bool checkViewRootPersistentTopology(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* reference_tv,
    ScheduleHeuristic heuristic) {
  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic, "Fusion requires view being reversible.");
      return false;
    }

    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic, "View may interfere with normalization scheduling.");
      return false;
    }
  }

  // Before examining the reduction axes want to quickly
  //   check the reductions have the same axis width
  //   to avoid building root domain map in easier cases
  bool valid_axis_count = false;
  size_t axis_count = 0;
  auto reduction_root_size = [](TensorView* red_tv) {
    size_t count = 0;
    for (auto id : red_tv->getRootDomain()) {
      if (!id->isBroadcast()) {
        count++;
      }
    }
    return count;
  };

  for (auto red : reduction_tvs) {
    if (!valid_axis_count) {
      valid_axis_count = true;
      axis_count = reduction_root_size(red);
    } else {
      if (reduction_root_size(red) != axis_count) {
        scheduler_debug_utils::canScheduleRejectReason(
            heuristic,
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "has unsupported gather-like ops before normalization");
    return false;
  }
  return true;
}

// used by inner persistent kernel and innerOuter persistent kernel for run time
// check.
bool runTimeCheckIterSize(
    const scheduler_utils::ReductionTvProperties& properties,
    ScheduleHeuristic heuristic) {
  // Don't go persistent if we can't use a small fraction of the
  // available SMs yet have a large reduction size.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)device_prop->multiProcessorCount;
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)device_prop->maxThreadsPerMultiProcessor;

  if ( // Large reduction dim
      properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 &&
      properties.total_iteration_numel <
          scheduler_utils::safeDiv(device_multiprocessor_count, 8)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "not enough blocks");
    return false;
  }
  return true;
}

// used by all persistent kernels through getPersistentHeuristic
std::tuple<TensorView*, scheduler_utils::ReductionTvProperties, int64_t>
getReductionPropertiesVectFactor(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* ref_red_tv) {
  // (1) check
  NVF_ERROR(ref_red_tv != nullptr, "Reduction TensorView wasn't found.");
  NVF_ERROR(ref_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  NVF_ERROR(
      ir_utils::isReductionOp(ref_red_tv->definition()),
      "TensorView doesn't have a reduction.");
  auto tv_inps = ir_utils::filterByType<TensorView>(fusion->inputs());
  NVF_ERROR(
      std::distance(tv_inps.begin(), tv_inps.end()) > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  // (2) reduction properties
  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);

  // (3) vectorization factor
  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);
  auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      vectorize_helper::getVectorizationBreakPointOfReductionProducer(
          ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));

  return std::make_tuple(reduced_tv, properties, vectorize_factor);
}

// used by all persistent kernels through getPersistentHeuristic
std::tuple<bool, scheduler_utils::PersistentBufferSizeReturn> getBufferSizeInfo(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });
  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");
  // Grab persistent buffer sizes
  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Figure out if we want to projet persistent buffers to the inputs for
  // exmaple if we have an input tensor t0 that's fp16:
  //
  // t0 = makeSymbolicTensor(2, DataType::Half)
  // t1 = castOp(DataType::Float, t0)
  // t2 = sum(t1, 1)
  // t3 = broadcast(t2, {false, true})
  // t4 = set(t1)
  // t5 = add(t4, t3)
  // t6 = castOp(DataType::Half, t5)
  //
  // The persistent buffer is detected as being t1, which would save the
  // persistent buffer as a float, however we could obviously just save t0 which
  // is half and would take half the memory. A more complex scenario of this
  // which requires more advanced analysis is batch norm backwards.
  // TODO: Fix projected persistent buffers with view
  // https://github.com/csarofeen/pytorch/issues/2054
  // If projected persistent buffers are smaller, they will be used.
  bool can_project = ir_utils::getViewOps(fusion).empty() &&
      persistent_buffer_size_info.projected_persistent_buffer_size > 0;

  return std::make_tuple(can_project, persistent_buffer_size_info);
}

// used by inner and outer persistent kernels through getHeuristi
std::pair<int64_t, int64_t> getTensorInputNumAndMaxTypeSize(
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    TensorView* reduced_tv) {
  auto unrollable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::UnrollableInputsAndOutputs>(
          data_cache, [&reduced_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reduced_tv, false, false));
          });

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();
  // Base max dtype and n_tensor_inputs on tensors that are vectorizable (i.e.
  // share inner dimension with data pattern we're looking at).
  int64_t max_dtype_size = 1;

  // TODO: This might be better if it was the larger of input or outputs. Would
  // be even better if we had better analysis as not all unrolled elements have
  // to be alive at the same time.
  int64_t n_tensor_inputs = 0;
  for (auto tv : unrollable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }

    max_dtype_size = std::max(
        max_dtype_size,
        dataTypeSize(tv->getDataType().value(), runtime_info.getIndexType()));
    n_tensor_inputs++;
  }

  // Protect heuristics div by 0:
  n_tensor_inputs = std::max(n_tensor_inputs, (int64_t)1);

  return std::make_pair(n_tensor_inputs, max_dtype_size);
}

// common prepare for all reduction types
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs) {
  // Project the persistent buffers to the inputs. Inputs will be cached in a
  // later step, this will move them to be in a register buffer as expected.
  // dummy outputs are helper tensors to make sure persistent buffer projection
  // does not create trouble for transform propagation.
  dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
      fusion, rparams.project_persistent_buffers);

  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.
  bool unroll = rparams.isUnrolled();
  // Cache inputs even if not unrolled, as otherwise we may not create a
  // persistent buffer if that persistent buffer would be the input.
  cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);
  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // Use shared memory to store persistent buffers
  if (rparams.shared_mem_persistent_buffer) {
    const auto& persistent_buffers =
        scheduler_utils::persistentBuffers(fusion).persistent_buffers;
    for (auto tv : persistent_buffers) {
      tv->setMemoryType(MemoryType::Shared);
    }
  }

  reduction_tvs = scheduler_utils::getReductionTvs(fusion);
}

// schedule inner or outer reduction tv
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& reduction_tvs) {
  NVF_ERROR(!reduction_tvs.empty());
  // Registry assumes the reference tv is the first reduction_tv, if this
  // changes registry needs to change.
  auto reduction_tv = reduction_tvs[0];

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion, ca_map);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reduction_tv->reorder(
        scheduler_utils::domainReorderAsRfactorMap(reduction_tv));
  }

  if (rparams.persistent_kernel && rparams.cross_grid_inner_reduction &&
      !rparams.fastest_dim && reduction_tvs.size() > 1 &&
      !rparams.combined_inner_outer) {
    groupReductions(reduction_tvs, false);
  }

  auto dim_analysis = scheduler_utils::canonicalDimReduction(
      fusion, reduction_tv, rparams.fastest_dim && rparams.schedule_3D);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  NVF_ERROR(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    NVF_ERROR(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  return reduction_scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);
}

// used by inner persistent kernel and outer persistent kernel.
// This function wraps the common compile time check for inner and outer
// persistent kernels using the above 4 functions.
bool innerOrOuterCompileTimeCheck(Fusion* fusion, ScheduleHeuristic heuristic) {
  NVF_ERROR(
      heuristic == ScheduleHeuristic::InnerPersistent ||
          heuristic == ScheduleHeuristic::OuterPersistent,
      "innerOrOuterCompileTimeCheck should only be used by inner or outer persistent schedulers.");

  // (1) leading common checks for all persistent kernels.
  if (!checkOpsAndInputs(fusion, heuristic)) {
    return false;
  }

  // (2) check reduction type.
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (!checkReductionType(reduction_tvs, heuristic)) {
    return false;
  }

  // (3) check reduction axis.
  if (!checkReductionAxis(fusion, reduction_tvs, heuristic)) {
    return false;
  }

  // (4) tailing common checks for all persistent kernels.
  if (!checkViewRootPersistentTopology(
          fusion, reduction_tvs, reduction_tvs[0], heuristic)) {
    return false;
  }

  return true;
}

PersistentHeuristicArgs getInnerOrOuterPersistentHeuristicArgs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    ScheduleHeuristic heuristic) {
  FUSER_PERF_SCOPE("getInnerOrOuterPersistentHeuristicArgs");
  FusionGuard fg(fusion);
  NVF_ERROR(
      heuristic == ScheduleHeuristic::InnerPersistent ||
          heuristic == ScheduleHeuristic::OuterPersistent,
      "getInnerOrOuterPersistentHeuristicArgs should only be used by inner or outer persistent schedulers.");

  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });
  auto& reduction_tvs = reduction_tv_entry.get();

  // (1) reduction properties and vectorization factor
  auto [reduced_tv, properties, vectorize_factor] =
      persistent_scheduler::getReductionPropertiesVectFactor(
          fusion, runtime_info, data_cache, reduction_tvs, reduction_tvs[0]);

  // (2) info about persistent buffer
  auto [can_project, persistent_buffer_size_info] =
      persistent_scheduler::getBufferSizeInfo(fusion, runtime_info, data_cache);
  bool project_persistent_buffers = can_project &&
      persistent_buffer_size_info.projected_persistent_buffer_size <
          persistent_buffer_size_info.persistent_buffer_size;
  auto max_persistent_buffer_size = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;

  // (3) info about input tensors
  auto [n_tensor_inputs, max_input_dtype_size] =
      persistent_scheduler::getTensorInputNumAndMaxTypeSize(
          runtime_info, data_cache, reduced_tv);

  return PersistentHeuristicArgs{
      properties.inner_most_dimension_numel,
      properties.total_reduction_numel,
      properties.total_iteration_numel,
      max_persistent_buffer_size,
      n_tensor_inputs,
      max_input_dtype_size,
      vectorize_factor,
      project_persistent_buffers};
}

// schedule inner or outer persistent kernel
void scheduleInnerOrOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams,
    ScheduleHeuristic heuristic) {
  FUSER_PERF_SCOPE("schedulePersistentKernel");
  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  persistent_scheduler::beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      cached_outputs);

  TensorView* reference_tv = persistent_scheduler::scheduleReductionGeneral(
      fusion, rparams, reduction_tvs);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  NVF_ERROR(
      reference_tv != nullptr && reduction_tvs[0] != nullptr,
      "Need these two tensor views to finish the scheduling.");

  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  const bool unroll = rparams.isUnrolled();
  const bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams.persistent_kernel &&
      rparams.cross_grid_inner_reduction && !rparams.fastest_dim;
  reduction_scheduler_utils::multiReductionInliner(
      fusion,
      reduction_tvs[0],
      reference_tv,
      unroll,
      vectorize,
      is_outer_grid_persistence,
      reduction_tvs,
      cached_inputs,
      cached_outputs,
      dummy_outputs);

  if (rparams.compute_persistent_buffer_with_first_consumer) {
    NVF_ERROR(
        rparams.persistent_kernel,
        "computeWith should be only used with persistent kernels");
    for (const auto persistent_buffer : cached_inputs) {
      persistent_buffer->computeWith(-1, true);
    }
  }

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);
}

} // namespace persistent_scheduler
} // namespace nvfuser
