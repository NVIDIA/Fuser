// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <inlining.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner_outer.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

InnerOuterPersistentKernelScheduler::InnerOuterPersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(heuristicType()) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void InnerOuterPersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule InnerOuterPersistent Fusion");
  scheduleInnerOuterPersistentKernel(fusion, reductionParams());
}

bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  // common checks for all persistent heuristics
  if (!normalization_scheduler_utils::checkOpsAndInputs(
          fusion, heuristicType())) {
    return false;
  }

  // check reduction type
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "no reduction tv");
    return false;
  }
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  if (normalization_scheduler_utils::getPersistentHeuristicFor(
          reduction_type) != heuristicType()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "heuristicType() doesn't match with reduction type.");
    return false;
  }
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }

  // check connections between inner reduction and outer reduction tvs.
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "to use combined reduction, inner reduction and outer reduction should have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristicType(), "Fusion requires view being reversible.");
      return false;
    }
    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    auto reference_tv = inner_reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristicType(), "View may interfere with normalization scheduling.");
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
            heuristicType(),
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // the reduction axis of outer reduction tv should match to the iteration axis
  // of the inner reduction tv.
  if (!normalization_scheduler_utils::isReductionIterationAxisMatched(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "to use combined reduction, every iteration axis in inner reduction tv should match to a reduction domain in outer reduction tv.");
    return false;
  }

  if (!normalization_scheduler_utils::checkReductionPattern(
          fusion, heuristicType(), inner_reduction_tvs, outer_reduction_tvs)) {
    return false;
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

namespace {

// The roundup is due to the fact that the shared memory buffer is allocated
// as: ceilDiv(dim_size / vectorize_factor, threads_per_block).
// Let after_vect = dim_size / vectorize_factor;
// n_batch = ceilDiv(after_vect, threads_per_block);
// Then the shared memory buffer size is n_batch * vectorize_factor *
// threads_per_block * data_type_size. This function returns the maximum
// possible shared memory buffer size considering all possible block sizes.
int64_t roundUpSharedMemory(
    TensorView* tv,
    int64_t tv_buffer_size,
    int64_t vectorize_factor,
    int64_t threads_per_block_min,
    int64_t threads_per_block_max,
    int64_t threads_per_block_step) {
  int64_t data_type_size = dataTypeSize(tv->getDataType().value());
  int64_t n_elements = tv_buffer_size / data_type_size;
  int64_t after_vect = n_elements / vectorize_factor;
  int64_t max_smem = 0;
  for (int64_t threads = threads_per_block_min;
       threads <= threads_per_block_max;
       threads += threads_per_block_step) {
    int64_t n_batch = ceilDiv(after_vect, threads);
    max_smem = std::max(
        max_smem, n_batch * vectorize_factor * threads * data_type_size);
  }
  return max_smem;
}

// Return the broadcast tvs that are broadcast to the iteration dimensions of
// the inner reduction tv. These tvs are reused in the loop over the iteration
// dimension. This resue reduced the number loads from gmem and this tensor
// is likely the first candidate to be moved to shared memory when the register
// space runs low.
std::vector<TensorView*> getOuterBroadcastTvs(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  // set reference broadcast mask using the first inner reduction tv
  std::vector<bool> ref_broadcast_mask;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      const auto& root = tv->getMaybeAllocationDomain();
      ref_broadcast_mask.reserve(root.size());
      for (const auto i : c10::irange(root.size())) {
        ref_broadcast_mask.push_back(!root.at(i)->isReduction());
      }
      break;
    }
  }
  NVF_ERROR(!ref_broadcast_mask.empty(), "ref_broadcast_mask is empty!");

  // find the brodcast tensor whose broadcast makse is same to the reference
  std::vector<TensorView*> outer_broadcast_tvs;
  for (auto tv : ir_utils::allTvs(fusion)) {
    if (std::any_of(
            tv->getLeafDomain().begin(),
            tv->getLeafDomain().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      if (auto bcast = dynamic_cast<BroadcastOp*>(tv->definition())) {
        if (bcast->getBroadcastDimFlags() == ref_broadcast_mask) {
          outer_broadcast_tvs.emplace_back(tv);
        }
      }
    }
  }
  return outer_broadcast_tvs;
}

// Size of buffers storing intermediate outer reduction results
int64_t partialOuterReductionBufferSize(
    const std::vector<TensorView*>& reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t partial_reduction_buffer_size = 0;
  for (auto buffer : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(buffer)) {
      continue;
    }
    int64_t buffer_size = -1;
    for (auto id : buffer->getMaybeRFactorDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(id_size.hasValue(), "Could not infer persistent buffer size.");
      if (buffer_size == -1) {
        buffer_size = id_size.as<int64_t>();
      } else {
        buffer_size *= id_size.as<int64_t>();
      }
    }
    buffer_size = (buffer_size == -1) ? 0
                                      : buffer_size *
            (int64_t)dataTypeSize(buffer->getDataType().value(),
                                  runtime_info.getIndexType());
    partial_reduction_buffer_size += buffer_size;
  }
  return partial_reduction_buffer_size;
}

// Sorts projectable_buffer_inputs based on the latency from smallest to
// largest. The latency is calculated based on the number of uses, the buffer
// data type size, and the pre-defined schedule structures. Assume each
// global memory load has latency 100, each shared memory load has latency 10,
// latency is proportional to the data type size. There is no write from
// register to shared memory, so this type of latency is not considered. The
// magic number 100 and 10 are not critical for our current major use case, e.g.
// layer norm backward, and rms norm backward. The outer broadcast tensor (the
// weight tensor) is almost always moved to shared memory before other buffers
// unless the outer dimension is very small, e.g. less than 1 sm count or other
// buffers are used 10 times more than the outer broadcast tensor.
std::vector<TensorView*> sortPersistentBuffers(
    Fusion* fusion,
    const std::vector<TensorView*>& persistent_buffers,
    const std::vector<TensorView*>& projectable_buffer_inputs,
    const std::vector<TensorView*>& reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  const int64_t n_buffers = (int64_t)persistent_buffers.size();

  // Set the number of uses for each projected buffer.
  // Includes direct use and the use of its corresponding persistent buffer.
  // When project to inputs, we are replacing the use of the original persistent
  // buffers except for the one goes to reduction.
  std::vector<int> n_proj_buffer_uses(n_buffers, 0);
  for (auto idx = 0; idx < n_buffers; idx++) {
    auto input_tv = projectable_buffer_inputs[idx];
    n_proj_buffer_uses[idx] = (int)input_tv->uses().size();
    for (auto buffer : persistent_buffers) {
      if (DependencyCheck::isDependencyOf(input_tv, buffer)) {
        n_proj_buffer_uses[idx] += (int)buffer->uses().size() - 1;
      }
    }
  }

  // assume the fusion is scheduled with a two-step approach, the first step
  // finishes the inner reduction and a partial outer reduction. The second
  // step finishes the outer reduction. In the first step, the outer dimension
  // is parallelized by BIDy, each thread loops over the outer dimension
  // for n_outer_loop = ceilDiv(outer_dim_size, gdimy) times. gdimy is a
  // heuristic para, here we assume gdimy = sm_count. For outer broadcast tv,
  // e.g. weight tensor, it is reused in every outer loop, when calculate the
  // latency of global memory load, 1 is used instead of n_outer_loop. For other
  // tensors, each loop is loading a different row, so n_outer_loop is used.
  std::vector<int> buffer_latency(n_buffers, 0);
  int sm_count =
      (int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int n_outer_loop = [&sm_count, &reduction_tvs, &runtime_info]() -> int {
    for (auto buffer : reduction_tvs) {
      if (scheduler_utils::isFastestDimReduction(buffer)) {
        continue;
      }
      int64_t outer_dim_size = -1;
      for (auto id : buffer->getMaybeRFactorDomain()) {
        if (id->isReduction()) {
          auto id_size =
              runtime_info.expressionEvaluator().evaluate(id->extent());
          NVF_ERROR(id_size.hasValue(), "Could not infer outer dim size.");
          if (outer_dim_size == -1) {
            outer_dim_size = id_size.as<int64_t>();
          } else {
            outer_dim_size *= id_size.as<int64_t>();
          }
        }
      }
      int gdimy = sm_count;
      int n_loops = (int)ceilDiv(outer_dim_size, gdimy);
      return n_loops;
    }
    NVF_ERROR(false, "Could not infer outer loop count.");
    return 1;
  }();

  // Get the outer broadcast tensors to check which input buffer is reused in
  // the loop over the outer dimension.
  const auto& outer_broadcast_tvs = getOuterBroadcastTvs(fusion, reduction_tvs);

  // set latency for each buffer.
  // Assume load from gmem to smem has a latency of 100, load from smem to
  // register has a latency of 10. Total latency is proportional to data type
  // size.
  for (auto idx = 0; idx < n_buffers; idx++) {
    int latency = 0;
    auto input_buffer = projectable_buffer_inputs.at(idx);
    bool is_reused = std::any_of(
        outer_broadcast_tvs.begin(),
        outer_broadcast_tvs.end(),
        [&input_buffer](TensorView* tv) {
          return DependencyCheck::isDependencyOf(input_buffer, tv);
        });
    // for each load from gmem to smem, latency += 100
    latency += 100 * (is_reused ? 1 : n_outer_loop);
    // for each load from smem to regs, latency += 10
    latency += 10 * n_outer_loop * n_proj_buffer_uses.at(idx);
    int dtype_size = (int)dataTypeSize(input_buffer->getDataType().value());
    buffer_latency[idx] = latency * dtype_size;
  }

  // reorder projectable_buffer_inputs based on the latency from smallest to
  // largest
  std::vector<int> idxs(n_buffers);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::stable_sort(idxs.begin(), idxs.end(), [&buffer_latency](int i, int j) {
    return buffer_latency[i] < buffer_latency[j];
  });
  std::vector<TensorView*> sorted_candidate_tvs;
  sorted_candidate_tvs.reserve(n_buffers);
  for (auto idx : idxs) {
    sorted_candidate_tvs.emplace_back(projectable_buffer_inputs.at(idx));
  }
  return sorted_candidate_tvs;
}

// Decide where to store persistent buffers.
// By default, they reside in registers.
// If register space runs low but there's ample shared memory,
// move one or more buffers to shared memory. Since shared memory
// is slower than registers, we should move the buffers that are
// less frequently accessed to shared memory.
struct PersistentBufferStorageParams {
  std::vector<TensorView*> smem_persistent_tvs;
  int64_t smem_buffer_size = -1;
  int64_t regs_buffer_size = -1;
  int64_t smem_overhead = -1;
  bool has_enough_regs_and_smem = false;
  bool project_to_input = false;
};

PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor) {
  PersistentBufferStorageParams buffer_params;

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  buffer_params.project_to_input = ir_utils::getViewOps(fusion).empty() &&
      persistent_buffer_size_info.projected_persistent_buffer_size > 0;

  const auto& persistent_buffers = buffer_params.project_to_input
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;

  auto total_buffer_size = buffer_params.project_to_input
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;
  total_buffer_size +=
      partialOuterReductionBufferSize(reduction_tvs, runtime_info);

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t smem_overhead =
      scheduler_utils::getSharedMemoryOverheadPerBlock(fusion, reduction_tvs);
  int64_t available_smem =
      (int64_t)dev_prop->sharedMemPerMultiprocessor - smem_overhead;
  int64_t available_regs = scheduler_utils::register_file_size_inner_outer;
  buffer_params.smem_overhead = smem_overhead;

  // Put all the persistent tensors in registers
  buffer_params.regs_buffer_size = total_buffer_size;
  buffer_params.smem_buffer_size = 0;

  // Relocate buffers to shared memory until the buffer size in registers is
  // within the allowable limit.
  if (buffer_params.regs_buffer_size > available_regs) {
    const auto sorted_candidate_tvs = sortPersistentBuffers(
        fusion,
        persistent_buffers,
        persistent_buffer_info.projectable_buffer_inputs,
        reduction_tvs,
        runtime_info);
    // calculate the accumulated buffer size of the first N buffers
    const int64_t n_buffers = (int64_t)persistent_buffers.size();
    std::vector<int64_t> acc_regs_buffer_sizes(n_buffers + 1, 0);
    std::vector<int64_t> acc_smem_buffer_sizes(n_buffers + 1, 0);
    for (int i = 1; i <= n_buffers; i++) {
      int64_t tv_buffer_size_regs =
          scheduler_utils::getPersistentBufferSizeOfTensor(
              sorted_candidate_tvs[i - 1],
              runtime_info,
              persistent_buffer_info);
      int64_t tv_buffer_size_smem = roundUpSharedMemory(
          sorted_candidate_tvs[i - 1],
          tv_buffer_size_regs,
          vectorize_factor,
          InnerOuterPersistentKernelScheduler::threads_per_block_min,
          InnerOuterPersistentKernelScheduler::threads_per_block_max,
          dev_prop->warpSize);

      acc_regs_buffer_sizes[i] =
          acc_regs_buffer_sizes[i - 1] + tv_buffer_size_regs;
      acc_smem_buffer_sizes[i] =
          acc_smem_buffer_sizes[i - 1] + tv_buffer_size_smem;
    }

    // check if we have enough shared memory and registers for the buffers.
    // start with an optimal register size, if it's not enough, try the full
    // size. Test shows using full size leads to severe register spills but it
    // is still faster than segmenting the fusion into multiple kernels.
    int64_t n_smem_buffer = -1;
    for (auto tmp_ava_regs :
         {available_regs, scheduler_utils::register_file_size_full}) {
      bool can_schedule = true;
      int64_t n_smem_buffer_tmp = -1;
      // Determine the least number of buffers to transfer to shared memory
      // to ensure the register buffer size doesn't exceed the available limit.
      for (int i = 1; i <= n_buffers; i++) {
        if (buffer_params.regs_buffer_size - acc_regs_buffer_sizes[i] <=
            tmp_ava_regs) {
          n_smem_buffer_tmp = i;
          break;
        }
      }

      // Can't be scheduled if n_smem_buffer is not set or requested shared
      // memory is larger than available.
      if (n_smem_buffer_tmp == -1 ||
          acc_smem_buffer_sizes[n_smem_buffer_tmp] > tmp_ava_regs) {
        can_schedule = false;
      }

      // no need to try the other tmp_ava_regs
      // mark can_schedule as true and break from the loop
      if (can_schedule) {
        available_regs = tmp_ava_regs;
        n_smem_buffer = n_smem_buffer_tmp;
        buffer_params.has_enough_regs_and_smem = true;
        break;
      }
    }
    if (!buffer_params.has_enough_regs_and_smem) {
      return buffer_params;
    }

    // move n_smem_buffer buffers to shared memory
    for (int i = 0; i < n_smem_buffer; i++) {
      buffer_params.smem_persistent_tvs.emplace_back(sorted_candidate_tvs[i]);
    }
    buffer_params.regs_buffer_size -= acc_regs_buffer_sizes[n_smem_buffer];
    buffer_params.smem_buffer_size = acc_smem_buffer_sizes[n_smem_buffer];
  }

  // Double check
  buffer_params.has_enough_regs_and_smem =
      (buffer_params.smem_buffer_size <= available_smem) &&
      (buffer_params.regs_buffer_size <= available_regs);
  NVF_ERROR(
      buffer_params.has_enough_regs_and_smem,
      "Not enough registers and shared memory for persistence! Should return early.");
  return buffer_params;
}

} // namespace

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canSchedule");
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto reference_tv = first_inner_reduction_tv;

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));

  // check if there is enough register and shared memory for persistence
  const auto buffer_params = getPersistentBufferStorageParams(
      fusion, runtime_info, data_cache, reduction_tvs, vectorize_factor);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (!buffer_params.has_enough_regs_and_smem) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(),
        "not enough registers or shared memory for persistence");
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      buffer_params.regs_buffer_size, scheduler_utils::register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "requires over half GPU persistence.");
    return false;
  }

  // Don't go persistent if we can't use a small fraction of the
  // available SMs yet have a large reduction size.
  if ( // Large reduction dim
      properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 &&
      properties.total_iteration_numel <
          (properties.fastest_dim_reduction
               ? scheduler_utils::safeDiv(device_multiprocessor_count, 8)
               // Make sure we at least use a quarter of the device * a
               // half warp
               : (warp_size / 8) * device_multiprocessor_count)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristicType(), "not enough blocks");
    return false;
  }

  return true;
}

void InnerOuterPersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getInnerOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

namespace {

// Parameters for inner reduction:
// Reduction dim: inner_vect, persistent_batch_size, bdimx and bdimy
// Iteration dim: gdimy

// Parameters for outer reduction:
// Reduction dim: bdimy
// Iteration dim: vectorization_factor_outer, bdimx, gdimy
struct InnerOuterParams {
  int64_t inner_vect = -1;
  int64_t persistent_batch_size = -1;
  int64_t bdimx = -1;
  int64_t bdimy = -1;
  int64_t gdimy = -1;
  int64_t occupancy = -1;
  int64_t register_per_thread = -1;
  int64_t n_persistent_tails = -1;
  int64_t non_buffer_registers = -1;

  int64_t tmp_gmem_write_vect = -1;
  int64_t vectorization_factor_outer = -1;
  void print() const {
    std::cout << "bdimx: " << bdimx << ", bdimy: " << bdimy
              << ", persistent_batch_size: " << persistent_batch_size
              << ", occupancy: " << occupancy
              << ", register_per_thread: " << register_per_thread
              << ", n_persistent_tails: " << n_persistent_tails
              << ", non_buffer_registers: " << non_buffer_registers
              << ", gdimy: " << gdimy << std::endl;
  }
};
int64_t getMaxPersistentBatch(
    const int64_t buffer_bytes_per_batch,
    const int64_t target_threads_per_sm,
    const int64_t register_overhead) {
  // (1) calculate the maximum register count given the target occupancy.
  int64_t total_register =
      getRegPerThreadGivenThreadsPerSM(target_threads_per_sm);
  int64_t register_for_buffer = total_register - register_overhead;

  // (2) calculate the maximum persistent batch size using the register count.
  int64_t batch_from_register = scheduler_utils::safeDiv(
      register_for_buffer * scheduler_utils::bytes_per_register,
      buffer_bytes_per_batch);

  // (3) Avoid using very large persistent buffer size, which may lead to low
  // occupancy due to the limitation of the current heuristics. TODO: remove
  // this parameter when we have a better heuristic to select the best
  // persistent batch size.
  constexpr int64_t max_batches_per_block = 10l;
  return std::min(max_batches_per_block, batch_from_register);
}
std::pair<int64_t, int64_t> getMaxRegisterCountPerThreadAndOccupancy(
    const int64_t buffer_size_per_thread,
    const int64_t threads_per_block,
    const int64_t target_warps_per_sm,
    const int64_t register_overhead) {
  // convert [target_warps_per_sm] to [target_blocks_per_sm]
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = dev_prop->warpSize;
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  // ensure higher than target by round up
  int64_t target_blocks_per_sm =
      ceilDiv(target_warps_per_sm * threads_per_warp, threads_per_block);
  // ensure lower than hardware limit by round down
  if (threads_per_block * target_blocks_per_sm > max_threads_per_sm) {
    target_blocks_per_sm = max_threads_per_sm / threads_per_block;
  }
  // minimum register each thread should use to avoid spills
  const int64_t register_per_thread_min =
      buffer_size_per_thread / scheduler_utils::bytes_per_register +
      register_overhead;

  // (1) use register calculated from target occupancy
  int64_t threads_per_sm = threads_per_block * target_blocks_per_sm;
  int64_t register_per_thread_target =
      getRegPerThreadGivenThreadsPerSM(threads_per_sm);

  if (register_per_thread_target >= register_per_thread_min) {
    return {register_per_thread_target, threads_per_sm / threads_per_warp};
  }

  //(2) can't achieve target occupancy. Estimate occupancy from minimum register
  // each thread should use, then derive register per thread from occupancy.
  int64_t blocks_per_sm_max = scheduler_utils::safeDiv(
      getThreadsPerSMGivenRegPerThread(register_per_thread_min),
      threads_per_block);
  threads_per_sm =
      std::min(blocks_per_sm_max * threads_per_block, max_threads_per_sm);

  return {
      getRegPerThreadGivenThreadsPerSM(threads_per_sm),
      threads_per_sm / threads_per_warp};
}
bool compareTwoHeuristics(
    const InnerOuterParams& ha,
    const InnerOuterParams& hb,
    const int64_t min_non_buffer_registers,
    const int64_t target_warps_per_sm) {
  auto compare = [](int64_t a, int64_t b) -> int {
    return a > b ? 1 : (a < b ? -1 : 0);
  };
  int score = 0;

  // prefer occupancy larger than target
  score = compare(
      ha.occupancy >= target_warps_per_sm, hb.occupancy >= target_warps_per_sm);
  if (score != 0) {
    return score > 0;
  }

  // prefer reduction count after vectorization is divisible by persistent
  // batch size
  score = compare(ha.n_persistent_tails == 0, hb.n_persistent_tails == 0);
  if (score != 0) {
    return score > 0;
  }

  // Ensure the count of non buffer registers is larger than the min overhead.
  // But don't want to achieve this goal at the cost of using a very large block
  // size, it avoids using a small persistent batch with a large block size,
  // which usually leads to 10% lower in performance.
  constexpr int64_t opt_max_threads_per_block = 512;
  score = compare(
      ha.non_buffer_registers > min_non_buffer_registers &&
          ha.bdimx * ha.bdimy <= opt_max_threads_per_block,
      hb.non_buffer_registers > min_non_buffer_registers &&
          hb.bdimx * hb.bdimy <= opt_max_threads_per_block);
  if (score != 0) {
    return score > 0;
  }

  // Prefer large occupancy
  score = compare(ha.occupancy, hb.occupancy);
  if (score != 0) {
    return score > 0;
  }

  // Tiebreaker, use large persistent batch size so more registers are used
  // for the persistent buffer.
  return ha.persistent_batch_size > hb.persistent_batch_size;
}

InnerOuterParams getHeuristicParamsGivenPerisisentBatchSize(
    const int64_t inner_dim_numel,
    const int64_t outer_dim_numel,
    const int64_t max_multi_reduction_factor,
    const int64_t min_threads_per_block,
    const int64_t register_buffer_per_batch,
    const int64_t smem_buffer_per_batch,
    const int64_t target_warps_per_sm,
    const int64_t register_overhead,
    const int64_t smem_overhead,
    const int64_t max_vectorize_factor,
    const int64_t persistent_batch_size) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  auto device_warp_size = dev_prop->warpSize;
  // auto max_threads_per_block = dev_prop->maxThreadsPerBlock;
  auto device_multiprocessor_count = dev_prop->multiProcessorCount;

  InnerOuterParams params;
  params.inner_vect = max_vectorize_factor;

  // Step-1, calc threads per block from persistent batch size
  params.persistent_batch_size = persistent_batch_size;
  auto threads_per_block = scheduler_utils::safeDiv(
      inner_dim_numel / params.inner_vect, persistent_batch_size);
  threads_per_block =
      scheduler_utils::roundUpToN(threads_per_block, device_warp_size);

  // Step-2, calc gdimy from occupancy
  int64_t persistent_buffer_size =
      register_buffer_per_batch * persistent_batch_size;
  auto reg_occ = getMaxRegisterCountPerThreadAndOccupancy(
      persistent_buffer_size,
      threads_per_block,
      target_warps_per_sm,
      register_overhead);
  params.register_per_thread = reg_occ.first;
  params.occupancy = reg_occ.second;
  int64_t warps_per_block = threads_per_block / device_warp_size;
  int64_t blocks_per_sm_regs = reg_occ.second / warps_per_block;
  int64_t blocks_per_sm_smem = (int64_t)dev_prop->sharedMemPerMultiprocessor /
      (smem_overhead +
       smem_buffer_per_batch * persistent_batch_size * threads_per_block);
  int64_t blocks_per_sm = std::min(blocks_per_sm_regs, blocks_per_sm_smem);
  int64_t gdimy_occupancy = blocks_per_sm * device_multiprocessor_count;
  // round down to a divisible value
  const int64_t outer_iter_min =
      std::max(8l, ceilDiv(outer_dim_numel, gdimy_occupancy));
  params.gdimy = ceilDiv(outer_dim_numel, outer_iter_min);
  std::cout << "gdimy_occupancy: " << gdimy_occupancy
            << ", params.gdimy: " << params.gdimy << std::endl;

  // inner dim: vect8 x batch x bdimx x bdimy >= N
  // inner dim: vect4 x bdimx x gdimy >=N
  // ---------> 2 x batch x bdimy ~=~ gdimy
  // thread serial reduction of the final outer reduction: ~= 2 x batch
  // Step-3, set OuterParams Iteration dim: vectorization_factor_outer, bdimx,
  // gdimy (already done) The partial outer reduction result is stored in tmp
  // gmem, set the vectorization factor for write and read
  params.tmp_gmem_write_vect = std::min(4l, max_vectorize_factor);
  params.vectorization_factor_outer = std::min(4l, max_vectorize_factor);
  // For widely used hidden sizes, threads_per_block has factor of 8, roundup to
  // increase the probability of bdimx * bdimy == threads_per_block.
  params.bdimx = scheduler_utils::roundUpPow2Or8(ceilDiv(
      inner_dim_numel / params.vectorization_factor_outer, params.gdimy));
  // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
  // increase bdimx to make it divisible. Under worst case, bdimx equals to
  // threads_per_block.
  while (threads_per_block % params.bdimx) {
    params.bdimx = std::min(params.bdimx + 8, threads_per_block);
  }
  // Step-4, set OuterParams Reduction dim: bdimy.
  params.bdimy = threads_per_block / params.bdimx;
  NVF_ERROR(
      params.bdimy * params.bdimx == threads_per_block,
      " threads_per_block must be divisible by bdimx and bdimy.");

  params.non_buffer_registers = params.register_per_thread -
      persistent_buffer_size / scheduler_utils::bytes_per_register;
  // (4) Calculate other quantities reflecting the quality of the heuristic.
  // when [reduction_count_after_vectorize] is not divisible by
  // [persistent_val], the last batch is not be fully utilized, the wasted
  // threads in the last batch is quantified as [n_persistent_tails].
  params.n_persistent_tails = threads_per_block * persistent_batch_size -
      inner_dim_numel / params.inner_vect;
  return params;
}

// The innerOuterPersistentHeuristic is tuned for layer_norm backward on A100
// ======= Method if hidden_size > 1024 =======
// (1) Inner reduction is one reduction per block. Reduction domain is
// parallelized by TIDx and TIDy, Iteration domain is parallelized by BIDy. (2)
// Outer reduction is done in two-steps. The first step is partial reduction,
// reduction domain is parallelized by BIDy, iteration domain is parallelized by
// TIDx and TIDy. The partial results are written to gmem followed by a grid
// sync. The second step is block reduction, the reduction domain is
// parallelized by TIDy, the iteration domain is parallelized by TIDx and BIDy.
// ======= Method if hidden_size <= 1024 =======
// (1) Inner reduction is multi-reductions per blocks. Reduction domain is
// parallelized by TIDx, Iteration domain is parallelized by BIDy and TIDy
// (2) Outer reduction is same to cases where hidden_size > 1024 except the
// second step where in this case, the reduction domain is parallelized by TIDx
// and the iteration domain is parallelized by TIDy and BIDy. This switch
// between TIDx and TIDy is because (a) We can do warp reduction with TIDx and
// (b) TIDx*BIDy is usually much larger than hidden_size, e.g. 128*216 = 1024*27
// this means without switch only 1/27 of the threads is used.
std::shared_ptr<ReductionParams> innerOuterPersistentHeuristic(
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t smem_buffer_size,
    const int64_t smem_overhead,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor,
    const bool project_to_input,
    const PrimDataType index_type) {
  auto rparams = std::make_shared<ReductionParams>();
  rparams->shared_mem_persistent_buffer = smem_buffer_size > 0;
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;

  const int64_t register_overhead = 32;
  const int64_t target_warps_per_sm = 8;
  // device properties
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = (int64_t)dev_prop->warpSize;
  const int64_t max_threads_in_block = 256;
  const int64_t max_threads_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // alwasy use [vectorize_factor]
  const int64_t parallel_after_vectorize =
      inner_dim_numel / (int64_t)vectorize_factor;

  // try to use at least 4 warps per block
  const int64_t min_threads_per_block = 4l * threads_per_warp;

  // set the min persistent buffer size to avoid requesting
  // a block size larger than device limit
  const int64_t pbs_min =
      ceilDiv(parallel_after_vectorize, max_threads_in_block);

  // set the max persistent batch size to avoid low occupancy
  // (1) limitation set by min_threads_per_block
  const int64_t pbs_max_1 =
      ceilDiv(parallel_after_vectorize, min_threads_per_block);
  // (2) derived the maximum persistent batch size from the target occupancy
  const int64_t regs_buffer_per_batch =
      regs_buffer_size / inner_dim_numel * (int64_t)vectorize_factor;
  const int64_t smem_buffer_per_batch =
      smem_buffer_size / inner_dim_numel * (int64_t)vectorize_factor;

  const int64_t target_threads_per_sm =
      std::min(target_warps_per_sm * threads_per_warp, max_threads_per_sm);
  const int64_t pbs_max_2 = getMaxPersistentBatch(
      regs_buffer_per_batch, target_threads_per_sm, register_overhead);
  std::cout << "pbs_max_2: " << pbs_max_2
            << " regs_buffer_per_batch: " << regs_buffer_per_batch
            << " target_threads_per_sm: " << target_threads_per_sm << std::endl;
  const int64_t pbs_max = std::max(pbs_min, std::min(pbs_max_1, pbs_max_2));

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size, regs_buffer_size),
      ceilDiv(outer_dim_numel, device_multiprocessor_count));

  // Generate a heuristic for each possible persistent batch size.
  // record which persistent batch size has the highest occupancy.
  int64_t idx_max_occupancy = -1;
  int64_t current_max_occupancy = -1;
  std::vector<InnerOuterParams> all_heuristics;
  all_heuristics.reserve(pbs_max - pbs_min + 1);
  for (int64_t pbs = pbs_min; pbs <= pbs_max; pbs++) {
    all_heuristics.push_back(getHeuristicParamsGivenPerisisentBatchSize(
        inner_dim_numel,
        outer_dim_numel,
        max_multi_reduction_factor,
        min_threads_per_block,
        regs_buffer_per_batch,
        smem_buffer_per_batch,
        target_warps_per_sm,
        register_overhead,
        smem_overhead,
        vectorize_factor,
        pbs));
    if (all_heuristics.back().occupancy > current_max_occupancy) {
      current_max_occupancy = all_heuristics.back().occupancy;
      idx_max_occupancy = (int64_t)all_heuristics.size() - 1;
    }
  }

  // Sort the heuristics and select the best one.
  // If no persistent batch size can achieve the target occupancy, and
  InnerOuterParams iop;
  if (current_max_occupancy < target_warps_per_sm) {
    iop = all_heuristics.at(idx_max_occupancy);
  } else {
    std::stable_sort(
        all_heuristics.begin(),
        all_heuristics.end(),
        [&register_overhead](
            const InnerOuterParams& a, const InnerOuterParams& b) {
          return compareTwoHeuristics(
              a, b, register_overhead, target_warps_per_sm);
        });
    iop = all_heuristics.at(0);
  }

  for (auto& heuristic : all_heuristics) {
    heuristic.print();
  }

  rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.register_per_thread;
  rparams->unroll_factor_inner_reduction = iop.inner_vect;
  rparams->batches_per_block_inner_reduction = iop.persistent_batch_size;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->vectorize_inner_reduction = iop.inner_vect > 1;
  rparams->split_grid_dim_iter_dom_outer = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      iop.gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      iop.bdimx,
      iop.bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  if (rparams->shared_mem_persistent_buffer) {
    rparams->tag =
        "InnerOuter Register and Shared Memory Persistent Heuristic.\n";
  } else {
    rparams->tag = "InnerOuter Register Persistent Heuristic.\n";
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "regs_buffer_size: " << regs_buffer_size << "\n"
            << "smem_buffer_size: " << smem_buffer_size << "\n"
            << "smem_overhead: " << smem_overhead << "\n"
            << "vectorize_factor_input: " << iop.inner_vect << "\n"
            << "vectorization_factor_tmp_gmem_write: "
            << iop.tmp_gmem_write_vect << "\n"
            << "vectorization_factor_outer: " << iop.vectorization_factor_outer
            << "\n"
            << "multiple_reds_per_blk: " << rparams->multiple_reds_per_blk
            << "\n"
            << "gdimy: " << iop.gdimy << "\n"
            << "block(" << (iop.bdimx) << ", " << iop.bdimy << ", " << 1 << ")";
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

} // namespace

std::shared_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getInnerOuterPersistentHeuristics");
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto ref_red_tv = first_inner_reduction_tv;

  // Verify the presence of a reduction TensorView connected to a Fusion input
  normalization_scheduler_utils::checkReductionTvForScheduling(
      fusion, ref_red_tv);

  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      vectorize_helper::getVectorizationBreakPointOfReductionProducer(
          ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));

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
  auto buffer_params = getPersistentBufferStorageParams(
      fusion, runtime_info, data_cache, reduction_tvs, vectorize_factor);

  // dtype used to store partial outer reduction in combined reduction
  int64_t max_outer_reduction_dtype_size = 1;
  for (auto tv : reduction_tvs) {
    if (!scheduler_utils::isFastestDimReduction(tv)) {
      max_outer_reduction_dtype_size = std::max(
          max_outer_reduction_dtype_size,
          dataTypeSize(tv->getDataType().value()));
    }
  }

  std::shared_ptr<ReductionParams> rparams = innerOuterPersistentHeuristic(
      properties.total_iteration_numel,
      properties.total_reduction_numel,
      buffer_params.regs_buffer_size,
      buffer_params.smem_buffer_size,
      buffer_params.smem_overhead,
      max_outer_reduction_dtype_size,
      vectorize_factor,
      buffer_params.project_to_input,
      runtime_info.getIndexType());

  // save persistent tvs should use shared memory, to avoid calling
  // getPersistentBufferStorageParams again during the scheduling.
  if (rparams->shared_mem_persistent_buffer) {
    rparams->smem_persistent_tvs = buffer_params.smem_persistent_tvs;
  }

  return rparams;
}

std::shared_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getInnerOuterPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  return getInnerOuterPersistentHeuristics(fusion, runtime_info, data_cache);
}

namespace {

void scheduleReductionCombinedOuter(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& outer_reduction_tvs,
    std::vector<TensorView*>& cached_gmem,
    std::vector<TensorView*>& cached_gmem_reload,
    std::vector<TensorView*>& outer_reference_tvs,
    std::unordered_set<TensorView*>& boundaryNodesSet) {
  auto mergeReductionOrIterDomains = [](TensorView* tv, bool mergeReduction) {
    int prev_i = -1;
    for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
      if (mergeReduction == tv->axis(i)->isReduction()) {
        if (prev_i == -1) {
          prev_i = i;
        } else {
          tv->merge(i, prev_i);
          prev_i = i;
        }
      }
    }
  };
  for (auto& outer_reduction_tv : outer_reduction_tvs) {
    // merge tensorview to [reduction, iteraiton] domains
    mergeReductionOrIterDomains(outer_reduction_tv, true);
    mergeReductionOrIterDomains(outer_reduction_tv, false);
    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(rparams.block_dim_iter_dom));
    }
    outer_reduction_tv->split(
        0, NamedScalar::getParallelDim(rparams.grid_dim_iter_dom), false);

    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->rFactor({1});
    }
    TensorView* partialResult = outer_reduction_tv->rFactor({1});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    if (rparams.multiple_reds_per_blk) {
      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        // to use warp reduction
        if (rparams.pad_outer_reduction_to_warp) {
          outer_reduction_tv->axis(1)->padToMultipleOfWarp();
        }
      } else {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
      }
      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDy);
      } else {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);

    } else {
      // reduction domain
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(ParallelType::TIDy));
      outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.lparams.bdimx() > 1) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));

      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);
    }
    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
  }
}

} // namespace

// fusion is the input IR that will be modified by this function
void scheduleInnerOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("scheduleInnerOuterPersistentKernel");

  FusionGuard fg(fusion);

  // Project the persistent buffers to the inputs. Inputs will be cached in a
  // later step, this will move them to be in a register buffer as expected.
  // dummy outputs are helper tensors to make sure persistent buffer projection
  // does not create trouble for transform propagation.
  const auto& dummy_outputs =
      reduction_scheduler_utils::projectPersistentBuffers(
          fusion, rparams.project_persistent_buffers);

  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.
  const bool unroll = rparams.isUnrolled();
  // Cache inputs even if not unrolled, as otherwise we may not create a
  // persistent buffer if that persistent buffer would be the input.
  const auto& cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  const auto& cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);
  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // Transfer the persistent buffer tensors to shared memory. These tensors are
  // housed in smem_persistent_tvs. If a candidate tensor is input, move its
  // associated cached tensors.
  if (rparams.shared_mem_persistent_buffer) {
    const auto& persistent_buffers =
        scheduler_utils::persistentBuffers(fusion).persistent_buffers;
    auto isSharedMemoryPersistent = [&rparams](const TensorView* lookup_tv) {
      return std::any_of(
          rparams.smem_persistent_tvs.begin(),
          rparams.smem_persistent_tvs.end(),
          [lookup_tv](const auto* tv) {
            // can't use `tv->sameAs(lookup_tv)` since the saved tvs in
            // smem_persistent_tvs are from a cloned fusion.
            return tv->name() == lookup_tv->name();
          });
    };
    for (auto tv : persistent_buffers) {
      bool use_smem = isSharedMemoryPersistent(tv);
      if (!use_smem &&
          std::find(cached_inputs.begin(), cached_inputs.end(), tv) !=
              cached_inputs.end()) {
        auto input_tv = ir_utils::producerTvsOf(tv).at(0);
        use_smem = isSharedMemoryPersistent(input_tv);
      }
      if (use_smem) {
        tv->setMemoryType(MemoryType::Shared);
      }
    }
  }

  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  // split reduction_tvs into inner and outer reduction_tvs
  std::vector<TensorView*> inner_reduction_tvs, outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  NVF_ERROR(
      !inner_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no inner reduction is found.");
  NVF_ERROR(
      !outer_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no outer reduction is found.");

  // schedule inner reduction, only schedule the first inner reduction tv,
  // then will be propagated to other inner reduction tvs.
  TensorView* inner_reference_tv =
      normalization_scheduler_utils::scheduleReductionGeneral(
          fusion,
          rparams,
          inner_reduction_tvs,
          InnerOuterPersistentKernelScheduler::heuristicType());

  // schedule outer reduction, schedule all the outer reduction tvs since we
  // need to store the intermediate results.
  std::vector<TensorView*> cached_gmem;
  std::vector<TensorView*> cached_gmem_reload;
  std::vector<TensorView*> outer_reference_tvs;
  std::unordered_set<TensorView*> boundaryNodesSet;
  scheduleReductionCombinedOuter(
      fusion,
      rparams,
      outer_reduction_tvs,
      cached_gmem,
      cached_gmem_reload,
      outer_reference_tvs,
      boundaryNodesSet);

  // Propagate inner reduction and outer reductions
  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  const bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams.persistent_kernel &&
      rparams.cross_grid_inner_reduction && !rparams.fastest_dim;

  // Propagate inner reduction. There is a cutoff at boundaryNodesSet, so this
  // propagation will not propagate to the final outer reduction.
  reduction_scheduler_utils::propagateTransformation(
      inner_reference_tv, boundaryNodesSet);
  reduction_scheduler_utils::propagateRFactor(
      inner_reference_tv, inner_reduction_tvs[0], inner_reduction_tvs);

  // Don't allow parallelization propagation goes through boundaryNodesSet
  const auto& selected_tvs_inner =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, boundaryNodesSet);
  reduction_scheduler_utils::propagateParallelization(
      fusion,
      inner_reduction_tvs[0],
      inner_reference_tv,
      unroll,
      vectorize,
      is_outer_grid_persistence,
      inner_reduction_tvs,
      cached_inputs,
      cached_outputs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at
  // boundaryNodesSet. Thus, we need a loop to initiate the propagation from
  // each outer reduction. Don't allow parallelization propagation goes
  // through cached_gmem, see issue 246.
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    reduction_scheduler_utils::propagateParallelization(
        fusion,
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        unroll,
        vectorize,
        is_outer_grid_persistence,
        outer_reduction_tvs,
        cached_inputs,
        cached_outputs,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // special vectorization of temp gmem, vectorization_factor_tmp_gmem_write
  // is guaranteed to be smaller or equal to input vectorization factor.
  if (rparams.vectorization_factor_tmp_gmem_write > 1) {
    for (auto tv : cached_gmem) {
      NVF_ERROR(
          rparams.vectorization_factor_tmp_gmem_write <=
              rparams.unroll_factor_inner_reduction,
          "vectorization factor of temp gmem write should be smaller than that of inner reduction.")
      if (rparams.vectorization_factor_tmp_gmem_write <
          rparams.unroll_factor_inner_reduction) {
        tv->split(-1, rparams.vectorization_factor_tmp_gmem_write);
      }
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorization propagate through propagateParallelization only works for
  // input and output tensors. propagate vectorization to cached_gmem_reload
  // directly from output tv using parallelizeAllLike. must propagate
  // seperaely for different tvs as outer reductions are transformed
  // seperately.
  if (rparams.vectorization_factor_outer > 1) {
    for (auto tv : cached_gmem_reload) {
      auto output_tvs = ir_utils::outputTvsOf(tv);
      NVF_ERROR(
          !output_tvs.empty(),
          "cached_gmem_reload should have at least one output tensor.")
      scheduler_utils::parallelizeAllLike(
          output_tvs[0],
          -1,
          {cached_gmem_reload.begin(), cached_gmem_reload.end()},
          {ParallelType::Vectorize});
    }
  }

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }
  inlineMost();
}

} // namespace nvfuser
