// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/reduction.h>

#include <debug.h>
#include <executor_utils.h>
#include <grouped_reduction.h>
#include <inlining.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_outer.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry.h>
#include <scheduler/registry_utils.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <transform_replay.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace nvfuser {

constexpr auto schedule_heuristic = ScheduleHeuristic::OuterPersistent;

OuterPersistentKernelScheduler::OuterPersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(schedule_heuristic) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void OuterPersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule Persistent Fusion");
  scheduleOuterPersistentKernel(fusion, reductionParams());
}

namespace {

bool checkReductionPattern(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  // Use root domain map to check the reduction ops have the same axes
  FusionGuard fg(fusion);
  ComputeAtRootDomainMap root_map;
  root_map.build(true);

  for (const auto it : c10::irange(1, reduction_tvs.size())) {
    if (!registry_utils::checkPatternEquivalence(
            reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedule_heuristic,
          "unmapped reduction ",
          reduction_tvs[it - 1],
          " and ",
          reduction_tvs[it]);
      return false;
    }
  }
  return true;
}

} // namespace

bool OuterPersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  // Needs at least one reduction to consider.
  auto reduction_ops = ir_utils::getAllTypesOfReductionOps(fusion);
  if (reduction_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "needs a reduction op");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedule_heuristic)) {
    return false;
  }

  // Fusions handled by persistent kernel scheduler cannot have MmaOp.
  if (!ir_utils::getOpsOfType<MmaOp>(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "no support for mma ops.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "no reduction tv");
    return false;
  }

  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);

  if (reduction_type != reduction_scheduler_utils::ReductionType::Outer) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "schedule_heuristic requires outer reduction tvs without inner reduction tvs.");
    return false;
  }

  if (!checkReductionPattern(fusion, reduction_tvs)) {
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedule_heuristic, "Fusion requires view being reversible.");
      return false;
    }

    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    TensorView* reference_tv = reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedule_heuristic,
          "View may interfere with normalization scheduling.");
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
            schedule_heuristic,
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
        schedule_heuristic, "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

bool OuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("OuterPersistentKernelScheduler::canSchedule");
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reduction_tvs[0]);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();

  const int64_t sm_register_file_size =
      static_cast<int64_t>(device_prop->regsPerBlock * sizeof(int));

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  const auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Note that projected buffer size can be zero
  auto persistent_buffer_size =
      persistent_buffer_size_info.projected_persistent_buffer_size == 0
      ? persistent_buffer_size_info.persistent_buffer_size
      : std::min(
            persistent_buffer_size_info.persistent_buffer_size,
            persistent_buffer_size_info.projected_persistent_buffer_size);

  const int64_t device_multiprocessor_count =
      (int64_t)device_prop->multiProcessorCount;

  const auto available_persistent_buffer_size =
      sm_register_file_size * device_multiprocessor_count;

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "not enough registers for persistence");
    return false;
  }

  auto reduced_tv = ir_utils::getSoleProducerTv(reduction_tvs.at(0));

  const int64_t vectorization_factor =
      (int64_t)vectorize_helper::getVectorizationFactor(
          runtime_info,
          reduced_tv,
          data_cache,
          (int)reduced_tv->nDims() -
              (int)properties.inner_most_dimension_ndims);

  // Minimum required multi reduction factor.
  const int64_t min_multi_reduction_factor = vectorization_factor *
      normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx;

  const int64_t required_sm_per_norm = ceilDiv(
      persistent_buffer_size * min_multi_reduction_factor,
      sm_register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "requires over half GPU persistence.",
        " required SMs per normalization: ",
        required_sm_per_norm);
    return false;
  }

  const bool is_cross_grid = required_sm_per_norm > 1;

  std::optional<normalization_scheduler_utils::GridOuterNormalizationParams>
      cross_grid_params;

  if (is_cross_grid) {
    // Don't try to be persistent unless at least 4-way vectorized
    // as register usage is hard to control
    // TODO: Is this necessary for block persistence as well?
    if (vectorization_factor < 4) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedule_heuristic, "not enough vectorized");
      return false;
    }

    // Make sure there's a valid grid persistence launch config
    cross_grid_params =
        normalization_scheduler_utils::getGridOuterNormalizationParams(
            properties.total_reduction_numel,
            properties.total_iteration_numel,
            vectorization_factor,
            persistent_buffer_size);

    if (!cross_grid_params.has_value()) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedule_heuristic, "no valid launch config found");
      return false;
    }
  }

  NVF_ERROR(!is_cross_grid || cross_grid_params.has_value())

  // Maximum number of iteration dimensions we can have and still be
  // persistent.
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      is_cross_grid ? available_persistent_buffer_size : sm_register_file_size,
      persistent_buffer_size);

  // Don't go persistent if we can't fit the minimum multi reduction
  // factor
  if (max_multi_reduction_factor < min_multi_reduction_factor) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "Not enough threads.",
        " Multi reduction factor, ",
        max_multi_reduction_factor,
        ", is smaller than minimum multi reduction factor, ",
        min_multi_reduction_factor);
    return false;
  }

  const int64_t max_used_sms = is_cross_grid
      ? ceilDiv(
            ceilDiv(properties.total_iteration_numel, vectorization_factor),
            cross_grid_params->launch_params.bdimx()) *
          cross_grid_params->launch_params.gdimy()
      : ceilDiv(
            properties.total_iteration_numel * persistent_buffer_size,
            sm_register_file_size);

  // Bandwidth suffers if the number of used SMs is small. This is
  // particularly impactful in the case of cross grid, so at least
  // half of the SMs are required to be used. In the case of cross
  // block, keep using the existing heuristics for now.
  if (is_cross_grid &&
      max_used_sms < scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "cross grid - not enough used SMs: ", max_used_sms);
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)device_prop->maxThreadsPerMultiProcessor;
  const int64_t min_fraction_of_sms =
      scheduler_utils::safeDiv(device_multiprocessor_count, 8);
  if (properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 && // Large reduction dim
      max_used_sms < min_fraction_of_sms) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "not enough used SMs");
    return false;
  }

  // The runtime kernel for grouped normal grid reductions is not
  // well tuned, and it turned out to be quite difficult to get
  // consistently better performances than non-persistent
  // schedules. Disabled for now.
  // TODO: Enable non-welford persistent reductions
  if (is_cross_grid &&
      std::any_of(
          reduction_tvs.begin(),
          reduction_tvs.end(),
          [](TensorView* reduction_tv) {
            return !reduction_tv->definition()->isA<WelfordOp>();
          })) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "non-Welford not enabled yet");
    return false;
  }

  // Had a hard time tuning on Titan RTX and V100 when the iteration
  // space is not evenly divided by threads and thread blocks. It
  // doesn't seem to be noticeably bad on A100, though. For now,
  // disable the schedule if not evenly divisible on Titan RTX and
  // V100, i.e., compute architecture version 7.
  // TODO: Revisit
  if (is_cross_grid &&
      (properties.total_iteration_numel %
           (vectorization_factor * cross_grid_params->launch_params.bdimx() *
            cross_grid_params->launch_params.gdimx()) !=
       0) &&
      device_prop->major == 7) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic, "iteration not evenly divided");
    return false;
  }

  return true;
}

void OuterPersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

namespace {
// In normalization heuristics, we usually have several free parameters, e.g.
// persistent batch size, unroll factors, thread block size, etc. This wrapper
// class is used to make sure the parameters are set before they are used and
// they will not be changed after they are finalized.
class HeuristicParameterWrapper {
 private:
  int64_t value_;
  bool mutable_;

 public:
  HeuristicParameterWrapper() : value_(-1), mutable_(true) {}
  void set(int64_t val) {
    if (mutable_) {
      value_ = val;
    } else {
      NVF_ERROR(false, "Trying to set a non-mutable heuristic parameter!");
    }
  }

  int64_t get() const {
    NVF_ERROR(value_ != -1, "Heuristic parameter is not set!");
    return value_;
  }

  void finalize() {
    NVF_ERROR(value_ != -1, "Heuristic parameter is not set!");
    mutable_ = false;
  }

  bool isMutable() const {
    return mutable_;
  }
};

} // namespace

namespace {

// Heuristics for grid outer normalizations
std::shared_ptr<ReductionParams> gridOuterPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    const int64_t max_persistent_buffer_size,
    const size_t vectorize_factor) {
  auto outer_params =
      normalization_scheduler_utils::getGridOuterNormalizationParams(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)vectorize_factor,
          max_persistent_buffer_size);

  NVF_ERROR(outer_params.has_value(), "No valid config found");

  const auto pb_size = outer_params->persistent_buffer_factor;
  const auto unswitch_factor = outer_params->unswitch_factor;

  auto rparams = std::make_shared<ReductionParams>();

  rparams->persistent_kernel = true;
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  rparams->grid_dim_inner_reduction = ParallelType::BIDy;
  rparams->block_dim_inner_reduction = ParallelType::TIDy;
  rparams->batches_per_block_inner_reduction = pb_size;
  rparams->multiple_reds_per_blk = true;
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = (int64_t)vectorize_factor;
  rparams->block_dim_iter_dom = ParallelType::TIDx;
  rparams->unroll_factor_inner_reduction = unswitch_factor;
  rparams->split_grid_dim_iter_dom_inner =
      ceilDiv(
          total_iteration_numel / (int64_t)vectorize_factor,
          outer_params->launch_params.bdimx()) >
      outer_params->launch_params.gdimx();
  rparams->compute_persistent_buffer_with_first_consumer = true;
  rparams->static_bdimx = true;
  rparams->static_bdimy = true;

  rparams->lparams = LaunchParams(
      rparams->split_grid_dim_iter_dom_inner
          ? outer_params->launch_params.gdimx()
          : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      outer_params->launch_params.bdimx(),
      outer_params->launch_params.bdimy(),
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "max_persistent_buffer_size: " << max_persistent_buffer_size
            << "\n"
            << "persistent_buffer_factor: " << pb_size << "\n"
            << "block(" << outer_params->launch_params.bdimx() << ", "
            << outer_params->launch_params.bdimy() << ", 1)" << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
// TODO: Check adding iteration domain unrolling
std::shared_ptr<ReductionParams> outerPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    const int64_t max_persistent_buffer_size,
    const size_t vectorize_factor) {
  // Set some targets for parallelization
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // If it fits in l2, we just want to make sure each warp uses 32Bytes. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size =
      n_elems * max_input_dtype_size * n_tensor_inputs < dev_prop->l2CacheSize
      ? (int64_t)32 / max_input_dtype_size
      : 16;

  const auto register_file_size =
      dev_prop->regsPerBlock * scheduler_utils::bytes_per_register;
  const int64_t device_warp_size = (int64_t)dev_prop->warpSize;

  // Each block runs N reductions, where N is defined as:
  // vectorize_factor * blockDim.x. The minimum number of SMs to run
  // this as a persistent kernel is thus defined as:
  const int64_t min_required_sm_per_norm = ceilDiv(
      max_persistent_buffer_size * (int64_t)vectorize_factor *
          normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx,
      (int64_t)register_file_size);

  if (min_required_sm_per_norm > 1) {
    return gridOuterPersistentHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_input_dtype_size,
        max_persistent_buffer_size,
        vectorize_factor);
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      scheduler_utils::register_file_size, max_persistent_buffer_size);

  struct HeuristicParams {
    // Iteration dim, each CTA covers [bdimx] * [iter_unroll_factor] reductions.
    // Needs total_iteration_numel / (bdimx * iter_unroll_factor) CTAs.
    HeuristicParameterWrapper iter_unroll_factor;
    HeuristicParameterWrapper bdimx;
    // Reduction dim, each thread do [batches_per_block * redu_unroll_factor]
    // serial reductions, then do block reductions along [bdimy].
    // Total_reduction_numel <= bdimy [dynamic] * batches_per_block *
    // redu_unroll_factor
    HeuristicParameterWrapper redu_unroll_factor;
    HeuristicParameterWrapper batches_per_block;
    HeuristicParameterWrapper bdimy;
    void verify() {
      NVF_ERROR(
          !iter_unroll_factor.isMutable(),
          "iter_unroll_factor is not finalized.");
      NVF_ERROR(!bdimx.isMutable(), "bdimx is not finalized.");
      NVF_ERROR(
          !redu_unroll_factor.isMutable(),
          "redu_unroll_factor is not finalized.");
      NVF_ERROR(
          !batches_per_block.isMutable(),
          "batches_per_block is not finalized.");
      NVF_ERROR(!bdimy.isMutable(), "bdimy is not finalized.");
    }
  };
  HeuristicParams hp;

  // set iter_unroll_factor
  // This controls vectorized load/store along the iteration dimension.
  // The kernel calls block reduction [iter_unroll_factor] times.
  // Test shows performance regression when iter_unroll_factor > 1 due to
  // the high cost of calling block reduction multiple times per block.
  hp.iter_unroll_factor.set(1l);
  hp.iter_unroll_factor.finalize();

  // set redu_unroll_factor
  // This controls unroll along the reduction dimension.
  // For case InstanceNormFP32 of [256, 28, 28, 128], if unroll 2, register
  // usage increased from 89 to 118 but the occupancy is not changed. However,
  // the bandwidth is dropped from 1029 GB/s to 840 GB/s due to more stalled
  // warps. Unroll by 4 increased performance for some cases but has regression
  // in many others. So we set redu_unroll_factor to 1.
  hp.redu_unroll_factor.set(1l);
  hp.redu_unroll_factor.finalize();

  // set bdimx
  // Start from warp_size, and decrease it until we can make more than 4 waves
  const int64_t bdimx_max =
      max_multi_reduction_factor / hp.iter_unroll_factor.get();
  int64_t tmp_bdimx = std::min(bdimx_max, warp_size);
  if (tmp_bdimx < warp_size) {
    tmp_bdimx = scheduler_utils::lastPow2(tmp_bdimx);
  }
  // check if we can make more than 4 waves to hide memory access latency.
  // InstanceNormFP32 of [32, 32, 32, 128] increased from 618 to 770 GB/s
  int64_t num_CTAs =
      ceilDiv(total_iteration_numel, tmp_bdimx * hp.iter_unroll_factor.get());
  while (
      num_CTAs < 4l * device_multiprocessor_count &&
      tmp_bdimx >= 2l *
              normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx) {
    tmp_bdimx /= 2l;
    num_CTAs =
        ceilDiv(total_iteration_numel, tmp_bdimx * hp.iter_unroll_factor.get());
  }
  // we are not finalizing bdimx here, because we may need to change it later if
  // bdimy is very small
  hp.bdimx.set(tmp_bdimx);

  // set bdimy and batches_per_block
  // These two parameters controls the reduction. Each reduction is split into
  // [batches_per_block] serial reductions and a block reduction along [bdimy].
  // Test shows setting a serial workload larger than 8 improves performance
  // since it reduces inter-threads communication.
  const int64_t batches_per_block_min = std::min(8l, total_reduction_numel);

  // A minimum of 128 threads in a block ensures the four warp schedulers are
  // fully utilized even in cases where only one CTA is active per SM.
  const int64_t min_threads_in_block = 128l;

  // A maximum of 256 threads in a block ensures each thread can use up to 255
  // registers.
  const int64_t max_threads_in_block = 256l;

  // Split reduction domain into redu_unroll_factor, bdimy, and
  // batches_per_block. redu_unroll_factor is already finalized, so the problem
  // changes to split after_unroll into bdimy and batches_per_block. The
  // strategy is: prioritize divisible splits and search for bdimy in a fixed
  // range under the constraint of batches_per_block_min.
  const int64_t after_unroll =
      total_reduction_numel / hp.redu_unroll_factor.get();
  const int64_t bdimy_max = std::min(
      ceilDiv(after_unroll, batches_per_block_min),
      max_threads_in_block / hp.bdimx.get());
  const int64_t bdimy_min =
      std::min(bdimy_max, min_threads_in_block / hp.bdimx.get());
  const int64_t bdimy_step = std::max(1l, device_warp_size / hp.bdimx.get());
  NVF_ERROR(
      device_warp_size % hp.bdimx.get() == 0,
      "bdimx is no divisible by warp_size. bdimx= ",
      hp.bdimx.get());

  auto maybeNextDivisibleFactor =
      [&after_unroll, &bdimy_step, &bdimy_max](int64_t cur) {
        auto next = cur + bdimy_step;
        while (next <= bdimy_max && after_unroll % next) {
          next += bdimy_step;
        }
        return std::min(next, bdimy_max);
      };
  int64_t tmp_bdimy = bdimy_min;
  int64_t tmp_batch = ceilDiv(after_unroll, tmp_bdimy);
  while (tmp_bdimy < bdimy_max) {
    int64_t next_bdimy = maybeNextDivisibleFactor(tmp_bdimy);
    int64_t next_batch = ceilDiv(after_unroll, next_bdimy);
    if (next_batch >= batches_per_block_min) {
      tmp_bdimy = next_bdimy;
      tmp_batch = next_batch;
    } else {
      break;
    }
  }
  hp.bdimy.set(tmp_bdimy);
  hp.bdimy.finalize();
  hp.batches_per_block.set(tmp_batch);
  hp.batches_per_block.finalize();

  // final check on bdimx to avoid small threads_in_block
  if (hp.bdimx.get() * hp.bdimy.get() < min_threads_in_block) {
    hp.bdimx.set(min_threads_in_block / hp.bdimy.get());
  }
  hp.bdimx.finalize();

  // make sure all paras are set
  hp.verify();

  // Final check of the requested registers
  int64_t sm_required_per_norm_set = ceilDiv(
      max_persistent_buffer_size * hp.bdimx.get() * hp.iter_unroll_factor.get(),
      scheduler_utils::register_file_size);
  NVF_ERROR(
      sm_required_per_norm_set == 1,
      "Tried to use multiple SMs on an outer persistent kernel ",
      "yet this kernel should have been within block persistent.",
      "\nbdimx= ",
      hp.bdimx.get(),
      ", iter_unroll_factor= ",
      hp.iter_unroll_factor.get());

  // copy to ReductionParams
  auto rparams = std::make_shared<ReductionParams>();
  auto gdimx = ceilDiv(total_iteration_numel, hp.bdimx.get());
  rparams->batches_per_block_inner_reduction = hp.batches_per_block.get();
  rparams->persistent_kernel = true;

  rparams->fastest_dim = false;
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = false;
  rparams->multiple_reds_per_blk = hp.bdimx.get() > 1;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDx;
  }

  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  rparams->split_grid_dim_iter_dom_outer =
      gdimx > scheduler_utils::x_grid_limit;

  if (rparams->block_dim_iter_dom == ParallelType::TIDx) {
    rparams->block_dim_inner_reduction = ParallelType::TIDy;
  } else {
    rparams->block_dim_inner_reduction = ParallelType::TIDx;
  }

  // Always need to mark inner reduction unroll for rfactor in outer persitent
  // kernels
  rparams->unroll_factor_inner_reduction = hp.redu_unroll_factor.get();

  rparams->unroll_factor_iter_dom = hp.iter_unroll_factor.get();

  rparams->vectorize_iter_dom =
      vectorize_factor > 1 && hp.iter_unroll_factor.get() > 1;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      rparams->multiple_reds_per_blk ? hp.bdimx.get() : hp.bdimy.get(),
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "Outer persistent kernel heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "max_persistent_buffer_size: " << max_persistent_buffer_size
            << "\n"
            << "max_multi_reduction_factor: " << max_multi_reduction_factor
            << "\n"
            << "block(" << hp.bdimx.get() << ", " << hp.bdimy.get() << ", 1)"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

std::shared_ptr<ReductionParams> persistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const size_t n_tensor_inputs,
    const size_t max_input_dtype_size,
    const int64_t max_persistent_buffer_size,
    size_t vectorize_factor,
    bool project_persistent_buffers) {
  std::shared_ptr<ReductionParams> rparams;

  rparams = outerPersistentHeuristic(
      total_reduction_numel,
      total_iteration_numel,
      (int64_t)n_tensor_inputs,
      (int64_t)max_input_dtype_size,
      max_persistent_buffer_size,
      vectorize_factor);

  rparams->project_persistent_buffers = project_persistent_buffers;
  return rparams;
}

} // namespace

std::shared_ptr<ReductionParams> getOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getOuterPersistentHeuristics");
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto ref_red_tv = reduction_tvs[0];

  NVF_ERROR(ref_red_tv != nullptr, "Reduction TensorView wasn't found.");

  NVF_ERROR(ref_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = ref_red_tv->definition();

  NVF_ERROR(
      ir_utils::isReductionOp(red_expr),
      "TensorView doesn't have a reduction.");

  auto tv_inps = ir_utils::filterByType<TensorView>(fusion->inputs());
  NVF_ERROR(
      std::distance(tv_inps.begin(), tv_inps.end()) > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

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

  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);

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
  bool project_persistent_buffers = can_project &&
      persistent_buffer_size_info.projected_persistent_buffer_size <
          persistent_buffer_size_info.persistent_buffer_size;

  auto max_persistent_size = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;

  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);

  auto unrollable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::UnrollableInputsAndOutputs>(
          data_cache, [&reduced_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reduced_tv, false, false));
          });

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();

  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      vectorize_helper::getVectorizationBreakPointOfReductionProducer(
          ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));

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

  auto heuristic = persistentHeuristic(
      properties.total_reduction_numel,
      properties.total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size,
      max_persistent_size,
      vectorize_factor,
      project_persistent_buffers);
  heuristic->cparams.index_type = runtime_info.getIndexType();
  return heuristic;
}

std::shared_ptr<ReductionParams> getOuterPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getOuterPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  return getOuterPersistentHeuristics(fusion, runtime_info, data_cache);
}

namespace {

// common prepare for both inner outer combined and seperated reductions
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

// If called from schedulePersistentKernel, reduction_tvs are either inner
// reductions or outer reductions. If called from
// schedulePersistentKernelInnerOuter, reduction_tvs are inner reductions, outer
// reductions are handled by scheduleCombinedOuter.
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
      reduction_tvs.size() > 1) {
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

} // namespace

// fusion is the input IR that will be modified by this function
void scheduleOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("scheduleOuterPersistentKernel");

  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      cached_outputs);

  TensorView* reference_tv =
      scheduleReductionGeneral(fusion, rparams, reduction_tvs);

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

} // namespace nvfuser
