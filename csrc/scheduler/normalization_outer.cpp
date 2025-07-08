// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_outer.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

namespace {
// In normalization heuristics, we usually have several free parameters, e.g.
// persistent batch size, unroll factors, thread block size, etc. This wrapper
// class is used to make sure the parameters are set before they are used and
// they will not be changed after they are finalized.
class ParameterWrapper {
 private:
  int64_t value_;
  bool mutable_;

 public:
  ParameterWrapper() : value_(-1), mutable_(true) {}
  void set(int64_t val) {
    if (mutable_) {
      value_ = val;
    } else {
      NVF_THROW("Trying to set a non-mutable heuristic parameter!");
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
std::unique_ptr<ReductionParams> gridOuterPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size_bit,
    const int64_t max_persistent_buffer_size_bit,
    const size_t vectorize_factor,
    const bool project_to_input,
    const PrimDataType index_type) {
  auto outer_params =
      normalization_scheduler_utils::getGridOuterNormalizationParams(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)vectorize_factor,
          max_persistent_buffer_size_bit);

  NVF_ERROR(outer_params.has_value(), "No valid config found");

  const auto pb_size = outer_params->persistent_buffer_factor;
  const auto unswitch_factor = outer_params->unswitch_factor;

  auto rparams = std::make_unique<ReductionParams>(
      OuterPersistentKernelScheduler::schedulerType());

  rparams->persistent_kernel = true;
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;
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
            << "max_input_dtype_size_bit: " << max_input_dtype_size_bit << "\n"
            << "max_persistent_buffer_size_bit: " << max_persistent_buffer_size_bit
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
std::unique_ptr<ReductionParams> outerPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size_bit,
    const int64_t max_persistent_buffer_size_bit,
    const size_t vectorize_factor,
    const bool project_to_input,
    const PrimDataType index_type) {
  // Set some targets for parallelization
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // If it fits in l2, we just want to make sure each warp uses 256Bits. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size =
      n_elems * max_input_dtype_size_bit * n_tensor_inputs < dev_prop->l2CacheSize * 8
      ? (int64_t)256 / max_input_dtype_size_bit
      : 16;

  const auto register_file_size_bit =
      dev_prop->regsPerBlock * scheduler_utils::bits_per_register;
  const int64_t device_warp_size = (int64_t)dev_prop->warpSize;

  // Each block runs N reductions, where N is defined as:
  // vectorize_factor * blockDim.x. The minimum number of SMs to run
  // this as a persistent kernel is thus defined as:
  const int64_t min_required_sm_per_norm = ceilDiv(
      max_persistent_buffer_size_bit * (int64_t)vectorize_factor *
          normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx,
      (int64_t)register_file_size_bit);

  if (min_required_sm_per_norm > 1) {
    return gridOuterPersistentHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_input_dtype_size_bit,
        max_persistent_buffer_size_bit,
        vectorize_factor,
        project_to_input,
        index_type);
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      scheduler_utils::register_file_size_bit, max_persistent_buffer_size_bit);

  struct NormOuterParams {
    // Iteration dim, each CTA covers [bdimx] * [iter_unroll_factor] reductions.
    // Needs total_iteration_numel / (bdimx * iter_unroll_factor) CTAs.
    ParameterWrapper iter_unroll_factor;
    ParameterWrapper bdimx;
    // Reduction dim, each thread do [batches_per_block * redu_unroll_factor]
    // serial reductions, then do block reductions along [bdimy].
    // Total_reduction_numel <= bdimy [dynamic] * batches_per_block *
    // redu_unroll_factor
    ParameterWrapper redu_unroll_factor;
    ParameterWrapper batches_per_block;
    ParameterWrapper bdimy;
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
  NormOuterParams params;

  // set iter_unroll_factor
  // This controls vectorized load/store along the iteration dimension.
  // The kernel calls block reduction [iter_unroll_factor] times.
  // Test shows performance regression when iter_unroll_factor > 1 due to
  // the high cost of calling block reduction multiple times per block.
  params.iter_unroll_factor.set(1l);
  params.iter_unroll_factor.finalize();

  // set redu_unroll_factor
  // This controls unroll along the reduction dimension.
  // For case InstanceNormFP32 of [256, 28, 28, 128], if unroll 2, register
  // usage increased from 89 to 118 but the occupancy is not changed. However,
  // the bandwidth is dropped from 1029 GB/s to 840 GB/s due to more stalled
  // warps. Unroll by 4 increased performance for some cases but has regression
  // in many others. So we set redu_unroll_factor to 1.
  params.redu_unroll_factor.set(1l);
  params.redu_unroll_factor.finalize();

  // set bdimx
  // Start from warp_size, and decrease it until we can make more than 4 waves
  const int64_t bdimx_max =
      max_multi_reduction_factor / params.iter_unroll_factor.get();
  int64_t tmp_bdimx = std::min(bdimx_max, warp_size);
  if (tmp_bdimx < warp_size) {
    tmp_bdimx = scheduler_utils::lastPow2(tmp_bdimx);
  }
  // check if we can make more than 4 waves to hide memory access latency.
  // InstanceNormFP32 of [32, 32, 32, 128] increased from 618 to 770 GB/s
  int64_t num_CTAs = ceilDiv(
      total_iteration_numel, tmp_bdimx * params.iter_unroll_factor.get());
  while (
      num_CTAs < 4l * device_multiprocessor_count &&
      tmp_bdimx >= 2l *
              normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx) {
    tmp_bdimx /= 2l;
    num_CTAs = ceilDiv(
        total_iteration_numel, tmp_bdimx * params.iter_unroll_factor.get());
  }
  // we are not finalizing bdimx here, because we may need to change it later if
  // bdimy is very small
  params.bdimx.set(tmp_bdimx);

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
      total_reduction_numel / params.redu_unroll_factor.get();
  const int64_t bdimy_max = std::min(
      ceilDiv(after_unroll, batches_per_block_min),
      max_threads_in_block / params.bdimx.get());
  const int64_t bdimy_min =
      std::min(bdimy_max, min_threads_in_block / params.bdimx.get());
  const int64_t bdimy_step =
      std::max(1l, device_warp_size / params.bdimx.get());
  NVF_ERROR(
      device_warp_size % params.bdimx.get() == 0,
      "bdimx is no divisible by warp_size. bdimx= ",
      params.bdimx.get());

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
  params.bdimy.set(tmp_bdimy);
  params.bdimy.finalize();
  params.batches_per_block.set(tmp_batch);
  params.batches_per_block.finalize();

  // final check on bdimx to avoid small threads_in_block
  if (params.bdimx.get() * params.bdimy.get() < min_threads_in_block) {
    params.bdimx.set(min_threads_in_block / params.bdimy.get());
  }
  params.bdimx.finalize();

  // make sure all paras are set
  params.verify();

  // Final check of the requested registers
  int64_t sm_required_per_norm_set = ceilDiv(
      max_persistent_buffer_size_bit * params.bdimx.get() *
          params.iter_unroll_factor.get(),
      scheduler_utils::register_file_size_bit);
  NVF_ERROR(
      sm_required_per_norm_set == 1,
      "Tried to use multiple SMs on an outer persistent kernel ",
      "yet this kernel should have been within block persistent.",
      "\nbdimx= ",
      params.bdimx.get(),
      ", iter_unroll_factor= ",
      params.iter_unroll_factor.get());

  // copy to ReductionParams
  auto rparams = std::make_unique<ReductionParams>(
      OuterPersistentKernelScheduler::schedulerType());
  auto gdimx = ceilDiv(total_iteration_numel, params.bdimx.get());
  rparams->batches_per_block_inner_reduction = params.batches_per_block.get();
  rparams->persistent_kernel = true;
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;

  rparams->fastest_dim = false;
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = false;
  rparams->multiple_reds_per_blk = params.bdimx.get() > 1;

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
  rparams->unroll_factor_inner_reduction = params.redu_unroll_factor.get();

  rparams->unroll_factor_iter_dom = params.iter_unroll_factor.get();

  rparams->vectorize_iter_dom =
      vectorize_factor > 1 && params.iter_unroll_factor.get() > 1;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      rparams->multiple_reds_per_blk ? params.bdimx.get() : params.bdimy.get(),
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "Outer persistent kernel heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size_bit: " << max_input_dtype_size_bit << "\n"
            << "max_persistent_buffer_size_bit: " << max_persistent_buffer_size_bit
            << "\n"
            << "max_multi_reduction_factor: " << max_multi_reduction_factor
            << "\n"
            << "block(" << params.bdimx.get() << ", " << params.bdimy.get()
            << ", 1)" << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

} // namespace

std::unique_ptr<ReductionParams> getOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  const auto& prop =
      normalization_scheduler_utils::getPersistentKernelProperties(
          fusion,
          runtime_info,
          data_cache,
          OuterPersistentKernelScheduler::schedulerType());

  std::unique_ptr<ReductionParams> rparams = outerPersistentHeuristic(
      prop.total_reduction_numel,
      prop.total_iteration_numel,
      prop.n_tensor_inputs,
      prop.max_dtype_size_bit,
      prop.max_persistent_buffer_size_bit,
      prop.vectorize_factor,
      prop.project_persistent_buffers,
      prop.index_type);
  return rparams;
}

// fusion is the input IR that will be modified by this function
void scheduleOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams* rparams) {
  NVF_ERROR(
      rparams->scheduler_type ==
      OuterPersistentKernelScheduler::schedulerType());
  normalization_scheduler_utils::schedulePersistentKernel(
      fusion, rparams, rparams->scheduler_type);
}

bool OuterPersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("OuterPersistentKernelScheduler::canScheduleCompileTime");
  return normalization_scheduler_utils::compileTimeCheck(
      fusion, schedulerType());
}

bool OuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("OuterPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
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
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  const auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSizeBit(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Note that projected buffer size can be zero
  auto persistent_buffer_size_bit =
      persistent_buffer_size_info.projected_persistent_buffer_size_bit == 0
      ? persistent_buffer_size_info.persistent_buffer_size_bit
      : std::min(
            persistent_buffer_size_info.persistent_buffer_size_bit,
            persistent_buffer_size_info.projected_persistent_buffer_size_bit);

  const int64_t device_multiprocessor_count =
      (int64_t)device_prop->multiProcessorCount;

  const auto available_persistent_buffer_size_bit =
      sm_register_file_size_bit * device_multiprocessor_count;

  if (persistent_buffer_size_bit > available_persistent_buffer_size_bit) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "not enough registers for persistence");
    return false;
  }

  auto reduced_tv = ir_utils::getSoleProducerTv(reduction_tvs.at(0));

  const int64_t vectorization_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      reduced_tv->nDims() - properties.inner_most_dimension_ndims);

  // Minimum required multi reduction factor.
  const int64_t min_multi_reduction_factor = vectorization_factor *
      normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx;

  const int64_t required_sm_per_norm = ceilDiv(
      persistent_buffer_size_bit * min_multi_reduction_factor,
      sm_register_file_size_bit);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
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
          schedulerType(), "not enough vectorized");
      return false;
    }

    // Make sure there's a valid grid persistence launch config
    cross_grid_params =
        normalization_scheduler_utils::getGridOuterNormalizationParams(
            properties.total_reduction_numel,
            properties.total_iteration_numel,
            vectorization_factor,
            persistent_buffer_size_bit);

    if (!cross_grid_params.has_value()) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "no valid launch config found");
      return false;
    }
  }

  NVF_ERROR(!is_cross_grid || cross_grid_params.has_value())

  // Maximum number of iteration dimensions we can have and still be
  // persistent.
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      is_cross_grid ? available_persistent_buffer_size_bit : sm_register_file_size_bit,
      persistent_buffer_size_bit);

  // Don't go persistent if we can't fit the minimum multi reduction
  // factor
  if (max_multi_reduction_factor < min_multi_reduction_factor) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
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
            properties.total_iteration_numel * persistent_buffer_size_bit,
            sm_register_file_size_bit);

  // Bandwidth suffers if the number of used SMs is small. This is
  // particularly impactful in the case of cross grid, so at least
  // half of the SMs are required to be used. In the case of cross
  // block, keep using the existing heuristics for now.
  if (is_cross_grid &&
      max_used_sms < scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "cross grid - not enough used SMs: ", max_used_sms);
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
        schedulerType(), "not enough used SMs");
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
        schedulerType(), "non-Welford not enabled yet");
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
        schedulerType(), "iteration not evenly divided");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> OuterPersistentKernelScheduler::
    computeHeuristics(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("OuterPersistentKernelScheduler::computeHeuristics");
  auto rparams = getOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void OuterPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("OuterPersistentKernelScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to OuterPersistentKernelScheduler::schedule",
      params);
  scheduleOuterPersistentKernel(fusion, rparams);
}

} // namespace nvfuser
