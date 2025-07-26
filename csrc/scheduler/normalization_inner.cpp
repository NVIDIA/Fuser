// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

namespace {

std::pair<int64_t, int64_t> getPersistentBufferSizeBit(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const bool can_use_smem_persistent) {
  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSizeBit(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  normalization_scheduler_utils::BufferProjectionStrategy project_strategy =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerPersistentKernelScheduler::schedulerType(),
          can_use_smem_persistent);
  bool project_persistent_buffers =
      (project_strategy ==
       normalization_scheduler_utils::BufferProjectionStrategy::
           ProjectToInputs);
  auto persistent_buffer_size_bit = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size_bit
      : persistent_buffer_size_info.persistent_buffer_size_bit;

  int64_t available_persistent_buffer_size_bit = normalization_scheduler_utils::
      getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          can_use_smem_persistent,
          project_persistent_buffers);
  return std::make_pair(
      persistent_buffer_size_bit, available_persistent_buffer_size_bit);
}

// Return the maximum register count each thread can use and achieved occupancy.
// We always guarantee the returned register count is at least as large as the
// buffer+overhead estimate. We meet the desired occupancy but don't try to
// maximize it further. As long as it's as large as a given target occupancy, we
// consider it's good enough. The idea is that as long as we have a good enough
// occupancy, we should be able to saturate the memory bandwidth. Within these
// constraints, we try to maximize the number of registers each thread can use.
// Para [target_warps_per_sm]: required occupancy to saturate memory bandwidth.
// Para [register_overhead]: registers except those for persistent buffers.
std::pair<int64_t, int64_t> getMaxRegisterCountPerThreadAndOccupancy(
    const int64_t buffer_size_per_thread_bit,
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
      buffer_size_per_thread_bit / scheduler_utils::bits_per_register +
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

// Returns the maximum persistent batch size.
// For example: assuming we have 64K registers per SM and 28 warps (864 threads)
// per SM. Each thread can use up to 72 registers. Then minus the register
// overhead 16, there are 56 registers or 224 * 8 = 1792 bits to store the
// persistent buffer.
// (1) If each reduction element has 1 fp32 buffer and vectorized by 8,
//     [buffer_bits_per_batch] = 4 * 8 * 8 = 256. Then the maximum persistent
//      batch size is 1792 / 256 = 7
// (2) If each reduction element has 1 fp16 buffer and vectorized by 8,
//     [buffer_bits_per_batch] = 2 * 8 * 8 = 128. Then the maximum persistent
//      batch size is 1792 / 128 = 14, which is then capped to
//      [max_batches_per_block] whose value is 10.
int64_t getMaxPersistentBatch(
    const int64_t buffer_bits_per_batch,
    const int64_t target_threads_per_sm,
    const int64_t register_overhead,
    const bool is_high_bandwidth_flops_ratio = false) {
  // (1) calculate the maximum register count given the target occupancy.
  int64_t total_register =
      getRegPerThreadGivenThreadsPerSM(target_threads_per_sm);
  int64_t register_for_buffer = total_register - register_overhead;

  // (2) calculate the maximum persistent batch size using the register count.
  int64_t batch_size = scheduler_utils::safeDiv(
      register_for_buffer * scheduler_utils::bits_per_register,
      buffer_bits_per_batch);

  // (3) Avoid using very large persistent buffer size, which may lead to low
  // occupancy due to the limitation of the current heuristics. TODO: remove
  // this parameter when we have a better heuristic to select the best
  // persistent batch size.
  int64_t max_batches_per_block =
      normalization_scheduler_utils::getInnerPersistentMaxBatchSize(
          is_high_bandwidth_flops_ratio);
  return std::min(max_batches_per_block, batch_size);
}

// calculate bdimx, bdimy, occupancy, given a persistent batch size
struct NormInnerParams {
  int64_t bdimx = -1;
  int64_t bdimy = -1;
  int64_t padded_bdimx = -1;
  int64_t persistent_batch_size = -1;
  int64_t register_per_thread = -1;
  int64_t non_buffer_registers = -1;
  int64_t occupancy = -1;
  int64_t n_wave = -1;
  int64_t n_persistent_tails = -1;
  bool is_pad_bdimx = false;
  void print() const {
    std::cout << "bdimx: " << bdimx << ", bdimy: " << bdimy
              << ", padded_bdimx: " << padded_bdimx
              << ", persistent_batch_size: " << persistent_batch_size
              << ", register_per_thread: " << register_per_thread
              << ", non_buffer_registers: " << non_buffer_registers
              << ", occupancy: " << occupancy << ", n_wave: " << n_wave
              << ", n_persistent_tails: " << n_persistent_tails
              << ", is_pad_bdimx: " << is_pad_bdimx << std::endl;
  }
};

NormInnerParams getNormInnerParamsGivenPerisisentBatchSize(
    const int64_t reduction_count_after_vectorize,
    const int64_t total_iteration_numel,
    const int64_t max_multi_reduction_factor,
    const int64_t min_threads_per_block,
    const int64_t buffer_bits_per_batch,
    const int64_t target_warps_per_sm,
    const int64_t register_overhead,
    const int64_t persistent_batch_size) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  auto device_warp_size = dev_prop->warpSize;
  auto max_threads_per_block = dev_prop->maxThreadsPerBlock;
  auto sm_count = dev_prop->multiProcessorCount;
  NormInnerParams params;
  params.persistent_batch_size = persistent_batch_size;

  // set bdimx and bdimy
  params.bdimx = scheduler_utils::safeDiv(
      reduction_count_after_vectorize, persistent_batch_size);
  NVF_ERROR(
      params.bdimx <= max_threads_per_block,
      "persistent batch size too small! bdimx should be less than ",
      max_threads_per_block,
      ", but got ",
      params.bdimx);
  params.bdimy = std::min(
      scheduler_utils::safeDiv(min_threads_per_block, params.bdimx),
      max_multi_reduction_factor);
  params.padded_bdimx = params.bdimx % device_warp_size == 0
      ? params.bdimx
      : params.bdimx + (device_warp_size - params.bdimx % device_warp_size);
  params.is_pad_bdimx = params.bdimx > 16 &&
      params.padded_bdimx * params.bdimy <= max_threads_per_block;

  // calculate register per thread and achieved occupancy
  int64_t threads_per_block = params.is_pad_bdimx
      ? params.padded_bdimx * params.bdimy
      : params.bdimx * params.bdimy;
  int64_t persistent_buffer_size_bit =
      buffer_bits_per_batch * persistent_batch_size;
  auto reg_occ = getMaxRegisterCountPerThreadAndOccupancy(
      persistent_buffer_size_bit,
      threads_per_block,
      target_warps_per_sm,
      register_overhead);
  params.register_per_thread = reg_occ.first;
  params.occupancy = reg_occ.second;
  int64_t blocks_per_sm = scheduler_utils::safeDiv(
      params.occupancy * device_warp_size, threads_per_block);
  params.n_wave = ceilDiv(total_iteration_numel, sm_count * blocks_per_sm);
  params.non_buffer_registers = params.register_per_thread -
      persistent_buffer_size_bit / scheduler_utils::bits_per_register;
  // (4) Calculate other quantities reflecting the quality of the heuristic.
  // when [reduction_count_after_vectorize] is not divisible by
  // [persistent_val], the last batch is not be fully utilized, the wasted
  // threads in the last batch is quantified as [n_persistent_tails].
  params.n_persistent_tails =
      ceilDiv(reduction_count_after_vectorize, persistent_batch_size) *
          persistent_batch_size -
      reduction_count_after_vectorize;
  return params;
}

// Return true if ha is better than hb

// This sorting function ensures the selected heuristic meeting target
// occupancy, promotes even workload distribution, enhances register
// optimization, and prefers higher occupancy.
// TODO: It leads to 10% regression for softmax around 2K to 6K and 16K.
// See https://github.com/NVIDIA/Fuser/issues/1876
bool compareTwoHeuristics(
    const NormInnerParams& pa,
    const NormInnerParams& pb,
    const int64_t min_non_buffer_registers,
    const int64_t target_warps_per_sm,
    const bool is_high_bandwidth_flops_ratio,
    const bool has_exp_ops) {
  // This lambda compares a parameter between two `NormInnerParams`
  // configurations. If the parameters for configuration A and B are the same,
  // continue along and compare other parameters. Otherwise, short-circuit
  // compareTwoHeuristics.
  auto compare = [](int64_t a, int64_t b) -> int {
    return a > b ? 1 : (a < b ? -1 : 0);
  };
  int score = 0;

  // prefer occupancy larger than target
  score = compare(
      pa.occupancy >= target_warps_per_sm, pb.occupancy >= target_warps_per_sm);
  if (score != 0) {
    return score > 0;
  }

  // prefer reduction count after vectorization is divisible by persistent
  // batch size. Skip this check when the bandwidth to flops ratio is high and
  // has expensive ops, under this condition, using a larger persistent batch
  // is more beneficial than using a smaller persistent batch that is divisible.
  if (!(is_high_bandwidth_flops_ratio && has_exp_ops)) {
    score = compare(pa.n_persistent_tails == 0, pb.n_persistent_tails == 0);
    if (score != 0) {
      return score > 0;
    }
  }

  // Ensure the count of non buffer registers is larger than (or equal to if the
  // bandwidth to flops ratio is high and fusion has expensive ops) the min
  // overhead. But don't want to achieve this goal at the cost of using a very
  // large block size, it avoids using a small persistent batch with a large
  // block size, which usually leads to 10% lower in performance.
  constexpr int64_t opt_max_threads_per_block = 512;
  if (is_high_bandwidth_flops_ratio && has_exp_ops) {
    score = compare(
        pa.non_buffer_registers >= min_non_buffer_registers &&
            pa.padded_bdimx <= opt_max_threads_per_block,
        pb.non_buffer_registers >= min_non_buffer_registers &&
            pb.padded_bdimx <= opt_max_threads_per_block);
  } else {
    score = compare(
        pa.non_buffer_registers > min_non_buffer_registers &&
            pa.padded_bdimx <= opt_max_threads_per_block,
        pb.non_buffer_registers > min_non_buffer_registers &&
            pb.padded_bdimx <= opt_max_threads_per_block);
  }

  if (score != 0) {
    return score > 0;
  }

  // when there are enough waves, prefer the one with less waves, less waves
  // means higher occupancy. We don't want to directly use occupancy as two
  // different occupancies may lead to the same number of waves.
  if (is_high_bandwidth_flops_ratio && (pa.n_wave >= 8 || pb.n_wave >= 8)) {
    score = compare(pa.n_wave, pb.n_wave);
    if (score != 0) {
      return score < 0;
    }
  }

  // Prefer large occupancy
  score = compare(pa.occupancy, pb.occupancy);
  if (score != 0) {
    return score > 0;
  }
  // Tiebreaker, use large persistent batch size so more registers are used
  // for the persistent buffer.
  return pa.persistent_batch_size > pb.persistent_batch_size;
}

// Generate a heuristic for each possible persistent batch size.
// (1) If the maximum occupancy is less than the target occupancy, use the batch
//     leads to the largest occupancy.
// (2) sort the heuristics as follows:
//     (a) Prefer occupancy exceeding target.
//         Ensures minimum required occupancy is surpassed.
//     (b) Prefer divisible by persistent batch size.
//         Aims for even workload distribution.
//     (c) Prefer non buffer register exceeds min overhead.
//         Maximizes compiler optimization potential.
//     (d) Seek larger occupancy.
//         Exceeds the target minimum for better performance.
//     (e) Use large persistent batch size as a tiebreaker.
//         Use more registers for persistent buffers.
// This sequence ensures meeting target occupancy, promotes even workload
// distribution, enhances register optimization, and prefers higher occupancy.
void innerPersistentHeuristic2D(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams) {
  bool is_high_bandwidth_flops_ratio =
      scheduler_utils::isHighBandwidthFlopsRatio();
  // Currently, we only considered the influence of exp op which is used in
  // softmax should extend to other MUFU units. Note that, rng op is an
  // expensive op however, test shows it can't be processed similarly to exp op
  // since it doesn't use the MUFU units.
  bool has_exp_op = properties.has_exp_op;
  bool disable_project_to_avoid_recompute =
      properties.disable_project_to_avoid_recompute;
  // Define two free parameters used in this heuristic.
  // register_overhead is all registers except those for the persistent
  // buffers. The register in each thread = register_overhead +
  // persistent_buffer_size_bit / bits_per_register
  // Current values are based on tests of sofmax, layer_norm, softmax_dropout,
  // dropout_layer_norm on A100 & H100. It directly affects maxregcount passed
  // to NVRTC and influences the occupancy.
  const int64_t register_overhead = has_exp_op ? 32l : 16l;

  // Target occupancy required to hide memory latency.
  // Used to calculate the maximum register count each thread can use.
  // Used to calculate the maximum persistent batch size.
  // Current value of 28 is based on tests of softmax, layer_norm,
  // softmax_dropout, dropout_layer_norm on A100 & H100. When bandwidth to flops
  // ratio is high, we may disable recompute persistent buffer from inputs to
  // reduce computation costs. when this happens, the target occupancy is set to
  // 16 to allow more registers per thread and larger persistent batch size for
  // better instruction level parallelism. This empirical value is based on
  // tests of softmax on B100.
  const int64_t target_warps_per_sm =
      is_high_bandwidth_flops_ratio && disable_project_to_avoid_recompute ? 16l
                                                                          : 28l;

  // device properties
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = (int64_t)dev_prop->warpSize;
  const int64_t max_threads_in_block = (int64_t)dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // alwasy use [vectorize_factor]
  const int64_t parallel_after_vectorize =
      properties.inner_most_dimension_numel / properties.vectorize_factor;

  // try to use at least 4 warps per block
  const int64_t min_threads_per_block = 4l * threads_per_warp;

  // set the min persistent buffer size to avoid requesting
  // a block size larger than device limit
  int64_t batches_per_block_inner_reduction_min =
      ceilDiv(parallel_after_vectorize, max_threads_in_block);

  // set the max persistent batch size to avoid low occupancy
  // (1) limitation set by min_threads_per_block
  const int64_t pbs_max_1 =
      ceilDiv(parallel_after_vectorize, min_threads_per_block);
  // (2) derived the maximum persistent batch size from the target occupancy
  const int64_t buffer_bits_per_batch =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * properties.vectorize_factor;
  const int64_t target_threads_per_sm =
      std::min(target_warps_per_sm * threads_per_warp, max_threads_per_sm);
  const int64_t pbs_max_2 = getMaxPersistentBatch(
      buffer_bits_per_batch,
      target_threads_per_sm,
      register_overhead,
      is_high_bandwidth_flops_ratio);
  int64_t batches_per_block_inner_reduction_max = std::max(
      batches_per_block_inner_reduction_min, std::min(pbs_max_1, pbs_max_2));

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size_bit,
          properties.max_persistent_buffer_size_bit),
      ceilDiv(properties.total_iteration_numel, device_multiprocessor_count));

  // Generate a heuristic for each possible persistent batch size.
  // record which persistent batch size has the highest occupancy.
  int64_t idx_max_occupancy = -1;
  int64_t current_max_occupancy = -1;
  std::vector<NormInnerParams> all_heuristics;

  all_heuristics.reserve(
      batches_per_block_inner_reduction_max -
      batches_per_block_inner_reduction_min + 1);
  for (int64_t pbs = batches_per_block_inner_reduction_min;
       pbs <= batches_per_block_inner_reduction_max;
       pbs++) {
    all_heuristics.push_back(getNormInnerParamsGivenPerisisentBatchSize(
        parallel_after_vectorize,
        properties.total_iteration_numel,
        max_multi_reduction_factor,
        min_threads_per_block,
        buffer_bits_per_batch,
        target_warps_per_sm,
        register_overhead,
        pbs));
    if (all_heuristics.back().occupancy > current_max_occupancy) {
      current_max_occupancy = all_heuristics.back().occupancy;
      idx_max_occupancy = (int64_t)all_heuristics.size() - 1;
    }
  }

  // Sort the heuristics and select the best one.
  // If no persistent batch size can achieve the target occupancy, and
  NormInnerParams best_heuristic;
  if (current_max_occupancy < target_warps_per_sm) {
    best_heuristic = all_heuristics.at(idx_max_occupancy);
  } else {
    std::stable_sort(
        all_heuristics.begin(),
        all_heuristics.end(),
        [&register_overhead,
         &target_warps_per_sm,
         &is_high_bandwidth_flops_ratio,
         &has_exp_op](const NormInnerParams& a, const NormInnerParams& b) {
          return compareTwoHeuristics(
              a,
              b,
              register_overhead,
              target_warps_per_sm,
              is_high_bandwidth_flops_ratio,
              has_exp_op);
        });
    best_heuristic = all_heuristics.at(0);
  }

  // Fill in the reduction params
  rparams->cparams.maxrregcount = best_heuristic.register_per_thread;

  // Disable magic zero to further reduce computation cost.
  // Magic zero reduces register usage, so only disble it when the register
  // usage is so low that we can disable project to avoid recompute.
  if (is_high_bandwidth_flops_ratio && disable_project_to_avoid_recompute) {
    rparams->cparams.enable_magic_zero = false;
  }
  // Inner reduction domain
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = best_heuristic.is_pad_bdimx;
  rparams->batches_per_block_inner_reduction =
      best_heuristic.persistent_batch_size;

  // For persistent schedules always have to mark the reduction unrolled
  // otherwise rfactor can fail
  rparams->unroll_factor_inner_reduction = properties.vectorize_factor;
  rparams->vectorize_inner_reduction = properties.vectorize_factor > 1;

  // Iter domain
  rparams->multiple_reds_per_blk = best_heuristic.bdimy > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t godim =
      ceilDiv(properties.total_iteration_numel, best_heuristic.bdimy);
  if (godim > 1) {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      best_heuristic.bdimy,
      LaunchParams::UNINITIALIZED_VAL);
}

void innerGridPersistentHeuristic2D(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // Inner reduction domain
  // This heuristic is only used for cases with large total_reduction_numel.
  // e.g. layer_norm with hidden size larger than 64K for fp16 or 32K for fp32.
  // fully vectorized, use maxThreadsPerBlock to reduce workload per threads
  int64_t vectorize_factor = properties.vectorize_factor;
  int64_t bdimx = 1024;
  int64_t gdimx = scheduler_utils::roundUpPow2(properties.max_persistent_buffer_size_bit / scheduler_utils::register_file_size_bit);
  NVF_ERROR(gdimx > 1, "gdimx should be larger than 1");
  int64_t persistent_batch =
      ceilDiv(properties.total_reduction_numel, vectorize_factor * bdimx * gdimx);
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = true;

  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->grid_dim_inner_reduction = ParallelType::BIDx;

  
  rparams->pad_inner_reduction_to_warp = true;
  rparams->batches_per_block_inner_reduction = persistent_batch;
  rparams->unroll_factor_inner_reduction = vectorize_factor;
  rparams->vectorize_inner_reduction = vectorize_factor > 1;

  // Iter
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->multiple_reds_per_blk = false;
  rparams->unroll_factor_iter_dom = 1;
  auto max_gdimy = dev_prop->multiProcessorCount / gdimx;
  rparams->split_grid_dim_iter_dom_inner = max_gdimy < properties.total_iteration_numel;
  rparams->lparams = LaunchParams(
      gdimx,
      rparams->split_grid_dim_iter_dom_inner ? max_gdimy
                                             : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);
}

// TODO: clean and revise the heuristics
void innerPersistentHeuristic3D(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams) {
  // Define two free parameters used in this heuristic.
  // register_overhead is all registers except those for the persistent
  // buffers. The register in each thread = register_overhead +
  // persistent_buffer_size_bit / bits_per_register
  // Current values are based on tests of sofmax, layer_norm, softmax_dropout,
  // dropout_layer_norm on A100 & H100. It directly affects maxregcount passed
  // to NVRTC and influences the occupancy.
  const int64_t register_overhead = properties.has_exp_op ? 32l : 16l;

  // Target occupancy required to hide memory latency
  // Current value is based on tests of sofmax, layer_norm, softmax_dropout,
  // dropout_layer_norm on A100 & H100.
  const int64_t target_warps_per_sm = 28l;

  // Set some targets for parallelization
  const int64_t n_elems =
      properties.total_reduction_numel * properties.total_iteration_numel;

  const int64_t outer_reduction_numel =
      properties.total_reduction_numel / properties.inner_most_dimension_numel;

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // WARNING: At some point we may want to generate heuristics for another
  // device that is not the current device.
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      128l / properties.max_dtype_size_bit,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(std::max(properties.n_tensor_inputs >> 2, 1l)));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache_bit = 32l * 1024l * 8;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 =
      n_elems * properties.max_dtype_size_bit * properties.n_tensor_inputs <
      dev_prop->l2CacheSize * 8;

  // If it fits in l2, we just want to make sure each warp uses 256Bits. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? 256l / properties.max_dtype_size_bit : 16l;

  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(
          properties.total_reduction_numel,
          scheduler_utils::safeDiv(
              l1_cache_bit,
              properties.n_tensor_inputs * properties.max_dtype_size_bit *
                  active_threads)),
      16l);

  // Take the smaller, warp_size may be a odd number, e.g. 15
  // Tracked at https://github.com/NVIDIA/Fuser/issues/107
  const int64_t warp_size =
      std::min(warp_size_based_on_l1, warp_size_based_on_l2);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations =
      scheduler_utils::safeDiv(256, properties.max_dtype_size_bit);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // max_threads_in_block is the cap on a thread block, the minimum is based on
  // warp_size
  int64_t max_threads_in_block = std::max(
      warp_size,
      ceilDiv(properties.total_reduction_numel, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = scheduler_utils::safeDiv(
        n_elems, warp_size * device_multiprocessor_count);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = true;

    while (available_unroll > 1 &&
           (target_unroll < max_unroll ||
            // Prefer unrolling
            target_iterations < max_unroll)) {
      if (target_unroll * 2 <= max_unroll && flip) {
        target_unroll *= 2;
      }

      if (target_iterations * 2 <= max_unroll && !flip) {
        target_iterations *= 2;
      }

      available_unroll = scheduler_utils::safeDiv(
          n_elems,
          warp_size * device_multiprocessor_count * target_unroll *
              target_iterations);
      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, warp_size * target_unroll * target_iterations);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_iterations < n_elems) {
    if (outer_reduction_numel == 1) {
      // set to hardware limit to use small persistent buffer for large
      // reductions
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          (int64_t)dev_prop->maxThreadsPerBlock);
    } else {
      // targetting 4 waves, so try to use a quarter of available threads
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
    }
  }

  // Round up to nearest warp.
  if (max_threads_in_block % warp_size != 0) {
    max_threads_in_block += warp_size - max_threads_in_block % warp_size;
    max_threads_in_block =
        std::min(max_threads_in_block, (int64_t)dev_prop->maxThreadsPerBlock);
  }
  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size_bit,
          properties.max_persistent_buffer_size_bit),
      ceilDiv(properties.total_iteration_numel, device_multiprocessor_count));
  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Blocks for outputs
  int64_t godim = 1;

  // Threads for reduction
  int64_t bdimx = 1;
  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for outer reduction dimension
  int64_t bdimz = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t outer_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  inner_reduction_unroll_factor =
      properties.vectorize_factor > 1 ? properties.vectorize_factor : 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(
      std::max(
          ceilDiv(
              properties.inner_most_dimension_numel,
              inner_reduction_unroll_factor),
          (int64_t)warp_size),
      max_threads_in_block);

  // If we're not just barely covering the dimension, round to a more friendly
  // number
  if (bdimx * inner_reduction_unroll_factor !=
      properties.inner_most_dimension_numel) {
    bdimx = bdimx > warp_size ? bdimx - bdimx % warp_size
                              : scheduler_utils::lastPow2(bdimx);

    // Round bdimx down to multiple of warp size or power 2
    if (bdimx < warp_size) {
      bdimx = scheduler_utils::lastPow2(bdimx);
    } else {
      bdimx = bdimx - bdimx % warp_size;
    }
  }

  // Put everything else in bdimy for now
  bdimy = std::min(
      scheduler_utils::safeDiv(warp_size, bdimx), max_multi_reduction_factor);
  // If 3D fill the rest of the threads into bdimz
  bdimz = std::min(
      std::min(
          scheduler_utils::safeDiv(max_threads_in_block, bdimx * bdimy),
          outer_reduction_numel),
      scheduler_utils::z_block_limit);

  bool vectorize = false;

  // Move unrolling factor into vectorization upto vectorization limit.
  if (properties.vectorize_factor > 1 && inner_reduction_unroll_factor > 1) {
    vectorize = true;
    inner_reduction_unroll_factor = std::min(
        scheduler_utils::lastPow2(inner_reduction_unroll_factor),
        properties.vectorize_factor);
  }

  // calculate the maximum persistent buffer size
  const int64_t buffer_bits_per_batch =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * inner_reduction_unroll_factor;
  const int64_t batches_per_block_inner_reduction_max = getMaxPersistentBatch(
      buffer_bits_per_batch,
      target_warps_per_sm * dev_prop->warpSize,
      register_overhead);

  // start from small block size to minimize expensive inter-threads reduction
  const int64_t threads_after_vectorize =
      properties.inner_most_dimension_numel / inner_reduction_unroll_factor;

  // Test min_threads_per_block using 3 values:
  // (1) One warp, so we can use single warp reduction and sync.
  // (2) Two warps, so we can achieve 100% occupancy since most GPUs allow 32
  //     blocks per SM.
  // (3) Four warps, number recommended by the cuda-c-best-practices-guide.
  const int64_t min_threads_per_block = 4l * dev_prop->warpSize;

  // start bdimx with min_threads_per_block then increase if we have too many
  // persistent buffer batches per block
  if (outer_reduction_numel == 1 && vectorize) {
    bdimx = std::min(min_threads_per_block, threads_after_vectorize);
  }

  // If we don't have enough threads, let's do multiple reductions per block.
  // Multiple reductions per block shows better performance than unroll
  // iterations. Still keep vectorization as it is important for performance
  // since V100.
  if (bdimx * bdimy * bdimz < min_threads_per_block) {
    bdimy = std::min(
        scheduler_utils::safeDiv(min_threads_per_block, bdimx * bdimz),
        max_multi_reduction_factor);
  }

  // Set size of persistent per thread buffer on inner reduction buffer
  // if too large, will be reduced later to reduce register usage
  int64_t batches_per_block_inner_reduction = ceilDiv(
      properties.inner_most_dimension_numel,
      bdimx * inner_reduction_unroll_factor);

  // Attempt to put some unrolling into the outer reduction if inner hasn't
  // taken the max unrolling
  if (inner_reduction_unroll_factor < max_unroll) {
    outer_reduction_unroll_factor = std::min(
        ceilDiv(max_unroll, inner_reduction_unroll_factor),
        ceilDiv(outer_reduction_numel, bdimz));
  }

  godim = ceilDiv(properties.total_iteration_numel, bdimy);

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (!vectorize && inner_reduction_unroll_factor < max_unroll &&
         batches_per_block_inner_reduction >= 2) {
    inner_reduction_unroll_factor *= 2;
    batches_per_block_inner_reduction = scheduler_utils::roundUpPow2Or8(ceilDiv(
        properties.inner_most_dimension_numel,
        bdimx * inner_reduction_unroll_factor));
  }

  // Set size of persistent per thread buffer on outer reduction buffer
  int64_t batches_per_block_outer_reduction =
      scheduler_utils::roundUpPow2Or8(ceilDiv(
          ceilDiv(
              properties.total_reduction_numel,
              properties.inner_most_dimension_numel),
          bdimz * outer_reduction_unroll_factor));

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (outer_reduction_unroll_factor < max_unroll &&
         batches_per_block_outer_reduction >= 2) {
    outer_reduction_unroll_factor *= 2;
    batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
        ceilDiv(outer_reduction_numel, bdimz * outer_reduction_unroll_factor));
  }

  // Adjust bdimx based on batches_per_block and unroll factor set as they could
  // have moved a bit since they're the free variables, not the buffers
  bdimx = ceilDiv(
      properties.inner_most_dimension_numel,
      inner_reduction_unroll_factor * batches_per_block_inner_reduction);
  bdimz = ceilDiv(
      outer_reduction_numel,
      outer_reduction_unroll_factor * batches_per_block_outer_reduction);

  // Try moving persistent buffer factors into threads until we have too many
  // threads.

  while (
      // If block size can be doubled
      bdimx * bdimy * bdimz * 2 <= max_threads_in_block &&
      // And batches_per_block_inner_reduction can be divided by two
      (batches_per_block_inner_reduction >
           batches_per_block_inner_reduction_max ||
       batches_per_block_outer_reduction >= 2)) {
    // Try to decrease per thread register allocation persistence size on inner
    // reduction by double bdimx.
    if (batches_per_block_inner_reduction >
        batches_per_block_inner_reduction_max) {
      bdimx *= 2;
      batches_per_block_inner_reduction = ceilDiv(
          properties.inner_most_dimension_numel,
          inner_reduction_unroll_factor * bdimx);
      continue;
    }

    // Try to decrease per thread register allocation persistence size on outer
    // reduction
    if (batches_per_block_outer_reduction >= 2 &&
        batches_per_block_outer_reduction !=
            scheduler_utils::roundUpPow2Or8(
                batches_per_block_outer_reduction / 2) &&
        bdimz * 2 <= scheduler_utils::z_block_limit) {
      batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
          batches_per_block_outer_reduction / 2);
      bdimz = ceilDiv(
          outer_reduction_numel,
          batches_per_block_outer_reduction * outer_reduction_unroll_factor);
      continue;
    }
    break;
  }

  // Register pressure is really high per thread, which could lead to local
  // memory leaks, if using less than maximum threads, decrease batches per
  // block by a factor of 2
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= max_threads_in_block &&
      batches_per_block_inner_reduction >
          batches_per_block_inner_reduction_max) {
    batches_per_block_inner_reduction = batches_per_block_inner_reduction / 2;
  }

  // Do the same on the outer reduction dimension
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= device_max_threads_per_multiprocessor &&
      batches_per_block_outer_reduction >= 2l) {
    batches_per_block_outer_reduction /= 2l;
  }

  auto device_warp_size = (int64_t)at::cuda::warp_size();
  auto padded_bdimx = bdimx % device_warp_size == 0
      ? bdimx
      : bdimx + (device_warp_size - bdimx % device_warp_size);

  bool pad_bdimx = bdimx > 16 &&
      padded_bdimx * bdimy * bdimz <= (int64_t)dev_prop->maxThreadsPerBlock;

  // estimate register usage and occupancy raito.
  // If occupancy raito is less than a preset occupancy_ratio, reduce register
  // usage register per thread is estimated as overhead + buffer_size_bit /
  // bits_per_register
  int64_t nvrtc_register_per_thread = scheduler_utils::max_registers_per_thread;
  const int64_t blocksPerKernel = godim;
  // register estimation is only valid for vectorized gmem access
  // we've seen unexpectedly high register counts with vectorization factor less
  // than 4, which would make the below estimate inaccurate.
  // TODO: support the non vectorized case. consider shmem.
  // only need to balance register and occupancy ratio if there are enough
  // blocks and buffers
  if (vectorize && blocksPerKernel > device_multiprocessor_count &&
      batches_per_block_inner_reduction > 1) {
    // Estimate register per thread based on buffer size, since inner reduction
    // dim is fully parallelized, the buffer size of each element equals the
    // total buffer size divide by inner_most_dimension_numel. Each thread will
    // hold batches_per_block_inner_reduction * inner_reduction_unroll_factor
    // elements.
    const int64_t persistent_buffer_size_bit =
        properties.max_persistent_buffer_size_bit /
        properties.inner_most_dimension_numel *
        batches_per_block_inner_reduction * inner_reduction_unroll_factor;
    const int64_t threads_per_block =
        pad_bdimx ? padded_bdimx * bdimy * bdimz : bdimx * bdimy * bdimz;

    // Calculate the max register count each thread can use.
    nvrtc_register_per_thread = getMaxRegisterCountPerThreadAndOccupancy(
                                    persistent_buffer_size_bit,
                                    threads_per_block,
                                    target_warps_per_sm,
                                    register_overhead)
                                    .first;
  }

  // Will be used once supporting inter-block persistence
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimz = LaunchParams::UNINITIALIZED_VAL;

  rparams->cparams.maxrregcount = nvrtc_register_per_thread;

  // Inner reduction domain
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;
  rparams->batches_per_block_inner_reduction =
      batches_per_block_inner_reduction;

  // For persistent schedules always have to mark the reduction unrolled
  // otherwise rfactor can fail
  rparams->unroll_factor_inner_reduction = inner_reduction_unroll_factor;
  rparams->vectorize_inner_reduction = vectorize;

  // Iter domain
  rparams->multiple_reds_per_blk = bdimy > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  if (godim > 1) {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  if (iter_unroll_factor > 1) {
    rparams->unroll_factor_iter_dom = iter_unroll_factor;
  }

  // Outer reduction domain
  rparams->schedule_3D =
      properties.total_reduction_numel != properties.inner_most_dimension_numel;
  if (rparams->schedule_3D) {
    rparams->batches_per_block_outer_reduction =
        batches_per_block_outer_reduction;
    rparams->block_dim_outer_reduction = ParallelType::TIDz;
    rparams->cross_block_outer_reduction = true;
    rparams->unroll_factor_outer_reduction = outer_reduction_unroll_factor;
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      gdimz,
      LaunchParams::UNINITIALIZED_VAL,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
}

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  // properties of the fusion
  const auto& prop =
      normalization_scheduler_utils::getPersistentKernelProperties(
          fusion,
          runtime_info,
          data_cache,
          InnerPersistentKernelScheduler::schedulerType());

  std::unique_ptr<ReductionParams> rparams = std::make_unique<ReductionParams>(
      InnerPersistentKernelScheduler::schedulerType());

  // shared heuristics for all cases
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->project_persistent_buffers = prop.project_persistent_buffers;
  rparams->cparams.index_type = prop.index_type;

  // specific heuristics for different cases
  if (prop.max_persistent_buffer_size_bit >
      scheduler_utils::register_file_size_bit) {
    rparams->tag = "Register Grid Inner Persistent Heuristic.\n";
    innerGridPersistentHeuristic2D(prop, rparams.get());
  } else if (prop.total_reduction_numel == prop.inner_most_dimension_numel) {
    rparams->tag = "2D Register Inner Persistent Heuristic.\n";
    innerPersistentHeuristic2D(prop, rparams.get());
  } else {
    rparams->tag = "3D Register Inner Persistent Heuristic.\n";
    innerPersistentHeuristic3D(prop, rparams.get());
  }

  // debug print
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << prop.toString() << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

} // namespace

bool InnerPersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::canScheduleCompileTime");
  return normalization_scheduler_utils::compileTimeCheck(
      fusion, schedulerType());
}

bool InnerPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  auto reference_tv = reduction_tvs[0];

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  // check reduction properties, don't use shared memory persistent if 3D
  // reduction
  bool can_use_smem_persistent =
      properties.total_reduction_numel == properties.inner_most_dimension_numel;
  // pair of persistent_buffer_size_bit and available_persistent_buffer_size_bit
  const std::pair<int64_t, int64_t> buffer_size_bit =
      getPersistentBufferSizeBit(
          fusion,
          runtime_info,
          data_cache,
          reduction_tvs,
          can_use_smem_persistent);
  const int64_t persistent_buffer_size_bit = buffer_size_bit.first;
  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (std::getenv("USE_MAIN")) {
    const int64_t available_persistent_buffer_size_bit = buffer_size_bit.second;
    if (persistent_buffer_size_bit > available_persistent_buffer_size_bit) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          can_use_smem_persistent
              ? "not enough registers or shared memory for persistence."
              : "not enough registers for persistence and shared memory "
                "persistence is not supported yet.");
      return false;
    }
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      persistent_buffer_size_bit, scheduler_utils::register_file_size_bit);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "requires over half GPU persistence.");
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
        schedulerType(), "not enough blocks");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> InnerPersistentKernelScheduler::
    computeHeuristics(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::computeHeuristics");
  auto rparams = getInnerPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void InnerPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to InnerPersistentKernelScheduler::schedule",
      params);
  NVF_ERROR(
      rparams->scheduler_type ==
      InnerPersistentKernelScheduler::schedulerType());
  normalization_scheduler_utils::schedulePersistentKernel(
      fusion, rparams, rparams->scheduler_type);
}
} // namespace nvfuser
